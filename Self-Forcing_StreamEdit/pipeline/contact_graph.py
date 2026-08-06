from typing import Dict, List

import torch
import torch.nn.functional as F

from .role_router import RoleState


CONTACT_GRAPH_MODES = {
    "no_graph",
    "distance_only",
    "shuffled",
    "source_qk",
}


def _pool_role(role: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch, frames, height, width = role.shape
    pooled = F.avg_pool2d(
        role.reshape(batch * frames, 1, height, width).float(),
        kernel_size=patch_size,
        stride=patch_size,
    )
    return pooled.reshape(
        batch,
        frames,
        pooled.shape[-2],
        pooled.shape[-1],
    )


def _empty_graph(device: torch.device, topk: int) -> Dict[str, torch.Tensor]:
    return {
        "object_indices": torch.empty(0, dtype=torch.long, device=device),
        "hand_indices": torch.empty(
            0, topk, dtype=torch.long, device=device
        ),
        "edge_confidence": torch.empty(
            0, topk, dtype=torch.float32, device=device
        ),
        "edge_valid": torch.empty(
            0, topk, dtype=torch.bool, device=device
        ),
        "object_confidence": torch.empty(
            0, dtype=torch.float32, device=device
        ),
    }


def build_oracle_contact_graphs(
    roles: RoleState,
    mode: str,
    topk: int = 4,
    radius: float = 2.5,
    min_confidence: float = 0.05,
    patch_size: int = 2,
    shuffle_seed: int = 0,
) -> List[Dict[str, torch.Tensor]]:
    """Build sparse object-to-hand edges on the transformer token grid."""
    if mode not in CONTACT_GRAPH_MODES:
        raise ValueError(f"Unsupported contact graph mode: {mode}")
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError(
            "min_confidence must lie in [0, 1], got "
            f"{min_confidence}"
        )
    if patch_size <= 0:
        raise ValueError(
            f"patch_size must be positive, got {patch_size}"
        )

    roles.validate()
    boundary = _pool_role(roles.boundary, patch_size)
    hand = _pool_role(roles.hand, patch_size)
    batch, frames, token_height, token_width = boundary.shape
    tokens_per_frame = token_height * token_width
    graphs = []

    for batch_index in range(batch):
        object_indices = []
        hand_indices = []
        edge_confidences = []
        edge_valids = []
        object_confidences = []

        for frame_index in range(frames):
            object_map = boundary[batch_index, frame_index]
            hand_map = hand[batch_index, frame_index]
            object_positions = torch.nonzero(
                object_map >= min_confidence,
                as_tuple=False,
            )
            hand_positions = torch.nonzero(
                hand_map >= min_confidence,
                as_tuple=False,
            )
            if object_positions.numel() == 0 or hand_positions.numel() == 0:
                continue

            distances = torch.cdist(
                object_positions.float(),
                hand_positions.float(),
            )
            object_values = object_map[
                object_positions[:, 0],
                object_positions[:, 1],
            ]
            hand_values = hand_map[
                hand_positions[:, 0],
                hand_positions[:, 1],
            ]
            confidence = (
                object_values[:, None]
                * hand_values[None, :]
                * torch.exp(-0.5 * (distances / radius) ** 2)
            )
            confidence = confidence.masked_fill(
                distances > radius,
                -1.0,
            )
            frame_topk = min(topk, hand_positions.shape[0])
            selected_confidence, selected_local_index = torch.topk(
                confidence,
                k=frame_topk,
                dim=1,
            )
            selected_valid = selected_confidence >= 0.0
            active_object = selected_valid.any(dim=1)
            if not active_object.any():
                continue

            object_positions = object_positions[active_object]
            object_values = object_values[active_object]
            selected_confidence = selected_confidence[active_object]
            selected_local_index = selected_local_index[active_object]
            selected_valid = selected_valid[active_object]
            selected_hand_positions = hand_positions[
                selected_local_index.clamp_min(0)
            ]

            object_token_index = (
                frame_index * tokens_per_frame
                + object_positions[:, 0] * token_width
                + object_positions[:, 1]
            )
            hand_token_index = (
                frame_index * tokens_per_frame
                + selected_hand_positions[..., 0] * token_width
                + selected_hand_positions[..., 1]
            )

            if frame_topk < topk:
                padding = topk - frame_topk
                hand_token_index = F.pad(
                    hand_token_index,
                    (0, padding),
                    value=0,
                )
                selected_confidence = F.pad(
                    selected_confidence,
                    (0, padding),
                    value=0.0,
                )
                selected_valid = F.pad(
                    selected_valid,
                    (0, padding),
                    value=False,
                )

            if mode == "shuffled":
                flat_valid = selected_valid.flatten()
                flat_hand = hand_token_index.flatten()
                valid_hand = flat_hand[flat_valid]
                if valid_hand.numel() > 1:
                    shift = 1 + (
                        shuffle_seed + batch_index + frame_index
                    ) % (valid_hand.numel() - 1)
                    flat_hand[flat_valid] = torch.roll(
                        valid_hand,
                        shifts=int(shift),
                    )
                    hand_token_index = flat_hand.reshape_as(
                        hand_token_index
                    )

            object_indices.append(object_token_index.long())
            hand_indices.append(hand_token_index.long())
            edge_confidences.append(
                selected_confidence.clamp_min(0.0).float()
            )
            edge_valids.append(selected_valid)
            object_confidences.append(object_values.float())

        if not object_indices:
            graphs.append(_empty_graph(boundary.device, topk))
            continue
        graphs.append({
            "object_indices": torch.cat(object_indices),
            "hand_indices": torch.cat(hand_indices),
            "edge_confidence": torch.cat(edge_confidences),
            "edge_valid": torch.cat(edge_valids),
            "object_confidence": torch.cat(object_confidences),
        })

    return graphs


def contact_graph_stats(
    graphs: List[Dict[str, torch.Tensor]],
) -> Dict[str, int]:
    return {
        "object_nodes": sum(
            int(graph["object_indices"].numel()) for graph in graphs
        ),
        "valid_edges": sum(
            int(graph["edge_valid"].sum().item()) for graph in graphs
        ),
    }
