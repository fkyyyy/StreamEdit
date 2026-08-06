import math
from typing import Dict

import torch


def apply_contact_graph_residual(
    target_output: torch.Tensor,
    source_query: torch.Tensor,
    target_query: torch.Tensor,
    source_key: torch.Tensor,
    target_key: torch.Tensor,
    source_value: torch.Tensor,
    target_value: torch.Tensor,
    graph: Dict[str, torch.Tensor],
    mode: str,
    strength: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Replace target contact messages with sparse source-relation messages."""
    if mode == "no_graph" or strength == 0.0:
        return target_output
    if mode not in {"distance_only", "shuffled", "source_qk"}:
        raise ValueError(f"Unsupported contact graph mode: {mode}")
    if strength < 0.0:
        raise ValueError(f"strength must be non-negative, got {strength}")
    if target_output.ndim != 4 or target_output.shape[0] != 1:
        raise ValueError(
            "target_output must have shape [1,L,H,D], got "
            f"{tuple(target_output.shape)}"
        )

    object_indices = graph["object_indices"]
    if object_indices.numel() == 0:
        return target_output
    hand_indices = graph["hand_indices"]
    edge_confidence = graph["edge_confidence"].float()
    edge_valid = graph["edge_valid"]
    object_confidence = graph["object_confidence"].float()
    sequence_length = source_query.shape[0]
    if (
        object_indices.min() < 0
        or object_indices.max() >= sequence_length
        or hand_indices[edge_valid].min() < 0
        or hand_indices[edge_valid].max() >= sequence_length
    ):
        raise ValueError("Contact graph contains an out-of-range token index")

    safe_hand_indices = hand_indices.masked_fill(~edge_valid, 0)
    source_object_query = source_query[object_indices].float()
    target_object_query = target_query[object_indices].float()
    source_hand_key = source_key[safe_hand_indices].float()
    target_hand_key = target_key[safe_hand_indices].float()
    prior_logits = torch.log(edge_confidence.clamp_min(eps))
    prior_logits = prior_logits.unsqueeze(-1)

    if mode == "distance_only":
        source_logits = prior_logits.expand(
            -1, -1, source_object_query.shape[1]
        )
        target_logits = source_logits
    else:
        scale = 1.0 / math.sqrt(source_object_query.shape[-1])
        source_logits = (
            torch.einsum(
                "nhd,nkhd->nkh",
                source_object_query,
                source_hand_key,
            )
            * scale
            + prior_logits
        )
        target_logits = (
            torch.einsum(
                "nhd,nkhd->nkh",
                target_object_query,
                target_hand_key,
            )
            * scale
            + prior_logits
        )

    valid = edge_valid.unsqueeze(-1)
    source_weight = torch.softmax(
        source_logits.masked_fill(~valid, -torch.inf),
        dim=1,
    )
    target_weight = torch.softmax(
        target_logits.masked_fill(~valid, -torch.inf),
        dim=1,
    )
    source_hand_value = source_value[safe_hand_indices].float()
    target_hand_value = target_value[safe_hand_indices].float()
    source_message = torch.einsum(
        "nkh,nkhd->nhd",
        source_weight,
        source_hand_value,
    )
    target_message = torch.einsum(
        "nkh,nkhd->nhd",
        target_weight,
        target_hand_value,
    )
    correction = (
        strength
        * object_confidence[:, None, None]
        * (source_message - target_message)
    )

    output = target_output.clone()
    output[0, object_indices] = (
        output[0, object_indices]
        + correction.to(dtype=target_output.dtype)
    )
    return output
