from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PairedEditMemoryState:
    """Sparse source-addressed target-residual memory for one layer."""

    source_key: torch.Tensor
    target_value_residual: torch.Tensor
    object_coordinate: torch.Tensor
    evidence: torch.Tensor
    source_part_signature: torch.Tensor | None = None
    lineage_id: torch.Tensor | None = None
    transport_confidence: torch.Tensor | None = None

    def validate(self) -> None:
        if self.source_key.ndim != 4:
            raise ValueError(
                "Paired edit-memory keys must have shape [B,P,H,D]"
            )
        if self.target_value_residual.shape != self.source_key.shape:
            raise ValueError(
                "Paired edit-memory residuals must align with source keys"
            )
        if self.object_coordinate.shape != (
            self.source_key.shape[0],
            self.source_key.shape[1],
            2,
        ):
            raise ValueError(
                "Object coordinates must have shape [B,P,2]"
            )
        if self.evidence.shape != self.source_key.shape[:2]:
            raise ValueError(
                "Paired edit-memory evidence must have shape [B,P]"
            )
        if (
            self.source_part_signature is not None
            and (
                self.source_part_signature.ndim != 3
                or self.source_part_signature.shape[:2]
                != self.source_key.shape[:2]
            )
        ):
            raise ValueError(
                "Source part signatures must align with memory slots"
            )
        if (
            self.lineage_id is not None
            and self.lineage_id.shape != self.source_key.shape[:2]
        ):
            raise ValueError(
                "Paired edit-memory lineage ids must align with slots"
            )
        if (
            self.transport_confidence is not None
            and self.transport_confidence.shape
            != self.source_key.shape[:2]
        ):
            raise ValueError(
                "Transport confidence must align with memory slots"
            )
        if not torch.isfinite(self.source_key.float()).all():
            raise ValueError("Paired edit-memory keys must be finite")
        if not torch.isfinite(
            self.target_value_residual.float()
        ).all():
            raise ValueError(
                "Paired edit-memory residuals must be finite"
            )
        if not torch.isfinite(self.object_coordinate.float()).all():
            raise ValueError(
                "Paired edit-memory coordinates must be finite"
            )
        if not torch.isfinite(self.evidence.float()).all():
            raise ValueError(
                "Paired edit-memory evidence must be finite"
            )
        if (
            self.source_part_signature is not None
            and not torch.isfinite(
                self.source_part_signature.float()
            ).all()
        ):
            raise ValueError(
                "Source part signatures must be finite"
            )
        if self.evidence.min() < 0 or self.evidence.max() > 1:
            raise ValueError(
                "Paired edit-memory evidence must lie in [0, 1]"
            )
        if (
            self.transport_confidence is not None
            and (
                not torch.isfinite(
                    self.transport_confidence.float()
                ).all()
                or self.transport_confidence.min() < 0
                or self.transport_confidence.max() > 1
            )
        ):
            raise ValueError(
                "Transport confidence must be finite and lie in [0, 1]"
            )


@dataclass(frozen=True)
class SourceAddressedRead:
    residual: torch.Tensor
    support: torch.Tensor
    best_similarity: torch.Tensor
    assigned_evidence: torch.Tensor
    residual_consensus: torch.Tensor | None = None
    source_value: torch.Tensor | None = None
    part_similarity: torch.Tensor | None = None
    part_confidence: torch.Tensor | None = None
    canonical_support: torch.Tensor | None = None
    transported_support: torch.Tensor | None = None
    transport_similarity: torch.Tensor | None = None
    transport_cycle_confidence: torch.Tensor | None = None
    lineage_id: torch.Tensor | None = None


@dataclass(frozen=True)
class SourceTransportResult:
    """A source-only frontier update and its counterfactual read."""

    read: SourceAddressedRead
    frontier: PairedEditMemoryState | None


@dataclass(frozen=True)
class ImmutableCanonicalAnchor:
    """Verified first-block K/delta-V and current read admission."""

    source_key: torch.Tensor
    target_value_residual: torch.Tensor
    object_coordinate: torch.Tensor
    evidence: torch.Tensor
    lineage_id: torch.Tensor
    query_support: torch.Tensor
    query_lineage_id: torch.Tensor
    query_key_mask: torch.Tensor

    def validate(self) -> None:
        if self.source_key.ndim != 4:
            raise ValueError(
                "Canonical anchor keys must have shape [B,M,H,D]"
            )
        if self.target_value_residual.shape != self.source_key.shape:
            raise ValueError(
                "Canonical anchor residuals must align with keys"
            )
        if self.object_coordinate.shape != (
            *self.source_key.shape[:2], 2
        ):
            raise ValueError(
                "Canonical anchor coordinates must have shape [B,M,2]"
            )
        if self.evidence.shape != self.source_key.shape[:2]:
            raise ValueError(
                "Canonical anchor evidence must have shape [B,M]"
            )
        if self.lineage_id.shape != self.source_key.shape[:2]:
            raise ValueError(
                "Canonical anchor lineage ids must have shape [B,M]"
            )
        if self.query_support.ndim != 2:
            raise ValueError(
                "Canonical query support must have shape [B,Q]"
            )
        if self.query_lineage_id.shape != self.query_support.shape:
            raise ValueError(
                "Canonical query lineage ids must align with support"
            )
        if self.query_key_mask.shape != (
            self.source_key.shape[0],
            self.query_support.shape[1],
            self.source_key.shape[1],
        ):
            raise ValueError(
                "Canonical query-key mask must have shape [B,Q,M]"
            )
        if self.query_support.shape[0] != self.source_key.shape[0]:
            raise ValueError(
                "Canonical anchors and queries must share batch size"
            )
        if not torch.isfinite(self.evidence.float()).all():
            raise ValueError(
                "Canonical anchor evidence must be finite"
            )
        if self.evidence.min() < 0 or self.evidence.max() > 1:
            raise ValueError(
                "Canonical anchor evidence must lie in [0, 1]"
            )


@dataclass(frozen=True)
class TransactionalCommit:
    write_weight: torch.Tensor
    source_match: torch.Tensor
    residual_agreement: torch.Tensor
    accepted: torch.Tensor
    canonical_residual: torch.Tensor
    lineage_id: torch.Tensor | None = None


def build_source_part_signature(
    source_value: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build a training-free appearance signature for each source token.

    Attention keys are optimized for correspondence and can regard adjacent
    parts of one object as interchangeable.  Clean-source values retain more
    local appearance/material information.  Per-token centering removes a
    shared activation offset and L2 normalization makes cosine similarity
    insensitive to activation magnitude and exposure.  No semantic part
    labels or target colors are used.
    """
    if source_value.ndim != 4:
        raise ValueError(
            "source_value must have shape [B,L,H,D]"
        )
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    flat = source_value.detach().float().flatten(2)
    centered = flat - flat.mean(dim=-1, keepdim=True)
    return F.normalize(centered, dim=-1, eps=eps)


def build_object_coordinates(
    owner_weight: torch.Tensor,
    *,
    tokens_per_frame: int,
    spatial_shape: tuple[int, int],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Express tokens in a per-frame object coordinate system.

    Coordinates are normalized by the visible object's bounding box. They are
    used only as a soft retrieval prior; values are never spatially warped.
    """
    if owner_weight.ndim != 2:
        raise ValueError("owner_weight must have shape [B,L]")
    height, width = spatial_shape
    if height <= 0 or width <= 0 or height * width != tokens_per_frame:
        raise ValueError(
            "spatial_shape must contain tokens_per_frame locations"
        )
    if tokens_per_frame <= 0 or owner_weight.shape[1] % tokens_per_frame:
        raise ValueError(
            "tokens_per_frame must evenly divide owner_weight"
        )
    batch = owner_weight.shape[0]
    frames = owner_weight.shape[1] // tokens_per_frame
    owner = owner_weight.detach().float().clamp(0.0, 1.0).reshape(
        batch, frames, height, width
    )
    rows = torch.arange(
        height, device=owner.device, dtype=torch.float32
    ).view(1, 1, height, 1).expand_as(owner)
    cols = torch.arange(
        width, device=owner.device, dtype=torch.float32
    ).view(1, 1, 1, width).expand_as(owner)
    support = owner > eps
    large_row = torch.full_like(rows, float(height))
    large_col = torch.full_like(cols, float(width))
    min_row = torch.where(support, rows, large_row).flatten(2).amin(-1)
    max_row = torch.where(support, rows, -large_row).flatten(2).amax(-1)
    min_col = torch.where(support, cols, large_col).flatten(2).amin(-1)
    max_col = torch.where(support, cols, -large_col).flatten(2).amax(-1)
    valid = support.flatten(2).any(-1)
    center_row = 0.5 * (min_row + max_row)
    center_col = 0.5 * (min_col + max_col)
    extent_row = (max_row - min_row + 1.0).clamp_min(1.0)
    extent_col = (max_col - min_col + 1.0).clamp_min(1.0)
    coord_row = (
        rows - center_row[:, :, None, None]
    ) / extent_row[:, :, None, None]
    coord_col = (
        cols - center_col[:, :, None, None]
    ) / extent_col[:, :, None, None]
    coordinates = torch.stack([coord_row, coord_col], dim=-1)
    coordinates = torch.where(
        valid[:, :, None, None, None],
        coordinates,
        torch.zeros_like(coordinates),
    )
    return coordinates.reshape(batch, -1, 2)


def build_object_interior_gate(
    owner_weight: torch.Tensor,
    *,
    object_role: torch.Tensor,
    boundary_role: torch.Tensor,
    hand_role: torch.Tensor,
    tokens_per_frame: int,
    spatial_shape: tuple[int, int],
    neighborhood_radius: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return a soft, role-aware interior gate on the attention grid.

    The gate is generic: it uses neither object labels nor appearance.  Role
    purity protects hand/contact tokens, while local owner density protects
    the silhouette boundary.  Thin or heavily occluded objects retain a soft
    path instead of being removed by a hard erosion.
    """
    if owner_weight.ndim != 2:
        raise ValueError("owner_weight must have shape [B,L]")
    if neighborhood_radius < 0:
        raise ValueError("neighborhood_radius must be non-negative")
    height, width = spatial_shape
    if height <= 0 or width <= 0 or height * width != tokens_per_frame:
        raise ValueError(
            "spatial_shape must contain tokens_per_frame locations"
        )
    if tokens_per_frame <= 0 or owner_weight.shape[1] % tokens_per_frame:
        raise ValueError(
            "tokens_per_frame must evenly divide owner_weight"
        )
    batch = owner_weight.shape[0]
    frames = owner_weight.shape[1] // tokens_per_frame
    expected_prefix = (batch, frames)
    roles = {
        "object_role": object_role,
        "boundary_role": boundary_role,
        "hand_role": hand_role,
    }
    for name, role in roles.items():
        if role.ndim != 4 or role.shape[:2] != expected_prefix:
            raise ValueError(
                f"{name} must have shape [B,T,H,W] aligned with owner"
            )

    def resize(role: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            role.detach().float().reshape(
                batch * frames, 1, *role.shape[-2:]
            ),
            output_size=spatial_shape,
        ).reshape(batch, frames, height, width)

    object_token = resize(object_role)
    boundary_token = resize(boundary_role)
    hand_token = resize(hand_role)
    role_purity = (
        object_token
        / (object_token + boundary_token + hand_token).clamp_min(eps)
    ).clamp(0.0, 1.0)

    owner = owner_weight.detach().float().clamp(0.0, 1.0).reshape(
        batch, frames, height, width
    )
    owner_support = (owner > eps).float()
    if neighborhood_radius > 0:
        kernel_size = 2 * neighborhood_radius + 1
        local_density = F.avg_pool2d(
            owner_support.reshape(batch * frames, 1, height, width),
            kernel_size=kernel_size,
            stride=1,
            padding=neighborhood_radius,
        ).reshape_as(owner)
    else:
        local_density = owner_support
    return (
        role_purity * local_density * owner_support
    ).clamp(0.0, 1.0).reshape(batch, -1)


def build_owner_attached_structure_gate(
    owner_weight: torch.Tensor,
    *,
    object_role: torch.Tensor,
    boundary_role: torch.Tensor,
    hand_role: torch.Tensor,
    tokens_per_frame: int,
    spatial_shape: tuple[int, int],
    neighborhood_radius: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Admit owner-attached object structure without opening the hand.

    The strict interior gate is deliberately conservative, but it removes
    silhouette and contact cells that determine part seams and object width.
    This gate restores a soft structural corridor only where the causal owner
    is present and the object-or-boundary role dominates the hand role.  It
    contains no category, color, or absolute-size assumption.  A paired read
    must still pass clean-source correspondence, part consistency, and cycle
    verification before its residual can affect attention.
    """
    if owner_weight.ndim != 2:
        raise ValueError("owner_weight must have shape [B,L]")
    if neighborhood_radius < 0:
        raise ValueError(
            "neighborhood_radius must be non-negative"
        )
    height, width = spatial_shape
    if height <= 0 or width <= 0 or height * width != tokens_per_frame:
        raise ValueError(
            "spatial_shape must contain tokens_per_frame locations"
        )
    if tokens_per_frame <= 0 or owner_weight.shape[1] % tokens_per_frame:
        raise ValueError(
            "tokens_per_frame must evenly divide owner_weight"
        )
    batch = owner_weight.shape[0]
    frames = owner_weight.shape[1] // tokens_per_frame
    expected_prefix = (batch, frames)
    for name, role in {
        "object_role": object_role,
        "boundary_role": boundary_role,
        "hand_role": hand_role,
    }.items():
        if role.ndim != 4 or role.shape[:2] != expected_prefix:
            raise ValueError(
                f"{name} must have shape [B,T,H,W] aligned with owner"
            )

    def resize(role: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            role.detach().float().reshape(
                batch * frames, 1, *role.shape[-2:]
            ),
            output_size=spatial_shape,
        ).reshape(batch, frames, height, width)

    strict_interior = build_object_interior_gate(
        owner_weight,
        object_role=object_role,
        boundary_role=boundary_role,
        hand_role=hand_role,
        tokens_per_frame=tokens_per_frame,
        spatial_shape=spatial_shape,
        neighborhood_radius=neighborhood_radius,
        eps=eps,
    ).reshape(batch, frames, height, width)

    object_token = resize(object_role)
    boundary_token = resize(boundary_role)
    hand_token = resize(hand_role)
    structure_role = (object_token + boundary_token).clamp(0.0, 1.0)
    # RoleState is a partition, so the omitted mass is background. Include it
    # explicitly here: normalizing only against hand would accidentally admit
    # a background-dominant cell containing a tiny interpolated object tail.
    background_token = (
        1.0 - object_token - boundary_token - hand_token
    ).clamp(0.0, 1.0)
    attached_role = (
        structure_role
        / (
            structure_role + hand_token + background_token
        ).clamp_min(eps)
    ).clamp(0.0, 1.0)
    # A hand-dominant mixed token remains exact native fallback.  Boundary
    # evidence can pass only when it belongs more strongly to the tracked
    # object than to the hand.
    object_dominant = (
        (structure_role > hand_token)
        & (structure_role > background_token)
    ).float()

    owner = owner_weight.detach().float().clamp(0.0, 1.0).reshape(
        batch, frames, height, width
    )
    owner_support = (owner > eps).float()
    if neighborhood_radius > 0:
        kernel_size = 2 * neighborhood_radius + 1
        local_density = F.avg_pool2d(
            owner_support.reshape(batch * frames, 1, height, width),
            kernel_size=kernel_size,
            stride=1,
            padding=neighborhood_radius,
        ).reshape_as(owner)
    else:
        local_density = owner_support
    # Square root retains thin silhouettes while still penalizing isolated
    # owner speckles.  The owner probability itself is multiplied later when
    # the request is assembled, so it is intentionally not duplicated here.
    structure_corridor = (
        attached_role
        * object_dominant
        * local_density.sqrt()
        * owner_support
    ).clamp(0.0, 1.0)
    # This mode is an additive ablation over the strict-interior baseline:
    # never remove a read that 934a would have made, only admit separately
    # verified owner-attached structure.
    return torch.maximum(
        strict_interior, structure_corridor
    ).reshape(batch, -1)


def source_addressed_residual_read(
    *,
    current_source_key: torch.Tensor,
    current_coordinate: torch.Tensor,
    current_owner: torch.Tensor,
    memory: PairedEditMemoryState | None,
    current_source_part_signature: torch.Tensor | None = None,
    topk: int = 8,
    min_similarity: float = 0.35,
    coordinate_bias: float = 1.0,
    coordinate_radius: float = 0.0,
    min_residual_consensus: float = 0.0,
    source_part_consistency: bool = False,
    min_part_similarity: float = 0.45,
    part_similarity_margin: float = 0.08,
    temperature: float = 0.07,
    eps: float = 1e-6,
) -> SourceAddressedRead:
    """Read target residuals using clean-source content as the address."""
    if current_source_key.ndim != 4:
        raise ValueError(
            "current_source_key must have shape [B,L,H,D]"
        )
    batch, length = current_source_key.shape[:2]
    if current_coordinate.shape != (batch, length, 2):
        raise ValueError(
            "current_coordinate must have shape [B,L,2]"
        )
    if current_owner.shape != (batch, length):
        raise ValueError("current_owner must have shape [B,L]")
    if topk <= 0:
        raise ValueError("topk must be positive")
    if not -1.0 < min_similarity < 1.0:
        raise ValueError("min_similarity must lie in (-1, 1)")
    if coordinate_bias < 0:
        raise ValueError("coordinate_bias must be non-negative")
    if coordinate_radius < 0:
        raise ValueError("coordinate_radius must be non-negative")
    if not 0.0 <= min_residual_consensus < 1.0:
        raise ValueError(
            "min_residual_consensus must lie in [0, 1)"
        )
    if not -1.0 < min_part_similarity < 1.0:
        raise ValueError("min_part_similarity must lie in (-1, 1)")
    if not 0.0 <= part_similarity_margin <= 2.0:
        raise ValueError(
            "part_similarity_margin must lie in [0, 2]"
        )
    if source_part_consistency:
        if current_source_part_signature is None:
            raise ValueError(
                "Part-consistent read requires current source signatures"
            )
        if current_source_part_signature.shape[:2] != (batch, length):
            raise ValueError(
                "Current source part signatures must align with tokens"
            )
    if temperature <= 0 or eps <= 0:
        raise ValueError("temperature and eps must be positive")

    residual = torch.zeros_like(current_source_key)
    support = current_owner.new_zeros(
        (batch, length), dtype=torch.float32
    )
    best_similarity = support.new_full((batch, length), -1.0)
    assigned_evidence = support.new_zeros((batch, length))
    residual_consensus = support.new_zeros((batch, length))
    part_similarity_output = support.new_full((batch, length), -1.0)
    part_confidence_output = support.new_zeros((batch, length))
    lineage_output = torch.full(
        (batch, length),
        -1,
        dtype=torch.long,
        device=current_source_key.device,
    )
    if memory is None:
        return SourceAddressedRead(
            residual=residual,
            support=support,
            best_similarity=best_similarity,
            assigned_evidence=assigned_evidence,
            residual_consensus=residual_consensus,
            part_similarity=part_similarity_output,
            part_confidence=part_confidence_output,
            lineage_id=lineage_output,
        )
    memory.validate()
    if memory.source_key.shape[0] != batch:
        raise ValueError(
            "Current tokens and paired memory must share batch size"
        )
    if memory.source_key.shape[2:] != current_source_key.shape[2:]:
        raise ValueError(
            "Current tokens and paired memory must share head dimensions"
        )
    if source_part_consistency and memory.source_part_signature is None:
        raise ValueError(
            "Part-consistent read requires stored source signatures"
        )
    if (
        source_part_consistency
        and memory.source_part_signature.shape[-1]
        != current_source_part_signature.shape[-1]
    ):
        raise ValueError(
            "Current and stored source part signatures must align"
        )

    for batch_index in range(batch):
        memory_evidence = memory.evidence[batch_index].to(
            current_source_key.device
        )
        query_index = torch.nonzero(
            current_owner[batch_index] > eps, as_tuple=False
        ).flatten()
        memory_index = torch.nonzero(
            memory_evidence > eps, as_tuple=False
        ).flatten()
        if query_index.numel() == 0 or memory_index.numel() == 0:
            continue
        query = F.normalize(
            current_source_key[batch_index, query_index].float(), dim=-1
        )
        memory_key = memory.source_key[batch_index].to(
            current_source_key.device
        )
        key = F.normalize(memory_key[memory_index].float(), dim=-1)
        similarity = torch.einsum("qhd,mhd->qmh", query, key).mean(-1)
        content_similarity = similarity.clone()
        memory_coordinate = memory.object_coordinate[batch_index].to(
            current_coordinate.device
        )[memory_index].float()
        coordinate_distance = (
            current_coordinate[batch_index, query_index].float()[:, None]
            - memory_coordinate[None]
        ).square().sum(-1)
        score = (
            similarity
            - float(coordinate_bias) * coordinate_distance
            + memory_evidence[memory_index].float()
            .clamp_min(eps)
            .log()[None]
        )
        coordinate_valid = torch.ones_like(score, dtype=torch.bool)
        if coordinate_radius > 0.0:
            coordinate_valid = coordinate_distance <= float(
                coordinate_radius
            ) ** 2
        part_similarity = torch.ones_like(score)
        if source_part_consistency:
            query_part = F.normalize(
                current_source_part_signature[
                    batch_index, query_index
                ].float(),
                dim=-1,
                eps=eps,
            )
            memory_part = F.normalize(
                memory.source_part_signature[batch_index]
                .to(current_source_key.device)[memory_index]
                .float(),
                dim=-1,
                eps=eps,
            )
            part_similarity = torch.einsum(
                "qc,mc->qm", query_part, memory_part
            )
            local_part = torch.where(
                coordinate_valid,
                part_similarity,
                torch.full_like(part_similarity, -1e4),
            )
            best_part = local_part.max(dim=-1, keepdim=True).values
            # The absolute threshold rejects an unseen/ambiguous part. The
            # relative margin prevents a strong cap match and a merely
            # plausible body match from sharing one top-k residual mixture.
            coordinate_valid = (
                coordinate_valid
                & (part_similarity >= float(min_part_similarity))
                & (
                    part_similarity
                    >= best_part - float(part_similarity_margin)
                )
            )
            score = score + part_similarity
        if min_residual_consensus > 0.0:
            # Consensus mode must not let a low-similarity neighbour
            # contribute its payload merely because another top-k slot
            # passed the query-level threshold.  Retain only source-content
            # matches within one retrieval temperature of the best match;
            # this adapts to pose without requiring semantic part labels.
            local_content = torch.where(
                coordinate_valid,
                content_similarity,
                torch.full_like(content_similarity, -1e4),
            )
            best_content = local_content.max(
                dim=-1, keepdim=True
            ).values
            coordinate_valid = (
                coordinate_valid
                & (content_similarity >= float(min_similarity))
                & (content_similarity >= best_content - float(temperature))
            )
        # A hard local candidate set prevents a high-confidence residual from
        # a different object part from winning solely through global K
        # similarity. Queries with no local candidate abstain exactly.
        candidate_score = torch.where(
            coordinate_valid, score, torch.full_like(score, -1e4)
        )
        selected_count = min(int(topk), int(memory_index.numel()))
        selected_score, selected_offset = candidate_score.topk(
            selected_count, dim=-1
        )
        selected_index = memory_index[selected_offset]
        selected_valid = torch.gather(
            coordinate_valid, dim=-1, index=selected_offset
        )
        stable_score = selected_score - selected_score.max(
            dim=-1, keepdim=True
        ).values
        assignment = (
            torch.exp(stable_score / float(temperature))
            * selected_valid.float()
        )
        assignment = assignment / assignment.sum(
            dim=-1, keepdim=True
        ).clamp_min(eps)
        memory_residual = memory.target_value_residual[batch_index].to(
            current_source_key.device
        )
        selected_residual = memory_residual[selected_index].float()
        selected_evidence = memory_evidence[selected_index].float()
        if memory.lineage_id is None:
            memory_lineage = torch.arange(
                memory.source_key.shape[1],
                device=current_source_key.device,
                dtype=torch.long,
            )
        else:
            memory_lineage = memory.lineage_id[batch_index].to(
                current_source_key.device
            )
        selected_lineage = memory_lineage[selected_index]
        selected_content = torch.gather(
            content_similarity, dim=-1, index=selected_offset
        )
        selected_content = torch.where(
            selected_valid,
            selected_content,
            torch.full_like(selected_content, -1.0),
        )
        selected_part = torch.gather(
            part_similarity, dim=-1, index=selected_offset
        )
        selected_part = torch.where(
            selected_valid,
            selected_part,
            torch.full_like(selected_part, -1.0),
        )
        retrieved = torch.einsum(
            "qk,qkhd->qhd", assignment, selected_residual
        )
        selected_flat = selected_residual.flatten(2)
        retrieved_flat = retrieved.flatten(1)
        residual_cosine = F.cosine_similarity(
            selected_flat, retrieved_flat[:, None], dim=-1, eps=eps
        ).clamp_min(0.0)
        selected_norm = selected_flat.norm(dim=-1).clamp_min(eps)
        retrieved_norm = retrieved_flat.norm(dim=-1).clamp_min(eps)
        magnitude_agreement = torch.exp(
            -(selected_norm / retrieved_norm[:, None]).log().abs()
        )
        consensus_score = torch.where(
            selected_valid,
            selected_content,
            torch.full_like(selected_content, -1e4),
        )
        consensus_score = consensus_score - consensus_score.max(
            dim=-1, keepdim=True
        ).values
        consensus_weight = (
            torch.exp(consensus_score / max(float(temperature), 0.25))
            * selected_valid.float()
            * selected_evidence.clamp_min(eps)
        )
        consensus_weight = consensus_weight / consensus_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(eps)
        consensus = (
            consensus_weight
            * torch.sqrt(residual_cosine * magnitude_agreement)
        ).sum(-1).clamp(0.0, 1.0)
        if min_residual_consensus > 0.0:
            consensus_gate = (
                (consensus - float(min_residual_consensus))
                / (1.0 - float(min_residual_consensus))
            ).clamp(0.0, 1.0)
        else:
            consensus_gate = torch.ones_like(consensus)
        evidence = (assignment * selected_evidence).sum(-1)
        # Confidence must describe the slots that actually supplied the
        # payload.  Using the global content maximum here could report a
        # confident read while the coordinate prior selected a different,
        # poorly matching slot.
        best = selected_content.max(dim=-1).values
        absolute = (
            (best - float(min_similarity))
            / (1.0 - float(min_similarity))
        ).clamp(0.0, 1.0)
        best_part = selected_part.max(dim=-1).values
        if source_part_consistency:
            part_confidence = (
                (best_part - float(min_part_similarity))
                / (1.0 - float(min_part_similarity))
            ).clamp(0.0, 1.0)
        else:
            part_confidence = torch.ones_like(best_part)
        current_support = (
            current_owner[batch_index, query_index].float()
            * torch.sqrt(absolute * evidence.clamp(0.0, 1.0))
            * consensus_gate
            * torch.sqrt(part_confidence)
            * selected_valid.any(dim=-1).float()
        ).clamp(0.0, 1.0)
        residual[batch_index, query_index] = retrieved.to(residual.dtype)
        support[batch_index, query_index] = current_support
        best_similarity[batch_index, query_index] = best
        assigned_evidence[batch_index, query_index] = evidence
        residual_consensus[batch_index, query_index] = consensus
        part_similarity_output[batch_index, query_index] = best_part
        part_confidence_output[batch_index, query_index] = part_confidence
        dominant = assignment.argmax(dim=-1, keepdim=True)
        lineage_output[batch_index, query_index] = torch.gather(
            selected_lineage, dim=-1, index=dominant
        ).squeeze(-1)

    return SourceAddressedRead(
        residual=residual,
        support=support,
        best_similarity=best_similarity,
        assigned_evidence=assigned_evidence,
        residual_consensus=residual_consensus,
        part_similarity=part_similarity_output,
        part_confidence=part_confidence_output,
        lineage_id=lineage_output,
    )


def source_transport_frontier(
    *,
    current_source_key: torch.Tensor,
    current_coordinate: torch.Tensor,
    current_owner: torch.Tensor,
    previous_frontier: PairedEditMemoryState | None,
    current_source_part_signature: torch.Tensor | None = None,
    max_frontier_tokens: int = 192,
    min_similarity: float = 0.10,
    min_part_similarity: float = 0.0,
    coordinate_bias: float = 0.50,
    coordinate_radius: float = 0.60,
    cycle_radius: float = 0.20,
    min_confidence: float = 0.05,
    eps: float = 1e-6,
) -> SourceTransportResult:
    """Transport immutable edit payloads through adjacent source states.

    Matching and cycle verification use only clean-source keys, source-part
    signatures, object coordinates, and the causal owner request.  The
    returned frontier receives current source addresses but copies payloads
    and lineage ids from the previous frontier.  Generated target values are
    deliberately absent from this operation.
    """
    if current_source_key.ndim != 4:
        raise ValueError(
            "current_source_key must have shape [B,L,H,D]"
        )
    batch, length = current_source_key.shape[:2]
    if current_coordinate.shape != (batch, length, 2):
        raise ValueError(
            "current_coordinate must have shape [B,L,2]"
        )
    if current_owner.shape != (batch, length):
        raise ValueError("current_owner must have shape [B,L]")
    if max_frontier_tokens <= 0:
        raise ValueError("max_frontier_tokens must be positive")
    if not -1.0 < min_similarity < 1.0:
        raise ValueError("min_similarity must lie in (-1, 1)")
    if not -1.0 < min_part_similarity < 1.0:
        raise ValueError(
            "min_part_similarity must lie in (-1, 1)"
        )
    if coordinate_bias < 0.0:
        raise ValueError("coordinate_bias must be non-negative")
    if coordinate_radius <= 0.0 or cycle_radius <= 0.0:
        raise ValueError(
            "transport coordinate and cycle radii must be positive"
        )
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must lie in [0, 1]")

    residual = torch.zeros_like(current_source_key)
    support = current_owner.new_zeros((batch, length), dtype=torch.float32)
    similarity_output = support.new_full((batch, length), -1.0)
    evidence_output = support.new_zeros((batch, length))
    cycle_output = support.new_zeros((batch, length))
    part_output = support.new_full((batch, length), -1.0)
    part_confidence_output = support.new_zeros((batch, length))
    lineage_output = torch.full(
        (batch, length),
        -1,
        dtype=torch.long,
        device=current_source_key.device,
    )

    def empty_read() -> SourceAddressedRead:
        return SourceAddressedRead(
            residual=residual,
            support=support,
            best_similarity=similarity_output,
            assigned_evidence=evidence_output,
            residual_consensus=torch.ones_like(support),
            part_similarity=part_output,
            part_confidence=part_confidence_output,
            transported_support=support,
            transport_similarity=similarity_output,
            transport_cycle_confidence=cycle_output,
            lineage_id=lineage_output,
        )

    if previous_frontier is None:
        return SourceTransportResult(read=empty_read(), frontier=None)
    previous_frontier.validate()
    if previous_frontier.source_key.shape[0] != batch:
        raise ValueError(
            "Current source and frontier must share batch size"
        )
    if previous_frontier.source_key.shape[2:] != current_source_key.shape[2:]:
        raise ValueError(
            "Current source and frontier must share head dimensions"
        )
    if (
        current_source_part_signature is not None
        and previous_frontier.source_part_signature is not None
        and current_source_part_signature.shape[-1]
        != previous_frontier.source_part_signature.shape[-1]
    ):
        raise ValueError(
            "Current and frontier source-part signatures must align"
        )

    for batch_index in range(batch):
        query_index = torch.nonzero(
            current_owner[batch_index] > eps, as_tuple=False
        ).flatten()
        previous_evidence = previous_frontier.evidence[batch_index].to(
            current_source_key.device
        ).float()
        memory_index = torch.nonzero(
            previous_evidence > eps, as_tuple=False
        ).flatten()
        if query_index.numel() == 0 or memory_index.numel() == 0:
            continue

        query = F.normalize(
            current_source_key[batch_index, query_index].float(), dim=-1
        )
        previous_key = F.normalize(
            previous_frontier.source_key[batch_index]
            .to(current_source_key.device)[memory_index]
            .float(),
            dim=-1,
        )
        similarity = torch.einsum(
            "qhd,mhd->qm", query, previous_key
        ) / float(query.shape[-2])
        previous_coordinate = previous_frontier.object_coordinate[
            batch_index
        ].to(current_coordinate.device)[memory_index].float()
        query_coordinate = current_coordinate[
            batch_index, query_index
        ].float()
        coordinate_distance = (
            query_coordinate[:, None] - previous_coordinate[None]
        ).square().sum(-1)
        valid = (
            (similarity >= float(min_similarity))
            & (coordinate_distance <= float(coordinate_radius) ** 2)
        )

        part_similarity = torch.ones_like(similarity)
        if (
            current_source_part_signature is not None
            and previous_frontier.source_part_signature is not None
        ):
            query_part = F.normalize(
                current_source_part_signature[
                    batch_index, query_index
                ].float(),
                dim=-1,
                eps=eps,
            )
            previous_part = F.normalize(
                previous_frontier.source_part_signature[batch_index]
                .to(current_source_key.device)[memory_index]
                .float(),
                dim=-1,
                eps=eps,
            )
            part_similarity = torch.einsum(
                "qc,mc->qm", query_part, previous_part
            )
            valid = valid & (
                part_similarity >= float(min_part_similarity)
            )

        score = (
            similarity
            + 0.25 * part_similarity
            - float(coordinate_bias) * coordinate_distance
            + previous_evidence[memory_index].clamp_min(eps).log()[None]
        )
        invalid_score = torch.full_like(score, -1e4)
        candidate_score = torch.where(valid, score, invalid_score)
        forward_score, forward_offset = candidate_score.max(dim=-1)
        forward_valid = forward_score > -1e3

        # Reverse matching closes the source-only correspondence loop.  We
        # permit a nearby return token instead of requiring an exact discrete
        # index, which is important when the visible object changes scale.
        reverse_score, reverse_offset = candidate_score.max(dim=0)
        selected_reverse_offset = reverse_offset[forward_offset]
        returned_coordinate = query_coordinate[selected_reverse_offset]
        cycle_distance = (
            query_coordinate - returned_coordinate
        ).square().sum(-1)
        reverse_valid = reverse_score[forward_offset] > -1e3
        cycle_valid = (
            reverse_valid
            & (cycle_distance <= float(cycle_radius) ** 2)
        )
        cycle_confidence = torch.exp(
            -cycle_distance.sqrt() / float(cycle_radius)
        )

        selected_index = memory_index[forward_offset]
        selected_similarity = torch.gather(
            similarity, 1, forward_offset[:, None]
        ).squeeze(1)
        selected_part = torch.gather(
            part_similarity, 1, forward_offset[:, None]
        ).squeeze(1)
        selected_evidence = previous_evidence[selected_index]
        similarity_confidence = (
            (selected_similarity - float(min_similarity))
            / (1.0 - float(min_similarity))
        ).clamp(0.0, 1.0)
        part_confidence = ((selected_part + 1.0) * 0.5).clamp(0.0, 1.0)
        confidence = (
            current_owner[batch_index, query_index].float()
            * torch.sqrt(
                similarity_confidence
                * selected_evidence.clamp(0.0, 1.0)
            )
            * cycle_confidence
            * torch.sqrt(part_confidence)
            * forward_valid.float()
            * cycle_valid.float()
        ).clamp(0.0, 1.0)
        accepted = confidence >= float(min_confidence)

        selected_residual = previous_frontier.target_value_residual[
            batch_index
        ].to(current_source_key.device)[selected_index]
        if previous_frontier.lineage_id is None:
            previous_lineage = torch.arange(
                previous_frontier.source_key.shape[1],
                device=current_source_key.device,
                dtype=torch.long,
            )
        else:
            previous_lineage = previous_frontier.lineage_id[
                batch_index
            ].to(current_source_key.device)
        selected_lineage = previous_lineage[selected_index]

        residual[batch_index, query_index] = torch.where(
            accepted[:, None, None],
            selected_residual,
            torch.zeros_like(selected_residual),
        ).to(residual.dtype)
        support[batch_index, query_index] = confidence * accepted.float()
        similarity_output[batch_index, query_index] = selected_similarity
        evidence_output[batch_index, query_index] = selected_evidence
        cycle_output[batch_index, query_index] = (
            cycle_confidence * cycle_valid.float()
        )
        part_output[batch_index, query_index] = selected_part
        part_confidence_output[batch_index, query_index] = part_confidence
        lineage_output[batch_index, query_index] = torch.where(
            accepted, selected_lineage, torch.full_like(selected_lineage, -1)
        )

    if not bool((support > eps).any()):
        # Do not let a stale frontier jump across an unsupported source
        # transition.  Long-range re-association remains the responsibility
        # of the immutable canonical bank.
        return SourceTransportResult(read=empty_read(), frontier=None)

    selected_keys = []
    selected_residuals = []
    selected_coordinates = []
    selected_evidence = []
    selected_part_signatures = []
    selected_lineages = []
    selected_transport_confidence = []
    for batch_index in range(batch):
        candidates = torch.nonzero(
            support[batch_index] > eps, as_tuple=False
        ).flatten()
        count = min(max_frontier_tokens, int(candidates.numel()))
        if count:
            chosen = candidates[
                support[batch_index, candidates].topk(count).indices
            ]
        else:
            chosen = candidates
        pad = max_frontier_tokens - count
        selected_keys.append(F.pad(
            current_source_key[batch_index, chosen],
            (0, 0, 0, 0, 0, pad),
        ))
        selected_residuals.append(F.pad(
            residual[batch_index, chosen],
            (0, 0, 0, 0, 0, pad),
        ))
        selected_coordinates.append(F.pad(
            current_coordinate[batch_index, chosen], (0, 0, 0, pad)
        ))
        # Evidence belongs to the immutable payload lineage.  It must not be
        # multiplied again at every transport hop; fresh correspondence and
        # cycle confidence are stored separately and recomputed on each hop.
        selected_evidence.append(F.pad(
            evidence_output[batch_index, chosen], (0, pad)
        ))
        selected_lineages.append(F.pad(
            lineage_output[batch_index, chosen], (0, pad), value=-1
        ))
        selected_transport_confidence.append(F.pad(
            cycle_output[batch_index, chosen], (0, pad)
        ))
        if current_source_part_signature is not None:
            selected_part_signatures.append(F.pad(
                current_source_part_signature[batch_index, chosen],
                (0, 0, 0, pad),
            ))

    frontier = PairedEditMemoryState(
        source_key=torch.stack(selected_keys).detach(),
        target_value_residual=torch.stack(selected_residuals).detach(),
        object_coordinate=torch.stack(selected_coordinates).detach(),
        evidence=torch.stack(selected_evidence).detach(),
        source_part_signature=(
            torch.stack(selected_part_signatures).detach()
            if selected_part_signatures
            else None
        ),
        lineage_id=torch.stack(selected_lineages).detach(),
        transport_confidence=torch.stack(
            selected_transport_confidence
        ).detach(),
    )
    frontier.validate()
    return SourceTransportResult(read=empty_read(), frontier=frontier)


class CausalPairedEditMemory:
    """Training-free sparse memory with transactional target writes.

    The address and payload deliberately come from different branches:
    clean-source keys address the memory, while target-minus-source values
    carry the edit. Low-confidence observations remain available through the
    native recent KV cache but cannot update this canonical memory.
    """

    def __init__(
        self,
        *,
        layers: tuple[int, ...],
        max_tokens: int = 1536,
        max_tokens_per_block: int = 192,
        min_commit_confidence: float = 0.20,
        min_similarity: float = 0.35,
        coordinate_bias: float = 1.0,
        coordinate_radius: float = 0.0,
        min_residual_consensus: float = 0.0,
        source_part_consistency: bool = False,
        min_part_similarity: float = 0.45,
        part_similarity_margin: float = 0.08,
        topk: int = 8,
        source_transport: bool = False,
        transport_min_similarity: float = 0.10,
        transport_coordinate_radius: float = 0.60,
        transport_cycle_radius: float = 0.20,
        transport_min_confidence: float = 0.05,
        single_confidence: bool = False,
        immutable_canonical_key_anchor: bool = False,
        eps: float = 1e-6,
    ):
        if not layers:
            raise ValueError("Paired edit memory requires at least one layer")
        if max_tokens <= 0 or max_tokens_per_block <= 0 or topk <= 0:
            raise ValueError("Paired edit-memory capacities must be positive")
        if not 0.0 <= min_commit_confidence <= 1.0:
            raise ValueError(
                "min_commit_confidence must lie in [0, 1]"
            )
        if coordinate_radius < 0.0:
            raise ValueError("coordinate_radius must be non-negative")
        if not 0.0 <= min_residual_consensus < 1.0:
            raise ValueError(
                "min_residual_consensus must lie in [0, 1)"
            )
        if not -1.0 < min_part_similarity < 1.0:
            raise ValueError(
                "min_part_similarity must lie in (-1, 1)"
            )
        if not 0.0 <= part_similarity_margin <= 2.0:
            raise ValueError(
                "part_similarity_margin must lie in [0, 2]"
            )
        if not -1.0 < transport_min_similarity < 1.0:
            raise ValueError(
                "transport_min_similarity must lie in (-1, 1)"
            )
        if transport_coordinate_radius <= 0.0:
            raise ValueError(
                "transport_coordinate_radius must be positive"
            )
        if transport_cycle_radius <= 0.0:
            raise ValueError(
                "transport_cycle_radius must be positive"
            )
        if not 0.0 <= transport_min_confidence <= 1.0:
            raise ValueError(
                "transport_min_confidence must lie in [0, 1]"
            )
        self.layers = tuple(int(layer) for layer in layers)
        self.max_tokens = int(max_tokens)
        self.max_tokens_per_block = int(max_tokens_per_block)
        self.min_commit_confidence = float(min_commit_confidence)
        self.min_similarity = float(min_similarity)
        self.coordinate_bias = float(coordinate_bias)
        self.coordinate_radius = float(coordinate_radius)
        self.min_residual_consensus = float(min_residual_consensus)
        self.source_part_consistency = bool(source_part_consistency)
        self.min_part_similarity = float(min_part_similarity)
        self.part_similarity_margin = float(part_similarity_margin)
        self.topk = int(topk)
        self.source_transport = bool(source_transport)
        self.transport_min_similarity = float(
            transport_min_similarity
        )
        self.transport_coordinate_radius = float(
            transport_coordinate_radius
        )
        self.transport_cycle_radius = float(transport_cycle_radius)
        self.transport_min_confidence = float(
            transport_min_confidence
        )
        self.single_confidence = bool(single_confidence)
        self.immutable_canonical_key_anchor = bool(
            immutable_canonical_key_anchor
        )
        self.eps = float(eps)
        self._states: Dict[int, PairedEditMemoryState] = {}
        self._frontiers: Dict[int, PairedEditMemoryState] = {}
        # The ignition bank is deliberately separate from ``_states``.
        # ``_states`` grows with later clean-source addresses so it can
        # follow pose and viewpoint changes; the ignition bank contains only
        # the first-block source keys and target-minus-source value residuals.
        # Replay may invalidate a lineage or lower its evidence, but may
        # never replace either immutable tensor.
        self._ignition_states: Dict[int, PairedEditMemoryState] = {}
        self._last_reads: Dict[int, SourceAddressedRead] = {}
        self._ignition_verified_layers: set[int] = set()
        # Once a source-transported layer has accepted an ignition payload,
        # it may never fall back to an unconditional write.  In particular,
        # a replay that rejects every lineage must leave the method in a safe
        # abstaining state instead of treating a later generated block as a
        # new identity source.
        self._payload_initialized_layers: set[int] = set()

    def export(self) -> Mapping[int, PairedEditMemoryState]:
        return self._states

    def export_frontier(self) -> Mapping[int, PairedEditMemoryState]:
        return self._frontiers

    def export_ignition(self) -> Mapping[int, PairedEditMemoryState]:
        """Return the immutable first-block canonical key/residual bank.

        Callers must treat the returned mapping as read-only.  The bank is
        intentionally not synthesized from ``_states`` because later
        source-address commits are allowed to extend that moving manifold.
        """
        return self._ignition_states

    def ignition_is_verified(self) -> bool:
        """Return whether every frozen layer survived replay checking."""
        return bool(self._ignition_states) and all(
            layer in self._ignition_verified_layers
            for layer in self._ignition_states
        )

    @torch.no_grad()
    def build_canonical_anchor_requests(
        self,
        reads: Mapping[int, SourceAddressedRead],
        current_coordinate: torch.Tensor,
    ) -> Dict[int, ImmutableCanonicalAnchor]:
        """Pair current binary admission with immutable ignition K/dV.

        Retrieval confidence decides whether a current owner token may read
        the bank, but it does not attenuate the value payload.  Stored
        ignition evidence is consumed later as an attention-logit prior.
        This separates uncertainty from edit magnitude and guarantees that
        unsupported queries retain the exact native output.
        """
        requests: Dict[int, ImmutableCanonicalAnchor] = {}
        if current_coordinate.ndim != 3 or current_coordinate.shape[-1] != 2:
            raise ValueError(
                "Current canonical coordinates must have shape [B,Q,2]"
            )
        if not self.ignition_is_verified():
            return requests
        for layer in self.layers:
            state = self._ignition_states.get(layer)
            read = reads.get(layer)
            if state is None or read is None:
                continue
            state.validate()
            if state.lineage_id is None:
                raise RuntimeError(
                    "Immutable canonical anchors require lineage ids"
                )
            query_lineage = read.lineage_id
            if query_lineage is None:
                raise RuntimeError(
                    "Canonical anchor reads require query lineage ids"
                )
            if query_lineage.shape != read.support.shape:
                raise ValueError(
                    "Canonical query lineage must align with read support"
                )
            request_device = query_lineage.device
            key_lineage = state.lineage_id.to(query_lineage.device)
            key_valid = state.evidence.to(query_lineage.device) > self.eps
            exact_lineage = (
                query_lineage[:, :, None] == key_lineage[:, None, :]
            ) & (query_lineage[:, :, None] >= 0)
            lineage_available = exact_lineage & key_valid[:, None, :]
            query_support = (
                (read.support > self.eps)
                & (query_lineage >= 0)
                & lineage_available.any(dim=-1)
            ).float()

            # Transported lineage supplies a precise seed.  Expand that seed
            # only to a local, source-part-consistent ignition neighborhood
            # so cross-attention has more than one useful key without mixing
            # cap/body or separate object parts.  This is geometry-relative,
            # not an absolute-size constraint.
            coordinate = current_coordinate.to(request_device).float()
            canonical_coordinate = state.object_coordinate.to(
                request_device
            ).float()
            coordinate_distance = (
                coordinate[:, :, None]
                - canonical_coordinate[:, None, :]
            ).square().sum(-1)
            neighborhood_radius = (
                self.coordinate_radius
                if self.coordinate_radius > 0.0
                else self.transport_cycle_radius
            )
            local_neighbor = coordinate_distance <= float(
                neighborhood_radius
            ) ** 2
            if (
                self.source_part_consistency
                and read.source_value is not None
                and state.source_part_signature is not None
            ):
                query_part = build_source_part_signature(
                    read.source_value, eps=self.eps
                ).to(request_device)
                key_part = state.source_part_signature.to(
                    request_device
                ).float()
                part_similarity = torch.einsum(
                    "bqc,bmc->bqm", query_part.float(), key_part
                )
                masked_part = torch.where(
                    local_neighbor & key_valid[:, None, :],
                    part_similarity,
                    torch.full_like(part_similarity, -1e4),
                )
                best_part = masked_part.max(dim=-1, keepdim=True).values
                local_neighbor = (
                    local_neighbor
                    & (part_similarity >= self.min_part_similarity)
                    & (
                        part_similarity
                        >= best_part - self.part_similarity_margin
                    )
                )
            query_key_mask = (
                exact_lineage | local_neighbor.to(exact_lineage.device)
            ) & key_valid[:, None, :]
            query_key_mask &= query_support.bool()[:, :, None]
            request = ImmutableCanonicalAnchor(
                # Materialize the small sparse bank once per block on the
                # active device.  Attention then reuses it across denoising
                # steps instead of copying CPU state at every layer call.
                source_key=state.source_key.to(request_device),
                target_value_residual=(
                    state.target_value_residual.to(request_device)
                ),
                object_coordinate=state.object_coordinate.to(
                    request_device
                ),
                evidence=state.evidence.to(request_device),
                lineage_id=state.lineage_id.to(request_device),
                query_support=query_support,
                query_lineage_id=query_lineage.detach(),
                query_key_mask=query_key_mask.detach().to(request_device),
            )
            request.validate()
            requests[layer] = request
        return requests

    @staticmethod
    def _state_to(
        state: PairedEditMemoryState, device: torch.device | str
    ) -> PairedEditMemoryState:
        return PairedEditMemoryState(
            source_key=state.source_key.to(device),
            target_value_residual=state.target_value_residual.to(device),
            object_coordinate=state.object_coordinate.to(device),
            evidence=state.evidence.to(device),
            source_part_signature=(
                None
                if state.source_part_signature is None
                else state.source_part_signature.to(device)
            ),
            lineage_id=(
                None
                if state.lineage_id is None
                else state.lineage_id.to(device)
            ),
            transport_confidence=(
                None
                if state.transport_confidence is None
                else state.transport_confidence.to(device)
            ),
        )

    def to(self, device: torch.device | str) -> None:
        """Move sparse state explicitly for low-memory inference."""
        self._states = {
            layer: self._state_to(state, device)
            for layer, state in self._states.items()
        }
        self._frontiers = {
            layer: self._state_to(state, device)
            for layer, state in self._frontiers.items()
        }
        self._ignition_states = {
            layer: self._state_to(state, device)
            for layer, state in self._ignition_states.items()
        }
        self._last_reads = {}

    def has_state(self) -> bool:
        has_valid_slots = any(
            bool((state.evidence > self.eps).any())
            for state in self._states.values()
        )
        return has_valid_slots or (
            self.source_transport
            and bool(self._payload_initialized_layers)
        )

    def compatible_with(self, other: "CausalPairedEditMemory") -> bool:
        """Return whether an injected rollout state matches this config."""
        return (
            self.layers == other.layers
            and self.max_tokens == other.max_tokens
            and self.max_tokens_per_block == other.max_tokens_per_block
            and self.min_commit_confidence
            == other.min_commit_confidence
            and self.min_similarity == other.min_similarity
            and self.coordinate_bias == other.coordinate_bias
            and self.coordinate_radius == other.coordinate_radius
            and self.min_residual_consensus
            == other.min_residual_consensus
            and self.source_part_consistency
            == other.source_part_consistency
            and self.min_part_similarity == other.min_part_similarity
            and self.part_similarity_margin
            == other.part_similarity_margin
            and self.topk == other.topk
            and self.source_transport == other.source_transport
            and self.transport_min_similarity
            == other.transport_min_similarity
            and self.transport_coordinate_radius
            == other.transport_coordinate_radius
            and self.transport_cycle_radius
            == other.transport_cycle_radius
            and self.transport_min_confidence
            == other.transport_min_confidence
            and self.single_confidence == other.single_confidence
            and self.immutable_canonical_key_anchor
            == other.immutable_canonical_key_anchor
        )

    @staticmethod
    def _merge_canonical_and_transport_reads(
        canonical: SourceAddressedRead,
        transported: SourceAddressedRead,
    ) -> SourceAddressedRead:
        """Choose the more reliable source-grounded payload per query."""
        use_transport = transported.support > canonical.support

        def choose(
            transported_value: torch.Tensor | None,
            canonical_value: torch.Tensor | None,
        ) -> torch.Tensor | None:
            if transported_value is None:
                return canonical_value
            if canonical_value is None:
                return transported_value
            gate = use_transport
            while gate.ndim < transported_value.ndim:
                gate = gate.unsqueeze(-1)
            return torch.where(gate, transported_value, canonical_value)

        return SourceAddressedRead(
            residual=choose(transported.residual, canonical.residual),
            support=torch.maximum(
                canonical.support, transported.support
            ),
            best_similarity=choose(
                transported.best_similarity, canonical.best_similarity
            ),
            assigned_evidence=choose(
                transported.assigned_evidence,
                canonical.assigned_evidence,
            ),
            residual_consensus=choose(
                transported.residual_consensus,
                canonical.residual_consensus,
            ),
            source_value=canonical.source_value,
            part_similarity=choose(
                transported.part_similarity, canonical.part_similarity
            ),
            part_confidence=choose(
                transported.part_confidence, canonical.part_confidence
            ),
            canonical_support=canonical.support,
            transported_support=transported.support,
            transport_similarity=transported.transport_similarity,
            transport_cycle_confidence=(
                transported.transport_cycle_confidence
            ),
            lineage_id=choose(
                transported.lineage_id, canonical.lineage_id
            ),
        )

    @torch.no_grad()
    def read(
        self,
        *,
        source_kv_cache,
        current_coordinate: torch.Tensor,
        current_object_request: torch.Tensor,
        current_transport_owner: torch.Tensor | None = None,
    ) -> Dict[int, SourceAddressedRead]:
        """Read every configured layer from its clean-source address.

        ``current_object_request`` is the role-specific gate assembled by
        the pipeline (causal owner times target-memory action).  The returned
        support additionally contains source-match and stored-evidence
        confidence, so a zero support is an exact abstention.
        """
        if current_object_request.ndim != 2:
            raise ValueError(
                "current_object_request must have shape [B,L]"
            )
        if current_coordinate.shape != (
            *current_object_request.shape, 2
        ):
            raise ValueError(
                "current_coordinate must have shape [B,L,2]"
            )
        if current_transport_owner is None:
            current_transport_owner = current_object_request
        if current_transport_owner.shape != current_object_request.shape:
            raise ValueError(
                "current_transport_owner must align with object request"
            )
        reads: Dict[int, SourceAddressedRead] = {}
        for layer in self.layers:
            source_cache = source_kv_cache[layer]
            num_new_tokens = int(source_cache.get("num_new_tokens", 0))
            if num_new_tokens != current_object_request.shape[1]:
                raise ValueError(
                    "Paired-memory read must align with the clean-source "
                    "KV write"
                )
            if int(source_cache.get("sink_tokens", 0)) != 0:
                raise ValueError(
                    "Paired edit memory requires unrotated sink-free keys"
                )
            source_end = int(source_cache["local_end_index"].item())
            current_source_key = source_cache["k"][
                :, source_end - num_new_tokens:source_end
            ].detach()
            source_value = source_cache["v"][
                :, source_end - num_new_tokens:source_end
            ].detach()
            current_part_signature = (
                build_source_part_signature(
                    source_value, eps=self.eps
                )
                if self.source_part_consistency
                else None
            )
            canonical_read = source_addressed_residual_read(
                current_source_key=current_source_key,
                current_coordinate=current_coordinate,
                current_owner=current_object_request,
                memory=self._states.get(layer),
                current_source_part_signature=(
                    current_part_signature
                ),
                topk=self.topk,
                min_similarity=self.min_similarity,
                coordinate_bias=self.coordinate_bias,
                coordinate_radius=self.coordinate_radius,
                min_residual_consensus=self.min_residual_consensus,
                source_part_consistency=(
                    self.source_part_consistency
                ),
                min_part_similarity=self.min_part_similarity,
                part_similarity_margin=self.part_similarity_margin,
                eps=self.eps,
            )
            read = canonical_read
            if self.source_transport:
                previous_frontier = self._frontiers.get(layer)
                transported = source_transport_frontier(
                    current_source_key=current_source_key,
                    current_coordinate=current_coordinate,
                    current_owner=current_transport_owner,
                    previous_frontier=previous_frontier,
                    current_source_part_signature=current_part_signature,
                    max_frontier_tokens=self.max_tokens_per_block,
                    min_similarity=self.transport_min_similarity,
                    # Adjacent source transport uses mutual-cycle geometry
                    # for precision, so part appearance is a soft cue here.
                    # The stricter canonical read keeps its configured part
                    # threshold for long-range re-association.
                    min_part_similarity=0.0,
                    coordinate_bias=self.coordinate_bias,
                    coordinate_radius=self.transport_coordinate_radius,
                    cycle_radius=self.transport_cycle_radius,
                    min_confidence=self.transport_min_confidence,
                    eps=self.eps,
                )
                transport_gate = torch.where(
                    current_transport_owner > self.eps,
                    current_object_request
                    / current_transport_owner.clamp_min(self.eps),
                    torch.zeros_like(current_object_request),
                ).clamp(0.0, 1.0)
                transported_read = SourceAddressedRead(
                    residual=transported.read.residual,
                    support=(
                        transported.read.support * transport_gate
                    ),
                    best_similarity=transported.read.best_similarity,
                    assigned_evidence=transported.read.assigned_evidence,
                    residual_consensus=(
                        transported.read.residual_consensus
                    ),
                    part_similarity=transported.read.part_similarity,
                    part_confidence=transported.read.part_confidence,
                    transported_support=(
                        transported.read.support * transport_gate
                    ),
                    transport_similarity=(
                        transported.read.transport_similarity
                    ),
                    transport_cycle_confidence=(
                        transported.read.transport_cycle_confidence
                    ),
                    lineage_id=transported.read.lineage_id,
                )
                read = self._merge_canonical_and_transport_reads(
                    canonical_read, transported_read
                )
                if transported.frontier is not None:
                    self._frontiers[layer] = transported.frontier
                else:
                    self._frontiers.pop(layer, None)
            reads[layer] = SourceAddressedRead(
                residual=read.residual,
                support=read.support,
                best_similarity=read.best_similarity,
                assigned_evidence=read.assigned_evidence,
                residual_consensus=read.residual_consensus,
                source_value=source_value,
                part_similarity=read.part_similarity,
                part_confidence=read.part_confidence,
                canonical_support=read.canonical_support,
                transported_support=read.transported_support,
                transport_similarity=read.transport_similarity,
                transport_cycle_confidence=(
                    read.transport_cycle_confidence
                ),
                lineage_id=read.lineage_id,
            )
        self._last_reads = reads
        return reads

    @torch.no_grad()
    def project_target_cache(
        self,
        *,
        source_kv_cache,
        target_kv_cache,
        reads: Mapping[int, SourceAddressedRead],
        strength: float,
    ) -> Dict[str, torch.Tensor]:
        """Project freshly written target values onto paired anchors.

        This runs after the transactional commit has inspected the untouched
        target observation.  It therefore closes the recurrent KV loop
        without letting a projection validate and rewrite itself.
        """
        if not 0.0 <= float(strength) <= 1.0:
            raise ValueError(
                "Paired cache projection strength must lie in [0, 1]"
            )
        corrections = []
        supports = []
        for layer in self.layers:
            read = reads.get(layer)
            if read is None:
                continue
            source_cache = source_kv_cache[layer]
            target_cache = target_kv_cache[layer]
            num_new_tokens = int(target_cache.get("num_new_tokens", 0))
            if (
                num_new_tokens <= 0
                or read.support.shape[1] != num_new_tokens
                or read.residual.shape[1] != num_new_tokens
            ):
                raise ValueError(
                    "Paired cache projection must align with current KV "
                    "tokens"
                )
            source_end = int(source_cache["local_end_index"].item())
            target_end = int(target_cache["local_end_index"].item())
            source_value = source_cache["v"][
                :, source_end - num_new_tokens:source_end
            ]
            target_value = target_cache["v"][
                :, target_end - num_new_tokens:target_end
            ]
            gate = (
                float(strength)
                * read.support.to(target_value.device).float()
                .clamp(0.0, 1.0)
            )[:, :, None, None]
            desired_value = (
                source_value.float()
                + read.residual.to(target_value.device).float()
            )
            corrected = target_value.float() + gate * (
                desired_value - target_value.float()
            )
            projected = torch.where(
                gate > 0.0,
                corrected.to(target_value.dtype),
                target_value,
            )
            correction = (
                projected.float() - target_value.float()
            ).abs().mean(dim=(-1, -2))
            gate = read.support.to(target_value.device).float() > self.eps
            target_cache["v"][
                :, target_end - num_new_tokens:target_end
            ] = projected
            corrections.append(correction)
            supports.append(gate.float())
        if not corrections:
            raise RuntimeError(
                "No configured layer was available for paired cache "
                "projection"
            )
        correction = torch.stack(corrections).mean(0)
        support = torch.stack(supports).mean(0)
        return {
            "correction": correction,
            "support": support,
        }

    @torch.no_grad()
    def _transactional_weight(
        self,
        *,
        source_key: torch.Tensor,
        source_part_signature: torch.Tensor | None = None,
        target_residual: torch.Tensor,
        coordinate: torch.Tensor,
        proposal_weight: torch.Tensor,
        previous: PairedEditMemoryState | None,
        transactional: bool,
    ) -> TransactionalCommit:
        zeros = proposal_weight.new_zeros(proposal_weight.shape)
        if previous is None or not transactional:
            write = proposal_weight.float().clamp(0.0, 1.0)
            accepted = write >= self.min_commit_confidence
            return TransactionalCommit(
                write_weight=write * accepted.float(),
                source_match=torch.where(accepted, torch.ones_like(write), zeros),
                residual_agreement=torch.where(
                    accepted, torch.ones_like(write), zeros
                ),
                accepted=accepted,
                canonical_residual=target_residual,
                lineage_id=torch.arange(
                    source_key.shape[1],
                    device=source_key.device,
                    dtype=torch.long,
                )[None].expand(source_key.shape[0], -1),
            )

        read = source_addressed_residual_read(
            current_source_key=source_key,
            current_coordinate=coordinate,
            # The proposal is multiplied exactly once below.  Retrieval is
            # an independent consistency check, not a second copy of the
            # role/ownership probability.
            current_owner=(proposal_weight > self.eps).float(),
            memory=previous,
            current_source_part_signature=source_part_signature,
            topk=self.topk,
            min_similarity=self.min_similarity,
            coordinate_bias=self.coordinate_bias,
            coordinate_radius=self.coordinate_radius,
            min_residual_consensus=self.min_residual_consensus,
            source_part_consistency=self.source_part_consistency,
            min_part_similarity=self.min_part_similarity,
            part_similarity_margin=self.part_similarity_margin,
            eps=self.eps,
        )
        current_flat = target_residual.float().flatten(2)
        retrieved_flat = read.residual.float().flatten(2)
        cosine = F.cosine_similarity(
            current_flat, retrieved_flat, dim=-1, eps=self.eps
        )
        cosine_agreement = cosine.clamp_min(0.0)
        current_norm = current_flat.norm(dim=-1).clamp_min(self.eps)
        retrieved_norm = retrieved_flat.norm(dim=-1).clamp_min(self.eps)
        magnitude_agreement = torch.exp(
            -(current_norm / retrieved_norm).log().abs()
        )
        residual_agreement = torch.sqrt(
            cosine_agreement * magnitude_agreement
        ).clamp(0.0, 1.0)
        # Transactional means all independent checks must agree. A failed
        # observation stays in the ordinary recent cache, never canonical.
        write = (
            proposal_weight.float().clamp(0.0, 1.0)
            * read.support
            * residual_agreement
        ).clamp(0.0, 1.0)
        accepted = write >= self.min_commit_confidence
        return TransactionalCommit(
            write_weight=write * accepted.float(),
            source_match=read.support,
            residual_agreement=residual_agreement,
            accepted=accepted,
            # Accepted later views extend the address manifold, but inherit
            # the matched canonical payload. Writing the current observation
            # here would slowly turn an agreed direction into appearance
            # drift through repeated self-generated commits.
            canonical_residual=read.residual,
            lineage_id=read.lineage_id,
        )

    def _source_transport_transaction(
        self,
        *,
        layer: int,
        source_key: torch.Tensor,
        target_residual: torch.Tensor,
        proposal_weight: torch.Tensor,
    ) -> TransactionalCommit:
        """Let target observations verify, but never rewrite, lineage."""
        read = self._last_reads.get(layer)
        if read is None or read.support.shape != proposal_weight.shape:
            zeros = proposal_weight.new_zeros(proposal_weight.shape)
            return TransactionalCommit(
                write_weight=zeros,
                source_match=zeros,
                residual_agreement=zeros,
                accepted=zeros.bool(),
                canonical_residual=torch.zeros_like(source_key),
                lineage_id=torch.full(
                    proposal_weight.shape,
                    -1,
                    device=proposal_weight.device,
                    dtype=torch.long,
                ),
            )
        current_flat = target_residual.float().flatten(2)
        inherited_flat = read.residual.float().flatten(2)
        cosine_agreement = F.cosine_similarity(
            current_flat, inherited_flat, dim=-1, eps=self.eps
        ).clamp_min(0.0)
        current_norm = current_flat.norm(dim=-1).clamp_min(self.eps)
        inherited_norm = inherited_flat.norm(dim=-1).clamp_min(self.eps)
        magnitude_agreement = torch.exp(
            -(current_norm / inherited_norm).log().abs()
        )
        residual_agreement = torch.sqrt(
            cosine_agreement * magnitude_agreement
        ).clamp(0.0, 1.0)
        # ``read.support`` already contains the role/owner proposal exactly
        # once.  Multiplying proposal again makes transactional promotion
        # quadratic in role confidence and prevents later poses from
        # extending the address manifold.  The single-confidence ablation
        # therefore uses the successful read directly; the legacy behavior
        # remains selectable for controlled comparison with 933.
        write = (
            read.support.float()
            * residual_agreement
            * (
                1.0
                if self.single_confidence
                else proposal_weight.float().clamp(0.0, 1.0)
            )
        ).clamp(0.0, 1.0)
        accepted = write >= self.min_commit_confidence
        return TransactionalCommit(
            write_weight=write * accepted.float(),
            source_match=read.support,
            residual_agreement=residual_agreement,
            accepted=accepted,
            canonical_residual=read.residual,
            lineage_id=read.lineage_id,
        )

    @torch.no_grad()
    def commit(
        self,
        *,
        source_kv_cache,
        target_kv_cache,
        proposal_weight: torch.Tensor,
        object_coordinate: torch.Tensor,
        transactional: bool,
        preserve_canonical_payload: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if proposal_weight.ndim != 2:
            raise ValueError("proposal_weight must have shape [B,L]")
        if object_coordinate.shape != (*proposal_weight.shape, 2):
            raise ValueError(
                "object_coordinate must have shape [B,L,2]"
            )
        diagnostics = []
        for layer in self.layers:
            source_cache = source_kv_cache[layer]
            target_cache = target_kv_cache[layer]
            num_new_tokens = int(target_cache["num_new_tokens"])
            if num_new_tokens != proposal_weight.shape[1]:
                raise ValueError(
                    "Paired memory proposal must align with the KV write"
                )
            if int(source_cache.get("sink_tokens", 0)) != 0 or int(
                target_cache.get("sink_tokens", 0)
            ) != 0:
                raise ValueError(
                    "Paired edit memory requires unrotated sink-free keys"
                )
            source_end = int(source_cache["local_end_index"].item())
            target_end = int(target_cache["local_end_index"].item())
            source_key = source_cache["k"][
                :, source_end - num_new_tokens:source_end
            ].detach()
            source_value = source_cache["v"][
                :, source_end - num_new_tokens:source_end
            ].detach()
            target_value = target_cache["v"][
                :, target_end - num_new_tokens:target_end
            ].detach()
            target_residual = target_value.float() - source_value.float()
            source_part_signature = (
                build_source_part_signature(
                    source_value, eps=self.eps
                )
                if self.source_part_consistency
                else None
            )
            if self.source_transport and transactional:
                transaction = self._source_transport_transaction(
                    layer=layer,
                    source_key=source_key,
                    target_residual=target_residual,
                    proposal_weight=proposal_weight,
                )
            else:
                transaction = self._transactional_weight(
                    source_key=source_key,
                    source_part_signature=source_part_signature,
                    target_residual=target_residual,
                    coordinate=object_coordinate,
                    proposal_weight=proposal_weight,
                    previous=self._states.get(layer),
                    transactional=transactional,
                )

            selected_keys = []
            selected_residuals = []
            selected_coordinates = []
            selected_evidence = []
            selected_part_signatures = []
            selected_lineages = []
            batch = proposal_weight.shape[0]
            for batch_index in range(batch):
                candidates = torch.nonzero(
                    transaction.accepted[batch_index], as_tuple=False
                ).flatten()
                count = min(
                    self.max_tokens_per_block, int(candidates.numel())
                )
                if count > 0:
                    chosen_offset = torch.topk(
                        transaction.write_weight[batch_index, candidates],
                        k=count,
                    ).indices
                    chosen = candidates[chosen_offset]
                else:
                    chosen = candidates
                pad = self.max_tokens_per_block - count
                selected_keys.append(
                    F.pad(
                        source_key[batch_index, chosen],
                        (0, 0, 0, 0, 0, pad),
                    )
                )
                selected_residuals.append(
                    F.pad(
                        (
                            transaction.canonical_residual
                            if (
                                preserve_canonical_payload
                                or (self.source_transport and transactional)
                            )
                            else target_residual
                        )[batch_index, chosen],
                        (0, 0, 0, 0, 0, pad),
                    )
                )
                selected_coordinates.append(
                    F.pad(
                        object_coordinate[batch_index, chosen],
                        (0, 0, 0, pad),
                    )
                )
                selected_evidence.append(
                    F.pad(transaction.write_weight[batch_index, chosen], (0, pad))
                )
                lineage = transaction.lineage_id
                if lineage is None:
                    lineage = torch.arange(
                        source_key.shape[1],
                        device=source_key.device,
                        dtype=torch.long,
                    )[None].expand(source_key.shape[0], -1)
                selected_lineages.append(
                    F.pad(
                        lineage[batch_index, chosen],
                        (0, pad),
                        value=-1,
                    )
                )
                if source_part_signature is not None:
                    selected_part_signatures.append(
                        F.pad(
                            source_part_signature[batch_index, chosen],
                            (0, 0, 0, pad),
                        )
                    )
            appended = PairedEditMemoryState(
                source_key=torch.stack(selected_keys),
                target_value_residual=torch.stack(selected_residuals).to(
                    target_value.dtype
                ),
                object_coordinate=torch.stack(selected_coordinates),
                evidence=torch.stack(selected_evidence),
                source_part_signature=(
                    torch.stack(selected_part_signatures)
                    if selected_part_signatures
                    else None
                ),
                lineage_id=torch.stack(selected_lineages),
                transport_confidence=torch.stack(selected_evidence),
            )
            current_addresses = appended
            previous = self._states.get(layer)
            if previous is not None:
                appended = PairedEditMemoryState(
                    source_key=torch.cat(
                        [
                            previous.source_key.to(source_key.device),
                            appended.source_key,
                        ],
                        dim=1,
                    ),
                    target_value_residual=torch.cat(
                        [
                            previous.target_value_residual.to(target_value.device),
                            appended.target_value_residual,
                        ],
                        dim=1,
                    ),
                    object_coordinate=torch.cat(
                        [
                            previous.object_coordinate.to(
                                object_coordinate.device
                            ),
                            appended.object_coordinate,
                        ],
                        dim=1,
                    ),
                    evidence=torch.cat(
                        [
                            previous.evidence.to(proposal_weight.device),
                            appended.evidence,
                        ],
                        dim=1,
                    ),
                    source_part_signature=(
                        torch.cat(
                            [
                                previous.source_part_signature.to(
                                    source_key.device
                                ),
                                appended.source_part_signature,
                            ],
                            dim=1,
                        )
                        if self.source_part_consistency
                        else None
                    ),
                    lineage_id=torch.cat(
                        [
                            (
                                previous.lineage_id.to(source_key.device)
                                if previous.lineage_id is not None
                                else torch.full(
                                    previous.evidence.shape,
                                    -1,
                                    device=source_key.device,
                                    dtype=torch.long,
                                )
                            ),
                            appended.lineage_id,
                        ],
                        dim=1,
                    ),
                    transport_confidence=torch.cat(
                        [
                            (
                                previous.transport_confidence.to(
                                    proposal_weight.device
                                )
                                if previous.transport_confidence is not None
                                else previous.evidence.to(
                                    proposal_weight.device
                                )
                            ),
                            appended.transport_confidence,
                        ],
                        dim=1,
                    ),
                )
            if appended.source_key.shape[1] > self.max_tokens:
                keep = min(self.max_tokens, appended.source_key.shape[1])
                keep_index = appended.evidence.topk(keep, dim=-1).indices
                def gather(value: torch.Tensor) -> torch.Tensor:
                    index = keep_index
                    while index.ndim < value.ndim:
                        index = index.unsqueeze(-1)
                    return torch.gather(
                        value,
                        1,
                        index.expand(
                            *keep_index.shape, *value.shape[2:]
                        ),
                    )
                appended = PairedEditMemoryState(
                    source_key=gather(appended.source_key),
                    target_value_residual=gather(
                        appended.target_value_residual
                    ),
                    object_coordinate=gather(appended.object_coordinate),
                    evidence=gather(appended.evidence),
                    source_part_signature=(
                        gather(appended.source_part_signature)
                        if appended.source_part_signature is not None
                        else None
                    ),
                    lineage_id=gather(appended.lineage_id),
                    transport_confidence=gather(
                        appended.transport_confidence
                    ),
                )
            appended.validate()
            self._states[layer] = PairedEditMemoryState(
                source_key=appended.source_key.detach(),
                target_value_residual=(
                    appended.target_value_residual.detach()
                ),
                object_coordinate=appended.object_coordinate.detach(),
                evidence=appended.evidence.detach(),
                source_part_signature=(
                    appended.source_part_signature.detach()
                    if appended.source_part_signature is not None
                    else None
                ),
                lineage_id=appended.lineage_id.detach(),
                transport_confidence=(
                    appended.transport_confidence.detach()
                ),
            )
            if (
                self.immutable_canonical_key_anchor
                and not transactional
                and layer not in self._ignition_states
            ):
                # Bootstrap exactly once from the proposal pass.  Later
                # source addresses may extend ``_states`` and move the
                # transport frontier, but neither can enter this bank.
                self._ignition_states[layer] = PairedEditMemoryState(
                    source_key=current_addresses.source_key.detach().clone(),
                    target_value_residual=(
                        current_addresses.target_value_residual
                        .detach().clone()
                    ),
                    object_coordinate=(
                        current_addresses.object_coordinate.detach().clone()
                    ),
                    evidence=current_addresses.evidence.detach().clone(),
                    source_part_signature=(
                        current_addresses.source_part_signature
                        .detach().clone()
                        if current_addresses.source_part_signature is not None
                        else None
                    ),
                    lineage_id=(
                        current_addresses.lineage_id.detach().clone()
                        if current_addresses.lineage_id is not None
                        else None
                    ),
                    transport_confidence=(
                        current_addresses.transport_confidence
                        .detach().clone()
                        if current_addresses.transport_confidence is not None
                        else None
                    ),
                )
            if (
                self.source_transport
                and not transactional
                and bool(transaction.accepted.any())
            ):
                self._payload_initialized_layers.add(layer)
            if (
                self.source_transport
                and transactional
                and layer not in self._ignition_verified_layers
            ):
                accepted_lineage = transaction.lineage_id
                if accepted_lineage is None:
                    accepted_lineage = torch.full(
                        proposal_weight.shape,
                        -1,
                        device=proposal_weight.device,
                        dtype=torch.long,
                    )
                verified_state = self._states[layer]
                verified_evidence = verified_state.evidence.clone()
                for batch_index in range(proposal_weight.shape[0]):
                    accepted_mask = transaction.accepted[batch_index]
                    lineage = accepted_lineage[batch_index][accepted_mask]
                    confidence = transaction.write_weight[batch_index][
                        accepted_mask
                    ]
                    valid_lineage = lineage >= 0
                    lineage = lineage[valid_lineage]
                    confidence = confidence[valid_lineage]
                    if lineage.numel() == 0:
                        verified_evidence[batch_index].zero_()
                        continue
                    slot_lineage = verified_state.lineage_id[batch_index]
                    match = (
                        slot_lineage[:, None] == lineage[None]
                    )
                    replay_confidence = torch.where(
                        match,
                        confidence[None],
                        torch.zeros_like(confidence)[None],
                    ).amax(dim=-1)
                    verified_evidence[batch_index] *= replay_confidence
                self._states[layer] = PairedEditMemoryState(
                    source_key=verified_state.source_key,
                    target_value_residual=(
                        verified_state.target_value_residual
                    ),
                    object_coordinate=verified_state.object_coordinate,
                    evidence=verified_evidence,
                    source_part_signature=(
                        verified_state.source_part_signature
                    ),
                    lineage_id=verified_state.lineage_id,
                    transport_confidence=(
                        verified_state.transport_confidence
                    ),
                )
                if self.immutable_canonical_key_anchor:
                    ignition_state = self._ignition_states.get(layer)
                    if ignition_state is None:
                        raise RuntimeError(
                            "Canonical-key replay requires a frozen "
                            "ignition bank from the proposal pass"
                        )
                    ignition_evidence = ignition_state.evidence.clone()
                    for batch_index in range(proposal_weight.shape[0]):
                        accepted_mask = transaction.accepted[batch_index]
                        lineage = accepted_lineage[batch_index][accepted_mask]
                        confidence = transaction.write_weight[batch_index][
                            accepted_mask
                        ]
                        valid_lineage = lineage >= 0
                        lineage = lineage[valid_lineage].to(
                            ignition_evidence.device
                        )
                        confidence = confidence[valid_lineage].to(
                            ignition_evidence.device
                        )
                        if lineage.numel() == 0:
                            ignition_evidence[batch_index].zero_()
                            continue
                        ignition_lineage = ignition_state.lineage_id[
                            batch_index
                        ]
                        match = (
                            ignition_lineage[:, None]
                            == lineage[None]
                        )
                        replay_confidence = torch.where(
                            match,
                            confidence[None],
                            torch.zeros_like(confidence)[None],
                        ).amax(dim=-1)
                        ignition_evidence[batch_index] *= replay_confidence
                    # Only evidence may change during replay validation.
                    # Source K, delta-V, coordinates and lineage ids retain
                    # object identity from the proposal pass bit-for-bit.
                    self._ignition_states[layer] = PairedEditMemoryState(
                        source_key=ignition_state.source_key,
                        target_value_residual=(
                            ignition_state.target_value_residual
                        ),
                        object_coordinate=ignition_state.object_coordinate,
                        evidence=ignition_evidence,
                        source_part_signature=(
                            ignition_state.source_part_signature
                        ),
                        lineage_id=ignition_state.lineage_id,
                        transport_confidence=(
                            ignition_state.transport_confidence
                        ),
                    )
                self._frontiers[layer] = self._states[layer]
                self._ignition_verified_layers.add(layer)
            elif (
                self.source_transport
                and transactional
                and layer not in self._frontiers
                and bool((current_addresses.evidence > self.eps).any())
            ):
                # A broken adjacent-source chain may restart only after the
                # strict canonical reader re-associates the current address
                # and the independent target observation accepts its inherited
                # residual.  Never apply the looser transport matcher directly
                # to the long-range canonical bank.
                self._frontiers[layer] = current_addresses
            if not transactional and self.source_transport:
                # Ignition creates the only new payload lineage.  The
                # frontier starts from these source addresses and thereafter
                # may move, but its residual can only be inherited.
                self._frontiers[layer] = self._states[layer]
            diagnostics.append(transaction)

        return {
            "proposal": proposal_weight.detach(),
            "write": torch.stack(
                [item.write_weight for item in diagnostics]
            ).mean(0),
            "source_match": torch.stack(
                [item.source_match for item in diagnostics]
            ).mean(0),
            "residual_agreement": torch.stack(
                [item.residual_agreement for item in diagnostics]
            ).mean(0),
            "accepted": torch.stack(
                [item.accepted.float() for item in diagnostics]
            ).mean(0),
        }
