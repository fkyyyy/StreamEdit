"""Training-free source-appearance leakage control for video editing.

The source reconstruction residual is useful for motion and geometry, but an
unqualified residual can also push the denoising trajectory back toward the
source object's appearance.  This module identifies a conservative,
interaction-conditioned target-change core and removes only the residual
component that is antagonistic to the target edit direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn.functional as F


def _resize_map(value: torch.Tensor, size) -> torch.Tensor:
    if value.ndim != 4:
        raise ValueError(
            "Spatial evidence must have shape [B,T,H,W], got "
            f"{tuple(value.shape)}"
        )
    batch, frames, height, width = value.shape
    if (height, width) == tuple(size):
        return value.float()
    return F.interpolate(
        value.float().reshape(batch * frames, 1, height, width),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).reshape(batch, frames, *size)


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    batch, frames, height, width = mask.shape
    return (
        F.max_pool2d(
            mask.float().reshape(batch * frames, 1, height, width),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        ).reshape(batch, frames, height, width)
        > 0.5
    )


def _robust_normalize(value: torch.Tensor, eps: float) -> torch.Tensor:
    flat = value.detach().float().flatten(2)
    low = torch.quantile(flat, 0.50, dim=-1, keepdim=True)
    high = torch.quantile(flat, 0.95, dim=-1, keepdim=True)
    return (
        (flat - low) / (high - low).clamp_min(eps)
    ).clamp(0.0, 1.0).reshape_as(value)


def _adaptive_two_class_threshold(
    score: torch.Tensor,
    valid: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Return a per-frame, parameter-free two-class variance threshold."""
    thresholds = torch.full(
        score.shape[:2],
        float("inf"),
        dtype=torch.float32,
        device=score.device,
    )
    for batch_index in range(score.shape[0]):
        for frame_index in range(score.shape[1]):
            values = score[batch_index, frame_index][
                valid[batch_index, frame_index]
            ].float()
            if values.numel() == 0:
                continue
            if values.numel() == 1:
                thresholds[batch_index, frame_index] = values[0]
                continue
            values = values.sort().values
            cumulative = values.cumsum(dim=0)
            count = torch.arange(
                1,
                values.numel() + 1,
                device=values.device,
                dtype=values.dtype,
            )
            left_count = count[:-1]
            right_count = values.numel() - left_count
            left_mean = cumulative[:-1] / left_count
            right_mean = (cumulative[-1] - cumulative[:-1]) / right_count
            between = (
                left_count
                * right_count
                * (left_mean - right_mean).square()
            )
            if between.max() <= eps:
                threshold = values.mean()
            else:
                split = int(between.argmax().item())
                threshold = 0.5 * (values[split] + values[split + 1])
            thresholds[batch_index, frame_index] = threshold
    return thresholds[:, :, None, None]


def _select_contact_component(
    candidate: torch.Tensor,
    weight: torch.Tensor,
    contact_ring: torch.Tensor,
) -> torch.Tensor:
    """Keep one 8-connected component that is supported by the hand."""
    candidate_cpu = candidate.detach().bool().cpu()
    weight_cpu = weight.detach().float().cpu()
    contact_cpu = contact_ring.detach().float().cpu()
    selected = torch.zeros_like(candidate_cpu)
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    batch, frames, height, width = candidate_cpu.shape
    for batch_index in range(batch):
        for frame_index in range(frames):
            visited = torch.zeros((height, width), dtype=torch.bool)
            best_component = None
            best_score = None
            for row in range(height):
                for col in range(width):
                    if (
                        visited[row, col]
                        or not candidate_cpu[
                            batch_index, frame_index, row, col
                        ]
                    ):
                        continue
                    stack = [(row, col)]
                    visited[row, col] = True
                    component = []
                    mass = 0.0
                    contact_mass = 0.0
                    while stack:
                        current_row, current_col = stack.pop()
                        component.append((current_row, current_col))
                        current_weight = float(
                            weight_cpu[
                                batch_index,
                                frame_index,
                                current_row,
                                current_col,
                            ]
                        )
                        mass += current_weight
                        contact_mass += current_weight * float(
                            contact_cpu[
                                batch_index,
                                frame_index,
                                current_row,
                                current_col,
                            ]
                        )
                        for row_offset, col_offset in neighbors:
                            next_row = current_row + row_offset
                            next_col = current_col + col_offset
                            if (
                                0 <= next_row < height
                                and 0 <= next_col < width
                                and not visited[next_row, next_col]
                                and candidate_cpu[
                                    batch_index,
                                    frame_index,
                                    next_row,
                                    next_col,
                                ]
                            ):
                                visited[next_row, next_col] = True
                                stack.append((next_row, next_col))
                    # A target-change core must be causally tied to the
                    # hand/object interaction.  No contact means no edit.
                    if contact_mass <= 0.0:
                        continue
                    component_score = (
                        contact_mass,
                        mass,
                        len(component),
                    )
                    if best_score is None or component_score > best_score:
                        best_score = component_score
                        best_component = component
            if best_component is not None:
                for row, col in best_component:
                    selected[batch_index, frame_index, row, col] = True
    return selected.to(device=candidate.device)


@dataclass(frozen=True)
class TargetChangeCore:
    mask: torch.Tensor
    change_score: torch.Tensor
    semantic_score: torch.Tensor
    joint_score: torch.Tensor
    candidate_mask: torch.Tensor
    hand_exclusion_mask: torch.Tensor
    contact_ring: torch.Tensor

    def as_debug_maps(self) -> Dict[str, torch.Tensor]:
        return {
            "ignition_change_score": self.change_score,
            "ignition_semantic_score": self.semantic_score,
            "ignition_joint_score": self.joint_score,
            "ignition_candidate_mask": self.candidate_mask.float(),
            "ignition_core_mask": self.mask.float(),
            "ignition_hand_exclusion": (
                self.hand_exclusion_mask.float()
            ),
            "ignition_contact_ring": self.contact_ring.float(),
        }


def build_target_change_core(
    source_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    source_semantic_attention: torch.Tensor,
    hand_mask: torch.Tensor,
    *,
    hand_exclusion_radius: int = 1,
    contact_radius: int = 3,
    eps: float = 1e-6,
) -> TargetChangeCore:
    """Build a conservative target-change core without learned modules.

    The core requires three independent signals: target/source field
    disagreement, source-object semantics, and spatial contact with the hand.
    Thresholding is selected independently per frame by two-class variance,
    rather than a sample-specific hand-tuned scalar.
    """
    if source_velocity.shape != target_velocity.shape:
        raise ValueError(
            "Source and target velocities must have the same shape"
        )
    if source_velocity.ndim != 5:
        raise ValueError(
            "Velocities must have shape [B,T,C,H,W], got "
            f"{tuple(source_velocity.shape)}"
        )
    if hand_mask.ndim != 4:
        raise ValueError("hand_mask must have shape [B,T,H,W]")
    if source_semantic_attention.ndim != 4:
        raise ValueError(
            "source_semantic_attention must have shape [B,T,H,W]"
        )
    if hand_exclusion_radius < 0 or contact_radius <= hand_exclusion_radius:
        raise ValueError(
            "contact_radius must be greater than the non-negative "
            "hand_exclusion_radius"
        )
    if eps <= 0:
        raise ValueError("eps must be positive")

    batch, frames, _, height, width = source_velocity.shape
    expected_prefix = (batch, frames)
    if hand_mask.shape[:2] != expected_prefix:
        raise ValueError("Hand mask and velocities must share [B,T]")
    if source_semantic_attention.shape[:2] != expected_prefix:
        raise ValueError(
            "Semantic attention and velocities must share [B,T]"
        )

    change_magnitude = (
        target_velocity.detach().float()
        - source_velocity.detach().float()
    ).square().mean(dim=2).sqrt()
    change_score = _robust_normalize(change_magnitude, eps)
    semantic_score = _robust_normalize(
        _resize_map(source_semantic_attention, (height, width)),
        eps,
    )
    resized_hand = _resize_map(hand_mask.float(), (height, width)) > 0.0
    hand_exclusion = _dilate(resized_hand, hand_exclusion_radius)
    contact_ring = (
        _dilate(resized_hand, contact_radius) & ~hand_exclusion
    )

    joint_score = torch.sqrt(
        (change_score * semantic_score).clamp_min(0.0)
    )
    valid = (
        (change_score > eps)
        & (semantic_score > eps)
        & ~hand_exclusion
    )
    threshold = _adaptive_two_class_threshold(
        joint_score,
        valid,
        eps,
    )
    candidate = valid & (joint_score >= threshold)
    core = _select_contact_component(
        candidate=candidate,
        weight=joint_score,
        contact_ring=contact_ring,
    )
    core &= ~hand_exclusion
    return TargetChangeCore(
        mask=core,
        change_score=change_score,
        semantic_score=semantic_score,
        joint_score=joint_score,
        candidate_mask=candidate,
        hand_exclusion_mask=hand_exclusion,
        contact_ring=contact_ring,
    )


def remove_antagonistic_source_residual(
    source_residual: torch.Tensor,
    edit_direction: torch.Tensor,
    target_change_core: torch.Tensor,
    *,
    protect_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
):
    """Remove only residual energy opposing the target edit direction."""
    if source_residual.shape != edit_direction.shape:
        raise ValueError(
            "Source residual and edit direction must have the same shape"
        )
    if source_residual.ndim != 5:
        raise ValueError(
            "Vector fields must have shape [B,T,C,H,W]"
        )
    expected_mask_shape = (
        source_residual.shape[0],
        source_residual.shape[1],
        source_residual.shape[3],
        source_residual.shape[4],
    )
    if target_change_core.shape != expected_mask_shape:
        raise ValueError(
            "target_change_core must align with the vector-field grid"
        )
    if protect_mask is not None and protect_mask.shape != expected_mask_shape:
        raise ValueError("protect_mask must align with the vector-field grid")
    if eps <= 0:
        raise ValueError("eps must be positive")

    residual = source_residual.float()
    direction = edit_direction.float()
    core = target_change_core.detach().bool()
    if protect_mask is not None:
        core &= ~protect_mask.detach().bool()

    dot = (residual * direction).sum(dim=2, keepdim=True)
    direction_energy = direction.square().sum(dim=2, keepdim=True)
    negative_coefficient = torch.minimum(dot, torch.zeros_like(dot)) / (
        direction_energy + eps
    )
    removed_vector = negative_coefficient * direction
    projected = residual - removed_vector
    active = core.unsqueeze(2) & (direction_energy > eps)
    filtered = torch.where(
        active,
        projected.to(source_residual.dtype),
        source_residual,
    )
    actual_removed = residual - filtered.float()

    residual_energy = residual.square().sum(dim=2, keepdim=True)
    filtered_energy = filtered.float().square().sum(dim=2, keepdim=True)
    removed_energy = actual_removed.square().sum(dim=2, keepdim=True)
    diagnostics = {
        "appearance_leakage_core": active.float(),
        "appearance_leakage_antagonism": (
            (-dot).clamp_min(0.0) / direction_energy.sqrt().clamp_min(eps)
        ),
        "appearance_leakage_removed_energy": removed_energy,
        "appearance_leakage_source_energy": residual_energy,
        "appearance_leakage_preserved_energy": filtered_energy,
    }
    return filtered, diagnostics


def restore_projected_residual_norm(
    source_residual: torch.Tensor,
    projected_residual: torch.Tensor,
    application_weight: torch.Tensor,
    *,
    eps: float = 1e-8,
    max_scale: float = 4.0,
):
    """Restore projected-residual magnitude without restoring its direction.

    The norm is measured after applying the native StreamGVE soft background
    weight, because that is the residual that is actually added to the target
    velocity.  One positive scalar is computed independently for every sample
    and latent timestep.  Positive scaling cannot reintroduce the antagonistic
    component removed by :func:`remove_antagonistic_source_residual`.

    ``max_scale`` is only a numerical degeneracy guard.  A projection that
    leaves almost no safe direction cannot recover the source norm by scalar
    rescaling without amplifying numerical noise.
    """
    if source_residual.shape != projected_residual.shape:
        raise ValueError(
            "source_residual and projected_residual must match"
        )
    if source_residual.ndim != 5:
        raise ValueError(
            "Residual fields must have shape [B,T,C,H,W]"
        )
    expected_weight_shape = (
        source_residual.shape[0],
        source_residual.shape[1],
        1,
        source_residual.shape[3],
        source_residual.shape[4],
    )
    if application_weight.shape != expected_weight_shape:
        raise ValueError(
            "application_weight must have shape "
            f"{expected_weight_shape}, got "
            f"{tuple(application_weight.shape)}"
        )
    if eps <= 0:
        raise ValueError("eps must be positive")
    if max_scale < 1.0:
        raise ValueError("max_scale must be at least 1")

    source = source_residual.float()
    projected = projected_residual.float()
    weight = application_weight.float()
    # Preserve temporal locality: reduce only channel and spatial dimensions.
    reduction_dims = (2, 3, 4)
    source_energy = (weight * source).square().sum(dim=reduction_dims)
    projected_energy = (weight * projected).square().sum(
        dim=reduction_dims
    )
    restorable = (source_energy > eps) & (projected_energy > eps)
    raw_scale = torch.sqrt(
        source_energy / projected_energy.clamp_min(eps)
    )
    scale = torch.where(
        restorable,
        raw_scale.clamp(max=max_scale),
        torch.ones_like(raw_scale),
    ).detach()
    scale_view = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    calibrated = projected * scale_view
    applied_energy = (weight * calibrated).square().sum(
        dim=reduction_dims
    )
    diagnostics = {
        "source_energy": source_energy.detach(),
        "projected_energy": projected_energy.detach(),
        "applied_energy": applied_energy.detach(),
        "norm_scale": scale,
        "scale_capped": (restorable & (raw_scale > max_scale)).detach(),
        "degenerate": (~restorable).detach(),
    }
    return calibrated.to(projected_residual.dtype), diagnostics


@dataclass
class CausalResidualEnergyBudget:
    """Freeze a causal projection-energy budget from the first block.

    A separate reference is recorded for every denoising-step index.  Later
    blocks may remove less antagonistic energy, but cannot remove a larger
    fraction of the source residual than the corresponding first-block step.
    The budget is global per sample and therefore introduces no moving spatial
    boundary.
    """

    eps: float = 1e-8
    reference_fractions: Dict[int, torch.Tensor] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    @staticmethod
    def _validate(
        source_residual: torch.Tensor,
        projected_residual: torch.Tensor,
        application_weight: torch.Tensor,
    ) -> None:
        if source_residual.shape != projected_residual.shape:
            raise ValueError(
                "source_residual and projected_residual must match"
            )
        if source_residual.ndim != 5:
            raise ValueError(
                "Residual fields must have shape [B,T,C,H,W]"
            )
        expected_weight_shape = (
            source_residual.shape[0],
            source_residual.shape[1],
            1,
            source_residual.shape[3],
            source_residual.shape[4],
        )
        if application_weight.shape != expected_weight_shape:
            raise ValueError(
                "application_weight must have shape "
                f"{expected_weight_shape}, got "
                f"{tuple(application_weight.shape)}"
            )

    def apply(
        self,
        source_residual: torch.Tensor,
        projected_residual: torch.Tensor,
        application_weight: torch.Tensor,
        *,
        denoising_step_index: int,
    ):
        """Apply the frozen first-block budget to one denoising step."""
        self._validate(
            source_residual, projected_residual, application_weight
        )
        if denoising_step_index < 0:
            raise ValueError("denoising_step_index must be non-negative")

        source = source_residual.float()
        projected = projected_residual.float()
        weight = application_weight.float()
        removed = source - projected
        reduction_dims = tuple(range(1, source.ndim))
        source_energy = (weight * source).square().sum(
            dim=reduction_dims
        )
        raw_removed_energy = (weight * removed).square().sum(
            dim=reduction_dims
        )
        raw_fraction = raw_removed_energy / (source_energy + self.eps)

        reference_initialized = (
            denoising_step_index not in self.reference_fractions
        )
        if reference_initialized:
            # Keep the tiny state on CPU: it is a causal scalar per sample and
            # denoising step, not a feature or appearance memory.
            self.reference_fractions[denoising_step_index] = (
                raw_fraction.detach().cpu()
            )
        reference_fraction = self.reference_fractions[
            denoising_step_index
        ].to(device=source.device, dtype=source.dtype)
        if reference_fraction.shape != raw_fraction.shape:
            raise ValueError(
                "Projection-budget batch size changed across blocks"
            )

        scale = torch.sqrt(
            (reference_fraction / raw_fraction.clamp_min(self.eps))
            .clamp(max=1.0)
        )
        scale = torch.where(
            raw_fraction > self.eps, scale, torch.ones_like(scale)
        ).detach()
        view_shape = (source.shape[0],) + (1,) * (source.ndim - 1)
        scale_view = scale.view(view_shape)
        scaled_projection = source - scale_view * removed
        # Preserve the original P0 tensor exactly whenever no cap is needed;
        # this avoids an unnecessary subtract/add round trip in low precision.
        budgeted = torch.where(
            scale_view == 1.0, projected, scaled_projection
        )
        applied_removed_energy = (
            weight * (source - budgeted)
        ).square().sum(dim=reduction_dims)
        applied_fraction = applied_removed_energy / (
            source_energy + self.eps
        )
        diagnostics = {
            "raw_removed_fraction": raw_fraction.detach(),
            "reference_removed_fraction": (
                reference_fraction.detach()
            ),
            "applied_removed_fraction": applied_fraction.detach(),
            "projection_scale": scale,
            "reference_initialized": reference_initialized,
        }
        return budgeted.to(source_residual.dtype), diagnostics
