from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, TYPE_CHECKING

import torch
import torch.nn.functional as F

from .appearance_leakage import remove_antagonistic_source_residual

if TYPE_CHECKING:
    from .control_belief import CausalControlBelief


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    batch, frames, height, width = mask.shape
    flat = mask.reshape(batch * frames, 1, height, width).float()
    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(
        flat,
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )
    return dilated.reshape(batch, frames, height, width) > 0.5


@dataclass(frozen=True)
class RoleState:
    """Disjoint spatial role probabilities on the VAE latent grid."""

    object: torch.Tensor
    boundary: torch.Tensor
    hand: torch.Tensor
    background: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "object": self.object,
            "boundary": self.boundary,
            "hand": self.hand,
            "background": self.background,
        }

    @property
    def edit_weight(self) -> torch.Tensor:
        return self.object + self.boundary

    @property
    def preserve_weight(self) -> torch.Tensor:
        return self.hand + self.background

    @property
    def contact(self) -> torch.Tensor:
        return self.boundary

    def validate(self) -> None:
        shapes = {tuple(value.shape) for value in self.as_dict().values()}
        if len(shapes) != 1:
            raise ValueError(f"Role maps must have one shape, got {sorted(shapes)}")
        total = sum(self.as_dict().values())
        if not torch.allclose(total, torch.ones_like(total), atol=1e-5):
            raise ValueError("Role probabilities must sum to one at every token")
        for name, value in self.as_dict().items():
            if value.min() < 0 or value.max() > 1:
                raise ValueError(f"Role '{name}' must lie in [0, 1]")


def build_oracle_roles(
    object_mask: torch.Tensor,
    hand_mask: torch.Tensor,
    boundary_radius: int = 1,
) -> RoleState:
    """Build an object-priority interaction partition from oracle masks.

    Boundary denotes object tokens that overlap or touch the hand. Those tokens
    remain editable; the hand is never used as a hard negative for the object.
    """
    if object_mask.shape != hand_mask.shape:
        raise ValueError(
            f"Object and hand masks must match, got "
            f"{tuple(object_mask.shape)} and {tuple(hand_mask.shape)}"
        )
    if object_mask.ndim != 4:
        raise ValueError(
            f"Role masks must have shape [B,T,H,W], got {tuple(object_mask.shape)}"
        )

    object_mask = object_mask.bool()
    hand_mask = hand_mask.bool()
    boundary = object_mask & _dilate(hand_mask, boundary_radius)
    object_core = object_mask & ~boundary
    hand_core = hand_mask & ~object_mask
    background = ~(object_core | boundary | hand_core)

    roles = RoleState(
        object=object_core.float(),
        boundary=boundary.float(),
        hand=hand_core.float(),
        background=background.float(),
    )
    roles.validate()
    return roles


class RoleFlowRouter:
    """Closed-form token-wise routing between edit and preservation fields."""

    @staticmethod
    def _resize_weight(weight: torch.Tensor, spatial_size) -> torch.Tensor:
        batch, frames, height, width = weight.shape
        flat = weight.reshape(batch * frames, 1, height, width)
        resized = F.interpolate(flat, size=spatial_size, mode="nearest")
        return resized.reshape(batch, frames, 1, *spatial_size)

    def __call__(
        self,
        target_velocity: torch.Tensor,
        source_reconstruction_velocity: torch.Tensor,
        roles: RoleState,
    ):
        if target_velocity.shape != source_reconstruction_velocity.shape:
            raise ValueError(
                "Target and source reconstruction velocities must have "
                f"the same shape, got {tuple(target_velocity.shape)} and "
                f"{tuple(source_reconstruction_velocity.shape)}"
            )
        roles.validate()
        spatial_size = target_velocity.shape[-2:]
        edit_weight = self._resize_weight(
            roles.edit_weight.to(target_velocity), spatial_size
        )
        preserve_weight = self._resize_weight(
            roles.preserve_weight.to(target_velocity), spatial_size
        )
        routed_velocity = (
            edit_weight * target_velocity
            + preserve_weight * source_reconstruction_velocity
        )
        return routed_velocity, edit_weight, preserve_weight


class ResidualRoleFlowRouter:
    """Role-routed residual source guidance around the target field."""

    @staticmethod
    def _resize_weight(weight: torch.Tensor, spatial_size) -> torch.Tensor:
        batch, frames, height, width = weight.shape
        flat = weight.reshape(batch * frames, 1, height, width)
        resized = F.interpolate(flat, size=spatial_size, mode="nearest")
        return resized.reshape(batch, frames, 1, *spatial_size)

    def __call__(
        self,
        target_velocity: torch.Tensor,
        source_velocity: torch.Tensor,
        source_reconstruction_velocity: torch.Tensor,
        roles: RoleState,
        contact_target_weight: float = 0.7,
    ):
        velocity_shapes = {
            tuple(target_velocity.shape),
            tuple(source_velocity.shape),
            tuple(source_reconstruction_velocity.shape),
        }
        if len(velocity_shapes) != 1:
            raise ValueError(
                "Target, source, and source reconstruction velocities must "
                f"have the same shape, got {sorted(velocity_shapes)}"
            )
        if not 0.0 <= contact_target_weight <= 1.0:
            raise ValueError(
                "contact_target_weight must lie in [0, 1], got "
                f"{contact_target_weight}"
            )

        roles.validate()
        correction_weight = (
            roles.hand
            + roles.background
            + (1.0 - contact_target_weight) * roles.boundary
        )
        correction_weight = self._resize_weight(
            correction_weight.to(target_velocity),
            target_velocity.shape[-2:],
        )
        source_residual = (
            source_reconstruction_velocity - source_velocity
        )
        routed_velocity = target_velocity + correction_weight * source_residual
        return routed_velocity, correction_weight


class PosteriorResidualFlowRouter:
    """Mix target and source-residual experts with role posteriors."""

    def __init__(self, eps: float = 1e-6):
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps

    @staticmethod
    def _resize_roles(
        roles: RoleState,
        spatial_size,
        dtype,
        device,
    ) -> torch.Tensor:
        role_tensor = torch.stack(
            [
                roles.object,
                roles.boundary,
                roles.hand,
                roles.background,
            ],
            dim=2,
        ).float()
        batch, frames, role_count, height, width = role_tensor.shape
        resized = F.interpolate(
            role_tensor.reshape(
                batch * frames,
                role_count,
                height,
                width,
            ),
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        ).reshape(
            batch,
            frames,
            role_count,
            *spatial_size,
        )
        resized = resized.clamp_min(0.0)
        resized = resized / resized.sum(
            dim=2,
            keepdim=True,
        ).clamp_min(1e-6)
        return resized.to(device=device, dtype=dtype)

    def __call__(
        self,
        target_velocity: torch.Tensor,
        source_velocity: torch.Tensor,
        source_reconstruction_velocity: torch.Tensor,
        roles: RoleState,
        hard_roles: bool = False,
    ):
        velocity_shapes = {
            tuple(target_velocity.shape),
            tuple(source_velocity.shape),
            tuple(source_reconstruction_velocity.shape),
        }
        if len(velocity_shapes) != 1:
            raise ValueError(
                "Target, source, and source reconstruction velocities must "
                f"have the same shape, got {sorted(velocity_shapes)}"
            )

        roles.validate()
        probabilities = self._resize_roles(
            roles,
            target_velocity.shape[-2:],
            target_velocity.dtype,
            target_velocity.device,
        )
        if hard_roles:
            hard_index = probabilities.argmax(dim=2, keepdim=True)
            probabilities = torch.zeros_like(probabilities).scatter_(
                2,
                hard_index,
                1.0,
            )

        object_probability = probabilities[:, :, 0:1]
        contact_probability = probabilities[:, :, 1:2]
        hand_probability = probabilities[:, :, 2:3]
        background_probability = probabilities[:, :, 3:4]
        preservation_probability = (
            hand_probability + background_probability
        )

        # Contact is split online by its competition with preservation roles.
        contact_denominator = (
            contact_probability + preservation_probability
        )
        contact_present = contact_probability > self.eps
        contact_target_weight = torch.where(
            contact_present,
            contact_probability
            / contact_denominator.clamp_min(self.eps),
            torch.zeros_like(contact_probability),
        ).clamp(0.0, 1.0)
        contact_residual_weight = torch.where(
            contact_present,
            1.0 - contact_target_weight,
            torch.zeros_like(contact_probability),
        )
        residual_expert_weight = (
            preservation_probability
            + contact_probability * contact_residual_weight
        ).clamp(0.0, 1.0)
        target_expert_weight = (
            object_probability
            + contact_probability * contact_target_weight
        ).clamp(0.0, 1.0)

        expert_sum = (
            target_expert_weight + residual_expert_weight
        ).clamp_min(self.eps)
        target_expert_weight = target_expert_weight / expert_sum
        residual_expert_weight = residual_expert_weight / expert_sum

        source_residual = (
            source_reconstruction_velocity - source_velocity
        )
        routed_velocity = (
            target_velocity
            + residual_expert_weight * source_residual
        )
        entropy = -(
            probabilities.float()
            * probabilities.float().clamp_min(self.eps).log()
        ).sum(dim=2, keepdim=True) / torch.log(
            probabilities.new_tensor(4.0).float()
        )
        entropy = entropy.clamp(0.0, 1.0)
        diagnostics = {
            "target_expert_weight": target_expert_weight,
            "residual_expert_weight": residual_expert_weight,
            "contact_target_weight": contact_target_weight,
            "contact_residual_weight": contact_residual_weight,
            "role_entropy": entropy.to(target_velocity),
            "role_probabilities": probabilities,
        }
        return routed_velocity, diagnostics


class BayesResidualFlowRouter:
    """Precision-weighted Bayes action for non-exclusive control beliefs."""

    def __init__(self, eps: float = 1e-6):
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps

    @staticmethod
    def _resize_map(
        value: torch.Tensor,
        spatial_size,
    ) -> torch.Tensor:
        batch, frames, height, width = value.shape
        resized = F.interpolate(
            value.float().reshape(batch * frames, 1, height, width),
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )
        return resized.reshape(batch, frames, 1, *spatial_size)

    def __call__(
        self,
        target_velocity: torch.Tensor,
        source_velocity: torch.Tensor,
        source_reconstruction_velocity: torch.Tensor,
        belief: CausalControlBelief,
        target_owned_mask: torch.Tensor | None = None,
        target_change_core: torch.Tensor | None = None,
        protect_mask: torch.Tensor | None = None,
        identity_owner_weight: torch.Tensor | None = None,
        identity_source_suppression: float = 0.0,
        denoising_fraction: float = 1.0,
    ):
        velocity_shapes = {
            tuple(target_velocity.shape),
            tuple(source_velocity.shape),
            tuple(source_reconstruction_velocity.shape),
        }
        if len(velocity_shapes) != 1:
            raise ValueError(
                "Target, source, and source reconstruction velocities must "
                f"have the same shape, got {sorted(velocity_shapes)}"
            )
        belief.validate()
        if not 0.0 <= identity_source_suppression <= 1.0:
            raise ValueError(
                "identity_source_suppression must lie in [0, 1]"
            )
        if not 0.0 <= denoising_fraction <= 1.0:
            raise ValueError(
                "denoising_fraction must lie in [0, 1]"
            )
        spatial_size = target_velocity.shape[-2:]
        edit_belief = self._resize_map(
            belief.edit_belief,
            spatial_size,
        )
        preserve_belief = self._resize_map(
            belief.preserve_belief,
            spatial_size,
        )
        edit_precision = self._resize_map(
            belief.edit_precision,
            spatial_size,
        )
        preserve_precision = self._resize_map(
            belief.preserve_precision,
            spatial_size,
        )

        edit_strength = edit_belief * edit_precision
        preserve_strength = preserve_belief * preserve_precision
        total_strength = edit_strength + preserve_strength
        no_evidence = total_strength <= self.eps
        edit_action_weight = torch.where(
            no_evidence,
            torch.zeros_like(total_strength),
            edit_strength / total_strength.clamp_min(self.eps),
        )
        preserve_action_weight = torch.where(
            no_evidence,
            torch.ones_like(total_strength),
            preserve_strength / total_strength.clamp_min(self.eps),
        )
        target_owned = None
        if target_owned_mask is not None:
            if target_owned_mask.ndim != 4:
                raise ValueError(
                    "target_owned_mask must have shape [B,T,H,W]"
                )
            if target_owned_mask.shape[:2] != target_velocity.shape[:2]:
                raise ValueError(
                    "Target-owned mask and velocity must share [B,T]"
                )
            target_owned = F.interpolate(
                target_owned_mask.detach().float().reshape(
                    target_owned_mask.shape[0]
                    * target_owned_mask.shape[1],
                    1,
                    *target_owned_mask.shape[-2:],
                ),
                size=spatial_size,
                mode="nearest",
            ).reshape(
                target_owned_mask.shape[0],
                target_owned_mask.shape[1],
                1,
                *spatial_size,
            ).bool()
            preserve_action_weight = torch.where(
                target_owned,
                torch.zeros_like(preserve_action_weight),
                preserve_action_weight,
            )
            edit_action_weight = torch.where(
                target_owned,
                torch.ones_like(edit_action_weight),
                edit_action_weight,
            )
        source_suppression = torch.zeros_like(preserve_action_weight)
        if identity_owner_weight is not None:
            if identity_owner_weight.ndim != 4:
                raise ValueError(
                    "identity_owner_weight must have shape [B,T,H,W]"
                )
            if identity_owner_weight.shape[:2] != target_velocity.shape[:2]:
                raise ValueError(
                    "Identity ownership and velocity must share [B,T]"
                )
            owner = self._resize_map(
                identity_owner_weight, spatial_size
            ).clamp(0.0, 1.0)
            source_suppression = (
                owner
                * float(identity_source_suppression)
                * float(denoising_fraction)
            ).clamp(0.0, 1.0)
            preserve_action_weight = (
                preserve_action_weight * (1.0 - source_suppression)
            )

        target_f32 = target_velocity.float()
        source_residual_f32 = (
            source_reconstruction_velocity.float()
            - source_velocity.float()
        )
        leakage_diagnostics = None
        if target_change_core is not None:
            source_residual_f32, leakage_diagnostics = (
                remove_antagonistic_source_residual(
                    source_residual=source_residual_f32,
                    edit_direction=(
                        target_velocity.float()
                        - source_velocity.float()
                    ),
                    target_change_core=target_change_core,
                    protect_mask=protect_mask,
                    eps=self.eps,
                )
            )
        routed_velocity = (
            target_f32
            + preserve_action_weight * source_residual_f32
        ).to(target_velocity.dtype)
        diagnostics = {
            "edit_belief": edit_belief,
            "preserve_belief": preserve_belief,
            "edit_precision": edit_precision,
            "preserve_precision": preserve_precision,
            "edit_strength": edit_strength,
            "preserve_strength": preserve_strength,
            "edit_action_weight": edit_action_weight,
            "preserve_action_weight": preserve_action_weight,
            "action_sum_error": (
                edit_action_weight + preserve_action_weight - 1.0
            ).abs(),
            "no_evidence": no_evidence.float(),
            "identity_source_suppression": source_suppression,
        }
        if target_owned is not None:
            diagnostics["target_owned_mask"] = target_owned.float()
        if leakage_diagnostics is not None:
            diagnostics.update(leakage_diagnostics)
        return routed_velocity, diagnostics
