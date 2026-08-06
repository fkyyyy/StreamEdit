from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


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
