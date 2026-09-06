from __future__ import annotations

from dataclasses import dataclass
import math
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


def select_hand_connected_support(
    candidate: torch.Tensor,
    weight: torch.Tensor,
    hand_anchor: torch.Tensor,
    *,
    anchor_radius: int = 1,
    previous_support: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keep one temporally coherent component attached to the hand.

    The first frame must be supported by the explicit hand anchor. Later
    frames may additionally inherit the component selected in the preceding
    frame. This makes a short hand/attention dropout fail to temporal support
    instead of switching ownership to an unrelated high-response object.
    """
    if candidate.ndim != 4 or candidate.shape != weight.shape:
        raise ValueError(
            "candidate and weight must share shape [B,T,H,W]"
        )
    if hand_anchor.shape != candidate.shape:
        raise ValueError(
            "hand_anchor must align with candidate on [B,T,H,W]"
        )
    if anchor_radius < 0:
        raise ValueError("anchor_radius must be non-negative")
    if previous_support is not None and previous_support.shape != (
        candidate.shape[0], *candidate.shape[-2:]
    ):
        raise ValueError(
            "previous_support must have shape [B,H,W]"
        )

    candidate_cpu = candidate.detach().bool().cpu()
    weight_cpu = weight.detach().float().cpu()
    anchor_cpu = _dilate(
        hand_anchor.detach().bool(), anchor_radius
    ).cpu()
    selected = torch.zeros_like(candidate_cpu)
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    batch, frames, height, width = candidate_cpu.shape
    for batch_index in range(batch):
        previous = (
            torch.zeros((height, width), dtype=torch.bool)
            if previous_support is None
            else previous_support[batch_index].detach().bool().cpu()
        )
        for frame_index in range(frames):
            visited = torch.zeros(
                (height, width), dtype=torch.bool
            )
            previous_corridor = (
                _dilate(
                    previous[None, None], anchor_radius
                )[0, 0].cpu()
                if previous.any()
                else previous
            )
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
                    anchor_mass = 0.0
                    temporal_mass = 0.0
                    while stack:
                        current_row, current_col = stack.pop()
                        component.append((current_row, current_col))
                        value = float(weight_cpu[
                            batch_index, frame_index,
                            current_row, current_col,
                        ])
                        mass += value
                        if anchor_cpu[
                            batch_index, frame_index,
                            current_row, current_col,
                        ]:
                            anchor_mass += value
                        if previous_corridor[current_row, current_col]:
                            temporal_mass += value
                        for row_offset, col_offset in neighbors:
                            next_row = current_row + row_offset
                            next_col = current_col + col_offset
                            if (
                                0 <= next_row < height
                                and 0 <= next_col < width
                                and not visited[next_row, next_col]
                                and candidate_cpu[
                                    batch_index, frame_index,
                                    next_row, next_col,
                                ]
                            ):
                                visited[next_row, next_col] = True
                                stack.append((next_row, next_col))
                    # Explicit hand contact has priority. Temporal overlap is
                    # only a dropout fallback and cannot initialize ownership.
                    attached = anchor_mass > 0.0
                    inherited = temporal_mass > 0.0 and previous.any()
                    if not attached and not inherited:
                        continue
                    score = (
                        int(attached),
                        anchor_mass + temporal_mass,
                        mass,
                        len(component),
                    )
                    if best_score is None or score > best_score:
                        best_score = score
                        best_component = component
            previous = torch.zeros_like(previous)
            if best_component is not None:
                for row, col in best_component:
                    selected[batch_index, frame_index, row, col] = True
                    previous[row, col] = True
    return selected.to(device=candidate.device)


def limit_connected_support_area(
    support: torch.Tensor,
    weight: torch.Tensor,
    hand_anchor: torch.Tensor,
    area_budget: torch.Tensor,
    *,
    previous_support: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causally trim a connected support without fragmenting its core.

    Tokens are admitted by an 8-connected best-first traversal rooted at the
    hand interaction or transported previous support.  This applies the area
    budget *after* hysteresis growth and therefore prevents low-threshold
    connected recovery from silently bypassing the adaptive coverage limit.
    """
    if support.ndim != 4 or weight.shape != support.shape:
        raise ValueError("support and weight must share shape [B,T,H,W]")
    if hand_anchor.shape != support.shape:
        raise ValueError("hand_anchor must align with support")
    if area_budget.shape != support.shape[:2] + (1, 1):
        raise ValueError("area_budget must have shape [B,T,1,1]")
    if previous_support is not None and previous_support.shape != (
        support.shape[0], *support.shape[-2:]
    ):
        raise ValueError("previous_support must have shape [B,H,W]")

    support_cpu = support.detach().bool().cpu()
    weight_cpu = weight.detach().float().cpu()
    anchor_cpu = _dilate(hand_anchor.detach().bool(), 1).cpu()
    budget_cpu = area_budget.detach().float().cpu()
    selected = torch.zeros_like(support_cpu)
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    batch, frames, height, width = support_cpu.shape
    token_count = height * width
    for batch_index in range(batch):
        previous = (
            torch.zeros((height, width), dtype=torch.bool)
            if previous_support is None
            else previous_support[batch_index].detach().bool().cpu()
        )
        for frame_index in range(frames):
            candidate = support_cpu[batch_index, frame_index]
            count = min(
                int(candidate.sum().item()),
                max(
                    1,
                    int(math.ceil(
                        float(budget_cpu[batch_index, frame_index])
                        * token_count - 1e-6
                    )),
                ),
            )
            if count <= 0 or not candidate.any():
                previous = torch.zeros_like(previous)
                continue
            root_region = (
                anchor_cpu[batch_index, frame_index]
                | _dilate(previous[None, None], 1)[0, 0]
            ) & candidate
            if not root_region.any():
                root_region = candidate
            score = weight_cpu[batch_index, frame_index]
            root_flat = score.masked_fill(~root_region, -1.0).argmax()
            root = (int(root_flat) // width, int(root_flat) % width)
            chosen = torch.zeros_like(candidate)
            frontier = [root]
            queued = {root}
            while frontier and int(chosen.sum().item()) < count:
                best_index = max(
                    range(len(frontier)),
                    key=lambda index: float(score[frontier[index]]),
                )
                row, col = frontier.pop(best_index)
                if chosen[row, col] or not candidate[row, col]:
                    continue
                chosen[row, col] = True
                for row_offset, col_offset in neighbors:
                    next_position = (row + row_offset, col + col_offset)
                    next_row, next_col = next_position
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and candidate[next_row, next_col]
                        and not chosen[next_row, next_col]
                        and next_position not in queued
                    ):
                        frontier.append(next_position)
                        queued.add(next_position)
            selected[batch_index, frame_index] = chosen
            previous = chosen
    return selected.to(device=support.device)


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
        editable_source_residual: torch.Tensor | None = None,
        object_residual_strength: float | None = None,
        contact_residual_strength: float | None = None,
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
        for name, strength in (
            ("object_residual_strength", object_residual_strength),
            ("contact_residual_strength", contact_residual_strength),
        ):
            if strength is not None and not 0.0 <= float(strength) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if editable_source_residual is not None and (
            editable_source_residual.shape != target_velocity.shape
        ):
            raise ValueError(
                "editable_source_residual must align with the velocity fields"
            )
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

        explicit_role_policy = (
            object_residual_strength is not None
            or contact_residual_strength is not None
        )
        if explicit_role_policy:
            object_strength = float(
                0.0
                if object_residual_strength is None
                else object_residual_strength
            )
            contact_strength = float(
                0.0
                if contact_residual_strength is None
                else contact_residual_strength
            )
            contact_residual_weight = contact_probability.new_full(
                contact_probability.shape, contact_strength
            )
            contact_target_weight = 1.0 - contact_residual_weight
            residual_expert_weight = (
                object_probability * object_strength
                + contact_probability * contact_strength
                + preservation_probability
            ).clamp(0.0, 1.0)
            target_expert_weight = (1.0 - residual_expert_weight).clamp(
                0.0, 1.0
            )
        else:
            # Legacy posterior routing: contact is split online by its
            # competition with preservation roles.
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
        if explicit_role_policy and editable_source_residual is not None:
            editable_residual = editable_source_residual.to(
                device=target_velocity.device, dtype=target_velocity.dtype
            )
            editable_weight = (
                object_probability * float(object_residual_strength or 0.0)
                + contact_probability * float(contact_residual_strength or 0.0)
            )
            preservation_weight = preservation_probability
            routed_velocity = target_velocity + (
                editable_weight * editable_residual
                + preservation_weight * source_residual
            )
        else:
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


def build_role_memory_gates(
    roles: RoleState,
    spatial_size,
    *,
    contact_read_weight: float = 0.5,
    object_write_threshold: float = 0.5,
    max_hand_write_probability: float = 0.1,
    hand_anchor: torch.Tensor | None = None,
    owner_support: torch.Tensor | None = None,
    component_probability_threshold: float = 0.2,
    evidence_reliability: torch.Tensor | None = None,
    temporal_recovery: torch.Tensor | None = None,
    adaptive_write: bool = False,
    temporal_write_consensus: bool = False,
):
    """Build asymmetric M2 read and write permissions from soft roles.

    Reads have high recall (object plus a discounted contact region). Writes
    are deliberately high precision: only confident object-core tokens that
    are not hand-like may enter the immutable appearance bank.
    """
    roles.validate()
    if not 0.0 <= float(contact_read_weight) <= 1.0:
        raise ValueError("contact_read_weight must lie in [0, 1]")
    if not 0.0 <= float(object_write_threshold) <= 1.0:
        raise ValueError("object_write_threshold must lie in [0, 1]")
    if not 0.0 <= float(max_hand_write_probability) <= 1.0:
        raise ValueError(
            "max_hand_write_probability must lie in [0, 1]"
        )
    if not 0.0 <= float(component_probability_threshold) <= 1.0:
        raise ValueError(
            "component_probability_threshold must lie in [0, 1]"
        )

    probabilities = PosteriorResidualFlowRouter._resize_roles(
        roles, spatial_size, torch.float32, roles.object.device
    )
    object_probability = probabilities[:, :, 0]
    contact_probability = probabilities[:, :, 1]
    hand_probability = probabilities[:, :, 2]
    entropy = -(
        probabilities * probabilities.clamp_min(1e-6).log()
    ).sum(dim=2) / torch.log(probabilities.new_tensor(4.0))
    entropy = entropy.clamp(0.0, 1.0)
    read_gate = (
        object_probability
        + float(contact_read_weight) * contact_probability
    ).clamp(0.0, 1.0) * (1.0 - entropy)
    raw_write_gate = (
        (object_probability >= float(object_write_threshold))
        & (object_probability >= contact_probability)
        & (hand_probability <= float(max_hand_write_probability))
        & (entropy <= 0.5)
    )
    candidate_write_gate = raw_write_gate.clone()
    connected_support = torch.ones_like(raw_write_gate)
    if owner_support is not None:
        if owner_support.ndim != 4 or owner_support.shape[:2] != (
            object_probability.shape[0], object_probability.shape[1]
        ):
            raise ValueError(
                "owner_support must align with roles on [B,T]"
            )
        support_batch, support_frames = owner_support.shape[:2]
        connected_support = F.interpolate(
            owner_support.float().reshape(
                support_batch * support_frames, 1,
                *owner_support.shape[-2:],
            ),
            size=spatial_size,
            mode="nearest",
        ).reshape(
            support_batch, support_frames, *spatial_size
        ).to(device=roles.object.device) > 0.5
    elif hand_anchor is not None:
        if hand_anchor.ndim != 4:
            raise ValueError(
                "hand_anchor must have shape [B,T,H,W]"
            )
        if hand_anchor.shape[:2] != object_probability.shape[:2]:
            raise ValueError(
                "hand_anchor must align with roles on [B,T]"
            )
        anchor_batch, anchor_frames = hand_anchor.shape[:2]
        resized_anchor = F.interpolate(
            hand_anchor.float().reshape(
                anchor_batch * anchor_frames, 1,
                *hand_anchor.shape[-2:],
            ),
            size=spatial_size,
            mode="nearest",
        ).reshape(
            anchor_batch, anchor_frames, *spatial_size
        ).to(device=roles.object.device)
        owner_probability = object_probability + contact_probability
        component_candidate = (
            owner_probability >= float(component_probability_threshold)
        )
        connected_support = select_hand_connected_support(
            component_candidate,
            owner_probability,
            resized_anchor > 0.0,
        )
    if owner_support is not None or hand_anchor is not None:
        # Reads retain the complete selected object/contact extent, while
        # writes remain restricted to the high-confidence non-hand core.
        read_gate = read_gate * connected_support.float()
    connected_write_gate = raw_write_gate & connected_support
    raw_write_gate = connected_write_gate
    if adaptive_write:
        expected_reliability_shape = object_probability.shape[:2] + (1, 1)
        if evidence_reliability is None:
            evidence_reliability = object_probability.new_ones(
                expected_reliability_shape
            )
        if evidence_reliability.shape != expected_reliability_shape:
            raise ValueError(
                "evidence_reliability must have shape [B,T,1,1]"
            )
        write_score = (
            object_probability
            * (1.0 - entropy)
            * (1.0 - contact_probability)
            * (1.0 - hand_probability)
        ).clamp(0.0, 1.0)
        adaptive_gate = torch.zeros_like(raw_write_gate)
        write_threshold = torch.ones_like(object_probability[:, :, :1, :1])
        for batch_index in range(object_probability.shape[0]):
            for frame_index in range(object_probability.shape[1]):
                eligible = raw_write_gate[batch_index, frame_index]
                values = write_score[batch_index, frame_index][eligible]
                if not values.numel():
                    continue
                reliability = float(evidence_reliability[
                    batch_index, frame_index, 0, 0
                ].clamp(0.0, 1.0))
                keep_fraction = 0.10 + 0.20 * reliability
                keep_count = max(
                    1,
                    int(math.ceil(values.numel() * keep_fraction)),
                )
                candidate_indices = torch.nonzero(
                    eligible, as_tuple=False
                )
                top_values, top_indices = torch.topk(
                    values.float(), keep_count
                )
                threshold = top_values[-1]
                write_threshold[batch_index, frame_index] = threshold
                selected_indices = candidate_indices[top_indices]
                adaptive_gate[
                    batch_index,
                    frame_index,
                    selected_indices[:, 0],
                    selected_indices[:, 1],
                ] = True
        raw_write_gate = adaptive_gate
    else:
        write_score = object_probability
        write_threshold = object_probability.new_full(
            object_probability.shape[:2] + (1, 1),
            float(object_write_threshold),
        )

    if temporal_recovery is not None:
        if temporal_recovery.shape != object_probability.shape[:2] + (1, 1):
            raise ValueError(
                "temporal_recovery must have shape [B,T,1,1]"
            )
        raw_write_gate = raw_write_gate & ~temporal_recovery.bool()

    preconsensus_write_gate = raw_write_gate.clone()

    if temporal_write_consensus:
        write_gate = torch.zeros_like(raw_write_gate)
        for frame_index in range(1, raw_write_gate.shape[1]):
            previous = _dilate(
                raw_write_gate[:, frame_index - 1:frame_index], 1
            )[:, 0]
            write_gate[:, frame_index] = (
                raw_write_gate[:, frame_index] & previous
            )
    else:
        write_gate = raw_write_gate
    return read_gate, write_gate, {
        "role_memory_object_probability": object_probability,
        "role_memory_contact_probability": contact_probability,
        "role_memory_hand_probability": hand_probability,
        "role_memory_background_probability": probabilities[:, :, 3],
        "role_memory_entropy": entropy,
        "role_memory_read_gate": read_gate,
        "role_memory_connected_support": connected_support.float(),
        "role_memory_candidate_write_gate": candidate_write_gate.float(),
        "role_memory_connected_write_gate": connected_write_gate.float(),
        "role_memory_write_score": write_score,
        "role_memory_write_threshold": write_threshold,
        "role_memory_raw_write_gate": candidate_write_gate.float(),
        "role_memory_preconsensus_write_gate": (
            preconsensus_write_gate.float()
        ),
        "role_memory_write_gate": write_gate.float(),
    }


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
