from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch
import torch.nn.functional as F

from .appearance_leakage import remove_antagonistic_source_residual
from .role_router import RoleState


def _resize_evidence(
    value: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Resize token/frame evidence to the VAE latent role grid."""
    if value.ndim != 4:
        raise ValueError(
            "Factorized evidence must have shape [B,T,H,W], got "
            f"{tuple(value.shape)}"
        )
    if value.shape[:2] != reference.shape[:2]:
        raise ValueError(
            "Factorized evidence and roles must share [B,T]"
        )
    if value.shape[-2:] == (1, 1):
        return value.float().expand_as(reference).clamp(0.0, 1.0)
    batch, frames = value.shape[:2]
    resized = F.interpolate(
        value.float().reshape(batch * frames, 1, *value.shape[-2:]),
        size=reference.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape_as(reference).clamp(0.0, 1.0)


def _resize_discrete_evidence(
    value: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Project token ownership without bleeding it across boundaries."""
    if value.ndim != 4 or value.shape[:2] != reference.shape[:2]:
        raise ValueError(
            "Discrete factorized evidence must align on [B,T]"
        )
    if value.shape[-2:] == (1, 1):
        return value.float().expand_as(reference).clamp(0.0, 1.0)
    batch, frames = value.shape[:2]
    resized = F.interpolate(
        value.float().reshape(batch * frames, 1, *value.shape[-2:]),
        size=reference.shape[-2:],
        mode="nearest",
    )
    return resized.reshape_as(reference).clamp(0.0, 1.0)


def _to_token_map(value: torch.Tensor) -> torch.Tensor:
    batch, frames, height, width = value.shape
    if height % 2 or width % 2:
        raise ValueError(
            "Factorized operator maps require an even latent grid"
        )
    return F.avg_pool2d(
        value.float().reshape(batch * frames, 1, height, width),
        kernel_size=2,
        stride=2,
    ).reshape(batch, frames, height // 2, width // 2)


@dataclass(frozen=True)
class FactorizedRolePosterior:
    """Disjoint role evidence plus an explicit unknown state.

    ``target_owned`` is deliberately an attribute of the object/boundary
    posterior rather than a fifth mutually-exclusive spatial role.
    """

    object: torch.Tensor
    boundary: torch.Tensor
    hand: torch.Tensor
    background: torch.Tensor
    unknown: torch.Tensor
    target_owned: torch.Tensor

    def validate(self) -> None:
        disjoint = {
            "object": self.object,
            "boundary": self.boundary,
            "hand": self.hand,
            "background": self.background,
            "unknown": self.unknown,
        }
        shapes = {tuple(value.shape) for value in disjoint.values()}
        if len(shapes) != 1 or self.object.ndim != 4:
            raise ValueError(
                "Factorized role maps must share shape [B,T,H,W]"
            )
        total = sum(disjoint.values())
        if not torch.allclose(total, torch.ones_like(total), atol=1e-5):
            raise ValueError(
                "Factorized role posterior must sum to one"
            )
        values = {**disjoint, "target_owned": self.target_owned}
        for name, value in values.items():
            if tuple(value.shape) != tuple(self.object.shape):
                raise ValueError(
                    f"Factorized role '{name}' has a different shape"
                )
            if not torch.isfinite(value.float()).all():
                raise ValueError(
                    f"Factorized role '{name}' is not finite"
                )
            if value.min() < 0 or value.max() > 1:
                raise ValueError(
                    f"Factorized role '{name}' must lie in [0, 1]"
                )
        role_support = (self.object + self.boundary) > 0
        if (self.target_owned > 0)[~role_support].any():
            raise ValueError(
                "Target ownership must overlap object or boundary support"
            )


@dataclass(frozen=True)
class FactorizedBayesOperators:
    """Independent operator actions and their token projections."""

    roles: FactorizedRolePosterior
    source_key_action_map: torch.Tensor
    source_value_action_map: torch.Tensor
    source_residual_action_map: torch.Tensor
    target_memory_action_map: torch.Tensor
    unknown_action_map: torch.Tensor
    source_key_action: torch.Tensor
    source_value_action: torch.Tensor
    source_residual_action: torch.Tensor
    target_memory_action: torch.Tensor
    unknown_action: torch.Tensor

    def as_debug_maps(self) -> Dict[str, torch.Tensor]:
        return {
            "factorized_role_object": self.roles.object,
            "factorized_role_boundary": self.roles.boundary,
            "factorized_role_hand": self.roles.hand,
            "factorized_role_background": self.roles.background,
            "factorized_role_unknown": self.roles.unknown,
            "factorized_target_owned": self.roles.target_owned,
            "factorized_source_key_action": (
                self.source_key_action_map
            ),
            "factorized_source_value_action": (
                self.source_value_action_map
            ),
            "factorized_source_residual_action": (
                self.source_residual_action_map
            ),
            "factorized_target_memory_action": (
                self.target_memory_action_map
            ),
            "factorized_unknown_action": self.unknown_action_map,
        }

    def validate(self, expected_token_length: int | None = None) -> None:
        self.roles.validate()
        maps = {
            "source_key_action_map": self.source_key_action_map,
            "source_value_action_map": self.source_value_action_map,
            "source_residual_action_map": (
                self.source_residual_action_map
            ),
            "target_memory_action_map": self.target_memory_action_map,
            "unknown_action_map": self.unknown_action_map,
        }
        tokens = {
            "source_key_action": self.source_key_action,
            "source_value_action": self.source_value_action,
            "source_residual_action": self.source_residual_action,
            "target_memory_action": self.target_memory_action,
            "unknown_action": self.unknown_action,
        }
        if len({tuple(value.shape) for value in maps.values()}) != 1:
            raise ValueError(
                "Factorized operator maps must share a shape"
            )
        if len({tuple(value.shape) for value in tokens.values()}) != 1:
            raise ValueError(
                "Factorized operator tokens must share a shape"
            )
        if any(value.ndim != 2 for value in tokens.values()):
            raise ValueError(
                "Factorized operator tokens must have shape [B,L]"
            )
        if expected_token_length is not None and (
            self.source_key_action.shape[1] != expected_token_length
        ):
            raise ValueError(
                "Factorized operator token length does not match attention"
            )
        for name, value in {**maps, **tokens}.items():
            if not torch.isfinite(value.float()).all():
                raise ValueError(
                    f"Factorized operator '{name}' is not finite"
                )
            if value.min() < 0 or value.max() > 1:
                raise ValueError(
                    f"Factorized operator '{name}' must lie in [0, 1]"
                )
        provenance_sum = (
            self.source_value_action_map
            + self.target_memory_action_map
            + self.unknown_action_map
        )
        if not torch.allclose(
            provenance_sum,
            torch.ones_like(provenance_sum),
            atol=1e-5,
        ):
            raise ValueError(
                "Source, target, and unknown provenance must sum to one"
            )
        token_provenance_sum = (
            self.source_value_action
            + self.target_memory_action
            + self.unknown_action
        )
        if not torch.allclose(
            token_provenance_sum,
            torch.ones_like(token_provenance_sum),
            atol=1e-5,
        ):
            raise ValueError(
                "Token source, target, and unknown provenance must sum to one"
            )


class FactorizedBayesOperatorBuilder:
    """Convert role evidence into asymmetric memory/flow operators.

    Missing object evidence is not background evidence. Background ownership
    requires reliable negative source-text attention away from the hand; all
    remaining mass is explicitly unknown and falls back to native StreamGVE.
    """

    def __init__(
        self,
        boundary_source_fraction: float = 0.25,
        eps: float = 1e-6,
    ):
        if not 0.0 <= boundary_source_fraction <= 1.0:
            raise ValueError(
                "boundary_source_fraction must lie in [0, 1]"
            )
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.boundary_source_fraction = boundary_source_fraction
        self.eps = eps

    @staticmethod
    def _required(
        evidence: Mapping[str, torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        value = evidence.get(name)
        if value is None:
            raise ValueError(
                f"Missing factorized Bayes evidence: {name}"
            )
        return value

    def __call__(
        self,
        roles: RoleState,
        evidence: Mapping[str, torch.Tensor],
        expected_token_length: int,
    ) -> FactorizedBayesOperators:
        if expected_token_length <= 0:
            raise ValueError(
                "expected_token_length must be positive"
            )
        roles.validate()
        reference = roles.object.float()
        object_posterior = _resize_evidence(
            self._required(evidence, "object_posterior"), reference
        )
        posterior_threshold = _resize_evidence(
            self._required(evidence, "posterior_threshold"), reference
        )
        source_attention = _resize_evidence(
            self._required(evidence, "source_attention"), reference
        )
        hand_proximity = _resize_evidence(
            self._required(evidence, "hand_proximity"), reference
        )
        attention_reliability = _resize_evidence(
            self._required(
                evidence, "adaptive_attention_reliability"
            ),
            reference,
        )
        visibility = _resize_evidence(
            self._required(evidence, "object_visible"), reference
        )
        temporal_confidence = _resize_evidence(
            self._required(evidence, "temporal_confidence"), reference
        )
        causal_owner_value = evidence.get("causal_owner_weight")
        causal_owner = (
            torch.zeros_like(reference)
            if causal_owner_value is None
            else _resize_evidence(causal_owner_value, reference)
        )
        causal_owner_support_value = evidence.get(
            "causal_owner_support"
        )
        causal_owner_support = (
            causal_owner > self.eps
            if causal_owner_support_value is None
            else _resize_discrete_evidence(
                causal_owner_support_value, reference
            ).bool()
        )
        hard_hand_value = evidence.get("hand_hard_exclusion")
        hard_hand = (
            roles.hand > (1.0 - self.eps)
            if hard_hand_value is None
            else _resize_discrete_evidence(
                hard_hand_value, reference
            ).bool()
        )
        edit_authority_value = evidence.get("edit_authority")
        semantic_authority_enabled = edit_authority_value is not None
        edit_authority = (
            torch.ones_like(reference)
            if edit_authority_value is None
            else _resize_evidence(edit_authority_value, reference)
        )

        relative_object = (
            object_posterior
            / posterior_threshold.clamp_min(self.eps)
        ).clamp(0.0, 1.0)
        semantic_support = torch.maximum(
            source_attention, temporal_confidence
        )
        object_confidence = torch.sqrt(
            (
                relative_object
                * semantic_support
                * attention_reliability
            ).clamp(0.0, 1.0)
        ) * visibility
        background_confidence = (
            attention_reliability
            * (1.0 - source_attention)
            * (1.0 - hand_proximity)
        ).clamp(0.0, 1.0)
        flow_object_value = evidence.get("flow_object_likelihood")
        flow_background_value = evidence.get("flow_background_likelihood")
        flow_boundary_value = evidence.get("flow_boundary_likelihood")
        flow_unknown_value = evidence.get("flow_unknown_likelihood")
        flow_transport_value = evidence.get("flow_transport_support")
        flow_roles_enabled = all(
            value is not None
            for value in (
                flow_object_value, flow_background_value,
                flow_boundary_value, flow_unknown_value,
                flow_transport_value,
            )
        )
        if flow_roles_enabled:
            flow_object = _resize_evidence(flow_object_value, reference)
            flow_background = _resize_evidence(
                flow_background_value, reference
            )
            flow_boundary = _resize_evidence(
                flow_boundary_value, reference
            )
            flow_unknown = _resize_evidence(flow_unknown_value, reference)
            flow_transport = _resize_evidence(
                flow_transport_value, reference
            )
            # Bayesian product-of-complements: source semantics and source
            # flow are independent positive observations. Neither must be
            # present for the other to survive.
            object_confidence = (
                1.0 - (1.0 - object_confidence) * (1.0 - flow_object)
            ).clamp(0.0, 1.0)
            background_confidence = (
                1.0
                - (1.0 - background_confidence)
                * (1.0 - flow_background)
            ).clamp(0.0, 1.0)

        object_role = roles.object.float() * object_confidence
        boundary_role = roles.boundary.float() * object_confidence
        # The provided hand mask is direct source evidence and is therefore
        # not weakened by text-attention uncertainty.
        hand_role = roles.hand.float()
        background_role = (
            roles.background.float() * background_confidence
        )
        # 919: once an object has been ignited, clean-source feature
        # correspondence is an independent ownership observation.  It may
        # reclaim mass that the per-block text detector labelled background
        # or unknown, but it cannot overwrite the supplied hand mask.
        causal_owner = (
            causal_owner
            * causal_owner_support.float()
            * (~hard_hand).float()
        ).clamp(0.0, 1.0)
        causal_owner_support = (
            causal_owner_support
            & ~hard_hand
            & (causal_owner > self.eps)
        )
        if flow_roles_enabled:
            # Flow is allowed to redistribute only non-hand provenance, and
            # only where the independently transported owner already exists.
            flow_owner = (
                flow_object * flow_transport * (~hard_hand).float()
            ).clamp(0.0, 1.0)
            causal_owner = torch.maximum(causal_owner, flow_owner)
            causal_owner_support = causal_owner_support | (
                flow_owner > self.eps
            )
        known_mass = (
            object_role + boundary_role + hand_role + background_role
        ).clamp(0.0, 1.0)
        unknown_role = (1.0 - known_mass).clamp(0.0, 1.0)
        # Ownership is a convex reallocation over all non-hand provenance.
        # Moving only background/unknown and then adding the full owner to
        # object double-counts mass already assigned to the soft object or
        # boundary roles.  The convex form is exactly conservative.
        transferable_mass = (
            object_role + boundary_role + background_role + unknown_role
        ).clamp(0.0, 1.0)
        retained = 1.0 - causal_owner
        object_role = (
            object_role * retained
            + transferable_mass * causal_owner
        )
        boundary_role = boundary_role * retained
        background_role = background_role * retained
        unknown_role = unknown_role * retained
        if flow_roles_enabled:
            # Boundary/unknown flow is uncertainty allocation, never edit
            # authority. Re-normalization below preserves a proper posterior.
            boundary_role = torch.maximum(
                boundary_role, flow_boundary * (1.0 - object_role)
            )
            unknown_role = torch.maximum(
                unknown_role, flow_unknown * (1.0 - object_role)
            )
            background_role = torch.maximum(
                background_role,
                flow_background
                * (1.0 - object_role)
                * (1.0 - boundary_role),
            )
            non_hand_total = (
                object_role + boundary_role + background_role + unknown_role
            ).clamp_min(self.eps)
            available = (1.0 - hand_role).clamp(0.0, 1.0)
            scale = available / non_hand_total
            object_role = object_role * scale
            boundary_role = boundary_role * scale
            background_role = background_role * scale
            unknown_role = unknown_role * scale
        known_mass = (
            object_role + boundary_role + hand_role + background_role
        ).clamp(0.0, 1.0)
        # Target ownership is an attribute of the *final calibrated* object
        # or boundary posterior.  A transported causal owner may reallocate
        # provenance above, but it must not create target ownership outside
        # that same role support domain.
        target_role_support = (object_role + boundary_role) > self.eps
        detector_owned = (
            (object_posterior >= posterior_threshold)
            & (roles.object > 0)
            & (roles.boundary <= self.eps)
        )
        target_owned = (
            (detector_owned | causal_owner_support)
            & target_role_support
            & ~hard_hand
        ).float()
        posterior = FactorizedRolePosterior(
            object=object_role,
            boundary=boundary_role,
            hand=hand_role,
            background=background_role,
            unknown=unknown_role,
            target_owned=target_owned,
        )
        posterior.validate()

        boundary_source = (
            self.boundary_source_fraction * boundary_role
        )
        source_key_map = torch.maximum(known_mass, target_owned)
        source_value_map = (
            hand_role + background_role + boundary_source
        ).clamp(0.0, 1.0)
        target_candidate_map = (
            object_role
            + (1.0 - self.boundary_source_fraction) * boundary_role
        ).clamp(0.0, 1.0)
        # Ownership identifies the complete moving instance, whereas edit
        # authority identifies only the prompt-requested part.  Conflating
        # them made a cap-color edit rewrite the bottle body and nearby pan.
        # Denied target mass is assigned to clean-source preservation; it is
        # not converted to uncertainty and cannot silently fall back to the
        # target branch later.
        if semantic_authority_enabled:
            # Competitive target semantics make the provenance decision
            # explicit: authority is target-valued and its complement is
            # clean-source-valued.  This closes both the target-memory and
            # unknown/native target paths outside the edited part.
            target_memory_map = edit_authority
            source_value_map = 1.0 - edit_authority
            unknown_action_map = torch.zeros_like(unknown_role)
        else:
            # Exact legacy behavior for all experiments that do not enable
            # target semantic competition.
            target_memory_map = target_candidate_map
            owner_transfer = source_value_map * target_owned
            owner_unknown_transfer = unknown_role * target_owned
            source_value_map = (
                source_value_map - owner_transfer
            ).clamp(0.0, 1.0)
            target_memory_map = (
                target_memory_map
                + owner_transfer
                + owner_unknown_transfer
            ).clamp(0.0, 1.0)
            unknown_action_map = (
                unknown_role - owner_unknown_transfer
            ).clamp(0.0, 1.0)
        source_residual_map = source_value_map.clone()

        token_maps = {
            "source_key": _to_token_map(source_key_map),
            "source_value": _to_token_map(source_value_map),
            "source_residual": _to_token_map(source_residual_map),
            "target_memory": _to_token_map(target_memory_map),
            "unknown": _to_token_map(unknown_action_map),
        }
        token_owner_key = F.max_pool2d(
            target_owned.reshape(
                reference.shape[0] * reference.shape[1],
                1,
                *reference.shape[-2:],
            ),
            kernel_size=2,
            stride=2,
        ).reshape_as(token_maps["source_key"])
        token_maps["source_key"] = torch.maximum(
            token_maps["source_key"], token_owner_key
        )
        if not semantic_authority_enabled:
            # Preserve the historical max-pool promotion only in legacy
            # owner-is-edit experiments.  Semantic authority has already
            # been applied before pooling and must not be spatially expanded.
            token_target_owned = F.max_pool2d(
                target_owned.reshape(
                    reference.shape[0] * reference.shape[1],
                    1,
                    *reference.shape[-2:],
                ),
                kernel_size=2,
                stride=2,
            ).reshape_as(token_maps["source_value"]) > 0
            token_owner_source = (
                token_maps["source_value"]
                * token_target_owned.float()
            )
            token_owner_unknown = (
                token_maps["unknown"]
                * token_target_owned.float()
            )
            token_maps["source_value"] = (
                token_maps["source_value"] - token_owner_source
            )
            token_maps["source_residual"] = (
                token_maps["source_residual"]
                * (~token_target_owned).float()
            )
            token_maps["unknown"] = (
                token_maps["unknown"] - token_owner_unknown
            )
            token_maps["target_memory"] = (
                token_maps["target_memory"]
                + token_owner_source
                + token_owner_unknown
            ).clamp(0.0, 1.0)
        operators = FactorizedBayesOperators(
            roles=posterior,
            source_key_action_map=source_key_map,
            source_value_action_map=source_value_map,
            source_residual_action_map=source_residual_map,
            target_memory_action_map=target_memory_map,
            unknown_action_map=unknown_action_map,
            source_key_action=token_maps["source_key"].reshape(
                reference.shape[0], -1
            ),
            source_value_action=token_maps["source_value"].reshape(
                reference.shape[0], -1
            ),
            source_residual_action=token_maps["source_residual"].reshape(
                reference.shape[0], -1
            ),
            target_memory_action=token_maps["target_memory"].reshape(
                reference.shape[0], -1
            ),
            unknown_action=token_maps["unknown"].reshape(
                reference.shape[0], -1
            ),
        )
        operators.validate(expected_token_length)
        return operators


def route_factorized_velocity(
    target_velocity: torch.Tensor,
    source_velocity: torch.Tensor,
    source_reconstruction_velocity: torch.Tensor,
    operators: FactorizedBayesOperators,
    native_fallback_action: torch.Tensor,
    *,
    target_owned_weight: torch.Tensor | None = None,
    block_target_owned_source: bool = True,
    geometry_owner_weight: torch.Tensor | None = None,
    geometry_strength: float = 0.0,
    denoising_fraction: float = 1.0,
    source_coordinate_target_delta: bool = False,
    owner_complement_source_weight: torch.Tensor | None = None,
    owner_complement_margin: int = 1,
    owner_complement_min_preserve_confidence: float = 0.0,
    paired_memory_support_weight: torch.Tensor | None = None,
    paired_memory_source_suppression: float = 0.0,
    verified_native_history_support_weight: torch.Tensor | None = None,
    verified_native_history_source_suppression: float = 0.0,
    verified_native_history_appearance_projection: bool = False,
    edit_authority_weight: torch.Tensor | None = None,
):
    """Route provenance and an independent source-geometry channel.

    Source/target/unknown remain an exhaustive Bayes provenance partition.
    Geometry is deliberately not another provenance class: when enabled, the
    clean-source reconstruction residual is projected away from the component
    opposing the current edit direction, then injected only inside the object
    owner.  This preserves motion/shape without reopening a direct source-
    appearance value path.
    """
    if not (
        target_velocity.shape
        == source_velocity.shape
        == source_reconstruction_velocity.shape
    ):
        raise ValueError(
            "Factorized velocity inputs must share a shape"
        )
    if native_fallback_action.shape != (
        target_velocity.shape[0],
        target_velocity.shape[1],
        1,
        *target_velocity.shape[-2:],
    ):
        raise ValueError(
            "Native fallback action must have shape [B,T,1,H,W]"
        )
    if not 0.0 <= geometry_strength <= 1.0:
        raise ValueError("geometry_strength must lie in [0, 1]")
    if not 0.0 <= denoising_fraction <= 1.0:
        raise ValueError("denoising_fraction must lie in [0, 1]")
    if not 0.0 <= paired_memory_source_suppression <= 1.0:
        raise ValueError(
            "paired_memory_source_suppression must lie in [0, 1]"
        )
    if not (
        0.0 <= verified_native_history_source_suppression <= 1.0
    ):
        raise ValueError(
            "Verified native-history source suppression must lie in "
            "[0, 1]"
        )
    if owner_complement_margin < 0:
        raise ValueError(
            "owner_complement_margin must be non-negative"
        )
    if not 0.0 <= owner_complement_min_preserve_confidence <= 1.0:
        raise ValueError(
            "owner_complement_min_preserve_confidence must lie in "
            "[0, 1]"
        )
    batch, frames = target_velocity.shape[:2]

    def resize(value: torch.Tensor) -> torch.Tensor:
        result = F.interpolate(
            value.float().reshape(batch * frames, 1, *value.shape[-2:]),
            size=target_velocity.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return result.reshape(
            batch, frames, 1, *target_velocity.shape[-2:]
        ).clamp(0.0, 1.0)

    def resize_discrete(value: torch.Tensor) -> torch.Tensor:
        result = F.interpolate(
            value.float().reshape(batch * frames, 1, *value.shape[-2:]),
            size=target_velocity.shape[-2:],
            mode="nearest",
        )
        return result.reshape(
            batch, frames, 1, *target_velocity.shape[-2:]
        ) > 0.0

    source_action = resize(operators.source_residual_action_map)
    unknown_action = resize(operators.unknown_action_map)
    target_owned_action = torch.zeros_like(source_action)
    if target_owned_weight is not None:
        if target_owned_weight.ndim != 4:
            raise ValueError(
                "target_owned_weight must have shape [B,T,H,W]"
            )
        if target_owned_weight.shape[:2] != (batch, frames):
            raise ValueError(
                "Target ownership and velocity must share [B,T]"
            )
        target_owned_action = resize(target_owned_weight)
    if block_target_owned_source:
        source_action = (
            source_action * (1.0 - target_owned_action)
        ).clamp(0.0, 1.0)
        owner_fallback_action = torch.zeros_like(source_action)
    else:
        # 923a counterfactual: reproduce native source-oriented fallback on
        # the owner while keeping all other Bayes routing unchanged.  923b
        # removes exactly this term, isolating source-appearance blocking.
        owner_fallback_action = (
            target_owned_action * (1.0 - unknown_action)
        ).clamp(0.0, 1.0)
    effective_action = (
        source_action
        + (unknown_action + owner_fallback_action).clamp(0.0, 1.0)
        * native_fallback_action.float()
    ).clamp(0.0, 1.0)
    effective_action_before_paired_arbitration = effective_action
    paired_memory_action = torch.zeros_like(effective_action)
    if paired_memory_support_weight is not None:
        if paired_memory_support_weight.ndim != 4:
            raise ValueError(
                "Paired-memory support must have shape [B,T,H,W]"
            )
        if paired_memory_support_weight.shape[:2] != (batch, frames):
            raise ValueError(
                "Paired-memory support and velocity must share [B,T]"
            )
        paired_memory_action = (
            resize(paired_memory_support_weight)
            * float(paired_memory_source_suppression)
        ).clamp(0.0, 1.0)
        # Only a successful source-addressed read arbitrates the competing
        # source residual. A missing or uncertain read is an exact fallback
        # to the native routing, unlike an owner-wide hard source block.
        effective_action = (
            effective_action * (1.0 - paired_memory_action)
        ).clamp(0.0, 1.0)
    verified_native_history_action = torch.zeros_like(effective_action)
    if verified_native_history_support_weight is not None:
        if verified_native_history_support_weight.ndim != 4:
            raise ValueError(
                "Verified native-history support must have shape "
                "[B,T,H,W]"
            )
        if verified_native_history_support_weight.shape[:2] != (
            batch, frames
        ):
            raise ValueError(
                "Verified native-history support and velocity must "
                "share [B,T]"
            )
        verified_native_history_action = (
            resize(verified_native_history_support_weight)
            * float(verified_native_history_source_suppression)
        ).clamp(0.0, 1.0)
        # This gate is deliberately downstream of attention retrieval. An
        # owner proposal alone cannot suppress source reconstruction: at
        # least one configured native-KV layer must have admitted the query.
        if not verified_native_history_appearance_projection:
            effective_action = (
                effective_action
                * (1.0 - verified_native_history_action)
            ).clamp(0.0, 1.0)
    source_residual = (
        source_reconstruction_velocity.float()
        - source_velocity.float()
    )
    verified_appearance_removed = torch.zeros_like(source_residual)
    if (
        verified_native_history_appearance_projection
        and bool((verified_native_history_action > 0.0).any())
    ):
        # Retrieval proves target appearance is available at this query.
        # Remove only the source-residual component opposing the target edit;
        # retain orthogonal motion, scale, illumination, and occlusion energy.
        appearance_safe_residual, _ = remove_antagonistic_source_residual(
            source_residual=source_residual,
            edit_direction=(
                target_velocity.float() - source_velocity.float()
            ),
            target_change_core=(
                verified_native_history_action.squeeze(2) > 0.0
            ),
        )
        verified_appearance_removed = (
            source_residual - appearance_safe_residual.float()
        ) * verified_native_history_action
        source_residual = (
            source_residual - verified_appearance_removed
        )
    geometry_action = torch.zeros_like(effective_action)
    geometry_residual = torch.zeros_like(source_residual)
    geometry_diagnostics = {}
    if geometry_owner_weight is not None:
        if geometry_owner_weight.ndim != 4:
            raise ValueError(
                "geometry_owner_weight must have shape [B,T,H,W]"
            )
        if geometry_owner_weight.shape[:2] != (batch, frames):
            raise ValueError(
                "Geometry owner and velocity must share [B,T]"
            )
        owner_map = geometry_owner_weight
        geometry_action = resize(owner_map)
        geometry_action = (
            geometry_action
            * float(geometry_strength)
            * float(denoising_fraction)
            * (1.0 - effective_action)
        ).clamp(0.0, 1.0)
        geometry_residual, geometry_diagnostics = (
            remove_antagonistic_source_residual(
                source_residual=source_residual,
                edit_direction=(
                    target_velocity.float() - source_velocity.float()
                ),
                target_change_core=(geometry_action.squeeze(2) > 0.0),
            )
        )
    target_delta_action = torch.zeros_like(effective_action)
    if source_coordinate_target_delta:
        # The source reconstruction is the per-frame coordinate/geometry
        # base. Only the target-minus-source semantic delta is spatially
        # gated, so background and hand receive no target-prompt change.
        # Unlike a fixed geometry envelope, the base follows the current
        # source frame and therefore retains real perspective and scale.
        target_delta_action = resize(
            operators.target_memory_action_map
        )
        routed = (
            source_reconstruction_velocity.float()
            + target_delta_action
            * (target_velocity.float() - source_velocity.float())
        ).to(target_velocity.dtype)
    else:
        routed = (
            target_velocity.float()
            + effective_action * source_residual
            + geometry_action * geometry_residual.float()
        ).to(target_velocity.dtype)

    # The residual formulation above preserves source information but still
    # contains the complete target-minus-source semantic direction.  That is
    # desirable around the edited object, yet it lets a strong color prompt
    # weakly tint distant background regions.  When an explicit source owner
    # is available, close that path only on its complement: the owner and a
    # small safety band keep the original factorized/native-KV result, while
    # definite non-owner pixels use the clean-source reconstruction velocity.
    # This is a per-frame coordinate constraint, not a fixed size envelope.
    owner_complement_source_action = torch.zeros_like(effective_action)
    owner_complement_abstain_action = torch.zeros_like(effective_action)
    owner_complement_preserve_confidence = torch.zeros_like(
        effective_action
    )
    if owner_complement_source_weight is not None:
        owner_reference = operators.target_memory_action_map
        if owner_reference.ndim != 4:
            raise ValueError(
                "Factorized target-memory action must have shape "
                "[B,T,H,W]"
            )
        owner_grid_shape = owner_reference.shape
        owner_weight = owner_complement_source_weight
        # CausalObjectOwnership stores token ownership canonically as [B,L],
        # while role posteriors use [B,T,H,W].  Accept both representations
        # here so a caller cannot accidentally make the routing depend on a
        # debug-only reshape.  Singleton-channel spatial masks are accepted
        # as well for compatibility with velocity-space utilities.
        if owner_weight.ndim == 2:
            if (
                owner_weight.shape[0] != batch
                or owner_weight.shape[1]
                != frames * owner_grid_shape[-2] * owner_grid_shape[-1]
            ):
                raise ValueError(
                    "Flattened owner-complement source weight must match "
                    "the factorized [B,T,H,W] token grid"
                )
            owner_weight = owner_weight.reshape(owner_grid_shape)
        elif owner_weight.ndim == 5 and owner_weight.shape[2] == 1:
            owner_weight = owner_weight.squeeze(2)
        elif owner_weight.ndim == 5 and owner_weight.shape[1] == 1:
            owner_weight = owner_weight.squeeze(1)
        if owner_weight.ndim != 4:
            raise ValueError(
                "owner_complement_source_weight must have shape "
                "[B,L], [B,T,H,W], or a singleton-channel spatial shape"
            )
        if owner_weight.shape[:2] != (batch, frames):
            raise ValueError(
                "Owner-complement source weight and velocity must "
                "share [B,T]"
            )
        owner_grid = owner_weight.detach().float().clamp(
            0.0, 1.0
        )
        raw_owner = owner_grid > 0.0
        # Missing owner evidence is not positive background evidence. During
        # hand occlusion or an image exit, the visible owner can vanish while
        # the edited target is still present. Close to the source only for
        # confident preserve roles; uncertain complement pixels abstain and
        # retain the native target/KV route. A zero threshold reproduces the
        # original 938 all-complement ablation exactly.
        preserve_confidence_grid = (
            operators.roles.hand.float()
            + operators.roles.background.float()
        ).clamp(0.0, 1.0)
        owner_complement_preserve_confidence = resize(
            preserve_confidence_grid
        )
        if owner_complement_min_preserve_confidence > 0.0:
            confident_preserve = (
                owner_complement_preserve_confidence
                >= owner_complement_min_preserve_confidence
            )
        else:
            confident_preserve = torch.ones_like(
                owner_complement_preserve_confidence, dtype=torch.bool
            )
        owner = owner_grid
        if owner_complement_margin > 0:
            owner = F.max_pool2d(
                owner.reshape(batch * frames, 1, *owner.shape[-2:]),
                kernel_size=2 * owner_complement_margin + 1,
                stride=1,
                padding=owner_complement_margin,
            ).reshape_as(owner)
        owner = F.interpolate(
            owner.reshape(batch * frames, 1, *owner.shape[-2:]),
            size=target_velocity.shape[-2:],
            mode="nearest",
        ).reshape(
            batch, frames, 1, *target_velocity.shape[-2:]
        ) > 0.0
        source_closure = (~owner) & confident_preserve
        owner_complement_source_action = source_closure.to(
            dtype=effective_action.dtype
        )
        owner_complement_abstain_action = (
            (~owner) & (~confident_preserve)
        ).to(dtype=effective_action.dtype)
        routed = torch.where(
            source_closure,
            source_reconstruction_velocity,
            routed,
        )
    semantic_edit_action = torch.ones_like(effective_action)
    semantic_preserve_action = torch.zeros_like(effective_action)
    if edit_authority_weight is not None:
        if edit_authority_weight.ndim != 4:
            raise ValueError(
                "edit_authority_weight must have shape [B,T,H,W]"
            )
        if edit_authority_weight.shape[:2] != (batch, frames):
            raise ValueError(
                "Edit authority and velocity must share [B,T]"
            )
        semantic_edit_action = resize(edit_authority_weight)
        semantic_preserve_action = 1.0 - semantic_edit_action
        # Source reconstruction is the closed/default branch.  Only local
        # competitive semantic authority opens the target-generation path.
        routed = (
            source_reconstruction_velocity.float()
            + semantic_edit_action
            * (
                routed.float()
                - source_reconstruction_velocity.float()
            )
        ).to(target_velocity.dtype)
    diagnostics = {
        "source_residual_action": source_action,
        "unknown_action": unknown_action,
        "target_owned_action": target_owned_action,
        "target_owned_native_fallback_action": (
            owner_fallback_action * native_fallback_action.float()
        ),
        "native_fallback_action": native_fallback_action.float(),
        "effective_source_residual_action": effective_action,
        "effective_source_residual_before_paired_arbitration": (
            effective_action_before_paired_arbitration
        ),
        "paired_memory_source_suppression_action": (
            paired_memory_action
        ),
        "verified_native_history_source_suppression_action": (
            verified_native_history_action
        ),
        "verified_native_history_appearance_removed_energy": (
            verified_appearance_removed.square().mean(
                dim=2, keepdim=True
            )
        ),
        "orthogonal_geometry_action": geometry_action,
        "source_coordinate_target_delta_action": (
            target_delta_action
        ),
        "owner_complement_source_action": (
            owner_complement_source_action
        ),
        "owner_complement_abstain_action": (
            owner_complement_abstain_action
        ),
        "owner_complement_preserve_confidence": (
            owner_complement_preserve_confidence
        ),
        "semantic_edit_action": semantic_edit_action,
        "semantic_preserve_action": semantic_preserve_action,
        "orthogonal_geometry_residual_abs": (
            geometry_residual.float().abs().mean(dim=2, keepdim=True)
        ),
    }
    diagnostics.update({
        f"orthogonal_geometry_{name}": value
        for name, value in geometry_diagnostics.items()
    })
    return routed, diagnostics
