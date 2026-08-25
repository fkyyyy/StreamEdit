from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Mapping

import torch
import torch.nn.functional as F

from .control_belief import CausalControlBelief


@dataclass(frozen=True)
class TargetIdentityLayerState:
    """Slow, position-free target appearance prototypes for one layer."""

    key: torch.Tensor
    value: torch.Tensor
    evidence: torch.Tensor

    def validate(self) -> None:
        if self.key.ndim != 4:
            raise ValueError(
                "Identity keys must have shape [B,P,H,D]"
            )
        if self.value.shape != self.key.shape:
            raise ValueError(
                "Identity keys and values must share shape"
            )
        if self.evidence.shape != self.key.shape[:2]:
            raise ValueError(
                "Identity evidence must have shape [B,P]"
            )
        for name, value in (
            ("key", self.key),
            ("value", self.value),
            ("evidence", self.evidence),
        ):
            if not torch.isfinite(value.float()).all():
                raise ValueError(
                    f"Target identity state '{name}' is not finite"
                )
        if self.evidence.min() < 0:
            raise ValueError(
                "Target identity evidence must be non-negative"
            )


@dataclass(frozen=True)
class TargetIdentityUpdate:
    """Diagnostics for one slow identity-memory update."""

    write_weight: torch.Tensor
    observation_evidence: torch.Tensor
    update_gain: torch.Tensor
    accumulated_evidence: torch.Tensor

    def validate(self) -> None:
        if self.write_weight.ndim != 2:
            raise ValueError(
                "Identity write weights must have shape [B,L]"
            )
        layer_shapes = {
            tuple(value.shape)
            for value in (
                self.observation_evidence,
                self.update_gain,
                self.accumulated_evidence,
            )
        }
        if len(layer_shapes) != 1:
            raise ValueError(
                "Identity update diagnostics must share [layers,B,P]"
            )
        for name, value in (
            ("write_weight", self.write_weight),
            ("observation_evidence", self.observation_evidence),
            ("update_gain", self.update_gain),
            ("accumulated_evidence", self.accumulated_evidence),
        ):
            if not torch.isfinite(value.float()).all():
                raise ValueError(
                    f"Identity update '{name}' is not finite"
                )
            if value.min() < 0:
                raise ValueError(
                    f"Identity update '{name}' must be non-negative"
                )


@dataclass(frozen=True)
class FirstFrameIdentityBootstrap:
    """High-confidence object core used for causal first-frame identity."""

    write_weight: torch.Tensor
    base_write_weight: torch.Tensor
    object_likelihood: torch.Tensor
    core_mask: torch.Tensor

    def validate(self) -> None:
        if self.base_write_weight.ndim != 4:
            raise ValueError(
                "Bootstrap maps must have shape [B,T,H,W]"
            )
        expected_shape = self.base_write_weight.shape
        for name, value in (
            ("object_likelihood", self.object_likelihood),
            ("core_mask", self.core_mask),
        ):
            if value.shape != expected_shape:
                raise ValueError(
                    f"Bootstrap '{name}' must match [B,T,H,W]"
                )
        if self.write_weight.shape != (
            expected_shape[0],
            math.prod(expected_shape[1:]),
        ):
            raise ValueError(
                "Bootstrap write weights must flatten [T,H,W]"
            )
        for name, value in (
            ("write_weight", self.write_weight),
            ("base_write_weight", self.base_write_weight),
            ("object_likelihood", self.object_likelihood),
            ("core_mask", self.core_mask.float()),
        ):
            if not torch.isfinite(value.float()).all():
                raise ValueError(
                    f"Bootstrap '{name}' is not finite"
                )
        for name, value in (
            ("write_weight", self.write_weight),
            ("base_write_weight", self.base_write_weight),
            ("object_likelihood", self.object_likelihood),
        ):
            if value.min() < 0 or value.max() > 1:
                raise ValueError(
                    f"Bootstrap '{name}' must lie in [0, 1]"
                )

    def as_debug_maps(self) -> Dict[str, torch.Tensor]:
        return {
            "identity_causal_bootstrap_base": (
                self.base_write_weight.float()
            ),
            "identity_causal_bootstrap_object_likelihood": (
                self.object_likelihood.float()
            ),
            "identity_causal_bootstrap_core": self.core_mask.float(),
            "identity_causal_bootstrap_weight": (
                self.write_weight.reshape_as(
                    self.base_write_weight
                ).float()
            ),
        }


@dataclass(frozen=True)
class TargetIdentityTokenPropagation:
    """Causal token-match gating applied before identity-memory writes."""

    write_weight: torch.Tensor
    base_write_weight: torch.Tensor
    support_weight: torch.Tensor
    match_confidence: torch.Tensor
    best_similarity: torch.Tensor
    matched_previous_weight: torch.Tensor
    has_previous: torch.Tensor

    def validate(self) -> None:
        if self.write_weight.ndim != 2:
            raise ValueError(
                "Token propagation weights must have shape [B,L]"
            )
        expected_shape = self.write_weight.shape
        for name, value in (
            ("base_write_weight", self.base_write_weight),
            ("support_weight", self.support_weight),
            ("match_confidence", self.match_confidence),
            ("best_similarity", self.best_similarity),
            (
                "matched_previous_weight",
                self.matched_previous_weight,
            ),
        ):
            if value.shape != expected_shape:
                raise ValueError(
                    f"Token propagation '{name}' must match [B,L]"
                )
        if self.has_previous.shape != tuple(expected_shape[:1]) + (1,):
            raise ValueError(
                "Token propagation has_previous must have shape [B,1]"
            )
        for name, value in (
            ("write_weight", self.write_weight),
            ("base_write_weight", self.base_write_weight),
            ("support_weight", self.support_weight),
            ("match_confidence", self.match_confidence),
            ("best_similarity", self.best_similarity),
            (
                "matched_previous_weight",
                self.matched_previous_weight,
            ),
            ("has_previous", self.has_previous.float()),
        ):
            if not torch.isfinite(value.float()).all():
                raise ValueError(
                    f"Token propagation '{name}' is not finite"
                )
        for name, value in (
            ("write_weight", self.write_weight),
            ("base_write_weight", self.base_write_weight),
            ("support_weight", self.support_weight),
            ("match_confidence", self.match_confidence),
            (
                "matched_previous_weight",
                self.matched_previous_weight,
            ),
        ):
            if value.min() < 0 or value.max() > 1:
                raise ValueError(
                    f"Token propagation '{name}' must lie in [0, 1]"
                )
        if self.best_similarity.min() < -1 or self.best_similarity.max() > 1:
            raise ValueError(
                "Token propagation best_similarity must lie in [-1, 1]"
            )


@dataclass(frozen=True)
class ConnectedIdentitySupport:
    """Spatially coherent causal support retained for identity transport."""

    weight: torch.Tensor
    candidate_mask: torch.Tensor
    keep_mask: torch.Tensor
    anchor_mask: torch.Tensor
    object_likelihood_mask: torch.Tensor
    budget_fraction: torch.Tensor

    def validate(self) -> None:
        if self.weight.ndim != 4:
            raise ValueError(
                "Connected support weight must have shape [B,T,H,W]"
            )
        expected_shape = self.weight.shape
        for name, value in (
            ("candidate_mask", self.candidate_mask),
            ("keep_mask", self.keep_mask),
            ("anchor_mask", self.anchor_mask),
            (
                "object_likelihood_mask",
                self.object_likelihood_mask,
            ),
        ):
            if value.shape != expected_shape:
                raise ValueError(
                    f"Connected support '{name}' must match [B,T,H,W]"
                )
        if self.budget_fraction.shape != expected_shape[:2] + (1, 1):
            raise ValueError(
                "Connected support budget must have shape [B,T,1,1]"
            )
        if not torch.isfinite(self.weight.float()).all():
            raise ValueError("Connected support weight is not finite")
        if self.weight.min() < 0 or self.weight.max() > 1:
            raise ValueError(
                "Connected support weight must lie in [0, 1]"
            )


class CausalConnectedSupportFilter:
    """Keep one causal object component and bound support-area growth."""

    def __init__(
        self,
        min_weight: float = 0.05,
        temporal_radius: int = 3,
        max_anchor_ratio: float = 2.0,
        min_area_fraction: float = 0.02,
        max_area_fraction: float = 0.20,
    ):
        if not 0.0 < min_weight < 1.0:
            raise ValueError("min_weight must lie in (0, 1)")
        if temporal_radius < 0:
            raise ValueError("temporal_radius must be non-negative")
        if max_anchor_ratio < 1.0:
            raise ValueError("max_anchor_ratio must be at least 1")
        if not 0.0 < min_area_fraction <= max_area_fraction <= 1.0:
            raise ValueError(
                "Support area fractions must satisfy 0 < min <= max <= 1"
            )
        self.min_weight = float(min_weight)
        self.temporal_radius = int(temporal_radius)
        self.max_anchor_ratio = float(max_anchor_ratio)
        self.min_area_fraction = float(min_area_fraction)
        self.max_area_fraction = float(max_area_fraction)
        self.previous_mask: torch.Tensor | None = None

    @staticmethod
    def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask.bool()
        return (
            F.max_pool2d(
                mask.float()[None, None],
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            )[0, 0]
            > 0
        )

    def _select_seed(
        self,
        weight: torch.Tensor,
        candidate: torch.Tensor,
        anchor: torch.Tensor,
        previous: torch.Tensor | None,
    ) -> torch.Tensor:
        searches = []
        if previous is not None and previous.any():
            searches.append(
                self._dilate(previous, self.temporal_radius)
            )
        if anchor.any():
            searches.append(
                self._dilate(anchor, self.temporal_radius)
            )
        eligible = torch.zeros_like(candidate)
        for search in searches:
            eligible = candidate & search
            if eligible.any():
                break
        if not eligible.any() and searches:
            combined_search = torch.stack(searches).any(dim=0)
            eligible = candidate & self._dilate(
                combined_search,
                self.temporal_radius,
            )
        if not eligible.any():
            if searches or not candidate.any():
                return torch.zeros_like(candidate)
            eligible = candidate
        score = weight.masked_fill(~eligible, -1.0)
        seed = torch.zeros_like(candidate)
        seed.flatten()[score.flatten().argmax()] = True
        return seed

    def _connected_component(
        self,
        candidate: torch.Tensor,
        seed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_hops = candidate.shape[-2] + candidate.shape[-1]
        unreachable = max_hops + 1
        distance = torch.full(
            candidate.shape,
            unreachable,
            dtype=torch.long,
            device=candidate.device,
        )
        reached = seed & candidate
        distance = torch.where(
            reached,
            torch.zeros_like(distance),
            distance,
        )
        for hop in range(1, max_hops + 1):
            expanded = candidate & self._dilate(reached, 1)
            newly_reached = expanded & ~reached
            distance = torch.where(
                newly_reached,
                torch.full_like(distance, hop),
                distance,
            )
            reached = expanded
        return reached, distance

    def _area_budget(
        self,
        anchor: torch.Tensor,
        previous: torch.Tensor | None,
    ) -> int:
        total = anchor.numel()
        minimum = max(
            1,
            math.ceil(total * self.min_area_fraction),
        )
        maximum = max(
            minimum,
            math.floor(total * self.max_area_fraction),
        )
        anchor_count = int(anchor.sum().item())
        previous_count = (
            0 if previous is None else int(previous.sum().item())
        )
        if anchor_count > 0:
            anchor_budget = math.ceil(
                max(anchor_count, minimum) * self.max_anchor_ratio
            )
            budget = max(anchor_budget, previous_count)
        elif previous_count > 0:
            budget = max(previous_count, minimum)
        else:
            budget = minimum
        return min(maximum, budget)

    @torch.no_grad()
    def __call__(
        self,
        support_weight: torch.Tensor,
        anchor_mask: torch.Tensor,
        object_likelihood_mask: torch.Tensor | None = None,
    ) -> ConnectedIdentitySupport:
        if support_weight.ndim != 4:
            raise ValueError(
                "support_weight must have shape [B,T,H,W]"
            )
        if anchor_mask.shape != support_weight.shape:
            raise ValueError(
                "anchor_mask and support_weight must share shape"
            )
        if (
            object_likelihood_mask is not None
            and object_likelihood_mask.shape != support_weight.shape
        ):
            raise ValueError(
                "object_likelihood_mask and support_weight must share shape"
            )
        support = support_weight.detach().float().clamp(0.0, 1.0)
        likelihood = (
            torch.ones_like(support, dtype=torch.bool)
            if object_likelihood_mask is None
            else object_likelihood_mask.detach().bool()
        )
        anchor = anchor_mask.detach().bool() & likelihood
        candidate = (support > self.min_weight) & likelihood
        keep = torch.zeros_like(candidate)
        budget_fraction = support.new_zeros(
            support.shape[:2] + (1, 1)
        )
        previous = (
            None
            if self.previous_mask is None
            else self.previous_mask.to(device=support.device)
        )
        if previous is not None and previous.shape != (
            support.shape[0],
            *support.shape[-2:],
        ):
            raise ValueError(
                "Previous connected support is incompatible with input"
            )
        next_previous = torch.zeros(
            (
                support.shape[0],
                *support.shape[-2:],
            ),
            dtype=torch.bool,
            device=support.device,
        )

        for batch_index in range(support.shape[0]):
            previous_frame = (
                None
                if previous is None
                else previous[batch_index]
            )
            for frame_index in range(support.shape[1]):
                frame_candidate = candidate[
                    batch_index,
                    frame_index,
                ]
                frame_anchor = anchor[
                    batch_index,
                    frame_index,
                ]
                seed = self._select_seed(
                    support[batch_index, frame_index],
                    frame_candidate,
                    frame_anchor,
                    previous_frame,
                )
                connected, distance = self._connected_component(
                    frame_candidate,
                    seed,
                )
                budget = self._area_budget(
                    frame_anchor,
                    previous_frame,
                )
                connected_count = int(connected.sum().item())
                if connected_count > budget:
                    max_hops = (
                        frame_candidate.shape[-2]
                        + frame_candidate.shape[-1]
                    )
                    rank = (
                        (max_hops + 1 - distance).float() * 2.0
                        + support[batch_index, frame_index]
                    ).masked_fill(~connected, -1.0)
                    selected = torch.topk(
                        rank.flatten(),
                        k=budget,
                    ).indices
                    frame_keep = torch.zeros_like(
                        frame_candidate.flatten()
                    )
                    frame_keep[selected] = True
                    frame_keep = frame_keep.reshape_as(
                        frame_candidate
                    )
                else:
                    frame_keep = connected
                keep[batch_index, frame_index] = frame_keep
                budget_fraction[batch_index, frame_index] = (
                    float(budget) / frame_candidate.numel()
                )
                if frame_keep.any():
                    previous_frame = frame_keep
            if previous_frame is not None:
                next_previous[batch_index] = previous_frame

        self.previous_mask = next_previous.detach().cpu()
        result = ConnectedIdentitySupport(
            weight=(support * keep.float()).clamp(0.0, 1.0),
            candidate_mask=candidate,
            keep_mask=keep,
            anchor_mask=anchor,
            object_likelihood_mask=likelihood,
            budget_fraction=budget_fraction,
        )
        result.validate()
        return result


def build_first_frame_object_core_bootstrap(
    base_write_weight: torch.Tensor,
    object_likelihood: torch.Tensor,
    object_threshold: torch.Tensor,
    hand_probability: torch.Tensor,
    hand_exclusion_threshold: float = 0.5,
    eps: float = 1e-6,
) -> FirstFrameIdentityBootstrap:
    """Restrict a causal bootstrap to the first frame's object interior."""
    if base_write_weight.ndim != 4:
        raise ValueError(
            "base_write_weight must have shape [B,T,H,W]"
        )
    expected_shape = base_write_weight.shape
    for name, value in (
        ("object_likelihood", object_likelihood),
        ("hand_probability", hand_probability),
    ):
        if value.shape != expected_shape:
            raise ValueError(
                f"{name} must match base_write_weight"
            )
    if object_threshold.shape not in {
        expected_shape,
        expected_shape[:2] + (1, 1),
    }:
        raise ValueError(
            "object_threshold must have shape [B,T,H,W] or [B,T,1,1]"
        )
    if not 0.0 <= hand_exclusion_threshold <= 1.0:
        raise ValueError(
            "hand_exclusion_threshold must lie in [0, 1]"
        )
    if eps <= 0:
        raise ValueError("eps must be positive")

    base = base_write_weight.detach().float().clamp(0.0, 1.0)
    likelihood = object_likelihood.detach().float().clamp(0.0, 1.0)
    threshold = object_threshold.detach().float().clamp(0.0, 1.0)
    hand = hand_probability.detach().float().clamp(0.0, 1.0)
    core = (
        (likelihood >= threshold)
        & (likelihood > eps)
        & (hand < hand_exclusion_threshold)
    )
    hand_contact = F.max_pool2d(
        hand.reshape(
            math.prod(expected_shape[:2]),
            1,
            *expected_shape[-2:],
        ),
        kernel_size=3,
        stride=1,
        padding=1,
    ).reshape_as(hand)
    component_mask = _largest_weighted_component_mask(
        likelihood * core.float(),
        eps=eps,
        hand_contact_score=hand_contact,
    )
    core = core & component_mask
    first_frame_core = torch.zeros_like(core)
    first_frame_core[:, 0] = core[:, 0]
    write_map = (
        base
        * likelihood
        * (1.0 - hand)
        * first_frame_core.float()
    ).clamp(0.0, 1.0)

    result = FirstFrameIdentityBootstrap(
        write_weight=write_map.flatten(1),
        base_write_weight=base,
        object_likelihood=likelihood,
        core_mask=first_frame_core,
    )
    result.validate()
    return result


class CausalObjectTokenPropagator:
    """Gate identity writes by matching current source tokens to prior ones."""

    def __init__(
        self,
        min_similarity: float = 0.55,
        gate_strength: float = 0.85,
        max_candidates: int = 512,
        eps: float = 1e-6,
    ):
        if not -1.0 < min_similarity < 1.0:
            raise ValueError("min_similarity must lie in (-1, 1)")
        if not 0.0 <= gate_strength <= 1.0:
            raise ValueError("gate_strength must lie in [0, 1]")
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.min_similarity = float(min_similarity)
        self.gate_strength = float(gate_strength)
        self.max_candidates = int(max_candidates)
        self.eps = float(eps)
        self.previous_features: torch.Tensor | None = None
        self.previous_weight: torch.Tensor | None = None

    def _candidate_indices(
        self,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        candidates = torch.nonzero(
            weight > self.eps,
            as_tuple=False,
        ).flatten()
        if candidates.numel() > self.max_candidates:
            selected = torch.topk(
                weight[candidates],
                k=self.max_candidates,
            ).indices
            candidates = candidates[selected]
        return candidates

    @torch.no_grad()
    def __call__(
        self,
        source_features: torch.Tensor,
        base_write_weight: torch.Tensor,
        support_weight: torch.Tensor | None = None,
    ) -> TargetIdentityTokenPropagation:
        if source_features.ndim != 3:
            raise ValueError(
                "Token propagation source_features must have shape [B,L,D]"
            )
        if base_write_weight.ndim != 2:
            raise ValueError(
                "Token propagation write weights must have shape [B,L]"
            )
        if source_features.shape[:2] != base_write_weight.shape:
            raise ValueError(
                "Token propagation features and weights must align"
            )

        current_features = F.normalize(
            source_features.detach().float(),
            dim=-1,
        )
        base_weight = base_write_weight.detach().float().clamp(0.0, 1.0)
        support = (
            base_weight
            if support_weight is None
            else support_weight.detach().float().clamp(0.0, 1.0)
        )
        if support.shape != base_weight.shape:
            raise ValueError(
                "Token propagation support_weight must match [B,L]"
            )
        write_weight = base_weight.clone()
        match_confidence = torch.ones_like(base_weight)
        best_similarity = torch.zeros_like(base_weight)
        matched_previous_weight = torch.zeros_like(base_weight)
        has_previous = torch.zeros(
            base_weight.shape[0],
            1,
            dtype=torch.bool,
            device=base_weight.device,
        )

        if (
            self.previous_features is not None
            and self.previous_weight is not None
        ):
            previous_features = self.previous_features.to(
                device=current_features.device,
            )
            previous_weight = self.previous_weight.to(
                device=base_weight.device,
            )
            if (
                previous_features.shape[0] != current_features.shape[0]
                or previous_features.shape[-1] != current_features.shape[-1]
                or previous_weight.shape[0] != base_weight.shape[0]
                or previous_features.shape[:2] != previous_weight.shape
            ):
                raise ValueError(
                    "Previous token propagation state is incompatible with "
                    "the current features"
                )
            for batch_index in range(base_weight.shape[0]):
                candidates = self._candidate_indices(
                    previous_weight[batch_index],
                )
                if candidates.numel() == 0:
                    continue
                has_previous[batch_index, 0] = True
                candidate_features = previous_features[
                    batch_index,
                    candidates,
                ]
                similarity = torch.matmul(
                    current_features[batch_index],
                    candidate_features.T,
                ).clamp(-1.0, 1.0)
                best, matched = similarity.max(dim=-1)
                matched_weight = previous_weight[
                    batch_index,
                    candidates[matched],
                ].clamp(0.0, 1.0)
                absolute_match = (
                    (best - self.min_similarity)
                    / (1.0 - self.min_similarity)
                ).clamp(0.0, 1.0)
                low = torch.quantile(best, 0.50)
                high = torch.quantile(best, 0.95)
                spread = (high - low).clamp_min(self.eps)
                relative_match = ((best - low) / spread).clamp(0.0, 1.0)
                relative_match = torch.where(
                    (high - low) > self.eps,
                    relative_match,
                    absolute_match,
                )
                confidence = torch.sqrt(
                    absolute_match * relative_match
                ).clamp(0.0, 1.0)
                propagated_weight = (
                    confidence * matched_weight
                ).clamp(0.0, 1.0)
                multiplier = (
                    1.0
                    - self.gate_strength
                    + self.gate_strength * propagated_weight
                )
                write_weight[batch_index] = (
                    base_weight[batch_index] * multiplier
                ).clamp(0.0, 1.0)
                match_confidence[batch_index] = confidence
                best_similarity[batch_index] = best
                matched_previous_weight[batch_index] = matched_weight

        self.previous_features = current_features.detach().cpu()
        self.previous_weight = support.detach().cpu()

        result = TargetIdentityTokenPropagation(
            write_weight=write_weight,
            base_write_weight=base_weight,
            support_weight=support,
            match_confidence=match_confidence,
            best_similarity=best_similarity,
            matched_previous_weight=matched_previous_weight,
            has_previous=has_previous,
        )
        result.validate()
        return result


@dataclass(frozen=True)
class ReferenceIdentityBootstrap:
    """Object evidence extracted from an aligned target reference."""

    write_weight: torch.Tensor
    change_score: torch.Tensor
    semantic_score: torch.Tensor
    joint_score: torch.Tensor
    hand_contact_score: torch.Tensor

    def validate(self) -> None:
        if self.write_weight.ndim != 2:
            raise ValueError(
                "Reference identity weights must have shape [B,L]"
            )
        if self.change_score.ndim != 4:
            raise ValueError(
                "Reference identity maps must have shape [B,T,H,W]"
            )
        shapes = {
            tuple(value.shape)
            for value in (
                self.change_score,
                self.semantic_score,
                self.joint_score,
                self.hand_contact_score,
            )
        }
        if len(shapes) != 1:
            raise ValueError(
                "Reference identity debug maps must share shape"
            )
        if self.write_weight.shape != (
            self.change_score.shape[0],
            math.prod(self.change_score.shape[1:]),
        ):
            raise ValueError(
                "Reference identity weights and maps must align"
            )
        for name, value in (
            ("write_weight", self.write_weight),
            ("change_score", self.change_score),
            ("semantic_score", self.semantic_score),
            ("joint_score", self.joint_score),
            ("hand_contact_score", self.hand_contact_score),
        ):
            if not torch.isfinite(value.float()).all():
                raise ValueError(
                    f"Reference identity '{name}' is not finite"
                )
            if value.min() < 0 or value.max() > 1:
                raise ValueError(
                    f"Reference identity '{name}' must lie in [0, 1]"
                )

    def as_debug_maps(self) -> Dict[str, torch.Tensor]:
        return {
            "reference_identity_change": self.change_score.float(),
            "reference_identity_semantic": self.semantic_score.float(),
            "reference_identity_joint": self.joint_score.float(),
            "reference_identity_hand_contact": (
                self.hand_contact_score.float()
            ),
            "reference_identity_write": self.write_weight.reshape_as(
                self.joint_score
            ).float(),
        }


def _largest_weighted_component_mask(
    weight: torch.Tensor,
    eps: float,
    hand_contact_score: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select one 8-connected edited instance per reference frame."""
    if weight.ndim != 4:
        raise ValueError(
            "Reference component weights must have shape [B,T,H,W]"
        )
    if (
        hand_contact_score is not None
        and hand_contact_score.shape != weight.shape
    ):
        raise ValueError(
            "Reference hand contact and component weights must align"
        )
    candidate = (weight.detach().float() > eps).cpu()
    cpu_weight = weight.detach().float().cpu()
    cpu_contact = (
        hand_contact_score.detach().float().cpu()
        if hand_contact_score is not None
        else torch.zeros_like(cpu_weight)
    )
    selected = torch.zeros_like(candidate)
    batch, frames, height, width = candidate.shape
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    for batch_index in range(batch):
        for frame_index in range(frames):
            visited = torch.zeros(
                (height, width),
                dtype=torch.bool,
            )
            best_component = []
            best_score = (-1.0, -1.0, -1)
            for row in range(height):
                for col in range(width):
                    if (
                        visited[row, col]
                        or not candidate[
                            batch_index,
                            frame_index,
                            row,
                            col,
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
                        component.append(
                            (current_row, current_col)
                        )
                        mass += float(
                            cpu_weight[
                                batch_index,
                                frame_index,
                                current_row,
                                current_col,
                            ]
                        )
                        contact_mass += float(
                            cpu_weight[
                                batch_index,
                                frame_index,
                                current_row,
                                current_col,
                            ]
                            * cpu_contact[
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
                                and candidate[
                                    batch_index,
                                    frame_index,
                                    next_row,
                                    next_col,
                                ]
                            ):
                                visited[next_row, next_col] = True
                                stack.append((next_row, next_col))
                    score = (
                        contact_mass,
                        mass,
                        len(component),
                    )
                    if score > best_score:
                        best_score = score
                        best_component = component
            for row, col in best_component:
                selected[
                    batch_index,
                    frame_index,
                    row,
                    col,
                ] = True
    return selected.to(device=weight.device)


def build_reference_identity_bootstrap(
    source_latent: torch.Tensor,
    target_latent: torch.Tensor,
    target_attention: torch.Tensor,
    hand_mask: torch.Tensor | None = None,
    patch_size: int = 2,
    eps: float = 1e-6,
) -> ReferenceIdentityBootstrap:
    """Locate customized content from an aligned source/reference pair."""
    if source_latent.shape != target_latent.shape:
        raise ValueError(
            "Source and target reference latents must share shape"
        )
    if source_latent.ndim != 5:
        raise ValueError(
            "Reference latents must have shape [B,T,C,H,W]"
        )
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")
    batch, frames, _, height, width = source_latent.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            "Reference latent size must be divisible by patch_size"
        )
    token_height = height // patch_size
    token_width = width // patch_size
    token_count = frames * token_height * token_width
    if target_attention.shape != (batch, token_count):
        raise ValueError(
            "Target attention and reference token grid must align"
        )
    if hand_mask is not None and hand_mask.shape != (
        batch,
        frames,
        height,
        width,
    ):
        raise ValueError(
            "Reference hand mask must match latent spatial shape"
        )

    latent_change = (
        target_latent.float() - source_latent.float()
    ).abs().mean(dim=2)
    change_score = F.avg_pool2d(
        latent_change.reshape(
            batch * frames,
            1,
            height,
            width,
        ),
        kernel_size=patch_size,
        stride=patch_size,
    ).reshape(batch, frames, token_height, token_width)

    def robust_unit(value: torch.Tensor) -> torch.Tensor:
        flat = value.flatten(2)
        median = torch.quantile(
            flat,
            0.50,
            dim=-1,
            keepdim=True,
        )
        residual = (flat - median).clamp_min(0.0)
        high = torch.quantile(
            residual,
            0.95,
            dim=-1,
            keepdim=True,
        )
        normalized = residual / high.clamp_min(eps)
        normalized = torch.where(
            high > eps,
            normalized,
            torch.zeros_like(normalized),
        )
        return normalized.clamp(0.0, 1.0).reshape_as(value)

    change_score = robust_unit(change_score)
    raw_semantic = target_attention.float().clamp_min(0.0).reshape(
        batch,
        frames,
        token_height,
        token_width,
    )
    semantic_score = robust_unit(raw_semantic)
    semantic_available = (
        semantic_score.flatten(2).amax(dim=-1, keepdim=True) > eps
    ).reshape(batch, frames, 1, 1)
    joint_score = torch.where(
        semantic_available,
        torch.sqrt(change_score * semantic_score),
        change_score,
    )

    hand_contact_score = torch.zeros_like(joint_score)
    if hand_mask is not None:
        hand_core = F.avg_pool2d(
            hand_mask.float().reshape(
                batch * frames,
                1,
                height,
                width,
            ),
            kernel_size=patch_size,
            stride=patch_size,
        ).reshape_as(joint_score).clamp(0.0, 1.0)
        contact_radius = max(
            1,
            round(min(token_height, token_width) * 0.067),
        )
        hand_contact_score = F.max_pool2d(
            hand_core.reshape(
                batch * frames,
                1,
                token_height,
                token_width,
            ),
            kernel_size=2 * contact_radius + 1,
            stride=1,
            padding=contact_radius,
        ).reshape_as(joint_score)
        joint_score = joint_score * (1.0 - hand_core)

    flat_joint = joint_score.flatten(2)
    center = torch.quantile(
        flat_joint,
        0.50,
        dim=-1,
        keepdim=True,
    )
    deviation = (flat_joint - center).abs()
    robust_scale = (
        torch.quantile(
            deviation,
            0.50,
            dim=-1,
            keepdim=True,
        )
        / 0.6745
    )
    threshold = center + robust_scale
    high = torch.quantile(
        flat_joint,
        0.95,
        dim=-1,
        keepdim=True,
    )
    write_weight = (
        (flat_joint - threshold)
        / (high - threshold).clamp_min(eps)
    ).clamp(0.0, 1.0)
    write_weight = torch.where(
        high > threshold + eps,
        write_weight,
        torch.zeros_like(write_weight),
    ).reshape(batch, frames, token_height, token_width)
    component_mask = _largest_weighted_component_mask(
        write_weight,
        eps=eps,
        hand_contact_score=hand_contact_score,
    )
    write_weight = (
        write_weight * component_mask.float()
    ).reshape(batch, -1)

    result = ReferenceIdentityBootstrap(
        write_weight=write_weight.float(),
        change_score=change_score.float(),
        semantic_score=semantic_score.float(),
        joint_score=joint_score.float(),
        hand_contact_score=hand_contact_score.float(),
    )
    result.validate()
    return result


class SlowTargetIdentityMemory:
    """Factor source correspondence from immutable target appearance."""

    def __init__(
        self,
        layers: Iterable[int] = (8, 12, 16, 20),
        num_prototypes: int = 4,
        reference_prior_evidence: float = 8.0,
        eps: float = 1e-6,
    ):
        self.layers = tuple(layers)
        if not self.layers:
            raise ValueError("Identity layers must not be empty")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("Identity layers must be unique")
        if any(layer < 0 for layer in self.layers):
            raise ValueError(
                "Identity layer indices must be non-negative"
            )
        if num_prototypes <= 0:
            raise ValueError(
                "num_prototypes must be positive"
            )
        if reference_prior_evidence <= 0:
            raise ValueError(
                "reference_prior_evidence must be positive"
            )
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.num_prototypes = num_prototypes
        self.reference_prior_evidence = (
            float(reference_prior_evidence)
        )
        self.eps = eps
        self.anchor_states: Dict[int, TargetIdentityLayerState] = {}
        self.states: Dict[int, TargetIdentityLayerState] = {}
        self.reference_bootstrapped = False
        self.causal_first_frame_bootstrapped = False

    @staticmethod
    def _descriptor(key: torch.Tensor) -> torch.Tensor:
        return F.normalize(
            key.float().flatten(2),
            dim=-1,
        )

    def _bootstrap_centers(
        self,
        descriptor: torch.Tensor,
        weight: torch.Tensor,
    ):
        batch, _, dim = descriptor.shape
        centers = descriptor.new_zeros(
            batch,
            self.num_prototypes,
            dim,
        )
        valid = torch.zeros(
            batch,
            self.num_prototypes,
            dtype=torch.bool,
            device=descriptor.device,
        )
        for batch_index in range(batch):
            candidates = torch.nonzero(
                weight[batch_index] > self.eps,
                as_tuple=False,
            ).flatten()
            if candidates.numel() == 0:
                continue
            selected = []
            first = candidates[
                weight[batch_index, candidates].argmax()
            ]
            selected.append(first)
            for slot in range(self.num_prototypes):
                if slot > 0:
                    if len(selected) >= candidates.numel():
                        break
                    selected_features = descriptor[
                        batch_index,
                        torch.stack(selected),
                    ]
                    similarity = torch.einsum(
                        "nd,pd->np",
                        descriptor[batch_index, candidates],
                        selected_features,
                    )
                    diversity = (
                        1.0 - similarity.max(dim=-1).values
                    ).clamp_min(0.0)
                    score = (
                        weight[batch_index, candidates] * diversity
                    )
                    used = torch.zeros_like(
                        candidates,
                        dtype=torch.bool,
                    )
                    for selected_index in selected:
                        used |= candidates == selected_index
                    score = score.masked_fill(used, -1.0)
                    selected.append(candidates[score.argmax()])
                centers[batch_index, slot] = descriptor[
                    batch_index,
                    selected[-1],
                ]
                valid[batch_index, slot] = True
        return centers, valid

    def _assign(
        self,
        descriptor: torch.Tensor,
        centers: torch.Tensor,
        center_valid: torch.Tensor,
    ) -> torch.Tensor:
        similarity = torch.einsum(
            "bld,bpd->blp",
            descriptor,
            F.normalize(centers.float(), dim=-1),
        ).clamp(-1.0, 1.0)
        flat_similarity = similarity.flatten(1)
        median = torch.quantile(
            flat_similarity,
            0.50,
            dim=-1,
            keepdim=True,
        )
        absolute_deviation = (
            flat_similarity - median
        ).abs()
        robust_scale = (
            torch.quantile(
                absolute_deviation,
                0.50,
                dim=-1,
                keepdim=True,
            )
            / 0.6745
        ).clamp_min(self.eps)
        logits = similarity / robust_scale.unsqueeze(-1)
        logits = logits.masked_fill(
            ~center_valid[:, None, :],
            torch.finfo(logits.dtype).min,
        )
        assignment = torch.softmax(logits, dim=-1)
        has_center = center_valid.any(dim=-1, keepdim=True)
        return torch.where(
            has_center.unsqueeze(-1),
            assignment,
            torch.zeros_like(assignment),
        )

    def _observe(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        weight: torch.Tensor,
        state: TargetIdentityLayerState | None,
    ):
        descriptor = self._descriptor(key)
        if state is None:
            centers, center_valid = self._bootstrap_centers(
                descriptor,
                weight,
            )
        else:
            centers = self._descriptor(state.key)
            center_valid = state.evidence > self.eps
            missing = ~center_valid.any(dim=-1)
            if missing.any():
                bootstrap_centers, bootstrap_valid = (
                    self._bootstrap_centers(
                        descriptor,
                        weight,
                    )
                )
                centers = torch.where(
                    missing[:, None, None],
                    bootstrap_centers,
                    centers,
                )
                center_valid = torch.where(
                    missing[:, None],
                    bootstrap_valid,
                    center_valid,
                )
        assignment = self._assign(
            descriptor,
            centers,
            center_valid,
        )
        responsibility = assignment * weight.unsqueeze(-1)
        mass = responsibility.sum(dim=1)
        normalizer = mass.clamp_min(self.eps)
        observation_key = torch.einsum(
            "blp,blhd->bphd",
            responsibility,
            key.float(),
        ) / normalizer[:, :, None, None]
        observation_value = torch.einsum(
            "blp,blhd->bphd",
            responsibility,
            value.float(),
        ) / normalizer[:, :, None, None]

        support = (weight > self.eps).float()
        assignment_support = (
            assignment * support.unsqueeze(-1)
        ).sum(dim=1)
        observation_evidence = (
            mass / assignment_support.clamp_min(self.eps)
        )
        observation_evidence = torch.where(
            mass > self.eps,
            observation_evidence,
            torch.zeros_like(observation_evidence),
        ).clamp(0.0, 1.0)
        observation_key = F.normalize(
            observation_key,
            dim=-1,
        ) * math.sqrt(key.shape[-1])
        return (
            observation_key,
            observation_value,
            observation_evidence,
        )

    @staticmethod
    def _current_tokens(
        cache,
        num_new_tokens: int,
        tensor_name: str,
    ) -> torch.Tensor:
        if tensor_name == "k":
            captured_key = cache.get("current_identity_key")
            if captured_key is not None:
                if captured_key.shape[1] != num_new_tokens:
                    raise ValueError(
                        "Captured source identity keys and target KV must "
                        "contain the same number of current tokens"
                    )
                return captured_key
        local_end = cache["local_end_index"].item()
        return cache[tensor_name][
            :,
            local_end - num_new_tokens:local_end,
        ]

    @torch.no_grad()
    def _update_store(
        self,
        kv_cache,
        write_weight: torch.Tensor,
        state_store: Dict[int, TargetIdentityLayerState],
        batch_slice: slice | None = None,
        source_kv_cache=None,
    ) -> TargetIdentityUpdate:
        if write_weight.ndim != 2:
            raise ValueError(
                "Identity write_weight must have shape [B,L]"
            )
        write_weight = write_weight.detach().float().clamp(
            0.0,
            1.0,
        )
        observation_evidence = []
        update_gains = []
        accumulated_evidence = []
        for layer in self.layers:
            target_cache = kv_cache[layer]
            source_cache = (
                target_cache
                if source_kv_cache is None
                else source_kv_cache[layer]
            )
            num_new_tokens = target_cache.get("num_new_tokens")
            if num_new_tokens != write_weight.shape[1]:
                raise ValueError(
                    "Identity write weights and target KV must align: "
                    f"{write_weight.shape[1]} != {num_new_tokens}"
                )
            key = self._current_tokens(
                source_cache,
                num_new_tokens,
                "k",
            )
            value = self._current_tokens(
                target_cache,
                num_new_tokens,
                "v",
            )
            if batch_slice is not None:
                value = value[batch_slice]
                if key.shape[0] != write_weight.shape[0]:
                    key = key[batch_slice]
            if key.shape[0] != write_weight.shape[0]:
                raise ValueError(
                    "Source identity keys and write weights must align: "
                    f"{key.shape[0]} != {write_weight.shape[0]}"
                )
            if value.shape[0] != write_weight.shape[0]:
                raise ValueError(
                    "Target identity values and write weights must align: "
                    f"{value.shape[0]} != {write_weight.shape[0]}"
                )
            previous = state_store.get(layer)
            (
                observed_key,
                observed_value,
                observed_evidence,
            ) = self._observe(
                key,
                value,
                write_weight,
                previous,
            )
            if previous is None:
                old_evidence = torch.zeros_like(observed_evidence)
                old_key = torch.zeros_like(observed_key)
                old_value = torch.zeros_like(observed_value)
            else:
                old_evidence = previous.evidence.float()
                old_key = previous.key.float()
                old_value = previous.value.float()
            total_evidence = old_evidence + observed_evidence
            gain = torch.where(
                total_evidence > self.eps,
                observed_evidence
                / total_evidence.clamp_min(self.eps),
                torch.zeros_like(total_evidence),
            )
            new_key = (
                old_key
                + gain[:, :, None, None]
                * (observed_key - old_key)
            )
            new_key = F.normalize(
                new_key,
                dim=-1,
            ) * math.sqrt(key.shape[-1])
            new_value = (
                old_value
                + gain[:, :, None, None]
                * (observed_value - old_value)
            )
            valid = total_evidence > self.eps
            new_key = torch.where(
                valid[:, :, None, None],
                new_key,
                torch.zeros_like(new_key),
            )
            new_value = torch.where(
                valid[:, :, None, None],
                new_value,
                torch.zeros_like(new_value),
            )
            state = TargetIdentityLayerState(
                key=new_key.to(key.dtype).detach(),
                value=new_value.to(value.dtype).detach(),
                evidence=total_evidence.detach(),
            )
            state.validate()
            state_store[layer] = state
            observation_evidence.append(observed_evidence)
            update_gains.append(gain)
            accumulated_evidence.append(total_evidence)

        update = TargetIdentityUpdate(
            write_weight=write_weight,
            observation_evidence=torch.stack(
                observation_evidence,
                dim=0,
            ),
            update_gain=torch.stack(update_gains, dim=0),
            accumulated_evidence=torch.stack(
                accumulated_evidence,
                dim=0,
            ),
        )
        update.validate()
        return update

    @torch.no_grad()
    def update(
        self,
        kv_cache,
        write_weight: torch.Tensor,
        batch_slice: slice | None = None,
        source_kv_cache=None,
    ) -> TargetIdentityUpdate:
        """Update the adaptive bank without changing an immutable anchor."""
        return self._update_store(
            kv_cache=kv_cache,
            write_weight=write_weight,
            state_store=self.states,
            batch_slice=batch_slice,
            source_kv_cache=source_kv_cache,
        )

    def _make_authoritative(
        self,
        states: Dict[int, TargetIdentityLayerState],
    ) -> None:
        for layer, state in tuple(states.items()):
            authoritative_evidence = torch.where(
                state.evidence > self.eps,
                torch.full_like(
                    state.evidence,
                    self.reference_prior_evidence,
                ),
                torch.zeros_like(state.evidence),
            )
            anchored_state = TargetIdentityLayerState(
                key=state.key,
                value=state.value,
                evidence=authoritative_evidence,
            )
            anchored_state.validate()
            states[layer] = anchored_state

    @torch.no_grad()
    def bootstrap_causal_first_frame(
        self,
        kv_cache,
        write_weight: torch.Tensor,
        num_frames: int,
        target_batch_start: int,
        source_kv_cache=None,
    ) -> TargetIdentityUpdate:
        """Freeze source-key/target-value identity from frame zero."""
        if self.states or self.anchor_states:
            raise RuntimeError(
                "Causal first-frame bootstrap requires empty identity state"
            )
        if self.causal_first_frame_bootstrapped:
            raise RuntimeError(
                "Causal first-frame identity was already bootstrapped"
            )
        if num_frames <= 1:
            raise ValueError(
                "Causal first-frame bootstrap requires multiple frames"
            )
        if write_weight.ndim != 2:
            raise ValueError(
                "Identity write_weight must have shape [B,L]"
            )
        if write_weight.shape[1] % num_frames != 0:
            raise ValueError(
                "Identity token count must be divisible by num_frames"
            )
        if target_batch_start < 0:
            raise ValueError(
                "target_batch_start must be non-negative"
            )

        first_frame_weight = torch.zeros_like(write_weight)
        tokens_per_frame = write_weight.shape[1] // num_frames
        first_frame_weight[:, :tokens_per_frame] = write_weight[
            :,
            :tokens_per_frame,
        ]
        batch_end = target_batch_start + write_weight.shape[0]
        update = self._update_store(
            kv_cache=kv_cache,
            write_weight=first_frame_weight,
            state_store=self.anchor_states,
            batch_slice=slice(target_batch_start, batch_end),
            source_kv_cache=source_kv_cache,
        )
        self._make_authoritative(self.anchor_states)
        self.causal_first_frame_bootstrapped = True
        anchored_update = TargetIdentityUpdate(
            write_weight=update.write_weight,
            observation_evidence=update.observation_evidence,
            update_gain=update.update_gain,
            accumulated_evidence=torch.stack(
                [
                    self.anchor_states[layer].evidence
                    for layer in self.layers
                ],
                dim=0,
            ),
        )
        anchored_update.validate()
        return anchored_update

    @torch.no_grad()
    def bootstrap_reference(
        self,
        kv_cache,
        write_weight: torch.Tensor,
        source_kv_cache=None,
    ) -> TargetIdentityUpdate:
        if self.reference_bootstrapped:
            raise RuntimeError(
                "Target identity reference was already bootstrapped"
            )
        if self.states or self.anchor_states:
            raise RuntimeError(
                "Target identity reference must be bootstrapped before "
                "online identity updates"
            )
        update = self._update_store(
            kv_cache=kv_cache,
            write_weight=write_weight,
            state_store=self.anchor_states,
            source_kv_cache=source_kv_cache,
        )
        self._make_authoritative(self.anchor_states)
        self.reference_bootstrapped = True
        anchored_update = TargetIdentityUpdate(
            write_weight=update.write_weight,
            observation_evidence=update.observation_evidence,
            update_gain=update.update_gain,
            accumulated_evidence=torch.stack(
                [
                    self.anchor_states[layer].evidence
                    for layer in self.layers
                ],
                dim=0,
            ),
        )
        anchored_update.validate()
        return anchored_update

    def export(
        self,
    ) -> Mapping[int, TargetIdentityLayerState]:
        if self.anchor_states:
            return self.anchor_states
        return self.states

    def export_adaptive(
        self,
    ) -> Mapping[int, TargetIdentityLayerState]:
        return self.states


def strengthen_belief_with_target_identity(
    belief: CausalControlBelief,
    identity_support: torch.Tensor,
    hand_mask: torch.Tensor,
    eps: float = 1e-6,
) -> CausalControlBelief:
    """Use reliable target identity as edit evidence, not as a new seed."""
    belief.validate()
    if identity_support.ndim != 4:
        raise ValueError(
            "Identity support must have shape [B,T,H,W]"
        )
    if hand_mask.shape != belief.edit_belief.shape:
        raise ValueError(
            "Hand mask and control belief must share shape"
        )
    if identity_support.shape[:2] != belief.edit_belief.shape[:2]:
        raise ValueError(
            "Identity support and belief must share [B,T]"
        )
    batch, frames = identity_support.shape[:2]
    target_size = belief.edit_belief.shape[-2:]
    support = F.interpolate(
        identity_support.float().reshape(
            batch * frames,
            1,
            *identity_support.shape[-2:],
        ),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).reshape(batch, frames, *target_size).clamp(0.0, 1.0)

    edit_belief = (
        1.0
        - (1.0 - belief.edit_belief)
        * (1.0 - support)
    ).clamp(0.0, 1.0)
    edit_precision = torch.maximum(
        belief.edit_precision,
        support,
    )
    preserve_release = support * (
        1.0 - hand_mask.float().clamp(0.0, 1.0)
    )
    preserve_belief = (
        belief.preserve_belief * (1.0 - preserve_release)
    ).clamp(0.0, 1.0)
    visibility = (
        1.0
        - (1.0 - belief.visibility) * (1.0 - support)
    ).clamp(0.0, 1.0)
    conflict = (edit_belief * preserve_belief).clamp(0.0, 1.0)
    responsibility = (
        edit_belief + preserve_belief
    ).clamp_min(eps)
    uncertainty = (
        edit_belief * (1.0 - edit_precision)
        + preserve_belief * (1.0 - belief.preserve_precision)
    ) / responsibility
    updated = CausalControlBelief(
        edit_belief=edit_belief.float(),
        preserve_belief=preserve_belief.float(),
        edit_precision=edit_precision.float(),
        preserve_precision=belief.preserve_precision.float(),
        visibility=visibility.float(),
        uncertainty=uncertainty.clamp(0.0, 1.0).float(),
        conflict=conflict.float(),
    )
    updated.validate()
    return updated


def _expand_token_map_to_belief(
    token_map: torch.Tensor,
    belief: CausalControlBelief,
) -> torch.Tensor:
    if token_map.ndim != 4:
        raise ValueError("token_map must have shape [B,T,H,W]")
    batch, frames, height, width = belief.edit_belief.shape
    if token_map.shape[:2] != (batch, frames):
        raise ValueError("token_map and belief must share [B,T]")
    return F.interpolate(
        token_map.float().reshape(
            batch * frames,
            1,
            *token_map.shape[-2:],
        ),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).reshape(batch, frames, height, width).clamp(0.0, 1.0)


def inject_committed_memory_into_belief(
    belief: CausalControlBelief,
    committed_token_edit: torch.Tensor,
    committed_token_precision: torch.Tensor,
    hand_mask: torch.Tensor,
    feedback_strength: float,
    identity_core_support: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[CausalControlBelief, Dict[str, torch.Tensor]]:
    """Materialize transported edit memory into the current action belief."""
    if feedback_strength <= 0:
        return belief, {}
    if committed_token_edit.shape != committed_token_precision.shape:
        raise ValueError(
            "Committed edit and precision token maps must share shape"
        )
    if hand_mask.shape != belief.edit_belief.shape:
        raise ValueError("hand_mask and belief must share shape")
    if (
        identity_core_support is not None
        and identity_core_support.shape != committed_token_edit.shape
    ):
        raise ValueError(
            "identity_core_support and committed edit must share shape"
        )

    committed_edit_full = _expand_token_map_to_belief(
        committed_token_edit,
        belief,
    )
    committed_precision_full = _expand_token_map_to_belief(
        committed_token_precision,
        belief,
    )
    object_space = (1.0 - hand_mask.float().clamp(0.0, 1.0))
    committed_evidence = (
        committed_edit_full
        * committed_precision_full.clamp(0.0, 1.0)
        * object_space
    ).clamp(0.0, 1.0)
    committed_evidence = (
        committed_evidence * float(feedback_strength)
    ).clamp(0.0, 1.0)
    identity_core_full = (
        torch.zeros_like(committed_edit_full)
        if identity_core_support is None
        else _expand_token_map_to_belief(
            identity_core_support,
            belief,
        )
    )
    identity_core_full = (
        identity_core_full * object_space
    ).clamp(0.0, 1.0)
    identity_core_evidence = (
        identity_core_full * float(feedback_strength)
    ).clamp(0.0, 1.0)
    feedback_evidence = torch.maximum(
        committed_evidence,
        identity_core_evidence,
    )
    feedback_precision = torch.maximum(
        committed_precision_full,
        identity_core_full,
    )

    old_edit_belief = belief.edit_belief.float()
    old_edit_precision = belief.edit_precision.float()
    edit_belief = (
        1.0
        - (1.0 - old_edit_belief)
        * (1.0 - feedback_evidence)
    ).clamp(0.0, 1.0)
    edit_precision = torch.where(
        edit_belief > eps,
        (
            old_edit_belief * old_edit_precision
            + feedback_evidence * feedback_precision
        )
        / (old_edit_belief + feedback_evidence).clamp_min(eps),
        old_edit_precision,
    ).clamp(0.0, 1.0)

    preserve_release = feedback_evidence.clamp(0.0, 0.90)
    preserve_belief = (
        belief.preserve_belief.float()
        * (1.0 - preserve_release)
    ).clamp(0.0, 1.0)
    visibility = torch.maximum(
        belief.visibility.float(),
        committed_evidence,
    )
    conflict = (edit_belief * preserve_belief).clamp(0.0, 1.0)
    responsibility = (edit_belief + preserve_belief).clamp_min(eps)
    updated_uncertainty = (
        edit_belief * (1.0 - edit_precision)
        + preserve_belief
        * (1.0 - belief.preserve_precision.float())
    ) / responsibility
    uncertainty = torch.where(
        feedback_evidence > eps,
        updated_uncertainty,
        belief.uncertainty.float(),
    )
    injected = CausalControlBelief(
        edit_belief=edit_belief.float(),
        preserve_belief=preserve_belief.float(),
        edit_precision=edit_precision.float(),
        preserve_precision=belief.preserve_precision.float(),
        visibility=visibility.float(),
        uncertainty=uncertainty.clamp(0.0, 1.0).float(),
        conflict=conflict.float(),
    )
    injected.validate()
    return injected, {
        "committed_memory_edit": committed_edit_full.float(),
        "committed_memory_precision": committed_precision_full.float(),
        "committed_memory_evidence": committed_evidence.float(),
        "committed_memory_identity_core": identity_core_full.float(),
        "committed_memory_feedback_evidence": feedback_evidence.float(),
        "committed_memory_preserve_release": preserve_release.float(),
    }
