from dataclasses import dataclass
import math
from typing import Dict

import torch
import torch.nn.functional as F

from .adaptive_role_calibrator import AdaptiveRoleCalibrator
from .motion.flow_role_evidence import FlowRoleEvidence
from .role_router import RoleState
from .source_flow_verified_region import (
    build_source_flow_verified_region,
)


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    batch, frames, height, width = mask.shape
    flat = mask.reshape(batch * frames, 1, height, width).float()
    value = F.max_pool2d(
        flat,
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )
    return value.reshape(batch, frames, height, width)


def _neighbor_max(value: torch.Tensor) -> torch.Tensor:
    batch, frames, height, width = value.shape
    flat = value.reshape(batch * frames, 1, height, width)
    propagated = F.max_pool2d(flat, kernel_size=3, stride=1, padding=1)
    return propagated.reshape(batch, frames, height, width)


def _connected_hysteresis_growth(
    seed: torch.Tensor,
    candidate: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    """Grow seed support through a low-threshold candidate corridor.

    Unlike unconstrained dilation, every admitted token must have an
    8-connected path of at most ``steps`` cells back to a high-confidence
    seed. This recovers low-attention object interiors and thin structure
    without letting an unrelated semantic response ignite on its own.
    """
    if seed.shape != candidate.shape or seed.ndim != 4:
        raise ValueError(
            "seed and candidate must share shape [B,T,H,W]"
        )
    if steps < 0:
        raise ValueError("steps must be non-negative")
    # A seed is already the high-confidence decision.  The lower-threshold
    # candidate mask is only a corridor through which that decision may grow;
    # it must never be allowed to erase the seed itself.
    support = seed.bool()
    for _ in range(steps):
        support = support | (
            _neighbor_max(support.float()).bool() & candidate.bool()
        )
    return support


def _quantile_normalize(
    value: torch.Tensor,
    low: float,
    high: float,
    eps: float,
) -> torch.Tensor:
    value = value.float()
    flat = value.flatten(2)
    low_value = torch.quantile(flat, low, dim=-1, keepdim=True)
    high_value = torch.quantile(flat, high, dim=-1, keepdim=True)
    normalized = (
        (flat - low_value) / (high_value - low_value + eps)
    ).clamp(0.0, 1.0)
    return normalized.reshape_as(value)


def _masked_quantile(
    value: torch.Tensor,
    mask: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    thresholds = []
    for batch_index in range(value.shape[0]):
        batch_thresholds = []
        for frame_index in range(value.shape[1]):
            selected = value[batch_index, frame_index][
                mask[batch_index, frame_index]
            ]
            threshold = (
                torch.quantile(selected.float(), quantile)
                if selected.numel()
                else value.new_tensor(float("inf"))
            )
            batch_thresholds.append(threshold)
        thresholds.append(torch.stack(batch_thresholds))
    return torch.stack(thresholds).unsqueeze(-1).unsqueeze(-1)


@dataclass(frozen=True)
class HandRoleInferenceResult:
    roles: RoleState
    token_edit_confidence: torch.Tensor
    debug: Dict[str, torch.Tensor]


class HandRoleInferencer:
    """Infer an active object from source text attention and a hand mask."""

    def __init__(
        self,
        attention_quantile_low: float = 0.50,
        attention_quantile_high: float = 0.95,
        seed_quantile: float = 0.85,
        candidate_quantile: float = 0.75,
        hand_proximity_radius: int = 3,
        propagation_steps: int = 2,
        propagation_alpha: float = 0.55,
        max_object_coverage: float = 0.18,
        visibility_ratio: float = 0.40,
        temporal_weight: float = 0.45,
        query_similarity_threshold: float = 0.65,
        query_temperature: float = 0.07,
        field_quantile_low: float = 0.50,
        field_quantile_high: float = 0.95,
        field_power: float = 1.5,
        field_weight: float = 0.65,
        field_candidate_radius: int = 2,
        connected_hysteresis: bool = False,
        connected_growth_steps: int = 3,
        connected_candidate_ratio: float = 1.0,
        soft_hand_contact: bool = False,
        adaptive: bool = False,
        eps: float = 1e-6,
    ):
        if not (
            0.0 <= attention_quantile_low < attention_quantile_high <= 1.0
        ):
            raise ValueError("Attention quantiles must satisfy 0 <= low < high <= 1")
        if not 0.0 <= seed_quantile <= 1.0:
            raise ValueError("seed_quantile must be in [0, 1]")
        if not 0.0 <= candidate_quantile <= 1.0:
            raise ValueError("candidate_quantile must be in [0, 1]")
        if hand_proximity_radius < 0:
            raise ValueError("hand_proximity_radius must be non-negative")
        if propagation_steps < 0:
            raise ValueError("propagation_steps must be non-negative")
        if not 0.0 <= propagation_alpha <= 1.0:
            raise ValueError("propagation_alpha must be in [0, 1]")
        if not 0.0 < max_object_coverage <= 1.0:
            raise ValueError("max_object_coverage must be in (0, 1]")
        if not 0.0 <= visibility_ratio <= 1.0:
            raise ValueError("visibility_ratio must be in [0, 1]")
        if not 0.0 <= temporal_weight <= 1.0:
            raise ValueError("temporal_weight must be in [0, 1]")
        if not -1.0 < query_similarity_threshold < 1.0:
            raise ValueError(
                "query_similarity_threshold must be in (-1, 1)"
            )
        if query_temperature <= 0:
            raise ValueError("query_temperature must be positive")
        if not (
            0.0 <= field_quantile_low < field_quantile_high <= 1.0
        ):
            raise ValueError(
                "Field quantiles must satisfy 0 <= low < high <= 1"
            )
        if field_power <= 0:
            raise ValueError("field_power must be positive")
        if not 0.0 <= field_weight <= 1.0:
            raise ValueError("field_weight must be in [0, 1]")
        if field_candidate_radius < 0:
            raise ValueError("field_candidate_radius must be non-negative")
        if connected_growth_steps < 0:
            raise ValueError(
                "connected_growth_steps must be non-negative"
            )
        if not 0.0 < connected_candidate_ratio <= 1.0:
            raise ValueError(
                "connected_candidate_ratio must lie in (0, 1]"
            )

        self.attention_quantile_low = attention_quantile_low
        self.attention_quantile_high = attention_quantile_high
        self.seed_quantile = seed_quantile
        self.candidate_quantile = candidate_quantile
        self.hand_proximity_radius = hand_proximity_radius
        self.propagation_steps = propagation_steps
        self.propagation_alpha = propagation_alpha
        self.max_object_coverage = max_object_coverage
        self.visibility_ratio = visibility_ratio
        self.temporal_weight = temporal_weight
        self.query_similarity_threshold = query_similarity_threshold
        self.query_temperature = query_temperature
        self.field_quantile_low = field_quantile_low
        self.field_quantile_high = field_quantile_high
        self.field_power = field_power
        self.field_weight = field_weight
        self.field_candidate_radius = field_candidate_radius
        self.connected_hysteresis = bool(connected_hysteresis)
        self.connected_growth_steps = int(connected_growth_steps)
        self.connected_candidate_ratio = float(
            connected_candidate_ratio
        )
        self.soft_hand_contact = bool(soft_hand_contact)
        self.adaptive = adaptive
        self.eps = eps
        self.previous_features = None
        self.previous_posterior = None
        self.reference_interaction_support = None
        self.adaptive_calibrator = (
            AdaptiveRoleCalibrator(eps=eps) if adaptive else None
        )

    @staticmethod
    def _build_roles(
        posterior: torch.Tensor,
        hand_occupancy: torch.Tensor,
        *,
        soft_hand_contact: bool = False,
    ) -> RoleState:
        batch, frames, height, width = hand_occupancy.shape
        token_height, token_width = posterior.shape[-2:]
        posterior_latent = F.interpolate(
            posterior.reshape(
                batch * frames, 1, token_height, token_width
            ),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, frames, height, width).clamp(0.0, 1.0)
        hand_latent = hand_occupancy.float().clamp(0.0, 1.0)
        if soft_hand_contact:
            # In causal-evidence mode a moving hand is soft contact evidence.
            # Dilating its temporal union would convert most of a held object
            # into read-only boundary.
            hand_band = hand_latent
        else:
            # Preserve the pre-950 role semantics for every baseline that does
            # not explicitly enable causal hand evidence.
            hand_band = _dilate(hand_latent, 1).clamp(0.0, 1.0)

        contact = posterior_latent * hand_band
        object_core = posterior_latent * (1.0 - hand_band)
        hand_core = (1.0 - posterior_latent) * hand_latent
        background = (1.0 - posterior_latent) * (1.0 - hand_latent)
        roles = RoleState(
            object=object_core,
            boundary=contact,
            hand=hand_core,
            background=background,
        )
        roles.validate()
        return roles

    def refine_with_field(
        self,
        prior: HandRoleInferenceResult,
        source_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        hand_mask: torch.Tensor,
        hand_occupancy: torch.Tensor | None = None,
        apply_update: bool = True,
    ) -> HandRoleInferenceResult:
        """Measure field disagreement and optionally refine a role prior."""
        if source_velocity.shape != target_velocity.shape:
            raise ValueError(
                "source_velocity and target_velocity must share a shape"
            )
        if source_velocity.ndim != 5:
            raise ValueError(
                "Velocities must have shape [B,T,C,H,W], got "
                f"{tuple(source_velocity.shape)}"
            )
        if hand_mask.ndim != 4:
            raise ValueError("hand_mask must have shape [B,T,H,W]")
        batch, frames, _, height, width = source_velocity.shape
        if tuple(hand_mask.shape) != (batch, frames, height, width):
            raise ValueError(
                "Velocity and hand-mask grids differ: "
                f"{tuple(source_velocity.shape)} vs {tuple(hand_mask.shape)}"
            )
        if hand_occupancy is None:
            hand_occupancy = hand_mask.float()
        if tuple(hand_occupancy.shape) != (batch, frames, height, width):
            raise ValueError(
                "hand_occupancy must align with the velocity grid"
            )

        prior_posterior = prior.debug["object_posterior"].float()
        source_attention = prior.debug["source_attention"].float()
        object_visible = prior.debug["object_visible"].float()
        token_height, token_width = prior_posterior.shape[-2:]
        expected_tokens = frames * token_height * token_width
        if prior.token_edit_confidence.shape != (batch, expected_tokens):
            raise ValueError("Prior posterior has an inconsistent token grid")

        field_latent = (
            target_velocity.float() - source_velocity.float()
        ).abs().mean(dim=2)
        field_latent = _quantile_normalize(
            field_latent,
            self.field_quantile_low,
            self.field_quantile_high,
            self.eps,
        )
        field_score = F.avg_pool2d(
            field_latent.reshape(batch * frames, 1, height, width),
            kernel_size=2,
            stride=2,
        ).reshape(batch, frames, token_height, token_width)

        hand_probability = F.avg_pool2d(
            hand_occupancy.reshape(
                batch * frames, 1, height, width
            ).float(),
            kernel_size=2,
            stride=2,
        ).reshape(batch, frames, token_height, token_width)
        if self.adaptive:
            posterior, posterior_threshold, adaptive_debug = (
                self.adaptive_calibrator.field_update(
                    prior_posterior=prior_posterior,
                    field_score=field_score,
                    object_seed=prior.debug["object_seed"].float(),
                    object_visible=object_visible,
                    coverage_budget=prior.debug[
                        "adaptive_coverage_budget"
                    ].float(),
                    prior_threshold=prior.debug[
                        "posterior_threshold"
                    ].float(),
                    # The final threshold is committed once, after the
                    # hand/semantic-constrained flow expansion below.
                    update_state=False,
                )
            )
            semantic_threshold = torch.quantile(
                source_attention.flatten(2),
                self.candidate_quantile,
                dim=-1,
                keepdim=True,
            ).reshape(batch, frames, 1, 1)
            semantic_candidate = (
                source_attention >= semantic_threshold
            )
            spatial_candidate = _dilate(
                (prior_posterior > self.eps)
                | (prior.debug["object_seed"] > self.eps),
                self.field_candidate_radius,
            ).bool()
            proximity = prior.debug["hand_proximity"].float()
            field_candidate = (
                spatial_candidate
                & semantic_candidate
                & (proximity > self.eps)
                & (hand_probability < 0.95)
                & object_visible.bool()
            )
            field_observation = (
                adaptive_debug["adaptive_field_likelihood"]
                * adaptive_debug["adaptive_field_reliability"]
                * (0.25 + 0.75 * source_attention)
                * torch.sqrt(proximity.clamp_min(0.0))
                * field_candidate.float()
                * (1.0 - hand_probability)
            ).clamp(0.0, 1.0)
            if apply_update:
                posterior = torch.maximum(
                    posterior,
                    self.field_weight * field_observation,
                )
                posterior = self.adaptive_calibrator.limit_posterior(
                    posterior,
                    prior.debug["adaptive_coverage_budget"].float(),
                    object_visible,
                )
                posterior_threshold = (
                    self.adaptive_calibrator.posterior_threshold(
                        posterior, update_state=True
                    )
                )
                posterior_threshold = torch.maximum(
                    posterior_threshold,
                    prior.debug["posterior_threshold"].float(),
                )
            if not apply_update:
                posterior = prior_posterior
                posterior_threshold = prior.debug[
                    "posterior_threshold"
                ]
        else:
            semantic_threshold = torch.quantile(
                source_attention.flatten(2),
                self.candidate_quantile,
                dim=-1,
                keepdim=True,
            ).reshape(batch, frames, 1, 1)
            semantic_candidate = source_attention >= semantic_threshold
            prior_candidate = _dilate(
                prior_posterior > 0,
                self.field_candidate_radius,
            ).bool()
            field_candidate = (
                semantic_candidate
                & prior_candidate
                & (hand_probability < 0.95)
                & object_visible.bool()
            )
            semantic_gate = 0.25 + 0.75 * source_attention
            field_observation = (
                field_score.pow(self.field_power)
                * semantic_gate
                * field_candidate.float()
                * (1.0 - hand_probability)
            )
            if apply_update:
                posterior = torch.maximum(
                    prior_posterior,
                    self.field_weight * field_observation,
                )
                coverage_threshold = torch.quantile(
                    posterior.flatten(2),
                    1.0 - self.max_object_coverage,
                    dim=-1,
                    keepdim=True,
                ).reshape(batch, frames, 1, 1)
                posterior = (
                    posterior
                    * (posterior >= coverage_threshold).float()
                    * object_visible
                ).clamp(0.0, 1.0)
            else:
                posterior = prior_posterior
            posterior_threshold = prior.debug.get(
                "posterior_threshold",
                posterior.new_full(
                    (batch, frames, 1, 1),
                    0.20,
                ),
            )

        if self.connected_hysteresis:
            connected_support = prior.debug.get(
                "connected_hysteresis_support"
            )
            if connected_support is not None:
                connected_support = connected_support.bool()
                if connected_support.shape != posterior.shape:
                    raise ValueError(
                        "Connected support must align with the posterior"
                    )
                # The flow pass may score the recovered extent, but its
                # top-k coverage limiter must not silently erase the
                # hand-connected object region before routing/KV consume it.
                posterior = torch.where(
                    connected_support,
                    torch.maximum(
                        posterior,
                        posterior_threshold.expand_as(posterior),
                    ),
                    posterior,
                )

        if apply_update:
            self.previous_posterior = posterior[:, -1].detach()

        debug = dict(prior.debug)
        debug.update({
            "object_posterior_prior": prior_posterior,
            "field_score_latent": field_latent,
            "field_score": field_score,
            "field_candidate": field_candidate.float(),
            "field_observation": field_observation,
            "object_posterior": posterior,
            "posterior_threshold": posterior_threshold,
        })
        if self.adaptive:
            debug.update(adaptive_debug)
        return HandRoleInferenceResult(
            roles=self._build_roles(
                posterior,
                hand_occupancy,
                soft_hand_contact=self.soft_hand_contact,
            ),
            token_edit_confidence=posterior.reshape(batch, -1),
            debug=debug,
        )

    @torch.no_grad()
    def refine_with_source_flow(
        self,
        prior: HandRoleInferenceResult,
        flow: FlowRoleEvidence,
        *,
        hand_occupancy: torch.Tensor,
        flow_weight: float = 0.75,
    ) -> HandRoleInferenceResult:
        """Fuse camera-compensated source flow into token ownership.

        The fusion is asymmetric. A transported owner may recover weak
        semantic tokens, while background flow cannot erase an already
        observed object token. This keeps flow useful for tracking without
        turning egocentric camera motion into a destructive segmentation cue.
        """
        flow.validate()
        if not 0.0 <= float(flow_weight) <= 1.0:
            raise ValueError("flow_weight must lie in [0, 1]")
        posterior = prior.debug["object_posterior"].float()
        if flow.object_likelihood.shape != posterior.shape:
            raise ValueError(
                "Source-flow evidence must align with object posterior"
            )
        if hand_occupancy.ndim != 4 or hand_occupancy.shape[:2] != (
            posterior.shape[0], posterior.shape[1]
        ):
            raise ValueError(
                "hand_occupancy must align with flow evidence on [B,T]"
            )

        transported_object = torch.maximum(
            flow.object_likelihood,
            flow.transport_support * flow.cycle_confidence,
        )
        flow_observation = (
            float(flow_weight) * transported_object
        ).clamp(0.0, 1.0)
        fused = torch.maximum(posterior, flow_observation)
        # Positive clean-source background evidence is retained separately for
        # the factorized Bayes operator. It is not allowed to crop the object
        # hypothesis; occlusion and weak flow therefore fail to unknown/read
        # abstention instead of accumulating chunk-wise shrinkage.
        threshold = prior.debug["posterior_threshold"].float()
        flow_selected = flow_observation >= threshold

        debug = dict(prior.debug)
        debug.update(flow.as_debug_maps())
        debug.update({
            "object_posterior_pre_source_flow": posterior,
            "source_flow_object_observation": flow_observation,
            "source_flow_recovered_support": (
                flow_selected & (posterior < threshold)
            ).float(),
            "object_posterior": fused,
        })
        return HandRoleInferenceResult(
            roles=self._build_roles(
                fused, hand_occupancy,
                soft_hand_contact=self.soft_hand_contact,
            ),
            token_edit_confidence=fused.reshape(fused.shape[0], -1),
            debug=debug,
        )

    @torch.no_grad()
    def apply_source_flow_verified_region(
        self,
        prior: HandRoleInferenceResult,
        flow: FlowRoleEvidence,
        *,
        owner_support: torch.Tensor,
        hand_exclusion: torch.Tensor,
        hand_occupancy: torch.Tensor,
        owner_radius: int = 1,
        background_veto_threshold: float = 0.55,
        background_veto_min_confidence: float = 0.50,
    ) -> HandRoleInferenceResult:
        """Turn high-recall token evidence into verified edit authority.

        The original posterior remains available in diagnostics, but the role
        maps and token confidence returned to routing are rebuilt from the
        flow-verified support.  This prevents a later OR from restoring hand
        or confident-background false positives.
        """
        # On the first pass, verify the semantic posterior from before the
        # positive source-flow fusion.  Flow ownership already has its own
        # explicit recovery path below, so reusing its positive observation
        # as a semantic proposal would count the same evidence twice.  A
        # later denoising-field pass is re-verified from its latest posterior.
        proposal_posterior = prior.debug["object_posterior"]
        if "source_flow_verified_support" not in prior.debug:
            proposal_posterior = prior.debug.get(
                "object_posterior_pre_source_flow",
                proposal_posterior,
            )
        verified = build_source_flow_verified_region(
            object_posterior=proposal_posterior,
            posterior_threshold=prior.debug["posterior_threshold"],
            owner_support=owner_support,
            hand_exclusion=hand_exclusion,
            background_likelihood=flow.background_likelihood,
            flow_confidence=flow.cycle_confidence,
            owner_radius=owner_radius,
            background_veto_threshold=background_veto_threshold,
            background_veto_min_confidence=(
                background_veto_min_confidence
            ),
        )
        debug = dict(prior.debug)
        # Preserve the first unverified proposal across the later denoising
        # field pass.  The latest map is also recorded so an offline replay
        # can distinguish the original token proposal from a second
        # verification of the field-refined posterior.
        debug.setdefault(
            "object_posterior_pre_flow_verification",
            proposal_posterior.float(),
        )
        debug["object_posterior_pre_latest_flow_verification"] = (
            proposal_posterior.float()
        )
        debug.update(verified.as_debug_maps())
        debug["object_posterior"] = verified.posterior
        # Temporal token propagation in the next causal block must start
        # from the same verified authority consumed by routing/KV.  Leaving
        # the pre-verification field posterior here would reintroduce the
        # discarded hand/background fringe at every chunk boundary.
        self.previous_posterior = verified.posterior[:, -1].detach()
        return HandRoleInferenceResult(
            roles=self._build_roles(
                verified.posterior,
                hand_occupancy,
                soft_hand_contact=self.soft_hand_contact,
            ),
            token_edit_confidence=verified.posterior.reshape(
                verified.posterior.shape[0], -1
            ),
            debug=debug,
        )

    def _query_affinity_propagation(
        self,
        current_features: torch.Tensor,
        reference_features: torch.Tensor,
        reference_posterior: torch.Tensor,
    ):
        similarity = torch.einsum(
            "bnd,bmd->bnm",
            current_features,
            reference_features,
        ).clamp(-1.0, 1.0)
        topk = min(4, similarity.shape[-1])
        top_similarity, top_index = similarity.topk(topk, dim=-1)
        reference_flat = reference_posterior.flatten(1)
        selected_posterior = reference_flat.unsqueeze(1).expand(
            -1,
            current_features.shape[1],
            -1,
        ).gather(2, top_index)
        if self.adaptive:
            best_similarity = top_similarity[..., 0]
            similarity_low = torch.quantile(
                best_similarity,
                0.25,
                dim=-1,
                keepdim=True,
            )
            similarity_high = torch.quantile(
                best_similarity,
                0.75,
                dim=-1,
                keepdim=True,
            )
            temperature = (
                similarity_high - similarity_low
            ).clamp(0.03, 0.20).unsqueeze(-1)
        else:
            temperature = self.query_temperature
        affinity_weight = torch.softmax(
            top_similarity / temperature,
            dim=-1,
        )
        if self.adaptive:
            best_similarity = top_similarity[..., 0]
            median_similarity = torch.quantile(
                best_similarity,
                0.50,
                dim=-1,
                keepdim=True,
            )
            high_similarity = torch.quantile(
                best_similarity,
                0.90,
                dim=-1,
                keepdim=True,
            )
            relative_confidence = (
                (best_similarity - median_similarity)
                / (
                    high_similarity
                    - median_similarity
                    + self.eps
                )
            ).clamp(0.0, 1.0)
            absolute_confidence = (
                0.5 * (best_similarity + 1.0)
            ).clamp(0.0, 1.0)
            confidence = torch.sqrt(
                relative_confidence * absolute_confidence
            )
        else:
            confidence = (
                (
                    top_similarity[..., 0]
                    - self.query_similarity_threshold
                )
                / (
                    1.0
                    - self.query_similarity_threshold
                    + self.eps
                )
            ).clamp(0.0, 1.0)
        propagated = (
            affinity_weight * selected_posterior
        ).sum(dim=-1) * confidence
        return propagated, confidence

    def __call__(
        self,
        source_attention: torch.Tensor,
        hand_mask: torch.Tensor,
        hand_occupancy: torch.Tensor | None = None,
        source_features: torch.Tensor = None,
    ) -> HandRoleInferenceResult:
        if source_attention.ndim != 2:
            raise ValueError(
                "source_attention must have shape [B,L], got "
                f"{tuple(source_attention.shape)}"
            )
        if hand_mask.ndim != 4:
            raise ValueError(
                "hand_mask must have shape [B,T,H,W], got "
                f"{tuple(hand_mask.shape)}"
            )
        batch, frames, height, width = hand_mask.shape
        if hand_occupancy is None:
            hand_occupancy = hand_mask.float()
        if tuple(hand_occupancy.shape) != tuple(hand_mask.shape):
            raise ValueError(
                "hand_occupancy must share hand_mask shape [B,T,H,W]"
            )
        if (hand_occupancy < 0).any() or (hand_occupancy > 1).any():
            raise ValueError("hand_occupancy must lie in [0, 1]")
        if height % 2 or width % 2:
            raise ValueError("Hand-mask height and width must be divisible by 2")
        token_height, token_width = height // 2, width // 2
        expected_length = frames * token_height * token_width
        if tuple(source_attention.shape) != (batch, expected_length):
            raise ValueError(
                "Source attention and hand mask imply different token grids: "
                f"expected {(batch, expected_length)}, got "
                f"{tuple(source_attention.shape)}"
            )
        if source_features is not None:
            if source_features.ndim != 3:
                raise ValueError(
                    "source_features must have shape [B,L,D], got "
                    f"{tuple(source_features.shape)}"
                )
            if source_features.shape[:2] != source_attention.shape:
                raise ValueError(
                    "source_features and source_attention must share [B,L]"
                )
            source_features = F.normalize(
                source_features.float(),
                dim=-1,
            ).reshape(
                batch,
                frames,
                token_height * token_width,
                -1,
            )

        attention = source_attention.reshape(
            batch, frames, token_height, token_width
        )
        attention = _quantile_normalize(
            attention,
            self.attention_quantile_low,
            self.attention_quantile_high,
            self.eps,
        )
        hand_probability = F.avg_pool2d(
            hand_occupancy.reshape(
                batch * frames, 1, height, width
            ).float(),
            kernel_size=2,
            stride=2,
        ).reshape(batch, frames, token_height, token_width)
        proximity_hand_probability = F.max_pool2d(
            hand_mask.reshape(
                batch * frames, 1, height, width
            ).float(),
            kernel_size=2,
            stride=2,
        ).reshape(batch, frames, token_height, token_width)
        adaptive_debug = {}
        adaptive_observation = None
        if self.adaptive:
            adaptive_observation = self.adaptive_calibrator.observe(
                attention,
                hand_probability,
                proximity_hand_probability=proximity_hand_probability,
            )
            proximity = adaptive_observation.debug["hand_proximity"]
            seed = adaptive_observation.seed
            interaction_support = adaptive_observation.debug[
                "interaction_support"
            ]
            visibility_threshold = adaptive_observation.debug[
                "visibility_threshold"
            ]
            object_visible = adaptive_observation.object_visible
            gate = adaptive_observation.gate
            adaptive_debug = adaptive_observation.debug
        else:
            hand_binary = proximity_hand_probability > 0.0
            hand_coverage = proximity_hand_probability.flatten(2).mean(
                dim=-1
            )
            hand_present = (
                hand_binary.flatten(2).any(dim=-1)
                & (hand_coverage < 0.95)
            ).view(
                batch, frames, 1, 1
            )

            proximity = torch.zeros_like(attention)
            for radius in range(self.hand_proximity_radius + 1):
                ring = _dilate(hand_binary, radius).bool()
                if radius:
                    ring &= ~_dilate(hand_binary, radius - 1).bool()
                proximity = torch.where(
                    ring,
                    proximity.new_tensor(
                        math.exp(
                            -float(radius)
                            / max(
                                float(self.hand_proximity_radius),
                                1.0,
                            )
                        )
                    ),
                    proximity,
                )
            near_hand = _dilate(
                hand_binary,
                self.hand_proximity_radius,
            ).bool()
            interaction_prior = 0.25 + 0.75 * proximity
            seed_score = attention.pow(1.5) * interaction_prior
            seed_threshold = _masked_quantile(
                seed_score,
                near_hand,
                self.seed_quantile,
            )
            seed = (
                seed_score
                * (seed_score >= seed_threshold).float()
                * near_hand.float()
                * hand_present.float()
            )
            seed_candidate = (
                (seed_score >= seed_threshold)
                & near_hand
                & hand_present
            )
            interaction_support = (
                seed_score * seed_candidate.float()
            ).mean(dim=(2, 3), keepdim=True)
            current_reference = interaction_support.amax(
                dim=1,
                keepdim=True,
            ).detach()
            if self.reference_interaction_support is None:
                self.reference_interaction_support = current_reference
            else:
                self.reference_interaction_support = torch.maximum(
                    self.reference_interaction_support,
                    current_reference,
                )
            visibility_threshold = (
                self.reference_interaction_support
                * self.visibility_ratio
            )
            object_visible = (
                (self.reference_interaction_support > self.eps)
                & (interaction_support >= visibility_threshold)
            )

            candidate_threshold = torch.quantile(
                attention.flatten(2),
                self.candidate_quantile,
                dim=-1,
                keepdim=True,
            ).reshape(batch, frames, 1, 1)
            gate = attention * (
                attention >= candidate_threshold
            ).float()
        posterior = seed
        for _ in range(self.propagation_steps):
            propagated = _neighbor_max(posterior)
            posterior = (
                self.propagation_alpha * seed
                + (1.0 - self.propagation_alpha) * propagated
            )
            posterior = posterior * gate
        posterior = posterior * object_visible.float()
        hysteresis_candidate = torch.zeros_like(
            posterior, dtype=torch.bool
        )
        hysteresis_support = torch.zeros_like(
            posterior, dtype=torch.bool
        )
        if self.connected_hysteresis:
            if self.adaptive:
                candidate_threshold = adaptive_observation.debug[
                    "adaptive_candidate_threshold"
                ]
                extended_hand = adaptive_observation.debug[
                    "adaptive_extended_hand"
                ].bool()
            else:
                candidate_threshold = torch.quantile(
                    attention.flatten(2),
                    self.candidate_quantile,
                    dim=-1,
                    keepdim=True,
                ).reshape(batch, frames, 1, 1)
                extended_hand = _dilate(
                    hand_probability > 0.0,
                    2 * self.hand_proximity_radius,
                ).bool()
            connected_candidate_threshold = (
                candidate_threshold * self.connected_candidate_ratio
            )
            hysteresis_candidate = (
                (attention >= connected_candidate_threshold)
                & (attention > self.eps)
                & extended_hand
                & object_visible.bool()
            )
            hysteresis_support = _connected_hysteresis_growth(
                seed > self.eps,
                hysteresis_candidate,
                steps=self.connected_growth_steps,
            )
            # Candidate attention is calibrated evidence, not a binary mask.
            # Preserve its confidence while ensuring connected low-score
            # interiors survive later posterior thresholding.
            connected_floor = torch.minimum(
                connected_candidate_threshold.expand_as(attention),
                attention.new_full(attention.shape, 0.20),
            )
            connected_posterior = torch.maximum(
                attention, connected_floor
            ) * hysteresis_support.float()
            posterior = torch.maximum(posterior, connected_posterior)

        temporal_posterior = torch.zeros_like(posterior)
        temporal_confidence = torch.zeros_like(posterior)
        adaptive_temporal_weight = torch.zeros(
            batch,
            frames,
            1,
            1,
            device=posterior.device,
            dtype=torch.float32,
        )
        if source_features is not None:
            for frame_index in range(frames):
                if frame_index == 0:
                    reference_features = self.previous_features
                    reference_posterior = self.previous_posterior
                else:
                    reference_features = source_features[:, frame_index - 1]
                    reference_posterior = posterior[:, frame_index - 1]
                if (
                    reference_features is None
                    or reference_posterior is None
                ):
                    continue
                propagated, confidence = self._query_affinity_propagation(
                    source_features[:, frame_index],
                    reference_features,
                    reference_posterior,
                )
                propagated = propagated.reshape(
                    batch,
                    token_height,
                    token_width,
                )
                confidence = confidence.reshape(
                    batch,
                    token_height,
                    token_width,
                )
                temporal_posterior[:, frame_index] = propagated
                temporal_confidence[:, frame_index] = confidence
                semantic_gate = 0.20 + 0.80 * attention[:, frame_index]
                if self.adaptive:
                    temporal_weight = (
                        self.adaptive_calibrator.temporal_weight(
                            propagated,
                            confidence,
                            adaptive_observation.attention_reliability[
                                :, frame_index
                            ],
                        )
                    )
                    adaptive_temporal_weight[:, frame_index] = (
                        temporal_weight
                    )
                else:
                    temporal_weight = self.temporal_weight
                posterior[:, frame_index] = torch.maximum(
                    posterior[:, frame_index],
                    temporal_weight
                    * propagated
                    * semantic_gate,
                ) * object_visible[:, frame_index].float()

        if self.adaptive:
            posterior = self.adaptive_calibrator.limit_posterior(
                posterior,
                adaptive_observation.coverage_budget,
                object_visible,
            )
            posterior_threshold = (
                self.adaptive_calibrator.posterior_threshold(posterior)
            )
        else:
            coverage_threshold = torch.quantile(
                posterior.flatten(2),
                1.0 - self.max_object_coverage,
                dim=-1,
                keepdim=True,
            ).reshape(batch, frames, 1, 1)
            posterior = (
                posterior
                * (posterior >= coverage_threshold).float()
                * object_visible.float()
            ).clamp(0.0, 1.0)
            posterior_threshold = posterior.new_full(
                (batch, frames, 1, 1),
                0.20,
            )
        if self.connected_hysteresis:
            posterior = torch.where(
                hysteresis_support,
                torch.maximum(
                    posterior, posterior_threshold.expand_as(posterior)
                ),
                posterior,
            )
        if source_features is not None:
            self.previous_features = source_features[:, -1].detach()
            self.previous_posterior = posterior[:, -1].detach()

        debug = {
            "source_attention": attention,
            "hand_probability": hand_probability,
            "hand_proximity": proximity,
            "object_seed": seed,
            "interaction_support": interaction_support,
            "visibility_threshold": visibility_threshold.expand_as(
                interaction_support
            ),
            "object_visible": object_visible.float(),
            "temporal_posterior": temporal_posterior,
            "temporal_confidence": temporal_confidence,
            "object_posterior": posterior,
            "posterior_threshold": posterior_threshold,
            "connected_hysteresis_candidate": (
                hysteresis_candidate.float()
            ),
            "connected_candidate_threshold": (
                connected_candidate_threshold.expand_as(attention)
                if self.connected_hysteresis
                else torch.zeros_like(attention)
            ),
            "connected_hysteresis_support": (
                hysteresis_support.float()
            ),
        }
        if self.adaptive:
            debug["adaptive_temporal_weight"] = (
                adaptive_temporal_weight
            )
        debug.update(adaptive_debug)
        return HandRoleInferenceResult(
            roles=self._build_roles(
                posterior,
                hand_occupancy,
                soft_hand_contact=self.soft_hand_contact,
            ),
            token_edit_confidence=posterior.reshape(batch, -1),
            debug=debug,
        )
