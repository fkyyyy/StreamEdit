from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.float()
    batch, frames, height, width = mask.shape
    flat = mask.reshape(batch * frames, 1, height, width).float()
    value = F.max_pool2d(
        flat,
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )
    return value.reshape(batch, frames, height, width)


def _masked_quantile(
    value: torch.Tensor,
    mask: torch.Tensor,
    quantile: float,
    default: float = 0.0,
) -> torch.Tensor:
    output = []
    for batch_index in range(value.shape[0]):
        batch_output = []
        for frame_index in range(value.shape[1]):
            selected = value[batch_index, frame_index][
                mask[batch_index, frame_index]
            ]
            if selected.numel():
                result = torch.quantile(selected.float(), quantile)
            else:
                result = value.new_tensor(default, dtype=torch.float32)
            batch_output.append(result)
        output.append(torch.stack(batch_output))
    return torch.stack(output).unsqueeze(-1).unsqueeze(-1)


def _masked_mad(
    value: torch.Tensor,
    mask: torch.Tensor,
    center: torch.Tensor,
) -> torch.Tensor:
    deviation = (value - center).abs()
    return _masked_quantile(deviation, mask, 0.50)


@dataclass(frozen=True)
class AdaptiveObservation:
    seed: torch.Tensor
    gate: torch.Tensor
    object_visible: torch.Tensor
    coverage_budget: torch.Tensor
    attention_reliability: torch.Tensor
    debug: Dict[str, torch.Tensor]


@dataclass
class AdaptiveCalibrationState:
    support_history: List[torch.Tensor] = field(default_factory=list)
    previous_posterior_threshold: Optional[torch.Tensor] = None
    current_attention_reliability: Optional[torch.Tensor] = None


class AdaptiveRoleCalibrator:
    """Causally calibrate role evidence from its online reliability."""

    def __init__(
        self,
        radius_scale: float = 0.55,
        min_radius: int = 1,
        max_radius: int = 6,
        max_coverage: float = 0.30,
        support_history_size: int = 32,
        visibility_mad_scale: float = 3.0,
        eps: float = 1e-6,
    ):
        if radius_scale <= 0:
            raise ValueError("radius_scale must be positive")
        if min_radius < 0 or max_radius < min_radius:
            raise ValueError("Invalid adaptive radius range")
        if not 0.0 < max_coverage <= 1.0:
            raise ValueError("max_coverage must be in (0, 1]")
        if support_history_size <= 0:
            raise ValueError("support_history_size must be positive")
        if visibility_mad_scale <= 0:
            raise ValueError("visibility_mad_scale must be positive")

        self.radius_scale = radius_scale
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.max_coverage = max_coverage
        self.support_history_size = support_history_size
        self.visibility_mad_scale = visibility_mad_scale
        self.eps = eps
        self.state = AdaptiveCalibrationState()

    def reset(self):
        self.state = AdaptiveCalibrationState()

    def _hand_geometry(self, hand_probability: torch.Tensor):
        hand_binary = hand_probability > 0.0
        hand_area = hand_binary.flatten(2).float().sum(dim=-1)
        equivalent_radius = torch.sqrt(hand_area / math.pi)
        radius = torch.ceil(
            equivalent_radius * self.radius_scale
        ).clamp(self.min_radius, self.max_radius)
        hand_coverage = hand_binary.flatten(2).float().mean(dim=-1)
        hand_present = (
            hand_binary.flatten(2).any(dim=-1)
            & (hand_coverage < 0.95)
        )
        radius = torch.where(
            hand_present,
            radius,
            torch.zeros_like(radius),
        ).long().unsqueeze(-1).unsqueeze(-1)

        max_distance = 2 * self.max_radius
        distance = torch.full_like(
            hand_probability,
            float(max_distance + 1),
            dtype=torch.float32,
        )
        previous = torch.zeros_like(hand_binary)
        for distance_index in range(max_distance + 1):
            current = _dilate(hand_binary, distance_index).bool()
            ring = current & ~previous
            distance = torch.where(
                ring,
                distance.new_tensor(float(distance_index)),
                distance,
            )
            previous = current

        radius_float = radius.float().clamp_min(1.0)
        near_hand = distance <= radius.float()
        extended_hand = distance <= 2.0 * radius.float()
        proximity = torch.where(
            extended_hand,
            torch.exp(-distance / radius_float),
            torch.zeros_like(distance),
        )
        proximity = proximity * hand_present.view(
            hand_probability.shape[0],
            hand_probability.shape[1],
            1,
            1,
        ).float()
        return (
            hand_binary,
            hand_present,
            radius,
            near_hand,
            extended_hand,
            proximity,
        )

    def _attention_calibration(
        self,
        attention: torch.Tensor,
        near_hand: torch.Tensor,
        extended_hand: torch.Tensor,
    ):
        far_region = ~extended_hand
        local_high = _masked_quantile(attention, near_hand, 0.75)
        far_high = _masked_quantile(attention, far_region, 0.75)
        global_low = torch.quantile(
            attention.flatten(2), 0.25, dim=-1, keepdim=True
        ).reshape(attention.shape[0], attention.shape[1], 1, 1)
        global_high = torch.quantile(
            attention.flatten(2), 0.75, dim=-1, keepdim=True
        ).reshape(attention.shape[0], attention.shape[1], 1, 1)
        separation = (
            (local_high - far_high).clamp_min(0.0)
            / (global_high - global_low + self.eps)
        )
        reliability = 1.0 - torch.exp(-separation)

        local_median = _masked_quantile(attention, near_hand, 0.50)
        local_peak = _masked_quantile(attention, near_hand, 0.90)
        seed_fraction = 0.75 - 0.20 * reliability
        seed_threshold = (
            local_median
            + seed_fraction * (local_peak - local_median)
        )

        global_median = torch.quantile(
            attention.flatten(2), 0.50, dim=-1, keepdim=True
        ).reshape(attention.shape[0], attention.shape[1], 1, 1)
        global_peak = torch.quantile(
            attention.flatten(2), 0.90, dim=-1, keepdim=True
        ).reshape(attention.shape[0], attention.shape[1], 1, 1)
        candidate_fraction = 0.70 - 0.20 * reliability
        candidate_threshold = (
            global_median
            + candidate_fraction * (global_peak - global_median)
        )
        return (
            reliability.clamp(0.0, 1.0),
            seed_threshold,
            candidate_threshold,
        )

    def _visibility(
        self,
        interaction_support: torch.Tensor,
        hand_present: torch.Tensor,
    ):
        batch, frames = interaction_support.shape[:2]
        visible_values = []
        threshold_values = []
        center_values = []
        mad_values = []
        for frame_index in range(frames):
            support = interaction_support[:, frame_index]
            present = hand_present[:, frame_index].view(batch, 1, 1)
            if self.state.support_history:
                history = torch.stack(
                    self.state.support_history,
                    dim=1,
                )
                center = history.median(dim=1).values
                upper_reference = torch.quantile(
                    history,
                    0.75,
                    dim=1,
                )
                mad = (
                    (history - center.unsqueeze(1))
                    .abs()
                    .median(dim=1)
                    .values
                )
                robust_lower = (
                    center
                    - self.visibility_mad_scale * 1.4826 * mad
                ).clamp_min(0.0)
                fallback_lower = 0.35 * upper_reference
                threshold = torch.where(
                    mad > self.eps,
                    torch.maximum(robust_lower, fallback_lower),
                    fallback_lower,
                )
            else:
                center = support.detach()
                mad = torch.zeros_like(support)
                threshold = torch.zeros_like(support)

            visible = (
                present
                & (support > self.eps)
                & (support >= threshold)
            )
            accepted = torch.where(visible, support, center).detach()
            self.state.support_history.append(accepted)
            if (
                len(self.state.support_history)
                > self.support_history_size
            ):
                self.state.support_history.pop(0)

            visible_values.append(visible)
            threshold_values.append(threshold)
            center_values.append(center)
            mad_values.append(mad)

        return (
            torch.stack(visible_values, dim=1),
            torch.stack(threshold_values, dim=1),
            torch.stack(center_values, dim=1),
            torch.stack(mad_values, dim=1),
        )

    def observe(
        self,
        attention: torch.Tensor,
        hand_probability: torch.Tensor,
    ) -> AdaptiveObservation:
        if attention.shape != hand_probability.shape:
            raise ValueError(
                "attention and hand_probability must share [B,T,H,W]"
            )
        attention = attention.float().clamp(0.0, 1.0)
        hand_probability = hand_probability.float().clamp(0.0, 1.0)
        (
            hand_binary,
            hand_present,
            radius,
            near_hand,
            extended_hand,
            proximity,
        ) = self._hand_geometry(hand_probability)
        (
            attention_reliability,
            seed_threshold,
            candidate_threshold,
        ) = self._attention_calibration(
            attention,
            near_hand,
            extended_hand,
        )
        self.state.current_attention_reliability = (
            attention_reliability.detach()
        )

        interaction_prior = 0.20 + 0.80 * proximity
        seed_score = attention.pow(1.5) * interaction_prior
        seed_candidate = (
            (seed_score >= seed_threshold)
            & near_hand
            & hand_present.unsqueeze(-1).unsqueeze(-1)
            & (attention > self.eps)
        )
        seed = seed_score * seed_candidate.float()
        near_count = near_hand.flatten(2).float().sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0).unsqueeze(-1)
        token_count = attention.shape[-2] * attention.shape[-1]
        near_coverage = near_count / float(token_count)
        interaction_support = (
            (seed_score * seed_candidate.float())
            .flatten(2)
            .sum(dim=-1, keepdim=True)
            .unsqueeze(-1)
            / near_count
            * torch.sqrt(near_coverage)
        )
        (
            object_visible,
            visibility_threshold,
            support_center,
            support_mad,
        ) = self._visibility(interaction_support, hand_present)

        semantic_candidate = (
            (attention >= candidate_threshold)
            & (attention > self.eps)
        )
        gate = attention * semantic_candidate.float()
        semantic_extent = semantic_candidate & extended_hand
        extent_coverage = semantic_extent.flatten(2).float().mean(dim=-1)
        hand_coverage = hand_binary.flatten(2).float().mean(dim=-1)
        minimum_coverage = attention.new_tensor(1.0 / token_count)
        coverage_budget = torch.maximum(
            1.25 * extent_coverage,
            1.50 * hand_coverage,
        )
        coverage_budget = torch.maximum(
            coverage_budget,
            minimum_coverage,
        ).clamp_max(self.max_coverage)
        coverage_budget = coverage_budget.unsqueeze(-1).unsqueeze(-1)

        debug = {
            "adaptive_hand_radius": radius.float(),
            "adaptive_attention_reliability": attention_reliability,
            "adaptive_seed_threshold": seed_threshold,
            "adaptive_candidate_threshold": candidate_threshold,
            "adaptive_coverage_budget": coverage_budget,
            "adaptive_support_center": support_center,
            "adaptive_support_mad": support_mad,
            "hand_proximity": proximity,
            "object_seed": seed,
            "interaction_support": interaction_support,
            "visibility_threshold": visibility_threshold,
            "object_visible": object_visible.float(),
        }
        return AdaptiveObservation(
            seed=seed,
            gate=gate,
            object_visible=object_visible,
            coverage_budget=coverage_budget,
            attention_reliability=attention_reliability,
            debug=debug,
        )

    def temporal_weight(
        self,
        propagated: torch.Tensor,
        confidence: torch.Tensor,
        attention_reliability: torch.Tensor,
    ) -> torch.Tensor:
        active = propagated > self.eps
        active_count = active.flatten(1).float().sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        match_reliability = (
            (confidence * active.float()).flatten(1).sum(
                dim=-1, keepdim=True
            )
            / active_count
        ).reshape(-1, 1, 1)
        return (
            match_reliability
            * (0.25 + 0.75 * attention_reliability)
        ).clamp(0.0, 0.90)

    def limit_posterior(
        self,
        posterior: torch.Tensor,
        coverage_budget: torch.Tensor,
        object_visible: torch.Tensor,
    ) -> torch.Tensor:
        flat = posterior.float().flatten(2)
        order = flat.argsort(dim=-1, descending=True)
        ranks = torch.empty_like(order)
        rank_values = torch.arange(
            flat.shape[-1],
            device=flat.device,
        ).view(1, 1, -1).expand_as(order)
        ranks.scatter_(-1, order, rank_values)
        keep_count = (
            coverage_budget.flatten(2)
            * flat.shape[-1]
        ).ceil().clamp_min(1.0)
        keep = ranks < keep_count
        limited = (
            flat
            * keep.float()
            * object_visible.flatten(2).float()
        )
        return limited.reshape_as(posterior).clamp(0.0, 1.0)

    def posterior_threshold(
        self,
        posterior: torch.Tensor,
    ) -> torch.Tensor:
        reliability = self.state.current_attention_reliability
        if (
            reliability is None
            or reliability.shape[:2] != posterior.shape[:2]
        ):
            reliability = posterior.new_zeros(
                posterior.shape[0],
                posterior.shape[1],
                1,
                1,
            )
        thresholds = []
        for frame_index in range(posterior.shape[1]):
            frame_thresholds = []
            for batch_index in range(posterior.shape[0]):
                positive = posterior[batch_index, frame_index]
                positive = positive[positive > self.eps]
                if positive.numel() < 2:
                    threshold = posterior.new_tensor(1.0)
                else:
                    quantile = (
                        0.35
                        + 0.25
                        * (
                            1.0
                            - reliability[
                                batch_index,
                                frame_index,
                                0,
                                0,
                            ]
                        )
                    )
                    threshold = torch.quantile(positive, quantile)
                frame_thresholds.append(threshold)
            current = torch.stack(frame_thresholds).view(-1, 1, 1)
            if self.state.previous_posterior_threshold is not None:
                current = (
                    0.50
                    * self.state.previous_posterior_threshold
                    + 0.50 * current
                )
            self.state.previous_posterior_threshold = (
                current.detach()
            )
            thresholds.append(current)
        return torch.stack(thresholds, dim=1)

    def field_update(
        self,
        prior_posterior: torch.Tensor,
        field_score: torch.Tensor,
        object_seed: torch.Tensor,
        object_visible: torch.Tensor,
        coverage_budget: torch.Tensor,
        prior_threshold: torch.Tensor,
    ):
        seed_mask = object_seed > self.eps
        radius = 1
        ring_mask = _dilate(seed_mask, radius).bool() & ~seed_mask
        seed_center = _masked_quantile(field_score, seed_mask, 0.50)
        ring_center = _masked_quantile(field_score, ring_mask, 0.50)
        seed_mad = _masked_mad(field_score, seed_mask, seed_center)
        ring_mad = _masked_mad(field_score, ring_mask, ring_center)
        separation = (
            (seed_center - ring_center).clamp_min(0.0)
            / (1.4826 * (seed_mad + ring_mad) + self.eps)
        )
        field_reliability = (
            separation / (1.0 + separation)
        ).clamp(0.0, 1.0)

        likelihood_scale = (
            1.4826 * (seed_mad + ring_mad)
        ).clamp_min(0.05)
        field_deficit = (
            seed_center - field_score
        ).clamp_min(0.0) / likelihood_scale
        field_likelihood = torch.exp(
            -field_reliability * field_deficit
        )
        posterior = (
            prior_posterior.float()
            * field_likelihood
            * (prior_posterior > self.eps).float()
            * object_visible.float()
        )
        posterior = self.limit_posterior(
            posterior,
            coverage_budget,
            object_visible,
        )
        threshold = self.posterior_threshold(posterior)
        threshold = torch.maximum(
            threshold,
            prior_threshold.float(),
        )
        debug = {
            "adaptive_field_reliability": field_reliability,
            "adaptive_field_seed_center": seed_center,
            "adaptive_field_ring_center": ring_center,
            "adaptive_field_likelihood": field_likelihood,
            "adaptive_posterior_threshold": threshold,
        }
        return posterior, threshold, debug
