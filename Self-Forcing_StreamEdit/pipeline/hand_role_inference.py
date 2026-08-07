from dataclasses import dataclass
import math
from typing import Dict

import torch
import torch.nn.functional as F

from .role_router import RoleState


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

        self.attention_quantile_low = attention_quantile_low
        self.attention_quantile_high = attention_quantile_high
        self.seed_quantile = seed_quantile
        self.candidate_quantile = candidate_quantile
        self.hand_proximity_radius = hand_proximity_radius
        self.propagation_steps = propagation_steps
        self.propagation_alpha = propagation_alpha
        self.max_object_coverage = max_object_coverage
        self.eps = eps

    def __call__(
        self,
        source_attention: torch.Tensor,
        hand_mask: torch.Tensor,
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
            hand_mask.reshape(batch * frames, 1, height, width).float(),
            kernel_size=2,
            stride=2,
        ).reshape(batch, frames, token_height, token_width)
        hand_binary = hand_probability > 0.0
        hand_coverage = hand_probability.flatten(2).mean(dim=-1)
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
                        / max(float(self.hand_proximity_radius), 1.0)
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

        candidate_threshold = torch.quantile(
            attention.flatten(2),
            self.candidate_quantile,
            dim=-1,
            keepdim=True,
        ).reshape(batch, frames, 1, 1)
        gate = attention * (attention >= candidate_threshold).float()
        posterior = seed
        for _ in range(self.propagation_steps):
            propagated = _neighbor_max(posterior)
            posterior = (
                self.propagation_alpha * seed
                + (1.0 - self.propagation_alpha) * propagated
            )
            posterior = posterior * gate

        coverage_threshold = torch.quantile(
            posterior.flatten(2),
            1.0 - self.max_object_coverage,
            dim=-1,
            keepdim=True,
        ).reshape(batch, frames, 1, 1)
        posterior = (
            posterior
            * (posterior >= coverage_threshold).float()
            * hand_present.float()
        ).clamp(0.0, 1.0)

        posterior_latent = F.interpolate(
            posterior.reshape(
                batch * frames, 1, token_height, token_width
            ),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, frames, height, width).clamp(0.0, 1.0)
        hand_latent = hand_mask.float().clamp(0.0, 1.0)
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

        return HandRoleInferenceResult(
            roles=roles,
            token_edit_confidence=posterior.reshape(batch, -1),
            debug={
                "source_attention": attention,
                "hand_probability": hand_probability,
                "hand_proximity": proximity,
                "object_seed": seed,
                "object_posterior": posterior,
            },
        )
