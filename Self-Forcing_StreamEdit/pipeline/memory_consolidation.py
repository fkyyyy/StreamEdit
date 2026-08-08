from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .belief_kv import BeliefKVWeights
    from .control_belief import CausalControlBelief


@dataclass(frozen=True)
class MemoryConsolidationPlan:
    """Causal transport and precision update for one token block."""

    observation_edit_action: torch.Tensor
    observation_precision: torch.Tensor
    transported_edit_action: torch.Tensor
    transported_precision: torch.Tensor
    consolidated_edit_action: torch.Tensor
    consolidated_precision: torch.Tensor
    materialized_edit_action: torch.Tensor
    observation_gain: torch.Tensor
    reference_index: torch.Tensor
    reference_weight: torch.Tensor
    reference_valid: torch.Tensor

    def validate(self) -> None:
        values = {
            "observation_edit_action": self.observation_edit_action,
            "observation_precision": self.observation_precision,
            "transported_edit_action": self.transported_edit_action,
            "transported_precision": self.transported_precision,
            "consolidated_edit_action": self.consolidated_edit_action,
            "consolidated_precision": self.consolidated_precision,
            "materialized_edit_action": self.materialized_edit_action,
            "observation_gain": self.observation_gain,
            "reference_valid": self.reference_valid,
        }
        shapes = {tuple(value.shape) for value in values.values()}
        if len(shapes) != 1 or self.observation_gain.ndim != 3:
            raise ValueError(
                "Memory consolidation values must share shape [B,T,N]"
            )
        if self.reference_index.ndim != 4:
            raise ValueError(
                "Memory reference indices must have shape [B,T,N,K]"
            )
        if self.reference_index.shape != self.reference_weight.shape:
            raise ValueError(
                "Memory reference indices and weights must share shape"
            )
        if self.reference_index.shape[:3] != self.observation_gain.shape:
            raise ValueError(
                "Memory references and token values must align"
            )
        for name, value in values.items():
            if not torch.isfinite(value.float()).all():
                raise ValueError(
                    f"Memory consolidation '{name}' is not finite"
                )
            if value.dtype != torch.bool and (
                value.min() < 0 or value.max() > 1
            ):
                raise ValueError(
                    f"Memory consolidation '{name}' must lie in [0, 1]"
                )
        if not torch.isfinite(self.reference_weight.float()).all():
            raise ValueError("Memory reference weights are not finite")
        if self.reference_weight.min() < 0:
            raise ValueError("Memory reference weights must be non-negative")

    def as_debug_maps(
        self,
        height: int,
        width: int,
    ) -> Dict[str, torch.Tensor]:
        if height * width != self.observation_gain.shape[-1]:
            raise ValueError("Debug map size must match token count")
        batch, frames, _ = self.observation_gain.shape

        def reshape(value: torch.Tensor) -> torch.Tensor:
            return value.float().reshape(
                batch,
                frames,
                height,
                width,
            )

        return {
            "memory_observation_edit": reshape(
                self.observation_edit_action
            ),
            "memory_observation_precision": reshape(
                self.observation_precision
            ),
            "memory_transport_edit": reshape(
                self.transported_edit_action
            ),
            "memory_transport_precision": reshape(
                self.transported_precision
            ),
            "memory_consolidated_edit": reshape(
                self.consolidated_edit_action
            ),
            "memory_consolidated_precision": reshape(
                self.consolidated_precision
            ),
            "memory_materialized_edit": reshape(
                self.materialized_edit_action
            ),
            "memory_observation_gain": reshape(
                self.observation_gain
            ),
            "memory_reference_valid": reshape(
                self.reference_valid
            ),
        }


class CausalMemoryConsolidator:
    """Transport and consolidate edit memory without fixed write gates."""

    def __init__(self, topk: int = 4, eps: float = 1e-6):
        if topk <= 0:
            raise ValueError("topk must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.topk = topk
        self.eps = eps
        self.previous_features = None
        self.previous_edit_action = None
        self.previous_precision = None

    def _transport(
        self,
        current_features: torch.Tensor,
        reference_features: torch.Tensor,
        reference_edit_action: torch.Tensor,
        reference_precision: torch.Tensor,
    ):
        similarity = torch.einsum(
            "bnd,bmd->bnm",
            current_features,
            reference_features,
        ).clamp(-1.0, 1.0)
        topk = min(self.topk, similarity.shape[-1])
        top_similarity, top_index = similarity.topk(topk, dim=-1)

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
        reference_weight = torch.softmax(
            top_similarity / temperature,
            dim=-1,
        )

        def gather(value: torch.Tensor) -> torch.Tensor:
            return value.unsqueeze(1).expand(
                -1,
                current_features.shape[1],
                -1,
            ).gather(2, top_index)

        transported_edit_action = (
            reference_weight * gather(reference_edit_action)
        ).sum(dim=-1)
        transported_precision = (
            reference_weight * gather(reference_precision)
        ).sum(dim=-1)

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
        similarity_spread = high_similarity - median_similarity
        relative_confidence = (
            (best_similarity - median_similarity)
            / (similarity_spread + self.eps)
        ).clamp(0.0, 1.0)
        relative_confidence = torch.where(
            similarity_spread <= self.eps,
            torch.ones_like(relative_confidence),
            relative_confidence,
        )
        absolute_confidence = (
            0.5 * (best_similarity + 1.0)
        ).clamp(0.0, 1.0)
        match_confidence = torch.sqrt(
            relative_confidence * absolute_confidence
        )
        transported_precision = (
            transported_precision * match_confidence
        ).clamp(0.0, 1.0)
        return (
            transported_edit_action,
            transported_precision,
            top_index,
            reference_weight,
        )

    def __call__(
        self,
        belief: "CausalControlBelief",
        weights: "BeliefKVWeights",
        source_features: torch.Tensor,
    ) -> MemoryConsolidationPlan:
        belief.validate()
        weights.validate()
        if source_features.ndim != 3:
            raise ValueError(
                "source_features must have shape [B,L,D]"
            )

        batch, frames, token_height, token_width = (
            weights.edit_action_map.shape
        )
        tokens_per_frame = token_height * token_width
        expected_tokens = frames * tokens_per_frame
        if source_features.shape[:2] != (batch, expected_tokens):
            raise ValueError(
                "Source features and memory actions must align"
            )
        features = F.normalize(
            source_features.float(),
            dim=-1,
        ).reshape(batch, frames, tokens_per_frame, -1)
        observation_edit_action = weights.edit_action.reshape(
            batch,
            frames,
            tokens_per_frame,
        ).float()

        _, _, height, width = belief.uncertainty.shape
        if height != token_height * 2 or width != token_width * 2:
            raise ValueError(
                "Belief and memory token grids must differ by patch size 2"
            )

        def downsample(value: torch.Tensor) -> torch.Tensor:
            return F.avg_pool2d(
                value.float().reshape(
                    batch * frames,
                    1,
                    height,
                    width,
                ),
                kernel_size=2,
                stride=2,
            ).reshape(batch, frames, tokens_per_frame)

        observation_precision = (
            1.0 - downsample(belief.uncertainty)
        ).clamp(0.0, 1.0)
        visibility = downsample(belief.visibility).clamp(0.0, 1.0)

        transported_actions = []
        transported_precisions = []
        consolidated_actions = []
        consolidated_precisions = []
        observation_gains = []
        reference_indices = []
        reference_weights = []
        reference_validity = []

        reference_features = self.previous_features
        reference_edit_action = self.previous_edit_action
        reference_precision = self.previous_precision
        for frame_index in range(frames):
            current_features = features[:, frame_index]
            current_action = observation_edit_action[:, frame_index]
            current_precision = observation_precision[:, frame_index]
            current_visibility = visibility[:, frame_index]

            if (
                reference_features is None
                or reference_edit_action is None
                or reference_precision is None
            ):
                topk = min(self.topk, tokens_per_frame)
                transported_action = torch.zeros_like(current_action)
                transported_precision = torch.zeros_like(
                    current_precision
                )
                reference_index = torch.zeros(
                    batch,
                    tokens_per_frame,
                    topk,
                    dtype=torch.long,
                    device=current_action.device,
                )
                reference_weight = torch.zeros(
                    batch,
                    tokens_per_frame,
                    topk,
                    dtype=torch.float32,
                    device=current_action.device,
                )
            else:
                (
                    transported_action,
                    transported_precision,
                    reference_index,
                    reference_weight,
                ) = self._transport(
                    current_features,
                    reference_features,
                    reference_edit_action,
                    reference_precision,
                )
                transported_action = (
                    transported_action * current_visibility
                )

            total_precision = (
                current_precision + transported_precision
            )
            has_reference = transported_precision > self.eps
            observation_gain = torch.where(
                has_reference,
                current_precision
                / total_precision.clamp_min(self.eps),
                torch.ones_like(current_precision),
            )
            consolidated_action = (
                transported_action
                + observation_gain
                * (current_action - transported_action)
            ).clamp(0.0, 1.0)
            consolidated_precision = (
                1.0
                - (1.0 - current_precision)
                * (1.0 - transported_precision)
            ).clamp(0.0, 1.0)

            transported_actions.append(transported_action)
            transported_precisions.append(transported_precision)
            consolidated_actions.append(consolidated_action)
            consolidated_precisions.append(consolidated_precision)
            observation_gains.append(observation_gain)
            reference_indices.append(reference_index)
            reference_weights.append(reference_weight)
            reference_validity.append(has_reference)

            reference_features = current_features
            reference_edit_action = consolidated_action
            reference_precision = consolidated_precision

        self.previous_features = reference_features.detach()
        self.previous_edit_action = reference_edit_action.detach()
        self.previous_precision = reference_precision.detach()

        plan = MemoryConsolidationPlan(
            observation_edit_action=observation_edit_action,
            observation_precision=observation_precision,
            transported_edit_action=torch.stack(
                transported_actions,
                dim=1,
            ),
            transported_precision=torch.stack(
                transported_precisions,
                dim=1,
            ),
            consolidated_edit_action=torch.stack(
                consolidated_actions,
                dim=1,
            ),
            consolidated_precision=torch.stack(
                consolidated_precisions,
                dim=1,
            ),
            materialized_edit_action=(
                torch.stack(consolidated_actions, dim=1)
                * torch.stack(consolidated_precisions, dim=1)
            ).clamp(0.0, 1.0),
            observation_gain=torch.stack(
                observation_gains,
                dim=1,
            ),
            reference_index=torch.stack(
                reference_indices,
                dim=1,
            ),
            reference_weight=torch.stack(
                reference_weights,
                dim=1,
            ),
            reference_valid=torch.stack(
                reference_validity,
                dim=1,
            ),
        )
        plan.validate()
        return plan
