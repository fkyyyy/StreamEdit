from dataclasses import dataclass
from typing import Dict, Mapping

import torch
import torch.nn.functional as F


def _resize_map(
    value: torch.Tensor,
    spatial_size,
) -> torch.Tensor:
    if value.ndim != 4:
        raise ValueError(
            "Control evidence must have shape [B,T,H,W], got "
            f"{tuple(value.shape)}"
        )
    batch, frames, height, width = value.shape
    resized = F.interpolate(
        value.float().reshape(batch * frames, 1, height, width),
        size=spatial_size,
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(batch, frames, *spatial_size)


def _expand_frame_value(
    value: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if value.ndim != 4 or value.shape[-2:] != (1, 1):
        raise ValueError(
            "Frame evidence must have shape [B,T,1,1], got "
            f"{tuple(value.shape)}"
        )
    if value.shape[:2] != reference.shape[:2]:
        raise ValueError("Frame evidence and reference must share [B,T]")
    return value.float().expand_as(reference)


def _probabilistic_union(*values: torch.Tensor) -> torch.Tensor:
    complement = torch.ones_like(values[0])
    for value in values:
        complement = complement * (1.0 - value.clamp(0.0, 1.0))
    return (1.0 - complement).clamp(0.0, 1.0)


@dataclass(frozen=True)
class CausalControlBelief:
    """Non-exclusive edit and preservation responsibilities."""

    edit_belief: torch.Tensor
    preserve_belief: torch.Tensor
    edit_precision: torch.Tensor
    preserve_precision: torch.Tensor
    visibility: torch.Tensor
    uncertainty: torch.Tensor
    conflict: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "edit_belief": self.edit_belief,
            "preserve_belief": self.preserve_belief,
            "edit_precision": self.edit_precision,
            "preserve_precision": self.preserve_precision,
            "visibility": self.visibility,
            "uncertainty": self.uncertainty,
            "conflict": self.conflict,
        }

    def validate(self) -> None:
        values = self.as_dict()
        shapes = {tuple(value.shape) for value in values.values()}
        if len(shapes) != 1:
            raise ValueError(
                f"Control belief maps must share one shape, got {shapes}"
            )
        if next(iter(values.values())).ndim != 4:
            raise ValueError("Control belief maps must have shape [B,T,H,W]")
        for name, value in values.items():
            if not torch.isfinite(value.float()).all():
                raise ValueError(f"Control belief '{name}' is not finite")
            if value.min() < 0 or value.max() > 1:
                raise ValueError(
                    f"Control belief '{name}' must lie in [0, 1]"
                )


class CausalControlBeliefBuilder:
    """Build causal edit/preserve beliefs from hand-only observations."""

    def __init__(
        self,
        precision_floor: float = 1e-3,
        eps: float = 1e-6,
    ):
        if not 0.0 < precision_floor <= 1.0:
            raise ValueError("precision_floor must lie in (0, 1]")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.precision_floor = precision_floor
        self.eps = eps

    @staticmethod
    def _required(
        debug: Mapping[str, torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        value = debug.get(name)
        if value is None:
            raise ValueError(f"Missing control evidence: {name}")
        return value

    def __call__(
        self,
        debug: Mapping[str, torch.Tensor],
        hand_mask: torch.Tensor,
    ) -> CausalControlBelief:
        if hand_mask.ndim != 4:
            raise ValueError("hand_mask must have shape [B,T,H,W]")

        object_posterior = self._required(
            debug,
            "object_posterior",
        ).float()
        source_attention = self._required(
            debug,
            "source_attention",
        ).float()
        temporal_confidence = self._required(
            debug,
            "temporal_confidence",
        ).float()
        object_visible = self._required(
            debug,
            "object_visible",
        ).float()
        hand_proximity = self._required(
            debug,
            "hand_proximity",
        ).float()
        attention_reliability = self._required(
            debug,
            "adaptive_attention_reliability",
        ).float()

        token_shape = object_posterior.shape
        for name, value in (
            ("source_attention", source_attention),
            ("temporal_confidence", temporal_confidence),
            ("hand_proximity", hand_proximity),
        ):
            if value.shape != token_shape:
                raise ValueError(
                    f"{name} must match object_posterior, got "
                    f"{tuple(value.shape)} and {tuple(token_shape)}"
                )
        if hand_mask.shape[:2] != token_shape[:2]:
            raise ValueError(
                "hand_mask and object posterior must share [B,T]"
            )

        spatial_size = hand_mask.shape[-2:]
        edit_belief = _resize_map(
            object_posterior.clamp(0.0, 1.0),
            spatial_size,
        )
        source_attention = _resize_map(
            source_attention.clamp(0.0, 1.0),
            spatial_size,
        )
        temporal_confidence = _resize_map(
            temporal_confidence.clamp(0.0, 1.0),
            spatial_size,
        )
        hand_proximity = _resize_map(
            hand_proximity.clamp(0.0, 1.0),
            spatial_size,
        )
        visibility = _expand_frame_value(
            object_visible.clamp(0.0, 1.0),
            edit_belief,
        )
        attention_reliability = _expand_frame_value(
            attention_reliability.clamp(0.0, 1.0),
            edit_belief,
        )
        edit_belief = edit_belief * visibility

        hand_probability = hand_mask.float().clamp(0.0, 1.0)
        batch, frames, height, width = hand_probability.shape
        hand_band = F.max_pool2d(
            hand_probability.reshape(batch * frames, 1, height, width),
            kernel_size=3,
            stride=1,
            padding=1,
        ).reshape_as(hand_probability)

        background_responsibility = (
            (1.0 - source_attention)
            * (1.0 - hand_proximity)
        )
        inactive_responsibility = (
            (1.0 - edit_belief)
            * (1.0 - hand_proximity)
        )
        contact_responsibility = edit_belief * hand_band
        invisible_responsibility = 1.0 - visibility
        preserve_belief = _probabilistic_union(
            hand_band,
            background_responsibility,
            inactive_responsibility,
            contact_responsibility,
            invisible_responsibility,
        )

        local_edit_evidence = torch.maximum(
            source_attention,
            temporal_confidence,
        )
        edit_precision = torch.sqrt(
            (
                attention_reliability
                * local_edit_evidence
            ).clamp_min(0.0)
        )

        field_score = debug.get("field_score")
        field_reliability = debug.get("adaptive_field_reliability")
        if field_score is None or field_reliability is None:
            field_stability = torch.zeros_like(edit_belief)
            field_consistency = torch.ones_like(edit_belief)
        else:
            field_score = _resize_map(
                field_score.float().clamp(0.0, 1.0),
                spatial_size,
            )
            field_reliability = _expand_frame_value(
                field_reliability.float().clamp(0.0, 1.0),
                edit_belief,
            )
            field_stability = (
                field_reliability * (1.0 - field_score)
            )
            field_consistency = (
                1.0
                - field_reliability
                * (1.0 - field_score)
            )
        edit_precision = (
            edit_precision * field_consistency
        ).clamp(self.precision_floor, 1.0)

        background_precision = (
            attention_reliability * (1.0 - source_attention)
        )
        inactive_precision = (
            attention_reliability * (1.0 - edit_belief)
        )
        preserve_evidence = torch.maximum(
            hand_band,
            torch.maximum(
                inactive_precision,
                torch.maximum(
                    background_precision,
                    field_stability,
                ),
            ),
        )
        preserve_precision = preserve_evidence.clamp(
            self.precision_floor,
            1.0,
        )

        conflict = (edit_belief * preserve_belief).clamp(0.0, 1.0)
        responsibility = (
            edit_belief + preserve_belief
        ).clamp_min(self.eps)
        uncertainty = (
            edit_belief * (1.0 - edit_precision)
            + preserve_belief * (1.0 - preserve_precision)
        ) / responsibility
        belief = CausalControlBelief(
            edit_belief=edit_belief.float(),
            preserve_belief=preserve_belief.float(),
            edit_precision=edit_precision.float(),
            preserve_precision=preserve_precision.float(),
            visibility=visibility.float(),
            uncertainty=uncertainty.clamp(0.0, 1.0).float(),
            conflict=conflict.float(),
        )
        belief.validate()
        return belief
