"""Camera-compensated source-flow evidence for hand-centric token roles.

The optical flow is computed only on the clean source video.  It is not an
object detector: a transported, hand-conditioned causal owner supplies the
object hypothesis, while bidirectional flow confidence and hand-relative
motion say how strongly that hypothesis should be trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..causal_ownership import CausalObjectOwnership


@dataclass(frozen=True)
class FlowRoleEvidence:
    object_likelihood: torch.Tensor
    background_likelihood: torch.Tensor
    boundary_likelihood: torch.Tensor
    unknown_likelihood: torch.Tensor
    cycle_confidence: torch.Tensor
    transport_support: torch.Tensor

    def validate(self) -> None:
        values = {
            "object_likelihood": self.object_likelihood,
            "background_likelihood": self.background_likelihood,
            "boundary_likelihood": self.boundary_likelihood,
            "unknown_likelihood": self.unknown_likelihood,
            "cycle_confidence": self.cycle_confidence,
            "transport_support": self.transport_support,
        }
        shapes = {tuple(value.shape) for value in values.values()}
        if len(shapes) != 1 or self.object_likelihood.ndim != 4:
            raise ValueError(
                "Flow role evidence must share shape [B,T,H,W]"
            )
        for name, value in values.items():
            if not torch.isfinite(value.float()).all():
                raise ValueError(f"Flow role evidence '{name}' is not finite")
            if value.min() < 0 or value.max() > 1:
                raise ValueError(
                    f"Flow role evidence '{name}' must lie in [0, 1]"
                )

    def as_debug_maps(self) -> dict[str, torch.Tensor]:
        return {
            "flow_object_likelihood": self.object_likelihood,
            "flow_background_likelihood": self.background_likelihood,
            "flow_boundary_likelihood": self.boundary_likelihood,
            "flow_unknown_likelihood": self.unknown_likelihood,
            "flow_cycle_confidence": self.cycle_confidence,
            "flow_transport_support": self.transport_support,
        }


def _reshape(value: torch.Tensor, shape: tuple[int, int, int, int]) -> torch.Tensor:
    if value.shape != (shape[0], shape[1] * shape[2] * shape[3]):
        raise ValueError(
            "Causal motion evidence does not align with the role grid"
        )
    return value.detach().float().reshape(shape)


@torch.no_grad()
def build_flow_role_evidence(
    ownership: CausalObjectOwnership,
    *,
    shape: tuple[int, int, int, int],
    hand_exclusion: torch.Tensor,
    min_transport_weight: float = 0.05,
) -> FlowRoleEvidence:
    """Factor source flow into object/background/boundary/unknown evidence.

    Flow magnitude alone is intentionally never used as object evidence.  In
    an egocentric video the background can move strongly with the camera.  The
    object likelihood therefore requires the transported causal owner and is
    calibrated by forward/backward consistency plus motion relative to the
    camera-compensated hand motion.
    """
    ownership.validate()
    if not 0.0 <= float(min_transport_weight) <= 1.0:
        raise ValueError("min_transport_weight must lie in [0, 1]")
    if tuple(hand_exclusion.shape) != shape:
        raise ValueError("hand_exclusion must align with the role grid")

    owner = _reshape(ownership.owner_weight, shape).clamp(0.0, 1.0)
    transported = _reshape(
        ownership.transported_weight, shape
    ).clamp(0.0, 1.0)
    confidence = _reshape(
        ownership.match_confidence, shape
    ).clamp(0.0, 1.0)
    diagnostics = ownership.diagnostics or {}
    affinity_value = diagnostics.get("motion_hand_affinity")
    affinity = (
        torch.full_like(owner, 0.5)
        if affinity_value is None
        else _reshape(affinity_value, shape).clamp(0.0, 1.0)
    )
    hand = hand_exclusion.detach().float().clamp(0.0, 1.0)
    non_hand = 1.0 - hand

    # The owner is the hypothesis; cycle confidence and hand-relative flow are
    # independent reliability observations.  A 0.5 affinity is deliberately
    # neutral for the first frame or a degenerate motion estimate.
    motion_reliability = (0.5 + 0.5 * affinity) * confidence
    object_likelihood = (
        torch.sqrt((owner * motion_reliability).clamp(0.0, 1.0)) * non_hand
    )

    flat_owner = owner.reshape(shape[0] * shape[1], 1, shape[2], shape[3])
    dilated = F.max_pool2d(flat_owner, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-flat_owner, kernel_size=3, stride=1, padding=1)
    boundary_likelihood = (
        (dilated - eroded).clamp(0.0, 1.0).reshape(shape)
        * confidence
        * non_hand
    )

    # Camera-consistent, confidently non-owned cells are positive background
    # evidence.  Hand-relative affinity only modulates this term; it cannot
    # independently turn motion into foreground or background.
    background_likelihood = (
        (1.0 - owner)
        * confidence
        * (0.5 + 0.5 * (1.0 - affinity))
        * non_hand
    ).clamp(0.0, 1.0)
    unknown_likelihood = (
        (1.0 - confidence) * non_hand
        + hand * (1.0 - object_likelihood)
    ).clamp(0.0, 1.0)
    transport_support = (
        (transported >= float(min_transport_weight)).float()
        * confidence
        * non_hand
    ).clamp(0.0, 1.0)

    result = FlowRoleEvidence(
        object_likelihood=object_likelihood,
        background_likelihood=background_likelihood,
        boundary_likelihood=boundary_likelihood,
        unknown_likelihood=unknown_likelihood,
        cycle_confidence=confidence,
        transport_support=transport_support,
    )
    result.validate()
    return result
