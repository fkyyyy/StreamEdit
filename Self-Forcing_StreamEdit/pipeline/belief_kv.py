from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .control_belief import CausalControlBelief


@dataclass(frozen=True)
class BeliefKVWeights:
    """Continuous target/source memory responsibilities on the token grid."""

    edit: torch.Tensor
    preserve: torch.Tensor
    edit_map: torch.Tensor
    preserve_map: torch.Tensor
    conflict_map: torch.Tensor

    def validate(self) -> None:
        if self.edit.ndim != 2 or self.preserve.ndim != 2:
            raise ValueError("Belief KV token weights must have shape [B,L]")
        if self.edit.shape != self.preserve.shape:
            raise ValueError("Edit and preserve KV weights must share shape")
        map_shapes = {
            tuple(self.edit_map.shape),
            tuple(self.preserve_map.shape),
            tuple(self.conflict_map.shape),
        }
        if len(map_shapes) != 1 or self.edit_map.ndim != 4:
            raise ValueError(
                "Belief KV maps must share shape [B,T,H,W]"
            )
        if self.edit_map.shape[0] != self.edit.shape[0]:
            raise ValueError("Belief KV maps and tokens must share batch size")
        if self.edit_map.numel() // self.edit_map.shape[0] != (
            self.edit.shape[1]
        ):
            raise ValueError("Belief KV map size must match token length")
        for name, value in (
            ("edit", self.edit),
            ("preserve", self.preserve),
            ("conflict", self.conflict_map),
        ):
            if not torch.isfinite(value.float()).all():
                raise ValueError(f"Belief KV {name} weights are not finite")
            if value.min() < 0 or value.max() > 1:
                raise ValueError(
                    f"Belief KV {name} weights must lie in [0, 1]"
                )


def build_belief_kv_weights(
    belief: "CausalControlBelief",
    expected_token_length: int,
    eps: float = 1e-6,
) -> BeliefKVWeights:
    """Project non-exclusive edit/preserve beliefs to transformer tokens."""
    if expected_token_length <= 0:
        raise ValueError("expected_token_length must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")
    belief.validate()

    batch, frames, height, width = belief.edit_belief.shape
    if height % 2 or width % 2:
        raise ValueError(
            "Belief height and width must be divisible by the patch size"
        )
    token_height = height // 2
    token_width = width // 2
    if frames * token_height * token_width != expected_token_length:
        raise ValueError(
            "Belief grid and KV sequence imply different token counts: "
            f"{frames * token_height * token_width} != "
            f"{expected_token_length}"
        )

    def downsample(value: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            value.float().reshape(batch * frames, 1, height, width),
            kernel_size=2,
            stride=2,
        ).reshape(batch, frames, token_height, token_width).clamp(0.0, 1.0)

    edit_map = downsample(belief.edit_belief)
    preserve_map = downsample(belief.preserve_belief)
    no_evidence = (edit_map + preserve_map) <= eps
    preserve_map = torch.where(
        no_evidence,
        torch.ones_like(preserve_map),
        preserve_map,
    )
    weights = BeliefKVWeights(
        edit=edit_map.reshape(batch, -1),
        preserve=preserve_map.reshape(batch, -1),
        edit_map=edit_map,
        preserve_map=preserve_map,
        conflict_map=(edit_map * preserve_map).clamp(0.0, 1.0),
    )
    weights.validate()
    return weights
