"""Geometry primitives for dense optical flow.

All flow tensors use pixel displacement in ``(dx, dy)`` order. A forward
flow ``F_ab`` is defined on frame A and maps ``x_a`` to
``x_b = x_a + F_ab(x_a)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_flow(flow: torch.Tensor, name: str = "flow") -> None:
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(
            f"{name} must have shape [B,2,H,W], got {tuple(flow.shape)}"
        )
    if not torch.isfinite(flow.float()).all():
        raise ValueError(f"{name} contains non-finite values")


def flow_coordinates(flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return displaced pixel coordinates and their validity mask."""

    _validate_flow(flow)
    batch, _, height, width = flow.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    base = torch.stack((x, y), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
    coordinates = base + flow
    valid = (
        (coordinates[:, 0] >= 0.0)
        & (coordinates[:, 0] <= max(width - 1, 0))
        & (coordinates[:, 1] >= 0.0)
        & (coordinates[:, 1] <= max(height - 1, 0))
    )
    return coordinates, valid.unsqueeze(1)


def _coordinates_to_grid(coordinates: torch.Tensor) -> torch.Tensor:
    _, _, height, width = coordinates.shape
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    if width > 1:
        x = 2.0 * x / float(width - 1) - 1.0
    else:
        x = torch.zeros_like(x)
    if height > 1:
        y = 2.0 * y / float(height - 1) - 1.0
    else:
        y = torch.zeros_like(y)
    return torch.stack((x, y), dim=-1)


def sample_with_flow(
    value: torch.Tensor,
    flow: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``value`` at ``x + flow(x)``.

    This is the primitive needed both for forward/backward consistency and
    for backward-warping a previous-frame ownership map into a new frame.
    """

    _validate_flow(flow)
    if value.ndim != 4:
        raise ValueError(
            f"value must have shape [B,C,H,W], got {tuple(value.shape)}"
        )
    if value.shape[0] != flow.shape[0] or value.shape[-2:] != flow.shape[-2:]:
        raise ValueError(
            "value and flow must share batch and spatial dimensions"
        )
    coordinates, valid = flow_coordinates(flow)
    sampled = F.grid_sample(
        value,
        _coordinates_to_grid(coordinates),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )
    return sampled, valid


def warp_with_backward_flow(
    previous: torch.Tensor,
    backward_flow: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull-warp a previous-frame map onto the current frame.

    ``backward_flow`` must map current coordinates to the previous frame.
    """

    return sample_with_flow(previous, backward_flow)


def forward_splat(
    value: torch.Tensor,
    forward_flow: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Push ``value`` along a forward flow using bilinear splatting.

    Unlike a backward warp this primitive does not require an inverse flow.
    Multiple source pixels are accumulated and normalized, which makes it a
    useful complementary estimate around disocclusions and thin boundaries.
    The returned coverage is the accumulated (unnormalized) splat weight.
    """

    _validate_flow(forward_flow, "forward_flow")
    if value.ndim != 4:
        raise ValueError(
            f"value must have shape [B,C,H,W], got {tuple(value.shape)}"
        )
    if (
        value.shape[0] != forward_flow.shape[0]
        or value.shape[-2:] != forward_flow.shape[-2:]
    ):
        raise ValueError(
            "value and forward_flow must share batch and spatial dimensions"
        )
    if weight is None:
        weight = torch.ones_like(value[:, :1], dtype=torch.float32)
    if weight.shape != value[:, :1].shape:
        raise ValueError(
            "weight must have shape [B,1,H,W] matching value"
        )

    batch, channels, height, width = value.shape
    coordinates, _ = flow_coordinates(forward_flow.float())
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x1 = x0 + 1.0
    y1 = y0 + 1.0

    source = value.float().reshape(batch, channels, -1)
    source_weight = weight.float().clamp_min(0.0).reshape(batch, 1, -1)
    numerator = torch.zeros(
        batch, channels, height * width,
        device=value.device, dtype=torch.float32,
    )
    denominator = torch.zeros(
        batch, 1, height * width,
        device=value.device, dtype=torch.float32,
    )

    for target_x, target_y, bilinear in (
        (x0, y0, (x1 - x) * (y1 - y)),
        (x1, y0, (x - x0) * (y1 - y)),
        (x0, y1, (x1 - x) * (y - y0)),
        (x1, y1, (x - x0) * (y - y0)),
    ):
        valid = (
            (target_x >= 0) & (target_x < width)
            & (target_y >= 0) & (target_y < height)
        )
        contribution = (
            bilinear * valid.float()
        ).reshape(batch, 1, -1) * source_weight
        index = (
            target_y.clamp(0, height - 1).long() * width
            + target_x.clamp(0, width - 1).long()
        ).reshape(batch, 1, -1)
        denominator.scatter_add_(2, index, contribution)
        numerator.scatter_add_(
            2, index.expand(-1, channels, -1), source * contribution
        )

    coverage = denominator.reshape(batch, 1, height, width)
    splatted = (
        numerator / denominator.clamp_min(1e-6)
    ).reshape(batch, channels, height, width)
    splatted = torch.where(coverage > 0.0, splatted, torch.zeros_like(splatted))
    return splatted, coverage


def compose_forward_flow(
    flow_ab: torch.Tensor,
    flow_bc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose A->B and B->C flows into an A->C flow."""

    _validate_flow(flow_ab, "flow_ab")
    _validate_flow(flow_bc, "flow_bc")
    if flow_ab.shape != flow_bc.shape:
        raise ValueError("Flows to compose must share shape")
    sampled_bc, valid = sample_with_flow(flow_bc, flow_ab)
    return flow_ab + sampled_bc, valid


def resize_flow(
    flow: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    """Resize dense flow while preserving displacement in pixel units."""

    _validate_flow(flow)
    output_height, output_width = size
    if output_height <= 0 or output_width <= 0:
        raise ValueError("Flow output size must be positive")
    input_height, input_width = flow.shape[-2:]
    resized = F.interpolate(
        flow.float(),
        size=size,
        mode="bilinear",
        align_corners=True,
    )
    resized[:, 0] *= output_width / float(input_width)
    resized[:, 1] *= output_height / float(input_height)
    return resized


def forward_backward_confidence(
    forward_flow: torch.Tensor,
    backward_flow: torch.Tensor,
    *,
    alpha: float = 0.01,
    beta: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a soft confidence and occlusion from flow consistency.

    For a forward flow A->B, the backward flow B->A is sampled at the
    forward endpoint. The consistency residual should be zero for a valid
    correspondence. ``occlusion`` includes out-of-frame endpoints.
    """

    _validate_flow(forward_flow, "forward_flow")
    _validate_flow(backward_flow, "backward_flow")
    if forward_flow.shape != backward_flow.shape:
        raise ValueError("Forward and backward flows must share shape")
    if alpha < 0.0 or beta <= 0.0:
        raise ValueError("alpha must be non-negative and beta positive")

    sampled_backward, valid = sample_with_flow(
        backward_flow.float(), forward_flow.float()
    )
    residual = forward_flow.float() + sampled_backward
    error_sq = residual.square().sum(dim=1, keepdim=True)
    motion_sq = (
        forward_flow.float().square().sum(dim=1, keepdim=True)
        + sampled_backward.square().sum(dim=1, keepdim=True)
    )
    threshold = alpha * motion_sq + beta
    confidence = torch.exp(-error_sq / threshold.clamp_min(1e-6))
    confidence = confidence * valid.float()
    occlusion = (~valid) | (error_sq > threshold)
    error = torch.sqrt(error_sq.clamp_min(0.0))
    return confidence.clamp(0.0, 1.0), occlusion, error
