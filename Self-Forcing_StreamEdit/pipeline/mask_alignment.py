"""Temporal alignment helpers for pixel-space controls and causal latents."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CausalHandEvidence:
    """Three non-interchangeable views of a moving hand mask.

    ``union`` answers whether the hand visited a location and is therefore a
    useful, high-recall interaction/proximity cue. ``occupancy`` measures how
    much of the causal pixel-frame group was occupied and is used as a soft
    contact probability. ``persistent`` is the conservative subset occupied
    throughout the group and is the only view suitable for hard owner
    exclusion.
    """

    union: torch.Tensor
    occupancy: torch.Tensor
    persistent: torch.Tensor

    def validate(self) -> None:
        shapes = {
            tuple(self.union.shape),
            tuple(self.occupancy.shape),
            tuple(self.persistent.shape),
        }
        if len(shapes) != 1 or self.union.ndim != 3:
            raise ValueError(
                "Causal hand evidence must share shape [T,H,W]"
            )
        if self.union.dtype != torch.bool:
            raise ValueError("Hand union must be boolean")
        if self.persistent.dtype != torch.bool:
            raise ValueError("Persistent hand evidence must be boolean")
        if (self.occupancy < 0).any() or (self.occupancy > 1).any():
            raise ValueError("Hand occupancy must lie in [0, 1]")
        if (self.persistent & ~self.union).any():
            raise ValueError("Persistent hand support must be inside union")


def causal_vae_frame_groups(
    pixel_frames: int,
    latent_frames: int,
    *,
    temporal_stride: int | None = None,
) -> list[tuple[int, int]]:
    """Return pixel-frame intervals represented by causal VAE latents.

    Wan's temporal encoder emits one latent from the first pixel frame and
    then one latent for each subsequent ``temporal_stride``-frame group.
    Uniform resampling is incorrect because it shifts every control mask after
    the first latent relative to the encoded video.
    """
    if pixel_frames <= 0 or latent_frames <= 0:
        raise ValueError(
            "pixel_frames and latent_frames must be positive"
        )
    if temporal_stride is None:
        if latent_frames == 1:
            if pixel_frames != 1:
                raise ValueError(
                    "Cannot infer a causal temporal stride from more than "
                    "one pixel frame and one latent frame"
                )
            temporal_stride = 1
        else:
            remaining_frames = pixel_frames - 1
            remaining_latents = latent_frames - 1
            if remaining_frames % remaining_latents != 0:
                raise ValueError(
                    "Pixel and latent frame counts do not define an exact "
                    "causal temporal stride"
                )
            temporal_stride = remaining_frames // remaining_latents
    if temporal_stride <= 0:
        raise ValueError(
            "temporal_stride must be positive"
        )
    expected_latent_frames = 1 + (pixel_frames - 1) // temporal_stride
    if expected_latent_frames != latent_frames:
        raise ValueError(
            "Pixel and latent frame counts do not match the causal VAE "
            f"mapping: pixel_frames={pixel_frames}, "
            f"latent_frames={latent_frames}, "
            f"temporal_stride={temporal_stride}, "
            f"expected_latent_frames={expected_latent_frames}"
        )
    if pixel_frames == 1:
        return [(0, 1)]
    if (pixel_frames - 1) % temporal_stride != 0:
        raise ValueError(
            "Pixel sequence must contain one causal first frame followed "
            "by complete temporal-stride groups"
        )
    return [(0, 1)] + [
        (
            1 + temporal_stride * (index - 1),
            1 + temporal_stride * index,
        )
        for index in range(1, latent_frames)
    ]


def project_hand_evidence_to_causal_latents(
    hand_pixel_mask: torch.Tensor,
    *,
    latent_frames: int,
    latent_spatial_shape: tuple[int, int],
    persistent_occupancy: float = 1.0,
) -> CausalHandEvidence:
    """Project a hand matte without turning hand motion into occlusion.

    The first causal latent represents pixel frame zero. Every later latent
    represents a temporal group. Reducing each group with ``max`` is valid for
    proximity, but invalid for exclusion: a hand that merely sweeps across an
    object would erase the complete trajectory. This function preserves all
    three statistics so downstream components can use the right semantics.

    This function consumes only the supplied hand mask; no object annotation
    or object trajectory is involved.
    """
    if hand_pixel_mask.ndim != 4 or hand_pixel_mask.shape[1] != 1:
        raise ValueError(
            "hand_pixel_mask must have shape [T,1,H,W]"
        )
    if not 0.0 < persistent_occupancy <= 1.0:
        raise ValueError(
            "persistent_occupancy must lie in (0, 1]"
        )
    latent_height, latent_width = latent_spatial_shape
    if latent_height <= 0 or latent_width <= 0:
        raise ValueError("latent_spatial_shape must be positive")

    pixel = hand_pixel_mask.detach().float().clamp(0.0, 1.0)
    temporal_groups = causal_vae_frame_groups(
        pixel.shape[0], latent_frames
    )
    union_groups = torch.stack([
        pixel[left:right].amax(dim=0)
        for left, right in temporal_groups
    ])
    occupancy_groups = torch.stack([
        pixel[left:right].mean(dim=0)
        for left, right in temporal_groups
    ])

    # Union and hard support stay conservative under spatial resampling. The
    # fractional occupancy uses area resampling so sub-latent hand motion is
    # retained as probability rather than rounded into a hard exclusion.
    union = F.interpolate(
        union_groups,
        size=(latent_height, latent_width),
        mode="nearest",
    ).squeeze(1) > 0.5
    occupancy = F.interpolate(
        occupancy_groups,
        size=(latent_height, latent_width),
        mode="area",
    ).squeeze(1).clamp(0.0, 1.0)
    persistent_groups = occupancy_groups >= persistent_occupancy
    persistent = F.interpolate(
        persistent_groups.float(),
        size=(latent_height, latent_width),
        mode="area",
    ).squeeze(1) >= 0.5
    persistent &= union

    result = CausalHandEvidence(
        union=union,
        occupancy=occupancy,
        persistent=persistent,
    )
    result.validate()
    return result


def project_visible_owner_to_causal_latents(
    source_owner_pixel_mask: torch.Tensor,
    hand_pixel_mask: torch.Tensor,
    *,
    latent_frames: int,
    latent_spatial_shape: tuple[int, int],
    min_latent_coverage: float = 0.0,
) -> torch.Tensor:
    """Remove hand occlusion per pixel frame before causal pooling.

    Temporal max pooling does not commute with mask subtraction.  In
    particular, ``max_t(object) & ~max_t(hand)`` removes every location
    visited by a moving hand from the whole causal frame group.  Ego videos
    therefore need ``max_t(object & ~hand)`` so a surface remains owned when
    it is visible in any represented pixel frame.

    Args:
        source_owner_pixel_mask: Boolean-like ``[T, 1, H, W]`` object mask.
        hand_pixel_mask: Boolean-like ``[T, 1, H, W]`` hand mask.
        latent_frames: Number of causal VAE frames.
        latent_spatial_shape: Spatial ``(height, width)`` of the latent mask.
        min_latent_coverage: Optional per-frame coverage floor.

    Returns:
        Boolean visible-owner mask with shape ``[T_latent, H_latent, W_latent]``.
    """
    if source_owner_pixel_mask.ndim != 4:
        raise ValueError(
            "source_owner_pixel_mask must have shape [T, 1, H, W]"
        )
    if hand_pixel_mask.shape != source_owner_pixel_mask.shape:
        raise ValueError(
            "source owner and hand pixel masks must share [T, 1, H, W]"
        )
    if source_owner_pixel_mask.shape[1] != 1:
        raise ValueError(
            "source owner pixel mask must have a singleton channel"
        )
    latent_height, latent_width = latent_spatial_shape
    if latent_height <= 0 or latent_width <= 0:
        raise ValueError("latent_spatial_shape must be positive")
    if not 0.0 <= min_latent_coverage <= 1.0:
        raise ValueError("min_latent_coverage must lie in [0, 1]")

    visible_pixel_owner = (
        source_owner_pixel_mask.detach().bool()
        & ~hand_pixel_mask.detach().bool()
    )
    temporal_groups = causal_vae_frame_groups(
        visible_pixel_owner.shape[0],
        latent_frames,
    )
    visible_groups = torch.stack([
        visible_pixel_owner[left:right].amax(dim=0)
        for left, right in temporal_groups
    ]).float()
    latent_owner = F.interpolate(
        visible_groups,
        size=(latent_height, latent_width),
        mode="nearest",
    ) > 0.5
    if min_latent_coverage > 0.0:
        frame_coverage = latent_owner.float().mean(dim=(1, 2, 3))
        latent_owner[frame_coverage < min_latent_coverage] = False
    return latent_owner.squeeze(1)
