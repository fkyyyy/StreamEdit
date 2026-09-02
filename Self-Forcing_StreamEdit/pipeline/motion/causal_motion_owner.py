"""Hand-conditioned geometry ownership transported by real optical flow.

This module deliberately keeps geometry state separate from appearance-memory
writes.  The only external spatial input is a hand mask; object observations
come from the existing source-text/hand role inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from ..causal_ownership import CausalObjectOwnership, CausalOwnershipState
from ..mask_alignment import causal_vae_frame_groups
from .flow_geometry import (
    compose_forward_flow,
    forward_splat,
    resize_flow,
    sample_with_flow,
    warp_with_backward_flow,
)


@dataclass(frozen=True)
class LatentMotionTransition:
    forward_flow: torch.Tensor
    backward_flow: torch.Tensor
    forward_confidence: torch.Tensor
    backward_confidence: torch.Tensor


class SourceFlowCache:
    """Read-only adjacent RGB flow with causal-latent composition."""

    REQUIRED_KEYS = {
        "forward_flow",
        "backward_flow",
        "forward_confidence",
        "backward_confidence",
    }

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        latent_pixel_indices: Sequence[int],
    ) -> None:
        missing = self.REQUIRED_KEYS.difference(tensors)
        if missing:
            raise ValueError(
                f"Source flow cache is missing keys: {sorted(missing)}"
            )
        self.tensors = {
            name: tensors[name].detach().cpu() for name in self.REQUIRED_KEYS
        }
        flow_shape = self.tensors["forward_flow"].shape
        if len(flow_shape) != 4 or flow_shape[1] != 2:
            raise ValueError(
                "Cached flows must have shape [T-1,2,H,W]"
            )
        for name in ("backward_flow",):
            if self.tensors[name].shape != flow_shape:
                raise ValueError(
                    f"Cached {name} must match forward flow shape"
                )
        confidence_shape = (flow_shape[0], 1, *flow_shape[-2:])
        for name in ("forward_confidence", "backward_confidence"):
            if tuple(self.tensors[name].shape) != confidence_shape:
                raise ValueError(
                    f"Cached {name} must have shape {confidence_shape}"
                )
        self.latent_pixel_indices = tuple(int(v) for v in latent_pixel_indices)
        if not self.latent_pixel_indices:
            raise ValueError("latent_pixel_indices must not be empty")
        if self.latent_pixel_indices[0] != 0:
            raise ValueError("The first causal latent must represent RGB frame 0")
        if any(
            right <= left
            for left, right in zip(
                self.latent_pixel_indices, self.latent_pixel_indices[1:]
            )
        ):
            raise ValueError("latent_pixel_indices must be strictly increasing")
        if self.latent_pixel_indices[-1] > flow_shape[0]:
            raise ValueError(
                "Latent/RGB mapping exceeds the source flow cache"
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        latent_frame_count: int,
        pixel_frame_count: int | None = None,
        source_video_path: str | Path | None = None,
    ) -> "SourceFlowCache":
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Source flow cache not found: {path}")
        if source_video_path is not None:
            metadata_path = path.with_name("metadata.json")
            if not metadata_path.is_file():
                raise RuntimeError(
                    "Source flow cache requires sibling metadata.json for "
                    "input identity verification"
                )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            digest = hashlib.sha256()
            with Path(source_video_path).expanduser().resolve().open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if metadata.get("video_sha256") != digest.hexdigest():
                raise RuntimeError(
                    "Source flow cache was computed from a different video"
                )
        tensors = torch.load(path, map_location="cpu", weights_only=True)
        cached_pair_count = int(tensors["forward_flow"].shape[0])
        pixel_frame_count = (
            cached_pair_count + 1
            if pixel_frame_count is None
            else int(pixel_frame_count)
        )
        pair_count = pixel_frame_count - 1
        if pair_count <= 0 or cached_pair_count < pair_count:
            raise ValueError(
                "Source flow cache has fewer RGB transitions than the "
                f"inference clip: {cached_pair_count} < {pair_count}"
            )
        # Inference may causally trim a source video to a valid VAE length.
        # Use the corresponding prefix without changing the read-only cache.
        if cached_pair_count != pair_count:
            tensors = {
                name: value[:pair_count]
                for name, value in tensors.items()
            }
        groups = causal_vae_frame_groups(
            pixel_frame_count, latent_frame_count
        )
        # A causal latent summarizes its complete RGB group.  Its geometry
        # timestamp is therefore the group's last RGB frame.
        latent_pixel_indices = [right - 1 for _, right in groups]
        return cls(tensors, latent_pixel_indices=latent_pixel_indices)

    @property
    def latent_frame_count(self) -> int:
        return len(self.latent_pixel_indices)

    def _adjacent(
        self,
        name: str,
        index: int,
        *,
        size: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        value = self.tensors[name][index:index + 1].to(
            device=device, dtype=torch.float32
        )
        if value.shape[1] == 2:
            return resize_flow(value, size)
        return F.interpolate(
            value, size=size, mode="bilinear", align_corners=True
        ).clamp(0.0, 1.0)

    def _compose(
        self,
        *,
        start_pixel: int,
        end_pixel: int,
        forward: bool,
        size: tuple[int, int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if forward:
            pair_indices = list(range(start_pixel, end_pixel))
            flow_name = "forward_flow"
            confidence_name = "forward_confidence"
        else:
            pair_indices = list(range(end_pixel - 1, start_pixel - 1, -1))
            flow_name = "backward_flow"
            confidence_name = "backward_confidence"
        if not pair_indices:
            raise ValueError("Motion composition requires distinct frames")

        flow = self._adjacent(
            flow_name, pair_indices[0], size=size, device=device
        )
        confidence = self._adjacent(
            confidence_name, pair_indices[0], size=size, device=device
        )
        for pair_index in pair_indices[1:]:
            next_flow = self._adjacent(
                flow_name, pair_index, size=size, device=device
            )
            next_confidence = self._adjacent(
                confidence_name, pair_index, size=size, device=device
            )
            sampled_confidence, valid = sample_with_flow(
                next_confidence, flow
            )
            confidence = (
                confidence * sampled_confidence * valid.float()
            ).clamp(0.0, 1.0)
            flow, valid_flow = compose_forward_flow(flow, next_flow)
            confidence = confidence * valid_flow.float()
        return flow, confidence.clamp(0.0, 1.0)

    def transition(
        self,
        source_latent_index: int,
        target_latent_index: int,
        *,
        size: tuple[int, int],
        device: torch.device,
    ) -> LatentMotionTransition:
        if target_latent_index != source_latent_index + 1:
            raise ValueError(
                "Geometry owner expects consecutive global latent indices"
            )
        if not 0 <= source_latent_index < self.latent_frame_count - 1:
            raise IndexError("Latent flow transition is outside the cache")
        start_pixel = self.latent_pixel_indices[source_latent_index]
        end_pixel = self.latent_pixel_indices[target_latent_index]
        forward_flow, forward_confidence = self._compose(
            start_pixel=start_pixel, end_pixel=end_pixel, forward=True,
            size=size, device=device,
        )
        backward_flow, backward_confidence = self._compose(
            start_pixel=start_pixel, end_pixel=end_pixel, forward=False,
            size=size, device=device,
        )
        return LatentMotionTransition(
            forward_flow=forward_flow,
            backward_flow=backward_flow,
            forward_confidence=forward_confidence,
            backward_confidence=backward_confidence,
        )


def _dilate(value: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return value
    return F.max_pool2d(
        value.float(), kernel_size=2 * radius + 1, stride=1, padding=radius
    )


def _masked_vector_median(
    flow: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    medians = []
    valid_batches = []
    for batch_index in range(flow.shape[0]):
        selected = flow[batch_index, :, mask[batch_index, 0]].float()
        if selected.shape[1] >= 4:
            medians.append(selected.median(dim=1).values)
            valid_batches.append(True)
        else:
            medians.append(flow.new_zeros(2, dtype=torch.float32))
            valid_batches.append(False)
    return (
        torch.stack(medians).reshape(flow.shape[0], 2, 1, 1),
        torch.tensor(valid_batches, device=flow.device).reshape(-1, 1, 1, 1),
    )


class MotionAwareGeometryOwnerTracker:
    """Transport a full soft object owner independently of KV writes."""

    def __init__(
        self,
        source_flow_cache: SourceFlowCache,
        *,
        min_owner_weight: float = 0.05,
        max_occluded_frames: int = 3,
        bootstrap_frames: int = 3,
        reidentify_confidence: float = 0.20,
        observation_correction: float = 0.45,
        eps: float = 1e-6,
    ) -> None:
        self.source_flow_cache = source_flow_cache
        self.min_owner_weight = float(min_owner_weight)
        self.max_occluded_frames = int(max_occluded_frames)
        if bootstrap_frames <= 0:
            raise ValueError("bootstrap_frames must be positive")
        if not 0.0 <= reidentify_confidence <= 1.0:
            raise ValueError(
                "reidentify_confidence must lie in [0, 1]"
            )
        self.bootstrap_frames = int(bootstrap_frames)
        self.reidentify_confidence = float(reidentify_confidence)
        self.observation_correction = float(observation_correction)
        self.eps = float(eps)
        self._geometry_state: torch.Tensor | None = None
        self._previous_hand: torch.Tensor | None = None
        self._last_frame_index: int | None = None
        self._missing_count: torch.Tensor | None = None
        self._frames_seen = 0
        self._last_flow_confidence: torch.Tensor | None = None

    @torch.no_grad()
    def commit_verified(self, **_: object) -> None:
        """Appearance writes never redefine the geometry owner."""

    @torch.no_grad()
    def correct_current_observation(
        self,
        *,
        observation_weight: torch.Tensor,
        tokens_per_frame: int,
    ) -> None:
        """Add step-zero edit-response evidence to the geometry state.

        This consumes the field-refined role observation, not the sparse KV
        write core.  It can recover support missed before denoising but cannot
        erase flow-transported geometry.
        """
        if self._geometry_state is None:
            raise RuntimeError(
                "Motion geometry must be initialized before correction"
            )
        if observation_weight.ndim != 2 or (
            observation_weight.shape[1] % tokens_per_frame
        ):
            raise ValueError(
                "observation_weight must have shape [B,T*tokens_per_frame]"
            )
        if tokens_per_frame != (
            self._geometry_state.shape[-2] * self._geometry_state.shape[-1]
        ):
            raise ValueError(
                "Correction token grid differs from motion geometry grid"
            )
        # Counterfactual velocity is a bootstrap/re-identification cue, not a
        # perpetual area-growth operator.
        allow_correction = self._frames_seen <= self.bootstrap_frames
        if self._last_flow_confidence is not None:
            allow_correction = allow_correction or bool(
                (self._last_flow_confidence < self.reidentify_confidence)
                .any().item()
            )
        if not allow_correction:
            return
        last_observation = observation_weight.float().reshape(
            observation_weight.shape[0], -1, 1,
            self._geometry_state.shape[-2], self._geometry_state.shape[-1],
        )[:, -1].clamp(0.0, 1.0)
        if self._previous_hand is not None:
            last_observation = last_observation * (
                1.0 - self._previous_hand.float()
            )
        self._geometry_state = torch.maximum(
            self._geometry_state.float(), last_observation
        ).detach()

    @torch.no_grad()
    def __call__(
        self,
        *,
        source_features: torch.Tensor,
        observation_weight: torch.Tensor,
        source_semantic: torch.Tensor,
        hand_mask: torch.Tensor,
        hand_proximity: torch.Tensor,
        tokens_per_frame: int,
        detector_visible: torch.Tensor,
        spatial_shape: tuple[int, int],
        frame_indices: Sequence[int] | None = None,
        update_state: bool = True,
    ) -> CausalObjectOwnership:
        batch, token_count = observation_weight.shape
        if token_count % tokens_per_frame:
            raise ValueError("tokens_per_frame must divide ownership tokens")
        frames = token_count // tokens_per_frame
        height, width = spatial_shape
        if height * width != tokens_per_frame:
            raise ValueError("spatial_shape does not match tokens_per_frame")
        expected = (batch, token_count)
        for name, value in (
            ("source_semantic", source_semantic),
            ("hand_mask", hand_mask),
            ("hand_proximity", hand_proximity),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        if frame_indices is None or len(frame_indices) != frames:
            raise ValueError(
                "Motion geometry owner requires one global index per frame"
            )
        indices = [int(value) for value in frame_indices]

        old_state = self._geometry_state
        old_hand = self._previous_hand
        old_index = self._last_frame_index
        old_missing = self._missing_count
        old_frames_seen = self._frames_seen
        old_last_flow_confidence = self._last_flow_confidence
        geometry = self._geometry_state
        previous_hand = self._previous_hand
        last_index = self._last_frame_index
        missing = (
            torch.zeros(batch, dtype=torch.long, device=observation_weight.device)
            if self._missing_count is None
            else self._missing_count.to(observation_weight.device).clone()
        )

        observation = observation_weight.float().reshape(
            batch, frames, 1, height, width
        ).clamp(0.0, 1.0)
        semantic = source_semantic.float().reshape_as(observation).clamp(0.0, 1.0)
        hand = hand_mask.float().reshape_as(observation).clamp(0.0, 1.0)
        proximity = hand_proximity.float().reshape_as(observation).clamp(0.0, 1.0)
        detector_visible = detector_visible.reshape(batch, frames).bool()

        owner_frames = []
        support_frames = []
        transport_frames = []
        confidence_frames = []
        similarity_frames = []
        semantic_frames = []
        state_frames = []
        missing_frames = []
        affinity_frames = []
        residual_frames = []
        occluded_frames = []

        for local_index, global_index in enumerate(indices):
            current_hand = hand[:, local_index]
            current_observation = observation[:, local_index]
            semantic_correction = (
                semantic[:, local_index] * proximity[:, local_index]
                * self.observation_correction
            )
            corrected_observation = torch.maximum(
                current_observation, semantic_correction
            ) * (1.0 - current_hand)

            if geometry is None:
                geometry = corrected_observation
                transported = torch.zeros_like(geometry)
                confidence = torch.ones_like(geometry)
                affinity = proximity[:, local_index]
                residual_magnitude = torch.zeros_like(geometry)
            else:
                if last_index is None or global_index != last_index + 1:
                    raise ValueError(
                        "Non-consecutive global latent indices would corrupt "
                        "the causal motion owner"
                    )
                transition = self.source_flow_cache.transition(
                    last_index, global_index, size=spatial_shape,
                    device=observation_weight.device,
                )
                pull, pull_valid = warp_with_backward_flow(
                    geometry, transition.backward_flow
                )
                pushed, push_coverage = forward_splat(
                    geometry, transition.forward_flow,
                    weight=transition.forward_confidence,
                )
                pull_confidence = (
                    transition.backward_confidence * pull_valid.float()
                ).clamp(0.0, 1.0)
                push_confidence = torch.minimum(
                    push_coverage, torch.ones_like(push_coverage)
                )
                use_push = pull_confidence < self.reidentify_confidence
                confidence = torch.where(
                    use_push, push_confidence, pull_confidence
                ).clamp(0.0, 1.0)
                transported = torch.where(
                    use_push, pushed, pull
                ).clamp(0.0, 1.0)

                prior_hand = (
                    torch.zeros_like(current_hand)
                    if previous_hand is None else previous_hand
                )
                exclusion = _dilate(
                    torch.maximum(prior_hand, geometry), 2
                ) > 0.10
                camera_motion, camera_valid = _masked_vector_median(
                    transition.forward_flow, ~exclusion
                )
                residual = transition.forward_flow - camera_motion
                hand_motion, hand_valid = _masked_vector_median(
                    residual, _dilate(prior_hand, 1) > 0.05
                )
                residual_scale = torch.linalg.vector_norm(
                    residual.flatten(2), dim=1
                ).median(dim=1).values.reshape(batch, 1, 1, 1).clamp_min(0.25)
                source_affinity = torch.exp(
                    -torch.linalg.vector_norm(
                        residual - hand_motion, dim=1, keepdim=True
                    ) / (2.0 * residual_scale)
                )
                source_affinity = torch.where(
                    camera_valid & hand_valid, source_affinity,
                    torch.full_like(source_affinity, 0.5),
                )
                affinity, _ = forward_splat(
                    source_affinity, transition.forward_flow,
                    weight=transition.forward_confidence,
                )
                residual_magnitude, _ = forward_splat(
                    torch.linalg.vector_norm(
                        residual, dim=1, keepdim=True
                    ),
                    transition.forward_flow,
                    weight=transition.forward_confidence,
                )
                # Confidence and hand-relative motion are diagnostics for
                # arbitration, never multiplicative decay on geometry.  A
                # repeated confidence product was precisely the old
                # chunk-by-chunk shrinking failure.  Flow moves the full soft
                # owner; source observations may only add/recover support.
                retained = transported
                bootstrap_active = (
                    self._frames_seen < self.bootstrap_frames
                )
                lost_track = not bool(
                    (retained >= self.min_owner_weight).any().item()
                )
                low_confidence = bool(
                    (
                        confidence.flatten(1).mean(dim=1)
                        < self.reidentify_confidence
                    ).any().item()
                )
                geometry = (
                    torch.maximum(retained, corrected_observation)
                    if bootstrap_active or lost_track or low_confidence
                    else retained
                ).clamp(0.0, 1.0)

            visible_owner = geometry * (1.0 - current_hand)
            support = visible_owner >= self.min_owner_weight
            visible_owner = visible_owner * support.float()
            present = support.flatten(1).any(dim=1)
            missing = torch.where(present, torch.zeros_like(missing), missing + 1)
            state = torch.where(
                present,
                torch.full_like(missing, int(CausalOwnershipState.VISIBLE)),
                torch.where(
                    missing <= self.max_occluded_frames,
                    torch.full_like(missing, int(CausalOwnershipState.OCCLUDED)),
                    torch.full_like(missing, int(CausalOwnershipState.ABSENT)),
                ),
            )

            owner_frames.append(visible_owner)
            support_frames.append(support)
            transport_frames.append(transported)
            confidence_frames.append(confidence)
            similarity_frames.append(confidence.mul(2.0).sub(1.0))
            semantic_frames.append(semantic[:, local_index])
            state_frames.append(state)
            missing_frames.append(missing.clone())
            affinity_frames.append(affinity)
            residual_frames.append(residual_magnitude)
            occluded_frames.append((confidence < 0.25).float())
            previous_hand = current_hand
            last_index = global_index
            self._frames_seen += 1
            self._last_flow_confidence = confidence.flatten(1).mean(dim=1)

        self._geometry_state = geometry.detach()
        self._previous_hand = previous_hand.detach()
        self._last_frame_index = last_index
        self._missing_count = missing.detach().cpu()

        def flatten(values: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack(values, dim=1).reshape(batch, -1)

        result = CausalObjectOwnership(
            owner_weight=flatten(owner_frames),
            owner_support=flatten(support_frames).bool(),
            transported_weight=flatten(transport_frames),
            observation_weight=observation.reshape(batch, -1),
            match_similarity=flatten(similarity_frames),
            match_confidence=flatten(confidence_frames),
            semantic_support=flatten(semantic_frames),
            state_code=torch.stack(state_frames, dim=1),
            missing_frames=torch.stack(missing_frames, dim=1),
            diagnostics={
                "motion_geometry_state": flatten(owner_frames),
                "motion_geometry_occluded": flatten(occluded_frames),
                "motion_hand_affinity": flatten(affinity_frames),
                "motion_flow_confidence": flatten(confidence_frames),
                "motion_residual_magnitude": flatten(residual_frames),
            },
        )
        result.validate()
        if not update_state:
            self._geometry_state = old_state
            self._previous_hand = old_hand
            self._last_frame_index = old_index
            self._missing_count = old_missing
            self._frames_seen = old_frames_seen
            self._last_flow_confidence = old_last_flow_confidence
        return result
