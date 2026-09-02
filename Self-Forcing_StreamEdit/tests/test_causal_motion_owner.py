from __future__ import annotations

import sys
import types
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if "pipeline" not in sys.modules:
    package = types.ModuleType("pipeline")
    package.__path__ = [str(REPO_ROOT / "pipeline")]
    sys.modules["pipeline"] = package

from pipeline.motion.causal_motion_owner import (
    MotionAwareGeometryOwnerTracker,
    SourceFlowCache,
)


def _translation_cache(latent_frames: int = 3) -> SourceFlowCache:
    pair_count = latent_frames - 1
    forward = torch.zeros(pair_count, 2, 5, 7)
    backward = torch.zeros_like(forward)
    forward[:, 0] = 1.0
    backward[:, 0] = -1.0
    confidence = torch.ones(pair_count, 1, 5, 7)
    return SourceFlowCache(
        {
            "forward_flow": forward,
            "backward_flow": backward,
            "forward_confidence": confidence,
            "backward_confidence": confidence,
        },
        latent_pixel_indices=list(range(latent_frames)),
    )


def test_motion_owner_survives_detector_dropout_and_moves() -> None:
    tracker = MotionAwareGeometryOwnerTracker(_translation_cache())
    observation = torch.zeros(1, 3, 35)
    observation[0, 0, 2 * 7 + 1] = 1.0
    zeros = torch.zeros_like(observation)
    result = tracker(
        source_features=torch.zeros(1, 105, 2),
        observation_weight=observation.reshape(1, -1),
        source_semantic=observation.reshape(1, -1),
        hand_mask=zeros.reshape(1, -1),
        hand_proximity=torch.ones_like(observation).reshape(1, -1),
        tokens_per_frame=35,
        detector_visible=torch.tensor([[True, False, False]]),
        spatial_shape=(5, 7),
        frame_indices=[0, 1, 2],
    )
    owner = result.owner_weight.reshape(1, 3, 5, 7)
    assert owner[0, 0, 2, 1] > 0.9
    assert owner[0, 1, 2, 2] > 0.7
    assert owner[0, 2, 2, 3] > 0.5


def test_appearance_commit_cannot_shrink_geometry_state() -> None:
    tracker = MotionAwareGeometryOwnerTracker(_translation_cache())
    observation = torch.zeros(1, 1, 35)
    observation[0, 0, 2 * 7 + 1] = 1.0
    tracker(
        source_features=torch.zeros(1, 35, 2),
        observation_weight=observation.reshape(1, -1),
        source_semantic=observation.reshape(1, -1),
        hand_mask=torch.zeros(1, 35),
        hand_proximity=torch.ones(1, 35),
        tokens_per_frame=35,
        detector_visible=torch.ones(1, 1, dtype=torch.bool),
        spatial_shape=(5, 7),
        frame_indices=[0],
    )
    tracker.commit_verified(
        source_features=torch.zeros(1, 35, 2),
        verified_weight=torch.zeros(1, 35),
        tokens_per_frame=35,
    )
    result = tracker(
        source_features=torch.zeros(1, 35, 2),
        observation_weight=torch.zeros(1, 35),
        source_semantic=torch.zeros(1, 35),
        hand_mask=torch.zeros(1, 35),
        hand_proximity=torch.ones(1, 35),
        tokens_per_frame=35,
        detector_visible=torch.zeros(1, 1, dtype=torch.bool),
        spatial_shape=(5, 7),
        frame_indices=[1],
    )
    assert result.owner_weight.reshape(1, 1, 5, 7)[0, 0, 2, 2] > 0.7


def test_high_confidence_transport_does_not_decay_across_frames() -> None:
    latent_frames = 5
    tracker = MotionAwareGeometryOwnerTracker(
        _translation_cache(latent_frames)
    )
    observation = torch.zeros(1, latent_frames, 35)
    observation[0, 0, 2 * 7] = 1.0
    result = tracker(
        source_features=torch.zeros(1, latent_frames * 35, 2),
        observation_weight=observation.reshape(1, -1),
        source_semantic=observation.reshape(1, -1),
        hand_mask=torch.zeros(1, latent_frames * 35),
        hand_proximity=torch.ones(1, latent_frames * 35),
        tokens_per_frame=35,
        detector_visible=torch.tensor([[True, False, False, False, False]]),
        spatial_shape=(5, 7),
        frame_indices=list(range(latent_frames)),
    )
    owner = result.owner_weight.reshape(1, latent_frames, 5, 7)
    assert owner[0, -1, 2, 4] > 0.99


def test_field_correction_adds_but_does_not_crop_geometry() -> None:
    tracker = MotionAwareGeometryOwnerTracker(_translation_cache())
    initial = torch.zeros(1, 35)
    initial[0, 2 * 7 + 1] = 1.0
    common = dict(
        source_features=torch.zeros(1, 35, 2),
        source_semantic=initial,
        hand_mask=torch.zeros(1, 35),
        hand_proximity=torch.ones(1, 35),
        tokens_per_frame=35,
        detector_visible=torch.ones(1, 1, dtype=torch.bool),
        spatial_shape=(5, 7),
        frame_indices=[0],
    )
    tracker(observation_weight=initial, **common)
    correction = torch.zeros(1, 35)
    correction[0, 2 * 7 + 2] = 0.8
    tracker.correct_current_observation(
        observation_weight=correction, tokens_per_frame=35
    )
    state = tracker._geometry_state
    assert state is not None
    assert state[0, 0, 2, 1] == 1.0
    assert state[0, 0, 2, 2] == 0.8
