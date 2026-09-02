from __future__ import annotations

import torch

from tests._pipeline_imports import load_pipeline_module


ownership_module = load_pipeline_module("causal_ownership")
CausalObjectOwnership = ownership_module.CausalObjectOwnership
flow_role_module = load_pipeline_module("motion/flow_role_evidence")
build_flow_role_evidence = flow_role_module.build_flow_role_evidence


def make_ownership() -> CausalObjectOwnership:
    owner = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    confidence = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
    affinity = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    return CausalObjectOwnership(
        owner_weight=owner,
        owner_support=owner.bool(),
        transported_weight=owner,
        observation_weight=owner,
        match_similarity=confidence.mul(2.0).sub(1.0),
        match_confidence=confidence,
        semantic_support=owner,
        state_code=torch.ones(1, 1, dtype=torch.long),
        missing_frames=torch.zeros(1, 1, dtype=torch.long),
        diagnostics={"motion_hand_affinity": affinity},
    )


def test_flow_role_requires_owner_for_object_evidence() -> None:
    result = build_flow_role_evidence(
        make_ownership(), shape=(1, 1, 2, 2),
        hand_exclusion=torch.zeros(1, 1, 2, 2),
    )
    assert result.object_likelihood[0, 0, 0, 1] > 0.9
    # Strong hand-like motion at a non-owned location is not an object cue.
    assert result.object_likelihood[0, 0, 1, 0] == 0.0


def test_flow_role_marks_confident_non_owner_as_background() -> None:
    result = build_flow_role_evidence(
        make_ownership(), shape=(1, 1, 2, 2),
        hand_exclusion=torch.zeros(1, 1, 2, 2),
    )
    assert result.background_likelihood[0, 0, 0, 0] > 0.9
    assert result.background_likelihood[0, 0, 0, 1] == 0.0
    assert result.unknown_likelihood[0, 0, 1, 0] > 0.9


def test_flow_role_never_relabels_hand_as_object() -> None:
    hand = torch.zeros(1, 1, 2, 2)
    hand[0, 0, 0, 1] = 1.0
    result = build_flow_role_evidence(
        make_ownership(), shape=(1, 1, 2, 2), hand_exclusion=hand
    )
    assert result.object_likelihood[0, 0, 0, 1] == 0.0
    assert result.transport_support[0, 0, 0, 1] == 0.0
