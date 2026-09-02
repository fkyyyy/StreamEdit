from __future__ import annotations

import torch

from tests._pipeline_imports import load_pipeline_module


module = load_pipeline_module("source_flow_verified_region")
build_source_flow_verified_region = (
    module.build_source_flow_verified_region
)


def _inputs():
    shape = (1, 1, 5, 7)
    posterior = torch.zeros(shape)
    threshold = torch.full((1, 1, 1, 1), 0.5)
    owner = torch.zeros(shape, dtype=torch.bool)
    hand = torch.zeros(shape, dtype=torch.bool)
    background = torch.zeros(shape)
    confidence = torch.ones(shape)
    return posterior, threshold, owner, hand, background, confidence


def test_semantic_proposal_requires_nearby_flow_owner() -> None:
    posterior, threshold, owner, hand, background, confidence = _inputs()
    owner[0, 0, 2, 2] = True
    posterior[0, 0, 2, 3] = 0.9
    posterior[0, 0, 0, 6] = 0.9

    result = build_source_flow_verified_region(
        object_posterior=posterior,
        posterior_threshold=threshold,
        owner_support=owner,
        hand_exclusion=hand,
        background_likelihood=background,
        flow_confidence=confidence,
        owner_radius=1,
    )

    assert result.support[0, 0, 2, 2]
    assert result.support[0, 0, 2, 3]
    assert not result.support[0, 0, 0, 6]


def test_reliable_background_vetoes_extension_but_not_owner() -> None:
    posterior, threshold, owner, hand, background, confidence = _inputs()
    owner[0, 0, 2, 2] = True
    posterior[0, 0, 2, 2:4] = 0.9
    background[0, 0, 2, 2:4] = 0.9

    result = build_source_flow_verified_region(
        object_posterior=posterior,
        posterior_threshold=threshold,
        owner_support=owner,
        hand_exclusion=hand,
        background_likelihood=background,
        flow_confidence=confidence,
        owner_radius=1,
        background_veto_threshold=0.55,
        background_veto_min_confidence=0.50,
    )

    assert result.support[0, 0, 2, 2]
    assert result.recovered_owner[0, 0, 2, 2]
    assert not result.support[0, 0, 2, 3]


def test_unreliable_background_does_not_veto_semantic_extension() -> None:
    posterior, threshold, owner, hand, background, confidence = _inputs()
    owner[0, 0, 2, 2] = True
    posterior[0, 0, 2, 3] = 0.9
    background[0, 0, 2, 3] = 0.9
    confidence[0, 0, 2, 3] = 0.49

    result = build_source_flow_verified_region(
        object_posterior=posterior,
        posterior_threshold=threshold,
        owner_support=owner,
        hand_exclusion=hand,
        background_likelihood=background,
        flow_confidence=confidence,
        owner_radius=1,
    )

    assert result.support[0, 0, 2, 3]
    assert not result.reliable_background_veto[0, 0, 2, 3]


def test_hand_exclusion_is_final_even_on_owner_token() -> None:
    posterior, threshold, owner, hand, background, confidence = _inputs()
    owner[0, 0, 2, 2] = True
    posterior[0, 0, 2, 2] = 0.9
    hand[0, 0, 2, 2] = True

    result = build_source_flow_verified_region(
        object_posterior=posterior,
        posterior_threshold=threshold,
        owner_support=owner,
        hand_exclusion=hand,
        background_likelihood=background,
        flow_confidence=confidence,
    )

    assert not result.support[0, 0, 2, 2]
    assert result.posterior[0, 0, 2, 2] == 0


def test_recovered_owner_gets_threshold_floor() -> None:
    posterior, threshold, owner, hand, background, confidence = _inputs()
    owner[0, 0, 2, 2] = True
    posterior[0, 0, 2, 2] = 0.1

    result = build_source_flow_verified_region(
        object_posterior=posterior,
        posterior_threshold=threshold,
        owner_support=owner,
        hand_exclusion=hand,
    )

    assert result.posterior[0, 0, 2, 2] == 0.5
    assert result.posterior[0, 0, 0, 0] == 0


def test_rejects_unpaired_background_evidence() -> None:
    posterior, threshold, owner, hand, background, _ = _inputs()

    try:
        build_source_flow_verified_region(
            object_posterior=posterior,
            posterior_threshold=threshold,
            owner_support=owner,
            hand_exclusion=hand,
            background_likelihood=background,
        )
    except ValueError as error:
        assert "supplied together" in str(error)
    else:
        raise AssertionError("Expected unpaired flow evidence to fail")
