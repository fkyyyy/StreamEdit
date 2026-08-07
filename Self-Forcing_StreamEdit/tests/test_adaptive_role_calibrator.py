import importlib.util
from pathlib import Path
import sys
import types

import torch


ROOT = Path(__file__).parents[1]
pipeline_package = types.ModuleType("pipeline")
pipeline_package.__path__ = [str(ROOT / "pipeline")]
sys.modules.setdefault("pipeline", pipeline_package)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


adaptive = _load(
    "pipeline.adaptive_role_calibrator",
    ROOT / "pipeline" / "adaptive_role_calibrator.py",
)
_load("pipeline.role_router", ROOT / "pipeline" / "role_router.py")
hand_role = _load(
    "pipeline.hand_role_inference",
    ROOT / "pipeline" / "hand_role_inference.py",
)


def _attention(frames=1, dtype=torch.float32):
    value = torch.zeros((1, frames, 8, 8), dtype=dtype)
    value[:, :, 2:6, 2:6] = 0.8
    value[:, :, 3:5, 3:5] = 1.0
    return value


def _hand(frames=1, dtype=torch.float32):
    value = torch.zeros((1, frames, 8, 8), dtype=dtype)
    value[:, :, 4:6, 2:4] = 1.0
    return value


def test_hand_radius_scales_with_online_hand_size():
    hand = torch.zeros((1, 2, 12, 12))
    hand[:, 0, 5:7, 5:7] = 1.0
    hand[:, 1, 2:10, 2:10] = 1.0
    observation = adaptive.AdaptiveRoleCalibrator(
        max_coverage=1.0,
    ).observe(
        torch.ones_like(hand),
        hand,
    )
    radius = observation.debug["adaptive_hand_radius"]
    budget = observation.debug["adaptive_coverage_budget"]

    assert radius[0, 1].item() > radius[0, 0].item()
    assert budget[0, 1].item() > budget[0, 0].item()


def test_visibility_uses_causal_robust_history():
    calibrator = adaptive.AdaptiveRoleCalibrator()
    support = torch.tensor([0.50, 0.45, 0.48, 0.04]).reshape(
        1, 4, 1, 1
    )
    hand_present = torch.ones((1, 4), dtype=torch.bool)

    visible, threshold, _, _ = calibrator._visibility(
        support,
        hand_present,
    )

    assert visible[0, :3].all()
    assert not visible[0, 3].item()
    assert threshold[0, 3].item() > support[0, 3].item()


def test_temporal_weight_follows_query_confidence():
    calibrator = adaptive.AdaptiveRoleCalibrator()
    propagated = torch.ones((1, 4, 4))
    reliability = torch.ones((1, 1, 1))
    low = calibrator.temporal_weight(
        propagated,
        torch.full_like(propagated, 0.1),
        reliability,
    )
    high = calibrator.temporal_weight(
        propagated,
        torch.full_like(propagated, 0.9),
        reliability,
    )

    assert high.item() > low.item()


def test_posterior_threshold_follows_attention_reliability():
    posterior = torch.linspace(0.05, 0.95, 16).reshape(
        1, 1, 4, 4
    )
    low_calibrator = adaptive.AdaptiveRoleCalibrator()
    low_calibrator.state.current_attention_reliability = torch.zeros(
        (1, 1, 1, 1)
    )
    high_calibrator = adaptive.AdaptiveRoleCalibrator()
    high_calibrator.state.current_attention_reliability = torch.ones(
        (1, 1, 1, 1)
    )

    low_reliability_threshold = low_calibrator.posterior_threshold(
        posterior
    )
    high_reliability_threshold = high_calibrator.posterior_threshold(
        posterior
    )

    assert (
        high_reliability_threshold.item()
        < low_reliability_threshold.item()
    )


def test_field_reliability_detects_seed_ring_separation():
    calibrator = adaptive.AdaptiveRoleCalibrator()
    prior = torch.zeros((1, 1, 8, 8))
    prior[:, :, 3:5, 3:5] = 0.8
    seed = prior.clone()
    visible = torch.ones((1, 1, 1, 1), dtype=torch.bool)
    budget = torch.full((1, 1, 1, 1), 0.25)

    separated_field = torch.zeros_like(prior)
    separated_field[:, :, 3:5, 3:5] = 1.0
    _, _, separated_debug = calibrator.field_update(
        prior,
        separated_field,
        seed,
        visible,
        budget,
        torch.full((1, 1, 1, 1), 0.2),
    )

    flat_field = torch.full_like(prior, 0.5)
    _, _, flat_debug = calibrator.field_update(
        prior,
        flat_field,
        seed,
        visible,
        budget,
        torch.full((1, 1, 1, 1), 0.2),
    )

    assert (
        separated_debug["adaptive_field_reliability"].item()
        > flat_debug["adaptive_field_reliability"].item()
    )


def test_field_update_never_creates_support_outside_prior():
    calibrator = adaptive.AdaptiveRoleCalibrator()
    prior = torch.zeros((1, 1, 8, 8))
    prior[:, :, 3:5, 3:5] = 0.6
    seed = prior.clone()
    field = torch.rand_like(prior)
    posterior, _, _ = calibrator.field_update(
        prior,
        field,
        seed,
        torch.ones((1, 1, 1, 1), dtype=torch.bool),
        torch.full((1, 1, 1, 1), 0.25),
        torch.full((1, 1, 1, 1), 0.2),
    )

    assert torch.count_nonzero(
        (posterior > 0) & ~(prior > 0)
    ) == 0
    assert torch.all(posterior <= prior)


def test_adaptive_hand_role_partition_and_debug():
    hand_latent = torch.zeros((1, 1, 16, 16))
    hand_latent[:, :, 8:12, 4:8] = 1.0
    attention = _attention().reshape(1, -1)
    result = hand_role.HandRoleInferencer(adaptive=True)(
        attention,
        hand_latent,
    )

    role_sum = sum(result.roles.as_dict().values())
    assert torch.allclose(role_sum, torch.ones_like(role_sum), atol=1e-5)
    assert torch.isfinite(result.token_edit_confidence).all()
    assert "adaptive_hand_radius" in result.debug
    assert "adaptive_attention_reliability" in result.debug
    assert "adaptive_coverage_budget" in result.debug
    assert "posterior_threshold" in result.debug


def test_adaptive_empty_and_full_hand_preserve_frame():
    attention = _attention().reshape(1, -1)
    for hand_latent in (
        torch.zeros((1, 1, 16, 16)),
        torch.ones((1, 1, 16, 16)),
    ):
        result = hand_role.HandRoleInferencer(adaptive=True)(
            attention,
            hand_latent,
        )
        assert torch.count_nonzero(
            result.token_edit_confidence
        ) == 0
        assert torch.count_nonzero(result.roles.edit_weight) == 0


def test_adaptive_bfloat16_input_returns_float32():
    hand_latent = torch.zeros(
        (1, 1, 16, 16),
        dtype=torch.bfloat16,
    )
    hand_latent[:, :, 8:12, 4:8] = 1.0
    result = hand_role.HandRoleInferencer(adaptive=True)(
        _attention(dtype=torch.bfloat16).reshape(1, -1),
        hand_latent,
    )

    assert result.token_edit_confidence.dtype == torch.float32
    assert torch.isfinite(result.token_edit_confidence).all()
