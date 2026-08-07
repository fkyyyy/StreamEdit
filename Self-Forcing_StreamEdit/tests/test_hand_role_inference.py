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


_load("pipeline.role_router", ROOT / "pipeline" / "role_router.py")
hand_role = _load(
    "pipeline.hand_role_inference",
    ROOT / "pipeline" / "hand_role_inference.py",
)


def _attention(dtype=torch.float32):
    value = torch.linspace(-1.0, 1.0, 16, dtype=dtype).reshape(1, 16)
    return value


def _hand():
    value = torch.zeros((1, 1, 8, 8))
    value[:, :, 2:5, 1:3] = 1.0
    return value


def test_hand_role_shapes_and_partition():
    result = hand_role.HandRoleInferencer()(_attention(), _hand())

    for value in result.roles.as_dict().values():
        assert value.shape == (1, 1, 8, 8)
        assert torch.isfinite(value).all()
        assert value.min() >= 0
        assert value.max() <= 1
    total = sum(result.roles.as_dict().values())
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)
    assert result.token_edit_confidence.shape == (1, 16)


def test_empty_hand_preserves_entire_frame():
    result = hand_role.HandRoleInferencer()(
        _attention(),
        torch.zeros_like(_hand()),
    )

    assert torch.count_nonzero(result.token_edit_confidence) == 0
    assert torch.count_nonzero(result.roles.edit_weight) == 0
    assert torch.allclose(
        result.roles.background,
        torch.ones_like(result.roles.background),
    )


def test_full_hand_does_not_create_object():
    result = hand_role.HandRoleInferencer()(
        _attention(),
        torch.ones_like(_hand()),
    )

    assert torch.count_nonzero(result.token_edit_confidence) == 0
    assert torch.count_nonzero(result.roles.edit_weight) == 0
    assert torch.allclose(
        result.roles.hand,
        torch.ones_like(result.roles.hand),
    )


def test_small_seed_propagates_without_filling_frame():
    result = hand_role.HandRoleInferencer(
        propagation_steps=2,
        max_object_coverage=0.25,
    )(_attention(), _hand())
    posterior = result.token_edit_confidence

    assert torch.count_nonzero(posterior) > 0
    assert (posterior > 0).float().mean() <= 0.25


def test_bfloat16_attention_is_normalized_in_float32():
    result = hand_role.HandRoleInferencer()(
        _attention(torch.bfloat16),
        _hand().to(torch.bfloat16),
    )

    assert result.token_edit_confidence.dtype == torch.float32
    assert torch.isfinite(result.token_edit_confidence).all()


def test_visibility_gate_clears_frame_without_interaction_support():
    attention = torch.cat(
        [
            _attention(),
            torch.zeros_like(_attention()),
        ],
        dim=1,
    )
    hand = _hand().repeat(1, 2, 1, 1)
    result = hand_role.HandRoleInferencer(
        visibility_ratio=0.40,
    )(attention, hand)
    posterior = result.token_edit_confidence.reshape(1, 2, 4, 4)

    assert result.debug["object_visible"][0, 0].item() == 1.0
    assert result.debug["object_visible"][0, 1].item() == 0.0
    assert torch.count_nonzero(posterior[:, 1]) == 0


def test_query_affinity_propagates_previous_object_posterior():
    attention = _attention().repeat(1, 2)
    hand = _hand().repeat(1, 2, 1, 1)
    token_identity = torch.eye(16)
    source_features = torch.cat(
        [token_identity, token_identity],
        dim=0,
    ).unsqueeze(0)

    result = hand_role.HandRoleInferencer(
        visibility_ratio=0.0,
        temporal_weight=0.75,
        query_similarity_threshold=0.5,
    )(
        attention,
        hand,
        source_features=source_features,
    )

    assert result.debug["temporal_posterior"][:, 1].max() > 0
    assert result.debug["temporal_confidence"][:, 1].max() > 0
