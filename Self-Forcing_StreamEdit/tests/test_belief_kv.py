import importlib.util
from pathlib import Path
import sys

import pytest
import torch


PIPELINE_ROOT = Path(__file__).parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


belief_kv = _load_module(
    "belief_kv",
    PIPELINE_ROOT / "pipeline" / "belief_kv.py",
)
attention_module = _load_module(
    "belief_kv_attention",
    PIPELINE_ROOT / "wan" / "modules" / "attention.py",
)


class _Belief:
    def __init__(
        self,
        edit,
        preserve,
        edit_precision=None,
        preserve_precision=None,
    ):
        self.edit_belief = edit
        self.preserve_belief = preserve
        self.edit_precision = (
            torch.ones_like(edit)
            if edit_precision is None
            else edit_precision
        )
        self.preserve_precision = (
            torch.ones_like(preserve)
            if preserve_precision is None
            else preserve_precision
        )

    def validate(self):
        values = (
            self.edit_belief,
            self.preserve_belief,
            self.edit_precision,
            self.preserve_precision,
        )
        assert len({tuple(value.shape) for value in values}) == 1


def test_belief_kv_preserves_nonexclusive_conflict():
    edit = torch.zeros((1, 1, 4, 4))
    preserve = torch.zeros_like(edit)
    edit[:, :, :2, :2] = 0.8
    preserve[:, :, :2, :2] = 0.6

    weights = belief_kv.build_belief_kv_weights(
        _Belief(edit, preserve),
        expected_token_length=4,
    )

    assert weights.edit.shape == (1, 4)
    assert weights.preserve.shape == (1, 4)
    assert torch.allclose(weights.edit[0, 0], torch.tensor(0.8))
    assert torch.allclose(
        weights.preserve[0, 0],
        torch.tensor(0.6),
    )
    assert torch.allclose(
        weights.conflict_map[0, 0, 0, 0],
        torch.tensor(0.48),
    )
    assert torch.allclose(
        weights.edit_action[0, 0],
        torch.tensor(0.8 / 1.4),
    )
    assert torch.allclose(
        weights.preserve_action[0, 0],
        torch.tensor(0.6 / 1.4),
    )


def test_belief_kv_falls_back_to_source_when_evidence_is_absent():
    zero = torch.zeros((1, 1, 4, 4))

    weights = belief_kv.build_belief_kv_weights(
        _Belief(zero, zero),
        expected_token_length=4,
    )

    assert torch.count_nonzero(weights.edit) == 0
    assert torch.equal(
        weights.preserve,
        torch.ones_like(weights.preserve),
    )
    assert torch.count_nonzero(weights.edit_action) == 0
    assert torch.equal(
        weights.preserve_action,
        torch.ones_like(weights.preserve_action),
    )


def test_belief_kv_query_action_uses_online_precision():
    edit = torch.full((1, 1, 4, 4), 0.8)
    preserve = torch.full_like(edit, 0.6)
    edit_precision = torch.full_like(edit, 0.25)
    preserve_precision = torch.ones_like(edit)

    weights = belief_kv.build_belief_kv_weights(
        _Belief(
            edit,
            preserve,
            edit_precision,
            preserve_precision,
        ),
        expected_token_length=4,
    )

    assert torch.allclose(
        weights.edit_action,
        torch.full_like(weights.edit_action, 0.25),
    )
    assert torch.allclose(
        weights.preserve_action,
        torch.full_like(weights.preserve_action, 0.75),
    )
    assert torch.allclose(
        weights.edit_action + weights.preserve_action,
        torch.ones_like(weights.edit_action),
    )


def test_belief_kv_rejects_misaligned_token_length():
    value = torch.ones((1, 1, 4, 4))

    with pytest.raises(ValueError, match="different token counts"):
        belief_kv.build_belief_kv_weights(
            _Belief(value, value),
            expected_token_length=5,
        )


def test_aligned_memory_fusion_preserves_endpoints_and_conflict():
    target_key = torch.tensor(
        [[[[0.0]], [[2.0]], [[4.0]]]]
    )
    target_value = target_key + 1.0
    source_key = torch.tensor(
        [[[[10.0]], [[12.0]], [[14.0]]]]
    )
    source_value = source_key + 1.0

    fused_key, fused_value = attention_module.fuse_aligned_memory(
        target_key,
        target_value,
        source_key,
        source_value,
        preserve_action=torch.tensor([[0.0, 0.5, 1.0]]),
    )

    assert torch.allclose(
        fused_key.flatten(),
        torch.tensor([0.0, 7.0, 14.0]),
    )
    assert torch.allclose(
        fused_value.flatten(),
        torch.tensor([1.0, 8.0, 15.0]),
    )


def test_aligned_memory_fusion_supports_bfloat16():
    target = torch.zeros((1, 2, 1, 1), dtype=torch.bfloat16)
    source = torch.full_like(target, 2.0)

    fused_key, fused_value = attention_module.fuse_aligned_memory(
        target,
        target,
        source,
        source,
        preserve_action=torch.tensor([[0.25, 0.75]]),
    )

    assert fused_key.dtype == torch.bfloat16
    assert fused_value.dtype == torch.bfloat16
    assert torch.allclose(
        fused_key.float().flatten(),
        torch.tensor([0.5, 1.5]),
    )


def test_aligned_memory_fusion_rejects_unaligned_tokens():
    target = torch.zeros((1, 2, 1, 1))
    source = torch.zeros((1, 3, 1, 1))

    with pytest.raises(ValueError, match="keys must share shape"):
        attention_module.fuse_aligned_memory(
            target,
            target,
            source,
            source,
            preserve_action=torch.ones((1, 2)),
        )


def test_current_qk_blend_keeps_edit_tokens_target_pure():
    target = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    source = torch.tensor([[[11.0]], [[12.0]], [[13.0]]])

    blended = attention_module.blend_current_target_state(
        target=target,
        source=source,
        blend_rate=0.25,
        edit_support=torch.tensor([False, True, False]),
    )

    assert torch.allclose(
        blended.flatten(),
        torch.tensor([8.5, 2.0, 10.5]),
    )


def test_current_qk_blend_preserves_scalar_baseline():
    target = torch.tensor([[[0.0]], [[2.0]]])
    source = torch.tensor([[[4.0]], [[6.0]]])

    blended = attention_module.blend_current_target_state(
        target=target,
        source=source,
        blend_rate=0.25,
    )

    assert torch.allclose(
        blended.flatten(),
        torch.tensor([3.0, 5.0]),
    )
