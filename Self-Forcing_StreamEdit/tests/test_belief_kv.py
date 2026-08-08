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
    def __init__(self, edit, preserve):
        self.edit_belief = edit
        self.preserve_belief = preserve

    def validate(self):
        assert self.edit_belief.shape == self.preserve_belief.shape


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


def test_belief_kv_rejects_misaligned_token_length():
    value = torch.ones((1, 1, 4, 4))

    with pytest.raises(ValueError, match="different token counts"):
        belief_kv.build_belief_kv_weights(
            _Belief(value, value),
            expected_token_length=5,
        )


def test_weighted_attention_keeps_batch_numerators_aligned(monkeypatch):
    def uniform_attention(query, key, value):
        del key
        return value.mean(dim=1, keepdim=True).expand(
            -1,
            query.shape[1],
            -1,
            -1,
        )

    monkeypatch.setattr(
        attention_module,
        "attention",
        uniform_attention,
    )
    query = torch.zeros((2, 1, 1, 1))
    key = torch.zeros((2, 2, 1, 1))
    value = torch.tensor(
        [
            [[[[1.0]]], [[[3.0]]]],
            [[[[2.0]]], [[[4.0]]]],
        ]
    ).reshape(2, 2, 1, 1)
    key_weight = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]]
    )

    output = attention_module.weighted_attention(
        query,
        key,
        value,
        key_weight,
    )

    assert torch.allclose(
        output.flatten(),
        torch.tensor([1.0, 4.0]),
    )
