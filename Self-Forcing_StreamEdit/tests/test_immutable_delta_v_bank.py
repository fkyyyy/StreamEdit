import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


attention_module = load_module(
    "streamedit_attention_m1_standalone",
    "wan/modules/attention.py",
)
bank_module = load_module(
    "streamedit_immutable_delta_v_bank_standalone",
    "pipeline/immutable_delta_v_bank.py",
)
immutable_delta_v_memory_attention = (
    attention_module.immutable_delta_v_memory_attention
)
ImmutableDeltaVBank = bank_module.ImmutableDeltaVBank


def make_cache(value):
    return [{
        "v": value.clone(),
        "num_new_tokens": value.shape[1],
        "local_end_index": torch.tensor(value.shape[1]),
    }]


def test_bank_freezes_compact_target_minus_source_once():
    source_value = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)
    target_value = source_value + 2.0
    source_key = source_value + 10.0
    support = torch.tensor([[True, False, True]])
    source_cache = make_cache(source_value)
    target_cache = make_cache(target_value)
    bank = ImmutableDeltaVBank([0])

    diagnostics = bank.freeze(
        source_kv_cache=source_cache,
        target_kv_cache=target_cache,
        source_keys={0: source_key},
        support=support,
    )
    state = bank.export()[0]

    assert bank.is_frozen
    assert state["source_key"].shape == (1, 2, 2, 4)
    assert torch.equal(state["source_key"], source_key[:, [0, 2]])
    assert torch.equal(state["delta_value"], torch.full_like(
        state["delta_value"], 2.0
    ))
    assert torch.equal(state["support"], torch.ones(1, 2, dtype=torch.bool))
    torch.testing.assert_close(diagnostics["support_fraction"], torch.tensor(2 / 3))
    assert torch.equal(source_cache[0]["v"], source_value)
    assert torch.equal(target_cache[0]["v"], target_value)
    source_key.zero_()
    target_cache[0]["v"].zero_()
    assert torch.count_nonzero(state["source_key"]) > 0
    assert torch.equal(
        state["delta_value"], torch.full_like(state["delta_value"], 2.0)
    )

    with pytest.raises(RuntimeError, match="already frozen"):
        bank.freeze(
            source_kv_cache=source_cache,
            target_kv_cache=target_cache,
            source_keys={0: source_key},
            support=support,
        )


def test_retrieval_adds_delta_only_to_admitted_owner_query():
    native = torch.ones(1, 2, 1, 2)
    query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    memory_key = query.clone()
    memory_delta = torch.tensor([[[[2.0, 0.0]], [[0.0, 4.0]]]])
    memory_support = torch.ones(1, 2, dtype=torch.bool)
    owner = torch.tensor([[1.0, 0.0]])

    output, diagnostics = immutable_delta_v_memory_attention(
        native, query, memory_key, memory_delta, memory_support, owner,
        topk=1, min_similarity=0.35, strength=0.25, max_rms_ratio=10.0,
    )

    torch.testing.assert_close(
        output[:, 0], native[:, 0] + 0.25 * memory_delta[:, 0]
    )
    assert torch.equal(output[:, 1], native[:, 1])
    torch.testing.assert_close(
        diagnostics["matched_query_fraction"], torch.tensor(0.5)
    )
    torch.testing.assert_close(
        diagnostics["retrieval_similarity"], torch.tensor(1.0)
    )


def test_low_similarity_abstains_bit_exactly_and_does_not_mutate_inputs():
    native = torch.randn(1, 3, 2, 2, dtype=torch.bfloat16)
    query = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]],
        dtype=torch.float32,
    ).expand(-1, -1, 2, -1).to(torch.bfloat16)
    memory_key = -query[:, :1].clone()
    memory_delta = torch.randn_like(memory_key)
    memory_support = torch.ones(1, 1, dtype=torch.bool)
    owner = torch.ones(1, 3)
    snapshots = [
        tensor.clone()
        for tensor in (native, query, memory_key, memory_delta, owner)
    ]

    output, diagnostics = immutable_delta_v_memory_attention(
        native, query, memory_key, memory_delta, memory_support, owner,
        topk=1, min_similarity=0.9, strength=1.0, max_rms_ratio=1.0,
    )

    assert torch.equal(output, native)
    assert diagnostics["matched_query_fraction"] == 0.0
    for value, snapshot in zip(
        (native, query, memory_key, memory_delta, owner), snapshots
    ):
        assert torch.equal(value, snapshot)


def test_retrieved_residual_is_rms_clipped_against_native_output():
    native = torch.ones(1, 1, 1, 2)
    query = torch.tensor([[[[1.0, 0.0]]]])
    memory_key = query.clone()
    memory_delta = torch.full_like(query, 10.0)
    support = torch.ones(1, 1, dtype=torch.bool)
    owner = torch.ones(1, 1)

    output, diagnostics = immutable_delta_v_memory_attention(
        native, query, memory_key, memory_delta, support, owner,
        topk=1, min_similarity=0.0, strength=1.0, max_rms_ratio=0.5,
    )

    torch.testing.assert_close(output, torch.full_like(output, 1.5))
    torch.testing.assert_close(
        diagnostics["memory_to_native_rms"], torch.tensor(0.5)
    )
    torch.testing.assert_close(
        diagnostics["applied_correction_rms"], torch.tensor(0.5)
    )
    assert diagnostics["cap_fraction"] == 1.0


def test_zero_strength_preserves_native_output_even_when_retrieval_matches():
    native = torch.randn(1, 2, 2, 3)
    query = torch.randn_like(native)
    memory_key = query.clone()
    memory_delta = torch.randn_like(native)
    support = torch.ones(1, 2, dtype=torch.bool)
    owner = torch.ones(1, 2)

    output, diagnostics = immutable_delta_v_memory_attention(
        native, query, memory_key, memory_delta, support, owner,
        topk=1, min_similarity=0.0, strength=0.0, max_rms_ratio=1.0,
    )

    assert torch.equal(output, native)
    assert diagnostics["matched_query_fraction"] == 1.0
    assert diagnostics["applied_correction_rms"] == 0.0


def test_invalid_bank_shapes_and_retrieval_limits_are_rejected():
    source = torch.ones(1, 2, 1, 2)
    target = source + 1.0
    bank = ImmutableDeltaVBank([0])
    with pytest.raises(ValueError, match="token counts"):
        bank.freeze(
            source_kv_cache=make_cache(source),
            target_kv_cache=make_cache(target),
            source_keys={0: source},
            support=torch.ones(1, 3, dtype=torch.bool),
        )

    with pytest.raises(ValueError, match="topk"):
        immutable_delta_v_memory_attention(
            source, source, source, target,
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(1, 2), topk=0,
        )
