import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]


def load_attention_module():
    spec = importlib.util.spec_from_file_location(
        "streamedit_attention_m2_standalone",
        ROOT / "wan/modules/attention.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


closed_loop_delta_v_memory_attention = (
    load_attention_module().closed_loop_delta_v_memory_attention
)


def run_closed_loop(
    native, query, current_key, source_value, target_value,
    canonical_key, canonical_delta, owner, **kwargs,
):
    return closed_loop_delta_v_memory_attention(
        native_output=native,
        current_source_query=query,
        current_source_key=current_key,
        current_source_value=source_value,
        current_target_value=target_value,
        canonical_source_key=canonical_key,
        canonical_delta_value=canonical_delta,
        canonical_support=torch.ones(
            canonical_key.shape[:2], dtype=torch.bool
        ),
        owner_gate=owner,
        topk=kwargs.pop("topk", 1),
        min_similarity=kwargs.pop("min_similarity", 0.0),
        strength=kwargs.pop("strength", 1.0),
        max_error_ratio=kwargs.pop("max_error_ratio", 10.0),
        **kwargs,
    )


def test_matching_current_response_abstains_exactly():
    native = torch.randn(1, 2, 1, 2, dtype=torch.bfloat16)
    query = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=torch.bfloat16
    )
    source = torch.zeros_like(query)
    target = torch.tensor(
        [[[[2.0, 0.0]], [[0.0, 3.0]]]], dtype=torch.bfloat16
    )
    owner = torch.ones(1, 2)

    output, diagnostics = run_closed_loop(
        native, query, query, source, target, query, target, owner
    )

    assert torch.equal(output, native)
    assert diagnostics["matched_query_fraction"] == 1.0
    assert diagnostics["raw_error_rms"] == 0.0
    assert diagnostics["applied_correction_rms"] == 0.0


def test_current_response_is_retrieved_by_current_source_key_not_position():
    native = torch.ones(1, 2, 1, 2)
    query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    canonical_key = query.clone()
    canonical_delta = torch.tensor(
        [[[[2.0, 0.0]], [[0.0, 4.0]]]]
    )
    # Current source coordinates are permuted.  Their target-source values
    # move with the keys, so source-addressed retrieval should still recover
    # the same response as the frozen canonical bank.
    current_key = current_key_expected = query[:, [1, 0]].clone()
    current_delta = canonical_delta[:, [1, 0]].clone()
    source = torch.zeros_like(current_delta)
    owner = torch.ones(1, 2)

    output, diagnostics = run_closed_loop(
        native, query, current_key_expected, source, current_delta,
        canonical_key, canonical_delta, owner,
    )

    assert torch.equal(output, native)
    assert diagnostics["raw_error_rms"] == 0.0
    assert diagnostics["desired_current_cosine"] == 1.0
    assert torch.equal(current_key, current_key_expected)


def test_missing_current_delta_receives_only_closed_loop_error():
    native = torch.ones(1, 1, 1, 2)
    query = torch.tensor([[[[1.0, 0.0]]]])
    source = torch.zeros_like(query)
    current_target = torch.tensor([[[[0.5, 0.0]]]])
    desired = torch.tensor([[[[2.0, 0.0]]]])
    owner = torch.ones(1, 1)

    output, diagnostics = run_closed_loop(
        native, query, query, source, current_target, query, desired, owner,
        strength=0.2,
    )

    expected = native + 0.2 * (desired - current_target)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(
        diagnostics["applied_correction_rms"],
        (0.2 * (desired - current_target)).square().mean().sqrt(),
    )


def test_non_owner_and_low_similarity_queries_are_bit_exact_native():
    native = torch.randn(1, 2, 1, 2)
    query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    current_key = query.clone()
    source = torch.zeros_like(query)
    target = torch.ones_like(query)
    canonical_key = torch.tensor([[[[-1.0, 0.0]]]])
    canonical_delta = torch.full_like(canonical_key, 3.0)
    owner = torch.tensor([[1.0, 0.0]])

    output, diagnostics = run_closed_loop(
        native, query, current_key, source, target, canonical_key,
        canonical_delta, owner, min_similarity=0.8,
    )

    assert torch.equal(output, native)
    assert diagnostics["matched_query_fraction"] == 0.0


def test_closed_loop_error_is_rms_clipped():
    native = torch.ones(1, 1, 1, 2)
    query = torch.tensor([[[[1.0, 0.0]]]])
    source = torch.zeros_like(query)
    current_target = torch.zeros_like(query)
    desired = torch.full_like(query, 10.0)
    owner = torch.ones(1, 1)

    output, diagnostics = run_closed_loop(
        native, query, query, source, current_target, query, desired, owner,
        max_error_ratio=0.5,
    )

    # Desired RMS is 10 and current RMS is zero, hence the error cap is 5.
    torch.testing.assert_close(output, torch.full_like(output, 6.0))
    assert diagnostics["cap_fraction"] == 1.0
    torch.testing.assert_close(
        diagnostics["clipped_error_rms"], torch.tensor(5.0)
    )
