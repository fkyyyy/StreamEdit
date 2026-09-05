import importlib.util
from pathlib import Path

import torch


def load_attention_module():
    path = (
        Path(__file__).parents[1] / "wan" / "modules" / "attention.py"
    )
    spec = importlib.util.spec_from_file_location(
        "streamedit_attention_standalone", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_segment_mass_matches_full_softmax():
    module = load_attention_module()
    torch.manual_seed(0)
    query = torch.randn(6, 3, 4)
    key = torch.randn(9, 3, 4)
    foreground = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.bool)

    result = module.diagnose_attention_segment(
        query,
        key,
        segment_start=2,
        segment_end=5,
        foreground_mask=foreground,
        max_query_samples=16,
        key_chunk_size=2,
    )

    logits = torch.einsum(
        "qhd,khd->hqk", query.float(), key.float()
    ) * (query.shape[-1] ** -0.5)
    exact = logits.softmax(dim=-1)[..., 2:5].sum(dim=-1)

    torch.testing.assert_close(
        result["foreground"], exact[:, foreground].mean()
    )
    torch.testing.assert_close(
        result["background"], exact[:, ~foreground].mean()
    )
    torch.testing.assert_close(result["all"], exact.mean())
    torch.testing.assert_close(
        result["all_per_head"], exact.mean(dim=1)
    )


def test_query_sampling_is_bounded_and_deterministic():
    module = load_attention_module()
    mask = torch.ones(100, dtype=torch.bool)
    first = module._evenly_spaced_mask_indices(mask, 4)
    second = module._evenly_spaced_mask_indices(mask, 4)

    assert first.tolist() == [0, 33, 66, 99]
    assert torch.equal(first, second)


def test_block_summary_reports_target_to_source_rms_ratio():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "summarize_source_bg_attention.py"
    )
    spec = importlib.util.spec_from_file_location(
        "source_bg_attention_summary_standalone", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = {name: 1.0 for name in module.SCALARS}
    rows = [
        {**base, "block": 1, "target_value_rms": 2.0},
        {**base, "block": 1, "target_value_rms": 4.0},
    ]

    summary = module.aggregate(rows, ("block",))

    assert len(summary) == 1
    assert summary[0]["target_value_rms"] == 3.0
    assert summary[0]["target_to_source_value_rms"] == 3.0


def test_source_background_route_preserves_native_pair():
    module = load_attention_module()
    source_key = torch.arange(12).reshape(3, 2, 2)
    source_value = source_key + 100
    target_value = source_key + 200
    background = torch.tensor([True, False, True])

    key, value = module.route_source_background_kv(
        source_key, source_value, target_value, background
    )

    torch.testing.assert_close(key, source_key[background])
    torch.testing.assert_close(value, source_value[background])


def test_source_background_route_drops_key_value_as_pair():
    module = load_attention_module()
    tensor = torch.randn(3, 2, 4)

    routed = module.route_source_background_kv(
        tensor,
        tensor + 1,
        tensor + 2,
        torch.tensor([True, False, True]),
        drop_pair=True,
    )

    assert routed is None


def test_source_background_route_preserves_967h_value_ablation():
    module = load_attention_module()
    source_key = torch.arange(12).reshape(3, 2, 2)
    source_value = source_key + 100
    target_value = source_key + 200
    background = torch.tensor([True, False, True])

    key, value = module.route_source_background_kv(
        source_key,
        source_value,
        target_value,
        background,
        suppress_value=True,
    )

    torch.testing.assert_close(key, source_key[background])
    torch.testing.assert_close(value, target_value[background])


def test_source_background_route_rejects_conflicting_ablations():
    module = load_attention_module()
    tensor = torch.randn(3, 2, 4)

    try:
        module.route_source_background_kv(
            tensor,
            tensor,
            tensor,
            torch.ones(3, dtype=torch.bool),
            suppress_value=True,
            drop_pair=True,
        )
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("conflicting source-bg routes must fail")
