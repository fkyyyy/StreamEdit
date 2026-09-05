import importlib.util
from pathlib import Path

import pytest
import torch


def load_attention_module():
    path = Path(__file__).parents[1] / "wan" / "modules" / "attention.py"
    spec = importlib.util.spec_from_file_location(
        "streamedit_attention_counterfactual_standalone", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


counterfactual_replace_attention_segment = (
    load_attention_module().counterfactual_replace_attention_segment
)
build_counterfactual_segment_values = (
    load_attention_module().build_counterfactual_segment_values
)


def query_rms(value):
    return value.float().square().mean(dim=(2, 3)).sqrt()


def test_counterfactual_value_rows_are_zero_outside_exact_segment():
    native = torch.arange(30).reshape(5, 2, 3)
    target = native[1:4] + 100

    rows = build_counterfactual_segment_values(
        native, target, segment_start=1, segment_end=4
    )

    assert rows.shape == (2, 5, 2, 3)
    assert torch.equal(rows[0, 1:4], native[1:4])
    assert torch.equal(rows[1, 1:4], target)
    assert torch.count_nonzero(rows[:, [0, 4]]) == 0


def test_replaces_only_foreground_queries_and_matches_source_rms():
    torch.manual_seed(0)
    native = torch.randn(1, 4, 2, 3)
    source = torch.randn_like(native)
    target = 0.75 * torch.randn_like(native)
    foreground = torch.tensor([[False, True, True, False]])

    output, diagnostics = counterfactual_replace_attention_segment(
        native, source, target, foreground
    )

    assert torch.equal(output[:, ~foreground[0]], native[:, ~foreground[0]])
    inserted = output - native + source
    torch.testing.assert_close(
        query_rms(inserted)[:, foreground[0]],
        query_rms(source)[:, foreground[0]],
    )
    torch.testing.assert_close(
        diagnostics["foreground_fraction"], torch.tensor(0.5)
    )
    torch.testing.assert_close(
        diagnostics["active_fraction"], torch.tensor(0.5)
    )


def test_counterfactual_direction_is_target_not_source():
    native = torch.zeros(1, 1, 1, 2)
    source = torch.tensor([[[[2.0, 0.0]]]])
    target = torch.tensor([[[[0.0, 1.0]]]])
    foreground = torch.ones(1, 1, dtype=torch.bool)

    output, _ = counterfactual_replace_attention_segment(
        native, source, target, foreground
    )

    # Native minus source contribution plus the RMS-matched target direction.
    torch.testing.assert_close(
        output, torch.tensor([[[[-2.0, 2.0]]]])
    )


def test_zero_target_contribution_abstains_bit_exactly():
    native = torch.randn(1, 3, 2, 2, dtype=torch.bfloat16)
    source = torch.randn_like(native)
    target = torch.zeros_like(native)
    foreground = torch.ones(1, 3, dtype=torch.bool)

    output, diagnostics = counterfactual_replace_attention_segment(
        native, source, target, foreground
    )

    assert torch.equal(output, native)
    assert diagnostics["degenerate_target_fraction"] == 1.0
    assert output.dtype == native.dtype


def test_scale_guard_limits_near_zero_target():
    native = torch.zeros(1, 1, 1, 1)
    source = torch.ones_like(native)
    target = torch.full_like(native, 0.01)
    foreground = torch.ones(1, 1, dtype=torch.bool)

    output, diagnostics = counterfactual_replace_attention_segment(
        native, source, target, foreground, max_scale=4.0
    )

    torch.testing.assert_close(output, torch.tensor([[[[-0.96]]]]))
    torch.testing.assert_close(
        diagnostics["replacement_scale"], torch.tensor(4.0)
    )
    assert diagnostics["scale_capped_fraction"] == 1.0


def test_rejects_invalid_shapes_and_limits():
    tensor = torch.ones(1, 2, 1, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="must align"):
        counterfactual_replace_attention_segment(
            tensor, tensor[:, :1], tensor, mask
        )
    with pytest.raises(ValueError, match="foreground_mask"):
        counterfactual_replace_attention_segment(
            tensor, tensor, tensor, torch.ones(2, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="max_scale"):
        counterfactual_replace_attention_segment(
            tensor, tensor, tensor, mask, max_scale=0.5
        )
