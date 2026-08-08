import importlib.util
from pathlib import Path
import sys

import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "pipeline"
    / "control_belief.py"
)
SPEC = importlib.util.spec_from_file_location(
    "control_belief",
    MODULE_PATH,
)
control_belief = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = control_belief
SPEC.loader.exec_module(control_belief)


def _debug(object_value=0.8, visible=1.0):
    shape = (1, 1, 4, 4)
    posterior = torch.zeros(shape)
    posterior[:, :, 1:3, 1:3] = object_value
    attention = torch.zeros(shape)
    attention[:, :, 1:3, 1:3] = 1.0
    proximity = torch.zeros(shape)
    proximity[:, :, 1:3, 1:3] = 1.0
    return {
        "object_posterior": posterior,
        "source_attention": attention,
        "temporal_confidence": posterior,
        "object_visible": torch.full((1, 1, 1, 1), visible),
        "hand_proximity": proximity,
        "adaptive_attention_reliability": torch.full(
            (1, 1, 1, 1),
            0.8,
        ),
        "field_score": attention,
        "adaptive_field_reliability": torch.full(
            (1, 1, 1, 1),
            0.5,
        ),
    }


def _hand():
    hand = torch.zeros((1, 1, 8, 8))
    hand[:, :, 3:6, 3:6] = 1.0
    return hand


def test_control_belief_shapes_ranges_and_finiteness():
    belief = control_belief.CausalControlBeliefBuilder()(
        _debug(),
        _hand(),
    )

    for value in belief.as_dict().values():
        assert value.shape == (1, 1, 8, 8)
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()
        assert value.min() >= 0
        assert value.max() <= 1


def test_contact_has_nonexclusive_edit_and_preserve_beliefs():
    belief = control_belief.CausalControlBeliefBuilder()(
        _debug(),
        _hand(),
    )
    contact = _hand().bool()

    assert belief.edit_belief[contact].mean() > 0.5
    assert belief.preserve_belief[contact].mean() > 0.5
    assert (
        belief.edit_belief[contact]
        + belief.preserve_belief[contact]
    ).mean() > 1.0
    assert belief.conflict[contact].mean() > 0.25


def test_background_has_preservation_without_edit_responsibility():
    debug = _debug(object_value=0.0)
    debug["source_attention"].zero_()
    debug["temporal_confidence"].zero_()
    belief = control_belief.CausalControlBeliefBuilder()(
        debug,
        torch.zeros_like(_hand()),
    )

    assert torch.count_nonzero(belief.edit_belief) == 0
    assert torch.allclose(
        belief.preserve_belief,
        torch.ones_like(belief.preserve_belief),
    )


def test_invisible_object_falls_back_to_preservation():
    belief = control_belief.CausalControlBeliefBuilder()(
        _debug(visible=0.0),
        _hand(),
    )

    assert torch.count_nonzero(belief.edit_belief) == 0
    assert torch.allclose(
        belief.preserve_belief,
        torch.ones_like(belief.preserve_belief),
    )


def test_reliable_low_field_response_reduces_edit_precision():
    low_field = _debug()
    low_field["field_score"].zero_()
    low_field["adaptive_field_reliability"].fill_(1.0)
    high_field = _debug()
    high_field["field_score"].fill_(1.0)
    high_field["adaptive_field_reliability"].fill_(1.0)
    builder = control_belief.CausalControlBeliefBuilder()

    low = builder(low_field, _hand())
    high = builder(high_field, _hand())

    assert low.edit_precision.mean() < high.edit_precision.mean()
    assert low.preserve_precision.mean() > high.preserve_precision.mean()


def test_bfloat16_evidence_is_calibrated_in_float32():
    debug = {
        name: value.to(torch.bfloat16)
        for name, value in _debug().items()
    }
    belief = control_belief.CausalControlBeliefBuilder()(
        debug,
        _hand().to(torch.bfloat16),
    )

    assert belief.edit_belief.dtype == torch.float32
    assert belief.preserve_precision.dtype == torch.float32
