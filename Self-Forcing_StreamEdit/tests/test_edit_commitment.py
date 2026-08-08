import importlib.util
from pathlib import Path
import sys
import types

import torch


PIPELINE_ROOT = Path(__file__).parents[1] / "pipeline"
PACKAGE_NAME = "edit_commitment_test_pipeline"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(PIPELINE_ROOT)]
sys.modules[PACKAGE_NAME] = package


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


control_belief = _load_module(
    f"{PACKAGE_NAME}.control_belief",
    PIPELINE_ROOT / "control_belief.py",
)
commitment_module = _load_module(
    f"{PACKAGE_NAME}.edit_commitment",
    PIPELINE_ROOT / "edit_commitment.py",
)


def _belief(edit=0.0, preserve=1.0, visibility=1.0):
    def value(x):
        return torch.full((1, 1, 4, 4), x, dtype=torch.float32)

    return control_belief.CausalControlBelief(
        edit_belief=value(edit),
        preserve_belief=value(preserve),
        edit_precision=value(1.0),
        preserve_precision=value(1.0),
        visibility=value(visibility),
        uncertainty=value(0.0),
        conflict=value(edit * preserve),
    )


def _debug(attention=1.0, proximity=0.0):
    return {
        "source_attention": torch.full(
            (1, 1, 2, 2),
            attention,
        ),
        "hand_proximity": torch.full(
            (1, 1, 2, 2),
            proximity,
        ),
    }


def _features(dtype=torch.float32):
    return torch.eye(4, dtype=dtype).unsqueeze(0)


def _hand(present=True):
    hand = torch.zeros((1, 1, 4, 4))
    if present:
        hand[:, :, :2, :2] = 1.0
    return hand


def test_hand_interaction_triggers_edit_commitment():
    result = commitment_module.EditCommitmentController()(
        belief=_belief(edit=0.8),
        debug=_debug(),
        hand_mask=_hand(),
        source_features=_features(),
    )

    assert result.trigger.max() > 0.5
    assert torch.count_nonzero(result.trigger) == 1
    assert result.effective_commitment.max() > 0.5
    assert result.commitment.max() > result.effective_commitment.max()
    assert result.belief.edit_belief.max() > 0.8


def test_commitment_persists_after_hand_leaves_while_object_is_present():
    controller = commitment_module.EditCommitmentController()
    controller(
        belief=_belief(edit=0.8),
        debug=_debug(),
        hand_mask=_hand(),
        source_features=_features(),
    )
    result = controller(
        belief=_belief(edit=0.0, visibility=0.0),
        debug=_debug(attention=1.0, proximity=0.0),
        hand_mask=_hand(present=False),
        source_features=_features(),
    )

    assert result.trigger.max() == 0
    assert result.semantic_presence.max() > 0.9
    assert torch.count_nonzero(result.semantic_absence) == 0
    assert result.effective_commitment.max() > 0.5
    assert result.belief.edit_belief.max() > 0.5
    assert result.belief.preserve_belief.min() < 0.5


def test_commitment_closes_when_semantic_object_disappears():
    controller = commitment_module.EditCommitmentController()
    controller(
        belief=_belief(edit=0.8),
        debug=_debug(),
        hand_mask=_hand(),
        source_features=_features(),
    )
    result = controller(
        belief=_belief(edit=0.0, visibility=0.0),
        debug=_debug(attention=0.0, proximity=0.0),
        hand_mask=_hand(present=False),
        source_features=_features(),
    )

    assert torch.count_nonzero(result.effective_commitment) == 0
    assert result.semantic_absence.max() > 0.9
    edit_strength = (
        result.belief.edit_belief
        * result.belief.edit_precision
    )
    assert torch.count_nonzero(edit_strength) == 0
    assert torch.allclose(
        result.belief.preserve_belief,
        torch.ones_like(result.belief.preserve_belief),
    )


def test_commitment_does_not_start_without_hand_interaction():
    result = commitment_module.EditCommitmentController()(
        belief=_belief(edit=0.8),
        debug=_debug(attention=1.0, proximity=0.0),
        hand_mask=_hand(present=False),
        source_features=_features(),
    )

    assert torch.count_nonzero(result.trigger) == 0
    assert torch.count_nonzero(result.effective_commitment) == 0


def test_commitment_keeps_contact_preservation_responsibility():
    result = commitment_module.EditCommitmentController()(
        belief=_belief(edit=0.8, preserve=1.0),
        debug=_debug(),
        hand_mask=_hand(),
        source_features=_features(),
    )
    hand = _hand().bool()

    assert torch.allclose(
        result.belief.preserve_belief[hand],
        torch.ones_like(result.belief.preserve_belief[hand]),
    )
    assert result.belief.preserve_belief[0, 0, -1, -1] == 1.0
    assert result.belief.preserve_belief[0, 0, 0, 2] < 1.0


def test_weak_presence_does_not_compound_state_precision():
    controller = commitment_module.EditCommitmentController()
    result = controller(
        belief=_belief(edit=0.8),
        debug=_debug(),
        hand_mask=_hand(),
        source_features=_features(),
    )
    initial_state_precision = result.state_precision.max()

    for _ in range(5):
        result = controller(
            belief=_belief(edit=0.0, visibility=0.0),
            debug=_debug(attention=0.25, proximity=0.0),
            hand_mask=_hand(present=False),
            source_features=_features(),
        )

    assert torch.allclose(
        result.state_precision.max(),
        initial_state_precision,
        atol=1e-6,
    )
    assert result.commitment_precision.max() > 0
    assert result.effective_commitment.max() > 0


def test_confident_absence_suppresses_action_but_keeps_latent_state():
    controller = commitment_module.EditCommitmentController()
    initial = controller(
        belief=_belief(edit=0.8),
        debug=_debug(),
        hand_mask=_hand(),
        source_features=_features(),
    )
    absent = controller(
        belief=_belief(edit=0.0, visibility=0.0),
        debug=_debug(attention=0.0, proximity=0.0),
        hand_mask=_hand(present=False),
        source_features=_features(),
    )
    recovered = controller(
        belief=_belief(edit=0.0, visibility=0.0),
        debug=_debug(attention=1.0, proximity=0.0),
        hand_mask=_hand(present=False),
        source_features=_features(),
    )

    assert torch.count_nonzero(absent.effective_commitment) == 0
    assert torch.allclose(
        absent.state_precision.max(),
        initial.state_precision.max(),
        atol=1e-6,
    )
    assert recovered.effective_commitment.max() > 0.5


def test_interaction_anchor_recovers_lost_short_term_commitment():
    controller = commitment_module.EditCommitmentController()
    controller(
        belief=_belief(edit=0.8),
        debug=_debug(),
        hand_mask=_hand(),
        source_features=_features(),
    )
    controller.previous_commitment.zero_()
    controller.previous_precision.zero_()

    recovered = controller(
        belief=_belief(edit=0.0, visibility=0.0),
        debug=_debug(attention=1.0, proximity=0.0),
        hand_mask=_hand(present=False),
        source_features=_features(),
    )

    assert recovered.transported.max() == 0
    assert recovered.anchor_transport.max() > 0.5
    assert recovered.anchor_precision.max() > 0.5
    assert recovered.effective_commitment.max() > 0.5


def test_commitment_uses_fp32_state_with_bfloat16_features():
    result = commitment_module.EditCommitmentController()(
        belief=_belief(edit=0.8),
        debug=_debug(),
        hand_mask=_hand(),
        source_features=_features(dtype=torch.bfloat16),
    )

    assert result.commitment.dtype == torch.float32
    assert result.belief.edit_belief.dtype == torch.float32
