import importlib.util
from pathlib import Path
import sys
import types

import pytest
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


def test_reference_bootstrap_starts_commitment_without_hand():
    controller = commitment_module.EditCommitmentController(topk=1)
    reference_precision = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]]
    )
    controller.bootstrap_reference(
        source_features=_features(),
        edit_precision=reference_precision,
    )

    result = controller(
        belief=_belief(edit=0.0, visibility=0.0),
        debug=_debug(attention=1.0, proximity=0.0),
        hand_mask=_hand(present=False),
        source_features=_features(),
    )

    assert torch.count_nonzero(result.trigger) == 0
    assert result.transported.max() == 1
    assert result.transport_precision.max() == 1
    assert torch.count_nonzero(result.anchor_transport) == 0
    assert torch.count_nonzero(result.anchor_precision) == 0
    assert result.effective_commitment.max() == 1
    assert result.belief.edit_belief.max() == 1
    assert result.belief.preserve_belief.min() == 0
    assert controller.last_spatial_radius == 1
    assert controller.reference_support_budget.item() == 1


def test_reference_track_is_not_overwritten_by_online_trigger():
    controller = commitment_module.EditCommitmentController(topk=1)
    controller.bootstrap_reference(
        source_features=_features(),
        edit_precision=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]]
        ),
    )
    hand = torch.zeros((1, 1, 4, 4))
    hand[:, :, -2:, -2:] = 1.0

    triggered = controller(
        belief=_belief(edit=0.8),
        debug=_debug(attention=1.0, proximity=0.0),
        hand_mask=hand,
        source_features=_features(),
    )

    assert triggered.transported[0, 0, 0, 0] == 1
    assert triggered.trigger[0, 0, 1, 1] > 0
    assert triggered.commitment[0, 0, 0, 0] == 1
    assert triggered.commitment[0, 0, 1, 1] > 0
    assert torch.count_nonzero(triggered.state_precision) == 1
    assert torch.count_nonzero(controller.previous_precision) == 1
    assert controller.previous_commitment[0, 0] == 1
    assert controller.previous_commitment[0, 3] == 0

    persisted = controller(
        belief=_belief(edit=0.0, visibility=0.0),
        debug=_debug(attention=1.0, proximity=0.0),
        hand_mask=_hand(present=False),
        source_features=_features(),
    )

    assert torch.count_nonzero(persisted.state_precision) == 1
    assert persisted.commitment[0, 0, 0, 0] == 1
    assert persisted.effective_commitment[0, 0, 0, 0] == 1


def test_empty_reference_bootstrap_does_not_create_edit_support():
    controller = commitment_module.EditCommitmentController(topk=1)
    controller.bootstrap_reference(
        source_features=_features(),
        edit_precision=torch.zeros((1, 4)),
    )

    result = controller(
        belief=_belief(edit=0.0, visibility=0.0),
        debug=_debug(attention=1.0, proximity=0.0),
        hand_mask=_hand(present=False),
        source_features=_features(),
    )

    assert torch.count_nonzero(result.effective_commitment) == 0
    assert torch.count_nonzero(result.edit_support) == 0


def test_reference_commitment_bootstrap_is_single_use():
    controller = commitment_module.EditCommitmentController()
    controller.bootstrap_reference(
        source_features=_features(),
        edit_precision=torch.ones((1, 4)),
    )

    with pytest.raises(RuntimeError, match="already bootstrapped"):
        controller.bootstrap_reference(
            source_features=_features(),
            edit_precision=torch.ones((1, 4)),
        )


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


def test_transport_splats_only_from_committed_reference_tokens():
    controller = commitment_module.EditCommitmentController(topk=1)
    reference_features = _features()
    current_features = reference_features[:, [1, 2, 0, 3]]
    reference_commitment = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    reference_precision = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    transported, precision, match = controller._transport(
        current_features,
        reference_features,
        reference_commitment,
        reference_precision,
    )

    assert transported.argmax(dim=-1).item() == 2
    assert precision.argmax(dim=-1).item() == 2
    assert transported[0, 2] > 0.99
    assert precision[0, 2] > 0.99
    assert torch.count_nonzero(precision) == 1
    assert match[0, 2] > 0.99
    assert torch.count_nonzero(match) == 1


def test_reference_transport_rejects_distant_global_match():
    controller = commitment_module.EditCommitmentController(topk=1)
    reference_features = torch.tensor(
        [[[-1.0, 0.0]] * 9]
    )
    reference_features[:, 0] = torch.tensor([1.0, 0.0])
    current_features = torch.tensor(
        [[[0.5, 0.0]] * 9]
    )
    current_features[:, 1] = torch.tensor([0.9, 0.0])
    current_features[:, 8] = torch.tensor([1.0, 0.0])
    reference_commitment = torch.zeros((1, 9))
    reference_commitment[:, 0] = 1.0
    reference_precision = reference_commitment.clone()

    _, global_precision, _ = controller._transport(
        current_features,
        reference_features,
        reference_commitment,
        reference_precision,
    )
    _, local_precision, _ = controller._transport(
        current_features,
        reference_features,
        reference_commitment,
        reference_precision,
        spatial_shape=(3, 3),
        spatial_radius=1,
    )

    assert global_precision.argmax(dim=-1).item() == 8
    assert local_precision.argmax(dim=-1).item() == 1
    assert global_precision[0, 8] > 0
    assert local_precision[0, 8] == 0
    assert local_precision[0, 1] > 0


def test_reference_precision_is_calibrated_over_active_matches():
    controller = commitment_module.EditCommitmentController()
    precision = torch.tensor(
        [[0.0, 0.02, 0.05, 0.10]]
    )

    calibrated, scale = (
        controller._calibrate_active_precision(precision)
    )

    assert calibrated[0, 0] == 0
    assert calibrated[0, 3] == 1
    assert calibrated[0, 2] > calibrated[0, 1] > 0
    assert 0.05 < scale.item() < 0.10


def test_reference_transport_keeps_initial_support_budget():
    controller = commitment_module.EditCommitmentController()
    controller.reference_support_budget = torch.tensor([2])
    transported = torch.ones((1, 5))
    precision = torch.tensor(
        [[0.1, 0.8, 0.3, 0.9, 0.2]]
    )
    match = torch.ones_like(precision)

    transported, precision, match = (
        controller._prune_to_reference_budget(
            transported,
            precision,
            match,
        )
    )

    assert torch.count_nonzero(precision) == 2
    assert precision[0, 1] == 0.8
    assert precision[0, 3] == 0.9
    assert transported[0, 0] == 0
    assert match[0, 4] == 0


def test_commitment_uses_fp32_state_with_bfloat16_features():
    result = commitment_module.EditCommitmentController()(
        belief=_belief(edit=0.8),
        debug=_debug(),
        hand_mask=_hand(),
        source_features=_features(dtype=torch.bfloat16),
    )

    assert result.commitment.dtype == torch.float32
    assert result.belief.edit_belief.dtype == torch.float32
