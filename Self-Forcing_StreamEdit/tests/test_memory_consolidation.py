import importlib.util
from pathlib import Path
import sys

import torch


PIPELINE_ROOT = Path(__file__).parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


memory_module = _load_module(
    "memory_consolidation",
    PIPELINE_ROOT / "pipeline" / "memory_consolidation.py",
)
belief_kv = _load_module(
    "memory_consolidation_belief_kv",
    PIPELINE_ROOT / "pipeline" / "belief_kv.py",
)


class _Belief:
    def __init__(
        self,
        edit,
        preserve,
        edit_precision,
        preserve_precision,
        uncertainty,
        visibility=None,
    ):
        self.edit_belief = edit
        self.preserve_belief = preserve
        self.edit_precision = edit_precision
        self.preserve_precision = preserve_precision
        self.uncertainty = uncertainty
        self.visibility = (
            torch.ones_like(edit)
            if visibility is None
            else visibility
        )
        self.conflict = edit * preserve

    def validate(self):
        values = (
            self.edit_belief,
            self.preserve_belief,
            self.edit_precision,
            self.preserve_precision,
            self.uncertainty,
            self.visibility,
            self.conflict,
        )
        assert len({tuple(value.shape) for value in values}) == 1


def _belief(
    edit_action=0.8,
    uncertainty=0.2,
    visibility=1.0,
):
    edit = torch.full((1, 1, 4, 4), edit_action)
    preserve = torch.full_like(edit, 1.0 - edit_action)
    precision = torch.ones_like(edit)
    return _Belief(
        edit,
        preserve,
        precision,
        precision,
        torch.full_like(edit, uncertainty),
        torch.full_like(edit, visibility),
    )


def _weights(belief):
    return belief_kv.build_belief_kv_weights(
        belief,
        expected_token_length=4,
    )


def _features():
    return torch.eye(4).unsqueeze(0)


def test_first_observation_is_not_suppressed_without_history():
    belief = _belief(edit_action=0.8, uncertainty=0.2)
    plan = memory_module.CausalMemoryConsolidator()(
        belief,
        _weights(belief),
        _features(),
    )

    assert torch.allclose(
        plan.observation_gain,
        torch.ones_like(plan.observation_gain),
    )
    assert torch.allclose(
        plan.consolidated_edit_action,
        plan.observation_edit_action,
    )
    assert torch.allclose(
        plan.materialized_edit_action,
        plan.consolidated_edit_action
        * plan.consolidated_precision,
    )
    assert torch.count_nonzero(plan.reference_valid) == 0


def test_uncertain_first_observation_has_low_write_responsibility():
    belief = _belief(edit_action=0.8, uncertainty=0.9)
    plan = memory_module.CausalMemoryConsolidator()(
        belief,
        _weights(belief),
        _features(),
    )

    assert torch.allclose(
        plan.consolidated_edit_action,
        plan.observation_edit_action,
    )
    assert plan.materialized_edit_action.mean() < 0.1


def test_uncertain_observation_retains_reliable_transported_memory():
    consolidator = memory_module.CausalMemoryConsolidator()
    reliable = _belief(edit_action=0.9, uncertainty=0.0)
    consolidator(reliable, _weights(reliable), _features())

    uncertain = _belief(edit_action=0.1, uncertainty=0.9)
    plan = consolidator(
        uncertain,
        _weights(uncertain),
        _features(),
    )

    assert plan.transported_edit_action.mean() > 0.8
    assert plan.observation_gain.mean() < 0.2
    assert plan.consolidated_edit_action.mean() > 0.7
    assert plan.materialized_edit_action.mean() > 0.7


def test_reliable_observation_overrides_low_precision_history():
    consolidator = memory_module.CausalMemoryConsolidator()
    uncertain = _belief(edit_action=0.9, uncertainty=0.9)
    consolidator(uncertain, _weights(uncertain), _features())

    reliable = _belief(edit_action=0.1, uncertainty=0.0)
    plan = consolidator(
        reliable,
        _weights(reliable),
        _features(),
    )

    assert plan.observation_gain.mean() > 0.8
    assert plan.consolidated_edit_action.mean() < 0.2


def test_invisibility_closes_transported_target_memory():
    consolidator = memory_module.CausalMemoryConsolidator()
    visible = _belief(edit_action=0.9, uncertainty=0.0)
    consolidator(visible, _weights(visible), _features())

    invisible = _belief(
        edit_action=0.0,
        uncertainty=0.0,
        visibility=0.0,
    )
    plan = consolidator(
        invisible,
        _weights(invisible),
        _features(),
    )

    assert torch.count_nonzero(plan.transported_edit_action) == 0
    assert torch.count_nonzero(plan.consolidated_edit_action) == 0
    assert torch.count_nonzero(plan.materialized_edit_action) == 0
