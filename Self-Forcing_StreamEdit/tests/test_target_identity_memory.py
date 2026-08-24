import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).parents[1]
PIPELINE_ROOT = ROOT / "pipeline"
PACKAGE_NAME = "target_identity_test_pipeline"
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
identity_module = _load_module(
    f"{PACKAGE_NAME}.target_identity_memory",
    PIPELINE_ROOT / "target_identity_memory.py",
)
attention_module = _load_module(
    "target_identity_test_attention",
    ROOT / "wan" / "modules" / "attention.py",
)


def _belief(edit=0.2, preserve=1.0):
    def value(number):
        return torch.full(
            (1, 1, 4, 4),
            number,
            dtype=torch.float32,
        )

    return control_belief.CausalControlBelief(
        edit_belief=value(edit),
        preserve_belief=value(preserve),
        edit_precision=value(0.5),
        preserve_precision=value(1.0),
        visibility=value(1.0),
        uncertainty=value(0.5),
        conflict=value(edit * preserve),
    )


def _cache(key, value):
    return [{
        "k": key.clone(),
        "v": value.clone(),
        "local_end_index": torch.tensor([key.shape[1]]),
        "num_new_tokens": key.shape[1],
    }]


def _reference_inputs():
    source = torch.zeros((1, 1, 2, 4, 4))
    target = source.clone()
    target[:, :, :, :2, :2] = 1.0
    attention = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    return source, target, attention


def test_token_propagation_first_block_keeps_base_write():
    propagator = identity_module.CausalObjectTokenPropagator(
        gate_strength=1.0,
    )
    features = torch.tensor([[
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]])
    base_write = torch.tensor([[1.0, 0.5, 0.0]])

    result = propagator(features, base_write)

    assert torch.allclose(result.write_weight, base_write)
    assert not result.has_previous.any()


def test_token_propagation_gates_unmatched_identity_writes():
    propagator = identity_module.CausalObjectTokenPropagator(
        min_similarity=0.5,
        gate_strength=1.0,
    )
    previous_features = torch.tensor([[
        [1.0, 0.0],
        [0.0, 1.0],
    ]])
    propagator(
        previous_features,
        torch.tensor([[1.0, 0.0]]),
    )
    current_features = torch.tensor([[
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ]])

    result = propagator(
        current_features,
        torch.ones((1, 3)),
    )

    assert result.has_previous.all()
    assert result.write_weight[0, 0].item() > 0.99
    assert result.write_weight[0, 1].item() == 0.0
    assert result.write_weight[0, 2].item() == 0.0


def test_token_propagation_keeps_stable_support_across_blocks():
    propagator = identity_module.CausalObjectTokenPropagator(
        min_similarity=0.5,
        gate_strength=1.0,
    )
    features = torch.tensor([[
        [1.0, 0.0],
        [0.0, 1.0],
    ]])

    first = propagator(
        features,
        base_write_weight=torch.tensor([[0.1, 0.0]]),
        support_weight=torch.tensor([[1.0, 0.0]]),
    )
    second = propagator(
        features,
        base_write_weight=torch.ones((1, 2)),
    )

    assert torch.allclose(
        first.support_weight,
        torch.tensor([[1.0, 0.0]]),
    )
    assert second.matched_previous_weight[0, 0].item() == 1.0
    assert second.write_weight[0, 0].item() > 0.99
    assert second.write_weight[0, 1].item() == 0.0


def test_committed_memory_feedback_updates_current_belief():
    belief = _belief(edit=0.1, preserve=1.0)
    hand = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    committed_edit = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    committed_precision = torch.ones_like(committed_edit)

    updated, debug = identity_module.inject_committed_memory_into_belief(
        belief=belief,
        committed_token_edit=committed_edit,
        committed_token_precision=committed_precision,
        hand_mask=hand,
        feedback_strength=0.5,
    )

    assert updated.edit_belief.mean() > belief.edit_belief.mean()
    assert updated.preserve_belief.mean() < belief.preserve_belief.mean()
    assert torch.allclose(
        debug["committed_memory_evidence"],
        torch.full_like(belief.edit_belief, 0.5),
    )


def test_committed_memory_feedback_does_not_release_hand_preserve():
    belief = _belief(edit=0.1, preserve=1.0)
    hand = torch.ones((1, 1, 4, 4), dtype=torch.float32)
    committed_edit = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    committed_precision = torch.ones_like(committed_edit)

    updated, debug = identity_module.inject_committed_memory_into_belief(
        belief=belief,
        committed_token_edit=committed_edit,
        committed_token_precision=committed_precision,
        hand_mask=hand,
        feedback_strength=1.0,
    )

    assert torch.allclose(updated.edit_belief, belief.edit_belief)
    assert torch.allclose(updated.preserve_belief, belief.preserve_belief)
    assert torch.allclose(updated.uncertainty, belief.uncertainty)
    assert torch.count_nonzero(debug["committed_memory_evidence"]) == 0


def test_reference_bootstrap_localizes_semantic_latent_change():
    source, target, attention = _reference_inputs()

    bootstrap = (
        identity_module.build_reference_identity_bootstrap(
            source_latent=source,
            target_latent=target,
            target_attention=attention,
        )
    )

    assert bootstrap.change_score[0, 0, 0, 0] == 1.0
    assert bootstrap.semantic_score[0, 0, 0, 0] == 1.0
    assert bootstrap.joint_score[0, 0, 0, 0] == 1.0
    assert bootstrap.write_weight.argmax(dim=-1).item() == 0
    assert bootstrap.write_weight[0, 0] == 1.0
    assert torch.count_nonzero(
        bootstrap.write_weight[0, 1:]
    ) == 0


def test_reference_component_filter_keeps_one_edited_instance():
    weight = torch.zeros((1, 1, 4, 5))
    weight[0, 0, 0, 0] = 0.8
    weight[0, 0, 0, 1] = 0.8
    weight[0, 0, 2, 2] = 0.3
    weight[0, 0, 2, 3] = 0.3
    weight[0, 0, 3, 3] = 0.3

    selected = identity_module._largest_weighted_component_mask(
        weight,
        eps=1e-6,
    )

    assert torch.count_nonzero(selected) == 2
    assert selected[0, 0, 0, 0]
    assert selected[0, 0, 0, 1]
    assert not selected[0, 0, 2, 2]


def test_reference_component_filter_prefers_hand_contact():
    weight = torch.zeros((1, 1, 4, 5))
    weight[0, 0, 0, :3] = 1.0
    weight[0, 0, 3, 3:] = 0.8
    contact = torch.zeros_like(weight)
    contact[0, 0, 3, 3:] = 1.0

    selected = identity_module._largest_weighted_component_mask(
        weight,
        eps=1e-6,
        hand_contact_score=contact,
    )

    assert torch.count_nonzero(selected) == 2
    assert selected[0, 0, 3, 3]
    assert selected[0, 0, 3, 4]
    assert not selected[0, 0, 0, 0]


def test_reference_bootstrap_excludes_hand_core():
    source, target, attention = _reference_inputs()
    hand = torch.zeros((1, 1, 4, 4))
    hand[:, :, :2, :2] = 1.0

    bootstrap = (
        identity_module.build_reference_identity_bootstrap(
            source_latent=source,
            target_latent=target,
            target_attention=attention,
            hand_mask=hand,
        )
    )

    assert torch.count_nonzero(bootstrap.joint_score) == 0
    assert torch.count_nonzero(bootstrap.write_weight) == 0


def test_reference_bootstrap_requires_semantic_change_agreement():
    source, target, _ = _reference_inputs()
    disjoint_attention = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0]]
    )

    bootstrap = (
        identity_module.build_reference_identity_bootstrap(
            source_latent=source,
            target_latent=target,
            target_attention=disjoint_attention,
        )
    )

    assert torch.count_nonzero(bootstrap.joint_score) == 0
    assert torch.count_nonzero(bootstrap.write_weight) == 0


def test_reference_bootstrap_falls_back_without_semantics():
    source, target, attention = _reference_inputs()

    bootstrap = (
        identity_module.build_reference_identity_bootstrap(
            source_latent=source,
            target_latent=target,
            target_attention=torch.zeros_like(attention),
        )
    )

    assert torch.equal(
        bootstrap.joint_score,
        bootstrap.change_score,
    )
    assert bootstrap.write_weight[0, 0] == 1.0


def test_reference_bootstrap_is_empty_without_latent_change():
    source, _, attention = _reference_inputs()

    bootstrap = (
        identity_module.build_reference_identity_bootstrap(
            source_latent=source,
            target_latent=source.clone(),
            target_attention=attention,
        )
    )

    assert torch.count_nonzero(bootstrap.change_score) == 0
    assert torch.count_nonzero(bootstrap.joint_score) == 0
    assert torch.count_nonzero(bootstrap.write_weight) == 0


def test_reference_bootstrap_validates_alignment():
    source, target, attention = _reference_inputs()

    with pytest.raises(ValueError, match="share shape"):
        identity_module.build_reference_identity_bootstrap(
            source_latent=source,
            target_latent=target[:, :, :, :, :-1],
            target_attention=attention,
        )

    with pytest.raises(ValueError, match="token grid"):
        identity_module.build_reference_identity_bootstrap(
            source_latent=source,
            target_latent=target,
            target_attention=attention[:, :-1],
        )


def test_reference_identity_bootstrap_is_authoritative_and_single_use():
    key = torch.randn((1, 4, 1, 2))
    value = torch.randn_like(key)
    memory = identity_module.SlowTargetIdentityMemory(
        layers=(0,),
        num_prototypes=2,
    )

    update = memory.bootstrap_reference(
        _cache(key, value),
        write_weight=torch.ones((1, 4)),
    )
    valid_evidence = memory.states[0].evidence > 0

    assert memory.reference_bootstrapped
    assert torch.all(
        memory.states[0].evidence[valid_evidence]
        == memory.reference_prior_evidence
    )
    assert torch.all(
        update.accumulated_evidence[0][valid_evidence]
        == memory.reference_prior_evidence
    )
    with pytest.raises(RuntimeError, match="already bootstrapped"):
        memory.bootstrap_reference(
            _cache(key, value),
            write_weight=torch.ones((1, 4)),
        )


def test_reference_identity_resists_low_confidence_online_update():
    key = torch.randn((1, 4, 1, 2))
    value = torch.randn_like(key)
    memory = identity_module.SlowTargetIdentityMemory(
        layers=(0,),
        num_prototypes=2,
    )
    memory.bootstrap_reference(
        _cache(key, value),
        write_weight=torch.ones((1, 4)),
    )
    reference_value = memory.states[0].value.float().clone()

    update = memory.update(
        _cache(key, value + 20.0),
        write_weight=torch.full((1, 4), 0.05),
    )

    assert update.update_gain.max() < 0.1
    assert (
        memory.states[0].value.float() - reference_value
    ).abs().max() < 2.0


def test_slow_identity_update_resists_low_precision_overwrite():
    key = torch.tensor(
        [[
            [[[1.0, 0.0]]],
            [[[0.9, 0.1]]],
            [[[0.0, 1.0]]],
            [[[0.1, 0.9]]],
        ]]
    ).squeeze(2)
    value = key + 1.0
    memory = identity_module.SlowTargetIdentityMemory(
        layers=(0,),
        num_prototypes=2,
    )
    first = memory.update(
        _cache(key, value),
        write_weight=torch.ones((1, 4)),
    )
    initial_value = memory.states[0].value.float().clone()

    contradictory_value = value + 20.0
    second = memory.update(
        _cache(key, contradictory_value),
        write_weight=torch.full((1, 4), 0.1),
    )
    updated_value = memory.states[0].value.float()

    assert first.update_gain.max() == 1.0
    assert second.update_gain.max() < 0.2
    assert (updated_value - initial_value).abs().max() < 4.0
    assert torch.all(
        second.accumulated_evidence
        > first.accumulated_evidence
    )


def test_identity_memory_ignores_empty_write_support():
    key = torch.randn((1, 4, 1, 2))
    value = torch.randn_like(key)
    memory = identity_module.SlowTargetIdentityMemory(
        layers=(0,),
        num_prototypes=2,
    )

    update = memory.update(
        _cache(key, value),
        write_weight=torch.zeros((1, 4)),
    )

    assert torch.count_nonzero(update.observation_evidence) == 0
    assert torch.count_nonzero(memory.states[0].evidence) == 0
    assert torch.count_nonzero(memory.states[0].key) == 0
    assert torch.count_nonzero(memory.states[0].value) == 0


def test_identity_memory_can_bootstrap_after_empty_block():
    key = torch.randn((1, 4, 1, 2))
    value = torch.randn_like(key)
    memory = identity_module.SlowTargetIdentityMemory(
        layers=(0,),
        num_prototypes=2,
    )
    memory.update(
        _cache(key, value),
        write_weight=torch.zeros((1, 4)),
    )

    update = memory.update(
        _cache(key, value),
        write_weight=torch.ones((1, 4)),
    )

    assert torch.count_nonzero(update.observation_evidence) == 2
    assert torch.count_nonzero(memory.states[0].evidence) == 2
    assert torch.count_nonzero(memory.states[0].value) > 0


def test_identity_value_correction_is_match_gated():
    target_key = torch.tensor(
        [[
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
            [[[-1.0, 0.0]]],
            [[[0.0, -1.0]]],
        ]]
    ).squeeze(2)
    target_value = torch.zeros_like(target_key)
    prototype_key = torch.tensor([[[[1.0, 0.0]]]])
    prototype_value = torch.tensor([[[[5.0, 7.0]]]])

    corrected, support = (
        attention_module.apply_target_identity_value_correction(
            target_key,
            target_value,
            prototype_key,
            prototype_value,
            prototype_evidence=torch.ones((1, 1)),
        )
    )

    assert support.argmax(dim=-1).item() == 0
    assert support[0, 0] > 0.9
    assert torch.allclose(
        corrected[0, 0],
        prototype_value[0, 0],
        atol=1e-5,
    )
    assert torch.allclose(
        corrected[0, 2],
        target_value[0, 2],
    )


def test_identity_value_correction_is_noop_without_evidence():
    target_key = torch.randn((1, 4, 1, 2))
    target_value = torch.randn_like(target_key)
    prototype_key = torch.randn((1, 2, 1, 2))
    prototype_value = torch.randn_like(prototype_key)

    corrected, support = (
        attention_module.apply_target_identity_value_correction(
            target_key,
            target_value,
            prototype_key,
            prototype_value,
            prototype_evidence=torch.zeros((1, 2)),
        )
    )

    assert torch.equal(corrected, target_value)
    assert torch.count_nonzero(support) == 0


def test_identity_feedback_preserves_hand_core():
    belief = _belief()
    identity_support = torch.ones((1, 1, 2, 2))
    hand = torch.zeros((1, 1, 4, 4))
    hand[:, :, :2, :2] = 1.0

    updated = (
        identity_module.strengthen_belief_with_target_identity(
            belief,
            identity_support,
            hand,
        )
    )

    assert updated.edit_belief.min() > belief.edit_belief.min()
    assert torch.allclose(
        updated.preserve_belief[hand.bool()],
        belief.preserve_belief[hand.bool()],
    )
    assert updated.preserve_belief[0, 0, -1, -1] == 0.0
    assert updated.visibility.min() == 1.0


def test_identity_correction_supports_bfloat16_values():
    target_key = torch.tensor(
        [[[[1.0, 0.0]], [[-1.0, 0.0]]]],
        dtype=torch.bfloat16,
    )
    target_value = torch.zeros_like(target_key)
    prototype_key = target_key[:, :1].clone()
    prototype_value = torch.ones_like(prototype_key)

    corrected, support = (
        attention_module.apply_target_identity_value_correction(
            target_key,
            target_value,
            prototype_key,
            prototype_value,
            prototype_evidence=torch.ones((1, 1)),
        )
    )

    assert corrected.dtype == torch.bfloat16
    assert support.dtype == torch.float32
