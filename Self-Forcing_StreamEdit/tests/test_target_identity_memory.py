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
    assert torch.all(memory.states[0].evidence[valid_evidence] == 1)
    assert torch.all(
        update.accumulated_evidence[0][valid_evidence] == 1
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


def test_identity_memory_prefers_captured_raw_keys():
    cached_key = torch.tensor(
        [[[[0.0, 1.0]]] * 4]
    )
    raw_key = torch.tensor(
        [[[[1.0, 0.0]]] * 4]
    )
    value = torch.randn_like(cached_key)
    cache = _cache(cached_key, value)
    cache[0]["current_identity_key"] = raw_key
    memory = identity_module.SlowTargetIdentityMemory(
        layers=(0,),
        num_prototypes=1,
    )

    memory.update(
        cache,
        write_weight=torch.ones((1, 4)),
    )
    prototype = memory.states[0].key.float()

    raw_similarity = torch.nn.functional.cosine_similarity(
        prototype,
        raw_key[:, :1],
        dim=-1,
    )
    cached_similarity = torch.nn.functional.cosine_similarity(
        prototype,
        cached_key[:, :1],
        dim=-1,
    )
    assert raw_similarity.min() > 0.99
    assert cached_similarity.max() < 0.01


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
