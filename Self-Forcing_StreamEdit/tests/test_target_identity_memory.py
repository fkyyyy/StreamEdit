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


def _cache(key, value, current_identity_key=None):
    cache = {
        "k": key.clone(),
        "v": value.clone(),
        "local_end_index": torch.tensor([key.shape[1]]),
        "num_new_tokens": key.shape[1],
    }
    if current_identity_key is not None:
        cache["current_identity_key"] = (
            current_identity_key.clone()
        )
    return [cache]


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


def test_connected_support_removes_detached_regions_and_tracks_forward():
    support = torch.zeros((1, 2, 6, 7), dtype=torch.float32)
    support[0, 0, 2, 2:4] = 0.9
    support[0, 0, 0, 6] = 1.0
    support[0, 1, 2, 3:5] = 0.8
    support[0, 1, 5, 0] = 1.0
    anchor = torch.zeros_like(support, dtype=torch.bool)
    anchor[0, 0, 2, 2] = True
    support_filter = identity_module.CausalConnectedSupportFilter(
        min_weight=0.05,
        temporal_radius=1,
        max_anchor_ratio=4.0,
        min_area_fraction=0.01,
        max_area_fraction=0.5,
    )

    result = support_filter(support, anchor)

    assert result.keep_mask[0, 0, 2, 2:4].all()
    assert not result.keep_mask[0, 0, 0, 6]
    assert result.keep_mask[0, 1, 2, 3:5].all()
    assert not result.keep_mask[0, 1, 5, 0]
    assert torch.count_nonzero(result.weight) == 4


def test_connected_support_caps_area_growth():
    support = torch.ones((1, 1, 10, 10), dtype=torch.float32)
    anchor = torch.zeros_like(support, dtype=torch.bool)
    anchor[0, 0, 5, 5] = True
    support_filter = identity_module.CausalConnectedSupportFilter(
        max_anchor_ratio=2.0,
        min_area_fraction=0.01,
        max_area_fraction=0.20,
    )

    result = support_filter(support, anchor)

    assert torch.count_nonzero(result.keep_mask) == 2
    assert result.budget_fraction.item() == pytest.approx(0.02)


def test_connected_support_weak_anchor_does_not_shrink_previous_area():
    support = torch.zeros((1, 2, 6, 8), dtype=torch.float32)
    support[0, 0, 2, 1:7] = 0.9
    support[0, 1, 2, 2:8] = 0.9
    anchor = torch.zeros_like(support, dtype=torch.bool)
    anchor[0, 0, 2, 1:4] = True
    anchor[0, 1, 2, 2] = True
    support_filter = identity_module.CausalConnectedSupportFilter(
        temporal_radius=1,
        max_anchor_ratio=2.0,
        min_area_fraction=0.01,
        max_area_fraction=0.5,
    )

    result = support_filter(support, anchor)

    assert torch.count_nonzero(result.keep_mask[0, 0]) == 6
    assert torch.count_nonzero(result.keep_mask[0, 1]) == 6


def test_connected_support_survives_one_empty_observation():
    support_filter = identity_module.CausalConnectedSupportFilter(
        temporal_radius=1,
        min_area_fraction=0.01,
        max_area_fraction=0.5,
    )
    initial = torch.zeros((1, 1, 5, 6), dtype=torch.float32)
    initial[0, 0, 2, 1:3] = 0.9
    initial_anchor = torch.zeros_like(initial, dtype=torch.bool)
    initial_anchor[0, 0, 2, 1] = True
    support_filter(initial, initial_anchor)

    resumed = torch.zeros((1, 2, 5, 6), dtype=torch.float32)
    resumed[0, 1, 2, 2:4] = 0.8
    result = support_filter(
        resumed,
        torch.zeros_like(resumed, dtype=torch.bool),
    )

    assert torch.count_nonzero(result.keep_mask[0, 0]) == 0
    assert result.keep_mask[0, 1, 2, 2:4].all()


def test_connected_support_stays_inside_object_likelihood():
    support = torch.ones((1, 1, 5, 6), dtype=torch.float32)
    anchor = torch.zeros_like(support, dtype=torch.bool)
    anchor[0, 0, 2, 1] = True
    object_likelihood = torch.zeros_like(support, dtype=torch.bool)
    object_likelihood[0, 0, 2, 1:3] = True
    support_filter = identity_module.CausalConnectedSupportFilter(
        min_area_fraction=0.01,
        max_area_fraction=0.5,
        max_anchor_ratio=4.0,
    )

    result = support_filter(
        support,
        anchor,
        object_likelihood_mask=object_likelihood,
    )

    assert torch.equal(
        result.object_likelihood_mask,
        object_likelihood,
    )
    assert result.keep_mask[0, 0, 2, 1:3].all()
    assert not (result.keep_mask & ~object_likelihood).any()


def test_first_frame_bootstrap_uses_non_hand_object_core_only():
    base = torch.ones((1, 3, 5, 6), dtype=torch.float32)
    likelihood = torch.zeros_like(base)
    likelihood[0, 0, 2, 1:3] = torch.tensor([0.9, 0.8])
    likelihood[0, 0, 0, 3:6] = 1.0
    likelihood[0, 1, 0, 0] = 1.0
    threshold = torch.full((1, 3, 1, 1), 0.5)
    hand = torch.zeros_like(base)
    hand[0, 0, 2, 0] = 1.0

    bootstrap = (
        identity_module.build_first_frame_object_core_bootstrap(
            base_write_weight=base,
            object_likelihood=likelihood,
            object_threshold=threshold,
            hand_probability=hand,
        )
    )
    write = bootstrap.write_weight.reshape_as(base)

    assert bootstrap.core_mask[0, 0, 2, 1:3].all()
    assert not bootstrap.core_mask[0, 0, 0, 3:6].any()
    assert write[0, 0, 2, 1] == pytest.approx(0.9)
    assert torch.count_nonzero(write[:, 1:]) == 0


def test_causal_first_frame_bootstrap_factorizes_source_key_target_value():
    source_key = torch.tensor(
        [[[[1.0, 0.0]]] * 6],
        dtype=torch.float32,
    )
    source_key[:, 2:] = torch.tensor([-1.0, 0.0])
    target_key = torch.tensor(
        [[[[0.0, 1.0]]] * 6],
        dtype=torch.float32,
    )
    target_value = torch.full_like(target_key, 100.0)
    target_value[:, :2] = 5.0
    source_cache = _cache(
        source_key,
        torch.zeros_like(source_key),
        current_identity_key=source_key,
    )
    target_cache = _cache(target_key, target_value)
    memory = identity_module.SlowTargetIdentityMemory(
        layers=(0,),
        num_prototypes=1,
    )

    update = memory.bootstrap_causal_first_frame(
        kv_cache=target_cache,
        write_weight=torch.ones((1, 6)),
        num_frames=3,
        target_batch_start=0,
        source_kv_cache=source_cache,
    )
    anchor = memory.export()[0]

    assert torch.equal(
        update.write_weight,
        torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    assert torch.allclose(
        anchor.value,
        torch.full_like(anchor.value, 5.0),
    )
    assert torch.allclose(
        torch.nn.functional.normalize(anchor.key.float(), dim=-1),
        torch.tensor([[[[1.0, 0.0]]]]),
    )
    assert anchor.evidence.item() == memory.reference_prior_evidence
    assert memory.causal_first_frame_bootstrapped
    assert not memory.states


def test_causal_first_frame_anchor_ignores_adaptive_updates():
    source_key = torch.tensor(
        [[[[1.0, 0.0]]] * 4],
        dtype=torch.float32,
    )
    target_key = torch.tensor(
        [[[[0.0, 1.0]]] * 4],
        dtype=torch.float32,
    )
    target_value = torch.full_like(target_key, 3.0)
    source_cache = _cache(
        source_key,
        torch.zeros_like(source_key),
        current_identity_key=source_key,
    )
    memory = identity_module.SlowTargetIdentityMemory(
        layers=(0,),
        num_prototypes=1,
    )
    memory.bootstrap_causal_first_frame(
        kv_cache=_cache(target_key, target_value),
        write_weight=torch.ones((1, 4)),
        num_frames=2,
        target_batch_start=0,
        source_kv_cache=source_cache,
    )
    anchor_value = memory.export()[0].value.clone()

    memory.update(
        kv_cache=_cache(target_key, target_value + 50.0),
        write_weight=torch.ones((1, 4)),
        source_kv_cache=source_cache,
    )

    assert torch.equal(memory.export()[0].value, anchor_value)
    assert torch.count_nonzero(
        memory.export_adaptive()[0].value
    ) > 0


def test_object_wise_reset_commits_final_clean_object_only():
    provisional_source_key = torch.tensor(
        [[[[1.0, 0.0]]] * 4],
        dtype=torch.float32,
    )
    provisional_target_key = torch.tensor(
        [[[[0.0, 1.0]]] * 4],
        dtype=torch.float32,
    )
    memory = identity_module.SlowTargetIdentityMemory(
        layers=(0,),
        num_prototypes=1,
    )
    memory.bootstrap_causal_first_frame(
        kv_cache=_cache(
            provisional_target_key,
            torch.full_like(provisional_target_key, 3.0),
        ),
        write_weight=torch.ones((1, 4)),
        num_frames=2,
        target_batch_start=0,
        source_kv_cache=_cache(
            provisional_source_key,
            torch.zeros_like(provisional_source_key),
            current_identity_key=provisional_source_key,
        ),
    )

    final_source_key = torch.tensor(
        [[
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            [[-1.0, 0.0]],
            [[-1.0, 0.0]],
        ]],
        dtype=torch.float32,
    )
    final_target_value = torch.full_like(
        final_source_key,
        100.0,
    )
    final_target_value[:, :2] = 9.0
    update = memory.reset_causal_edit_anchor(
        kv_cache=_cache(
            torch.zeros_like(final_source_key),
            final_target_value,
        ),
        write_weight=torch.ones((1, 4)),
        num_frames=2,
        source_kv_cache=_cache(
            final_source_key,
            torch.zeros_like(final_source_key),
            current_identity_key=final_source_key,
        ),
    )
    anchor = memory.export()[0]

    assert torch.equal(
        update.write_weight,
        torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
    )
    assert torch.allclose(
        anchor.value,
        torch.full_like(anchor.value, 9.0),
    )
    assert torch.allclose(
        torch.nn.functional.normalize(anchor.key.float(), dim=-1),
        torch.tensor([[[[0.0, 1.0]]]]),
    )
    assert anchor.evidence.item() == memory.reference_prior_evidence
    assert memory.causal_edit_anchor_reset

    committed_value = anchor.value.clone()
    memory.update(
        kv_cache=_cache(
            torch.zeros_like(final_source_key),
            final_target_value + 50.0,
        ),
        write_weight=torch.ones((1, 4)),
        source_kv_cache=_cache(
            final_source_key,
            torch.zeros_like(final_source_key),
            current_identity_key=final_source_key,
        ),
    )
    assert torch.equal(memory.export()[0].value, committed_value)


def test_object_wise_reset_requires_support_and_runs_once():
    key = torch.tensor(
        [[[[1.0, 0.0]]] * 4],
        dtype=torch.float32,
    )
    memory = identity_module.SlowTargetIdentityMemory(
        layers=(0,),
        num_prototypes=1,
    )
    source_cache = _cache(
        key,
        torch.zeros_like(key),
        current_identity_key=key,
    )
    target_cache = _cache(key, torch.ones_like(key))
    memory.bootstrap_causal_first_frame(
        kv_cache=target_cache,
        write_weight=torch.ones((1, 4)),
        num_frames=2,
        target_batch_start=0,
        source_kv_cache=source_cache,
    )

    with pytest.raises(RuntimeError, match="no verified object"):
        memory.reset_causal_edit_anchor(
            kv_cache=target_cache,
            write_weight=torch.zeros((1, 4)),
            num_frames=2,
            source_kv_cache=source_cache,
        )

    memory.reset_causal_edit_anchor(
        kv_cache=target_cache,
        write_weight=torch.ones((1, 4)),
        num_frames=2,
        source_kv_cache=source_cache,
    )
    with pytest.raises(RuntimeError, match="already reset"):
        memory.reset_causal_edit_anchor(
            kv_cache=target_cache,
            write_weight=torch.ones((1, 4)),
            num_frames=2,
            source_kv_cache=source_cache,
        )


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


def test_identity_core_actively_releases_preserve_belief():
    belief = _belief(edit=0.1, preserve=1.0)
    hand = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    committed_edit = torch.full(
        (1, 1, 2, 2),
        0.1,
        dtype=torch.float32,
    )
    committed_precision = torch.full_like(committed_edit, 0.1)
    identity_core = torch.zeros_like(committed_edit)
    identity_core[0, 0, 0, 0] = 1.0

    updated, debug = identity_module.inject_committed_memory_into_belief(
        belief=belief,
        committed_token_edit=committed_edit,
        committed_token_precision=committed_precision,
        hand_mask=hand,
        feedback_strength=0.75,
        identity_core_support=identity_core,
    )

    assert debug["committed_memory_preserve_release"][0, 0, 0, 0] == (
        pytest.approx(0.75)
    )
    assert updated.preserve_belief[0, 0, 0, 0] == pytest.approx(0.25)
    assert updated.preserve_belief[0, 0, -1, -1] == pytest.approx(0.9925)
    assert updated.edit_belief[0, 0, 0, 0] > belief.edit_belief[
        0, 0, 0, 0
    ]


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
    anchor = memory.export()[0]
    valid_evidence = anchor.evidence > 0

    assert memory.reference_bootstrapped
    assert torch.all(
        anchor.evidence[valid_evidence]
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


def test_reference_identity_anchor_is_separate_from_online_update():
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
    reference_value = memory.export()[0].value.float().clone()

    update = memory.update(
        _cache(key, value + 20.0),
        write_weight=torch.full((1, 4), 0.05),
    )

    assert update.update_gain.max() == 1.0
    assert torch.equal(
        memory.export()[0].value.float(),
        reference_value,
    )
    assert not torch.equal(
        memory.export_adaptive()[0].value.float(),
        reference_value,
    )


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


def test_identity_value_correction_rejects_weak_source_match():
    correspondence_key = torch.tensor(
        [[[[0.0, 1.0]], [[0.0, -1.0]]]],
        dtype=torch.float32,
    )
    target_value = torch.zeros_like(correspondence_key)
    prototype_key = torch.tensor([[[[1.0, 0.0]]]])
    prototype_value = torch.ones_like(prototype_key) * 7.0

    corrected, support = (
        attention_module.apply_target_identity_value_correction(
            correspondence_key,
            target_value,
            prototype_key,
            prototype_value,
            prototype_evidence=torch.ones((1, 1)),
        )
    )

    assert torch.equal(corrected, target_value)
    assert torch.count_nonzero(support) == 0


def test_identity_value_correction_respects_role_support_mask():
    correspondence_key = torch.tensor(
        [[[[1.0, 0.0]], [[1.0, 0.0]]]],
        dtype=torch.float32,
    )
    target_value = torch.zeros_like(correspondence_key)
    prototype_key = correspondence_key[:, :1].clone()
    prototype_value = torch.ones_like(prototype_key) * 5.0
    support_mask = torch.tensor([[True, False]])

    corrected, support = (
        attention_module.apply_target_identity_value_correction(
            correspondence_key,
            target_value,
            prototype_key,
            prototype_value,
            prototype_evidence=torch.ones((1, 1)),
            support_mask=support_mask,
        )
    )

    assert support[0, 0] > 0
    assert support[0, 1] == 0
    assert torch.count_nonzero(corrected[0, 0]) > 0
    assert torch.equal(corrected[0, 1], target_value[0, 1])


def test_identity_value_correction_normalizes_support_per_frame():
    cosine = torch.tensor([
        1.00, 0.99, 0.98, 0.97, 0.96,
        0.95, 0.94, 0.93, 0.92, 0.91,
        0.80, 0.70, 0.60, 0.50, 0.40,
        0.30, 0.20, 0.10, 0.00, -0.10,
    ])
    sine = torch.sqrt((1.0 - cosine.square()).clamp_min(0.0))
    target_key = torch.stack((cosine, sine), dim=-1)[
        None, :, None, :
    ]
    target_value = torch.zeros_like(target_key)
    prototype_key = torch.tensor([[[[1.0, 0.0]]]])
    prototype_value = torch.ones_like(prototype_key)

    _, global_support = (
        attention_module.apply_target_identity_value_correction(
            target_key,
            target_value,
            prototype_key,
            prototype_value,
            prototype_evidence=torch.ones((1, 1)),
        )
    )
    _, frame_support = (
        attention_module.apply_target_identity_value_correction(
            target_key,
            target_value,
            prototype_key,
            prototype_value,
            prototype_evidence=torch.ones((1, 1)),
            tokens_per_frame=10,
        )
    )

    assert torch.count_nonzero(global_support[:, 10:]) == 0
    assert frame_support[:, 10:].max() > 0


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
