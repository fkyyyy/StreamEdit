from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch

from tests._pipeline_imports import REPO_ROOT, load_pipeline_module


def load_attention_module():
    path = REPO_ROOT / "wan" / "modules" / "attention.py"
    spec = importlib.util.spec_from_file_location(
        "streamedit_factorized_attention",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


attention_module = load_attention_module()
identity_module = load_pipeline_module("target_identity_memory")

apply_target_identity_value_correction = (
    attention_module.apply_target_identity_value_correction
)
SlowTargetIdentityMemory = identity_module.SlowTargetIdentityMemory
TargetIdentityLayerState = identity_module.TargetIdentityLayerState
TargetIdentityLifecycle = identity_module.TargetIdentityLifecycle
CausalIdentityOwnerTracker = identity_module.CausalIdentityOwnerTracker
SourceCoordinateResidualCarry = (
    identity_module.SourceCoordinateResidualCarry
)
apply_source_owner_residual_constraint = (
    identity_module.apply_source_owner_residual_constraint
)
apply_source_owner_geometry_envelope = (
    identity_module.apply_source_owner_geometry_envelope
)
build_oracle_source_owner_weight = (
    identity_module.build_oracle_source_owner_weight
)


class FactorizedIdentityCorrectionTests(unittest.TestCase):
    def test_soft_correction_changes_only_current_object_core(self):
        correspondence = torch.tensor(
            [[[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]]]
        )
        appearance = correspondence.clone()
        target_value = torch.zeros_like(correspondence)
        prototype_key = torch.tensor([[[[1.0, 0.0]]]])
        prototype_value = torch.tensor([[[[4.0, 2.0]]]])
        support = torch.tensor([[True, False, True]])

        corrected, confidence = apply_target_identity_value_correction(
            correspondence_key=correspondence,
            target_value=target_value,
            prototype_key=prototype_key,
            prototype_value=prototype_value,
            prototype_evidence=torch.ones(1, 1),
            target_appearance_key=appearance,
            prototype_appearance_key=prototype_key,
            tokens_per_frame=3,
            support_mask=support,
            min_similarity=0.5,
            correction_strength=0.25,
        )

        torch.testing.assert_close(
            corrected[:, 1],
            target_value[:, 1],
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue((corrected[:, (0, 2)] > 0).all())
        self.assertEqual(confidence[:, 1].item(), 0.0)

    def test_appearance_mismatch_blocks_high_confidence_read(self):
        correspondence = torch.tensor([[[[1.0, 0.0]]]])
        target_value = torch.zeros_like(correspondence)
        prototype_key = torch.tensor([[[[1.0, 0.0]]]])
        mismatch = torch.tensor([[[[-1.0, 0.0]]]])

        corrected, confidence = apply_target_identity_value_correction(
            correspondence_key=correspondence,
            target_value=target_value,
            prototype_key=prototype_key,
            prototype_value=torch.ones_like(prototype_key),
            prototype_evidence=torch.ones(1, 1),
            target_appearance_key=mismatch,
            prototype_appearance_key=prototype_key,
            support_mask=torch.ones(1, 1),
            min_similarity=0.5,
            correction_strength=1.0,
        )

        torch.testing.assert_close(corrected, target_value)
        torch.testing.assert_close(confidence, torch.zeros_like(confidence))

    def test_assignment_uses_same_prototype_for_both_factors(self):
        correspondence = torch.tensor([[[[1.0, 0.0]]]])
        appearance = torch.tensor([[[[1.0, 0.0]]]])
        prototype_key = torch.tensor(
            [[[[1.0, 0.0]], [[0.7, 0.7141428]]]]
        )
        prototype_appearance = torch.tensor(
            [[[[0.5, 0.8660254]], [[0.7, 0.7141428]]]]
        )
        prototype_value = torch.tensor(
            [[[[10.0, 0.0]], [[0.0, 10.0]]]]
        )

        corrected, confidence = apply_target_identity_value_correction(
            correspondence_key=correspondence,
            target_value=torch.zeros_like(correspondence),
            prototype_key=prototype_key,
            prototype_value=prototype_value,
            prototype_evidence=torch.ones(1, 2),
            target_appearance_key=appearance,
            prototype_appearance_key=prototype_appearance,
            support_mask=torch.ones(1, 1),
            min_similarity=0.4,
            correction_strength=1.0,
        )

        self.assertGreater(confidence.item(), 0.0)
        self.assertGreater(
            corrected[0, 0, 0, 1].item(),
            3.0 * corrected[0, 0, 0, 0].item(),
        )

    def test_residual_value_is_reconstructed_in_source_coordinates(self):
        correspondence = torch.tensor([[[[1.0, 0.0]]]])
        source_value = torch.tensor([[[[10.0, 10.0]]]])
        target_value = torch.zeros_like(source_value)
        target_residual = torch.tensor([[[[2.0, -2.0]]]])

        corrected, support, diagnostics = (
            apply_target_identity_value_correction(
                correspondence_key=correspondence,
                target_value=target_value,
                source_value=source_value,
                prototype_key=correspondence,
                prototype_value=target_residual,
                prototype_evidence=torch.ones(1, 1),
                prototype_value_is_residual=True,
                support_mask=torch.ones(1, 1),
                min_similarity=0.5,
                correction_strength=0.25,
                return_diagnostics=True,
            )
        )

        torch.testing.assert_close(
            corrected,
            torch.tensor([[[[3.0, 2.0]]]]),
        )
        torch.testing.assert_close(support, torch.ones_like(support))
        self.assertGreater(diagnostics["correction_ratio"].item(), 0.0)

    def test_residual_value_requires_source_value(self):
        value = torch.zeros(1, 1, 1, 2)
        with self.assertRaisesRegex(ValueError, "requires source_value"):
            apply_target_identity_value_correction(
                correspondence_key=torch.ones_like(value),
                target_value=value,
                prototype_key=torch.ones_like(value),
                prototype_value=torch.ones_like(value),
                prototype_evidence=torch.ones(1, 1),
                prototype_value_is_residual=True,
            )

    def test_support_floor_keeps_owner_coverage_when_match_is_weak(self):
        correspondence = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        prototype = torch.tensor([[[[1.0, 0.0]]]])
        target = torch.zeros_like(correspondence)

        corrected, support = apply_target_identity_value_correction(
            correspondence_key=correspondence,
            target_value=target,
            prototype_key=prototype,
            prototype_value=torch.ones_like(prototype),
            prototype_evidence=torch.ones(1, 1),
            support_mask=torch.ones(1, 2),
            support_floor=0.4,
            min_similarity=0.5,
            correction_strength=1.0,
        )

        self.assertGreaterEqual(support.min().item(), 0.4)
        self.assertTrue((corrected > 0).all())


class OracleSourceOwnerTests(unittest.TestCase):
    def test_projects_visible_source_owner_and_removes_hand(self):
        source_owner = torch.zeros(1, 2, 4, 4, dtype=torch.bool)
        source_owner[0, 0, :2, :2] = True
        source_owner[0, 1, 2:, 2:] = True
        hand = torch.zeros_like(source_owner)
        hand[0, 0, 0, 0] = True

        owner = build_oracle_source_owner_weight(
            source_owner,
            hand,
            spatial_shape=(2, 2),
        )

        self.assertEqual(tuple(owner.shape), (1, 2, 2, 2))
        self.assertEqual(owner[0, 0, 0, 0].item(), 0.75)
        self.assertEqual(owner[0, 1, 1, 1].item(), 1.0)
        self.assertEqual(owner[0, 1, 0, 0].item(), 0.0)

    def test_empty_source_frame_stays_empty_without_ghosting(self):
        source_owner = torch.zeros(1, 3, 4, 4, dtype=torch.bool)
        source_owner[0, 0, 1:3, 1:3] = True
        source_owner[0, 2, 1:3, 1:3] = True

        owner = build_oracle_source_owner_weight(
            source_owner,
            torch.zeros_like(source_owner),
            spatial_shape=(2, 2),
        )

        self.assertGreater(owner[:, 0].sum().item(), 0.0)
        self.assertEqual(owner[:, 1].sum().item(), 0.0)
        self.assertGreater(owner[:, 2].sum().item(), 0.0)

    def test_does_not_reapply_temporally_pooled_hand_exclusion(self):
        source_owner = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
        source_owner[0, 0, 0, :] = True
        temporally_pooled_hand = source_owner.clone()

        owner = build_oracle_source_owner_weight(
            source_owner,
            temporally_pooled_hand,
            spatial_shape=(2, 2),
            hand_already_excluded=True,
        )

        torch.testing.assert_close(owner, source_owner.float())

    def test_rejects_misaligned_hand_mask(self):
        with self.assertRaisesRegex(ValueError, "must share"):
            build_oracle_source_owner_weight(
                torch.zeros(1, 2, 4, 4),
                torch.zeros(1, 1, 4, 4),
                spatial_shape=(2, 2),
            )


class IdentityLifecycleTests(unittest.TestCase):
    def test_memory_separates_source_and_target_keys(self):
        memory = SlowTargetIdentityMemory(
            layers=(0,),
            num_prototypes=1,
        )
        source_key = torch.tensor([[[[1.0, 0.0]]]])
        target_key = torch.tensor([[[[0.0, 1.0]]]])
        target_value = torch.tensor([[[[2.0, 3.0]]]])
        target_cache = [{
            "k": target_key.clone(),
            "v": target_value.clone(),
            "current_identity_key": target_key.clone(),
            "local_end_index": torch.tensor([1]),
            "num_new_tokens": 1,
        }]
        source_cache = [{
            "k": source_key.clone(),
            "v": torch.zeros_like(source_key),
            "current_identity_key": source_key.clone(),
            "local_end_index": torch.tensor([1]),
            "num_new_tokens": 1,
        }]

        memory.update(
            kv_cache=target_cache,
            write_weight=torch.ones(1, 1),
            source_kv_cache=source_cache,
        )

        state = memory.export_adaptive()[0]
        source_direction = torch.nn.functional.normalize(
            state.key.float(), dim=-1
        )
        appearance_direction = torch.nn.functional.normalize(
            state.appearance_key.float(), dim=-1
        )
        torch.testing.assert_close(
            source_direction,
            source_key,
        )
        torch.testing.assert_close(
            appearance_direction,
            target_key,
        )

    def test_memory_stores_target_minus_source_value(self):
        memory = SlowTargetIdentityMemory(
            layers=(0,),
            num_prototypes=1,
            store_value_residual=True,
        )
        source_key = torch.tensor([[[[1.0, 0.0]]]])
        source_value = torch.tensor([[[[1.0, 2.0]]]])
        target_value = torch.tensor([[[[4.0, 6.0]]]])
        target_cache = [{
            "k": source_key.clone(),
            "v": target_value.clone(),
            "current_identity_key": source_key.clone(),
            "local_end_index": torch.tensor([1]),
            "num_new_tokens": 1,
        }]
        source_cache = [{
            "k": source_key.clone(),
            "v": source_value.clone(),
            "current_identity_key": source_key.clone(),
            "local_end_index": torch.tensor([1]),
            "num_new_tokens": 1,
        }]

        memory.update(
            kv_cache=target_cache,
            write_weight=torch.ones(1, 1),
            source_kv_cache=source_cache,
        )

        state = memory.export_adaptive()[0]
        torch.testing.assert_close(
            state.value, target_value - source_value
        )
        self.assertTrue(state.value_is_residual)

    def test_absent_object_never_reads_stale_identity(self):
        memory = SlowTargetIdentityMemory(layers=(0,))
        core = torch.tensor([[True, False, False, True]])

        visible = memory.update_visibility_lifecycle(
            core,
            torch.tensor([[[[True]], [[True]]]]),
            tokens_per_frame=2,
            max_occluded_blocks=1,
        )
        occluded = memory.update_visibility_lifecycle(
            torch.zeros_like(core),
            torch.tensor([[[[False]], [[False]]]]),
            tokens_per_frame=2,
            max_occluded_blocks=1,
        )
        absent = memory.update_visibility_lifecycle(
            torch.zeros_like(core),
            torch.tensor([[[[False]], [[False]]]]),
            tokens_per_frame=2,
            max_occluded_blocks=1,
        )

        self.assertEqual(
            visible.state_code.item(),
            TargetIdentityLifecycle.VISIBLE,
        )
        self.assertEqual(
            occluded.state_code.item(),
            TargetIdentityLifecycle.OCCLUDED,
        )
        self.assertEqual(
            absent.state_code.item(),
            TargetIdentityLifecycle.ABSENT,
        )
        self.assertFalse(occluded.read_mask.any())
        self.assertFalse(absent.read_mask.any())

    def test_replay_promotion_preserves_target_appearance_key(self):
        memory = SlowTargetIdentityMemory(layers=(0,))
        state = TargetIdentityLayerState(
            key=torch.ones(1, 1, 1, 2),
            value=torch.full((1, 1, 1, 2), 2.0),
            evidence=torch.full((1, 1), 0.5),
            appearance_key=torch.full((1, 1, 1, 2), 3.0),
        )
        memory.states[0] = state

        memory.promote_adaptive_to_replay_anchor()

        promoted = memory.export()[0]
        torch.testing.assert_close(
            promoted.appearance_key,
            state.appearance_key,
        )
        torch.testing.assert_close(
            promoted.evidence,
            torch.full((1, 1), 8.0),
        )
        self.assertFalse(memory.states)

    def test_replay_anchor_remains_authoritative_after_later_update(self):
        memory = SlowTargetIdentityMemory(
            layers=(0,),
            num_prototypes=1,
            store_value_residual=True,
        )
        initial_source_key = torch.tensor([[[[1.0, 0.0]]]])
        initial_source_value = torch.tensor([[[[1.0, 2.0]]]])
        initial_target_value = torch.tensor([[[[4.0, 6.0]]]])

        def cache(key, value):
            return [{
                "k": key.clone(),
                "v": value.clone(),
                "current_identity_key": key.clone(),
                "local_end_index": torch.tensor([1]),
                "num_new_tokens": 1,
            }]

        memory.update(
            kv_cache=cache(initial_source_key, initial_target_value),
            write_weight=torch.ones(1, 1),
            source_kv_cache=cache(
                initial_source_key,
                initial_source_value,
            ),
        )
        memory.promote_adaptive_to_replay_anchor()
        frozen = memory.export()[0]
        frozen_value = frozen.value.clone()
        frozen_key = frozen.key.clone()
        frozen_evidence = frozen.evidence.clone()

        later_key = torch.tensor([[[[0.0, 1.0]]]])
        memory.update(
            kv_cache=cache(
                later_key,
                torch.full_like(later_key, 100.0),
            ),
            write_weight=torch.ones(1, 1),
            source_kv_cache=cache(
                later_key,
                torch.full_like(later_key, -100.0),
            ),
        )

        exported = memory.export()[0]
        torch.testing.assert_close(exported.key, frozen_key)
        torch.testing.assert_close(exported.value, frozen_value)
        torch.testing.assert_close(exported.evidence, frozen_evidence)
        self.assertIn(0, memory.export_adaptive())
        self.assertIsNot(memory.export_adaptive()[0], exported)

    def test_replay_promotion_rejects_zero_evidence_state(self):
        memory = SlowTargetIdentityMemory(layers=(0,))
        memory.states[0] = TargetIdentityLayerState(
            key=torch.zeros(1, 1, 1, 2),
            value=torch.zeros(1, 1, 1, 2),
            evidence=torch.zeros(1, 1),
            appearance_key=torch.zeros(1, 1, 1, 2),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "no verified target identity support",
        ):
            memory.promote_adaptive_to_replay_anchor()


class CausalIdentityOwnerTrackerTests(unittest.TestCase):
    @staticmethod
    def _features(object_index):
        features = torch.tensor(
            [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]
        )
        features[0, object_index] = torch.tensor([1.0, 0.0])
        return features

    @staticmethod
    def _semantic(object_index):
        semantic = torch.zeros(1, 4)
        semantic[0, object_index] = 1.0
        return semantic

    def test_transport_survives_missing_current_observation(self):
        tracker = CausalIdentityOwnerTracker(max_area_fraction=0.5)
        tracker.commit_verified(
            source_features=self._features(0),
            verified_weight=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            tokens_per_frame=4,
        )

        result = tracker(
            source_features=self._features(1),
            observation_weight=torch.zeros(1, 4),
            source_semantic=self._semantic(1),
            hand_mask=torch.zeros(1, 4, dtype=torch.bool),
            tokens_per_frame=4,
            frame_visible=torch.ones(1, 1, 1, 1),
        )

        self.assertGreater(result.transported_weight[0, 1].item(), 0.9)
        self.assertGreater(result.read_weight[0, 1].item(), 0.9)
        self.assertEqual(result.read_weight[0, 0].item(), 0.0)

    def test_absence_disables_read_without_erasing_last_visible_state(self):
        tracker = CausalIdentityOwnerTracker(max_area_fraction=0.5)
        tracker.commit_verified(
            source_features=self._features(0),
            verified_weight=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            tokens_per_frame=4,
        )
        absent = tracker(
            source_features=self._features(1),
            observation_weight=torch.zeros(1, 4),
            source_semantic=self._semantic(1),
            hand_mask=torch.zeros(1, 4, dtype=torch.bool),
            tokens_per_frame=4,
            frame_visible=torch.zeros(1, 1),
        )
        visible_again = tracker(
            source_features=self._features(2),
            observation_weight=torch.zeros(1, 4),
            source_semantic=self._semantic(2),
            hand_mask=torch.zeros(1, 4, dtype=torch.bool),
            tokens_per_frame=4,
            frame_visible=torch.ones(1, 1),
        )

        self.assertFalse(absent.read_weight.bool().any())
        self.assertGreater(visible_again.read_weight[0, 2].item(), 0.9)

    def test_source_match_can_recover_a_detector_visibility_dropout(self):
        tracker = CausalIdentityOwnerTracker(
            max_area_fraction=0.5,
            recover_visibility_from_source_match=True,
        )
        tracker.commit_verified(
            source_features=self._features(0),
            verified_weight=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            tokens_per_frame=4,
        )

        recovered = tracker(
            source_features=self._features(1),
            observation_weight=torch.zeros(1, 4),
            source_semantic=self._semantic(1),
            hand_mask=torch.zeros(1, 4, dtype=torch.bool),
            tokens_per_frame=4,
            frame_visible=torch.zeros(1, 1),
        )

        self.assertGreater(recovered.read_weight[0, 1].item(), 0.9)

    def test_visibility_recovery_does_not_ghost_without_source_match(self):
        tracker = CausalIdentityOwnerTracker(
            max_area_fraction=0.5,
            recover_visibility_from_source_match=True,
        )
        tracker.commit_verified(
            source_features=self._features(0),
            verified_weight=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            tokens_per_frame=4,
        )
        no_object_features = torch.tensor(
            [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]
        )

        absent = tracker(
            source_features=no_object_features,
            observation_weight=torch.zeros(1, 4),
            source_semantic=torch.zeros(1, 4),
            hand_mask=torch.zeros(1, 4, dtype=torch.bool),
            tokens_per_frame=4,
            frame_visible=torch.zeros(1, 1),
        )

        self.assertFalse(absent.read_weight.bool().any())

    def test_prediction_advances_persistent_owner_in_source_coordinates(self):
        tracker = CausalIdentityOwnerTracker(max_area_fraction=0.5)
        tracker.commit_verified(
            source_features=self._features(0),
            verified_weight=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            tokens_per_frame=4,
        )
        committed_features = tracker.previous_features.clone()
        committed_weight = tracker.previous_weight.clone()

        tracker(
            source_features=self._features(1),
            observation_weight=torch.zeros(1, 4),
            source_semantic=self._semantic(1),
            hand_mask=torch.zeros(1, 4, dtype=torch.bool),
            tokens_per_frame=4,
            frame_visible=torch.ones(1, 1),
        )

        self.assertFalse(torch.equal(
            tracker.previous_features, committed_features
        ))
        self.assertFalse(torch.equal(
            tracker.previous_weight, committed_weight
        ))
        self.assertGreater(tracker.previous_weight[0, 1].item(), 0.9)

    def test_broad_observation_cannot_open_verified_read_gate(self):
        tracker = CausalIdentityOwnerTracker(max_area_fraction=0.75)
        tracker.commit_verified(
            source_features=self._features(0),
            verified_weight=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            tokens_per_frame=4,
        )

        result = tracker(
            source_features=self._features(1),
            observation_weight=torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
            source_semantic=torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
            hand_mask=torch.zeros(1, 4, dtype=torch.bool),
            tokens_per_frame=4,
            frame_visible=torch.ones(1, 1),
        )

        self.assertGreater(result.read_weight[0, 1].item(), 0.9)
        self.assertEqual(result.read_weight[0, 2].item(), 0.0)
        self.assertGreater(result.observation_weight[0, 2].item(), 0.0)

    def test_verified_write_replaces_broad_predicted_owner(self):
        tracker = CausalIdentityOwnerTracker(max_area_fraction=0.75)
        tracker(
            source_features=self._features(0),
            observation_weight=torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            source_semantic=torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            hand_mask=torch.zeros(1, 4, dtype=torch.bool),
            tokens_per_frame=4,
            frame_visible=torch.ones(1, 1),
        )
        tracker.commit_verified(
            source_features=self._features(0),
            verified_weight=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            tokens_per_frame=4,
        )

        self.assertEqual(tracker.previous_weight[0, 0].item(), 1.0)
        self.assertEqual(tracker.previous_weight[0, 1].item(), 0.0)

    def test_area_growth_is_bounded_during_ambiguous_transport(self):
        tracker = CausalIdentityOwnerTracker(
            max_area_fraction=1.0,
            max_area_growth=1.5,
        )
        features = torch.ones(1, 8, 2)
        tracker.commit_verified(
            source_features=features,
            verified_weight=torch.tensor(
                [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            ),
            tokens_per_frame=8,
        )

        result = tracker(
            source_features=features,
            observation_weight=torch.ones(1, 8),
            source_semantic=torch.ones(1, 8),
            hand_mask=torch.zeros(1, 8, dtype=torch.bool),
            tokens_per_frame=8,
            frame_visible=torch.ones(1, 1),
        )

        self.assertLessEqual(
            torch.count_nonzero(result.read_weight).item(), 3
        )


class SourceCoordinateResidualCarryTests(unittest.TestCase):
    @staticmethod
    def _features(object_index):
        features = torch.tensor(
            [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]
        )
        features[0, object_index] = torch.tensor([1.0, 0.0])
        return features

    def test_carry_moves_only_target_minus_source_object_patch(self):
        carry = SourceCoordinateResidualCarry(
            local_radius=2, min_similarity=0.5, patch_size=2
        )
        source = torch.zeros(1, 1, 1, 4, 4)
        target = source.clone()
        target[:, :, :, :2, :2] = 2.0
        carry.commit(
            source_features=self._features(0),
            owner_weight=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            source_latent=source,
            target_latent=target,
            tokens_per_frame=4,
            spatial_shape=(2, 2),
        )

        result = carry.prepare(
            source_features=self._features(1),
            owner_weight=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            source_latent=source,
            tokens_per_frame=4,
            spatial_shape=(2, 2),
        )

        self.assertGreater(result.support[0, 1].item(), 0.9)
        torch.testing.assert_close(
            result.residual[:, :, :, :2, 2:],
            torch.full((1, 1, 1, 2, 2), 2.0),
        )
        self.assertEqual(
            torch.count_nonzero(result.residual[:, :, :, 2:]).item(), 0
        )

    def test_empty_owner_does_not_create_a_ghost(self):
        carry = SourceCoordinateResidualCarry(patch_size=2)
        source = torch.zeros(1, 1, 1, 4, 4)
        carry.commit(
            source_features=self._features(0),
            owner_weight=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            source_latent=source,
            target_latent=source + 1.0,
            tokens_per_frame=4,
            spatial_shape=(2, 2),
        )

        result = carry.prepare(
            source_features=self._features(1),
            owner_weight=torch.zeros(1, 4),
            source_latent=source,
            tokens_per_frame=4,
            spatial_shape=(2, 2),
        )

        self.assertFalse(result.support.bool().any())
        self.assertFalse(result.residual.bool().any())

    def test_frozen_residual_advances_through_source_correspondence(self):
        carry = SourceCoordinateResidualCarry(
            local_radius=2, min_similarity=0.5, patch_size=2
        )
        source = torch.zeros(1, 1, 1, 4, 4)
        target = source.clone()
        target[:, :, :, :2, :2] = 3.0
        carry.commit(
            source_features=self._features(0),
            owner_weight=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            source_latent=source,
            target_latent=target,
            tokens_per_frame=4,
            spatial_shape=(2, 2),
        )
        carry.prepare(
            source_features=self._features(1),
            owner_weight=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            source_latent=source,
            tokens_per_frame=4,
            spatial_shape=(2, 2),
        )

        second_hop = carry.prepare(
            source_features=self._features(2),
            owner_weight=torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
            source_latent=source,
            tokens_per_frame=4,
            spatial_shape=(2, 2),
        )

        self.assertGreater(second_hop.support[0, 2].item(), 0.9)
        torch.testing.assert_close(
            second_hop.residual[:, :, :, 2:, :2],
            torch.full((1, 1, 1, 2, 2), 3.0),
        )


class SourceOwnerResidualConstraintTests(unittest.TestCase):
    def test_constraint_moves_toward_fixed_residual_without_adding_it(self):
        source = torch.zeros(1, 1, 1, 4, 4)
        residual = torch.full_like(source, 2.0)
        support = torch.ones(1, 4)

        first = apply_source_owner_residual_constraint(
            current_latent=source,
            source_latent=source,
            carried_residual=residual,
            support=support,
            spatial_shape=(2, 2),
            strength=0.5,
            denoising_progress=1.0,
        )
        second = apply_source_owner_residual_constraint(
            current_latent=first.latent,
            source_latent=source,
            carried_residual=residual,
            support=support,
            spatial_shape=(2, 2),
            strength=0.5,
            denoising_progress=1.0,
        )

        torch.testing.assert_close(first.latent, torch.ones_like(source))
        torch.testing.assert_close(
            second.latent, torch.full_like(source, 1.5)
        )
        self.assertTrue((second.latent <= source + residual).all())

    def test_constraint_is_late_rising_and_owner_scoped(self):
        source = torch.zeros(1, 1, 1, 4, 4)
        residual = torch.ones_like(source)
        result = apply_source_owner_residual_constraint(
            current_latent=source,
            source_latent=source,
            carried_residual=residual,
            support=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            spatial_shape=(2, 2),
            strength=0.8,
            denoising_progress=0.5,
            schedule_power=2.0,
        )

        torch.testing.assert_close(
            result.latent[:, :, :, :2, :2],
            torch.full((1, 1, 1, 2, 2), 0.2),
        )
        self.assertEqual(
            torch.count_nonzero(result.latent[:, :, :, 2:]).item(), 0
        )

    def test_constraint_protects_hand_pixels_and_absent_frames(self):
        source = torch.zeros(1, 2, 1, 4, 4)
        protect = torch.zeros(1, 2, 4, 4, dtype=torch.bool)
        protect[0, 0, :2, :2] = True
        result = apply_source_owner_residual_constraint(
            current_latent=source,
            source_latent=source,
            carried_residual=torch.ones_like(source),
            support=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            ),
            spatial_shape=(2, 2),
            strength=1.0,
            denoising_progress=1.0,
            protect_mask=protect,
        )

        self.assertFalse(result.latent[0, 0, :, :2, :2].bool().any())
        self.assertFalse(result.latent[0, 1].bool().any())


class SourceOwnerGeometryEnvelopeTests(unittest.TestCase):
    def test_envelope_preserves_source_outside_owner_only(self):
        source = torch.zeros(1, 1, 1, 4, 4)
        current = torch.ones_like(source)
        owner = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        owner[:, :, :2, :2] = True

        result = apply_source_owner_geometry_envelope(
            current_latent=current,
            source_latent=source,
            source_owner_mask=owner,
            strength=1.0,
            denoising_progress=1.0,
            margin=0,
        )

        torch.testing.assert_close(
            result.latent[:, :, :, :2, :2],
            torch.ones(1, 1, 1, 2, 2),
        )
        self.assertFalse(result.latent[:, :, :, 2:].bool().any())
        self.assertFalse(result.latent[:, :, :, :2, 2:].bool().any())

    def test_envelope_is_late_rising_and_non_accumulating(self):
        source = torch.zeros(1, 1, 1, 2, 2)
        current = torch.ones_like(source)
        owner = torch.zeros(1, 1, 2, 2, dtype=torch.bool)

        first = apply_source_owner_geometry_envelope(
            current_latent=current,
            source_latent=source,
            source_owner_mask=owner,
            strength=0.8,
            denoising_progress=0.5,
            schedule_power=2.0,
            margin=0,
        )
        second = apply_source_owner_geometry_envelope(
            current_latent=first.latent,
            source_latent=source,
            source_owner_mask=owner,
            strength=0.8,
            denoising_progress=0.5,
            schedule_power=2.0,
            margin=0,
        )

        torch.testing.assert_close(
            first.latent, torch.full_like(source, 0.8)
        )
        torch.testing.assert_close(
            second.latent, torch.full_like(source, 0.64)
        )
        self.assertTrue((second.latent >= source).all())

    def test_envelope_preserves_hand_inside_owner(self):
        source = torch.zeros(1, 1, 1, 4, 4)
        current = torch.ones_like(source)
        owner = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        hand = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        hand[:, :, 1, 2] = True

        result = apply_source_owner_geometry_envelope(
            current_latent=current,
            source_latent=source,
            source_owner_mask=owner,
            strength=1.0,
            denoising_progress=1.0,
            margin=0,
            protect_mask=hand,
        )

        self.assertEqual(result.latent[0, 0, 0, 1, 2].item(), 0.0)
        self.assertEqual(result.latent[0, 0, 0, 0, 0].item(), 1.0)

    def test_absent_owner_restores_whole_frame_to_source(self):
        source = torch.zeros(1, 1, 1, 2, 2)
        current = torch.ones_like(source)
        result = apply_source_owner_geometry_envelope(
            current_latent=current,
            source_latent=source,
            source_owner_mask=torch.zeros(
                1, 1, 2, 2, dtype=torch.bool
            ),
            strength=1.0,
            denoising_progress=1.0,
            margin=0,
        )

        torch.testing.assert_close(result.latent, source)


if __name__ == "__main__":
    unittest.main()
