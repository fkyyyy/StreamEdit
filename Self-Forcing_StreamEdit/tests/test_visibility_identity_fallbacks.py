from __future__ import annotations

import unittest

import torch

from tests._pipeline_imports import load_pipeline_module


control_belief_module = load_pipeline_module("control_belief")
target_identity_module = load_pipeline_module("target_identity_memory")
role_router_module = load_pipeline_module("role_router")
belief_kv_module = load_pipeline_module("belief_kv")
edit_commitment_module = load_pipeline_module("edit_commitment")

CausalControlBelief = control_belief_module.CausalControlBelief
CausalControlBeliefBuilder = control_belief_module.CausalControlBeliefBuilder
CausalObjectTokenPropagator = (
    target_identity_module.CausalObjectTokenPropagator
)
BayesResidualFlowRouter = role_router_module.BayesResidualFlowRouter
EditCommitmentController = (
    edit_commitment_module.EditCommitmentController
)
EditCommitmentResult = edit_commitment_module.EditCommitmentResult
apply_edit_commitment_policy = (
    edit_commitment_module.apply_edit_commitment_policy
)
build_belief_kv_weights = belief_kv_module.build_belief_kv_weights
build_cached_preserve_action = (
    belief_kv_module.build_cached_preserve_action
)
build_persistent_identity_read_mask = (
    target_identity_module.build_persistent_identity_read_mask
)
restrict_identity_support_to_observation = (
    target_identity_module.restrict_identity_support_to_observation
)


def make_belief(
    *,
    edit: float,
    preserve: float,
    edit_precision: float,
    preserve_precision: float,
    shape=(1, 1, 2, 2),
):
    def full(value: float):
        return torch.full(shape, value, dtype=torch.float32)

    return CausalControlBelief(
        edit_belief=full(edit),
        preserve_belief=full(preserve),
        edit_precision=full(edit_precision),
        preserve_precision=full(preserve_precision),
        visibility=full(1.0),
        uncertainty=full(0.0),
        conflict=full(0.0),
    )


class CausalObjectTokenPropagatorTests(unittest.TestCase):
    def test_default_empty_support_keeps_legacy_reset_behavior(self):
        propagator = CausalObjectTokenPropagator()
        features = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]],
            dtype=torch.float32,
        )
        propagator(
            source_features=features,
            base_write_weight=torch.tensor([[1.0, 0.0]]),
            support_weight=torch.tensor([[1.0, 0.0]]),
        )
        propagator(
            source_features=features,
            base_write_weight=torch.zeros(1, 2),
            support_weight=torch.zeros(1, 2),
        )

        self.assertEqual(
            torch.count_nonzero(propagator.previous_weight).item(),
            0,
        )

    def test_empty_support_does_not_erase_history_and_zero_base_does_not_write(self):
        propagator = CausalObjectTokenPropagator(
            min_similarity=0.55,
            gate_strength=0.85,
            max_candidates=8,
            retain_on_empty_support=True,
        )
        identity_features = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]],
            dtype=torch.float32,
        )

        initial = propagator(
            source_features=identity_features,
            base_write_weight=torch.tensor([[1.0, 0.0]]),
            support_weight=torch.tensor([[1.0, 0.0]]),
        )
        self.assertFalse(initial.has_previous.item())

        invisible = propagator(
            source_features=torch.flip(identity_features, dims=(1,)),
            base_write_weight=torch.zeros(1, 2),
            support_weight=torch.zeros(1, 2),
        )
        torch.testing.assert_close(
            invisible.write_weight,
            torch.zeros_like(invisible.write_weight),
        )

        reappeared = propagator(
            source_features=identity_features,
            base_write_weight=torch.tensor([[1.0, 0.0]]),
            support_weight=torch.tensor([[1.0, 0.0]]),
        )
        self.assertTrue(reappeared.has_previous.item())
        self.assertGreater(reappeared.matched_previous_weight[0, 0].item(), 0.0)
        self.assertGreater(reappeared.match_confidence[0, 0].item(), 0.0)
        self.assertGreater(reappeared.write_weight[0, 0].item(), 0.15)


class CausalControlBeliefVisibilityTests(unittest.TestCase):
    def test_default_invisibility_preserves_legacy_source_policy(self):
        builder = CausalControlBeliefBuilder()
        token_shape = (1, 1, 1, 1)
        belief = builder(
            debug={
                "object_posterior": torch.ones(token_shape),
                "source_attention": torch.ones(token_shape),
                "temporal_confidence": torch.zeros(token_shape),
                "object_visible": torch.zeros(token_shape),
                "hand_proximity": torch.ones(token_shape),
                "adaptive_attention_reliability": torch.ones(
                    token_shape
                ),
            },
            hand_mask=torch.zeros(1, 1, 2, 2),
        )

        torch.testing.assert_close(
            belief.preserve_belief,
            torch.ones_like(belief.preserve_belief),
        )

    def test_invisibility_alone_does_not_force_preserve(self):
        builder = CausalControlBeliefBuilder(
            preserve_on_observation_dropout=False,
        )
        token_shape = (1, 1, 1, 1)
        debug = {
            "object_posterior": torch.ones(token_shape),
            "source_attention": torch.ones(token_shape),
            "temporal_confidence": torch.zeros(token_shape),
            "observation_visible": torch.zeros(token_shape),
            "hand_proximity": torch.ones(token_shape),
            "adaptive_attention_reliability": torch.ones(token_shape),
        }
        hand_mask = torch.zeros(1, 1, 2, 2)

        belief = builder(debug=debug, hand_mask=hand_mask)

        torch.testing.assert_close(
            belief.visibility,
            torch.zeros_like(belief.visibility),
        )
        torch.testing.assert_close(
            belief.edit_belief,
            torch.zeros_like(belief.edit_belief),
        )
        torch.testing.assert_close(
            belief.preserve_belief,
            torch.zeros_like(belief.preserve_belief),
        )
        torch.testing.assert_close(
            belief.uncertainty,
            torch.ones_like(belief.uncertainty),
        )

    def test_target_conditioned_field_can_be_diagnostic_only(self):
        builder = CausalControlBeliefBuilder(
            preserve_on_observation_dropout=False,
            use_target_field_evidence=False,
        )
        token_shape = (1, 1, 1, 1)
        common = {
            "object_posterior": torch.full(token_shape, 0.8),
            "source_attention": torch.full(token_shape, 0.7),
            "temporal_confidence": torch.full(token_shape, 0.6),
            "observation_visible": torch.ones(token_shape),
            "hand_proximity": torch.zeros(token_shape),
            "adaptive_attention_reliability": torch.ones(token_shape),
            "adaptive_field_reliability": torch.ones(token_shape),
        }
        low_field = builder(
            debug={**common, "field_score": torch.zeros(token_shape)},
            hand_mask=torch.zeros(1, 1, 2, 2),
        )
        high_field = builder(
            debug={**common, "field_score": torch.ones(token_shape)},
            hand_mask=torch.zeros(1, 1, 2, 2),
        )

        for name in low_field.as_dict():
            torch.testing.assert_close(
                low_field.as_dict()[name],
                high_field.as_dict()[name],
            )

        field_aware_builder = CausalControlBeliefBuilder(
            preserve_on_observation_dropout=False,
        )
        field_aware_low = field_aware_builder(
            debug={**common, "field_score": torch.zeros(token_shape)},
            hand_mask=torch.zeros(1, 1, 2, 2),
        )
        field_aware_high = field_aware_builder(
            debug={**common, "field_score": torch.ones(token_shape)},
            hand_mask=torch.zeros(1, 1, 2, 2),
        )
        self.assertFalse(
            torch.equal(
                field_aware_low.edit_precision,
                field_aware_high.edit_precision,
            )
        )

    def test_tracked_dropout_is_target_only_without_erasing_background(self):
        builder = CausalControlBeliefBuilder(
            preserve_on_observation_dropout=False,
            use_target_field_evidence=False,
        )
        token_shape = (1, 1, 1, 2)
        belief = builder(
            debug={
                "object_posterior": torch.zeros(token_shape),
                "source_attention": torch.zeros(token_shape),
                "temporal_confidence": torch.zeros(token_shape),
                "observation_visible": torch.zeros(1, 1, 1, 1),
                "hand_proximity": torch.zeros(token_shape),
                "adaptive_attention_reliability": torch.ones(
                    1, 1, 1, 1
                ),
                "identity_track_tokens_preforward": torch.tensor(
                    [[[[1.0, 0.0]]]]
                ),
            },
            # A hand pixel in the neighboring patch dilates into the tracked
            # patch.  The tracked corridor must still remain target-only.
            hand_mask=torch.tensor(
                [[[[0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0]]]]
            ),
        )

        # The first 2x2 patch is the tracked dropout corridor; it has no
        # preserve evidence and therefore takes the target-only fallback.
        torch.testing.assert_close(
            belief.preserve_belief[..., :, :2],
            torch.zeros_like(belief.preserve_belief[..., :, :2]),
        )
        torch.testing.assert_close(
            belief.uncertainty[..., :, :2],
            torch.ones_like(belief.uncertainty[..., :, :2]),
        )
        # Untracked background still uses source preservation.
        torch.testing.assert_close(
            belief.preserve_belief[..., :, 2:],
            torch.ones_like(belief.preserve_belief[..., :, 2:]),
        )


class NoEvidenceTargetFallbackTests(unittest.TestCase):
    def test_bayes_residual_router_uses_target_only_without_evidence(self):
        router = BayesResidualFlowRouter(preserve_on_no_evidence=False)
        target = torch.full((1, 1, 1, 2, 2), 3.0)
        source = torch.full_like(target, 10.0)
        source_reconstruction = torch.full_like(target, 14.0)
        belief = make_belief(
            edit=0.0,
            preserve=0.0,
            edit_precision=0.0,
            preserve_precision=0.0,
        )

        routed, diagnostics = router(
            target_velocity=target,
            source_velocity=source,
            source_reconstruction_velocity=source_reconstruction,
            belief=belief,
        )

        torch.testing.assert_close(routed, target)
        torch.testing.assert_close(
            diagnostics["no_evidence"],
            torch.ones_like(diagnostics["no_evidence"]),
        )
        torch.testing.assert_close(
            diagnostics["edit_action_weight"],
            torch.ones_like(diagnostics["edit_action_weight"]),
        )
        torch.testing.assert_close(
            diagnostics["preserve_action_weight"],
            torch.zeros_like(diagnostics["preserve_action_weight"]),
        )

    def test_belief_kv_uses_target_only_without_evidence(self):
        belief = make_belief(
            edit=0.0,
            preserve=0.0,
            edit_precision=0.0,
            preserve_precision=0.0,
        )

        weights = build_belief_kv_weights(
            belief=belief,
            expected_token_length=1,
            preserve_on_no_evidence=False,
        )

        for value in (weights.edit, weights.edit_map):
            torch.testing.assert_close(value, torch.zeros_like(value))
        for value in (weights.edit_action, weights.edit_action_map):
            torch.testing.assert_close(value, torch.ones_like(value))
        for value in (
            weights.preserve,
            weights.preserve_map,
            weights.preserve_action,
            weights.preserve_action_map,
            weights.conflict_map,
        ):
            torch.testing.assert_close(value, torch.zeros_like(value))

    def test_default_router_and_kv_keep_legacy_preserve_fallback(self):
        belief = make_belief(
            edit=0.0,
            preserve=0.0,
            edit_precision=0.0,
            preserve_precision=0.0,
        )
        target = torch.full((1, 1, 1, 2, 2), 3.0)
        source = torch.full_like(target, 10.0)
        source_reconstruction = torch.full_like(target, 14.0)

        routed, diagnostics = BayesResidualFlowRouter()(
            target_velocity=target,
            source_velocity=source,
            source_reconstruction_velocity=source_reconstruction,
            belief=belief,
        )
        torch.testing.assert_close(routed, torch.full_like(target, 7.0))
        torch.testing.assert_close(
            diagnostics["preserve_action_weight"],
            torch.ones_like(diagnostics["preserve_action_weight"]),
        )

        weights = build_belief_kv_weights(
            belief=belief,
            expected_token_length=1,
        )
        torch.testing.assert_close(
            weights.preserve,
            torch.ones_like(weights.preserve),
        )
        torch.testing.assert_close(
            weights.preserve_action,
            torch.ones_like(weights.preserve_action),
        )

    def test_unknown_stays_target_only_after_memory_consolidation(self):
        belief = make_belief(
            edit=0.0,
            preserve=0.0,
            edit_precision=0.0,
            preserve_precision=0.0,
        )
        weights = build_belief_kv_weights(
            belief=belief,
            expected_token_length=1,
            preserve_on_no_evidence=False,
        )

        preserve_action = build_cached_preserve_action(
            weights,
            materialized_edit_action=torch.zeros_like(weights.edit),
            preserve_on_no_evidence=False,
        )

        torch.testing.assert_close(
            preserve_action,
            torch.zeros_like(preserve_action),
        )


class EditCommitmentPredictionTests(unittest.TestCase):
    def test_prediction_does_not_advance_state(self):
        controller = EditCommitmentController(
            promote_track_to_visibility=False,
        )
        belief = make_belief(
            edit=0.8,
            preserve=0.0,
            edit_precision=1.0,
            preserve_precision=0.0,
        )
        debug = {
            "source_attention": torch.ones(1, 1, 1, 1),
            "hand_proximity": torch.ones(1, 1, 1, 1),
        }
        features = torch.tensor([[[1.0, 0.0]]])
        hand_mask = torch.ones(1, 1, 2, 2)

        prediction = controller(
            belief=belief,
            debug=debug,
            hand_mask=hand_mask,
            source_features=features,
            update_state=False,
        )
        self.assertIsNone(controller.previous_features)
        self.assertIsNone(controller.previous_commitment)
        self.assertIsNone(controller.previous_precision)

        committed = controller(
            belief=belief,
            debug=debug,
            hand_mask=hand_mask,
            source_features=features,
        )
        torch.testing.assert_close(
            prediction.commitment,
            committed.commitment,
        )
        self.assertIsNotNone(controller.previous_features)

        state_before_prediction = (
            controller.previous_features.clone(),
            controller.previous_commitment.clone(),
            controller.previous_precision.clone(),
        )
        controller(
            belief=belief,
            debug=debug,
            hand_mask=hand_mask,
            source_features=torch.tensor([[[0.0, 1.0]]]),
            update_state=False,
        )
        for actual, expected in zip(
            (
                controller.previous_features,
                controller.previous_commitment,
                controller.previous_precision,
            ),
            state_before_prediction,
        ):
            torch.testing.assert_close(actual, expected)

    def test_read_only_policy_does_not_expand_control_or_role_mask(self):
        observation_belief = make_belief(
            edit=0.0,
            preserve=0.0,
            edit_precision=0.0,
            preserve_precision=0.0,
        )
        controller = EditCommitmentController(
            promote_track_to_visibility=False,
        )
        seeded_belief = make_belief(
            edit=1.0,
            preserve=0.0,
            edit_precision=1.0,
            preserve_precision=0.0,
        )
        debug = {
            "source_attention": torch.ones(1, 1, 1, 1),
            "hand_proximity": torch.ones(1, 1, 1, 1),
        }
        features = torch.tensor([[[1.0, 0.0]]])
        hand_mask = torch.zeros(1, 1, 2, 2)
        controller(
            belief=seeded_belief,
            debug=debug,
            hand_mask=hand_mask,
            source_features=features,
        )
        transported = controller(
            belief=observation_belief,
            debug=debug,
            hand_mask=hand_mask,
            source_features=features,
        )
        self.assertGreater(transported.transported.max().item(), 0.0)

        routed_belief, routed_tokens = apply_edit_commitment_policy(
            observation_belief=observation_belief,
            observation_edit_tokens=torch.zeros(
                1, 1, dtype=torch.bool
            ),
            commitment=transported,
            transport_read_only=True,
        )

        self.assertIs(routed_belief, observation_belief)
        torch.testing.assert_close(
            routed_tokens,
            torch.zeros_like(routed_tokens),
        )

    def test_legacy_policy_still_applies_commitment_to_control(self):
        observation_belief = make_belief(
            edit=0.0,
            preserve=1.0,
            edit_precision=0.0,
            preserve_precision=1.0,
        )
        committed_belief = make_belief(
            edit=1.0,
            preserve=0.0,
            edit_precision=1.0,
            preserve_precision=0.0,
        )
        commitment = EditCommitmentResult(
            belief=committed_belief,
            trigger=torch.ones(1, 1, 1, 1),
            transported=torch.ones(1, 1, 1, 1),
            transport_precision=torch.ones(1, 1, 1, 1),
            anchor_transport=torch.zeros(1, 1, 1, 1),
            anchor_precision=torch.zeros(1, 1, 1, 1),
            semantic_presence=torch.ones(1, 1, 1, 1),
            semantic_absence=torch.zeros(1, 1, 1, 1),
            commitment=torch.ones(1, 1, 1, 1),
            commitment_precision=torch.ones(1, 1, 1, 1),
            state_precision=torch.ones(1, 1, 1, 1),
            effective_commitment=torch.ones(1, 1, 1, 1),
            edit_support=torch.ones(1, 1, 1, 1, dtype=torch.bool),
        )

        routed_belief, routed_tokens = apply_edit_commitment_policy(
            observation_belief=observation_belief,
            observation_edit_tokens=torch.zeros(
                1, 1, dtype=torch.bool
            ),
            commitment=commitment,
        )

        self.assertIs(routed_belief, committed_belief)
        torch.testing.assert_close(
            routed_tokens,
            torch.ones_like(routed_tokens),
        )


class PersistentIdentityReadMaskTests(unittest.TestCase):
    def test_read_mask_unions_observation_and_track_outside_hand(self):
        observation = torch.tensor(
            [[True, False, False, False]],
            dtype=torch.bool,
        )
        observation_before = observation.clone()
        transported_commitment = torch.tensor(
            [[[[0.0, 0.8], [0.9, 0.9]]]],
            dtype=torch.float32,
        )
        transport_precision = torch.ones_like(transported_commitment)
        # Object space already excludes the hand token at bottom left.
        object_space = torch.tensor(
            [[[[1.0, 1.0], [0.0, 1.0]]]],
            dtype=torch.float32,
        )

        read_mask, track_evidence, track_mask = (
            build_persistent_identity_read_mask(
                observation_mask=observation,
                transported_commitment=transported_commitment,
                transport_precision=transport_precision,
                object_space=object_space,
                min_evidence=0.5,
            )
        )

        expected_track = torch.tensor(
            [[[[False, True], [False, True]]]],
            dtype=torch.bool,
        )
        expected_read = torch.tensor(
            [[True, True, False, True]],
            dtype=torch.bool,
        )
        torch.testing.assert_close(track_mask, expected_track)
        torch.testing.assert_close(
            track_evidence,
            transported_commitment * object_space,
        )
        torch.testing.assert_close(read_mask, expected_read)
        self.assertFalse(read_mask[0, 2].item())
        torch.testing.assert_close(observation, observation_before)

    def test_track_only_support_cannot_expand_routing(self):
        identity_support = torch.tensor(
            [[[[0.8, 0.7], [0.6, 0.5]]]],
            dtype=torch.float32,
        )
        observation = torch.tensor(
            [[True, False, False, False]],
            dtype=torch.bool,
        )

        routing_support = restrict_identity_support_to_observation(
            identity_support=identity_support,
            observation_mask=observation,
        )

        torch.testing.assert_close(
            routing_support,
            torch.tensor([[[[0.8, 0.0], [0.0, 0.0]]]]),
        )


class VariableLengthTokenHistoryTests(unittest.TestCase):
    def test_mixed_batch_retention_handles_changed_token_length(self):
        propagator = CausalObjectTokenPropagator(
            retain_on_empty_support=True,
            max_candidates=8,
        )
        initial_features = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]],
            ]
        )
        initial_support = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        propagator(
            source_features=initial_features,
            base_write_weight=initial_support,
            support_weight=initial_support,
        )
        previous_second_features = (
            propagator.previous_features[1].clone()
        )
        previous_second_weight = propagator.previous_weight[1].clone()

        shorter_features = torch.tensor(
            [
                [[0.0, 1.0], [1.0, 0.0]],
                [[1.0, 1.0], [1.0, -1.0]],
            ]
        )
        shorter_support = torch.tensor(
            [[0.5, 0.0], [0.0, 0.0]]
        )
        propagator(
            source_features=shorter_features,
            base_write_weight=torch.zeros(2, 2),
            support_weight=shorter_support,
        )

        self.assertEqual(propagator.previous_weight.shape, (2, 3))
        torch.testing.assert_close(
            propagator.previous_weight[0],
            torch.tensor([0.5, 0.0, 0.0]),
        )
        torch.testing.assert_close(
            propagator.previous_features[1],
            previous_second_features,
        )
        torch.testing.assert_close(
            propagator.previous_weight[1],
            previous_second_weight,
        )


if __name__ == "__main__":
    unittest.main()
