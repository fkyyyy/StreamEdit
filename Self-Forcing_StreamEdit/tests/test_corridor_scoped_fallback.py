from __future__ import annotations

import unittest

import torch

from tests._pipeline_imports import load_pipeline_module


control_belief_module = load_pipeline_module("control_belief")
role_router_module = load_pipeline_module("role_router")
belief_kv_module = load_pipeline_module("belief_kv")

CausalControlBelief = control_belief_module.CausalControlBelief
CausalControlBeliefBuilder = (
    control_belief_module.CausalControlBeliefBuilder
)
BayesResidualFlowRouter = role_router_module.BayesResidualFlowRouter
build_belief_kv_weights = belief_kv_module.build_belief_kv_weights
build_cached_preserve_action = (
    belief_kv_module.build_cached_preserve_action
)


def make_unknown_belief(
    shape: tuple[int, int, int, int] = (1, 1, 2, 4),
):
    zeros = torch.zeros(shape, dtype=torch.float32)
    return CausalControlBelief(
        edit_belief=zeros.clone(),
        preserve_belief=zeros.clone(),
        edit_precision=zeros.clone(),
        preserve_precision=zeros.clone(),
        visibility=zeros.clone(),
        uncertainty=torch.ones_like(zeros),
        conflict=zeros.clone(),
    )


def make_velocity_inputs():
    target = torch.full((1, 1, 1, 2, 4), 3.0)
    source = torch.full_like(target, 10.0)
    source_reconstruction = torch.full_like(target, 14.0)
    return target, source, source_reconstruction


class ScopedControlBeliefTests(unittest.TestCase):
    def test_tracked_dropout_only_is_target_only_in_worst_case(self):
        builder = CausalControlBeliefBuilder(
            preserve_on_observation_dropout=False,
            use_target_field_evidence=False,
            scope_target_only_to_tracked_dropout=True,
        )
        token_shape = (1, 1, 1, 2)

        belief = builder(
            debug={
                "object_posterior": torch.ones(token_shape),
                "source_attention": torch.ones(token_shape),
                "temporal_confidence": torch.zeros(token_shape),
                "observation_visible": torch.zeros(1, 1, 1, 1),
                "hand_proximity": torch.ones(token_shape),
                "adaptive_attention_reliability": torch.ones(
                    1, 1, 1, 1
                ),
                "identity_track_tokens_preforward": torch.tensor(
                    [[[[1.0, 0.0]]]]
                ),
            },
            hand_mask=torch.zeros(1, 1, 2, 4),
        )

        expected_preserve = torch.tensor(
            [[[[0.0, 0.0, 1.0, 1.0],
               [0.0, 0.0, 1.0, 1.0]]]]
        )
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
            expected_preserve,
        )


class ScopedResidualRouterTests(unittest.TestCase):
    def test_mask_scopes_target_only_fallback_and_is_detached(self):
        target, source, source_reconstruction = make_velocity_inputs()
        target_only_mask = torch.tensor(
            [[[[1.0, 0.0]]]],
            requires_grad=True,
        )

        routed, diagnostics = BayesResidualFlowRouter(
            preserve_on_no_evidence=False
        )(
            target_velocity=target,
            source_velocity=source,
            source_reconstruction_velocity=source_reconstruction,
            belief=make_unknown_belief(),
            target_only_no_evidence_mask=target_only_mask,
        )

        expected_preserve = torch.tensor(
            [[[[[0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0]]]]]
        )
        torch.testing.assert_close(
            routed,
            target + expected_preserve * 4.0,
        )
        torch.testing.assert_close(
            diagnostics["edit_action_weight"],
            1.0 - expected_preserve,
        )
        torch.testing.assert_close(
            diagnostics["preserve_action_weight"],
            expected_preserve,
        )
        torch.testing.assert_close(
            diagnostics["no_evidence"],
            torch.ones_like(diagnostics["no_evidence"]),
        )
        self.assertTrue(target_only_mask.requires_grad)
        self.assertFalse(routed.requires_grad)
        self.assertFalse(
            diagnostics["preserve_action_weight"].requires_grad
        )

    def test_no_mask_keeps_908_global_target_only_fallback(self):
        target, source, source_reconstruction = make_velocity_inputs()

        routed, diagnostics = BayesResidualFlowRouter(
            preserve_on_no_evidence=False
        )(
            target_velocity=target,
            source_velocity=source,
            source_reconstruction_velocity=source_reconstruction,
            belief=make_unknown_belief(),
        )

        torch.testing.assert_close(routed, target)
        torch.testing.assert_close(
            diagnostics["edit_action_weight"],
            torch.ones_like(diagnostics["edit_action_weight"]),
        )
        torch.testing.assert_close(
            diagnostics["preserve_action_weight"],
            torch.zeros_like(diagnostics["preserve_action_weight"]),
        )

    def test_rejects_invalid_target_only_mask_shapes(self):
        target, source, source_reconstruction = make_velocity_inputs()
        router = BayesResidualFlowRouter(preserve_on_no_evidence=False)
        common = {
            "target_velocity": target,
            "source_velocity": source,
            "source_reconstruction_velocity": source_reconstruction,
            "belief": make_unknown_belief(),
        }

        with self.assertRaisesRegex(ValueError, r"shape \[B,T,H,W\]"):
            router(
                **common,
                target_only_no_evidence_mask=torch.ones(1, 1, 2),
            )
        with self.assertRaisesRegex(ValueError, r"share \[B,T\]"):
            router(
                **common,
                target_only_no_evidence_mask=torch.ones(1, 2, 1, 2),
            )


class ScopedBeliefKVTests(unittest.TestCase):
    def test_mask_scopes_preserve_action_and_is_detached(self):
        target_only_mask = torch.tensor(
            [[[[1.0, 0.0]]]],
            requires_grad=True,
        )

        weights = build_belief_kv_weights(
            belief=make_unknown_belief(),
            expected_token_length=2,
            preserve_on_no_evidence=False,
            target_only_no_evidence_mask=target_only_mask,
        )

        torch.testing.assert_close(
            weights.edit,
            torch.tensor([[0.0, 0.0]]),
        )
        torch.testing.assert_close(
            weights.preserve,
            torch.tensor([[0.0, 1.0]]),
        )
        torch.testing.assert_close(
            weights.edit_action,
            torch.tensor([[1.0, 0.0]]),
        )
        torch.testing.assert_close(
            weights.preserve_action,
            torch.tensor([[0.0, 1.0]]),
        )
        self.assertTrue(target_only_mask.requires_grad)
        self.assertFalse(weights.preserve.requires_grad)
        self.assertFalse(weights.preserve_action.requires_grad)

    def test_partial_patch_is_not_allowed_to_leak_target_only_to_kv(self):
        target_only_mask = torch.tensor(
            [[[[1.0, 0.0], [1.0, 0.0]]]]
        )

        weights = build_belief_kv_weights(
            belief=make_unknown_belief(shape=(1, 1, 2, 2)),
            expected_token_length=1,
            preserve_on_no_evidence=False,
            target_only_no_evidence_mask=target_only_mask,
        )

        torch.testing.assert_close(
            weights.edit_action,
            torch.zeros_like(weights.edit_action),
        )
        torch.testing.assert_close(
            weights.preserve_action,
            torch.ones_like(weights.preserve_action),
        )

    def test_no_mask_keeps_908_global_target_only_fallback(self):
        weights = build_belief_kv_weights(
            belief=make_unknown_belief(),
            expected_token_length=2,
            preserve_on_no_evidence=False,
        )

        torch.testing.assert_close(
            weights.edit_action,
            torch.ones_like(weights.edit_action),
        )
        torch.testing.assert_close(
            weights.preserve_action,
            torch.zeros_like(weights.preserve_action),
        )
        torch.testing.assert_close(
            weights.preserve,
            torch.zeros_like(weights.preserve),
        )

    def test_cached_preserve_action_keeps_the_same_corridor_partition(self):
        weights = build_belief_kv_weights(
            belief=make_unknown_belief(),
            expected_token_length=2,
            preserve_on_no_evidence=False,
            target_only_no_evidence_mask=torch.tensor(
                [[[[1.0, 0.0]]]]
            ),
        )

        preserve_action = build_cached_preserve_action(
            weights,
            materialized_edit_action=torch.zeros_like(weights.edit),
            preserve_on_no_evidence=False,
        )

        torch.testing.assert_close(
            preserve_action,
            torch.tensor([[0.0, 1.0]]),
        )

    def test_rejects_invalid_target_only_mask_shapes(self):
        common = {
            "belief": make_unknown_belief(),
            "expected_token_length": 2,
            "preserve_on_no_evidence": False,
        }

        with self.assertRaisesRegex(ValueError, r"shape \[B,T,H,W\]"):
            build_belief_kv_weights(
                **common,
                target_only_no_evidence_mask=torch.ones(1, 1, 2),
            )
        with self.assertRaisesRegex(ValueError, r"share \[B,T\]"):
            build_belief_kv_weights(
                **common,
                target_only_no_evidence_mask=torch.ones(1, 2, 1, 2),
            )


if __name__ == "__main__":
    unittest.main()
