import unittest

import torch

from tests._pipeline_imports import load_pipeline_module


semantic = load_pipeline_module("semantic_edit_authority")
factorized = load_pipeline_module("factorized_bayes")
role_router = load_pipeline_module("role_router")

build_semantic_edit_authority = semantic.build_semantic_edit_authority
apply_semantic_transaction_gate = (
    semantic.apply_semantic_transaction_gate
)
FactorizedBayesOperatorBuilder = factorized.FactorizedBayesOperatorBuilder
route_factorized_velocity = factorized.route_factorized_velocity
RoleState = role_router.RoleState


class SemanticEditAuthorityTest(unittest.TestCase):
    def test_semantic_permission_does_not_attenuate_transaction(self):
        transaction = torch.tensor([[0.75, 0.60, 0.40]])
        support = torch.tensor([[True, False, True]])
        gated = apply_semantic_transaction_gate(transaction, support)
        torch.testing.assert_close(
            gated, torch.tensor([[0.75, 0.0, 0.40]])
        )

    def test_preserve_semantics_veto_background_and_body(self):
        edit = torch.tensor([[[[0.0, 4.0, 1.0, 0.0]]]])
        preserve = torch.tensor([[[[3.0, 0.0, 4.0, 0.0]]]])
        owner = torch.tensor([[[[0.0, 1.0, 1.0, 0.0]]]])
        result = build_semantic_edit_authority(
            edit_attention={"cap": edit},
            preserve_attention={"body_or_pan": preserve},
            owner_weight=owner,
            margin=0.10,
            min_confidence=0.20,
            low_quantile=0.0,
            high_quantile=1.0,
        )
        self.assertFalse(result.support[0, 0, 0, 0])
        self.assertTrue(result.support[0, 0, 0, 1])
        self.assertFalse(result.support[0, 0, 0, 2])
        self.assertFalse(result.support[0, 0, 0, 3])

    def test_factorized_target_memory_uses_part_authority_not_owner(self):
        shape = (1, 1, 2, 4)
        roles = RoleState(
            object=torch.ones(shape),
            boundary=torch.zeros(shape),
            hand=torch.zeros(shape),
            background=torch.zeros(shape),
        )
        evidence = {
            "object_posterior": torch.ones(shape),
            "posterior_threshold": torch.full(shape, 0.2),
            "source_attention": torch.ones(shape),
            "hand_proximity": torch.ones(shape),
            "adaptive_attention_reliability": torch.ones(shape),
            "object_visible": torch.ones(shape),
            "temporal_confidence": torch.ones(shape),
            "causal_owner_weight": torch.ones(shape),
            "causal_owner_support": torch.ones(shape),
            "edit_authority": torch.tensor(
                [[[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]]]
            ),
        }
        operators = FactorizedBayesOperatorBuilder()(
            roles=roles, evidence=evidence, expected_token_length=2
        )
        torch.testing.assert_close(
            operators.target_memory_action, torch.tensor([[0.0, 1.0]])
        )
        torch.testing.assert_close(
            operators.source_value_action, torch.tensor([[1.0, 0.0]])
        )
        # Whole-object ownership remains available for geometry/source keys.
        torch.testing.assert_close(
            operators.roles.target_owned, torch.ones(shape)
        )

    def test_velocity_is_exact_source_outside_authority(self):
        shape = (1, 1, 2, 4)
        roles = RoleState(
            object=torch.ones(shape),
            boundary=torch.zeros(shape),
            hand=torch.zeros(shape),
            background=torch.zeros(shape),
        )
        authority = torch.tensor(
            [[[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]]]
        )
        evidence = {
            "object_posterior": torch.ones(shape),
            "posterior_threshold": torch.full(shape, 0.2),
            "source_attention": torch.ones(shape),
            "hand_proximity": torch.ones(shape),
            "adaptive_attention_reliability": torch.ones(shape),
            "object_visible": torch.ones(shape),
            "temporal_confidence": torch.ones(shape),
            "edit_authority": authority,
        }
        operators = FactorizedBayesOperatorBuilder()(
            roles=roles, evidence=evidence, expected_token_length=2
        )
        target = torch.full((1, 1, 1, 2, 4), 10.0)
        source = torch.zeros_like(target)
        reconstruction = torch.full_like(target, 2.0)
        routed, diagnostics = route_factorized_velocity(
            target,
            source,
            reconstruction,
            operators,
            torch.zeros(1, 1, 1, 2, 4),
            target_owned_weight=authority,
            block_target_owned_source=True,
            edit_authority_weight=authority,
        )
        torch.testing.assert_close(
            routed[..., :2], reconstruction[..., :2]
        )
        torch.testing.assert_close(routed[..., 2:], target[..., 2:])
        torch.testing.assert_close(
            diagnostics["semantic_preserve_action"][..., :2],
            torch.ones_like(reconstruction[..., :2]),
        )


if __name__ == "__main__":
    unittest.main()
