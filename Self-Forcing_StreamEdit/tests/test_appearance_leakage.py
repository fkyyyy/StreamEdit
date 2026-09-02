from __future__ import annotations

import unittest

import torch

from tests._pipeline_imports import load_pipeline_module


appearance_module = load_pipeline_module("appearance_leakage")
role_router_module = load_pipeline_module("role_router")
control_belief_module = load_pipeline_module("control_belief")
build_target_change_core = appearance_module.build_target_change_core
remove_antagonistic_source_residual = (
    appearance_module.remove_antagonistic_source_residual
)
BayesResidualFlowRouter = role_router_module.BayesResidualFlowRouter
CausalControlBelief = control_belief_module.CausalControlBelief


class AppearanceLeakageProjectionTests(unittest.TestCase):
    def test_disabled_core_is_exact_legacy_residual(self):
        residual = torch.randn(1, 2, 4, 3, 5, dtype=torch.bfloat16)
        direction = torch.randn_like(residual)
        core = torch.zeros(1, 2, 3, 5, dtype=torch.bool)

        filtered, _ = remove_antagonistic_source_residual(
            residual, direction, core
        )

        self.assertTrue(torch.equal(filtered, residual))

    def test_only_antagonistic_component_is_removed(self):
        residual = torch.tensor(
            [[[[[-2.0]], [[3.0]], [[0.0]]]]],
            dtype=torch.float32,
        )
        direction = torch.tensor(
            [[[[[1.0]], [[0.0]], [[0.0]]]]],
            dtype=torch.float32,
        )
        core = torch.ones(1, 1, 1, 1, dtype=torch.bool)

        filtered, diagnostics = remove_antagonistic_source_residual(
            residual, direction, core
        )

        torch.testing.assert_close(
            filtered,
            torch.tensor([[[[[0.0]], [[3.0]], [[0.0]]]]]),
            rtol=0.0,
            atol=2e-5,
        )
        self.assertGreater(
            diagnostics["appearance_leakage_removed_energy"].item(),
            0.0,
        )

    def test_compatible_residual_is_unchanged(self):
        residual = torch.tensor(
            [[[[[2.0]], [[3.0]], [[0.0]]]]],
            dtype=torch.float32,
        )
        direction = torch.tensor(
            [[[[[1.0]], [[0.0]], [[0.0]]]]],
            dtype=torch.float32,
        )
        filtered, _ = remove_antagonistic_source_residual(
            residual,
            direction,
            torch.ones(1, 1, 1, 1, dtype=torch.bool),
        )
        torch.testing.assert_close(filtered, residual, rtol=0.0, atol=0.0)

    def test_core_outside_and_hand_are_bitwise_unchanged(self):
        residual = torch.tensor(
            [[[[[-1.0, -2.0, -3.0]]]]],
            dtype=torch.float32,
        )
        direction = torch.ones_like(residual)
        core = torch.tensor([[[[False, True, True]]]])
        hand = torch.tensor([[[[False, False, True]]]])

        filtered, _ = remove_antagonistic_source_residual(
            residual, direction, core, protect_mask=hand
        )

        self.assertTrue(torch.equal(filtered[..., 0], residual[..., 0]))
        self.assertTrue(torch.equal(filtered[..., 2], residual[..., 2]))
        self.assertAlmostEqual(filtered[..., 1].item(), 0.0, places=5)

    def test_bayes_router_keeps_weighted_orthogonal_motion(self):
        map_shape = (1, 1, 1, 1)
        belief = CausalControlBelief(
            edit_belief=torch.full(map_shape, 0.5),
            preserve_belief=torch.full(map_shape, 0.5),
            edit_precision=torch.ones(map_shape),
            preserve_precision=torch.ones(map_shape),
            visibility=torch.ones(map_shape),
            uncertainty=torch.zeros(map_shape),
            conflict=torch.zeros(map_shape),
        )
        target = torch.tensor(
            [[[[[1.0]], [[0.0]], [[0.0]]]]]
        )
        source = torch.zeros_like(target)
        reconstruction = torch.tensor(
            [[[[[-2.0]], [[4.0]], [[0.0]]]]]
        )

        routed, diagnostics = BayesResidualFlowRouter()(
            target_velocity=target,
            source_velocity=source,
            source_reconstruction_velocity=reconstruction,
            belief=belief,
            target_change_core=torch.ones(
                1, 1, 1, 1, dtype=torch.bool
            ),
        )

        torch.testing.assert_close(
            routed,
            torch.tensor([[[[[1.0]], [[2.0]], [[0.0]]]]]),
            rtol=0.0,
            atol=2e-5,
        )
        self.assertGreater(
            diagnostics["appearance_leakage_removed_energy"].item(),
            0.0,
        )

    def test_identity_owner_softly_suppresses_only_early_source_residual(self):
        shape = (1, 1, 1, 2)
        zeros = torch.zeros(shape)
        belief = CausalControlBelief(
            edit_belief=zeros.clone(),
            preserve_belief=torch.ones(shape),
            edit_precision=zeros.clone(),
            preserve_precision=torch.ones(shape),
            visibility=torch.ones(shape),
            uncertainty=zeros.clone(),
            conflict=zeros.clone(),
        )
        target = torch.zeros(1, 1, 1, 1, 2)
        source = torch.zeros_like(target)
        reconstruction = torch.full_like(target, 4.0)

        routed, diagnostics = BayesResidualFlowRouter()(
            target_velocity=target,
            source_velocity=source,
            source_reconstruction_velocity=reconstruction,
            belief=belief,
            identity_owner_weight=torch.tensor([[[[1.0, 0.0]]]]),
            identity_source_suppression=0.25,
            denoising_fraction=0.8,
        )

        torch.testing.assert_close(
            routed, torch.tensor([[[[[3.2, 4.0]]]]])
        )
        torch.testing.assert_close(
            diagnostics["identity_source_suppression"],
            torch.tensor([[[[[0.2, 0.0]]]]]),
        )


class TargetChangeCoreTests(unittest.TestCase):
    def test_core_is_semantic_connected_and_excludes_hand(self):
        source = torch.zeros(1, 1, 2, 7, 7)
        target = source.clone()
        target[:, :, :, 2:5, 2:5] = 4.0
        semantic = torch.zeros(1, 1, 7, 7)
        semantic[:, :, 2:5, 2:5] = 1.0
        hand = torch.zeros(1, 1, 7, 7, dtype=torch.bool)
        hand[:, :, 5, 3] = True

        result = build_target_change_core(
            source_velocity=source,
            target_velocity=target,
            source_semantic_attention=semantic,
            hand_mask=hand,
            hand_exclusion_radius=0,
            contact_radius=2,
        )

        self.assertTrue(result.mask.any())
        self.assertFalse((result.mask & hand).any())
        self.assertFalse(result.mask[:, :, 0, 0].any())
        self.assertTrue(
            (result.mask & result.contact_ring).any(),
            "Selected component must be attached to the hand interaction",
        )

    def test_no_contact_produces_no_core(self):
        source = torch.zeros(1, 1, 1, 8, 8)
        target = source.clone()
        target[:, :, :, :2, :2] = 4.0
        semantic = torch.zeros(1, 1, 8, 8)
        semantic[:, :, :2, :2] = 1.0
        hand = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
        hand[:, :, 7, 7] = True

        result = build_target_change_core(
            source, target, semantic, hand,
            hand_exclusion_radius=0,
            contact_radius=1,
        )

        self.assertFalse(result.mask.any())


if __name__ == "__main__":
    unittest.main()
