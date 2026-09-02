from __future__ import annotations

import unittest

import torch

from tests._pipeline_imports import load_pipeline_module


hand_role_module = load_pipeline_module("hand_role_inference")
HandRoleInferenceResult = hand_role_module.HandRoleInferenceResult
HandRoleInferencer = hand_role_module.HandRoleInferencer
FlowRoleEvidence = load_pipeline_module(
    "motion/flow_role_evidence"
).FlowRoleEvidence
connected_hysteresis_growth = (
    hand_role_module._connected_hysteresis_growth
)


class ConnectedHysteresisTests(unittest.TestCase):
    def test_recovers_connected_thin_extent_but_not_disconnected_response(self):
        seed = torch.zeros(1, 1, 5, 7, dtype=torch.bool)
        seed[0, 0, 2, 1] = True
        candidate = torch.zeros_like(seed)
        candidate[0, 0, 2, 1:6] = True
        candidate[0, 0, 0, 6] = True

        grown = connected_hysteresis_growth(
            seed, candidate, steps=4
        )

        self.assertTrue(grown[0, 0, 2, 1:6].all())
        self.assertFalse(grown[0, 0, 0, 6])

    def test_growth_is_distance_bounded(self):
        seed = torch.zeros(1, 1, 3, 7, dtype=torch.bool)
        seed[0, 0, 1, 0] = True
        candidate = torch.zeros_like(seed)
        candidate[0, 0, 1, :] = True

        grown = connected_hysteresis_growth(
            seed, candidate, steps=3
        )

        self.assertTrue(grown[0, 0, 1, :4].all())
        self.assertFalse(grown[0, 0, 1, 4:].any())

    def test_high_confidence_seed_survives_candidate_mismatch(self):
        seed = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
        seed[0, 0, 1, 1] = True
        candidate = torch.zeros_like(seed)

        grown = connected_hysteresis_growth(seed, candidate, steps=3)

        self.assertTrue(grown[0, 0, 1, 1])

    def test_soft_hand_occupancy_does_not_turn_full_contact_band_read_only(self):
        posterior = torch.ones(1, 1, 2, 3)
        occupancy = torch.zeros(1, 1, 4, 6)
        occupancy[:, :, 1:3, 2:4] = 0.25

        roles = HandRoleInferencer._build_roles(
            posterior, occupancy, soft_hand_contact=True
        )

        self.assertGreater(roles.object.mean().item(), 0.90)
        self.assertLess(roles.boundary.mean().item(), 0.10)

    def test_default_role_partition_retains_legacy_dilated_hand_band(self):
        posterior = torch.ones(1, 1, 2, 3)
        hand = torch.zeros(1, 1, 4, 6)
        hand[:, :, 2, 3] = 1.0

        roles = HandRoleInferencer._build_roles(posterior, hand)

        self.assertEqual(roles.boundary.sum().item(), 9.0)


class HandRoleFieldInferenceTests(unittest.TestCase):
    def test_field_refinement_cannot_prune_connected_extent(self):
        inferencer = HandRoleInferencer(
            adaptive=False,
            connected_hysteresis=True,
            max_object_coverage=0.01,
        )
        posterior = torch.zeros(1, 1, 4, 4)
        posterior[0, 0, 0, 0] = 1.0
        posterior[0, 0, 3, 3] = 0.20
        connected = torch.zeros_like(posterior)
        connected[0, 0, 3, 3] = 1.0
        hand = torch.zeros(1, 1, 8, 8)
        prior = HandRoleInferenceResult(
            roles=inferencer._build_roles(posterior, hand),
            token_edit_confidence=posterior.reshape(1, -1),
            debug={
                "object_posterior": posterior,
                "source_attention": torch.ones_like(posterior),
                "object_visible": torch.ones(1, 1, 1, 1),
                "object_seed": posterior.clone(),
                "hand_proximity": torch.ones_like(posterior),
                "posterior_threshold": torch.full(
                    (1, 1, 1, 1), 0.20
                ),
                "connected_hysteresis_support": connected,
            },
        )
        source_velocity = torch.zeros(1, 1, 2, 8, 8)

        refined = inferencer.refine_with_field(
            prior=prior,
            source_velocity=source_velocity,
            target_velocity=source_velocity.clone(),
            hand_mask=hand,
            apply_update=True,
        )

        self.assertGreaterEqual(
            refined.debug["object_posterior"][0, 0, 3, 3].item(),
            refined.debug["posterior_threshold"][0, 0, 0, 0].item(),
        )

    def test_nonadaptive_proximity_uses_union_not_soft_occupancy(self):
        inferencer = HandRoleInferencer(adaptive=False)
        hand_union = torch.zeros(1, 1, 4, 4)
        hand_union[0, 0, 0, 0] = 1.0
        occupancy = torch.zeros_like(hand_union)
        source_attention = torch.tensor([[0.0, 1.0, 0.5, 0.2]])

        result = inferencer(
            source_attention=source_attention,
            hand_mask=hand_union,
            hand_occupancy=occupancy,
        )

        self.assertEqual(result.debug["hand_probability"].sum().item(), 0.0)
        self.assertGreater(result.debug["hand_proximity"].sum().item(), 0.0)

    def test_adaptive_field_can_expand_only_inside_hand_semantic_corridor(self):
        inferencer = HandRoleInferencer(
            adaptive=True,
            field_weight=1.0,
            field_candidate_radius=1,
        )
        shape = (1, 1, 4, 4)
        prior_posterior = torch.zeros(shape)
        prior_posterior[0, 0, 1, 1] = 1.0
        source_attention = torch.zeros(shape)
        source_attention[0, 0, 1, 1] = 1.0
        source_attention[0, 0, 1, 2] = 1.0
        source_attention[0, 0, 3, 3] = 1.0
        proximity = torch.zeros(shape)
        proximity[0, 0, 1, 1] = 1.0
        proximity[0, 0, 1, 2] = 0.8
        proximity[0, 0, 3, 3] = 0.0
        hand_mask = torch.zeros(1, 1, 8, 8)
        prior = HandRoleInferenceResult(
            roles=inferencer._build_roles(prior_posterior, hand_mask),
            token_edit_confidence=prior_posterior.reshape(1, -1),
            debug={
                "object_posterior": prior_posterior,
                "source_attention": source_attention,
                "object_visible": torch.ones(1, 1, 1, 1),
                "object_seed": prior_posterior.clone(),
                "hand_proximity": proximity,
                "adaptive_coverage_budget": torch.full(
                    (1, 1, 1, 1), 0.5
                ),
                "posterior_threshold": torch.full(
                    (1, 1, 1, 1), 0.1
                ),
            },
        )
        source_velocity = torch.zeros(1, 1, 2, 8, 8)
        target_velocity = source_velocity.clone()
        target_velocity[:, :, :, 2:4, 2:6] = 1.0

        refined = inferencer.refine_with_field(
            prior=prior,
            source_velocity=source_velocity,
            target_velocity=target_velocity,
            hand_mask=hand_mask,
            apply_update=True,
        )

        # The adjacent semantic/hand-corridor token is recovered by flow.
        self.assertGreater(
            refined.debug["object_posterior"][0, 0, 1, 2].item(),
            0.0,
        )
        # A strong semantic/field response outside the hand corridor cannot
        # independently create object ownership.
        self.assertEqual(
            refined.debug["object_posterior"][0, 0, 3, 3].item(),
            0.0,
        )


class SourceFlowVerifiedHandRoleTests(unittest.TestCase):
    def test_rebuilds_roles_and_preserves_first_unverified_proposal(self):
        inferencer = HandRoleInferencer(adaptive=False)
        shape = (1, 1, 3, 4)
        posterior = torch.zeros(shape)
        posterior[0, 0, 1, 1:4] = 0.9
        fused_posterior = posterior.clone()
        fused_posterior[0, 0, 0, 3] = 0.95
        threshold = torch.full((1, 1, 1, 1), 0.5)
        hand_latent = torch.zeros(1, 1, 6, 8)
        prior = HandRoleInferenceResult(
            roles=inferencer._build_roles(posterior, hand_latent),
            token_edit_confidence=posterior.reshape(1, -1),
            debug={
                "object_posterior": fused_posterior,
                "object_posterior_pre_source_flow": posterior,
                "posterior_threshold": threshold,
            },
        )
        owner = torch.zeros(shape)
        owner[0, 0, 1, 1] = 1.0
        zeros = torch.zeros(shape)
        ones = torch.ones(shape)
        flow = FlowRoleEvidence(
            object_likelihood=owner,
            background_likelihood=zeros,
            boundary_likelihood=zeros,
            unknown_likelihood=zeros,
            cycle_confidence=ones,
            transport_support=owner,
        )

        first = inferencer.apply_source_flow_verified_region(
            prior,
            flow,
            owner_support=owner.bool(),
            hand_exclusion=zeros,
            hand_occupancy=hand_latent,
            owner_radius=1,
        )
        second = inferencer.apply_source_flow_verified_region(
            first,
            flow,
            owner_support=owner.bool(),
            hand_exclusion=zeros,
            hand_occupancy=hand_latent,
            owner_radius=1,
        )

        self.assertTrue(
            torch.equal(
                second.debug["object_posterior_pre_flow_verification"],
                posterior,
            )
        )
        # A positive flow-fusion response is not counted twice as a token
        # proposal; it can enter only through explicit owner recovery.
        self.assertFalse(first.debug["source_flow_verified_support"][
            0, 0, 0, 3
        ].bool())
        self.assertTrue(
            torch.equal(
                second.debug[
                    "object_posterior_pre_latest_flow_verification"
                ],
                first.debug["object_posterior"],
            )
        )
        self.assertTrue(
            torch.equal(
                second.token_edit_confidence,
                second.debug["object_posterior"].reshape(1, -1),
            )
        )
        self.assertLess(
            second.roles.edit_weight.mean().item(),
            prior.roles.edit_weight.mean().item(),
        )
        self.assertTrue(
            torch.equal(
                inferencer.previous_posterior,
                second.debug["object_posterior"][:, -1],
            )
        )


if __name__ == "__main__":
    unittest.main()
