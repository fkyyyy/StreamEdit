from __future__ import annotations

import unittest

import torch

from tests._pipeline_imports import load_pipeline_module


role_router = load_pipeline_module("role_router")
factorized_bayes = load_pipeline_module("factorized_bayes")
causal_ownership = load_pipeline_module("causal_ownership")

RoleState = role_router.RoleState
FactorizedBayesOperatorBuilder = (
    factorized_bayes.FactorizedBayesOperatorBuilder
)
CausalObjectOwnershipTracker = (
    causal_ownership.CausalObjectOwnershipTracker
)
CausalReadOnlyOwnerTracker = (
    causal_ownership.CausalReadOnlyOwnerTracker
)
AutomaticTransactionalOwnerTracker = (
    causal_ownership.AutomaticTransactionalOwnerTracker
)
CausalObjectOwnership = causal_ownership.CausalObjectOwnership
CausalOwnershipState = causal_ownership.CausalOwnershipState
TransactionalOwnerSupport = causal_ownership.TransactionalOwnerSupport
build_motion_owner_read_weight = (
    causal_ownership.build_motion_owner_read_weight
)
build_topology_complete_motion_owner_read_weight = (
    causal_ownership.build_topology_complete_motion_owner_read_weight
)
build_oracle_causal_ownership = (
    causal_ownership.build_oracle_causal_ownership
)


def _features(object_index: int) -> torch.Tensor:
    features = torch.tensor(
        [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    features[0, object_index] = torch.tensor([1.0, 0.0])
    return features


class CausalObjectOwnershipTests(unittest.TestCase):
    @staticmethod
    def _ownership(
        *,
        owner: torch.Tensor,
        observation: torch.Tensor | None = None,
        transported: torch.Tensor | None = None,
        match_confidence: torch.Tensor | None = None,
        state: int = int(CausalOwnershipState.VISIBLE),
        missing: int = 0,
    ):
        observation = owner if observation is None else observation
        transported = owner if transported is None else transported
        match_confidence = (
            torch.ones_like(owner)
            if match_confidence is None
            else match_confidence
        )
        frames = 1
        return CausalObjectOwnership(
            owner_weight=owner,
            owner_support=owner > 0.0,
            transported_weight=transported,
            observation_weight=observation,
            match_similarity=match_confidence.mul(2.0).sub(1.0),
            match_confidence=match_confidence,
            semantic_support=owner.clone(),
            state_code=torch.full(
                (owner.shape[0], frames), state, dtype=torch.long
            ),
            missing_frames=torch.full(
                (owner.shape[0], frames), missing, dtype=torch.long
            ),
        )

    def test_automatic_transaction_uses_no_object_matte_and_separates_roles(self):
        tracker = AutomaticTransactionalOwnerTracker(
            max_missing_frames=1
        )
        owner = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        result = tracker(
            ownership=self._ownership(owner=owner),
            object_posterior=owner.reshape(1, 1, 2, 2),
            posterior_threshold=torch.full((1, 1, 1, 1), 0.2),
            source_attention=owner.reshape(1, 1, 2, 2),
            hand_probability=torch.tensor(
                [[[[0.0, 0.8], [0.0, 0.0]]]]
            ),
            hand_proximity=torch.tensor(
                [[[[0.2, 1.0], [0.0, 0.0]]]]
            ),
            object_role=torch.tensor(
                [[[[1.0, 0.0], [0.0, 0.0]]]]
            ),
            boundary_role=torch.tensor(
                [[[[0.0, 1.0], [0.0, 0.0]]]]
            ),
            field_likelihood=torch.ones(1, 1, 2, 2),
            field_reliability=torch.ones(1, 1, 1, 1),
        )

        self.assertGreater(result.write_weight[0, 0].item(), 0.5)
        self.assertEqual(result.write_weight[0, 1].item(), 0.0)
        self.assertGreater(result.contact_weight[0, 1].item(), 0.0)
        self.assertGreater(result.read_weight[0, 1].item(), 0.0)
        self.assertFalse(bool(result.read_weight[0, 2:].any()))
        self.assertGreaterEqual(
            result.read_weight[0, 0].item(),
            result.write_weight[0, 0].item(),
        )

    def test_motion_owner_recovers_reads_without_widening_writes(self):
        owner = torch.tensor([[0.2, 0.8, 0.0, 0.4]])
        ownership = self._ownership(owner=owner)
        transaction = TransactionalOwnerSupport(
            read_weight=torch.tensor([[0.1, 0.3, 0.5, 0.0]]),
            write_weight=torch.tensor([[0.0, 0.7, 0.0, 0.0]]),
            contact_weight=torch.zeros_like(owner),
            lifecycle_weight=torch.zeros_like(owner),
            missing_observation_frames=torch.zeros(
                1, 1, dtype=torch.long
            ),
        )
        write_before = transaction.write_weight.clone()

        recovered, increment = build_motion_owner_read_weight(
            ownership, transaction
        )

        torch.testing.assert_close(
            recovered, torch.tensor([[0.2, 0.8, 0.5, 0.4]])
        )
        torch.testing.assert_close(
            increment, torch.tensor([[0.1, 0.5, 0.0, 0.4]])
        )
        torch.testing.assert_close(
            transaction.write_weight, write_before, rtol=0, atol=0
        )

    def test_topology_read_fills_only_enclosed_evidence_supported_holes(self):
        shape = (1, 1, 5, 7)
        owner_map = torch.zeros(shape)
        # Closed ring around (2, 2).
        owner_map[0, 0, 1:4, 1] = 1.0
        owner_map[0, 0, 1:4, 3] = 1.0
        owner_map[0, 0, 1, 1:4] = 1.0
        owner_map[0, 0, 3, 1:4] = 1.0
        # Open U around (2, 5): this background remains connected outside.
        owner_map[0, 0, 1:4, 4] = 1.0
        owner_map[0, 0, 1:4, 6] = 1.0
        owner_map[0, 0, 3, 4:7] = 1.0
        owner = owner_map.reshape(1, -1)
        ownership = self._ownership(owner=owner)
        ownership = CausalObjectOwnership(
            **{
                **ownership.__dict__,
                "semantic_support": torch.ones_like(owner),
                "diagnostics": {
                    "motion_hand_affinity": torch.ones_like(owner)
                },
            }
        )
        zeros = torch.zeros_like(owner)
        transaction = TransactionalOwnerSupport(
            read_weight=zeros.clone(), write_weight=zeros.clone(),
            contact_weight=zeros.clone(), lifecycle_weight=zeros.clone(),
            missing_observation_frames=torch.zeros(1, 1, dtype=torch.long),
        )
        write_before = transaction.write_weight.clone()

        recovered, _, topology_increment, holes = (
            build_topology_complete_motion_owner_read_weight(
                ownership, transaction, shape=shape,
                hand_exclusion=torch.zeros(shape),
            )
        )
        hole_map = holes.reshape(shape)
        increment_map = topology_increment.reshape(shape)
        self.assertTrue(bool(hole_map[0, 0, 2, 2]))
        self.assertGreater(increment_map[0, 0, 2, 2].item(), 0.0)
        self.assertFalse(bool(hole_map[0, 0, 2, 5]))
        self.assertEqual(increment_map[0, 0, 2, 5].item(), 0.0)
        self.assertGreater(recovered[0, 2 * 7 + 2].item(), 0.0)
        torch.testing.assert_close(
            transaction.write_weight, write_before, rtol=0, atol=0
        )

    def test_topology_read_uses_hand_only_as_exclusion(self):
        shape = (1, 1, 3, 3)
        owner_map = torch.ones(shape)
        owner_map[..., 1, 1] = 0.0
        owner = owner_map.reshape(1, -1)
        ownership = self._ownership(owner=owner)
        ownership = CausalObjectOwnership(
            **{
                **ownership.__dict__,
                "semantic_support": torch.ones_like(owner),
                "diagnostics": {
                    "motion_hand_affinity": torch.ones_like(owner)
                },
            }
        )
        zeros = torch.zeros_like(owner)
        transaction = TransactionalOwnerSupport(
            read_weight=zeros.clone(), write_weight=zeros.clone(),
            contact_weight=zeros.clone(), lifecycle_weight=zeros.clone(),
            missing_observation_frames=torch.zeros(1, 1, dtype=torch.long),
        )
        hand = torch.zeros(shape)
        hand[..., 1, 1] = 1.0
        recovered, _, topology_increment, holes = (
            build_topology_complete_motion_owner_read_weight(
                ownership, transaction, shape=shape, hand_exclusion=hand
            )
        )
        self.assertTrue(bool(holes.reshape(shape)[..., 1, 1]))
        self.assertEqual(
            topology_increment.reshape(shape)[..., 1, 1].item(), 0.0
        )
        self.assertEqual(recovered.reshape(shape)[..., 1, 1].item(), 0.0)

    def test_flow_contradiction_abstains_from_automatic_write(self):
        tracker = AutomaticTransactionalOwnerTracker()
        owner = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        shape = (1, 1, 2, 2)
        result = tracker(
            ownership=self._ownership(owner=owner),
            object_posterior=owner.reshape(shape),
            posterior_threshold=torch.full((1, 1, 1, 1), 0.2),
            source_attention=owner.reshape(shape),
            hand_probability=torch.zeros(shape),
            hand_proximity=owner.reshape(shape),
            object_role=owner.reshape(shape),
            boundary_role=torch.zeros(shape),
            field_likelihood=torch.zeros(shape),
            field_reliability=torch.ones(1, 1, 1, 1),
        )

        self.assertFalse(bool(result.write_weight.any()))
        self.assertFalse(bool(result.read_weight.any()))

    def test_flow_can_verify_an_automatic_detector_dropout(self):
        tracker = AutomaticTransactionalOwnerTracker()
        owner = torch.tensor([[0.8, 0.0, 0.0, 0.0]])
        shape = (1, 1, 2, 2)
        result = tracker(
            ownership=self._ownership(
                owner=owner, observation=torch.zeros_like(owner)
            ),
            object_posterior=owner.reshape(shape),
            posterior_threshold=torch.full((1, 1, 1, 1), 0.2),
            source_attention=owner.reshape(shape),
            hand_probability=torch.zeros(shape),
            hand_proximity=owner.reshape(shape),
            object_role=owner.reshape(shape),
            boundary_role=torch.zeros(shape),
            field_likelihood=torch.ones(shape),
            field_reliability=torch.ones(1, 1, 1, 1),
        )

        self.assertGreater(result.write_weight[0, 0].item(), 0.5)
        self.assertGreater(result.read_weight[0, 0].item(), 0.5)

    def test_automatic_lifecycle_is_read_only_and_bounded(self):
        tracker = AutomaticTransactionalOwnerTracker(
            max_missing_frames=1
        )
        owner = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        shape = (1, 1, 2, 2)
        common = dict(
            object_posterior=torch.zeros(shape),
            posterior_threshold=torch.full((1, 1, 1, 1), 0.2),
            source_attention=owner.reshape(shape),
            hand_probability=torch.zeros(shape),
            hand_proximity=owner.reshape(shape),
            object_role=torch.zeros(shape),
            boundary_role=torch.zeros(shape),
            field_likelihood=torch.ones(shape),
            field_reliability=torch.ones(1, 1, 1, 1),
        )
        missing_owner = self._ownership(
            owner=torch.zeros_like(owner),
            observation=torch.zeros_like(owner),
            transported=owner,
            match_confidence=owner,
            state=int(CausalOwnershipState.OCCLUDED),
            missing=1,
        )
        first = tracker(ownership=missing_owner, **common)
        second = tracker(ownership=missing_owner, **common)

        self.assertGreater(first.read_weight[0, 0].item(), 0.9)
        self.assertFalse(bool(first.write_weight.any()))
        self.assertFalse(bool(second.read_weight.any()))

    def test_automatic_lifecycle_budget_advances_once_per_block(self):
        tracker = AutomaticTransactionalOwnerTracker(
            max_missing_frames=1, blockwise_lifecycle=True
        )
        tokens = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(1, 3)
        shape = (1, 3, 2, 2)
        ownership = CausalObjectOwnership(
            owner_weight=torch.zeros_like(tokens),
            owner_support=torch.zeros_like(tokens, dtype=torch.bool),
            transported_weight=tokens,
            observation_weight=torch.zeros_like(tokens),
            match_similarity=tokens.mul(2.0).sub(1.0),
            match_confidence=tokens,
            semantic_support=tokens,
            state_code=torch.full(
                (1, 3), int(CausalOwnershipState.OCCLUDED), dtype=torch.long
            ),
            missing_frames=torch.ones(1, 3, dtype=torch.long),
        )
        common = dict(
            ownership=ownership,
            object_posterior=torch.zeros(shape),
            posterior_threshold=torch.full((1, 3, 1, 1), 0.2),
            source_attention=tokens.reshape(shape),
            hand_probability=torch.zeros(shape),
            hand_proximity=tokens.reshape(shape),
            object_role=torch.zeros(shape),
            boundary_role=torch.zeros(shape),
            field_likelihood=torch.ones(shape),
            field_reliability=torch.ones(1, 3, 1, 1),
        )
        first = tracker(**common)
        second = tracker(**common)
        self.assertTrue(bool(first.read_weight.reshape(shape)[:, :, 0, 0].all()))
        self.assertEqual(
            first.missing_observation_frames.unique().tolist(), [1]
        )
        self.assertFalse(bool(second.read_weight.any()))
        self.assertEqual(
            second.missing_observation_frames.unique().tolist(), [2]
        )

    def test_oracle_owner_is_exact_and_empty_frame_is_absent(self):
        source_owner = torch.zeros(1, 2, 4, 4, dtype=torch.bool)
        source_owner[0, 0, :2, :2] = True
        hand = torch.zeros_like(source_owner)
        hand[0, 0, 0, 0] = True

        result = build_oracle_causal_ownership(
            source_owner_mask=source_owner,
            hand_mask=hand,
            spatial_shape=(2, 2),
        )

        self.assertEqual(result.owner_weight[0, 0].item(), 0.75)
        self.assertFalse(result.owner_support[0, 4:].any())
        self.assertEqual(
            result.state_code[0, 0].item(),
            int(CausalOwnershipState.VISIBLE),
        )
        self.assertEqual(
            result.state_code[0, 1].item(),
            int(CausalOwnershipState.ABSENT),
        )
        self.assertEqual(result.missing_frames[0, 1].item(), 1)

    def _track(
        self,
        tracker: CausalObjectOwnershipTracker,
        object_index: int,
        observation: torch.Tensor,
        *,
        visible: bool,
        proximity: torch.Tensor | None = None,
    ):
        semantic = torch.zeros(1, 4)
        semantic[0, object_index] = 1.0
        if proximity is None:
            proximity = semantic.clone()
        return tracker(
            source_features=_features(object_index),
            observation_weight=observation,
            source_semantic=semantic,
            hand_mask=torch.zeros(1, 4, dtype=torch.bool),
            hand_proximity=proximity,
            tokens_per_frame=4,
            detector_visible=torch.tensor([[visible]]),
            spatial_shape=(2, 2),
        )

    def test_clean_source_transport_survives_detector_dropout(self):
        tracker = CausalObjectOwnershipTracker(
            max_area_fraction=0.5,
            min_owner_weight=0.01,
        )
        self._track(
            tracker,
            0,
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            visible=True,
        )
        moved = self._track(
            tracker,
            1,
            torch.zeros(1, 4),
            visible=False,
        )

        self.assertGreater(moved.owner_weight[0, 1].item(), 0.9)
        self.assertEqual(moved.owner_weight[0, 0].item(), 0.0)
        self.assertEqual(
            moved.state_code.item(),
            int(CausalOwnershipState.VISIBLE),
        )

    def test_hand_proximity_recovers_complete_semantic_dropout(self):
        tracker = CausalObjectOwnershipTracker(
            max_area_fraction=0.5,
            min_owner_weight=0.01,
        )
        self._track(
            tracker,
            0,
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            visible=True,
        )
        proximity = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        moved = tracker(
            source_features=_features(1),
            observation_weight=torch.zeros(1, 4),
            source_semantic=torch.zeros(1, 4),
            hand_mask=torch.zeros(1, 4, dtype=torch.bool),
            hand_proximity=proximity,
            tokens_per_frame=4,
            detector_visible=torch.tensor([[False]]),
            spatial_shape=(2, 2),
        )

        self.assertGreater(moved.owner_weight[0, 1].item(), 0.9)
        self.assertEqual(
            moved.state_code.item(),
            int(CausalOwnershipState.VISIBLE),
        )

    def test_later_frame_can_ignite_after_empty_first_frame(self):
        tracker = CausalObjectOwnershipTracker(
            max_area_fraction=0.5,
            min_owner_weight=0.01,
        )
        features = torch.cat([_features(0), _features(1)], dim=1)
        observation = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
        )
        semantic = observation.clone()
        result = tracker(
            source_features=features,
            observation_weight=observation,
            source_semantic=semantic,
            hand_mask=torch.zeros(1, 8, dtype=torch.bool),
            hand_proximity=semantic,
            tokens_per_frame=4,
            detector_visible=torch.tensor([[False, True]]),
            spatial_shape=(2, 2),
        )

        self.assertFalse(result.owner_support[0, :4].any())
        self.assertGreater(result.owner_weight[0, 5].item(), 0.9)
        self.assertEqual(
            result.state_code[0, 1].item(),
            int(CausalOwnershipState.VISIBLE),
        )

    def test_absence_emits_no_owner_but_keeps_reassociation_state(self):
        tracker = CausalObjectOwnershipTracker(
            max_area_fraction=0.5,
            min_owner_weight=0.01,
            max_occluded_frames=1,
        )
        self._track(
            tracker,
            0,
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            visible=True,
        )
        absent_features = torch.tensor(
            [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]],
            dtype=torch.float32,
        )
        kwargs = dict(
            source_features=absent_features,
            observation_weight=torch.zeros(1, 4),
            source_semantic=torch.zeros(1, 4),
            hand_mask=torch.zeros(1, 4, dtype=torch.bool),
            hand_proximity=torch.zeros(1, 4),
            tokens_per_frame=4,
            detector_visible=torch.tensor([[False]]),
            spatial_shape=(2, 2),
        )
        occluded = tracker(**kwargs)
        absent = tracker(**kwargs)
        reappeared = self._track(
            tracker,
            2,
            torch.zeros(1, 4),
            visible=False,
        )

        self.assertFalse(occluded.owner_support.any())
        self.assertEqual(
            occluded.state_code.item(),
            int(CausalOwnershipState.OCCLUDED),
        )
        self.assertFalse(absent.owner_support.any())
        self.assertEqual(
            absent.state_code.item(),
            int(CausalOwnershipState.ABSENT),
        )
        self.assertGreater(reappeared.owner_weight[0, 2].item(), 0.9)

    def test_preview_does_not_advance_causal_owner_state(self):
        tracker = CausalObjectOwnershipTracker(
            max_area_fraction=0.5,
            min_owner_weight=0.01,
        )
        result = tracker(
            source_features=_features(0),
            observation_weight=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]]
            ),
            source_semantic=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]]
            ),
            hand_mask=torch.zeros(1, 4, dtype=torch.bool),
            hand_proximity=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]]
            ),
            tokens_per_frame=4,
            detector_visible=torch.tensor([[True]]),
            spatial_shape=(2, 2),
            update_state=False,
        )

        self.assertGreater(result.owner_weight[0, 0].item(), 0.0)
        self.assertIsNone(tracker.transport.previous_features)
        self.assertIsNone(tracker.transport.previous_weight)
        self.assertIsNone(tracker._missing_count)

    def test_verified_commit_advances_only_supplied_core(self):
        tracker = CausalObjectOwnershipTracker(
            max_area_fraction=0.5,
            min_owner_weight=0.01,
        )
        tracker.commit_verified(
            source_features=_features(0),
            verified_weight=torch.tensor(
                [[0.8, 0.0, 0.0, 0.0]]
            ),
            tokens_per_frame=4,
        )

        self.assertIsNotNone(tracker.transport.previous_weight)
        self.assertGreater(
            tracker.transport.previous_weight[0, 0].item(), 0.0
        )
        self.assertFalse(
            bool(tracker.transport.previous_weight[0, 1:].any())
        )

    def test_causal_owner_overrides_background_provenance(self):
        shape = (1, 1, 2, 2)
        zero = torch.zeros(shape)
        roles = RoleState(
            object=zero.clone(),
            boundary=zero.clone(),
            hand=zero.clone(),
            background=torch.ones(shape),
        )
        evidence = {
            "object_posterior": zero.clone(),
            "posterior_threshold": torch.full(shape, 0.2),
            "source_attention": zero.clone(),
            "hand_proximity": zero.clone(),
            "adaptive_attention_reliability": torch.ones(shape),
            "object_visible": zero.clone(),
            "temporal_confidence": zero.clone(),
            "causal_owner_weight": torch.ones(shape),
        }
        operators = FactorizedBayesOperatorBuilder()(
            roles=roles, evidence=evidence, expected_token_length=1
        )

        torch.testing.assert_close(
            operators.source_value_action, torch.zeros(1, 1)
        )
        torch.testing.assert_close(
            operators.source_residual_action, torch.zeros(1, 1)
        )
        torch.testing.assert_close(
            operators.unknown_action, torch.zeros(1, 1)
        )
        torch.testing.assert_close(
            operators.target_memory_action, torch.ones(1, 1)
        )
        torch.testing.assert_close(
            operators.source_key_action, torch.ones(1, 1)
        )

    def test_soft_owner_transfer_conserves_probability_mass(self):
        shape = (1, 1, 2, 2)
        roles = RoleState(
            object=torch.full(shape, 0.13),
            boundary=torch.zeros(shape),
            hand=torch.zeros(shape),
            background=torch.full(shape, 0.87),
        )
        evidence = {
            "object_posterior": torch.full(shape, 0.10),
            "posterior_threshold": torch.full(shape, 0.20),
            "source_attention": torch.full(shape, 0.50),
            "hand_proximity": torch.zeros(shape),
            "adaptive_attention_reliability": torch.full(
                shape, 0.55
            ),
            "object_visible": torch.ones(shape),
            "temporal_confidence": torch.zeros(shape),
            "causal_owner_weight": torch.full(shape, 0.54),
            "causal_owner_support": torch.ones(shape),
        }
        operators = FactorizedBayesOperatorBuilder()(
            roles=roles, evidence=evidence, expected_token_length=1
        )
        total = (
            operators.roles.object
            + operators.roles.boundary
            + operators.roles.hand
            + operators.roles.background
            + operators.roles.unknown
        )

        torch.testing.assert_close(total, torch.ones_like(total))
        provenance = (
            operators.source_value_action
            + operators.target_memory_action
            + operators.unknown_action
        )
        torch.testing.assert_close(
            provenance, torch.ones_like(provenance)
        )

    def test_hand_remains_source_owned_under_spurious_owner(self):
        shape = (1, 1, 2, 2)
        zero = torch.zeros(shape)
        roles = RoleState(
            object=zero.clone(),
            boundary=zero.clone(),
            hand=torch.ones(shape),
            background=zero.clone(),
        )
        evidence = {
            "object_posterior": zero.clone(),
            "posterior_threshold": torch.full(shape, 0.2),
            "source_attention": zero.clone(),
            "hand_proximity": torch.ones(shape),
            "adaptive_attention_reliability": torch.ones(shape),
            "object_visible": zero.clone(),
            "temporal_confidence": zero.clone(),
            "causal_owner_weight": torch.ones(shape),
        }
        operators = FactorizedBayesOperatorBuilder()(
            roles=roles, evidence=evidence, expected_token_length=1
        )

        torch.testing.assert_close(
            operators.source_value_action, torch.ones(1, 1)
        )
        torch.testing.assert_close(
            operators.target_memory_action, torch.zeros(1, 1)
        )

    def test_transactional_owner_contact_is_read_only(self):
        tracker = CausalReadOnlyOwnerTracker(
            max_area_fraction=1.0,
            min_similarity=0.5,
            max_missing_frames=1,
        )
        full = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        core = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        result = tracker(
            source_features=_features(0),
            full_owner_weight=full,
            core_owner_weight=core,
            source_semantic=full,
            hand_proximity=full,
            tokens_per_frame=4,
            spatial_shape=(2, 2),
        )

        torch.testing.assert_close(result.read_weight, full)
        torch.testing.assert_close(result.write_weight, core)
        torch.testing.assert_close(
            result.contact_weight,
            torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        )
        self.assertFalse(bool(result.lifecycle_weight.any()))

    def test_transactional_owner_lifecycle_is_bounded_and_read_only(self):
        tracker = CausalReadOnlyOwnerTracker(
            max_area_fraction=1.0,
            min_similarity=0.5,
            max_missing_frames=1,
        )
        full = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        tracker(
            source_features=_features(0),
            full_owner_weight=full,
            core_owner_weight=full,
            source_semantic=full,
            hand_proximity=full,
            tokens_per_frame=4,
            spatial_shape=(2, 2),
        )
        missing = torch.zeros_like(full)
        first_missing = tracker(
            source_features=_features(1),
            full_owner_weight=missing,
            core_owner_weight=missing,
            source_semantic=missing,
            hand_proximity=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            tokens_per_frame=4,
            spatial_shape=(2, 2),
        )
        second_missing = tracker(
            source_features=_features(1),
            full_owner_weight=missing,
            core_owner_weight=missing,
            source_semantic=missing,
            hand_proximity=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            tokens_per_frame=4,
            spatial_shape=(2, 2),
        )

        self.assertGreater(first_missing.read_weight[0, 1].item(), 0.9)
        self.assertFalse(bool(first_missing.write_weight.any()))
        self.assertFalse(bool(second_missing.read_weight.any()))


if __name__ == "__main__":
    unittest.main()
