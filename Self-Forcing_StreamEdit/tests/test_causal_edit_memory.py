import unittest
import importlib.util
from pathlib import Path

import torch

from tests._pipeline_imports import load_pipeline_module


causal_edit_memory = load_pipeline_module("causal_edit_memory")
CausalPairedEditMemory = causal_edit_memory.CausalPairedEditMemory
PairedEditMemoryState = causal_edit_memory.PairedEditMemoryState
build_object_coordinates = causal_edit_memory.build_object_coordinates
build_object_interior_gate = (
    causal_edit_memory.build_object_interior_gate
)
build_owner_attached_structure_gate = (
    causal_edit_memory.build_owner_attached_structure_gate
)
build_source_part_signature = (
    causal_edit_memory.build_source_part_signature
)
source_addressed_residual_read = (
    causal_edit_memory.source_addressed_residual_read
)
source_transport_frontier = (
    causal_edit_memory.source_transport_frontier
)


def load_attention_module():
    path = Path(__file__).resolve().parents[1] / "wan/modules/attention.py"
    spec = importlib.util.spec_from_file_location(
        "streamedit_paired_memory_attention", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


blend_source_addressed_residual = (
    load_attention_module().blend_source_addressed_residual
)
arbitrate_projected_attention_output = (
    load_attention_module().arbitrate_projected_attention_output
)
project_source_addressed_target_value = (
    load_attention_module().project_source_addressed_target_value
)
scatter_source_addressed_anchor_delta = (
    load_attention_module().scatter_source_addressed_anchor_delta
)
source_addressed_anchor_attention_delta = (
    load_attention_module().source_addressed_anchor_attention_delta
)
immutable_canonical_anchor_attention_delta = (
    load_attention_module().immutable_canonical_anchor_attention_delta
)


def make_cache(key, value):
    length = key.shape[1]
    return [{
        "k": key.clone(),
        "v": value.clone(),
        "local_end_index": torch.tensor([length]),
        "global_end_index": torch.tensor([length]),
        "num_new_tokens": length,
        "sink_tokens": 0,
    }]


class CausalEditMemoryTest(unittest.TestCase):
    @staticmethod
    def make_transported_memory(**overrides):
        config = dict(
            layers=(0,),
            max_tokens=8,
            max_tokens_per_block=2,
            min_commit_confidence=0.05,
            # Keep the canonical matcher deliberately stricter than the
            # adjacent clean-source transport matcher.
            min_similarity=0.90,
            coordinate_bias=0.0,
            coordinate_radius=0.25,
            topk=1,
            source_transport=True,
            transport_min_similarity=0.50,
            transport_coordinate_radius=1.0,
            transport_cycle_radius=0.25,
            transport_min_confidence=0.05,
        )
        config.update(overrides)
        return CausalPairedEditMemory(**config)

    @staticmethod
    def commit_ignition(memory, key, residual, coordinate=None):
        if coordinate is None:
            coordinate = torch.zeros(*key.shape[:2], 2)
        source_value = torch.zeros_like(key)
        return memory.commit(
            source_kv_cache=make_cache(key, source_value),
            target_kv_cache=make_cache(key, source_value + residual),
            proposal_weight=torch.ones(key.shape[:2]),
            object_coordinate=coordinate,
            transactional=False,
        )

    def test_source_transport_recovers_when_canonical_address_fails(self):
        memory = self.make_transported_memory(max_tokens_per_block=1)
        ignition_key = torch.tensor([[[[1.0, 0.0]]]])
        canonical_residual = torch.tensor([[[[2.0, -1.0]]]])
        self.commit_ignition(memory, ignition_key, canonical_residual)

        # Cosine 0.8 is below the canonical threshold but above the
        # adjacent source-transport threshold.
        next_key = torch.tensor([[[[0.8, 0.6]]]])
        read = memory.read(
            source_kv_cache=make_cache(next_key, torch.zeros_like(next_key)),
            current_coordinate=torch.zeros(1, 1, 2),
            current_object_request=torch.ones(1, 1),
            current_transport_owner=torch.ones(1, 1),
        )[0]

        self.assertEqual(read.canonical_support.item(), 0.0)
        self.assertGreater(read.transported_support.item(), 0.0)
        self.assertGreater(read.support.item(), 0.0)
        torch.testing.assert_close(read.residual, canonical_residual)

    def test_two_hop_transport_preserves_payload_and_lineage(self):
        memory = self.make_transported_memory(max_tokens_per_block=1)
        ignition_key = torch.tensor([[[[1.0, 0.0]]]])
        canonical_residual = torch.tensor([[[[3.0, 4.0]]]])
        self.commit_ignition(memory, ignition_key, canonical_residual)

        first_hop = memory.read(
            source_kv_cache=make_cache(
                torch.tensor([[[[0.8, 0.6]]]]),
                torch.zeros(1, 1, 1, 2),
            ),
            current_coordinate=torch.zeros(1, 1, 2),
            current_object_request=torch.ones(1, 1),
        )[0]
        second_hop = memory.read(
            source_kv_cache=make_cache(
                torch.tensor([[[[0.0, 1.0]]]]),
                torch.zeros(1, 1, 1, 2),
            ),
            current_coordinate=torch.zeros(1, 1, 2),
            current_object_request=torch.ones(1, 1),
        )[0]

        self.assertGreater(first_hop.transported_support.item(), 0.0)
        self.assertGreater(second_hop.transported_support.item(), 0.0)
        self.assertEqual(first_hop.lineage_id.item(), 0)
        self.assertEqual(second_hop.lineage_id.item(), 0)
        torch.testing.assert_close(first_hop.residual, canonical_residual)
        torch.testing.assert_close(second_hop.residual, canonical_residual)

    def test_conflicting_target_observation_cannot_rewrite_lineage(self):
        memory = self.make_transported_memory(max_tokens_per_block=1)
        key = torch.tensor([[[[1.0, 0.0]]]])
        canonical_residual = torch.tensor([[[[2.0, 0.0]]]])
        self.commit_ignition(memory, key, canonical_residual)

        # First-block replay verifies the ignition lineage.
        memory.read(
            source_kv_cache=make_cache(key, torch.zeros_like(key)),
            current_coordinate=torch.zeros(1, 1, 2),
            current_object_request=torch.ones(1, 1),
        )
        verified = memory.commit(
            source_kv_cache=make_cache(key, torch.zeros_like(key)),
            target_kv_cache=make_cache(key, canonical_residual),
            proposal_weight=torch.ones(1, 1),
            object_coordinate=torch.zeros(1, 1, 2),
            transactional=True,
        )
        self.assertEqual(verified["accepted"].item(), 1.0)

        moved_key = torch.tensor([[[[0.8, 0.6]]]])
        memory.read(
            source_kv_cache=make_cache(
                moved_key, torch.zeros_like(moved_key)
            ),
            current_coordinate=torch.zeros(1, 1, 2),
            current_object_request=torch.ones(1, 1),
        )
        rejected = memory.commit(
            source_kv_cache=make_cache(
                moved_key, torch.zeros_like(moved_key)
            ),
            target_kv_cache=make_cache(
                moved_key, -canonical_residual
            ),
            proposal_weight=torch.ones(1, 1),
            object_coordinate=torch.zeros(1, 1, 2),
            transactional=True,
        )
        self.assertEqual(rejected["accepted"].item(), 0.0)

        state = memory.export()[0]
        valid = state.evidence[0] > 0
        self.assertTrue(bool(valid.any()))
        torch.testing.assert_close(
            state.target_value_residual[0, valid],
            canonical_residual[0].expand(int(valid.sum()), -1, -1),
        )
        frontier = memory.export_frontier()[0]
        frontier_valid = frontier.evidence[0] > 0
        torch.testing.assert_close(
            frontier.target_value_residual[0, frontier_valid],
            canonical_residual[0].expand(
                int(frontier_valid.sum()), -1, -1
            ),
        )

    def test_source_transport_cycle_and_owner_abstain_exactly(self):
        previous = PairedEditMemoryState(
            source_key=torch.tensor([[[[1.0, 0.0]]]]),
            target_value_residual=torch.tensor([[[[5.0, 1.0]]]]),
            object_coordinate=torch.zeros(1, 1, 2),
            evidence=torch.ones(1, 1),
            lineage_id=torch.tensor([[7]]),
        )
        current_key = torch.tensor(
            [[[[1.0, 0.0]], [[1.0, 0.0]]]]
        )
        coordinate = torch.tensor([[[0.0, 0.0], [0.8, 0.0]]])
        result = source_transport_frontier(
            current_source_key=current_key,
            current_coordinate=coordinate,
            current_owner=torch.ones(1, 2),
            previous_frontier=previous,
            max_frontier_tokens=2,
            min_similarity=0.0,
            coordinate_bias=0.0,
            coordinate_radius=1.0,
            cycle_radius=0.2,
            min_confidence=0.01,
        )
        self.assertGreater(result.read.support[0, 0].item(), 0.0)
        self.assertEqual(result.read.support[0, 1].item(), 0.0)
        torch.testing.assert_close(
            result.read.residual[0, 1],
            torch.zeros_like(result.read.residual[0, 1]),
            rtol=0.0,
            atol=0.0,
        )

        absent = source_transport_frontier(
            current_source_key=current_key,
            current_coordinate=coordinate,
            current_owner=torch.zeros(1, 2),
            previous_frontier=previous,
            max_frontier_tokens=2,
        )
        self.assertIsNone(absent.frontier)
        torch.testing.assert_close(
            absent.read.support,
            torch.zeros_like(absent.read.support),
            rtol=0.0,
            atol=0.0,
        )

    def test_single_confidence_transport_commit_does_not_square_request(self):
        key = torch.tensor([[[[1.0, 0.0]]]])
        residual = torch.tensor([[[[2.0, -1.0]]]])
        coordinate = torch.zeros(1, 1, 2)
        memory = self.make_transported_memory(
            min_commit_confidence=0.30,
            min_similarity=0.0,
            single_confidence=True,
        )
        self.commit_ignition(memory, key, residual, coordinate)
        request = torch.tensor([[0.5]])
        read = memory.read(
            source_kv_cache=make_cache(key, torch.zeros_like(key)),
            current_coordinate=coordinate,
            current_object_request=request,
            current_transport_owner=request,
        )[0]
        self.assertGreaterEqual(read.support.item(), 0.49)
        committed = memory.commit(
            source_kv_cache=make_cache(key, torch.zeros_like(key)),
            target_kv_cache=make_cache(key, residual),
            proposal_weight=request,
            object_coordinate=coordinate,
            transactional=True,
        )
        # A second proposal multiplication would reduce 0.5 to 0.25 and
        # fail the 0.30 transactional threshold.
        self.assertEqual(committed["accepted"].item(), 1.0)
        self.assertGreaterEqual(committed["write"].item(), 0.49)

    def test_first_replay_keeps_only_verified_ignition_lineage(self):
        memory = self.make_transported_memory(
            max_tokens=8, max_tokens_per_block=2
        )
        key = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        residual = torch.tensor(
            [[[[2.0, 0.0]], [[0.0, 3.0]]]]
        )
        self.commit_ignition(memory, key, residual)

        request = torch.tensor([[1.0, 0.0]])
        memory.read(
            source_kv_cache=make_cache(key, torch.zeros_like(key)),
            current_coordinate=torch.zeros(1, 2, 2),
            current_object_request=request,
            current_transport_owner=torch.ones(1, 2),
        )
        memory.commit(
            source_kv_cache=make_cache(key, torch.zeros_like(key)),
            target_kv_cache=make_cache(key, residual),
            proposal_weight=request,
            object_coordinate=torch.zeros(1, 2, 2),
            transactional=True,
        )

        state = memory.export()[0]
        valid = state.evidence[0] > 0
        self.assertTrue(bool(valid.any()))
        self.assertEqual(
            set(state.lineage_id[0, valid].tolist()), {0}
        )
        frontier = memory.export_frontier()[0]
        frontier_valid = frontier.evidence[0] > 0
        self.assertEqual(
            set(frontier.lineage_id[0, frontier_valid].tolist()), {0}
        )

    def test_rejected_replay_cannot_reseed_payload_later(self):
        memory = self.make_transported_memory(max_tokens_per_block=1)
        key = torch.tensor([[[[1.0, 0.0]]]])
        ignition_residual = torch.tensor([[[[2.0, 0.0]]]])
        self.commit_ignition(memory, key, ignition_residual)
        memory.read(
            source_kv_cache=make_cache(key, torch.zeros_like(key)),
            current_coordinate=torch.zeros(1, 1, 2),
            current_object_request=torch.ones(1, 1),
        )
        memory.commit(
            source_kv_cache=make_cache(key, torch.zeros_like(key)),
            target_kv_cache=make_cache(key, -ignition_residual),
            proposal_weight=torch.ones(1, 1),
            object_coordinate=torch.zeros(1, 1, 2),
            transactional=True,
        )
        self.assertEqual(memory.export()[0].evidence.sum().item(), 0.0)
        # Payload initialization is irreversible: an empty verified bank is
        # an abstaining state, not permission to learn from a later target.
        self.assertTrue(memory.has_state())

        memory.read(
            source_kv_cache=make_cache(key, torch.zeros_like(key)),
            current_coordinate=torch.zeros(1, 1, 2),
            current_object_request=torch.ones(1, 1),
        )
        later = memory.commit(
            source_kv_cache=make_cache(key, torch.zeros_like(key)),
            target_kv_cache=make_cache(
                key, torch.tensor([[[[99.0, 0.0]]]])
            ),
            proposal_weight=torch.ones(1, 1),
            object_coordinate=torch.zeros(1, 1, 2),
            transactional=memory.has_state(),
        )
        self.assertEqual(later["write"].item(), 0.0)
        self.assertEqual(memory.export()[0].evidence.sum().item(), 0.0)

    def test_broken_frontier_cannot_bypass_strict_canonical_gate(self):
        memory = self.make_transported_memory(max_tokens_per_block=1)
        ignition_key = torch.tensor([[[[1.0, 0.0]]]])
        residual = torch.tensor([[[[2.0, 0.0]]]])
        self.commit_ignition(memory, ignition_key, residual)

        # No source-only match: this explicitly terminates the adjacent
        # transport chain.
        broken_key = torch.tensor([[[[-1.0, 0.0]]]])
        broken = memory.read(
            source_kv_cache=make_cache(
                broken_key, torch.zeros_like(broken_key)
            ),
            current_coordinate=torch.zeros(1, 1, 2),
            current_object_request=torch.ones(1, 1),
        )[0]
        self.assertEqual(broken.support.item(), 0.0)
        self.assertEqual(len(memory.export_frontier()), 0)

        # This key passes the looser transport threshold but fails the strict
        # canonical threshold. It must not restart from the canonical bank.
        ambiguous_key = torch.tensor([[[[0.8, 0.6]]]])
        ambiguous = memory.read(
            source_kv_cache=make_cache(
                ambiguous_key, torch.zeros_like(ambiguous_key)
            ),
            current_coordinate=torch.zeros(1, 1, 2),
            current_object_request=torch.ones(1, 1),
        )[0]
        self.assertEqual(ambiguous.canonical_support.item(), 0.0)
        self.assertEqual(ambiguous.transported_support.item(), 0.0)
        self.assertEqual(ambiguous.support.item(), 0.0)

        # A strict canonical re-association plus an agreeing target view may
        # transactionally restart the moving frontier.
        reassociated = memory.read(
            source_kv_cache=make_cache(
                ignition_key, torch.zeros_like(ignition_key)
            ),
            current_coordinate=torch.zeros(1, 1, 2),
            current_object_request=torch.ones(1, 1),
        )[0]
        self.assertGreater(reassociated.canonical_support.item(), 0.0)
        memory.commit(
            source_kv_cache=make_cache(
                ignition_key, torch.zeros_like(ignition_key)
            ),
            target_kv_cache=make_cache(ignition_key, residual),
            proposal_weight=torch.ones(1, 1),
            object_coordinate=torch.zeros(1, 1, 2),
            transactional=True,
        )
        self.assertIn(0, memory.export_frontier())
        resumed = memory.read(
            source_kv_cache=make_cache(
                ambiguous_key, torch.zeros_like(ambiguous_key)
            ),
            current_coordinate=torch.zeros(1, 1, 2),
            current_object_request=torch.ones(1, 1),
        )[0]
        self.assertGreater(resumed.transported_support.item(), 0.0)
        torch.testing.assert_close(resumed.residual, residual)

    def test_query_arbitration_is_exact_outside_read_support(self):
        native = torch.randn(1, 3, 2, 4)
        projected = native + 7.0
        corrected = arbitrate_projected_attention_output(
            native, projected, torch.tensor([[1.0, 0.0, 0.25]])
        )
        torch.testing.assert_close(corrected[:, 0], projected[:, 0])
        torch.testing.assert_close(
            corrected[:, 1], native[:, 1], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            corrected[:, 2], native[:, 2] + 1.75
        )

    def test_query_arbitration_zero_support_is_exact_native(self):
        native = torch.randn(1, 3, 2, 4)
        corrected = arbitrate_projected_attention_output(
            native, torch.randn_like(native), torch.zeros(1, 3)
        )
        torch.testing.assert_close(
            corrected, native, rtol=0.0, atol=0.0
        )

    def test_binary_query_access_does_not_square_value_confidence(self):
        native_value = torch.zeros(1, 2, 1, 1)
        source_value = torch.zeros_like(native_value)
        residual = torch.full_like(native_value, 8.0)
        support = torch.tensor([[0.25, 0.0]])
        projected_value = project_source_addressed_target_value(
            native_value,
            source_value,
            residual,
            support,
            strength=0.75,
        )
        # Identity attention makes the composed confidence semantics
        # explicit: V projection contributes 0.75 * support exactly once.
        corrected = arbitrate_projected_attention_output(
            native_value,
            projected_value,
            support,
            binary_access=True,
        )
        torch.testing.assert_close(
            corrected[:, 0], torch.tensor([[[1.5]]])
        )
        torch.testing.assert_close(
            corrected[:, 1], native_value[:, 1], rtol=0.0, atol=0.0
        )

    def test_zero_support_is_exact_native_fallback(self):
        native = torch.randn(1, 3, 2, 4)
        residual = torch.randn_like(native)
        corrected = blend_source_addressed_residual(
            native, residual, torch.zeros(1, 3), strength=0.35
        )
        torch.testing.assert_close(
            corrected, native, rtol=0.0, atol=0.0
        )

    def test_residual_changes_only_supported_tokens(self):
        native = torch.zeros(1, 2, 1, 2)
        residual = torch.ones_like(native)
        corrected = blend_source_addressed_residual(
            native, residual, torch.tensor([[1.0, 0.0]]), strength=0.5
        )
        torch.testing.assert_close(
            corrected[:, 0], torch.full((1, 1, 2), 0.5)
        )
        torch.testing.assert_close(
            corrected[:, 1], native[:, 1], rtol=0.0, atol=0.0
        )

    def test_value_projection_materializes_source_plus_residual(self):
        target = torch.tensor([[[[9.0, 9.0]], [[7.0, 7.0]]]])
        source = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
        residual = torch.tensor([[[[5.0, 6.0]], [[8.0, 9.0]]]])
        projected = project_source_addressed_target_value(
            target,
            source,
            residual,
            torch.tensor([[1.0, 0.0]]),
            strength=1.0,
        )
        torch.testing.assert_close(
            projected[:, 0], torch.tensor([[[6.0, 8.0]]])
        )
        torch.testing.assert_close(
            projected[:, 1], target[:, 1], rtol=0.0, atol=0.0
        )

    def test_value_projection_zero_support_is_exact_fallback(self):
        target = torch.randn(1, 3, 2, 4)
        projected = project_source_addressed_target_value(
            target,
            torch.randn_like(target),
            torch.randn_like(target),
            torch.zeros(1, 3),
            strength=0.75,
        )
        torch.testing.assert_close(
            projected, target, rtol=0.0, atol=0.0
        )

    def test_object_coordinates_follow_each_frame_box(self):
        owner = torch.tensor([[
            0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ]])
        coordinate = build_object_coordinates(
            owner, tokens_per_frame=16, spatial_shape=(4, 4)
        )
        self.assertEqual(tuple(coordinate.shape), (1, 16, 2))
        torch.testing.assert_close(
            coordinate[0, 5], torch.tensor([-0.25, -0.25])
        )
        torch.testing.assert_close(
            coordinate[0, 10], torch.tensor([0.25, 0.25])
        )

    def test_interior_gate_protects_boundary_and_hand_tokens(self):
        owner = torch.ones(1, 9)
        object_role = torch.zeros(1, 1, 6, 6)
        object_role[:, :, 2:4, 2:4] = 1.0
        boundary_role = torch.ones_like(object_role) - object_role
        hand_role = torch.zeros_like(object_role)
        gate = build_object_interior_gate(
            owner,
            object_role=object_role,
            boundary_role=boundary_role,
            hand_role=hand_role,
            tokens_per_frame=9,
            spatial_shape=(3, 3),
            neighborhood_radius=0,
        ).reshape(1, 3, 3)
        self.assertGreater(gate[0, 1, 1].item(), 0.99)
        self.assertLess(gate[0, 0, 0].item(), 0.01)

    def test_owner_attached_structure_gate_admits_boundary_not_hand(self):
        owner = torch.tensor([[
            0.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
            0.0, 0.0, 0.0,
        ]])
        object_role = torch.zeros(1, 1, 3, 3)
        boundary_role = torch.zeros_like(object_role)
        hand_role = torch.zeros_like(object_role)
        boundary_role[0, 0, 1, 1] = 0.8
        hand_role[0, 0, 1, 1] = 0.2
        boundary_role[0, 0, 1, 2] = 0.4
        hand_role[0, 0, 1, 2] = 0.6
        gate = build_owner_attached_structure_gate(
            owner,
            object_role=object_role,
            boundary_role=boundary_role,
            hand_role=hand_role,
            tokens_per_frame=9,
            spatial_shape=(3, 3),
            neighborhood_radius=0,
        ).reshape(1, 3, 3)
        self.assertGreater(gate[0, 1, 1].item(), 0.79)
        # A tracker-owner tail without an object/boundary role must not turn
        # background into a paired-memory query.
        self.assertEqual(gate[0, 1, 0].item(), 0.0)
        self.assertEqual(gate[0, 1, 2].item(), 0.0)
        self.assertEqual(gate[0, 0, 0].item(), 0.0)

    def test_source_part_signature_is_scale_and_offset_invariant(self):
        source = torch.tensor([[[[1.0, 2.0], [3.0, 5.0]]]])
        transformed = source * 3.0 + 7.0
        signature = build_source_part_signature(source)
        transformed_signature = build_source_part_signature(transformed)
        torch.testing.assert_close(signature, transformed_signature)
        self.assertEqual(tuple(signature.shape), (1, 1, 4))

    def test_source_part_consistency_blocks_cross_part_residual(self):
        memory = PairedEditMemoryState(
            # Correspondence keys and coordinates are intentionally
            # ambiguous: only clean-source part appearance separates slots.
            source_key=torch.tensor(
                [[[[1.0, 0.0]], [[1.0, 0.0]]]]
            ),
            target_value_residual=torch.tensor(
                [[[[9.0, 0.0]], [[0.0, 2.0]]]]
            ),
            object_coordinate=torch.zeros(1, 2, 2),
            evidence=torch.ones(1, 2),
            source_part_signature=torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]]
            ),
        )
        result = source_addressed_residual_read(
            current_source_key=torch.tensor([[[[1.0, 0.0]]]]),
            current_coordinate=torch.zeros(1, 1, 2),
            current_owner=torch.ones(1, 1),
            memory=memory,
            current_source_part_signature=torch.tensor(
                [[[0.0, 1.0]]]
            ),
            topk=2,
            min_similarity=0.0,
            coordinate_bias=0.0,
            source_part_consistency=True,
            min_part_similarity=0.45,
            part_similarity_margin=0.08,
        )
        torch.testing.assert_close(
            result.residual, torch.tensor([[[[0.0, 2.0]]]])
        )
        self.assertGreater(result.support.item(), 0.99)
        self.assertGreater(result.part_similarity.item(), 0.99)

    def test_unseen_source_part_abstains_exactly(self):
        memory = PairedEditMemoryState(
            source_key=torch.tensor([[[[1.0, 0.0]]]]),
            target_value_residual=torch.tensor([[[[9.0, 0.0]]]]),
            object_coordinate=torch.zeros(1, 1, 2),
            evidence=torch.ones(1, 1),
            source_part_signature=torch.tensor([[[1.0, 0.0]]]),
        )
        result = source_addressed_residual_read(
            current_source_key=torch.tensor([[[[1.0, 0.0]]]]),
            current_coordinate=torch.zeros(1, 1, 2),
            current_owner=torch.ones(1, 1),
            memory=memory,
            current_source_part_signature=torch.tensor(
                [[[0.0, 1.0]]]
            ),
            topk=1,
            min_similarity=0.0,
            coordinate_bias=0.0,
            source_part_consistency=True,
            min_part_similarity=0.45,
        )
        self.assertEqual(result.support.item(), 0.0)
        torch.testing.assert_close(
            result.residual, torch.zeros_like(result.residual)
        )

    def test_memory_commit_and_read_preserve_source_part_identity(self):
        memory = CausalPairedEditMemory(
            layers=(0,),
            max_tokens=4,
            max_tokens_per_block=2,
            min_commit_confidence=0.2,
            min_similarity=0.0,
            coordinate_bias=0.0,
            source_part_consistency=True,
            min_part_similarity=0.45,
            part_similarity_margin=0.08,
            topk=2,
        )
        ambiguous_key = torch.tensor(
            [[[[1.0, 0.0]], [[1.0, 0.0]]]]
        )
        source_value = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        target_value = source_value + torch.tensor(
            [[[[7.0, 0.0]], [[0.0, 3.0]]]]
        )
        memory.commit(
            source_kv_cache=make_cache(ambiguous_key, source_value),
            target_kv_cache=make_cache(ambiguous_key, target_value),
            proposal_weight=torch.ones(1, 2),
            object_coordinate=torch.zeros(1, 2, 2),
            transactional=False,
        )
        state = memory.export()[0]
        self.assertIsNotNone(state.source_part_signature)

        read = memory.read(
            source_kv_cache=make_cache(
                ambiguous_key[:, :1], source_value[:, 1:2]
            ),
            current_coordinate=torch.zeros(1, 1, 2),
            current_object_request=torch.ones(1, 1),
        )[0]
        torch.testing.assert_close(
            read.residual, torch.tensor([[[[0.0, 3.0]]]])
        )
        self.assertGreater(read.part_confidence.item(), 0.99)

    def test_local_coordinate_read_does_not_cross_object_parts(self):
        memory = PairedEditMemoryState(
            source_key=torch.tensor(
                [[[[0.8, 0.6]], [[1.0, 0.0]]]]
            ),
            target_value_residual=torch.tensor(
                [[[[2.0, 0.0]], [[0.0, 9.0]]]]
            ),
            object_coordinate=torch.tensor(
                [[[0.0, 0.0], [0.8, 0.8]]]
            ),
            evidence=torch.ones(1, 2),
        )
        result = source_addressed_residual_read(
            current_source_key=torch.tensor([[[[1.0, 0.0]]]]),
            current_coordinate=torch.tensor([[[0.0, 0.0]]]),
            current_owner=torch.ones(1, 1),
            memory=memory,
            topk=2,
            min_similarity=0.35,
            coordinate_bias=0.0,
            coordinate_radius=0.25,
            min_residual_consensus=0.35,
        )
        torch.testing.assert_close(
            result.residual, torch.tensor([[[[2.0, 0.0]]]])
        )
        self.assertGreater(result.support.item(), 0.0)

    def test_conflicting_local_residuals_abstain(self):
        memory = PairedEditMemoryState(
            source_key=torch.tensor(
                [[[[1.0, 0.0]], [[1.0, 0.0]]]]
            ),
            target_value_residual=torch.tensor(
                [[[[1.0, 0.0]], [[-1.0, 0.0]]]]
            ),
            object_coordinate=torch.zeros(1, 2, 2),
            evidence=torch.ones(1, 2),
        )
        result = source_addressed_residual_read(
            current_source_key=torch.tensor([[[[1.0, 0.0]]]]),
            current_coordinate=torch.zeros(1, 1, 2),
            current_owner=torch.ones(1, 1),
            memory=memory,
            topk=2,
            min_similarity=0.35,
            coordinate_bias=0.0,
            coordinate_radius=0.25,
            min_residual_consensus=0.35,
        )
        self.assertEqual(result.support.item(), 0.0)
        self.assertEqual(result.residual_consensus.item(), 0.0)

    def test_coherent_local_residuals_pass_consensus_gate(self):
        memory = PairedEditMemoryState(
            source_key=torch.tensor(
                [[[[1.0, 0.0]], [[1.0, 0.0]]]]
            ),
            target_value_residual=torch.tensor(
                [[[[1.0, 0.0]], [[0.9, 0.0]]]]
            ),
            object_coordinate=torch.zeros(1, 2, 2),
            evidence=torch.ones(1, 2),
        )
        result = source_addressed_residual_read(
            current_source_key=torch.tensor([[[[1.0, 0.0]]]]),
            current_coordinate=torch.zeros(1, 1, 2),
            current_owner=torch.ones(1, 1),
            memory=memory,
            topk=2,
            min_similarity=0.35,
            coordinate_bias=0.0,
            coordinate_radius=0.25,
            min_residual_consensus=0.35,
        )
        self.assertGreater(result.residual_consensus.item(), 0.9)
        self.assertGreater(result.support.item(), 0.8)

    def test_read_uses_source_address_and_abstains_off_object(self):
        memory = PairedEditMemoryState(
            source_key=torch.tensor(
                [[[[1.0, 0.0]], [[0.0, 1.0]]]]
            ),
            target_value_residual=torch.tensor(
                [[[[3.0, 4.0]], [[8.0, 9.0]]]]
            ),
            object_coordinate=torch.tensor(
                [[[0.0, 0.0], [1.0, 1.0]]]
            ),
            evidence=torch.ones(1, 2),
        )
        result = source_addressed_residual_read(
            current_source_key=torch.tensor(
                [[[[1.0, 0.0]], [[0.0, 1.0]], [[-1.0, 0.0]]]]
            ),
            current_coordinate=torch.tensor(
                [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]
            ),
            current_owner=torch.tensor([[1.0, 0.0, 1.0]]),
            memory=memory,
            topk=1,
            min_similarity=0.35,
            coordinate_bias=1.0,
        )
        torch.testing.assert_close(
            result.residual[0, 0], torch.tensor([[3.0, 4.0]])
        )
        self.assertGreater(result.support[0, 0].item(), 0.99)
        torch.testing.assert_close(
            result.residual[0, 1], torch.zeros(1, 2)
        )
        self.assertEqual(result.support[0, 1].item(), 0.0)
        self.assertEqual(result.support[0, 2].item(), 0.0)

    def test_transaction_rejects_inconsistent_residual(self):
        memory = CausalPairedEditMemory(
            layers=(0,),
            max_tokens=4,
            max_tokens_per_block=1,
            min_commit_confidence=0.2,
            min_similarity=0.0,
            coordinate_bias=0.0,
            topk=1,
        )
        key = torch.tensor([[[[1.0, 0.0]]]])
        source_value = torch.zeros_like(key)
        coordinate = torch.zeros(1, 1, 2)
        proposal = torch.ones(1, 1)

        first = memory.commit(
            source_kv_cache=make_cache(key, source_value),
            target_kv_cache=make_cache(
                key, torch.tensor([[[[1.0, 0.0]]]])
            ),
            proposal_weight=proposal,
            object_coordinate=coordinate,
            transactional=False,
        )
        self.assertEqual(first["accepted"].item(), 1.0)
        self.assertTrue(memory.has_state())

        rejected = memory.commit(
            source_kv_cache=make_cache(key, source_value),
            target_kv_cache=make_cache(
                key, torch.tensor([[[[-1.0, 0.0]]]])
            ),
            proposal_weight=proposal,
            object_coordinate=coordinate,
            transactional=True,
        )
        self.assertEqual(rejected["accepted"].item(), 0.0)
        self.assertEqual(rejected["write"].item(), 0.0)

    def test_transaction_accepts_consistent_residual(self):
        memory = CausalPairedEditMemory(
            layers=(0,),
            max_tokens=4,
            max_tokens_per_block=1,
            min_commit_confidence=0.2,
            min_similarity=0.0,
            coordinate_bias=0.0,
            topk=1,
        )
        key = torch.tensor([[[[1.0, 0.0]]]])
        source_value = torch.zeros_like(key)
        target_value = torch.tensor([[[[1.0, 0.0]]]])
        coordinate = torch.zeros(1, 1, 2)
        proposal = torch.ones(1, 1)
        memory.commit(
            source_kv_cache=make_cache(key, source_value),
            target_kv_cache=make_cache(key, target_value),
            proposal_weight=proposal,
            object_coordinate=coordinate,
            transactional=False,
        )
        accepted = memory.commit(
            source_kv_cache=make_cache(key, source_value),
            target_kv_cache=make_cache(key, target_value),
            proposal_weight=proposal,
            object_coordinate=coordinate,
            transactional=True,
        )
        self.assertEqual(accepted["accepted"].item(), 1.0)
        self.assertGreater(accepted["write"].item(), 0.99)

    def test_transaction_can_extend_address_without_payload_drift(self):
        memory = CausalPairedEditMemory(
            layers=(0,),
            max_tokens=4,
            max_tokens_per_block=1,
            min_commit_confidence=0.1,
            min_similarity=0.0,
            coordinate_bias=0.0,
            topk=1,
        )
        key = torch.tensor([[[[1.0, 0.0]]]])
        source_value = torch.zeros_like(key)
        coordinate = torch.zeros(1, 1, 2)
        memory.commit(
            source_kv_cache=make_cache(key, source_value),
            target_kv_cache=make_cache(
                key, torch.tensor([[[[1.0, 0.0]]]])
            ),
            proposal_weight=torch.ones(1, 1),
            object_coordinate=coordinate,
            transactional=False,
            preserve_canonical_payload=True,
        )
        memory.commit(
            source_kv_cache=make_cache(key, source_value),
            target_kv_cache=make_cache(
                key, torch.tensor([[[[0.8, 0.0]]]])
            ),
            proposal_weight=torch.ones(1, 1),
            object_coordinate=coordinate,
            transactional=True,
            preserve_canonical_payload=True,
        )
        state = memory.export()[0]
        valid = state.evidence[0] > 0
        torch.testing.assert_close(
            state.target_value_residual[0, valid],
            torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]]),
        )

    def test_cache_projection_changes_only_supported_current_values(self):
        memory = CausalPairedEditMemory(
            layers=(0,),
            max_tokens=4,
            max_tokens_per_block=2,
            min_commit_confidence=0.2,
            min_similarity=0.0,
            coordinate_bias=0.0,
            topk=1,
        )
        source_key = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        source_value = torch.tensor(
            [[[[1.0, 2.0]], [[3.0, 4.0]]]]
        )
        target_value = torch.tensor(
            [[[[9.0, 9.0]], [[7.0, 7.0]]]]
        )
        source_cache = make_cache(source_key, source_value)
        target_cache = make_cache(source_key, target_value)
        read = causal_edit_memory.SourceAddressedRead(
            residual=torch.tensor(
                [[[[5.0, 6.0]], [[8.0, 9.0]]]]
            ),
            support=torch.tensor([[1.0, 0.0]]),
            best_similarity=torch.ones(1, 2),
            assigned_evidence=torch.ones(1, 2),
        )
        diagnostics = memory.project_target_cache(
            source_kv_cache=source_cache,
            target_kv_cache=target_cache,
            reads={0: read},
            strength=1.0,
        )
        torch.testing.assert_close(
            target_cache[0]["v"][:, 0],
            torch.tensor([[[6.0, 8.0]]]),
        )
        torch.testing.assert_close(
            target_cache[0]["v"][:, 1],
            target_value[:, 1],
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(diagnostics["support"][0, 0].item(), 1.0)
        self.assertEqual(diagnostics["support"][0, 1].item(), 0.0)

    def test_dual_timescale_anchor_scatter_is_exact_off_owner(self):
        native = torch.tensor(
            [[
                [[1.0, 2.0]],
                [[3.0, 4.0]],
                [[5.0, 6.0]],
            ]]
        )
        source_anchor = torch.tensor(
            [[[[10.0, 20.0]], [[30.0, 40.0]]]]
        )
        target_anchor = torch.tensor(
            [[[[12.0, 19.0]], [[34.0, 42.0]]]]
        )
        support = torch.tensor([[1.0, 0.0, 1.0]])

        output = scatter_source_addressed_anchor_delta(
            native,
            target_anchor - source_anchor,
            support,
            strength=0.5,
        )

        torch.testing.assert_close(
            output[:, 1], native[:, 1], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            output[:, 0], torch.tensor([[[2.0, 1.5]]])
        )
        torch.testing.assert_close(
            output[:, 2], torch.tensor([[[7.0, 7.0]]])
        )

    def test_dual_timescale_anchor_empty_support_returns_same_tensor(self):
        native = torch.randn(1, 3, 1, 2)
        output = scatter_source_addressed_anchor_delta(
            native,
            native[:, :0],
            torch.zeros(1, 3),
            strength=1.0,
        )
        self.assertIs(output, native)

    def test_anchor_attention_is_lineage_consistent(self):
        query = torch.tensor(
            [[[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]]]
        )
        key = query.clone()
        residual = torch.tensor(
            [[[[2.0, 0.0]], [[0.0, 4.0]], [[6.0, 0.0]]]]
        )
        output = source_addressed_anchor_attention_delta(
            query,
            key,
            residual,
            torch.ones(1, 3),
            torch.tensor([[3, 7, 3]]),
        )
        torch.testing.assert_close(
            output[:, 0], torch.tensor([[[4.0, 0.0]]])
        )
        torch.testing.assert_close(
            output[:, 1], torch.tensor([[[0.0, 4.0]]])
        )
        torch.testing.assert_close(
            output[:, 2], torch.tensor([[[4.0, 0.0]]])
        )

    def test_anchor_attention_consumes_confidence_once(self):
        query = torch.tensor([[[[1.0, 0.0]]]])
        residual = torch.tensor([[[[4.0, 2.0]]]])
        output = source_addressed_anchor_attention_delta(
            query,
            query,
            residual,
            torch.tensor([[0.5]]),
            torch.tensor([[2]]),
        )
        torch.testing.assert_close(
            output, torch.tensor([[[[2.0, 1.0]]]])
        )

    def test_anchor_attention_casts_fp32_memory_to_bfloat16_compute(self):
        query = torch.tensor(
            [[[[1.0, 0.0]]]], dtype=torch.bfloat16
        )
        residual = torch.tensor(
            [[[[4.0, 2.0]]]], dtype=torch.float32
        )
        output = source_addressed_anchor_attention_delta(
            query,
            query.clone(),
            residual,
            torch.tensor([[0.5]], dtype=torch.float32),
            torch.tensor([[2]]),
        )
        self.assertEqual(output.dtype, torch.bfloat16)
        torch.testing.assert_close(
            output.float(), torch.tensor([[[[2.0, 1.0]]]])
        )

    def test_immutable_ignition_bank_does_not_grow_or_rewrite(self):
        memory = self.make_transported_memory(
            max_tokens_per_block=2,
            immutable_canonical_key_anchor=True,
        )
        ignition_key = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        ignition_residual = torch.tensor(
            [[[[2.0, 0.0]], [[0.0, 3.0]]]]
        )
        self.commit_ignition(memory, ignition_key, ignition_residual)
        self.assertFalse(memory.ignition_is_verified())
        original = memory.export_ignition()[0]
        original_key = original.source_key.clone()
        original_residual = original.target_value_residual.clone()

        memory.read(
            source_kv_cache=make_cache(
                ignition_key, torch.zeros_like(ignition_key)
            ),
            current_coordinate=torch.zeros(1, 2, 2),
            current_object_request=torch.ones(1, 2),
        )
        memory.commit(
            source_kv_cache=make_cache(
                ignition_key, torch.zeros_like(ignition_key)
            ),
            target_kv_cache=make_cache(
                ignition_key, ignition_residual * 0.9
            ),
            proposal_weight=torch.ones(1, 2),
            object_coordinate=torch.zeros(1, 2, 2),
            transactional=True,
            preserve_canonical_payload=True,
        )
        ignition = memory.export_ignition()[0]
        self.assertTrue(memory.ignition_is_verified())
        self.assertEqual(ignition.source_key.shape[1], 2)
        torch.testing.assert_close(
            ignition.source_key, original_key, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            ignition.target_value_residual,
            original_residual,
            rtol=0.0,
            atol=0.0,
        )
        self.assertGreater(memory.export()[0].source_key.shape[1], 2)

    def test_immutable_ignition_replay_only_filters_evidence(self):
        memory = self.make_transported_memory(
            max_tokens_per_block=2,
            immutable_canonical_key_anchor=True,
        )
        key = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
        residual = torch.tensor(
            [[[[2.0, 0.0]], [[0.0, 3.0]]]]
        )
        self.commit_ignition(memory, key, residual)
        memory.read(
            source_kv_cache=make_cache(key, torch.zeros_like(key)),
            current_coordinate=torch.zeros(1, 2, 2),
            current_object_request=torch.tensor([[1.0, 0.0]]),
        )
        memory.commit(
            source_kv_cache=make_cache(key, torch.zeros_like(key)),
            target_kv_cache=make_cache(key, residual),
            proposal_weight=torch.tensor([[1.0, 0.0]]),
            object_coordinate=torch.zeros(1, 2, 2),
            transactional=True,
            preserve_canonical_payload=True,
        )
        ignition = memory.export_ignition()[0]
        self.assertGreater(ignition.evidence[0, 0].item(), 0.0)
        self.assertEqual(ignition.evidence[0, 1].item(), 0.0)
        torch.testing.assert_close(
            ignition.target_value_residual, residual
        )

    def test_canonical_anchor_request_uses_binary_query_admission(self):
        memory = self.make_transported_memory(
            max_tokens_per_block=1,
            immutable_canonical_key_anchor=True,
        )
        key = torch.tensor([[[[1.0, 0.0]]]])
        residual = torch.tensor([[[[2.0, 0.0]]]])
        self.commit_ignition(memory, key, residual)
        unverified_read = memory.read(
            source_kv_cache=make_cache(key, torch.zeros_like(key)),
            current_coordinate=torch.zeros(1, 1, 2),
            current_object_request=torch.ones(1, 1),
        )[0]
        self.assertEqual(
            memory.build_canonical_anchor_requests(
                {0: unverified_read}, torch.zeros(1, 1, 2)
            ),
            {},
        )
        memory.commit(
            source_kv_cache=make_cache(key, torch.zeros_like(key)),
            target_kv_cache=make_cache(key, residual),
            proposal_weight=torch.ones(1, 1),
            object_coordinate=torch.zeros(1, 1, 2),
            transactional=True,
            preserve_canonical_payload=True,
        )
        read = memory.read(
            source_kv_cache=make_cache(key, torch.zeros_like(key)),
            current_coordinate=torch.zeros(1, 1, 2),
            current_object_request=torch.tensor([[0.25]]),
        )[0]
        request = memory.build_canonical_anchor_requests(
            {0: read}, torch.zeros(1, 1, 2)
        )[0]
        self.assertEqual(request.query_support.item(), 1.0)
        torch.testing.assert_close(
            request.target_value_residual, residual
        )

    def test_immutable_anchor_cross_attention_supports_q_not_equal_m(self):
        query = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]]]]
        )
        key = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        residual = torch.tensor(
            [[[[2.0, 0.0]], [[0.0, 4.0]]]]
        )
        output = immutable_canonical_anchor_attention_delta(
            query,
            key,
            residual,
            torch.tensor([[1.0, 0.5]]),
            torch.tensor([[3, 7, 3]]),
            torch.tensor([[3, 7]]),
            query_key_mask=torch.tensor(
                [[[True, False], [False, True], [True, False]]]
            ),
        )
        torch.testing.assert_close(
            output,
            torch.tensor(
                [[[[2.0, 0.0]], [[0.0, 4.0]], [[2.0, 0.0]]]]
            ),
        )

    def test_immutable_anchor_evidence_is_logit_prior_not_value_scale(self):
        query = torch.tensor([[[[1.0, 0.0]]]])
        key = torch.tensor([[[[1.0, 0.0]]]])
        residual = torch.tensor([[[[4.0, 2.0]]]])
        output = immutable_canonical_anchor_attention_delta(
            query,
            key,
            residual,
            torch.tensor([[0.05]]),
            torch.tensor([[2]]),
            torch.tensor([[2]]),
        )
        torch.testing.assert_close(output, residual)

    def test_immutable_anchor_query_key_mask_can_expand_a_lineage_locally(self):
        query = torch.tensor([[[[1.0, 0.0]]]])
        key = torch.tensor(
            [[[[1.0, 0.0]], [[1.0, 0.0]]]]
        )
        residual = torch.tensor(
            [[[[2.0, 0.0]], [[6.0, 0.0]]]]
        )
        output = immutable_canonical_anchor_attention_delta(
            query,
            key,
            residual,
            torch.ones(1, 2),
            torch.tensor([[3]]),
            torch.tensor([[3, 4]]),
            query_key_mask=torch.ones(1, 1, 2, dtype=torch.bool),
        )
        torch.testing.assert_close(
            output, torch.tensor([[[[4.0, 0.0]]]])
        )

    def test_immutable_anchor_rejects_query_without_eligible_key(self):
        with self.assertRaisesRegex(
            ValueError, "verified lineage key"
        ):
            immutable_canonical_anchor_attention_delta(
                torch.tensor([[[[1.0, 0.0]]]]),
                torch.tensor([[[[1.0, 0.0]]]]),
                torch.tensor([[[[2.0, 0.0]]]]),
                torch.ones(1, 1),
                torch.tensor([[3]]),
                torch.tensor([[4]]),
                query_key_mask=torch.zeros(
                    1, 1, 1, dtype=torch.bool
                ),
            )

    def test_memory_compatibility_includes_part_consistency(self):
        baseline = CausalPairedEditMemory(
            layers=(0,),
            coordinate_radius=0.25,
            min_residual_consensus=0.45,
        )
        matching = CausalPairedEditMemory(
            layers=(0,),
            coordinate_radius=0.25,
            min_residual_consensus=0.45,
        )
        different_radius = CausalPairedEditMemory(
            layers=(0,),
            coordinate_radius=0.30,
            min_residual_consensus=0.45,
        )
        different_consensus = CausalPairedEditMemory(
            layers=(0,),
            coordinate_radius=0.25,
            min_residual_consensus=0.50,
        )
        part_consistent = CausalPairedEditMemory(
            layers=(0,),
            coordinate_radius=0.25,
            min_residual_consensus=0.45,
            source_part_consistency=True,
        )
        different_part_threshold = CausalPairedEditMemory(
            layers=(0,),
            coordinate_radius=0.25,
            min_residual_consensus=0.45,
            min_part_similarity=0.60,
        )
        self.assertTrue(baseline.compatible_with(matching))
        self.assertFalse(baseline.compatible_with(different_radius))
        self.assertFalse(baseline.compatible_with(different_consensus))
        self.assertFalse(baseline.compatible_with(part_consistent))
        self.assertFalse(
            baseline.compatible_with(different_part_threshold)
        )


if __name__ == "__main__":
    unittest.main()
