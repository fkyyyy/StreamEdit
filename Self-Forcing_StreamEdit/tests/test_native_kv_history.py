import importlib.util
import sys
import unittest
from pathlib import Path

import torch

from tests._pipeline_imports import load_pipeline_module


def load_module(relative_path, name):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


native_history = load_module(
    "pipeline/native_kv_history.py", "streamedit_native_kv_history"
)
attention = load_module(
    "wan/modules/attention.py", "streamedit_native_kv_attention"
)
RoleConditionedNativeKVHistory = (
    native_history.RoleConditionedNativeKVHistory
)
NativeFlowResidualFrame = native_history.NativeFlowResidualFrame
validate_recent_entry_hand_only_contract = (
    native_history.validate_recent_entry_hand_only_contract
)
source_addressed_native_history_attention = (
    attention.source_addressed_native_history_attention
)
source_flow_gated_multiframe_sink_attention = (
    attention.source_flow_gated_multiframe_sink_attention
)
closed_loop_counterfactual_memory_attention = (
    attention.closed_loop_counterfactual_memory_attention
)
arbitrate_verified_factorized_attention = (
    attention.arbitrate_verified_factorized_attention
)


def make_cache(key, value):
    length = key.shape[1]
    return [{
        "k": key.clone(),
        "v": value.clone(),
        "local_end_index": torch.tensor([length]),
        "global_end_index": torch.tensor([length]),
        "num_new_tokens": length,
    }]


def translation_flow_cache(frames=4):
    module = load_pipeline_module("motion.causal_motion_owner")
    forward = torch.zeros(frames - 1, 2, 2, 4)
    backward = torch.zeros_like(forward)
    forward[:, 0] = 1.0
    backward[:, 0] = -1.0
    confidence = torch.ones(frames - 1, 1, 2, 4)
    return module.SourceFlowCache(
        {
            "forward_flow": forward,
            "backward_flow": backward,
            "forward_confidence": confidence,
            "backward_confidence": confidence,
        },
        latent_pixel_indices=list(range(frames)),
    )


def identity_flow_cache(frames=10, confidence=0.8, displacement=0.0):
    module = load_pipeline_module("motion.causal_motion_owner")
    forward = torch.zeros(frames - 1, 2, 2, 4)
    backward = torch.zeros_like(forward)
    forward[:, 0] = float(displacement)
    backward[:, 0] = -float(displacement)
    reliability = torch.full(
        (frames - 1, 1, 2, 4), float(confidence)
    )
    return module.SourceFlowCache(
        {
            "forward_flow": forward,
            "backward_flow": backward,
            "forward_confidence": reliability,
            "backward_confidence": reliability,
        },
        latent_pixel_indices=list(range(frames)),
    )


class NativeKVHistoryTest(unittest.TestCase):
    def test_tccm_zero_error_is_exact_native_fallback(self):
        native = torch.randn(1, 2, 1, 2)
        source_key = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        source_value = torch.zeros_like(source_key)
        target_key = source_key.clone()
        target_value = torch.tensor(
            [[[[2.0, 0.0]], [[0.0, 3.0]]]]
        )
        output, diagnostics = closed_loop_counterfactual_memory_attention(
            native_output=native,
            current_source_query=source_key,
            current_source_key=source_key,
            current_source_value=source_value,
            current_target_key=target_key,
            current_target_value=target_value,
            canonical_source_key=source_key,
            canonical_source_value=source_value,
            canonical_target_key=target_key,
            canonical_target_value=target_value,
            canonical_support=torch.ones(1, 2, dtype=torch.bool),
            canonical_token_index=torch.tensor([[0, 1]]),
            mapped_current_index=torch.tensor([[[0, 1]]]),
            correspondence_support=torch.ones(
                1, 1, 2, dtype=torch.bool
            ),
            correspondence_confidence=torch.ones(1, 1, 2),
            owner_gate=torch.ones(1, 2),
            appearance_trust=torch.ones(1, 2),
            transport_confidence=torch.ones(1, 2),
            tokens_per_frame=2, spatial_shape=(1, 2),
            canonical_frame_count=1, current_frame_count=1,
            topk_per_frame=2, min_source_similarity=-0.5,
            flow_radius=2.0, strength=1.0,
        )
        torch.testing.assert_close(output, native, rtol=0, atol=0)
        torch.testing.assert_close(
            diagnostics["tccm_error_norm"],
            torch.zeros(1, 2), rtol=0, atol=1e-6,
        )

    def test_tccm_corrects_desired_minus_current_response(self):
        native = torch.zeros(1, 1, 1, 2)
        output, diagnostics = closed_loop_counterfactual_memory_attention(
            native_output=native,
            current_source_query=torch.tensor([[[[1.0, 0.0]]]]),
            current_source_key=torch.tensor([[[[1.0, 0.0]]]]),
            current_source_value=torch.zeros(1, 1, 1, 2),
            current_target_key=torch.tensor([[[[1.0, 0.0]]]]),
            current_target_value=torch.zeros(1, 1, 1, 2),
            canonical_source_key=torch.tensor([[[[1.0, 0.0]]]]),
            canonical_source_value=torch.zeros(1, 1, 1, 2),
            canonical_target_key=torch.tensor([[[[1.0, 0.0]]]]),
            canonical_target_value=torch.tensor([[[[2.0, 0.0]]]]),
            canonical_support=torch.ones(1, 1, dtype=torch.bool),
            canonical_token_index=torch.zeros(1, 1, dtype=torch.long),
            mapped_current_index=torch.zeros(1, 1, 1, dtype=torch.long),
            correspondence_support=torch.ones(
                1, 1, 1, dtype=torch.bool
            ),
            correspondence_confidence=torch.ones(1, 1, 1),
            owner_gate=torch.ones(1, 1),
            appearance_trust=torch.ones(1, 1),
            transport_confidence=torch.ones(1, 1),
            tokens_per_frame=1, spatial_shape=(1, 1),
            canonical_frame_count=1, current_frame_count=1,
            topk_per_frame=1, min_source_similarity=-0.5,
            flow_radius=0.0, strength=1.0, max_error_ratio=1.0,
        )
        torch.testing.assert_close(
            output, torch.tensor([[[[2.0, 0.0]]]]), rtol=0, atol=1e-6
        )
        self.assertTrue(bool(diagnostics["tccm_admitted"].all()))
        self.assertGreater(diagnostics["tccm_error_norm"].item(), 0.0)

    def test_tccm_abstention_is_bit_exact(self):
        native = torch.randn(1, 1, 1, 2)
        output, diagnostics = closed_loop_counterfactual_memory_attention(
            native_output=native,
            current_source_query=torch.ones_like(native),
            current_source_key=torch.ones_like(native),
            current_source_value=torch.zeros_like(native),
            current_target_key=torch.ones_like(native),
            current_target_value=torch.ones_like(native),
            canonical_source_key=torch.ones_like(native),
            canonical_source_value=torch.zeros_like(native),
            canonical_target_key=torch.ones_like(native),
            canonical_target_value=torch.full_like(native, 4.0),
            canonical_support=torch.ones(1, 1, dtype=torch.bool),
            canonical_token_index=torch.zeros(1, 1, dtype=torch.long),
            mapped_current_index=torch.zeros(1, 1, 1, dtype=torch.long),
            correspondence_support=torch.zeros(
                1, 1, 1, dtype=torch.bool
            ),
            correspondence_confidence=torch.zeros(1, 1, 1),
            owner_gate=torch.ones(1, 1),
            appearance_trust=torch.ones(1, 1),
            transport_confidence=torch.ones(1, 1),
            tokens_per_frame=1, spatial_shape=(1, 1),
            canonical_frame_count=1, current_frame_count=1,
            topk_per_frame=1, min_source_similarity=-0.5,
            flow_radius=0.0, strength=1.0,
        )
        torch.testing.assert_close(output, native, rtol=0, atol=0)
        self.assertFalse(bool(diagnostics["tccm_admitted"].any()))

    def test_tccm_chunked_queries_with_padded_candidates_stay_finite(self):
        query_count = 130
        native = torch.zeros(1, query_count, 1, 2)
        query = torch.tensor([1.0, 0.0]).reshape(1, 1, 1, 2).expand(
            1, query_count, 1, 2
        ).clone()
        current_key = query.clone()
        current_value = torch.zeros_like(query)
        canonical_key = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]], [[-1.0, 0.0]]]]
        )
        canonical_source_value = torch.zeros_like(canonical_key)
        canonical_target_value = torch.tensor(
            [[[[2.0, 0.0]], [[9.0, 9.0]], [[9.0, 9.0]]]]
        )
        output, diagnostics = closed_loop_counterfactual_memory_attention(
            native_output=native,
            current_source_query=query,
            current_source_key=current_key,
            current_source_value=current_value,
            current_target_key=current_key,
            current_target_value=current_value,
            canonical_source_key=canonical_key,
            canonical_source_value=canonical_source_value,
            canonical_target_key=canonical_key,
            canonical_target_value=canonical_target_value,
            canonical_support=torch.tensor([[True, False, False]]),
            canonical_token_index=torch.tensor([[0, 1, 130]]),
            mapped_current_index=torch.tensor([[[0, 1, 2]]]),
            correspondence_support=torch.ones(
                1, 1, 3, dtype=torch.bool
            ),
            correspondence_confidence=torch.ones(1, 1, 3),
            owner_gate=torch.ones(1, query_count),
            appearance_trust=torch.ones(1, query_count),
            transport_confidence=torch.ones(1, query_count),
            tokens_per_frame=query_count, spatial_shape=(10, 13),
            canonical_frame_count=2, current_frame_count=1,
            topk_per_frame=2, min_source_similarity=-0.5,
            flow_radius=20.0, strength=1.0, max_error_ratio=1.0,
        )
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertTrue(bool(torch.isfinite(
            diagnostics["tccm_error_norm"]
        ).all()))
        torch.testing.assert_close(
            output, torch.tensor([2.0, 0.0]).reshape(1, 1, 1, 2).expand_as(
                output
            ), rtol=0, atol=1e-6,
        )

    def test_tccm_bank_freezes_on_first_canonical_commit(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=4, max_tokens_per_frame=1,
            min_write_confidence=0.5, transactional_dense_recent=True,
            token_atomic_dense_recent=True, persistent_residual_upsert=True,
            last_trusted_residual_lineage=True,
            flow_indexed_residual_ledger=True, decoupled_flow_trust=True,
            multiframe_identity_sink=True,
            timestep_counterfactual_memory=True,
            source_flow_cache=identity_flow_cache(frames=5, confidence=1.0),
        )
        source = torch.arange(24, dtype=torch.float32).reshape(1, 12, 1, 2)
        target = source + 5.0
        selection = torch.zeros(1, 12)
        selection[:, (1, 6, 11)] = 1.0
        self.assertTrue(history.stage_timestep_counterfactual(
            layer=0, timestep_index=3, source_key=source,
            source_value=source + 1, target_key=target,
            target_value=target + 1, selection_weight=selection,
        ))
        history.commit(
            source_kv_cache=make_cache(source, source + 1),
            target_kv_cache=make_cache(target, target + 1),
            write_confidence=selection,
            retention_confidence=torch.ones_like(selection),
            frame_indices=(0, 1, 2), spatial_shape=(2, 2),
        )
        frozen = history.read_timestep_counterfactual(0, 3)
        self.assertTrue(history.timestep_bank_frozen)
        self.assertIsNotNone(frozen)
        torch.testing.assert_close(
            frozen.token_index, torch.tensor([[1, 6, 11]])
        )
        frozen_value = frozen.target_value.clone()
        self.assertFalse(history.stage_timestep_counterfactual(
            layer=0, timestep_index=3, source_key=source + 100,
            source_value=source + 100, target_key=target + 100,
            target_value=target + 100, selection_weight=selection,
        ))
        torch.testing.assert_close(
            history.read_timestep_counterfactual(0, 3).target_value,
            frozen_value, rtol=0, atol=0,
        )

    def test_tccm_correspondence_tracks_canonical_slots_across_chunk(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=8, max_tokens_per_frame=1,
            min_write_confidence=0.5, transactional_dense_recent=True,
            token_atomic_dense_recent=True, persistent_residual_upsert=True,
            last_trusted_residual_lineage=True,
            flow_indexed_residual_ledger=True, decoupled_flow_trust=True,
            multiframe_identity_sink=True,
            timestep_counterfactual_memory=True,
            source_flow_cache=translation_flow_cache(frames=7),
        )
        source = torch.arange(48, dtype=torch.float32).reshape(1, 24, 1, 2)
        write = torch.zeros(1, 24)
        write[:, (0, 8, 16)] = 1.0
        history.stage_timestep_counterfactual(
            layer=0, timestep_index=0, source_key=source,
            source_value=source, target_key=source + 1,
            target_value=source + 2, selection_weight=write,
        )
        history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(source + 1, source + 2),
            write_confidence=write,
            retention_confidence=torch.ones_like(write),
            frame_indices=(0, 1, 2), spatial_shape=(2, 4),
        )
        correspondence = history.prepare_canonical_correspondence(
            frame_indices=(3, 4, 5), spatial_shape=(2, 4),
            device=torch.device("cpu"),
        )[0]
        self.assertEqual(correspondence.current_index.shape, (1, 3, 3))
        self.assertTrue(bool(correspondence.support.any()))
        # The encoded indices are local to each current block frame.
        self.assertTrue(bool((correspondence.current_index[:, 0] < 8).all()))
        self.assertTrue(bool((correspondence.current_index[:, 1] >= 8).all()))
        self.assertTrue(bool((correspondence.current_index[:, 2] >= 16).all()))

    def test_multiframe_sink_restricts_target_attention_to_source_candidates(
        self,
    ):
        native = torch.zeros(1, 1, 1, 2)
        output, diagnostics = source_flow_gated_multiframe_sink_attention(
            native_output=native,
            target_query=torch.tensor([[[[10.0, 0.0]]]]),
            current_source_key=torch.tensor([[[[1.0, 0.0]]]]),
            canonical_source_key=torch.tensor(
                [[[[1.0, 0.0]], [[0.0, 1.0]]]]
            ),
            # The source-incompatible second slot is overwhelmingly favored
            # by target attention, but must never enter its candidate set.
            canonical_target_key=torch.tensor(
                [[[[0.0, 1.0]], [[100.0, 0.0]]]]
            ),
            canonical_source_value=torch.zeros(1, 2, 1, 2),
            canonical_target_value=torch.tensor(
                [[[[1.0, 0.0]], [[0.0, 100.0]]]]
            ),
            canonical_support=torch.ones(1, 2, dtype=torch.bool),
            canonical_token_index=torch.tensor([[0, 1]]),
            owner_gate=torch.ones(1, 1),
            appearance_trust=torch.ones(1, 1),
            transport_confidence=torch.ones(1, 1),
            tokens_per_frame=2, frame_count=1, topk_per_frame=1,
            min_source_similarity=0.5, source_logit_bias=0.0,
        )
        torch.testing.assert_close(
            output, torch.tensor([[[[1.0, 0.0]]]]), rtol=0, atol=1e-6
        )
        self.assertEqual(diagnostics["sink_candidate_count"].item(), 1.0)

    def test_multiframe_sink_target_query_selects_within_candidates(self):
        common = dict(
            native_output=torch.zeros(1, 2, 1, 2),
            current_source_key=torch.tensor(
                [[[[1.0, 0.0]], [[1.0, 0.0]]]]
            ),
            canonical_source_key=torch.tensor(
                [[[[1.0, 0.0]], [[1.0, 0.0]]]]
            ),
            canonical_target_key=torch.tensor(
                [[[[1.0, 0.0]], [[0.0, 1.0]]]]
            ),
            canonical_source_value=torch.zeros(1, 2, 1, 2),
            canonical_target_value=torch.tensor(
                [[[[2.0, 0.0]], [[0.0, 3.0]]]]
            ),
            canonical_support=torch.ones(1, 2, dtype=torch.bool),
            canonical_token_index=torch.tensor([[0, 1]]),
            owner_gate=torch.ones(1, 2),
            appearance_trust=torch.ones(1, 2),
            transport_confidence=torch.ones(1, 2),
            tokens_per_frame=2, frame_count=1, topk_per_frame=2,
            min_source_similarity=0.5, source_logit_bias=0.0,
        )
        output, diagnostics = source_flow_gated_multiframe_sink_attention(
            target_query=torch.tensor(
                [[[[10.0, 0.0]], [[0.0, 10.0]]]]
            ),
            **common,
        )
        self.assertGreater(output[0, 0, 0, 0], output[0, 0, 0, 1])
        self.assertGreater(output[0, 1, 0, 1], output[0, 1, 0, 0])
        self.assertTrue(bool((diagnostics["sink_attention_peak"] > 0.99).all()))

    def test_multiframe_sink_preserves_frame_diversity_and_selects_view(self):
        output, diagnostics = source_flow_gated_multiframe_sink_attention(
            native_output=torch.zeros(1, 1, 1, 2),
            target_query=torch.tensor([[[[0.0, 10.0]]]]),
            current_source_key=torch.tensor([[[[1.0, 0.0]]]]),
            canonical_source_key=torch.tensor(
                [[[[1.0, 0.0]], [[1.0, 0.0]]]]
            ),
            canonical_target_key=torch.tensor(
                [[[[1.0, 0.0]], [[0.0, 1.0]]]]
            ),
            canonical_source_value=torch.zeros(1, 2, 1, 2),
            canonical_target_value=torch.tensor(
                [[[[2.0, 0.0]], [[0.0, 3.0]]]]
            ),
            canonical_support=torch.ones(1, 2, dtype=torch.bool),
            canonical_token_index=torch.tensor([[0, 2]]),
            owner_gate=torch.ones(1, 1),
            appearance_trust=torch.ones(1, 1),
            transport_confidence=torch.ones(1, 1),
            tokens_per_frame=2, frame_count=2, topk_per_frame=1,
            min_source_similarity=0.5, source_logit_bias=0.0,
        )
        torch.testing.assert_close(
            diagnostics["sink_frame_candidate_count"],
            torch.ones(1, 1, 2), rtol=0, atol=0,
        )
        self.assertEqual(diagnostics["sink_selected_frame"].item(), 1.0)
        self.assertGreater(output[0, 0, 0, 1], output[0, 0, 0, 0])

    def test_multiframe_sink_owner_and_flow_abstain_exactly(self):
        native = torch.randn(1, 2, 1, 2)
        output, diagnostics = source_flow_gated_multiframe_sink_attention(
            native_output=native, target_query=torch.randn_like(native),
            current_source_key=torch.ones_like(native),
            canonical_source_key=torch.ones(1, 1, 1, 2),
            canonical_target_key=torch.ones(1, 1, 1, 2),
            canonical_source_value=torch.zeros(1, 1, 1, 2),
            canonical_target_value=torch.ones(1, 1, 1, 2),
            canonical_support=torch.ones(1, 1, dtype=torch.bool),
            canonical_token_index=torch.zeros(1, 1, dtype=torch.long),
            owner_gate=torch.tensor([[0.0, 1.0]]),
            appearance_trust=torch.ones(1, 2),
            transport_confidence=torch.tensor([[1.0, 0.0]]),
            tokens_per_frame=1, frame_count=1, min_source_similarity=-0.5,
        )
        torch.testing.assert_close(output, native, rtol=0, atol=0)
        self.assertFalse(bool(diagnostics["sink_admitted"].any()))

    def test_multiframe_sink_canonical_is_balanced_and_immutable(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=8, max_tokens_per_frame=1,
            min_write_confidence=0.5, transactional_dense_recent=True,
            token_atomic_dense_recent=True, persistent_residual_upsert=True,
            last_trusted_residual_lineage=True,
            flow_indexed_residual_ledger=True, decoupled_flow_trust=True,
            multiframe_identity_sink=True,
            source_flow_cache=identity_flow_cache(frames=6, confidence=1.0),
        )
        source = torch.arange(48, dtype=torch.float32).reshape(1, 24, 1, 2)
        target = source + 10.0
        write = torch.zeros(1, 24)
        write[:, (1, 10, 20)] = torch.tensor([0.7, 0.8, 0.9])
        history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(target, target),
            write_confidence=write, retention_confidence=torch.ones_like(write),
            frame_indices=(0, 1, 2), spatial_shape=(2, 4),
        )
        canonical = history.read()[0].canonical
        torch.testing.assert_close(
            canonical.token_index, torch.tensor([[1, 10, 20]])
        )
        frozen = tuple(
            value.clone() for value in (
                canonical.source_key, canonical.source_value,
                canonical.target_key, canonical.target_value,
                canonical.token_index, canonical.support,
            )
        )

        history.prepare_flow_read(
            frame_indices=(3, 4, 5), spatial_shape=(2, 4),
            device=torch.device("cpu"),
        )
        later = source + 1000.0
        history.commit(
            source_kv_cache=make_cache(later, later),
            target_kv_cache=make_cache(later + 100.0, later + 100.0),
            write_confidence=torch.zeros_like(write),
            retention_confidence=torch.ones_like(write),
            frame_indices=(3, 4, 5), spatial_shape=(2, 4),
        )
        canonical_after = history.read()[0].canonical
        for actual, expected in zip((
            canonical_after.source_key, canonical_after.source_value,
            canonical_after.target_key, canonical_after.target_value,
            canonical_after.token_index, canonical_after.support,
        ), frozen):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_source_addressed_wrapper_uses_multiframe_sink_not_flow_delta(self):
        native = torch.zeros(1, 1, 1, 2)
        output, diagnostics = source_addressed_native_history_attention(
            native, torch.tensor([[[[1.0, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0]]]]),
            torch.tensor([[[[3.0, 0.0]]]]),
            torch.empty(1, 0, 1, 2), torch.empty(1, 0, 1, 2),
            torch.empty(1, 0, dtype=torch.bool),
            torch.zeros_like(native), torch.zeros_like(native),
            torch.tensor([[[[1.0, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0]]]]),
            torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1),
            canonical_source_value=torch.tensor([[[[1.0, 0.0]]]]),
            recent_source_value=torch.empty(1, 0, 1, 2),
            recent_payload_support=torch.empty(1, 0, dtype=torch.bool),
            residual_rebased_payload=True, last_trusted_appearance=True,
            consistent_transaction=True, entry_bridge=True,
            motion_owner_dense_read=True, entry_query_count=1,
            flow_indexed_value_residual=torch.full_like(native, 100.0),
            flow_indexed_support=torch.ones(1, 1, dtype=torch.bool),
            flow_indexed_confidence=torch.ones(1, 1),
            flow_indexed_appearance_trust=torch.full((1, 1), 0.5),
            flow_indexed_transport_confidence=torch.full((1, 1), 0.4),
            multiframe_identity_sink=True,
            canonical_token_index=torch.zeros(1, 1, dtype=torch.long),
            canonical_tokens_per_frame=1, canonical_frame_count=1,
            multiframe_sink_topk_per_frame=1,
            multiframe_sink_source_logit_bias=0.0,
            multiframe_sink_strength=1.0,
        )
        # Canonical residual is [2, 0], gated once by 0.5 * 0.4.  The
        # deliberately huge mutable flow delta must not enter the result.
        torch.testing.assert_close(
            output, torch.tensor([[[[0.4, 0.0]]]]), rtol=0, atol=1e-6
        )
        self.assertTrue(bool(diagnostics["sink_admitted"].all()))
        self.assertEqual(diagnostics["multiframe_identity_sink"].item(), 1.0)

    def test_recent_entry_contract_accepts_hand_only_input(self):
        validate_recent_entry_hand_only_contract(
            enabled=True,
            routing_mode="hand_role_factorized_causal_owner_kv",
            hand_only_mask=object(),
            oracle_object_mask=None,
            oracle_source_owner_mask=None,
            oracle_source_owner_full_mask=None,
        )

    def test_recent_entry_contract_rejects_missing_hand_or_oracle_masks(self):
        valid = {
            "enabled": True,
            "routing_mode": "hand_role_factorized_causal_owner_kv",
            "hand_only_mask": object(),
            "oracle_object_mask": None,
            "oracle_source_owner_mask": None,
            "oracle_source_owner_full_mask": None,
        }
        invalid_overrides = (
            {"hand_only_mask": None},
            {"oracle_object_mask": object()},
            {"oracle_source_owner_mask": object()},
            {"oracle_source_owner_full_mask": object()},
        )
        for override in invalid_overrides:
            with self.subTest(override=tuple(override)):
                arguments = valid | override
                with self.assertRaisesRegex(ValueError, "hand-only"):
                    validate_recent_entry_hand_only_contract(**arguments)

    def test_verified_attention_authority_zero_gate_is_exact_native(self):
        native_source_mixed = torch.randn(1, 4, 2, 3)
        native_with_memory = native_source_mixed + 0.125
        factorized = torch.randn_like(native_source_mixed)
        output = arbitrate_verified_factorized_attention(
            native_with_memory,
            native_source_mixed,
            factorized,
            torch.zeros(1, 4),
            strength=1.0,
        )
        torch.testing.assert_close(
            output, native_with_memory, rtol=0, atol=0
        )

    def test_verified_attention_authority_preserves_memory_residual(self):
        native_source_mixed = torch.zeros(1, 2, 1, 2)
        memory_residual = torch.tensor(
            [[[[0.25, -0.50]], [[0.25, -0.50]]]]
        )
        native_with_memory = native_source_mixed + memory_residual
        factorized = torch.full_like(native_source_mixed, 2.0)
        output = arbitrate_verified_factorized_attention(
            native_with_memory,
            native_source_mixed,
            factorized,
            torch.tensor([[1.0, 0.5]]),
            strength=1.0,
        )
        expected = torch.tensor(
            [[[[2.25, 1.50]], [[1.25, 0.50]]]]
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_verified_attention_authority_rejects_invalid_gate(self):
        native = torch.zeros(1, 2, 1, 2)
        with self.assertRaisesRegex(ValueError, "shape"):
            arbitrate_verified_factorized_attention(
                native, native, native, torch.zeros(1, 2, 1)
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            arbitrate_verified_factorized_attention(
                native, native, native, torch.zeros(1, 2), strength=1.1
            )

    def test_payload_invariant_history_never_stores_recent_target_kv(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=2, max_tokens_per_frame=2,
            min_write_confidence=0.5, payload_invariant_lineage=True,
            min_lineage_similarity=0.2,
        )
        source = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        target = source + 10.0
        history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(target, target + 1.0),
            write_confidence=torch.ones(1, 2),
            lineage_confidence=torch.ones(1, 2),
        )
        canonical_value = history.read()[0].canonical.target_value.clone()

        drifted_target = target + 1000.0
        diagnostics = history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(
                drifted_target, drifted_target + 1.0
            ),
            write_confidence=torch.ones(1, 2),
            lineage_confidence=torch.ones(1, 2),
        )
        read = history.read()[0]
        self.assertIsNone(read.recent)
        self.assertIsNotNone(read.source_lineage)
        self.assertFalse(hasattr(read.source_lineage, "target_value"))
        torch.testing.assert_close(
            read.canonical.target_value, canonical_value, rtol=0, atol=0
        )
        self.assertEqual(
            read.source_lineage.canonical_index.tolist(), [[0, 1]]
        )
        self.assertEqual(
            diagnostics[0]["lineage_tokens"].item(), 2.0
        )

    def test_payload_invariant_lineage_abstention_holds_last_address(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=2, max_tokens_per_frame=2,
            min_write_confidence=0.5, payload_invariant_lineage=True,
            min_lineage_similarity=0.2,
        )
        source = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        target = source + 10.0
        history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(target, target),
            write_confidence=torch.ones(1, 2),
            lineage_confidence=torch.ones(1, 2),
        )
        lineage_before = history.read()[0].source_lineage
        diagnostics = history.commit(
            source_kv_cache=make_cache(source.flip(1), source.flip(1)),
            target_kv_cache=make_cache(target + 100.0, target + 100.0),
            write_confidence=torch.zeros(1, 2),
            lineage_confidence=torch.zeros(1, 2),
        )
        lineage_after = history.read()[0].source_lineage
        torch.testing.assert_close(
            lineage_after.source_key, lineage_before.source_key, rtol=0, atol=0
        )
        torch.testing.assert_close(
            lineage_after.canonical_index,
            lineage_before.canonical_index,
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            diagnostics[0]["lineage_held_tokens"].item(), 2.0
        )

    def test_canonical_native_payload_is_immutable(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=2, max_tokens_per_frame=2,
            min_write_confidence=0.5,
        )
        source = torch.arange(8, dtype=torch.float32).reshape(1, 2, 1, 4)
        target = source + 10
        history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(target, target + 1),
            write_confidence=torch.ones(1, 2),
        )
        canonical_before = history.read()[0].canonical.target_value.clone()
        second_target = target + 100
        history.commit(
            source_kv_cache=make_cache(source + 1, source + 1),
            target_kv_cache=make_cache(
                second_target, second_target + 1
            ),
            write_confidence=torch.ones(1, 2),
        )
        read = history.read()[0]
        torch.testing.assert_close(
            read.canonical.target_value, canonical_before
        )
        torch.testing.assert_close(
            read.canonical.source_value, source
        )
        torch.testing.assert_close(
            read.recent.target_value, second_target + 1
        )
        torch.testing.assert_close(
            read.recent.source_value, source + 1
        )

    def test_write_gate_selects_only_confident_native_tokens(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=4, max_tokens_per_frame=2,
            min_write_confidence=0.6,
        )
        key = torch.arange(16, dtype=torch.float32).reshape(1, 4, 1, 4)
        history.commit(
            source_kv_cache=make_cache(key, key),
            target_kv_cache=make_cache(key + 20, key + 30),
            write_confidence=torch.tensor([[0.1, 0.9, 0.8, 0.2]]),
        )
        canonical = history.read()[0].canonical
        self.assertEqual(canonical.token_index.tolist(), [[1, 2]])
        self.assertTrue(bool(canonical.support.all()))
        torch.testing.assert_close(
            canonical.target_value, (key + 30)[:, [1, 2]]
        )
        # The immediately recent tier is still the complete clean block,
        # with only duplicated canonical slots masked from attention.
        recent = history.read()[0].recent
        self.assertEqual(recent.target_value.shape[1], 4)
        self.assertEqual(recent.support.tolist(), [[True, False, False, True]])

    def test_transactional_recent_contains_only_write_approved_tokens(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=4, max_tokens_per_frame=2,
            min_write_confidence=0.6,
            transactional_compact_recent=True,
        )
        key = torch.arange(16, dtype=torch.float32).reshape(1, 4, 1, 4)
        history.commit(
            source_kv_cache=make_cache(key, key),
            target_kv_cache=make_cache(key + 20, key + 30),
            write_confidence=torch.tensor([[0.1, 0.9, 0.8, 0.2]]),
        )
        recent = history.read()[0].recent
        self.assertEqual(recent.target_value.shape[1], 2)
        self.assertEqual(recent.token_index.tolist(), [[1, 2]])
        # On the ignition read these slots duplicate canonical and are masked;
        # later successful commits replace them with the new transaction.
        self.assertFalse(bool(recent.support.any()))

        next_key = key + 100.0
        history.commit(
            source_kv_cache=make_cache(next_key, next_key),
            target_kv_cache=make_cache(next_key + 20, next_key + 30),
            write_confidence=torch.tensor([[0.8, 0.1, 0.9, 0.2]]),
        )
        recent = history.read()[0].recent
        self.assertEqual(recent.token_index.tolist(), [[2, 0]])
        self.assertTrue(bool(recent.support.all()))
        torch.testing.assert_close(
            recent.target_value, (next_key + 30)[:, [2, 0]]
        )

    def test_transactional_recent_abstention_holds_last_commit(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=2, max_tokens_per_frame=2,
            min_write_confidence=0.5,
            transactional_compact_recent=True,
        )
        key = torch.arange(8, dtype=torch.float32).reshape(1, 2, 1, 4)
        history.commit(
            source_kv_cache=make_cache(key, key),
            target_kv_cache=make_cache(key + 10, key + 20),
            write_confidence=torch.ones(1, 2),
        )
        before = history.read()[0].recent.target_value.clone()
        diagnostics = history.commit(
            source_kv_cache=make_cache(key + 100, key + 100),
            target_kv_cache=make_cache(key + 200, key + 300),
            write_confidence=torch.zeros(1, 2),
        )
        after = history.read()[0].recent.target_value
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        self.assertEqual(
            diagnostics[0]["recent_held_tokens"].item(), 2.0
        )

    def test_dense_recent_transaction_keeps_complete_clean_target_block(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=2, max_tokens_per_frame=1,
            min_write_confidence=0.5, transactional_dense_recent=True,
            dense_recent_min_residual_consensus=0.05,
        )
        source = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]],
              [[1.0, 1.0]], [[1.0, -1.0]]]]
        )
        target = source + torch.tensor([[[[0.0, 2.0]]]])
        diagnostics = history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(target, target),
            write_confidence=torch.tensor([[0.9, 0.0, 0.8, 0.0]]),
        )[0]
        recent = history.read()[0].recent
        self.assertEqual(recent.target_value.shape[1], 4)
        self.assertTrue(bool(recent.support.all()))
        self.assertIsNone(recent.payload_support)
        torch.testing.assert_close(recent.target_value, target)
        self.assertEqual(diagnostics["dense_recent_accepted"].item(), 1.0)

    def test_token_atomic_dense_recent_separates_address_and_payload(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=2, max_tokens_per_frame=1,
            min_write_confidence=0.5, transactional_dense_recent=True,
            token_atomic_dense_recent=True,
            dense_recent_min_residual_consensus=0.05,
        )
        source = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]],
              [[1.0, 1.0]], [[1.0, -1.0]]]]
        )
        target = source + torch.tensor([[[[0.0, 2.0]]]])
        diagnostics = history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(target, target),
            write_confidence=torch.tensor([[0.9, 0.1, 0.8, 0.0]]),
        )[0]

        recent = history.read()[0].recent
        self.assertTrue(bool(recent.support.all()))
        torch.testing.assert_close(
            recent.payload_support,
            torch.tensor([[True, False, True, False]]),
            rtol=0, atol=0,
        )
        self.assertEqual(
            diagnostics["mutable_target_payload_written"].item(), 2.0
        )

    def test_persistent_residual_upsert_retains_unwritten_matched_token(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=2, max_tokens_per_frame=2,
            min_write_confidence=0.5, transactional_dense_recent=True,
            token_atomic_dense_recent=True, persistent_residual_upsert=True,
            min_lineage_similarity=0.8,
        )
        source = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        first_target = source + torch.tensor([[[[2.0, 3.0]]]])
        history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(first_target, first_target),
            write_confidence=torch.ones(1, 2),
            retention_confidence=torch.ones(1, 2),
        )

        # The second token receives no new target write, but its source address
        # remains inside the automatic owner and matches the previous block.
        moved_source_value = source + 10.0
        proposed_target_key = source + torch.tensor([[[[7.0, 9.0]]]])
        proposed_target_value = (
            moved_source_value + torch.tensor([[[[7.0, 9.0]]]])
        )
        diagnostics = history.commit(
            source_kv_cache=make_cache(source, moved_source_value),
            target_kv_cache=make_cache(
                proposed_target_key, proposed_target_value
            ),
            write_confidence=torch.tensor([[1.0, 0.0]]),
            retention_confidence=torch.ones(1, 2),
        )[0]
        recent = history.read()[0].recent

        self.assertTrue(recent.residual_rebased_payload)
        self.assertEqual(recent.payload_support.tolist(), [[True, True]])
        self.assertEqual(
            diagnostics["persistent_direct_support"].tolist(),
            [[True, False]],
        )
        self.assertEqual(
            diagnostics["persistent_retained_support"].tolist(),
            [[False, True]],
        )
        torch.testing.assert_close(
            recent.target_value[:, :1], proposed_target_value[:, :1],
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            recent.target_value[:, 1:] - recent.source_value[:, 1:],
            first_target[:, 1:] - source[:, 1:],
            rtol=0, atol=0,
        )
        self.assertEqual(
            diagnostics["mutable_target_payload_written"].item(), 1.0
        )

    def test_persistent_residual_upsert_does_not_clone_one_payload(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=2, max_tokens_per_frame=2,
            min_write_confidence=0.5, transactional_dense_recent=True,
            token_atomic_dense_recent=True, persistent_residual_upsert=True,
            min_lineage_similarity=0.8,
        )
        source = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(source + 2.0, source + 2.0),
            write_confidence=torch.tensor([[1.0, 0.0]]),
            retention_confidence=torch.tensor([[1.0, 0.0]]),
        )
        # Both current source tokens resemble the sole trusted old address.
        # Mutual nearest-neighbour transport must retain at most one of them.
        ambiguous_source = torch.tensor(
            [[[[1.0, 0.0]], [[0.99, 0.01]]]]
        )
        diagnostics = history.commit(
            source_kv_cache=make_cache(ambiguous_source, ambiguous_source),
            target_kv_cache=make_cache(
                ambiguous_source + 100.0, ambiguous_source + 100.0
            ),
            write_confidence=torch.zeros(1, 2),
            retention_confidence=torch.ones(1, 2),
        )[0]
        self.assertEqual(
            diagnostics["persistent_retained_support"].sum().item(), 1
        )
        self.assertEqual(
            history.read()[0].recent.payload_support.sum().item(), 1
        )

    def test_persistent_residual_upsert_drops_match_outside_owner(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=2, max_tokens_per_frame=2,
            min_write_confidence=0.5, transactional_dense_recent=True,
            token_atomic_dense_recent=True, persistent_residual_upsert=True,
            min_lineage_similarity=0.8,
        )
        source = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(source + 2.0, source + 2.0),
            write_confidence=torch.ones(1, 2),
            retention_confidence=torch.ones(1, 2),
        )
        diagnostics = history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(source + 100.0, source + 100.0),
            write_confidence=torch.zeros(1, 2),
            retention_confidence=torch.tensor([[1.0, 0.0]]),
        )[0]
        self.assertEqual(
            diagnostics["persistent_retained_support"].tolist(),
            [[True, False]],
        )
        self.assertEqual(
            history.read()[0].recent.payload_support.tolist(),
            [[True, False]],
        )

    def test_last_trusted_lineage_rejects_source_regressed_direct_write(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=2, max_tokens_per_frame=2,
            min_write_confidence=0.5, transactional_dense_recent=True,
            token_atomic_dense_recent=True, persistent_residual_upsert=True,
            last_trusted_residual_lineage=True,
            residual_update_min_cosine=0.5,
            residual_update_min_magnitude_ratio=0.9,
            min_lineage_similarity=0.8,
        )
        source = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        trusted_target = source + torch.tensor(
            [[[[2.0, 1.0]], [[1.0, 2.0]]]]
        )
        history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(trusted_target, trusted_target),
            write_confidence=torch.ones(1, 2),
            retention_confidence=torch.ones(1, 2),
        )

        # The current block is confidently writable but has collapsed back
        # to source appearance. It must not overwrite the trusted residual.
        diagnostics = history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(source, source),
            write_confidence=torch.ones(1, 2),
            retention_confidence=torch.ones(1, 2),
        )[0]
        recent = history.read()[0].recent
        self.assertEqual(
            diagnostics["persistent_guarded_update_support"].tolist(),
            [[True, True]],
        )
        self.assertEqual(
            diagnostics["persistent_direct_support"].tolist(),
            [[False, False]],
        )
        self.assertEqual(
            diagnostics["persistent_retained_support"].tolist(),
            [[True, True]],
        )
        torch.testing.assert_close(
            recent.target_value - recent.source_value,
            trusted_target - source, rtol=0, atol=0,
        )

    def test_last_trusted_lineage_accepts_consistent_direct_write(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=1, max_tokens_per_frame=1,
            min_write_confidence=0.5, transactional_dense_recent=True,
            token_atomic_dense_recent=True, persistent_residual_upsert=True,
            last_trusted_residual_lineage=True,
            residual_update_min_cosine=0.5,
            residual_update_min_magnitude_ratio=0.9,
            min_lineage_similarity=0.8,
        )
        source = torch.tensor([[[[1.0, 0.0]]]])
        history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(source + 2.0, source + 2.0),
            write_confidence=torch.ones(1, 1),
            retention_confidence=torch.ones(1, 1),
        )
        improved = source + 3.0
        diagnostics = history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(improved, improved),
            write_confidence=torch.ones(1, 1),
            retention_confidence=torch.ones(1, 1),
        )[0]
        self.assertEqual(
            diagnostics["persistent_direct_support"].tolist(), [[True]]
        )
        self.assertFalse(
            bool(diagnostics["persistent_guarded_update_support"].any())
        )
        torch.testing.assert_close(
            history.read()[0].recent.target_value, improved, rtol=0, atol=0
        )

    def test_last_trusted_assignment_recovers_non_mutual_second_match(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=2, max_tokens_per_frame=2,
            min_write_confidence=0.5, transactional_dense_recent=True,
            token_atomic_dense_recent=True, persistent_residual_upsert=True,
            last_trusted_residual_lineage=True,
            min_lineage_similarity=0.8,
        )
        previous_source = torch.tensor(
            [[[[1.0, 0.0]], [[0.99, 0.1]]]]
        )
        history.commit(
            source_kv_cache=make_cache(previous_source, previous_source),
            target_kv_cache=make_cache(
                previous_source + 2.0, previous_source + 2.0
            ),
            write_confidence=torch.ones(1, 2),
            retention_confidence=torch.ones(1, 2),
        )
        current_source = torch.tensor(
            [[[[1.0, 0.0]], [[0.98, 0.2]]]]
        )
        diagnostics = history.commit(
            source_kv_cache=make_cache(current_source, current_source),
            target_kv_cache=make_cache(
                current_source + 100.0, current_source + 100.0
            ),
            write_confidence=torch.zeros(1, 2),
            retention_confidence=torch.ones(1, 2),
        )[0]
        # One-to-one top-k assignment preserves both lineages without
        # allowing either old payload to clone.
        self.assertEqual(
            diagnostics["persistent_retained_support"].sum().item(), 2
        )

    def test_residual_rebased_attention_injects_only_appearance_delta(self):
        native = torch.full((1, 1, 1, 2), 10.0)
        query = torch.ones_like(native)
        source_key = torch.tensor([[[[1.0, 0.0]]]])
        source_value = torch.tensor([[[[20.0, 30.0]]]])
        target_value = source_value + torch.tensor([[[[2.0, -3.0]]]])
        output, diagnostics = source_addressed_native_history_attention(
            native, query,
            torch.ones_like(native), torch.full_like(native, 999.0),
            torch.ones_like(native), target_value,
            torch.ones(1, 1, dtype=torch.bool),
            torch.ones_like(native), torch.full_like(native, -1000.0),
            source_key, torch.tensor([[[[-1.0, 0.0]]]]),
            torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1),
            canonical_source_value=torch.zeros_like(native),
            recent_source_value=source_value,
            recent_source_key=source_key,
            recent_payload_support=torch.ones(1, 1, dtype=torch.bool),
            residual_rebased_payload=True,
            topk=1, min_similarity=0.8, min_request=0.5,
            consistent_transaction=True, entry_bridge=True,
            motion_owner_dense_read=True, entry_query_count=1,
            entry_bridge_strength=1.0,
        )
        torch.testing.assert_close(
            output, torch.tensor([[[[12.0, 7.0]]]]), rtol=0, atol=1e-6
        )
        self.assertTrue(
            bool(diagnostics["residual_rebased_payload"].bool().all())
        )

    def test_last_trusted_read_uses_joint_evidence_geometric_mean(self):
        native = torch.zeros(1, 1, 1, 2)
        query = torch.ones_like(native)
        source_key = torch.tensor([[[[1.0, 0.0]]]])
        source_value = torch.zeros_like(source_key)
        target_value = torch.tensor([[[[2.0, 0.0]]]])
        common = dict(
            native_output=native, target_query=query,
            canonical_target_key=torch.tensor([[[[-1.0, 0.0]]]]),
            canonical_target_value=torch.zeros_like(native),
            recent_target_key=torch.ones_like(native),
            recent_target_value=target_value,
            recent_support=torch.ones(1, 1, dtype=torch.bool),
            current_target_key=torch.ones_like(native),
            current_target_value=torch.zeros_like(native),
            current_source_key=torch.tensor([[[[0.8, 0.6]]]]),
            canonical_source_key=torch.tensor([[[[-1.0, 0.0]]]]),
            canonical_support=torch.ones(1, 1, dtype=torch.bool),
            query_request=torch.full((1, 1), 0.25),
            canonical_source_value=torch.zeros_like(native),
            recent_source_value=source_value,
            recent_source_key=source_key,
            recent_payload_support=torch.ones(1, 1, dtype=torch.bool),
            residual_rebased_payload=True,
            topk=1, min_similarity=0.5, min_request=0.5,
            consistent_transaction=True, entry_bridge=True,
            motion_owner_dense_read=True, entry_query_count=1,
            entry_bridge_strength=1.0,
        )
        _, baseline = source_addressed_native_history_attention(**common)
        _, protected = source_addressed_native_history_attention(
            **common, last_trusted_appearance=True
        )
        self.assertGreater(
            protected["read_strength"].item(),
            baseline["read_strength"].item(),
        )
        self.assertAlmostEqual(
            protected["read_strength"].item(),
            baseline["read_strength"].item() ** 0.5,
            places=6,
        )

    def test_token_atomic_attention_abstains_without_payload_authority(self):
        native = torch.tensor([[[[7.0, 8.0]], [[9.0, 10.0]]]])
        query = torch.ones_like(native)
        recent_source = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        current_source = recent_source.clone()
        recent_key = torch.ones(1, 2, 1, 2)
        recent_value = torch.full_like(recent_key, 3.0)

        output, diagnostics = source_addressed_native_history_attention(
            native, query,
            torch.ones(1, 1, 1, 2),
            torch.full((1, 1, 1, 2), 9.0),
            recent_key, recent_value,
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones_like(native), torch.full_like(native, -1000.0),
            current_source, torch.tensor([[[[-1.0, 0.0]]]]),
            torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 2),
            recent_source_key=recent_source,
            recent_payload_support=torch.tensor([[True, False]]),
            topk=2, min_similarity=0.8, min_request=0.5,
            consistent_transaction=True, entry_bridge=True,
            motion_owner_dense_read=True, entry_query_count=2,
            entry_bridge_strength=1.0,
        )

        torch.testing.assert_close(
            output[:, :1], torch.full_like(output[:, :1], 3.0),
            rtol=0, atol=1e-6,
        )
        torch.testing.assert_close(
            output[:, 1:], native[:, 1:], rtol=0, atol=0
        )
        self.assertTrue(bool(diagnostics["recent_entry_admitted"][0, 0]))
        self.assertFalse(bool(diagnostics["admitted"][0, 1]))
        self.assertFalse(
            bool(diagnostics["canonical_fallback_admitted"][0, 1])
        )

    def test_token_atomic_addressing_does_not_reroute_around_denied_payload(self):
        native = torch.tensor([[[[7.0, 8.0]]]])
        query = torch.ones_like(native)
        current_source = torch.tensor([[[[1.0, 0.0]]]])
        recent_source = torch.tensor(
            [[[[1.0, 0.0]], [[0.9, 0.4358899]]]]
        )
        output, diagnostics = source_addressed_native_history_attention(
            native, query,
            torch.tensor([[[[-1.0, 0.0]]]]),
            torch.full((1, 1, 1, 2), 9.0),
            torch.ones(1, 2, 1, 2),
            torch.full((1, 2, 1, 2), 3.0),
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones_like(native), torch.full_like(native, -1000.0),
            current_source, torch.tensor([[[[-1.0, 0.0]]]]),
            torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1),
            recent_source_key=recent_source,
            recent_payload_support=torch.tensor([[False, True]]),
            topk=2, min_similarity=0.8, min_request=0.5,
            consistent_transaction=True, entry_bridge=True,
            motion_owner_dense_read=True, entry_query_count=1,
            entry_bridge_strength=1.0,
        )
        # The exact source address is payload-denied. The implementation must
        # abstain rather than hide that address and redirect the query to the
        # slightly less similar authorized token.
        torch.testing.assert_close(output, native, rtol=0, atol=0)
        self.assertFalse(bool(diagnostics["admitted"].any()))

    def test_last_trusted_read_uses_nearest_authorized_topk_payload(self):
        native = torch.tensor([[[[7.0, 8.0]]]])
        current_source = torch.tensor([[[[1.0, 0.0]]]])
        recent_source = torch.tensor(
            [[[[1.0, 0.0]], [[0.9, 0.4358899]]]]
        )
        output, diagnostics = source_addressed_native_history_attention(
            native, torch.ones_like(native),
            torch.tensor([[[[-1.0, 0.0]]]]),
            torch.zeros_like(native),
            torch.ones(1, 2, 1, 2),
            torch.tensor([[[[100.0, 100.0]], [[2.0, 3.0]]]]),
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones_like(native), torch.zeros_like(native),
            current_source, torch.tensor([[[[-1.0, 0.0]]]]),
            torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1),
            canonical_source_value=torch.zeros_like(native),
            recent_source_value=torch.zeros(1, 2, 1, 2),
            recent_source_key=recent_source,
            recent_payload_support=torch.tensor([[False, True]]),
            residual_rebased_payload=True,
            last_trusted_appearance=True,
            topk=2, min_similarity=0.8, min_request=0.5,
            consistent_transaction=True, entry_bridge=True,
            motion_owner_dense_read=True, entry_query_count=1,
            entry_bridge_strength=1.0,
        )
        self.assertTrue(bool(diagnostics["admitted"].all()))
        self.assertGreater(diagnostics["output_delta"].item(), 0.0)
        self.assertFalse(torch.equal(output, native))

    def test_dense_recent_transaction_holds_source_regressed_proposal(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=2, max_tokens_per_frame=2,
            min_write_confidence=0.5, transactional_dense_recent=True,
            dense_recent_min_residual_consensus=0.05,
        )
        source = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        target = source + torch.tensor([[[[0.0, 2.0]]]])
        history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(target, target),
            write_confidence=torch.ones(1, 2),
        )
        trusted = history.read()[0].recent.target_value.clone()
        diagnostics = history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(source, source),
            write_confidence=torch.ones(1, 2),
        )[0]
        torch.testing.assert_close(
            history.read()[0].recent.target_value, trusted, rtol=0, atol=0
        )
        self.assertEqual(diagnostics["dense_recent_accepted"].item(), 0.0)
        self.assertEqual(diagnostics["recent_held_tokens"].item(), 2.0)

    def test_entry_bridge_changes_only_first_frame_and_prefers_recent(self):
        native = torch.zeros(1, 4, 1, 2)
        query = torch.ones_like(native)
        canonical_key = torch.ones(1, 1, 1, 2)
        canonical_value = torch.full_like(canonical_key, 9.0)
        recent_key = torch.ones(1, 2, 1, 2)
        recent_value = torch.full_like(recent_key, 3.0)
        recent_source = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        current_source = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]],
              [[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        output, diagnostics = source_addressed_native_history_attention(
            native, query, canonical_key, canonical_value,
            recent_key, recent_value, torch.ones(1, 2, dtype=torch.bool),
            torch.ones_like(native), torch.full_like(native, -1000.0),
            current_source, torch.tensor([[[[0.0, -1.0]]]]),
            torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 4),
            recent_source_key=recent_source, topk=1, min_similarity=0.8,
            min_request=0.5, payload_blend_strength=0.2,
            consistent_transaction=True, entry_bridge=True,
            entry_query_count=2, entry_bridge_strength=1.0,
        )
        torch.testing.assert_close(
            output[:, :2], torch.full_like(output[:, :2], 3.0),
            rtol=0, atol=1e-6,
        )
        torch.testing.assert_close(output[:, 2:], native[:, 2:], rtol=0, atol=0)
        self.assertTrue(bool(diagnostics["recent_entry_admitted"][:, :2].all()))
        self.assertFalse(bool(diagnostics["canonical_fallback_admitted"].any()))

    def test_motion_owner_dense_read_covers_later_frames_only_on_requests(self):
        # Three latent frames with two spatial tokens each.  The owner asks
        # for the first token in every frame; the paired background token must
        # remain an exact native fallback even though its source address also
        # matches recent memory.
        native = torch.randn(1, 6, 1, 2)
        query = torch.ones_like(native)
        recent_source = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        current_source = recent_source.repeat(1, 3, 1, 1)
        recent_key = torch.ones(1, 2, 1, 2)
        recent_value = torch.full_like(recent_key, 3.0)
        request = torch.tensor([[1.0, 0.0, 0.5, 0.0, 0.25, 0.0]])

        output, diagnostics = source_addressed_native_history_attention(
            native, query,
            torch.ones(1, 1, 1, 2),
            torch.full((1, 1, 1, 2), 9.0),
            recent_key, recent_value,
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones_like(native), torch.full_like(native, -1000.0),
            current_source, torch.tensor([[[[-1.0, 0.0]]]]),
            torch.ones(1, 1, dtype=torch.bool), request,
            recent_source_key=recent_source,
            topk=1, min_similarity=0.8, min_request=0.5,
            consistent_transaction=True, entry_bridge=True,
            motion_owner_dense_read=True, entry_query_count=2,
            entry_bridge_strength=1.0,
        )

        owner_indices = torch.tensor([0, 2, 4])
        background_indices = torch.tensor([1, 3, 5])
        self.assertTrue(
            bool(diagnostics["recent_entry_admitted"][0, owner_indices].all())
        )
        self.assertTrue(bool(diagnostics["read_scope"].all()))
        torch.testing.assert_close(
            diagnostics["request_support"][0],
            (request[0] > 0.0).float(),
            rtol=0, atol=0,
        )
        self.assertTrue(
            bool((output[0, owner_indices] != native[0, owner_indices]).any(dim=-1).all())
        )
        torch.testing.assert_close(
            output[0, background_indices], native[0, background_indices],
            rtol=0, atol=0,
        )

    def test_dual_evidence_rejects_inconsistent_recent_payload(self):
        native = torch.zeros(1, 2, 1, 2)
        query = torch.ones_like(native)
        source_key = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]])
        canonical_key = torch.ones(1, 1, 1, 2)
        canonical_source_value = torch.tensor([[[[1.0, 1.0]]]])
        canonical_target_value = torch.tensor([[[[3.0, 1.0]]]])
        recent_key = torch.ones(1, 1, 1, 2)
        recent_source_value = torch.tensor([[[[1.0, 1.0]]]])
        # The recent residual [0, 2] is orthogonal to the immutable canonical
        # residual [2, 0], despite a perfect clean-source address match.
        recent_target_value = torch.tensor([[[[1.0, 3.0]]]])
        request = torch.tensor([[1.0, 0.0]])

        output, diagnostics = source_addressed_native_history_attention(
            native, query, canonical_key, canonical_target_value,
            recent_key, recent_target_value,
            torch.ones(1, 1, dtype=torch.bool),
            torch.ones_like(native), torch.full_like(native, -1000.0),
            source_key, source_key[:, :1],
            torch.ones(1, 1, dtype=torch.bool), request,
            canonical_source_value=canonical_source_value,
            recent_source_value=recent_source_value,
            recent_source_key=source_key[:, :1],
            topk=1, min_similarity=0.8, min_request=0.5,
            consistent_transaction=True, entry_bridge=True,
            motion_owner_dense_read=True, entry_query_count=1,
            entry_bridge_strength=1.0,
            dual_evidence_arbitration=True,
            min_payload_consistency=0.15,
        )

        torch.testing.assert_close(
            output[:, :1], canonical_target_value, rtol=0, atol=1e-6
        )
        torch.testing.assert_close(
            output[:, 1:], native[:, 1:], rtol=0, atol=0
        )
        self.assertTrue(bool(diagnostics["recent_payload_rejected"][0, 0]))
        self.assertTrue(bool(diagnostics["canonical_fallback_admitted"][0, 0]))
        self.assertEqual(diagnostics["recent_payload_trust"][0, 0].item(), 0.0)

    def test_dual_evidence_convexly_trusts_consistent_recent_payload(self):
        native = torch.zeros(1, 1, 1, 2)
        query = torch.ones_like(native)
        source_key = torch.tensor([[[[1.0, 0.0]]]])
        canonical_key = torch.ones_like(native)
        canonical_source_value = torch.tensor([[[[1.0, 1.0]]]])
        canonical_target_value = torch.tensor([[[[3.0, 1.0]]]])
        recent_source_value = torch.tensor([[[[1.0, 1.0]]]])
        # Same residual direction at twice the magnitude gives a scale
        # agreement/trust of 0.5.  The output must remain on the segment
        # between immutable canonical [3, 1] and recent [5, 1].
        recent_target_value = torch.tensor([[[[5.0, 1.0]]]])

        output, diagnostics = source_addressed_native_history_attention(
            native, query, canonical_key, canonical_target_value,
            canonical_key, recent_target_value,
            torch.ones(1, 1, dtype=torch.bool),
            canonical_key, torch.full_like(native, -1000.0),
            source_key, source_key,
            torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1),
            canonical_source_value=canonical_source_value,
            recent_source_value=recent_source_value,
            recent_source_key=source_key,
            topk=1, min_similarity=0.8, min_request=0.5,
            consistent_transaction=True, entry_bridge=True,
            motion_owner_dense_read=True, entry_query_count=1,
            entry_bridge_strength=1.0,
            dual_evidence_arbitration=True,
            min_payload_consistency=0.15,
        )

        torch.testing.assert_close(
            output, torch.tensor([[[[4.0, 1.0]]]]), rtol=0, atol=1e-6
        )
        torch.testing.assert_close(
            diagnostics["recent_payload_trust"],
            torch.full((1, 1), 0.5), rtol=0, atol=1e-6,
        )
        self.assertFalse(bool(diagnostics["recent_payload_rejected"].any()))
        self.assertTrue(bool(diagnostics["recent_entry_admitted"].all()))

    def test_first_read_marks_same_commit_temporal_alias(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=2, max_tokens_per_frame=1,
            min_write_confidence=0.5,
        )
        key = torch.arange(24, dtype=torch.float32).reshape(1, 6, 1, 4)
        history.commit(
            source_kv_cache=make_cache(key, key),
            target_kv_cache=make_cache(key + 10, key + 20),
            write_confidence=torch.ones(1, 6),
        )
        first_read = history.read()[0]
        self.assertTrue(first_read.recent_shares_canonical_time)
        self.assertEqual(first_read.temporal_origins(), (3, 6))
        self.assertEqual(
            first_read.temporal_origins(
                coalesce_bootstrap_alias=True
            ),
            (0, 3),
        )

        history.commit(
            source_kv_cache=make_cache(key + 1, key + 1),
            target_kv_cache=make_cache(key + 30, key + 40),
            write_confidence=torch.ones(1, 6),
        )
        second_read = history.read()[0]
        self.assertFalse(second_read.recent_shares_canonical_time)
        self.assertEqual(
            second_read.temporal_origins(
                coalesce_bootstrap_alias=True
            ),
            (3, 6),
        )

    def test_unsupported_queries_are_exact_native_fallback(self):
        native = torch.randn(1, 3, 1, 4)
        query = torch.randn_like(native)
        canonical_key = torch.randn(1, 2, 1, 4)
        canonical_value = torch.randn_like(canonical_key)
        current_key = torch.randn_like(native)
        current_value = torch.randn_like(native)
        current_source = torch.randn_like(native)
        canonical_source = torch.randn_like(canonical_key)
        output, diagnostics = source_addressed_native_history_attention(
            native, query, canonical_key, canonical_value,
            canonical_key[:, :0], canonical_value[:, :0],
            torch.empty(1, 0, dtype=torch.bool),
            current_key, current_value, current_source, canonical_source,
            torch.ones(1, 2, dtype=torch.bool),
            torch.zeros(1, 3),
        )
        torch.testing.assert_close(output, native, rtol=0, atol=0)
        self.assertFalse(bool(diagnostics["admitted"].any()))

    def test_source_match_changes_only_admitted_query(self):
        native = torch.zeros(1, 2, 1, 2)
        query = torch.ones_like(native)
        canonical_source = torch.tensor([[[[1.0, 0.0]]]])
        current_source = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        canonical_key = torch.ones(1, 1, 1, 2)
        canonical_value = torch.full_like(canonical_key, 4.0)
        current_key = torch.ones_like(native)
        current_value = torch.ones_like(native)
        output, diagnostics = source_addressed_native_history_attention(
            native, query, canonical_key, canonical_value,
            canonical_key[:, :0], canonical_value[:, :0],
            torch.empty(1, 0, dtype=torch.bool),
            current_key, current_value, current_source, canonical_source,
            torch.ones(1, 1, dtype=torch.bool),
            torch.ones(1, 2), min_similarity=0.8, topk=1,
        )
        self.assertTrue(bool(diagnostics["admitted"][0, 0]))
        self.assertFalse(bool(diagnostics["admitted"][0, 1]))
        self.assertFalse(torch.equal(output[:, :1], native[:, :1]))
        torch.testing.assert_close(
            output[:, 1:], native[:, 1:], rtol=0, atol=0
        )

    def test_consistent_transaction_uses_soft_read_and_compact_payload(self):
        native = torch.zeros(1, 2, 1, 2)
        query = torch.ones_like(native)
        canonical_source = torch.tensor([[[[1.0, 0.0]]]])
        current_source = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        canonical_key = torch.ones(1, 1, 1, 2)
        canonical_value = torch.full_like(canonical_key, 4.0)
        recent_key = torch.ones_like(canonical_key)
        recent_value = torch.full_like(recent_key, 4.0)
        output, diagnostics = source_addressed_native_history_attention(
            native, query, canonical_key, canonical_value,
            recent_key, recent_value,
            torch.ones(1, 1, dtype=torch.bool),
            torch.ones_like(native), torch.full_like(native, -1000.0),
            current_source, canonical_source,
            torch.ones(1, 1, dtype=torch.bool),
            torch.tensor([[0.25, 0.0]]), min_request=0.5,
            min_similarity=0.8, topk=1,
            recent_source_key=canonical_source,
            payload_blend_strength=0.5,
            consistent_transaction=True,
        )
        self.assertTrue(bool(diagnostics["admitted"][0, 0]))
        self.assertFalse(bool(diagnostics["admitted"][0, 1]))
        self.assertGreater(diagnostics["read_strength"][0, 0], 0.0)
        self.assertLess(diagnostics["read_strength"][0, 0], 1.0)
        # Current target value is deliberately extreme; the transactional
        # branch must use only canonical/recent committed payload.
        self.assertGreater(output[0, 0].mean().item(), 0.0)
        self.assertLess(output[0, 0].mean().item(), 4.0)
        torch.testing.assert_close(output[:, 1:], native[:, 1:])

    def test_payload_invariant_lineage_reads_only_canonical_target_value(self):
        native = torch.zeros(1, 1, 1, 2)
        query = torch.ones_like(native)
        canonical_source = torch.tensor([[[[0.0, 1.0]]]])
        current_source = torch.tensor([[[[1.0, 0.0]]]])
        lineage_source = current_source.clone()
        canonical_key = torch.ones(1, 1, 1, 2)
        canonical_value = torch.full_like(canonical_key, 4.0)
        current_key = torch.ones_like(native)
        current_value = torch.full_like(native, -20.0)
        recent_key = torch.ones_like(canonical_key)
        recent_support = torch.ones(1, 1, dtype=torch.bool)

        def run(drifted_recent_value):
            return source_addressed_native_history_attention(
                native, query, canonical_key, canonical_value,
                recent_key, drifted_recent_value, recent_support,
                current_key, current_value, current_source,
                canonical_source, torch.ones(1, 1, dtype=torch.bool),
                torch.ones(1, 1), topk=1, min_similarity=0.8,
                payload_invariant_lineage=True,
                recent_source_key=lineage_source,
                recent_lineage_index=torch.zeros(1, 1, dtype=torch.long),
                recent_lineage_support=torch.ones(1, 1, dtype=torch.bool),
                recent_lineage_confidence=torch.ones(1, 1),
                payload_blend_strength=0.5,
            )

        first, diagnostics = run(torch.full_like(recent_key, -10000.0))
        second, _ = run(torch.full_like(recent_key, 10000.0))
        self.assertTrue(bool(diagnostics["admitted"].all()))
        self.assertTrue(bool(diagnostics["lineage_admitted"].all()))
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(
            first, torch.full_like(first, 2.0), rtol=0, atol=1e-6
        )

    def test_canonical_appearance_transaction_adds_only_value_residual(self):
        native = torch.full((1, 1, 1, 2), 10.0)
        query = torch.ones_like(native)
        source_key = torch.tensor([[[[1.0, 0.0]]]])
        canonical_key = torch.ones_like(native)
        canonical_source_value = torch.ones_like(native)
        canonical_target_value = torch.full_like(native, 4.0)
        recent_key = torch.ones_like(native)

        def run(recent_value):
            return source_addressed_native_history_attention(
                native, query, canonical_key, canonical_target_value,
                recent_key, recent_value,
                torch.ones(1, 1, dtype=torch.bool),
                torch.ones_like(native), torch.full_like(native, -1000.0),
                source_key, source_key,
                torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1),
                canonical_source_value=canonical_source_value,
                topk=1, min_similarity=0.8,
                payload_invariant_lineage=True,
                recent_source_key=source_key,
                recent_lineage_index=torch.zeros(1, 1, dtype=torch.long),
                recent_lineage_support=torch.ones(1, 1, dtype=torch.bool),
                recent_lineage_confidence=torch.ones(1, 1),
                payload_blend_strength=0.5,
                consistent_transaction=True,
            )

        first, diagnostics = run(torch.full_like(recent_key, -10000.0))
        second, _ = run(torch.full_like(recent_key, 10000.0))
        # Native geometry (10) plus half the immutable appearance residual
        # (target 4 - source 1), independent of generated recent payload.
        torch.testing.assert_close(
            first, torch.full_like(first, 11.5), rtol=0, atol=1e-6
        )
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(
            diagnostics["read_strength"], torch.ones(1, 1),
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            diagnostics["applied_read_strength"],
            torch.full((1, 1), 0.5), rtol=0, atol=0,
        )
        torch.testing.assert_close(
            diagnostics["canonical_appearance_delta"],
            torch.full((1, 1), 3.0), rtol=0, atol=1e-6,
        )
        self.assertEqual(
            diagnostics["mutable_target_payload_enabled"].sum().item(),
            0.0,
        )

    def test_invariant_commit_reports_rejected_mutable_payload_drift(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=2, max_tokens_per_frame=2,
            min_write_confidence=0.5, payload_invariant_lineage=True,
            min_lineage_similarity=0.2,
        )
        source = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        canonical_target = source + 2.0
        history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(
                canonical_target, canonical_target
            ),
            write_confidence=torch.ones(1, 2),
            lineage_confidence=torch.ones(1, 2),
        )
        proposed_source_like_target = source.clone()
        diagnostics = history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(
                proposed_source_like_target, proposed_source_like_target
            ),
            write_confidence=torch.ones(1, 2),
            lineage_confidence=torch.ones(1, 2),
        )[0]
        self.assertEqual(
            diagnostics["mutable_target_payload_written"].item(), 0.0
        )
        self.assertGreater(
            diagnostics["candidate_target_source_similarity"].item(),
            diagnostics["candidate_target_canonical_similarity"].item(),
        )
        self.assertIsNone(history.read()[0].recent)

    def test_payload_invariant_lineage_abstention_is_exact_native(self):
        native = torch.randn(1, 1, 1, 2)
        key = torch.ones_like(native)
        source = torch.tensor([[[[1.0, 0.0]]]])
        output, diagnostics = source_addressed_native_history_attention(
            native, key, key, torch.full_like(key, 7.0),
            key[:, :0], key[:, :0],
            torch.empty(1, 0, dtype=torch.bool),
            key, key, source, source,
            torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1),
            payload_invariant_lineage=True,
            recent_source_key=source,
            recent_lineage_index=torch.zeros(1, 1, dtype=torch.long),
            recent_lineage_support=torch.ones(1, 1, dtype=torch.bool),
            recent_lineage_confidence=torch.ones(1, 1),
        )
        self.assertFalse(bool(diagnostics["admitted"].any()))
        torch.testing.assert_close(output, native, rtol=0, atol=0)

    def test_payload_invariant_lineage_confidence_scales_soft_read(self):
        native = torch.zeros(1, 1, 1, 2)
        key = torch.ones_like(native)
        current_source = torch.tensor([[[[1.0, 0.0]]]])
        canonical_source = torch.tensor([[[[0.0, 1.0]]]])
        output, diagnostics = source_addressed_native_history_attention(
            native, key, key, torch.full_like(key, 4.0),
            key[:, :0], key[:, :0],
            torch.empty(1, 0, dtype=torch.bool),
            key, key, current_source, canonical_source,
            torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1),
            payload_invariant_lineage=True,
            recent_source_key=current_source,
            recent_lineage_index=torch.zeros(1, 1, dtype=torch.long),
            recent_lineage_support=torch.ones(1, 1, dtype=torch.bool),
            recent_lineage_confidence=torch.full((1, 1), 0.25),
            payload_blend_strength=0.5,
            min_similarity=0.8,
            topk=1,
        )
        torch.testing.assert_close(
            output, torch.full_like(output, 0.5), rtol=0, atol=1e-6
        )
        torch.testing.assert_close(
            diagnostics["lineage_confidence"],
            torch.full((1, 1), 0.25),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            diagnostics["read_strength"],
            torch.full((1, 1), 0.125),
            rtol=0,
            atol=0,
        )

    def test_source_part_soft_bias_keeps_recent_and_current_support(self):
        native = torch.zeros(1, 1, 1, 4)
        query = torch.ones_like(native)
        current_key = torch.ones_like(native)
        canonical_key = torch.ones_like(native)
        recent_key = torch.ones_like(native)
        current_source_key = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
        canonical_source_key = current_source_key.clone()
        body_part = torch.tensor([[[[1.0, -1.0, 1.0, -1.0]]]])
        cap_part = torch.tensor([[[[1.0, 1.0, -1.0, -1.0]]]])

        baseline, baseline_diagnostics = (
            source_addressed_native_history_attention(
                native, query,
                canonical_key, torch.full_like(canonical_key, 10.0),
                recent_key, torch.full_like(recent_key, 2.0),
                torch.ones(1, 1, dtype=torch.bool),
                current_key, torch.full_like(current_key, 2.0),
                current_source_key, canonical_source_key,
                torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1),
                topk=1,
            )
        )
        output, diagnostics = source_addressed_native_history_attention(
            native, query,
            canonical_key, torch.full_like(canonical_key, 10.0),
            recent_key, torch.full_like(recent_key, 2.0),
            torch.ones(1, 1, dtype=torch.bool),
            current_key, torch.full_like(current_key, 2.0),
            current_source_key, canonical_source_key,
            torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1),
            current_source_value=body_part,
            canonical_source_value=cap_part,
            recent_source_value=body_part,
            source_part_consistency=True,
            min_part_similarity=0.8,
            part_similarity_margin=0.05,
            part_bias_strength=0.5,
            part_refinement_ratio=0.25,
            topk=1,
        )
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(bool(diagnostics["admitted"].all()))
        torch.testing.assert_close(
            diagnostics["admitted"],
            baseline_diagnostics["admitted"],
            rtol=0,
            atol=0,
        )
        # A mismatched canonical token may be de-emphasized, but the dense
        # recent/current path remains present instead of being hard-pruned.
        self.assertLess(output.mean().item(), baseline.mean().item())
        self.assertGreater(output.mean().item(), 2.0)

    def test_source_part_refinement_is_bounded_by_role_fixed_read(self):
        native = torch.randn(1, 1, 1, 4)
        query = torch.ones_like(native)
        key = torch.ones_like(native)
        source_key = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
        body_part = torch.tensor([[[[1.0, -1.0, 1.0, -1.0]]]])
        cap_part = torch.tensor([[[[1.0, 1.0, -1.0, -1.0]]]])
        baseline, baseline_diagnostics = (
            source_addressed_native_history_attention(
                native, query, key, torch.full_like(key, 10.0),
                key, torch.full_like(key, 10.0),
                torch.ones(1, 1, dtype=torch.bool),
                key, torch.full_like(key, 2.0), source_key, source_key,
                torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1),
                topk=1,
            )
        )
        ratio = 0.25
        output, diagnostics = source_addressed_native_history_attention(
            native, query, key, torch.full_like(key, 10.0),
            key, torch.full_like(key, 10.0),
            torch.ones(1, 1, dtype=torch.bool),
            key, torch.full_like(key, 2.0), source_key, source_key,
            torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1),
            current_source_value=body_part,
            canonical_source_value=cap_part,
            recent_source_value=cap_part,
            source_part_consistency=True,
            min_part_similarity=0.8,
            part_similarity_margin=0.05,
            part_bias_strength=0.5,
            part_refinement_ratio=ratio,
            topk=1,
        )
        torch.testing.assert_close(
            diagnostics["admitted"],
            baseline_diagnostics["admitted"],
            rtol=0,
            atol=0,
        )
        baseline_rms = (baseline.float() - native.float()).square().mean().sqrt()
        refinement_rms = (output.float() - baseline.float()).square().mean().sqrt()
        self.assertLessEqual(
            refinement_rms.item(),
            ratio * baseline_rms.item() + 1e-6,
        )

    def test_zero_part_refinement_exactly_recovers_role_fixed_read(self):
        native = torch.randn(1, 2, 1, 4)
        query = torch.randn_like(native)
        key = torch.randn_like(native)
        value = torch.randn_like(native)
        source_key = torch.randn_like(native)
        source_value = torch.randn_like(native)
        arguments = (
            native, query, key, value, key, value,
            torch.ones(1, 2, dtype=torch.bool), key, value,
            source_key, source_key, torch.ones(1, 2, dtype=torch.bool),
            torch.ones(1, 2),
        )
        baseline, _ = source_addressed_native_history_attention(
            *arguments, topk=2, min_similarity=-0.99
        )
        output, _ = source_addressed_native_history_attention(
            *arguments,
            current_source_value=source_value,
            canonical_source_value=source_value,
            recent_source_value=source_value,
            source_part_consistency=True,
            part_refinement_ratio=0.0,
            topk=2,
            min_similarity=-0.99,
        )
        torch.testing.assert_close(output, baseline, rtol=0, atol=0)

    def test_flow_indexed_attention_ignores_target_keys(self):
        native = torch.tensor([[[[10.0, 20.0]], [[30.0, 40.0]]]])
        residual = torch.tensor([[[[2.0, -3.0]], [[4.0, -5.0]]]])
        common = dict(
            native_output=native, target_query=torch.randn_like(native),
            canonical_target_key=torch.randn(1, 1, 1, 2),
            canonical_target_value=torch.randn(1, 1, 1, 2),
            recent_target_key=torch.randn_like(native),
            recent_target_value=torch.randn_like(native),
            recent_support=torch.ones(1, 2, dtype=torch.bool),
            recent_payload_support=torch.ones(1, 2, dtype=torch.bool),
            current_target_key=torch.randn_like(native),
            current_target_value=torch.randn_like(native),
            current_source_key=torch.randn_like(native),
            canonical_source_key=torch.randn(1, 1, 1, 2),
            canonical_support=torch.ones(1, 1, dtype=torch.bool),
            query_request=torch.ones(1, 2),
            recent_source_value=torch.zeros_like(native),
            residual_rebased_payload=True,
            last_trusted_appearance=True,
            consistent_transaction=True, entry_bridge=True,
            motion_owner_dense_read=True, entry_query_count=2,
            flow_indexed_value_residual=residual,
            flow_indexed_support=torch.tensor([[True, False]]),
            flow_indexed_confidence=torch.ones(1, 2),
        )
        output_a, diagnostics = source_addressed_native_history_attention(
            **common
        )
        common["recent_target_key"] = torch.full_like(native, 10000.0)
        common["current_target_key"] = torch.full_like(native, -10000.0)
        output_b, _ = source_addressed_native_history_attention(**common)
        torch.testing.assert_close(output_a, output_b, rtol=0, atol=0)
        torch.testing.assert_close(
            output_a[:, :1], native[:, :1] + residual[:, :1], rtol=0, atol=0
        )
        torch.testing.assert_close(
            output_a[:, 1:], native[:, 1:], rtol=0, atol=0
        )
        self.assertTrue(bool(diagnostics["flow_indexed_read"].all()))

    def test_flow_indexed_ledger_transports_residual_by_source_motion(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=8, max_tokens_per_frame=8,
            min_write_confidence=0.5, transactional_dense_recent=True,
            token_atomic_dense_recent=True, persistent_residual_upsert=True,
            last_trusted_residual_lineage=True,
            flow_indexed_residual_ledger=True,
            source_flow_cache=translation_flow_cache(),
        )
        source = torch.zeros(1, 8, 1, 2)
        target = source.clone()
        target[:, 4] = torch.tensor([[[2.0, 3.0]]])
        write = torch.zeros(1, 8)
        write[:, 4] = 1.0
        history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(target, target),
            write_confidence=write, retention_confidence=torch.ones_like(write),
            frame_indices=(0,), spatial_shape=(2, 4),
        )
        prepared = history.prepare_flow_read(
            frame_indices=(1,), spatial_shape=(2, 4), device=torch.device("cpu")
        )[0]
        # Source coordinate (row=1,col=0) moves one cell to the right.
        self.assertTrue(bool(prepared.support[0, 5]))
        torch.testing.assert_close(
            prepared.value_residual[:, 5], target[:, 4], rtol=0, atol=1e-6
        )
        self.assertFalse(bool(prepared.support[0, 4]))

    def test_flow_ledger_retained_confidence_comes_from_transport(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=8, max_tokens_per_frame=8,
            min_write_confidence=0.5, transactional_dense_recent=True,
            token_atomic_dense_recent=True, persistent_residual_upsert=True,
            last_trusted_residual_lineage=True,
            flow_indexed_residual_ledger=True,
            source_flow_cache=translation_flow_cache(),
        )
        source = torch.zeros(1, 8, 1, 2)
        target = source.clone()
        target[:, 4] = torch.tensor([[[2.0, 3.0]]])
        first_write = torch.zeros(1, 8)
        first_write[:, 4] = 0.8
        history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(target, target),
            write_confidence=first_write,
            retention_confidence=torch.ones_like(first_write),
            frame_indices=(0,), spatial_shape=(2, 4),
        )
        transported = history.prepare_flow_read(
            frame_indices=(1,), spatial_shape=(2, 4),
            device=torch.device("cpu"),
        )[0]
        transported_confidence = transported.confidence[:, 5].clone()

        # A positive but sub-threshold write proposal is not a new payload
        # observation.  The retained residual must keep the confidence of its
        # flow transport instead of being silently overwritten by 0.2.
        weak_write = torch.zeros(1, 8)
        weak_write[:, 5] = 0.2
        diagnostics = history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(source + 100.0, source + 100.0),
            write_confidence=weak_write,
            retention_confidence=torch.ones_like(weak_write),
            frame_indices=(1,), spatial_shape=(2, 4),
        )[0]
        self.assertTrue(bool(
            diagnostics["persistent_retained_support"][0, 5]
        ))
        torch.testing.assert_close(
            diagnostics["flow_indexed_state_confidence"][0, 5],
            transported_confidence[0], rtol=0, atol=0,
        )
        self.assertGreater(
            diagnostics["flow_indexed_state_confidence"][0, 5].item(),
            weak_write[0, 5].item(),
        )

    def test_decoupled_flow_trust_does_not_decay_across_blocks(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=8, max_tokens_per_frame=8,
            min_write_confidence=0.5, transactional_dense_recent=True,
            token_atomic_dense_recent=True, persistent_residual_upsert=True,
            last_trusted_residual_lineage=True,
            flow_indexed_residual_ledger=True, decoupled_flow_trust=True,
            source_flow_cache=identity_flow_cache(confidence=0.8),
        )
        # Match inference's three latent frames per causal chunk.  The trusted
        # payload sits in the last frame that becomes the next flow state.
        source = torch.zeros(1, 24, 1, 2)
        target = source.clone()
        target[:, 20] = torch.tensor([[[2.0, 3.0]]])
        initial_write = torch.zeros(1, 24)
        initial_write[:, 20] = 0.9
        history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(target, target),
            write_confidence=initial_write,
            retention_confidence=torch.ones_like(initial_write),
            frame_indices=(0, 1, 2), spatial_shape=(2, 4),
        )

        weak_write = torch.zeros_like(initial_write)
        effective_confidences = []
        for frame_indices in ((3, 4, 5), (6, 7, 8)):
            prepared = history.prepare_flow_read(
                frame_indices=frame_indices, spatial_shape=(2, 4),
                device=torch.device("cpu"),
            )[0]
            tracked = torch.tensor([4, 12, 20])
            effective_confidences.append(
                prepared.confidence[0, tracked].clone()
            )
            torch.testing.assert_close(
                prepared.appearance_trust[0, tracked],
                torch.full((3,), 0.9), rtol=0, atol=1e-6,
            )
            torch.testing.assert_close(
                prepared.transport_confidence[0, tracked],
                torch.tensor([0.8, 0.64, 0.512]), rtol=0, atol=1e-6,
            )
            diagnostics = history.commit(
                source_kv_cache=make_cache(source, source),
                target_kv_cache=make_cache(source + 100.0, source + 100.0),
                write_confidence=weak_write,
                retention_confidence=torch.ones_like(weak_write),
                frame_indices=frame_indices, spatial_shape=(2, 4),
            )[0]
            torch.testing.assert_close(
                diagnostics["flow_indexed_appearance_trust"][0, 20],
                torch.tensor(0.9), rtol=0, atol=1e-6,
            )
            torch.testing.assert_close(
                diagnostics[
                    "flow_indexed_local_transport_confidence"
                ][0, 20],
                torch.tensor(1.0), rtol=0, atol=0,
            )

        torch.testing.assert_close(
            torch.stack(effective_confidences),
            torch.tensor([[0.72, 0.576, 0.4608]]).expand(2, -1),
            rtol=0, atol=1e-6,
        )

    def test_decoupled_attention_uses_trust_times_local_transport(self):
        native = torch.zeros(1, 1, 1, 2)
        residual = torch.ones_like(native)
        output, diagnostics = source_addressed_native_history_attention(
            native, torch.randn_like(native),
            torch.randn_like(native), torch.randn_like(native),
            torch.randn_like(native), torch.randn_like(native),
            torch.ones(1, 1, dtype=torch.bool),
            torch.randn_like(native), torch.randn_like(native),
            torch.randn_like(native), torch.randn_like(native),
            torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1),
            recent_source_value=torch.zeros_like(native),
            recent_payload_support=torch.ones(1, 1, dtype=torch.bool),
            residual_rebased_payload=True, last_trusted_appearance=True,
            consistent_transaction=True, entry_bridge=True,
            motion_owner_dense_read=True, entry_query_count=1,
            flow_indexed_value_residual=residual,
            flow_indexed_support=torch.ones(1, 1, dtype=torch.bool),
            # Deliberately inconsistent legacy product: the decoupled fields
            # are authoritative when both are supplied.
            flow_indexed_confidence=torch.tensor([[0.01]]),
            flow_indexed_appearance_trust=torch.tensor([[0.81]]),
            flow_indexed_transport_confidence=torch.tensor([[0.64]]),
        )
        effective = torch.tensor(0.81 * 0.64)
        torch.testing.assert_close(
            diagnostics["flow_transport_confidence"][0, 0],
            effective, rtol=0, atol=1e-6,
        )
        # Full request plus entry bridge uses sqrt(effective confidence).
        torch.testing.assert_close(
            output[0, 0, 0, 0], effective.sqrt(), rtol=0, atol=1e-6,
        )

    def test_decoupled_flow_transport_preserves_residual_amplitude(self):
        history = RoleConditionedNativeKVHistory(
            layers=(0,), tokens_per_frame=8, max_tokens_per_frame=8,
            min_write_confidence=0.5, transactional_dense_recent=True,
            token_atomic_dense_recent=True, persistent_residual_upsert=True,
            last_trusted_residual_lineage=True,
            flow_indexed_residual_ledger=True, decoupled_flow_trust=True,
            source_flow_cache=identity_flow_cache(
                frames=4, confidence=1.0, displacement=0.5
            ),
        )
        source = torch.zeros(1, 8, 1, 2)
        target = source.clone()
        target[:, 5] = torch.tensor([[[2.0, 3.0]]])
        write = torch.zeros(1, 8)
        write[:, 5] = 0.9
        history.commit(
            source_kv_cache=make_cache(source, source),
            target_kv_cache=make_cache(target, target),
            write_confidence=write, retention_confidence=torch.ones_like(write),
            frame_indices=(0,), spatial_shape=(2, 4),
        )
        prepared = history.prepare_flow_read(
            frame_indices=(1, 2, 3), spatial_shape=(2, 4),
            device=torch.device("cpu"),
        )[0]
        supported = prepared.support[0]
        expected = torch.tensor([2.0, 3.0]).expand(
            int(supported.sum()), -1
        )
        torch.testing.assert_close(
            prepared.value_residual[0, supported, 0],
            expected, rtol=0, atol=1e-5,
        )
        torch.testing.assert_close(
            prepared.appearance_trust[0, supported],
            torch.full((int(supported.sum()),), 0.9),
            rtol=0, atol=1e-5,
        )

    def test_flow_indexed_attention_abstains_on_zero_confidence(self):
        native = torch.randn(1, 1, 1, 2)
        output, diagnostics = source_addressed_native_history_attention(
            native, torch.randn_like(native),
            torch.randn_like(native), torch.randn_like(native),
            torch.randn_like(native), torch.randn_like(native),
            torch.ones(1, 1, dtype=torch.bool),
            torch.randn_like(native), torch.randn_like(native),
            torch.randn_like(native), torch.randn_like(native),
            torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1),
            recent_source_value=torch.zeros_like(native),
            recent_payload_support=torch.ones(1, 1, dtype=torch.bool),
            residual_rebased_payload=True, last_trusted_appearance=True,
            consistent_transaction=True, entry_bridge=True,
            motion_owner_dense_read=True, entry_query_count=1,
            flow_indexed_value_residual=torch.ones_like(native),
            flow_indexed_support=torch.ones(1, 1, dtype=torch.bool),
            flow_indexed_confidence=torch.zeros(1, 1),
        )
        torch.testing.assert_close(output, native, rtol=0, atol=0)
        self.assertFalse(bool(diagnostics["admitted"].any()))


if __name__ == "__main__":
    unittest.main()
