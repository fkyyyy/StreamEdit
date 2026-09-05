from typing import List, Optional, Iterable
import torch
import torch.nn.functional as F
import math
from tqdm import tqdm 

import os
import numpy as np
from PIL import Image

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation
from .utils import (
    find_phrase_group_token_indices,
    find_phrase_token_indices,
)
from .contact_graph import (
    CONTACT_GRAPH_MODES,
    build_oracle_contact_graphs,
    contact_graph_stats,
)
from .belief_kv import build_belief_kv_weights
from .appearance_leakage import (
    build_target_change_core,
    remove_antagonistic_source_residual,
)
from .control_belief import CausalControlBeliefBuilder
from .causal_ownership import (
    AutomaticTransactionalOwnerTracker,
    CausalReadOnlyOwnerTracker,
    CausalObjectOwnershipTracker,
    build_motion_owner_read_weight,
    build_topology_complete_motion_owner_read_weight,
    build_oracle_causal_ownership,
)
from .motion.causal_motion_owner import (
    MotionAwareGeometryOwnerTracker,
    SourceFlowCache,
)
from .motion.flow_role_evidence import build_flow_role_evidence
from .causal_edit_memory import (
    CausalPairedEditMemory,
    build_object_coordinates,
    build_object_interior_gate,
    build_owner_attached_structure_gate,
)
from .native_kv_history import (
    RoleConditionedNativeKVHistory,
    validate_recent_entry_hand_only_contract,
)
from .semantic_edit_authority import (
    apply_semantic_transaction_gate,
    build_semantic_edit_authority,
)
from .edit_commitment import (
    EditCommitmentController,
    EditCommitmentResult,
)
from .factorized_bayes import (
    FactorizedBayesOperatorBuilder,
    FactorizedBayesOperators,
    route_factorized_velocity,
)
from .hand_role_inference import HandRoleInferencer
from .memory_consolidation import (
    CausalMemoryConsolidator,
    MemoryConsolidationPlan,
)
from .target_identity_memory import (
    CausalIdentityOwnerTracker,
    CausalConnectedSupportFilter,
    CausalObjectTokenPropagator,
    ConnectedIdentitySupport,
    FirstFrameIdentityBootstrap,
    SlowTargetIdentityMemory,
    SourceCoordinateResidualCarry,
    TargetIdentityTokenPropagation,
    TargetIdentityUpdate,
    apply_source_owner_geometry_envelope,
    apply_source_owner_residual_constraint,
    build_first_frame_object_core_bootstrap,
    build_oracle_source_owner_weight,
    build_reference_identity_bootstrap,
    inject_committed_memory_into_belief,
    strengthen_belief_with_target_identity,
)
from .role_router import (
    BayesResidualFlowRouter,
    PosteriorResidualFlowRouter,
    ResidualRoleFlowRouter,
    RoleFlowRouter,
    build_oracle_roles,
)
from wan.modules.attention import (
    materialize_immutable_target_value,
    project_source_addressed_target_value,
)

class EditCausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _check_prompts(self, *args):
        ret_list = []
        for itm in args:
            if isinstance(itm, str):
                itm = [itm]
            ret_list.append(itm)
        return ret_list


    def rollout_inference(
        self,
        src_video: torch.Tensor,
        src_prompts: str | List[str],
        trg_prompts: str | List[str],
        src_trigger_words: str | List[str],
        trg_trigger_words: str | List[str],
        return_latents: bool = False,
        wo_video_decode: bool = False,
        profile: bool = False,
        low_memory: bool = False,

        independent_first_frame: bool = False,
        triple_first_frame: bool = False,
        src_initial_latent: Optional[torch.Tensor] = None,  
        trg_initial_latent: Optional[torch.Tensor] = None,

        fg_boost_factor=2.0,
        blend_power=2.0,

        mask_layers: Iterable = range(20),
        enhance_layers: Iterable = range(30),

        fg_scale=1.0,
        reuse_noise_temporal_mean=True,

        rollout_chunk_size: int = 21,
        rollout_overlap_block_num: int = 1,
        routing_mode: str = "dynamic_sog",
        identity_first_latent_bootstrap: bool = False,
        object_wise_anchor_reset: bool = False,
        target_owned_object_handoff: bool = False,
        target_owned_min_similarity: float = 0.55,
        first_chunk_identity_replay: bool = False,
        factorized_target_identity: bool = False,
        factorized_immutable_target_memory: bool = False,
        factorized_native_target_history: bool = False,
        factorized_owner_source_block: bool = False,
        target_semantic_competition: bool = False,
        target_edit_phrases: Optional[List[str]] = None,
        target_preserve_phrases: Optional[List[str]] = None,
        target_semantic_margin: float = 0.10,
        target_semantic_min_confidence: float = 0.20,
        causal_paired_edit_memory: bool = False,
        paired_memory_layers: Optional[Iterable] = None,
        paired_memory_max_tokens: int = 1536,
        paired_memory_max_tokens_per_block: int = 192,
        paired_memory_topk: int = 8,
        paired_memory_min_similarity: float = 0.35,
        paired_memory_min_commit_confidence: float = 0.20,
        paired_memory_coordinate_bias: float = 1.0,
        paired_memory_coordinate_radius: float = 0.0,
        paired_memory_min_residual_consensus: float = 0.0,
        paired_memory_source_part_consistency: bool = False,
        paired_memory_min_part_similarity: float = 0.45,
        paired_memory_part_similarity_margin: float = 0.08,
        paired_memory_read_strength: float = 0.35,
        paired_memory_value_projection: bool = False,
        paired_memory_query_gated_projection: bool = False,
        paired_memory_disable_persistent_projection: bool = False,
        paired_memory_source_suppression: float = 0.0,
        paired_memory_interior_projection: bool = False,
        paired_memory_first_block_replay: bool = False,
        paired_memory_source_transport: bool = False,
        paired_memory_single_confidence: bool = False,
        paired_memory_owner_attached_boundary: bool = False,
        paired_memory_dual_timescale_anchor: bool = False,
        paired_memory_canonical_key_anchor: bool = False,
        role_fixed_native_history: bool = False,
        native_history_layers: Optional[Iterable] = None,
        native_history_max_tokens_per_frame: int = 256,
        native_history_topk: int = 8,
        native_history_min_similarity: float = 0.35,
        native_history_min_write_confidence: float = 0.50,
        native_history_min_query_confidence: float = 0.50,
        native_history_canonical_logit_bias: float = 1.0,
        native_history_coalesce_bootstrap_time: bool = False,
        native_history_bypass_blocks: Optional[Iterable[int]] = None,
        native_history_source_part_consistency: bool = False,
        native_history_min_part_similarity: float = 0.45,
        native_history_part_similarity_margin: float = 0.08,
        native_history_part_bias_strength: float = 0.5,
        native_history_part_refinement_ratio: float = 0.25,
        native_history_transactional_owner: bool = False,
        native_history_consistent_transaction: bool = False,
        native_history_verified_attention_authority: bool = False,
        native_history_attention_authority_strength: float = 1.0,
        native_history_payload_invariant_lineage: bool = False,
        native_history_payload_blend_strength: float = 0.35,
        native_history_recent_entry_bridge: bool = False,
        native_history_motion_owner_dense_read: bool = False,
        native_history_entry_bridge_strength: float = 1.0,
        native_history_dual_evidence_arbitration: bool = False,
        native_history_token_atomic_payload: bool = False,
        native_history_persistent_residual_upsert: bool = False,
        native_history_last_trusted_appearance: bool = False,
        native_history_flow_indexed_residual: bool = False,
        native_history_decoupled_flow_trust: bool = False,
        native_history_multiframe_identity_sink: bool = False,
        native_history_multiframe_sink_topk_per_frame: int = 8,
        native_history_multiframe_sink_source_logit_bias: float = 1.0,
        native_history_multiframe_sink_strength: float = 1.0,
        native_history_timestep_counterfactual_memory: bool = False,
        native_history_tccm_flow_radius: float = 2.0,
        native_history_tccm_strength: float = 1.0,
        native_history_tccm_max_error_ratio: float = 1.0,
        native_history_flow_min_confidence: float = 0.10,
        native_history_residual_update_min_cosine: float = 0.50,
        native_history_residual_update_min_magnitude_ratio: float = 0.90,
        native_history_topology_complete_read: bool = False,
        native_history_min_payload_consistency: float = 0.15,
        native_history_dense_recent_min_residual_consensus: float = 0.05,
        native_history_owner_max_missing_frames: int = 1,
        native_history_verified_source_suppression: float = 0.35,
        paired_memory_transport_min_similarity: float = 0.10,
        paired_memory_transport_coordinate_radius: float = 0.60,
        paired_memory_transport_cycle_radius: float = 0.20,
        paired_memory_transport_min_confidence: float = 0.05,
        immutable_target_layers: Optional[Iterable] = None,
        immutable_target_num_prototypes: int = 4,
        immutable_target_value_mode: str = "residual",
        immutable_target_hard_owner: bool = False,
        factorized_orthogonal_geometry: bool = False,
        factorized_geometry_strength: float = 1.0,
        identity_correction_strength: float = 0.35,
        identity_visibility_lifecycle: bool = False,
        identity_max_occluded_blocks: int = 1,
        appearance_leakage_decomposition: bool = False,
        source_coordinate_identity: bool = False,
        identity_source_suppression: float = 0.35,
        identity_support_floor: float = 0.0,
        source_identity_residual_carry: bool = False,
        identity_residual_carry_strength: float = 0.25,
        source_owner_residual_constraint: bool = False,
        identity_residual_constraint_strength: float = 0.35,
        identity_residual_constraint_power: float = 2.0,
        source_owner_geometry_envelope: bool = False,
        source_geometry_strength: float = 0.35,
        source_geometry_power: float = 2.0,
        source_geometry_margin: int = 1,
        ignition_hand_exclusion_radius: int = 1,
        ignition_contact_radius: int = 3,
        oracle_source_owner_mask: Optional[torch.Tensor] = None,
        oracle_source_owner_full_mask: Optional[torch.Tensor] = None,
        source_owner_prepool_hand_exclusion: bool = False,
        causal_owner_consistent_kv_metadata: bool = False,
        factorized_source_coordinate_target_delta: bool = False,
        factorized_owner_complement_source: bool = False,
        factorized_owner_complement_margin: int = 1,
        factorized_owner_complement_min_preserve_confidence: float = 0.0,
        oracle_object_mask: Optional[torch.Tensor] = None,
        oracle_hand_mask: Optional[torch.Tensor] = None,
        hand_only_mask: Optional[torch.Tensor] = None,
        hand_occupancy_mask: Optional[torch.Tensor] = None,
        hand_persistent_mask: Optional[torch.Tensor] = None,
        hand_causal_evidence: bool = False,
        motion_geometry_owner: bool = False,
        source_flow_cache: Optional[SourceFlowCache] = None,
        source_flow_role_fusion: bool = False,
        source_flow_role_weight: float = 0.75,
        source_flow_verified_region: bool = False,
        source_flow_verified_owner_radius: int = 1,
        source_flow_background_veto_threshold: float = 0.55,
        source_flow_background_veto_min_confidence: float = 0.50,
        soft_region_modulation: bool = False,
        soft_region_blend_strength: float = 0.5,
        first_block_identity_anchor: bool = False,
        identity_anchor_scale: float = 1.5,
        suppress_source_bg_value: bool = False,
        projected_source_residual: bool = False,
        role_boundary_radius: int = 1,
        contact_target_weight: float = 0.7,
        posterior_flow_mode: str = "soft",
        posterior_flow_use_field: bool = False,
        hand_posterior_threshold: float = 0.20,
        hand_max_object_coverage: float = 0.18,
        hand_proximity_radius: int = 3,
        hand_propagation_steps: int = 2,
        hand_connected_hysteresis: bool = False,
        hand_connected_growth_steps: int = 3,
        hand_connected_candidate_ratio: float = 1.0,
        hand_visibility_ratio: float = 0.40,
        hand_temporal_weight: float = 0.45,
        hand_query_similarity_threshold: float = 0.65,
        hand_query_layers: Iterable = (8, 12, 16, 20),
        hand_field_quantile_low: float = 0.50,
        hand_field_quantile_high: float = 0.95,
        hand_field_power: float = 1.5,
        hand_field_weight: float = 0.65,
        hand_field_candidate_radius: int = 2,
        hand_field_update_mode: str = "diagnostic",
        identity_tokenprop_min_similarity: float = 0.55,
        identity_tokenprop_gate_strength: float = 0.85,
        identity_tokenprop_max_candidates: int = 512,
        committed_memory_feedback_strength: float = 0.75,
        contact_graph_mode: str = "no_graph",
        contact_graph_topk: int = 4,
        contact_graph_radius: float = 2.5,
        contact_graph_min_confidence: float = 0.05,
        contact_graph_strength: float = 0.25,
        contact_graph_layer_start: int = 10,
        contact_graph_layer_end: int = 20,
        contact_graph_seed: int = 0,
        save_role_dir: Optional[str] = None,
        _hand_role_inferencer: Optional[HandRoleInferencer] = None,
        _memory_consolidator: Optional[
            CausalMemoryConsolidator
        ] = None,
        _edit_commitment_controller: Optional[
            EditCommitmentController
        ] = None,
        _target_identity_memory: Optional[
            SlowTargetIdentityMemory
        ] = None,
        _identity_token_propagator: Optional[
            CausalObjectTokenPropagator
        ] = None,
        _identity_support_filter: Optional[
            CausalConnectedSupportFilter
        ] = None,
        _identity_owner_tracker: Optional[
            CausalIdentityOwnerTracker
        ] = None,
        _causal_ownership_tracker: Optional[
            CausalObjectOwnershipTracker | MotionAwareGeometryOwnerTracker
        ] = None,
        _native_owner_tracker: Optional[
            CausalReadOnlyOwnerTracker
            | AutomaticTransactionalOwnerTracker
        ] = None,
        _identity_residual_carry: Optional[
            SourceCoordinateResidualCarry
        ] = None,
        _causal_paired_edit_memory: Optional[
            CausalPairedEditMemory
        ] = None,
        _role_native_kv_history: Optional[
            RoleConditionedNativeKVHistory
        ] = None,
    ) -> torch.Tensor:
        expected_role_shape = (
            src_video.shape[0], src_video.shape[1],
            src_video.shape[-2], src_video.shape[-1],
        )
        for name, mask in (
            ("oracle_source_owner_mask", oracle_source_owner_mask),
            (
                "oracle_source_owner_full_mask",
                oracle_source_owner_full_mask,
            ),
            ("oracle_object_mask", oracle_object_mask),
            ("oracle_hand_mask", oracle_hand_mask),
            ("hand_only_mask", hand_only_mask),
            ("hand_occupancy_mask", hand_occupancy_mask),
            ("hand_persistent_mask", hand_persistent_mask),
        ):
            if mask is not None and tuple(mask.shape) != expected_role_shape:
                raise ValueError(
                    f"{name} must have shape {expected_role_shape}, "
                    f"got {tuple(mask.shape)}"
                )
        if hand_only_mask is not None:
            if hand_occupancy_mask is None:
                hand_occupancy_mask = hand_only_mask.float()
        if motion_geometry_owner:
            if routing_mode != "hand_role_factorized_causal_owner_kv":
                raise ValueError(
                    "motion_geometry_owner requires the causal-owner routing mode"
                )
            if source_flow_cache is None:
                raise ValueError(
                    "motion_geometry_owner requires source_flow_cache"
                )
            if source_flow_cache.latent_frame_count != src_video.shape[1]:
                raise ValueError(
                    "Source flow cache latent count does not match the full "
                    f"source video: {source_flow_cache.latent_frame_count} != "
                    f"{src_video.shape[1]}"
                )
            if hand_persistent_mask is None:
                hand_persistent_mask = hand_only_mask.bool()
        if source_flow_role_fusion and not motion_geometry_owner:
            raise ValueError(
                "source_flow_role_fusion requires motion_geometry_owner"
            )
        if source_flow_verified_region and not source_flow_role_fusion:
            raise ValueError(
                "source_flow_verified_region requires "
                "source_flow_role_fusion"
            )
        if not 0.0 <= source_flow_role_weight <= 1.0:
            raise ValueError("source_flow_role_weight must lie in [0, 1]")
        if source_flow_verified_owner_radius < 0:
            raise ValueError(
                "source_flow_verified_owner_radius must be non-negative"
            )
        if not 0.0 <= source_flow_background_veto_threshold <= 1.0:
            raise ValueError(
                "source_flow_background_veto_threshold must lie in [0, 1]"
            )
        if not 0.0 <= source_flow_background_veto_min_confidence <= 1.0:
            raise ValueError(
                "source_flow_background_veto_min_confidence must lie in "
                "[0, 1]"
            )
        if rollout_chunk_size < 0:
            # for testing local attn
            return self.inference(
                src_video=src_video,
                src_prompts=src_prompts,
                trg_prompts=trg_prompts,
                src_trigger_words=src_trigger_words,
                trg_trigger_words=trg_trigger_words,
                return_latents=return_latents,
                wo_video_decode=wo_video_decode,
                profile=profile,
                low_memory=low_memory,

                independent_first_frame=independent_first_frame,
                triple_first_frame=triple_first_frame,
                src_initial_latent=src_initial_latent,
                trg_initial_latent=trg_initial_latent,

                mask_layers=mask_layers,
                enhance_layers=enhance_layers,
                reuse_noise_temporal_mean=reuse_noise_temporal_mean,

                fg_scale=fg_scale,
                fg_boost_factor=fg_boost_factor,

                blend_power=blend_power,
                routing_mode=routing_mode,
                identity_first_latent_bootstrap=(
                    identity_first_latent_bootstrap
                ),
                object_wise_anchor_reset=object_wise_anchor_reset,
                target_owned_object_handoff=(
                    target_owned_object_handoff
                ),
                target_owned_min_similarity=(
                    target_owned_min_similarity
                ),
                factorized_target_identity=(
                    factorized_target_identity
                ),
                factorized_immutable_target_memory=(
                    factorized_immutable_target_memory
                ),
                factorized_native_target_history=(
                    factorized_native_target_history
                ),
                factorized_owner_source_block=(
                    factorized_owner_source_block
                ),
                target_semantic_competition=(
                    target_semantic_competition
                ),
                target_edit_phrases=target_edit_phrases,
                target_preserve_phrases=target_preserve_phrases,
                target_semantic_margin=target_semantic_margin,
                target_semantic_min_confidence=(
                    target_semantic_min_confidence
                ),
                causal_paired_edit_memory=causal_paired_edit_memory,
                paired_memory_layers=paired_memory_layers,
                paired_memory_max_tokens=paired_memory_max_tokens,
                paired_memory_max_tokens_per_block=(
                    paired_memory_max_tokens_per_block
                ),
                paired_memory_topk=paired_memory_topk,
                paired_memory_min_similarity=(
                    paired_memory_min_similarity
                ),
                paired_memory_min_commit_confidence=(
                    paired_memory_min_commit_confidence
                ),
                paired_memory_coordinate_bias=(
                    paired_memory_coordinate_bias
                ),
                paired_memory_coordinate_radius=(
                    paired_memory_coordinate_radius
                ),
                paired_memory_min_residual_consensus=(
                    paired_memory_min_residual_consensus
                ),
                paired_memory_source_part_consistency=(
                    paired_memory_source_part_consistency
                ),
                paired_memory_min_part_similarity=(
                    paired_memory_min_part_similarity
                ),
                paired_memory_part_similarity_margin=(
                    paired_memory_part_similarity_margin
                ),
                paired_memory_read_strength=(
                    paired_memory_read_strength
                ),
                paired_memory_value_projection=(
                    paired_memory_value_projection
                ),
                paired_memory_query_gated_projection=(
                    paired_memory_query_gated_projection
                ),
                paired_memory_disable_persistent_projection=(
                    paired_memory_disable_persistent_projection
                ),
                paired_memory_source_suppression=(
                    paired_memory_source_suppression
                ),
                paired_memory_interior_projection=(
                    paired_memory_interior_projection
                ),
                paired_memory_source_transport=(
                    paired_memory_source_transport
                ),
                paired_memory_single_confidence=(
                    paired_memory_single_confidence
                ),
                paired_memory_owner_attached_boundary=(
                    paired_memory_owner_attached_boundary
                ),
                paired_memory_dual_timescale_anchor=(
                    paired_memory_dual_timescale_anchor
                ),
                paired_memory_canonical_key_anchor=(
                    paired_memory_canonical_key_anchor
                ),
                role_fixed_native_history=role_fixed_native_history,
                native_history_layers=native_history_layers,
                native_history_max_tokens_per_frame=(
                    native_history_max_tokens_per_frame
                ),
                native_history_topk=native_history_topk,
                native_history_min_similarity=(
                    native_history_min_similarity
                ),
                native_history_min_write_confidence=(
                    native_history_min_write_confidence
                ),
                native_history_min_query_confidence=(
                    native_history_min_query_confidence
                ),
                native_history_canonical_logit_bias=(
                    native_history_canonical_logit_bias
                ),
                native_history_coalesce_bootstrap_time=(
                    native_history_coalesce_bootstrap_time
                ),
                native_history_bypass_blocks=(
                    native_history_bypass_blocks
                ),
                native_history_source_part_consistency=(
                    native_history_source_part_consistency
                ),
                native_history_min_part_similarity=(
                    native_history_min_part_similarity
                ),
                native_history_part_similarity_margin=(
                    native_history_part_similarity_margin
                ),
                native_history_part_bias_strength=(
                    native_history_part_bias_strength
                ),
                native_history_part_refinement_ratio=(
                    native_history_part_refinement_ratio
                ),
                native_history_transactional_owner=(
                    native_history_transactional_owner
                ),
                native_history_consistent_transaction=(
                    native_history_consistent_transaction
                ),
                native_history_verified_attention_authority=(
                    native_history_verified_attention_authority
                ),
                native_history_attention_authority_strength=(
                    native_history_attention_authority_strength
                ),
                native_history_payload_invariant_lineage=(
                    native_history_payload_invariant_lineage
                ),
                native_history_payload_blend_strength=(
                    native_history_payload_blend_strength
                ),
                native_history_recent_entry_bridge=(
                    native_history_recent_entry_bridge
                ),
                native_history_motion_owner_dense_read=(
                    native_history_motion_owner_dense_read
                ),
                native_history_entry_bridge_strength=(
                    native_history_entry_bridge_strength
                ),
                native_history_dual_evidence_arbitration=(
                    native_history_dual_evidence_arbitration
                ),
                native_history_token_atomic_payload=(
                    native_history_token_atomic_payload
                ),
                native_history_persistent_residual_upsert=(
                    native_history_persistent_residual_upsert
                ),
                native_history_last_trusted_appearance=(
                    native_history_last_trusted_appearance
                ),
                native_history_flow_indexed_residual=(
                    native_history_flow_indexed_residual
                ),
                native_history_decoupled_flow_trust=(
                    native_history_decoupled_flow_trust
                ),
                native_history_multiframe_identity_sink=(
                    native_history_multiframe_identity_sink
                ),
                native_history_multiframe_sink_topk_per_frame=(
                    native_history_multiframe_sink_topk_per_frame
                ),
                native_history_multiframe_sink_source_logit_bias=(
                    native_history_multiframe_sink_source_logit_bias
                ),
                native_history_multiframe_sink_strength=(
                    native_history_multiframe_sink_strength
                ),
                native_history_timestep_counterfactual_memory=(
                    native_history_timestep_counterfactual_memory
                ),
                native_history_tccm_flow_radius=(
                    native_history_tccm_flow_radius
                ),
                native_history_tccm_strength=(
                    native_history_tccm_strength
                ),
                native_history_tccm_max_error_ratio=(
                    native_history_tccm_max_error_ratio
                ),
                native_history_flow_min_confidence=(
                    native_history_flow_min_confidence
                ),
                native_history_residual_update_min_cosine=(
                    native_history_residual_update_min_cosine
                ),
                native_history_residual_update_min_magnitude_ratio=(
                    native_history_residual_update_min_magnitude_ratio
                ),
                native_history_topology_complete_read=(
                    native_history_topology_complete_read
                ),
                native_history_min_payload_consistency=(
                    native_history_min_payload_consistency
                ),
                native_history_dense_recent_min_residual_consensus=(
                    native_history_dense_recent_min_residual_consensus
                ),
                native_history_owner_max_missing_frames=(
                    native_history_owner_max_missing_frames
                ),
                native_history_verified_source_suppression=(
                    native_history_verified_source_suppression
                ),
                paired_memory_transport_min_similarity=(
                    paired_memory_transport_min_similarity
                ),
                paired_memory_transport_coordinate_radius=(
                    paired_memory_transport_coordinate_radius
                ),
                paired_memory_transport_cycle_radius=(
                    paired_memory_transport_cycle_radius
                ),
                paired_memory_transport_min_confidence=(
                    paired_memory_transport_min_confidence
                ),
                immutable_target_layers=immutable_target_layers,
                immutable_target_num_prototypes=(
                    immutable_target_num_prototypes
                ),
                immutable_target_value_mode=immutable_target_value_mode,
                immutable_target_hard_owner=immutable_target_hard_owner,
                factorized_orthogonal_geometry=(
                    factorized_orthogonal_geometry
                ),
                factorized_geometry_strength=(
                    factorized_geometry_strength
                ),
                identity_correction_strength=(
                    identity_correction_strength
                ),
                identity_visibility_lifecycle=(
                    identity_visibility_lifecycle
                ),
                identity_max_occluded_blocks=(
                    identity_max_occluded_blocks
                ),
                appearance_leakage_decomposition=(
                    appearance_leakage_decomposition
                ),
                source_coordinate_identity=source_coordinate_identity,
                identity_source_suppression=(
                    identity_source_suppression
                ),
                identity_support_floor=identity_support_floor,
                source_identity_residual_carry=(
                    source_identity_residual_carry
                ),
                identity_residual_carry_strength=(
                    identity_residual_carry_strength
                ),
                source_owner_residual_constraint=(
                    source_owner_residual_constraint
                ),
                identity_residual_constraint_strength=(
                    identity_residual_constraint_strength
                ),
                identity_residual_constraint_power=(
                    identity_residual_constraint_power
                ),
                source_owner_geometry_envelope=(
                    source_owner_geometry_envelope
                ),
                source_geometry_strength=source_geometry_strength,
                source_geometry_power=source_geometry_power,
                source_geometry_margin=source_geometry_margin,
                ignition_hand_exclusion_radius=(
                    ignition_hand_exclusion_radius
                ),
                ignition_contact_radius=ignition_contact_radius,
                oracle_source_owner_mask=oracle_source_owner_mask,
                oracle_source_owner_full_mask=(
                    oracle_source_owner_full_mask
                ),
                source_owner_prepool_hand_exclusion=(
                    source_owner_prepool_hand_exclusion
                ),
                causal_owner_consistent_kv_metadata=(
                    causal_owner_consistent_kv_metadata
                ),
                factorized_source_coordinate_target_delta=(
                    factorized_source_coordinate_target_delta
                ),
                factorized_owner_complement_source=(
                    factorized_owner_complement_source
                ),
                factorized_owner_complement_margin=(
                    factorized_owner_complement_margin
                ),
                factorized_owner_complement_min_preserve_confidence=(
                    factorized_owner_complement_min_preserve_confidence
                ),
                oracle_object_mask=oracle_object_mask,
                oracle_hand_mask=oracle_hand_mask,
                hand_only_mask=hand_only_mask,
                hand_occupancy_mask=hand_occupancy_mask,
                hand_persistent_mask=hand_persistent_mask,
                hand_causal_evidence=hand_causal_evidence,
                motion_geometry_owner=motion_geometry_owner,
                source_flow_cache=source_flow_cache,
                source_flow_role_fusion=source_flow_role_fusion,
                source_flow_role_weight=source_flow_role_weight,
                source_flow_verified_region=source_flow_verified_region,
                source_flow_verified_owner_radius=(
                    source_flow_verified_owner_radius
                ),
                source_flow_background_veto_threshold=(
                    source_flow_background_veto_threshold
                ),
                source_flow_background_veto_min_confidence=(
                    source_flow_background_veto_min_confidence
                ),
                soft_region_modulation=soft_region_modulation,
                soft_region_blend_strength=soft_region_blend_strength,
                first_block_identity_anchor=first_block_identity_anchor,
                identity_anchor_scale=identity_anchor_scale,
                suppress_source_bg_value=suppress_source_bg_value,
                projected_source_residual=projected_source_residual,
                global_frame_indices=list(range(src_video.shape[1])),
                role_boundary_radius=role_boundary_radius,
                contact_target_weight=contact_target_weight,
                posterior_flow_mode=posterior_flow_mode,
                posterior_flow_use_field=posterior_flow_use_field,
                hand_posterior_threshold=hand_posterior_threshold,
                hand_max_object_coverage=hand_max_object_coverage,
                hand_proximity_radius=hand_proximity_radius,
                hand_propagation_steps=hand_propagation_steps,
                hand_connected_hysteresis=hand_connected_hysteresis,
                hand_connected_growth_steps=(
                    hand_connected_growth_steps
                ),
                hand_connected_candidate_ratio=(
                    hand_connected_candidate_ratio
                ),
                hand_visibility_ratio=hand_visibility_ratio,
                hand_temporal_weight=hand_temporal_weight,
                hand_query_similarity_threshold=(
                    hand_query_similarity_threshold
                ),
                hand_query_layers=hand_query_layers,
                hand_field_quantile_low=hand_field_quantile_low,
                hand_field_quantile_high=hand_field_quantile_high,
                hand_field_power=hand_field_power,
                hand_field_weight=hand_field_weight,
                hand_field_candidate_radius=hand_field_candidate_radius,
                hand_field_update_mode=hand_field_update_mode,
                identity_tokenprop_min_similarity=(
                    identity_tokenprop_min_similarity
                ),
                identity_tokenprop_gate_strength=(
                    identity_tokenprop_gate_strength
                ),
                identity_tokenprop_max_candidates=(
                    identity_tokenprop_max_candidates
                ),
                committed_memory_feedback_strength=(
                    committed_memory_feedback_strength
                ),
                contact_graph_mode=contact_graph_mode,
                contact_graph_topk=contact_graph_topk,
                contact_graph_radius=contact_graph_radius,
                contact_graph_min_confidence=contact_graph_min_confidence,
                contact_graph_strength=contact_graph_strength,
                contact_graph_layer_start=contact_graph_layer_start,
                contact_graph_layer_end=contact_graph_layer_end,
                contact_graph_seed=contact_graph_seed,
                save_role_dir=save_role_dir,
                _hand_role_inferencer=_hand_role_inferencer,
                _memory_consolidator=_memory_consolidator,
                _edit_commitment_controller=(
                    _edit_commitment_controller
                ),
                _target_identity_memory=_target_identity_memory,
                _identity_token_propagator=_identity_token_propagator,
                _identity_support_filter=_identity_support_filter,
                _identity_owner_tracker=_identity_owner_tracker,
                _causal_ownership_tracker=(
                    _causal_ownership_tracker
                ),
                _native_owner_tracker=_native_owner_tracker,
                _identity_residual_carry=_identity_residual_carry,
                _causal_paired_edit_memory=(
                    _causal_paired_edit_memory
                ),
                _role_native_kv_history=_role_native_kv_history,
            )

        rollout_overlap = rollout_overlap_block_num * self.num_frame_per_block
        identity_memory_layers = tuple(
            hand_query_layers
            if immutable_target_layers is None
            else immutable_target_layers
        )
        paired_edit_memory_layers = tuple(
            hand_query_layers
            if paired_memory_layers is None
            else paired_memory_layers
        )
        role_native_history_layers = tuple(
            hand_query_layers
            if native_history_layers is None
            else native_history_layers
        )
        role_native_history_bypass_blocks = tuple(
            ()
            if native_history_bypass_blocks is None
            else native_history_bypass_blocks
        )
        if target_semantic_competition:
            if (
                routing_mode
                != "hand_role_factorized_causal_owner_kv"
            ):
                raise ValueError(
                    "Target semantic competition requires factorized "
                    "causal-owner routing"
                )
            if not target_edit_phrases or not target_preserve_phrases:
                raise ValueError(
                    "Target semantic competition requires non-empty edit "
                    "and preserve phrase groups"
                )
            if not 0.0 <= target_semantic_margin < 1.0:
                raise ValueError(
                    "Target semantic margin must lie in [0, 1)"
                )
            if not 0.0 <= target_semantic_min_confidence <= 1.0:
                raise ValueError(
                    "Target semantic confidence must lie in [0, 1]"
                )
        if role_fixed_native_history:
            if not factorized_native_target_history:
                raise ValueError(
                    "Role-fixed native history requires native target "
                    "history as its exact fallback"
                )
            if routing_mode != "hand_role_factorized_causal_owner_kv":
                raise ValueError(
                    "Role-fixed native history requires factorized "
                    "causal-owner routing"
                )
            if not role_native_history_layers or len(set(
                role_native_history_layers
            )) != len(role_native_history_layers):
                raise ValueError(
                    "Native history layers must be unique and nonempty"
                )
            if any(
                layer < 0 or layer >= self.num_transformer_blocks
                for layer in role_native_history_layers
            ):
                raise ValueError(
                    "Native history layers must be valid transformer layers"
                )
            if native_history_max_tokens_per_frame <= 0:
                raise ValueError(
                    "Native history token budget must be positive"
                )
            if native_history_topk <= 0:
                raise ValueError("Native history topk must be positive")
            if not -1.0 < native_history_min_similarity < 1.0:
                raise ValueError(
                    "Native history similarity must lie in (-1, 1)"
                )
            for name, value in (
                (
                    "native_history_min_write_confidence",
                    native_history_min_write_confidence,
                ),
                (
                    "native_history_min_query_confidence",
                    native_history_min_query_confidence,
                ),
            ):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name} must lie in [0, 1]")
            if not math.isfinite(native_history_canonical_logit_bias):
                raise ValueError(
                    "Native history canonical logit bias must be finite"
                )
            if not -1.0 < native_history_min_part_similarity < 1.0:
                raise ValueError(
                    "Native history part similarity must lie in (-1, 1)"
                )
            if not (
                0.0 <= native_history_part_similarity_margin <= 2.0
            ):
                raise ValueError(
                    "Native history part similarity margin must lie in "
                    "[0, 2]"
                )
            if not 0.0 <= native_history_part_bias_strength <= 4.0:
                raise ValueError(
                    "Native history part bias strength must lie in [0, 4]"
                )
            if not 0.0 <= native_history_part_refinement_ratio <= 1.0:
                raise ValueError(
                    "Native history part refinement ratio must lie in "
                    "[0, 1]"
                )
            if native_history_owner_max_missing_frames < 0:
                raise ValueError(
                    "Native-history owner missing-frame limit must be "
                    "non-negative"
                )
            if not (
                0.0
                <= native_history_verified_source_suppression
                <= 1.0
            ):
                raise ValueError(
                    "Native-history verified source suppression must "
                    "lie in [0, 1]"
                )
            if (
                len(set(role_native_history_bypass_blocks))
                != len(role_native_history_bypass_blocks)
                or any(
                    block < 0
                    for block in role_native_history_bypass_blocks
                )
            ):
                raise ValueError(
                    "Native history bypass blocks must be unique and "
                    "non-negative"
                )
        if native_history_transactional_owner:
            if not role_fixed_native_history:
                raise ValueError(
                    "Transactional native owner requires role-fixed "
                    "native history"
                )
        if (
            native_history_consistent_transaction
            and not native_history_transactional_owner
        ):
            raise ValueError(
                "Consistent native transaction requires transactional "
                "owner reads"
            )
        if (
            native_history_verified_attention_authority
            and not native_history_consistent_transaction
        ):
            raise ValueError(
                "Verified attention authority requires a consistent native "
                "transaction"
            )
        if not (
            0.0 <= native_history_attention_authority_strength <= 1.0
        ):
            raise ValueError(
                "Native-history attention authority strength must lie in "
                "[0, 1]"
            )
        if (
            native_history_payload_invariant_lineage
            and not native_history_transactional_owner
        ):
            raise ValueError(
                "Payload-invariant native lineage requires "
                "transactional owner reads"
            )
        if not 0.0 <= native_history_payload_blend_strength <= 1.0:
            raise ValueError(
                "Native-history payload blend strength must lie in [0, 1]"
            )
        if native_history_recent_entry_bridge:
            if not native_history_consistent_transaction:
                raise ValueError(
                    "Recent-entry native history requires a consistent "
                    "transaction"
                )
            if native_history_payload_invariant_lineage:
                raise ValueError(
                    "Recent-entry native history cannot use payload-invariant "
                    "lineage"
                )
            validate_recent_entry_hand_only_contract(
                enabled=True,
                routing_mode=routing_mode,
                hand_only_mask=hand_only_mask,
                oracle_object_mask=oracle_object_mask,
                oracle_source_owner_mask=oracle_source_owner_mask,
                oracle_source_owner_full_mask=(
                    oracle_source_owner_full_mask
                ),
            )
        if native_history_motion_owner_dense_read and not (
            native_history_recent_entry_bridge
            and native_history_consistent_transaction
            and motion_geometry_owner
        ):
            raise ValueError(
                "Motion-owner dense native read requires the recent-entry "
                "consistent transaction and motion geometry owner"
            )
        if not 0.0 <= native_history_entry_bridge_strength <= 1.0:
            raise ValueError(
                "Native-history entry bridge strength must lie in [0, 1]"
            )
        if native_history_dual_evidence_arbitration and not (
            native_history_recent_entry_bridge
            and native_history_consistent_transaction
        ):
            raise ValueError(
                "Dual-evidence native history requires the consistent "
                "recent-entry bridge"
            )
        if native_history_token_atomic_payload and not (
            native_history_recent_entry_bridge
            and native_history_consistent_transaction
            and native_history_motion_owner_dense_read
        ):
            raise ValueError(
                "Token-atomic native payloads require the consistent "
                "dense motion-owner recent bridge"
            )
        if native_history_topology_complete_read and not (
            native_history_token_atomic_payload and motion_geometry_owner
        ):
            raise ValueError(
                "Topology-complete native reads require token-atomic "
                "payloads and motion geometry ownership"
            )
        if native_history_persistent_residual_upsert and not (
            native_history_token_atomic_payload and motion_geometry_owner
        ):
            raise ValueError(
                "Persistent residual native history requires token-atomic "
                "payloads and motion geometry ownership"
            )
        if native_history_last_trusted_appearance and not (
            native_history_persistent_residual_upsert
            and native_history_transactional_owner
        ):
            raise ValueError(
                "Last-trusted appearance requires persistent residual "
                "upserts and transactional owner arbitration"
            )
        if source_flow_role_fusion and not motion_geometry_owner:
            raise ValueError(
                "Source-flow role fusion requires motion geometry ownership"
            )
        if source_flow_verified_region and not source_flow_role_fusion:
            raise ValueError(
                "Source-flow verified region requires source-flow role "
                "fusion"
            )
        if not 0.0 <= source_flow_role_weight <= 1.0:
            raise ValueError("Source-flow role weight must lie in [0, 1]")
        if source_flow_verified_owner_radius < 0:
            raise ValueError(
                "Source-flow verified owner radius must be non-negative"
            )
        if not 0.0 <= source_flow_background_veto_threshold <= 1.0:
            raise ValueError(
                "Source-flow background veto threshold must lie in [0, 1]"
            )
        if not 0.0 <= source_flow_background_veto_min_confidence <= 1.0:
            raise ValueError(
                "Source-flow background veto confidence must lie in [0, 1]"
            )
        if native_history_flow_indexed_residual and not (
            native_history_last_trusted_appearance
            and source_flow_cache is not None
        ):
            raise ValueError(
                "Flow-indexed native residual requires last-trusted "
                "appearance and clean-source flow"
            )
        if native_history_decoupled_flow_trust and not (
            native_history_flow_indexed_residual
        ):
            raise ValueError(
                "Decoupled flow trust requires flow-indexed native residuals"
            )
        if native_history_multiframe_identity_sink and not (
            native_history_decoupled_flow_trust
        ):
            raise ValueError(
                "Multi-frame identity sink requires decoupled flow trust"
            )
        if native_history_timestep_counterfactual_memory and not (
            native_history_multiframe_identity_sink
        ):
            raise ValueError(
                "Timestep counterfactual memory requires the multi-frame "
                "identity sink contract"
            )
        if native_history_tccm_flow_radius < 0.0:
            raise ValueError(
                "TCCM flow radius must be non-negative"
            )
        if not 0.0 <= native_history_tccm_strength <= 1.0:
            raise ValueError("TCCM strength must lie in [0, 1]")
        if native_history_tccm_max_error_ratio <= 0.0:
            raise ValueError(
                "TCCM maximum error ratio must be positive"
            )
        if native_history_multiframe_sink_topk_per_frame <= 0:
            raise ValueError(
                "Multi-frame sink top-k per frame must be positive"
            )
        if not math.isfinite(
            native_history_multiframe_sink_source_logit_bias
        ):
            raise ValueError(
                "Multi-frame sink source logit bias must be finite"
            )
        if not 0.0 <= native_history_multiframe_sink_strength <= 1.0:
            raise ValueError(
                "Multi-frame sink strength must lie in [0, 1]"
            )
        if not 0.0 <= native_history_flow_min_confidence <= 1.0:
            raise ValueError(
                "Native flow-ledger confidence must lie in [0, 1]"
            )
        if not (
            -1.0 <= native_history_residual_update_min_cosine <= 1.0
        ):
            raise ValueError(
                "Native-history residual update cosine must lie in [-1, 1]"
            )
        if not (
            0.0
            <= native_history_residual_update_min_magnitude_ratio
            <= 1.0
        ):
            raise ValueError(
                "Native-history residual magnitude ratio must lie in [0, 1]"
            )
        if not 0.0 <= native_history_min_payload_consistency <= 1.0:
            raise ValueError(
                "Native-history payload consistency must lie in [0, 1]"
            )
        if not (
            0.0
            <= native_history_dense_recent_min_residual_consensus
            <= 1.0
        ):
            raise ValueError(
                "Dense recent residual consensus must lie in [0, 1]"
            )
        rollout_role_native_kv_history = _role_native_kv_history
        if (
            role_fixed_native_history
            and rollout_role_native_kv_history is None
        ):
            rollout_role_native_kv_history = (
                RoleConditionedNativeKVHistory(
                    layers=role_native_history_layers,
                    tokens_per_frame=self.frame_seq_length,
                    max_tokens_per_frame=(
                        native_history_max_tokens_per_frame
                    ),
                    min_write_confidence=(
                        native_history_min_write_confidence
                    ),
                    payload_invariant_lineage=(
                        native_history_payload_invariant_lineage
                    ),
                    transactional_compact_recent=(
                        native_history_consistent_transaction
                        and not native_history_recent_entry_bridge
                    ),
                    transactional_dense_recent=(
                        native_history_recent_entry_bridge
                    ),
                    token_atomic_dense_recent=(
                        native_history_token_atomic_payload
                    ),
                    persistent_residual_upsert=(
                        native_history_persistent_residual_upsert
                    ),
                    last_trusted_residual_lineage=(
                        native_history_last_trusted_appearance
                    ),
                    flow_indexed_residual_ledger=(
                        native_history_flow_indexed_residual
                    ),
                    decoupled_flow_trust=(
                        native_history_decoupled_flow_trust
                    ),
                    multiframe_identity_sink=(
                        native_history_multiframe_identity_sink
                    ),
                    timestep_counterfactual_memory=(
                        native_history_timestep_counterfactual_memory
                    ),
                    source_flow_cache=source_flow_cache,
                    flow_min_confidence=(
                        native_history_flow_min_confidence
                    ),
                    residual_update_min_cosine=(
                        native_history_residual_update_min_cosine
                    ),
                    residual_update_min_magnitude_ratio=(
                        native_history_residual_update_min_magnitude_ratio
                    ),
                    dense_recent_min_residual_consensus=(
                        native_history_dense_recent_min_residual_consensus
                    ),
                    min_lineage_similarity=native_history_min_similarity,
                )
            )
        if (
            paired_memory_value_projection
            or paired_memory_query_gated_projection
            or paired_memory_disable_persistent_projection
            or paired_memory_source_suppression != 0.0
            or paired_memory_interior_projection
            or paired_memory_coordinate_radius != 0.0
            or paired_memory_min_residual_consensus != 0.0
            or paired_memory_source_part_consistency
            or paired_memory_min_part_similarity != 0.45
            or paired_memory_part_similarity_margin != 0.08
            or paired_memory_source_transport
            or paired_memory_single_confidence
            or paired_memory_owner_attached_boundary
            or paired_memory_dual_timescale_anchor
            or paired_memory_canonical_key_anchor
            or paired_memory_transport_min_similarity != 0.10
            or paired_memory_transport_coordinate_radius != 0.60
            or paired_memory_transport_cycle_radius != 0.20
            or paired_memory_transport_min_confidence != 0.05
        ) and not causal_paired_edit_memory:
            raise ValueError(
                "Paired-memory projection and source suppression require "
                "causal paired edit memory"
            )
        if (
            paired_memory_query_gated_projection
            and not paired_memory_value_projection
        ):
            raise ValueError(
                "Query-gated projection requires paired value projection"
            )
        if (
            paired_memory_disable_persistent_projection
            and not paired_memory_value_projection
        ):
            raise ValueError(
                "Disabling persistent projection requires paired value "
                "projection"
            )
        if (
            paired_memory_query_gated_projection
            and not paired_memory_disable_persistent_projection
        ):
            raise ValueError(
                "Query-gated projection requires persistent cache "
                "projection to be disabled until historical owner "
                "metadata is available"
            )
        if (
            paired_memory_source_part_consistency
            and not paired_memory_query_gated_projection
        ):
            raise ValueError(
                "Source-part consistency requires query-gated projection"
            )
        if paired_memory_source_transport and not causal_paired_edit_memory:
            raise ValueError(
                "Source-transported memory requires causal paired edit "
                "memory"
            )
        if (
            paired_memory_owner_attached_boundary
            and not paired_memory_interior_projection
        ):
            raise ValueError(
                "Owner-attached boundary memory requires the paired "
                "projection gate"
            )
        if (
            paired_memory_single_confidence
            and not paired_memory_query_gated_projection
        ):
            raise ValueError(
                "Single-confidence memory requires query-gated "
                "projection"
            )
        if paired_memory_owner_attached_boundary and (
            not paired_memory_source_part_consistency
            or not paired_memory_source_transport
        ):
            raise ValueError(
                "Owner-attached boundary memory requires source-part "
                "consistency and source transport"
            )
        if paired_memory_dual_timescale_anchor and not (
            paired_memory_value_projection
            and paired_memory_query_gated_projection
            and paired_memory_disable_persistent_projection
            and paired_memory_single_confidence
            and paired_memory_source_transport
        ):
            raise ValueError(
                "Dual-timescale anchor requires transient, query-gated, "
                "single-confidence source-transported paired memory"
            )
        if paired_memory_canonical_key_anchor and not (
            paired_memory_dual_timescale_anchor
            and paired_memory_first_block_replay
        ):
            raise ValueError(
                "Immutable canonical-key anchor requires dual-timescale "
                "attention and first-block replay"
            )
        if not identity_memory_layers:
            raise ValueError("Immutable target layers must not be empty")
        if len(set(identity_memory_layers)) != len(identity_memory_layers):
            raise ValueError("Immutable target layers must be unique")
        if any(
            layer < 0 or layer >= self.num_transformer_blocks
            for layer in identity_memory_layers
        ):
            raise ValueError(
                "Immutable target layers must contain valid transformer layers"
            )
        if immutable_target_num_prototypes <= 0:
            raise ValueError(
                "immutable_target_num_prototypes must be positive"
            )
        if causal_paired_edit_memory:
            if routing_mode != "hand_role_factorized_causal_owner_kv":
                raise ValueError(
                    "Paired edit memory requires factorized "
                    "causal-owner routing"
                )
            if not factorized_native_target_history:
                raise ValueError(
                    "Paired edit memory requires native target-history "
                    "fallback"
                )
            if not paired_edit_memory_layers:
                raise ValueError(
                    "Paired edit-memory layers must not be empty"
                )
            if (
                len(set(paired_edit_memory_layers))
                != len(paired_edit_memory_layers)
                or any(
                    layer < 0 or layer >= self.num_transformer_blocks
                    for layer in paired_edit_memory_layers
                )
            ):
                raise ValueError(
                    "Paired edit-memory layers must contain unique "
                    "transformer indices"
                )
            if paired_memory_max_tokens <= 0:
                raise ValueError(
                    "paired_memory_max_tokens must be positive"
                )
            if paired_memory_max_tokens_per_block <= 0:
                raise ValueError(
                    "paired_memory_max_tokens_per_block must be positive"
                )
            if paired_memory_topk <= 0:
                raise ValueError(
                    "paired_memory_topk must be positive"
                )
            if not -1.0 < paired_memory_min_similarity < 1.0:
                raise ValueError(
                    "paired_memory_min_similarity must lie in (-1, 1)"
                )
            if not 0.0 <= paired_memory_min_commit_confidence <= 1.0:
                raise ValueError(
                    "paired_memory_min_commit_confidence must lie in "
                    "[0, 1]"
                )
            if paired_memory_coordinate_bias < 0.0:
                raise ValueError(
                    "paired_memory_coordinate_bias must be non-negative"
                )
            if paired_memory_coordinate_radius < 0.0:
                raise ValueError(
                    "paired_memory_coordinate_radius must be non-negative"
                )
            if not (
                0.0 <= paired_memory_min_residual_consensus < 1.0
            ):
                raise ValueError(
                    "paired_memory_min_residual_consensus must lie in "
                    "[0, 1)"
                )
            if not 0.0 <= paired_memory_read_strength <= 1.0:
                raise ValueError(
                    "paired_memory_read_strength must lie in [0, 1]"
                )
            if not 0.0 <= paired_memory_source_suppression <= 1.0:
                raise ValueError(
                    "paired_memory_source_suppression must lie in [0, 1]"
                )
        if immutable_target_value_mode not in {
            "residual", "subspace", "absolute"
        }:
            raise ValueError(
                "immutable_target_value_mode must be 'residual', "
                "'subspace', or 'absolute'"
            )
        if not 0.0 <= factorized_geometry_strength <= 1.0:
            raise ValueError(
                "factorized_geometry_strength must lie in [0, 1]"
            )
        if (
            factorized_orthogonal_geometry
            and not factorized_immutable_target_memory
        ):
            raise ValueError(
                "Orthogonal geometry requires immutable target memory"
            )
        if (
            first_chunk_identity_replay
            or paired_memory_first_block_replay
        ) and rollout_chunk_size <= 0:
            raise ValueError(
                "First-block replay requires rollout inference"
            )
        if (
            paired_memory_first_block_replay
            and not causal_paired_edit_memory
        ):
            raise ValueError(
                "Paired-memory first-block replay requires causal paired "
                "edit memory"
            )
        if source_coordinate_identity and not first_chunk_identity_replay:
            raise ValueError(
                "Source-coordinate identity requires first-chunk "
                "identity replay"
            )
        if (
            oracle_source_owner_mask is not None
            and not source_coordinate_identity
            and routing_mode
            != "hand_role_factorized_causal_owner_kv"
        ):
            raise ValueError(
                "Oracle source owner mask requires source-coordinate "
                "identity or factorized causal-owner routing"
            )
        identity_routing_modes = {
            "hand_role_bayes_flow_identity_kv",
            "hand_role_bayes_flow_tokenprop_kv",
            "hand_role_bayes_flow_customized_kv",
        }
        if factorized_immutable_target_memory:
            identity_routing_modes.add(
                "hand_role_factorized_causal_owner_kv"
            )
        if (
            first_chunk_identity_replay
            and routing_mode not in identity_routing_modes
        ):
            raise ValueError(
                "First-chunk identity replay requires an identity "
                "routing mode"
            )
        rollout_hand_role_inferencer = _hand_role_inferencer
        if (
            rollout_hand_role_inferencer is None
            and routing_mode in {
                "hand_role_adaptive_kv",
                "hand_role_posterior_flow_kv",
                "hand_role_bayes_flow_kv",
                "hand_role_bayes_flow_dual_kv",
                "hand_role_bayes_flow_consolidated_kv",
                "hand_role_bayes_flow_commitment_kv",
                "hand_role_bayes_flow_identity_kv",
                "hand_role_bayes_flow_tokenprop_kv",
                "hand_role_bayes_flow_customized_kv",
                "hand_role_factorized_bayes_kv",
                "hand_role_factorized_causal_owner_kv",
            }
        ):
            rollout_hand_role_inferencer = HandRoleInferencer(
                hand_proximity_radius=hand_proximity_radius,
                propagation_steps=hand_propagation_steps,
                max_object_coverage=hand_max_object_coverage,
                visibility_ratio=hand_visibility_ratio,
                temporal_weight=hand_temporal_weight,
                query_similarity_threshold=(
                    hand_query_similarity_threshold
                ),
                field_quantile_low=hand_field_quantile_low,
                field_quantile_high=hand_field_quantile_high,
                field_power=hand_field_power,
                field_weight=hand_field_weight,
                field_candidate_radius=hand_field_candidate_radius,
                connected_hysteresis=hand_connected_hysteresis,
                connected_growth_steps=hand_connected_growth_steps,
                connected_candidate_ratio=(
                    hand_connected_candidate_ratio
                ),
                soft_hand_contact=hand_causal_evidence,
                adaptive=(
                    routing_mode in {
                        "hand_role_adaptive_kv",
                        "hand_role_posterior_flow_kv",
                        "hand_role_bayes_flow_kv",
                        "hand_role_bayes_flow_dual_kv",
                        "hand_role_bayes_flow_consolidated_kv",
                        "hand_role_bayes_flow_commitment_kv",
                        "hand_role_bayes_flow_identity_kv",
                        "hand_role_bayes_flow_tokenprop_kv",
                        "hand_role_bayes_flow_customized_kv",
                        "hand_role_factorized_bayes_kv",
                        "hand_role_factorized_causal_owner_kv",
                    }
                ),
            )
        rollout_memory_consolidator = _memory_consolidator
        if (
            rollout_memory_consolidator is None
            and routing_mode in {
                "hand_role_bayes_flow_consolidated_kv",
                "hand_role_bayes_flow_commitment_kv",
                "hand_role_bayes_flow_identity_kv",
                "hand_role_bayes_flow_tokenprop_kv",
                "hand_role_bayes_flow_customized_kv",
            }
        ):
            rollout_memory_consolidator = CausalMemoryConsolidator()
        rollout_commitment_controller = _edit_commitment_controller
        if (
            rollout_commitment_controller is None
            and routing_mode in {
                "hand_role_bayes_flow_commitment_kv",
                "hand_role_bayes_flow_identity_kv",
                "hand_role_bayes_flow_tokenprop_kv",
                "hand_role_bayes_flow_customized_kv",
            }
        ):
            rollout_commitment_controller = EditCommitmentController()
        rollout_target_identity_memory = _target_identity_memory
        if (
            rollout_target_identity_memory is None
            and (
                routing_mode in {
                    "hand_role_bayes_flow_identity_kv",
                    "hand_role_bayes_flow_tokenprop_kv",
                    "hand_role_bayes_flow_customized_kv",
                }
                or factorized_immutable_target_memory
            )
        ):
            rollout_target_identity_memory = SlowTargetIdentityMemory(
                layers=identity_memory_layers,
                num_prototypes=immutable_target_num_prototypes,
                store_value_residual=(
                    source_coordinate_identity
                    or (
                        factorized_immutable_target_memory
                        and immutable_target_value_mode in {
                            "residual", "subspace"
                        }
                    )
                ),
            )
        rollout_identity_owner_tracker = _identity_owner_tracker
        if (
            rollout_identity_owner_tracker is None
            and source_coordinate_identity
        ):
            rollout_identity_owner_tracker = CausalIdentityOwnerTracker(
                max_candidates=identity_tokenprop_max_candidates,
                max_area_fraction=hand_max_object_coverage,
                recover_visibility_from_source_match=(
                    source_identity_residual_carry
                ),
            )
        rollout_causal_ownership_tracker = _causal_ownership_tracker
        if (
            rollout_causal_ownership_tracker is None
            and routing_mode == "hand_role_factorized_causal_owner_kv"
        ):
            rollout_causal_ownership_tracker = (
                MotionAwareGeometryOwnerTracker(
                    source_flow_cache,
                    bootstrap_frames=self.num_frame_per_block,
                    max_occluded_frames=(
                        identity_max_occluded_blocks
                        * self.num_frame_per_block
                    ),
                )
                if motion_geometry_owner
                else CausalObjectOwnershipTracker(
                    max_candidates=identity_tokenprop_max_candidates,
                    max_area_fraction=hand_max_object_coverage,
                    min_similarity=identity_tokenprop_min_similarity,
                    max_occluded_frames=(
                        identity_max_occluded_blocks
                        * self.num_frame_per_block
                    ),
                )
            )
        rollout_native_owner_tracker = _native_owner_tracker
        if (
            rollout_native_owner_tracker is None
            and native_history_transactional_owner
        ):
            if oracle_source_owner_full_mask is None:
                rollout_native_owner_tracker = (
                    AutomaticTransactionalOwnerTracker(
                        max_missing_frames=(
                            native_history_owner_max_missing_frames
                        ),
                        blockwise_lifecycle=(
                            native_history_consistent_transaction
                        ),
                    )
                )
            else:
                rollout_native_owner_tracker = CausalReadOnlyOwnerTracker(
                    max_candidates=identity_tokenprop_max_candidates,
                    max_area_fraction=hand_max_object_coverage,
                    min_similarity=identity_tokenprop_min_similarity,
                    max_missing_frames=(
                        native_history_owner_max_missing_frames
                    ),
                )
        rollout_paired_edit_memory = _causal_paired_edit_memory
        if (
            causal_paired_edit_memory
            and rollout_paired_edit_memory is None
        ):
            rollout_paired_edit_memory = CausalPairedEditMemory(
                layers=paired_edit_memory_layers,
                max_tokens=paired_memory_max_tokens,
                max_tokens_per_block=(
                    paired_memory_max_tokens_per_block
                ),
                min_commit_confidence=(
                    paired_memory_min_commit_confidence
                ),
                min_similarity=paired_memory_min_similarity,
                coordinate_bias=paired_memory_coordinate_bias,
                coordinate_radius=paired_memory_coordinate_radius,
                min_residual_consensus=(
                    paired_memory_min_residual_consensus
                ),
                source_part_consistency=(
                    paired_memory_source_part_consistency
                ),
                min_part_similarity=paired_memory_min_part_similarity,
                part_similarity_margin=(
                    paired_memory_part_similarity_margin
                ),
                topk=paired_memory_topk,
                source_transport=paired_memory_source_transport,
                transport_min_similarity=(
                    paired_memory_transport_min_similarity
                ),
                transport_coordinate_radius=(
                    paired_memory_transport_coordinate_radius
                ),
                transport_cycle_radius=(
                    paired_memory_transport_cycle_radius
                ),
                transport_min_confidence=(
                    paired_memory_transport_min_confidence
                ),
                single_confidence=paired_memory_single_confidence,
                immutable_canonical_key_anchor=(
                    paired_memory_canonical_key_anchor
                ),
            )
        rollout_identity_residual_carry = _identity_residual_carry
        if (
            rollout_identity_residual_carry is None
            and source_identity_residual_carry
        ):
            rollout_identity_residual_carry = (
                SourceCoordinateResidualCarry(
                    max_candidates=identity_tokenprop_max_candidates,
                )
            )
        rollout_identity_token_propagator = _identity_token_propagator
        if (
            rollout_identity_token_propagator is None
            and routing_mode == "hand_role_bayes_flow_tokenprop_kv"
        ):
            rollout_identity_token_propagator = CausalObjectTokenPropagator(
                min_similarity=identity_tokenprop_min_similarity,
                gate_strength=identity_tokenprop_gate_strength,
                max_candidates=identity_tokenprop_max_candidates,
            )
        rollout_identity_support_filter = _identity_support_filter
        if (
            rollout_identity_support_filter is None
            and routing_mode == "hand_role_bayes_flow_tokenprop_kv"
        ):
            rollout_identity_support_filter = (
                CausalConnectedSupportFilter()
            )

        total_frame_num = src_video.shape[1]
        ret_latent_list = []
        start_idx = 0

        while True:
            chunk_right_idx = start_idx + rollout_chunk_size
            if (start_idx == 0) and (independent_first_frame or triple_first_frame):
                # provide kv_cache space for image condition
                chunk_right_idx -= self.num_frame_per_block

            if start_idx == 0:
                rollout_global_frame_indices = list(
                    range(start_idx, min(chunk_right_idx, total_frame_num))
                )
                rollout_src_video = src_video[:, start_idx: chunk_right_idx]
                rollout_source_owner_mask = (
                    None
                    if oracle_source_owner_mask is None
                    else oracle_source_owner_mask[
                        :, start_idx:chunk_right_idx
                    ]
                )
                rollout_source_owner_full_mask = (
                    None
                    if oracle_source_owner_full_mask is None
                    else oracle_source_owner_full_mask[
                        :, start_idx:chunk_right_idx
                    ]
                )
                rollout_object_mask = (
                    None if oracle_object_mask is None
                    else oracle_object_mask[:, start_idx:chunk_right_idx]
                )
                rollout_hand_mask = (
                    None if oracle_hand_mask is None
                    else oracle_hand_mask[:, start_idx:chunk_right_idx]
                )
                rollout_hand_only_mask = (
                    None if hand_only_mask is None
                    else hand_only_mask[:, start_idx:chunk_right_idx]
                )
                rollout_hand_occupancy_mask = (
                    None if hand_occupancy_mask is None
                    else hand_occupancy_mask[:, start_idx:chunk_right_idx]
                )
                rollout_hand_persistent_mask = (
                    None if hand_persistent_mask is None
                    else hand_persistent_mask[:, start_idx:chunk_right_idx]
                )
            else:
                rollout_global_frame_indices = list(
                    range(
                        start_idx + rollout_overlap,
                        min(chunk_right_idx, total_frame_num),
                    )
                )
                rollout_src_video = src_video[:, start_idx + rollout_overlap: chunk_right_idx]
                rollout_source_owner_mask = (
                    None
                    if oracle_source_owner_mask is None
                    else oracle_source_owner_mask[
                        :, start_idx + rollout_overlap:chunk_right_idx
                    ]
                )
                rollout_source_owner_full_mask = (
                    None
                    if oracle_source_owner_full_mask is None
                    else oracle_source_owner_full_mask[
                        :, start_idx + rollout_overlap:chunk_right_idx
                    ]
                )
                rollout_object_mask = (
                    None if oracle_object_mask is None
                    else oracle_object_mask[:, start_idx + rollout_overlap:chunk_right_idx]
                )
                rollout_hand_mask = (
                    None if oracle_hand_mask is None
                    else oracle_hand_mask[:, start_idx + rollout_overlap:chunk_right_idx]
                )
                rollout_hand_only_mask = (
                    None if hand_only_mask is None
                    else hand_only_mask[
                        :, start_idx + rollout_overlap:chunk_right_idx
                    ]
                )
                rollout_hand_occupancy_mask = (
                    None if hand_occupancy_mask is None
                    else hand_occupancy_mask[
                        :, start_idx + rollout_overlap:chunk_right_idx
                    ]
                )
                rollout_hand_persistent_mask = (
                    None if hand_persistent_mask is None
                    else hand_persistent_mask[
                        :, start_idx + rollout_overlap:chunk_right_idx
                    ]
                )

            if start_idx == 0 and (
                first_chunk_identity_replay
                or paired_memory_first_block_replay
            ):
                if rollout_src_video.shape[1] < self.num_frame_per_block:
                    raise ValueError(
                        "First-chunk replay requires one complete model "
                        "generation block"
                    )
                proposal_frame_count = self.num_frame_per_block
                proposal_src_video = rollout_src_video[
                    :, :proposal_frame_count
                ]
                proposal_source_owner_mask = (
                    None
                    if rollout_source_owner_mask is None
                    else rollout_source_owner_mask[
                        :, :proposal_frame_count
                    ]
                )
                proposal_source_owner_full_mask = (
                    None
                    if rollout_source_owner_full_mask is None
                    else rollout_source_owner_full_mask[
                        :, :proposal_frame_count
                    ]
                )
                proposal_object_mask = (
                    None
                    if rollout_object_mask is None
                    else rollout_object_mask[:, :proposal_frame_count]
                )
                proposal_hand_mask = (
                    None
                    if rollout_hand_mask is None
                    else rollout_hand_mask[:, :proposal_frame_count]
                )
                proposal_hand_only_mask = (
                    None
                    if rollout_hand_only_mask is None
                    else rollout_hand_only_mask[:, :proposal_frame_count]
                )
                proposal_hand_occupancy_mask = (
                    None
                    if rollout_hand_occupancy_mask is None
                    else rollout_hand_occupancy_mask[
                        :, :proposal_frame_count
                    ]
                )
                proposal_hand_persistent_mask = (
                    None
                    if rollout_hand_persistent_mask is None
                    else rollout_hand_persistent_mask[
                        :, :proposal_frame_count
                    ]
                )
                proposal_memory = (
                    SlowTargetIdentityMemory(
                        layers=identity_memory_layers,
                        num_prototypes=immutable_target_num_prototypes,
                        store_value_residual=(
                            source_coordinate_identity
                            or (
                                factorized_immutable_target_memory
                                and immutable_target_value_mode in {
                                    "residual", "subspace"
                                }
                            )
                        ),
                    )
                    if first_chunk_identity_replay
                    else None
                )
                proposal_paired_memory = (
                    CausalPairedEditMemory(
                        layers=paired_edit_memory_layers,
                        max_tokens=paired_memory_max_tokens,
                        max_tokens_per_block=(
                            paired_memory_max_tokens_per_block
                        ),
                        min_commit_confidence=(
                            paired_memory_min_commit_confidence
                        ),
                        min_similarity=paired_memory_min_similarity,
                        coordinate_bias=paired_memory_coordinate_bias,
                        coordinate_radius=paired_memory_coordinate_radius,
                        min_residual_consensus=(
                            paired_memory_min_residual_consensus
                        ),
                        source_part_consistency=(
                            paired_memory_source_part_consistency
                        ),
                        min_part_similarity=(
                            paired_memory_min_part_similarity
                        ),
                        part_similarity_margin=(
                            paired_memory_part_similarity_margin
                        ),
                        topk=paired_memory_topk,
                        source_transport=paired_memory_source_transport,
                        transport_min_similarity=(
                            paired_memory_transport_min_similarity
                        ),
                        transport_coordinate_radius=(
                            paired_memory_transport_coordinate_radius
                        ),
                        transport_cycle_radius=(
                            paired_memory_transport_cycle_radius
                        ),
                        transport_min_confidence=(
                            paired_memory_transport_min_confidence
                        ),
                        single_confidence=(
                            paired_memory_single_confidence
                        ),
                        immutable_canonical_key_anchor=(
                            paired_memory_canonical_key_anchor
                        ),
                    )
                    if paired_memory_first_block_replay
                    else None
                )
                cpu_rng_state = torch.random.get_rng_state()
                cuda_rng_state = (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                )
                proposal_role_dir = (
                    os.path.join(save_role_dir, "proposal")
                    if (
                        save_role_dir is not None
                        and first_chunk_identity_replay
                    )
                    else None
                )
                print(
                    "FIRST_BLOCK_REPLAY phase=proposal "
                    f"frames={proposal_frame_count} "
                    f"identity={int(first_chunk_identity_replay)} "
                    "paired_memory="
                    f"{int(paired_memory_first_block_replay)}"
                )
                self.inference(
                    src_video=proposal_src_video,
                    src_prompts=src_prompts,
                    trg_prompts=trg_prompts,
                    src_trigger_words=src_trigger_words,
                    trg_trigger_words=trg_trigger_words,
                    return_latents=False,
                    wo_video_decode=True,
                    profile=False,
                    low_memory=low_memory,
                    independent_first_frame=(
                        independent_first_frame
                    ),
                    triple_first_frame=triple_first_frame,
                    src_initial_latent=src_initial_latent,
                    trg_initial_latent=trg_initial_latent,
                    mask_layers=mask_layers,
                    enhance_layers=enhance_layers,
                    reuse_noise_temporal_mean=reuse_noise_temporal_mean,
                    fg_scale=fg_scale,
                    fg_boost_factor=fg_boost_factor,
                    blend_power=blend_power,
                    routing_mode=routing_mode,
                    identity_first_latent_bootstrap=False,
                    object_wise_anchor_reset=False,
                    target_owned_object_handoff=False,
                    target_owned_min_similarity=(
                        target_owned_min_similarity
                    ),
                    factorized_target_identity=False,
                    factorized_immutable_target_memory=(
                        factorized_immutable_target_memory
                    ),
                    factorized_native_target_history=(
                        factorized_native_target_history
                    ),
                    factorized_owner_source_block=(
                        factorized_owner_source_block
                    ),
                    target_semantic_competition=(
                        target_semantic_competition
                    ),
                    target_edit_phrases=target_edit_phrases,
                    target_preserve_phrases=target_preserve_phrases,
                    target_semantic_margin=target_semantic_margin,
                    target_semantic_min_confidence=(
                        target_semantic_min_confidence
                    ),
                    causal_paired_edit_memory=(
                        paired_memory_first_block_replay
                    ),
                    paired_memory_layers=(
                        paired_edit_memory_layers
                        if paired_memory_first_block_replay
                        else None
                    ),
                    paired_memory_max_tokens=(
                        paired_memory_max_tokens
                        if paired_memory_first_block_replay
                        else 1536
                    ),
                    paired_memory_max_tokens_per_block=(
                        paired_memory_max_tokens_per_block
                        if paired_memory_first_block_replay
                        else 192
                    ),
                    paired_memory_topk=(
                        paired_memory_topk
                        if paired_memory_first_block_replay
                        else 8
                    ),
                    paired_memory_min_similarity=(
                        paired_memory_min_similarity
                        if paired_memory_first_block_replay
                        else 0.35
                    ),
                    paired_memory_min_commit_confidence=(
                        paired_memory_min_commit_confidence
                        if paired_memory_first_block_replay
                        else 0.20
                    ),
                    paired_memory_coordinate_bias=(
                        paired_memory_coordinate_bias
                        if paired_memory_first_block_replay
                        else 1.0
                    ),
                    paired_memory_coordinate_radius=(
                        paired_memory_coordinate_radius
                        if paired_memory_first_block_replay
                        else 0.0
                    ),
                    paired_memory_min_residual_consensus=(
                        paired_memory_min_residual_consensus
                        if paired_memory_first_block_replay
                        else 0.0
                    ),
                    paired_memory_source_part_consistency=(
                        paired_memory_first_block_replay
                        and paired_memory_source_part_consistency
                    ),
                    paired_memory_min_part_similarity=(
                        paired_memory_min_part_similarity
                        if paired_memory_first_block_replay
                        else 0.45
                    ),
                    paired_memory_part_similarity_margin=(
                        paired_memory_part_similarity_margin
                        if paired_memory_first_block_replay
                        else 0.08
                    ),
                    paired_memory_read_strength=(
                        paired_memory_read_strength
                        if paired_memory_first_block_replay
                        else 0.35
                    ),
                    paired_memory_value_projection=(
                        paired_memory_first_block_replay
                        and paired_memory_value_projection
                    ),
                    paired_memory_query_gated_projection=(
                        paired_memory_first_block_replay
                        and paired_memory_query_gated_projection
                    ),
                    paired_memory_disable_persistent_projection=(
                        paired_memory_first_block_replay
                        and paired_memory_disable_persistent_projection
                    ),
                    paired_memory_source_suppression=(
                        paired_memory_source_suppression
                        if paired_memory_first_block_replay
                        else 0.0
                    ),
                    paired_memory_interior_projection=(
                        paired_memory_first_block_replay
                        and paired_memory_interior_projection
                    ),
                    paired_memory_source_transport=(
                        paired_memory_first_block_replay
                        and paired_memory_source_transport
                    ),
                    paired_memory_single_confidence=(
                        paired_memory_first_block_replay
                        and paired_memory_single_confidence
                    ),
                    paired_memory_owner_attached_boundary=(
                        paired_memory_first_block_replay
                        and paired_memory_owner_attached_boundary
                    ),
                    paired_memory_dual_timescale_anchor=(
                        paired_memory_first_block_replay
                        and paired_memory_dual_timescale_anchor
                    ),
                    paired_memory_canonical_key_anchor=(
                        paired_memory_first_block_replay
                        and paired_memory_canonical_key_anchor
                    ),
                    role_fixed_native_history=role_fixed_native_history,
                    native_history_layers=role_native_history_layers,
                    native_history_max_tokens_per_frame=(
                        native_history_max_tokens_per_frame
                    ),
                    native_history_topk=native_history_topk,
                    native_history_min_similarity=(
                        native_history_min_similarity
                    ),
                    native_history_min_write_confidence=(
                        native_history_min_write_confidence
                    ),
                    native_history_min_query_confidence=(
                        native_history_min_query_confidence
                    ),
                    native_history_canonical_logit_bias=(
                        native_history_canonical_logit_bias
                    ),
                    native_history_coalesce_bootstrap_time=(
                        native_history_coalesce_bootstrap_time
                    ),
                    native_history_bypass_blocks=(
                        role_native_history_bypass_blocks
                    ),
                    native_history_source_part_consistency=(
                        native_history_source_part_consistency
                    ),
                    native_history_min_part_similarity=(
                        native_history_min_part_similarity
                    ),
                    native_history_part_similarity_margin=(
                        native_history_part_similarity_margin
                    ),
                    native_history_part_bias_strength=(
                        native_history_part_bias_strength
                    ),
                    native_history_part_refinement_ratio=(
                        native_history_part_refinement_ratio
                    ),
                    native_history_transactional_owner=(
                        native_history_transactional_owner
                    ),
                    native_history_consistent_transaction=(
                        native_history_consistent_transaction
                    ),
                    native_history_verified_attention_authority=(
                        native_history_verified_attention_authority
                    ),
                    native_history_attention_authority_strength=(
                        native_history_attention_authority_strength
                    ),
                    native_history_payload_invariant_lineage=(
                        native_history_payload_invariant_lineage
                    ),
                    native_history_payload_blend_strength=(
                        native_history_payload_blend_strength
                    ),
                    native_history_recent_entry_bridge=(
                        native_history_recent_entry_bridge
                    ),
                    native_history_motion_owner_dense_read=(
                        native_history_motion_owner_dense_read
                    ),
                    native_history_entry_bridge_strength=(
                        native_history_entry_bridge_strength
                    ),
                    native_history_dual_evidence_arbitration=(
                        native_history_dual_evidence_arbitration
                    ),
                    native_history_token_atomic_payload=(
                        native_history_token_atomic_payload
                    ),
                    native_history_persistent_residual_upsert=(
                        native_history_persistent_residual_upsert
                    ),
                    native_history_last_trusted_appearance=(
                        native_history_last_trusted_appearance
                    ),
                    native_history_flow_indexed_residual=(
                        native_history_flow_indexed_residual
                    ),
                    native_history_decoupled_flow_trust=(
                        native_history_decoupled_flow_trust
                    ),
                    native_history_multiframe_identity_sink=(
                        native_history_multiframe_identity_sink
                    ),
                    native_history_multiframe_sink_topk_per_frame=(
                        native_history_multiframe_sink_topk_per_frame
                    ),
                    native_history_multiframe_sink_source_logit_bias=(
                        native_history_multiframe_sink_source_logit_bias
                    ),
                    native_history_multiframe_sink_strength=(
                        native_history_multiframe_sink_strength
                    ),
                    native_history_timestep_counterfactual_memory=(
                        native_history_timestep_counterfactual_memory
                    ),
                    native_history_tccm_flow_radius=(
                        native_history_tccm_flow_radius
                    ),
                    native_history_tccm_strength=(
                        native_history_tccm_strength
                    ),
                    native_history_tccm_max_error_ratio=(
                        native_history_tccm_max_error_ratio
                    ),
                    native_history_flow_min_confidence=(
                        native_history_flow_min_confidence
                    ),
                    native_history_residual_update_min_cosine=(
                        native_history_residual_update_min_cosine
                    ),
                    native_history_residual_update_min_magnitude_ratio=(
                        native_history_residual_update_min_magnitude_ratio
                    ),
                    native_history_topology_complete_read=(
                        native_history_topology_complete_read
                    ),
                    native_history_min_payload_consistency=(
                        native_history_min_payload_consistency
                    ),
                    native_history_dense_recent_min_residual_consensus=(
                        native_history_dense_recent_min_residual_consensus
                    ),
                    native_history_owner_max_missing_frames=(
                        native_history_owner_max_missing_frames
                    ),
                    native_history_verified_source_suppression=(
                        native_history_verified_source_suppression
                    ),
                    paired_memory_transport_min_similarity=(
                        paired_memory_transport_min_similarity
                    ),
                    paired_memory_transport_coordinate_radius=(
                        paired_memory_transport_coordinate_radius
                    ),
                    paired_memory_transport_cycle_radius=(
                        paired_memory_transport_cycle_radius
                    ),
                    paired_memory_transport_min_confidence=(
                        paired_memory_transport_min_confidence
                    ),
                    immutable_target_layers=identity_memory_layers,
                    immutable_target_num_prototypes=(
                        immutable_target_num_prototypes
                    ),
                    immutable_target_value_mode=(
                        immutable_target_value_mode
                    ),
                    immutable_target_hard_owner=(
                        immutable_target_hard_owner
                    ),
                    factorized_orthogonal_geometry=False,
                    factorized_geometry_strength=0.0,
                    identity_correction_strength=(
                        identity_correction_strength
                    ),
                    identity_visibility_lifecycle=False,
                    identity_max_occluded_blocks=(
                        identity_max_occluded_blocks
                    ),
                    # Identity replay needs its legacy decomposition to
                    # construct the immutable identity anchor. Paired-memory
                    # replay already restricts writes with causal ownership,
                    # target-memory action and the role-aware interior gate;
                    # enabling the identity-only option here is invalid for
                    # factorized native-history routing.
                    appearance_leakage_decomposition=(
                        first_chunk_identity_replay
                    ),
                    source_coordinate_identity=source_coordinate_identity,
                    identity_source_suppression=0.0,
                    identity_support_floor=identity_support_floor,
                    source_identity_residual_carry=False,
                    identity_residual_carry_strength=0.0,
                    source_owner_residual_constraint=False,
                    identity_residual_constraint_strength=0.0,
                    identity_residual_constraint_power=(
                        identity_residual_constraint_power
                    ),
                    source_owner_geometry_envelope=(
                        source_owner_geometry_envelope
                    ),
                    source_geometry_strength=source_geometry_strength,
                    source_geometry_power=source_geometry_power,
                    source_geometry_margin=source_geometry_margin,
                    ignition_hand_exclusion_radius=(
                        ignition_hand_exclusion_radius
                    ),
                    ignition_contact_radius=ignition_contact_radius,
                    oracle_source_owner_mask=(
                        proposal_source_owner_mask
                    ),
                    oracle_source_owner_full_mask=(
                        proposal_source_owner_full_mask
                    ),
                    source_owner_prepool_hand_exclusion=(
                        source_owner_prepool_hand_exclusion
                    ),
                    causal_owner_consistent_kv_metadata=(
                        causal_owner_consistent_kv_metadata
                    ),
                    factorized_source_coordinate_target_delta=(
                        factorized_source_coordinate_target_delta
                    ),
                    factorized_owner_complement_source=(
                        factorized_owner_complement_source
                    ),
                    factorized_owner_complement_margin=(
                        factorized_owner_complement_margin
                    ),
                    factorized_owner_complement_min_preserve_confidence=(
                        factorized_owner_complement_min_preserve_confidence
                    ),
                    oracle_object_mask=proposal_object_mask,
                    oracle_hand_mask=proposal_hand_mask,
                    hand_only_mask=proposal_hand_only_mask,
                    hand_occupancy_mask=proposal_hand_occupancy_mask,
                    hand_persistent_mask=proposal_hand_persistent_mask,
                    hand_causal_evidence=hand_causal_evidence,
                    motion_geometry_owner=motion_geometry_owner,
                    source_flow_cache=source_flow_cache,
                    source_flow_role_fusion=source_flow_role_fusion,
                    source_flow_role_weight=source_flow_role_weight,
                    source_flow_verified_region=(
                        source_flow_verified_region
                    ),
                    source_flow_verified_owner_radius=(
                        source_flow_verified_owner_radius
                    ),
                    source_flow_background_veto_threshold=(
                        source_flow_background_veto_threshold
                    ),
                    source_flow_background_veto_min_confidence=(
                        source_flow_background_veto_min_confidence
                    ),
                    soft_region_modulation=soft_region_modulation,
                    soft_region_blend_strength=soft_region_blend_strength,
                    first_block_identity_anchor=first_block_identity_anchor,
                identity_anchor_scale=identity_anchor_scale,
                suppress_source_bg_value=suppress_source_bg_value,
                projected_source_residual=projected_source_residual,
                    global_frame_indices=(
                        rollout_global_frame_indices[:proposal_frame_count]
                    ),
                    role_boundary_radius=role_boundary_radius,
                    contact_target_weight=contact_target_weight,
                    posterior_flow_mode=posterior_flow_mode,
                    posterior_flow_use_field=posterior_flow_use_field,
                    hand_posterior_threshold=hand_posterior_threshold,
                    hand_max_object_coverage=hand_max_object_coverage,
                    hand_proximity_radius=hand_proximity_radius,
                    hand_propagation_steps=hand_propagation_steps,
                    hand_connected_hysteresis=(
                        hand_connected_hysteresis
                    ),
                    hand_connected_growth_steps=(
                        hand_connected_growth_steps
                    ),
                    hand_connected_candidate_ratio=(
                        hand_connected_candidate_ratio
                    ),
                    hand_visibility_ratio=hand_visibility_ratio,
                    hand_temporal_weight=hand_temporal_weight,
                    hand_query_similarity_threshold=(
                        hand_query_similarity_threshold
                    ),
                    hand_query_layers=hand_query_layers,
                    hand_field_quantile_low=hand_field_quantile_low,
                    hand_field_quantile_high=hand_field_quantile_high,
                    hand_field_power=hand_field_power,
                    hand_field_weight=hand_field_weight,
                    hand_field_candidate_radius=(
                        hand_field_candidate_radius
                    ),
                    hand_field_update_mode=hand_field_update_mode,
                    identity_tokenprop_min_similarity=(
                        identity_tokenprop_min_similarity
                    ),
                    identity_tokenprop_gate_strength=(
                        identity_tokenprop_gate_strength
                    ),
                    identity_tokenprop_max_candidates=(
                        identity_tokenprop_max_candidates
                    ),
                    committed_memory_feedback_strength=(
                        committed_memory_feedback_strength
                    ),
                    contact_graph_mode=contact_graph_mode,
                    contact_graph_topk=contact_graph_topk,
                    contact_graph_radius=contact_graph_radius,
                    contact_graph_min_confidence=(
                        contact_graph_min_confidence
                    ),
                    contact_graph_strength=contact_graph_strength,
                    contact_graph_layer_start=(
                        contact_graph_layer_start
                    ),
                    contact_graph_layer_end=contact_graph_layer_end,
                    contact_graph_seed=contact_graph_seed,
                    save_role_dir=proposal_role_dir,
                    _target_identity_memory=proposal_memory,
                    _causal_paired_edit_memory=(
                        proposal_paired_memory
                    ),
                    _native_owner_tracker=(
                        (
                            AutomaticTransactionalOwnerTracker(
                                max_missing_frames=(
                                    native_history_owner_max_missing_frames
                                ),
                                blockwise_lifecycle=(
                                    native_history_consistent_transaction
                                ),
                            )
                            if oracle_source_owner_full_mask is None
                            else CausalReadOnlyOwnerTracker(
                                max_candidates=(
                                    identity_tokenprop_max_candidates
                                ),
                                max_area_fraction=hand_max_object_coverage,
                                min_similarity=(
                                    identity_tokenprop_min_similarity
                                ),
                                max_missing_frames=(
                                    native_history_owner_max_missing_frames
                                ),
                            )
                        )
                        if native_history_transactional_owner
                        else None
                    ),
                    # Proposal ownership is source-derived as well, but its
                    # causal state must not jump from proposal frame 8 back
                    # to replay frame 0.  Replay starts a fresh source track.
                    _identity_owner_tracker=(
                        CausalIdentityOwnerTracker(
                            max_candidates=(
                                identity_tokenprop_max_candidates
                            ),
                            max_area_fraction=(
                                hand_max_object_coverage
                            ),
                            recover_visibility_from_source_match=(
                                source_identity_residual_carry
                            ),
                        )
                        if source_coordinate_identity
                        else None
                    ),
                )
                if proposal_memory is not None:
                    proposal_memory.promote_adaptive_to_replay_anchor()
                    rollout_target_identity_memory = proposal_memory
                if proposal_paired_memory is not None:
                    if not proposal_paired_memory.has_state():
                        raise RuntimeError(
                            "Paired first-block proposal produced no "
                            "canonical memory"
                        )
                    rollout_paired_edit_memory = proposal_paired_memory
                torch.random.set_rng_state(cpu_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_state)
                replay_memory = (
                    proposal_paired_memory
                    if proposal_paired_memory is not None
                    else proposal_memory
                )
                replay_states = replay_memory.export()
                replay_valid_slots_min = min(
                    int(
                        (state.evidence > replay_memory.eps)
                        .sum(dim=-1)
                        .min()
                        .item()
                    )
                    for state in replay_states.values()
                )
                replay_evidence_mean = torch.stack([
                    state.evidence.float().mean()
                    for state in replay_states.values()
                ]).mean().item()
                print(
                    "FIRST_BLOCK_REPLAY phase=commit "
                    f"layers={len(replay_states)} "
                    f"valid_slots_min={replay_valid_slots_min} "
                    f"evidence_mean={replay_evidence_mean:.4f} "
                    + (
                        "memory=source_addressed_target_residual"
                        if proposal_paired_memory is not None
                        else "memory=frozen_target_identity"
                    )
                )

            # inference
            _, rollout_latent = self.inference(
                src_video=rollout_src_video,
                src_prompts=src_prompts,
                trg_prompts=trg_prompts,
                src_trigger_words=src_trigger_words,
                trg_trigger_words=trg_trigger_words,

                return_latents=True,
                wo_video_decode=True,

                profile=profile,
                low_memory=low_memory,

                independent_first_frame=independent_first_frame if start_idx == 0 else False,
                triple_first_frame=triple_first_frame if start_idx == 0 else False,

                src_initial_latent=src_initial_latent,
                trg_initial_latent=trg_initial_latent,

                mask_layers=mask_layers,
                enhance_layers=enhance_layers,
                reuse_noise_temporal_mean=reuse_noise_temporal_mean,

                fg_scale=fg_scale,
                fg_boost_factor=fg_boost_factor,

                blend_power=blend_power,
                routing_mode=routing_mode,
                identity_first_latent_bootstrap=(
                    identity_first_latent_bootstrap
                ),
                object_wise_anchor_reset=object_wise_anchor_reset,
                target_owned_object_handoff=(
                    target_owned_object_handoff
                ),
                target_owned_min_similarity=(
                    target_owned_min_similarity
                ),
                factorized_target_identity=(
                    factorized_target_identity
                ),
                factorized_immutable_target_memory=(
                    factorized_immutable_target_memory
                ),
                factorized_native_target_history=(
                    factorized_native_target_history
                ),
                factorized_owner_source_block=(
                    factorized_owner_source_block
                ),
                target_semantic_competition=(
                    target_semantic_competition
                ),
                target_edit_phrases=target_edit_phrases,
                target_preserve_phrases=target_preserve_phrases,
                target_semantic_margin=target_semantic_margin,
                target_semantic_min_confidence=(
                    target_semantic_min_confidence
                ),
                causal_paired_edit_memory=causal_paired_edit_memory,
                paired_memory_layers=paired_edit_memory_layers,
                paired_memory_max_tokens=paired_memory_max_tokens,
                paired_memory_max_tokens_per_block=(
                    paired_memory_max_tokens_per_block
                ),
                paired_memory_topk=paired_memory_topk,
                paired_memory_min_similarity=(
                    paired_memory_min_similarity
                ),
                paired_memory_min_commit_confidence=(
                    paired_memory_min_commit_confidence
                ),
                paired_memory_coordinate_bias=(
                    paired_memory_coordinate_bias
                ),
                paired_memory_coordinate_radius=(
                    paired_memory_coordinate_radius
                ),
                paired_memory_min_residual_consensus=(
                    paired_memory_min_residual_consensus
                ),
                paired_memory_source_part_consistency=(
                    paired_memory_source_part_consistency
                ),
                paired_memory_min_part_similarity=(
                    paired_memory_min_part_similarity
                ),
                paired_memory_part_similarity_margin=(
                    paired_memory_part_similarity_margin
                ),
                paired_memory_read_strength=(
                    paired_memory_read_strength
                ),
                paired_memory_value_projection=(
                    paired_memory_value_projection
                ),
                paired_memory_query_gated_projection=(
                    paired_memory_query_gated_projection
                ),
                paired_memory_disable_persistent_projection=(
                    paired_memory_disable_persistent_projection
                ),
                paired_memory_source_suppression=(
                    paired_memory_source_suppression
                ),
                paired_memory_interior_projection=(
                    paired_memory_interior_projection
                ),
                paired_memory_source_transport=(
                    paired_memory_source_transport
                ),
                paired_memory_single_confidence=(
                    paired_memory_single_confidence
                ),
                paired_memory_owner_attached_boundary=(
                    paired_memory_owner_attached_boundary
                ),
                paired_memory_dual_timescale_anchor=(
                    paired_memory_dual_timescale_anchor
                ),
                paired_memory_canonical_key_anchor=(
                    paired_memory_canonical_key_anchor
                ),
                role_fixed_native_history=role_fixed_native_history,
                native_history_layers=role_native_history_layers,
                native_history_max_tokens_per_frame=(
                    native_history_max_tokens_per_frame
                ),
                native_history_topk=native_history_topk,
                native_history_min_similarity=(
                    native_history_min_similarity
                ),
                native_history_min_write_confidence=(
                    native_history_min_write_confidence
                ),
                native_history_min_query_confidence=(
                    native_history_min_query_confidence
                ),
                native_history_canonical_logit_bias=(
                    native_history_canonical_logit_bias
                ),
                native_history_coalesce_bootstrap_time=(
                    native_history_coalesce_bootstrap_time
                ),
                native_history_bypass_blocks=(
                    role_native_history_bypass_blocks
                ),
                native_history_source_part_consistency=(
                    native_history_source_part_consistency
                ),
                native_history_min_part_similarity=(
                    native_history_min_part_similarity
                ),
                native_history_part_similarity_margin=(
                    native_history_part_similarity_margin
                ),
                native_history_part_bias_strength=(
                    native_history_part_bias_strength
                ),
                native_history_part_refinement_ratio=(
                    native_history_part_refinement_ratio
                ),
                native_history_transactional_owner=(
                    native_history_transactional_owner
                ),
                native_history_consistent_transaction=(
                    native_history_consistent_transaction
                ),
                native_history_verified_attention_authority=(
                    native_history_verified_attention_authority
                ),
                native_history_attention_authority_strength=(
                    native_history_attention_authority_strength
                ),
                native_history_payload_invariant_lineage=(
                    native_history_payload_invariant_lineage
                ),
                native_history_payload_blend_strength=(
                    native_history_payload_blend_strength
                ),
                native_history_recent_entry_bridge=(
                    native_history_recent_entry_bridge
                ),
                native_history_motion_owner_dense_read=(
                    native_history_motion_owner_dense_read
                ),
                native_history_entry_bridge_strength=(
                    native_history_entry_bridge_strength
                ),
                native_history_dual_evidence_arbitration=(
                    native_history_dual_evidence_arbitration
                ),
                native_history_token_atomic_payload=(
                    native_history_token_atomic_payload
                ),
                native_history_persistent_residual_upsert=(
                    native_history_persistent_residual_upsert
                ),
                native_history_last_trusted_appearance=(
                    native_history_last_trusted_appearance
                ),
                native_history_flow_indexed_residual=(
                    native_history_flow_indexed_residual
                ),
                native_history_decoupled_flow_trust=(
                    native_history_decoupled_flow_trust
                ),
                native_history_multiframe_identity_sink=(
                    native_history_multiframe_identity_sink
                ),
                native_history_multiframe_sink_topk_per_frame=(
                    native_history_multiframe_sink_topk_per_frame
                ),
                native_history_multiframe_sink_source_logit_bias=(
                    native_history_multiframe_sink_source_logit_bias
                ),
                native_history_multiframe_sink_strength=(
                    native_history_multiframe_sink_strength
                ),
                native_history_timestep_counterfactual_memory=(
                    native_history_timestep_counterfactual_memory
                ),
                native_history_tccm_flow_radius=(
                    native_history_tccm_flow_radius
                ),
                native_history_tccm_strength=(
                    native_history_tccm_strength
                ),
                native_history_tccm_max_error_ratio=(
                    native_history_tccm_max_error_ratio
                ),
                native_history_flow_min_confidence=(
                    native_history_flow_min_confidence
                ),
                native_history_residual_update_min_cosine=(
                    native_history_residual_update_min_cosine
                ),
                native_history_residual_update_min_magnitude_ratio=(
                    native_history_residual_update_min_magnitude_ratio
                ),
                native_history_topology_complete_read=(
                    native_history_topology_complete_read
                ),
                native_history_min_payload_consistency=(
                    native_history_min_payload_consistency
                ),
                native_history_dense_recent_min_residual_consensus=(
                    native_history_dense_recent_min_residual_consensus
                ),
                native_history_owner_max_missing_frames=(
                    native_history_owner_max_missing_frames
                ),
                native_history_verified_source_suppression=(
                    native_history_verified_source_suppression
                ),
                paired_memory_transport_min_similarity=(
                    paired_memory_transport_min_similarity
                ),
                paired_memory_transport_coordinate_radius=(
                    paired_memory_transport_coordinate_radius
                ),
                paired_memory_transport_cycle_radius=(
                    paired_memory_transport_cycle_radius
                ),
                paired_memory_transport_min_confidence=(
                    paired_memory_transport_min_confidence
                ),
                immutable_target_layers=identity_memory_layers,
                immutable_target_num_prototypes=(
                    immutable_target_num_prototypes
                ),
                immutable_target_value_mode=immutable_target_value_mode,
                immutable_target_hard_owner=immutable_target_hard_owner,
                factorized_orthogonal_geometry=(
                    factorized_orthogonal_geometry
                ),
                factorized_geometry_strength=(
                    factorized_geometry_strength
                ),
                identity_correction_strength=(
                    identity_correction_strength
                ),
                identity_visibility_lifecycle=(
                    identity_visibility_lifecycle
                ),
                identity_max_occluded_blocks=(
                    identity_max_occluded_blocks
                ),
                appearance_leakage_decomposition=(
                    appearance_leakage_decomposition
                ),
                source_coordinate_identity=source_coordinate_identity,
                identity_source_suppression=(
                    identity_source_suppression
                ),
                identity_support_floor=identity_support_floor,
                source_identity_residual_carry=(
                    source_identity_residual_carry
                ),
                identity_residual_carry_strength=(
                    identity_residual_carry_strength
                ),
                source_owner_residual_constraint=(
                    source_owner_residual_constraint
                ),
                identity_residual_constraint_strength=(
                    identity_residual_constraint_strength
                ),
                identity_residual_constraint_power=(
                    identity_residual_constraint_power
                ),
                source_owner_geometry_envelope=(
                    source_owner_geometry_envelope
                ),
                source_geometry_strength=source_geometry_strength,
                source_geometry_power=source_geometry_power,
                source_geometry_margin=source_geometry_margin,
                ignition_hand_exclusion_radius=(
                    ignition_hand_exclusion_radius
                ),
                ignition_contact_radius=ignition_contact_radius,
                oracle_source_owner_mask=rollout_source_owner_mask,
                oracle_source_owner_full_mask=(
                    rollout_source_owner_full_mask
                ),
                source_owner_prepool_hand_exclusion=(
                    source_owner_prepool_hand_exclusion
                ),
                causal_owner_consistent_kv_metadata=(
                    causal_owner_consistent_kv_metadata
                ),
                factorized_source_coordinate_target_delta=(
                    factorized_source_coordinate_target_delta
                ),
                factorized_owner_complement_source=(
                    factorized_owner_complement_source
                ),
                factorized_owner_complement_margin=(
                    factorized_owner_complement_margin
                ),
                factorized_owner_complement_min_preserve_confidence=(
                    factorized_owner_complement_min_preserve_confidence
                ),
                oracle_object_mask=rollout_object_mask,
                oracle_hand_mask=rollout_hand_mask,
                hand_only_mask=rollout_hand_only_mask,
                hand_occupancy_mask=rollout_hand_occupancy_mask,
                hand_persistent_mask=rollout_hand_persistent_mask,
                hand_causal_evidence=hand_causal_evidence,
                motion_geometry_owner=motion_geometry_owner,
                source_flow_cache=source_flow_cache,
                source_flow_role_fusion=source_flow_role_fusion,
                source_flow_role_weight=source_flow_role_weight,
                source_flow_verified_region=source_flow_verified_region,
                source_flow_verified_owner_radius=(
                    source_flow_verified_owner_radius
                ),
                source_flow_background_veto_threshold=(
                    source_flow_background_veto_threshold
                ),
                source_flow_background_veto_min_confidence=(
                    source_flow_background_veto_min_confidence
                ),
                soft_region_modulation=soft_region_modulation,
                soft_region_blend_strength=soft_region_blend_strength,
                first_block_identity_anchor=first_block_identity_anchor,
                identity_anchor_scale=identity_anchor_scale,
                suppress_source_bg_value=suppress_source_bg_value,
                projected_source_residual=projected_source_residual,
                global_frame_indices=rollout_global_frame_indices,
                role_boundary_radius=role_boundary_radius,
                contact_target_weight=contact_target_weight,
                posterior_flow_mode=posterior_flow_mode,
                posterior_flow_use_field=posterior_flow_use_field,
                hand_posterior_threshold=hand_posterior_threshold,
                hand_max_object_coverage=hand_max_object_coverage,
                hand_proximity_radius=hand_proximity_radius,
                hand_propagation_steps=hand_propagation_steps,
                hand_connected_hysteresis=hand_connected_hysteresis,
                hand_connected_growth_steps=(
                    hand_connected_growth_steps
                ),
                hand_connected_candidate_ratio=(
                    hand_connected_candidate_ratio
                ),
                hand_visibility_ratio=hand_visibility_ratio,
                hand_temporal_weight=hand_temporal_weight,
                hand_query_similarity_threshold=(
                    hand_query_similarity_threshold
                ),
                hand_query_layers=hand_query_layers,
                hand_field_quantile_low=hand_field_quantile_low,
                hand_field_quantile_high=hand_field_quantile_high,
                hand_field_power=hand_field_power,
                hand_field_weight=hand_field_weight,
                hand_field_candidate_radius=hand_field_candidate_radius,
                hand_field_update_mode=hand_field_update_mode,
                identity_tokenprop_min_similarity=(
                    identity_tokenprop_min_similarity
                ),
                identity_tokenprop_gate_strength=(
                    identity_tokenprop_gate_strength
                ),
                identity_tokenprop_max_candidates=(
                    identity_tokenprop_max_candidates
                ),
                committed_memory_feedback_strength=(
                    committed_memory_feedback_strength
                ),
                contact_graph_mode=contact_graph_mode,
                contact_graph_topk=contact_graph_topk,
                contact_graph_radius=contact_graph_radius,
                contact_graph_min_confidence=contact_graph_min_confidence,
                contact_graph_strength=contact_graph_strength,
                contact_graph_layer_start=contact_graph_layer_start,
                contact_graph_layer_end=contact_graph_layer_end,
                contact_graph_seed=contact_graph_seed,
                save_role_dir=save_role_dir,
                _hand_role_inferencer=rollout_hand_role_inferencer,
                _memory_consolidator=rollout_memory_consolidator,
                _edit_commitment_controller=(
                    rollout_commitment_controller
                ),
                _target_identity_memory=(
                    rollout_target_identity_memory
                ),
                _identity_token_propagator=(
                    rollout_identity_token_propagator
                ),
                _identity_support_filter=(
                    rollout_identity_support_filter
                ),
                _identity_owner_tracker=(
                    rollout_identity_owner_tracker
                ),
                _causal_ownership_tracker=(
                    rollout_causal_ownership_tracker
                ),
                _native_owner_tracker=rollout_native_owner_tracker,
                _identity_residual_carry=(
                    rollout_identity_residual_carry
                ),
                _causal_paired_edit_memory=(
                    rollout_paired_edit_memory
                ),
                _role_native_kv_history=(
                    rollout_role_native_kv_history
                ),
            )

            # store results
            if start_idx == 0:
                ret_latent_list.append(rollout_latent)
            else:
                ret_latent_list.append(rollout_latent[:, rollout_overlap: ])

            # finish, end loop
            if chunk_right_idx >= total_frame_num:
                break

            # index update
            start_idx = chunk_right_idx - rollout_overlap

            # prepare prev_cond
            src_initial_latent = rollout_src_video[:, -rollout_overlap: ]
            trg_initial_latent = rollout_latent[:, -rollout_overlap: ]

        output = torch.cat(ret_latent_list, dim=1)
        assert src_video.shape == output.shape, 'noise shape: %s, but output: %s.' % (str(src_video.shape), str(output.shape))

        # clean cache before decode to avoid OOM
        torch.cuda.empty_cache()

        if wo_video_decode:
            video = None
        else:
            dec_latent = output
            video = self.vae.decode_to_pixel(dec_latent, use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)
        if profile:
            torch.cuda.synchronize()

        if return_latents:
            return video, output
        else:
            return video

    def inference(
        self,
        src_video: torch.Tensor,
        src_prompts: str | List[str],
        trg_prompts: str | List[str],
        src_trigger_words: str | List[str],
        trg_trigger_words: str | List[str],
        return_latents: bool = False,
        wo_video_decode: bool = False,
        profile: bool = False,
        low_memory: bool = False,

        independent_first_frame: bool = False,
        triple_first_frame: bool = False,
        src_initial_latent: Optional[torch.Tensor] = None,
        trg_initial_latent: Optional[torch.Tensor] = None,

        fg_boost_factor=2.0,
        blend_power=2.0,

        mask_layers: Iterable = range(20),
        enhance_layers: Iterable = range(30),

        fg_scale=1.0,
        reuse_noise_temporal_mean=True,
        routing_mode: str = "dynamic_sog",
        identity_first_latent_bootstrap: bool = False,
        object_wise_anchor_reset: bool = False,
        target_owned_object_handoff: bool = False,
        target_owned_min_similarity: float = 0.55,
        factorized_target_identity: bool = False,
        factorized_immutable_target_memory: bool = False,
        factorized_native_target_history: bool = False,
        factorized_owner_source_block: bool = False,
        target_semantic_competition: bool = False,
        target_edit_phrases: Optional[List[str]] = None,
        target_preserve_phrases: Optional[List[str]] = None,
        target_semantic_margin: float = 0.10,
        target_semantic_min_confidence: float = 0.20,
        causal_paired_edit_memory: bool = False,
        paired_memory_layers: Optional[Iterable] = None,
        paired_memory_max_tokens: int = 1536,
        paired_memory_max_tokens_per_block: int = 192,
        paired_memory_topk: int = 8,
        paired_memory_min_similarity: float = 0.35,
        paired_memory_min_commit_confidence: float = 0.20,
        paired_memory_coordinate_bias: float = 1.0,
        paired_memory_coordinate_radius: float = 0.0,
        paired_memory_min_residual_consensus: float = 0.0,
        paired_memory_source_part_consistency: bool = False,
        paired_memory_min_part_similarity: float = 0.45,
        paired_memory_part_similarity_margin: float = 0.08,
        paired_memory_read_strength: float = 0.35,
        paired_memory_value_projection: bool = False,
        paired_memory_query_gated_projection: bool = False,
        paired_memory_disable_persistent_projection: bool = False,
        paired_memory_source_suppression: float = 0.0,
        paired_memory_interior_projection: bool = False,
        paired_memory_source_transport: bool = False,
        paired_memory_single_confidence: bool = False,
        paired_memory_owner_attached_boundary: bool = False,
        paired_memory_dual_timescale_anchor: bool = False,
        paired_memory_canonical_key_anchor: bool = False,
        role_fixed_native_history: bool = False,
        native_history_layers: Optional[Iterable] = None,
        native_history_max_tokens_per_frame: int = 256,
        native_history_topk: int = 8,
        native_history_min_similarity: float = 0.35,
        native_history_min_write_confidence: float = 0.50,
        native_history_min_query_confidence: float = 0.50,
        native_history_canonical_logit_bias: float = 1.0,
        native_history_coalesce_bootstrap_time: bool = False,
        native_history_bypass_blocks: Optional[Iterable[int]] = None,
        native_history_source_part_consistency: bool = False,
        native_history_min_part_similarity: float = 0.45,
        native_history_part_similarity_margin: float = 0.08,
        native_history_part_bias_strength: float = 0.5,
        native_history_part_refinement_ratio: float = 0.25,
        native_history_transactional_owner: bool = False,
        native_history_consistent_transaction: bool = False,
        native_history_verified_attention_authority: bool = False,
        native_history_attention_authority_strength: float = 1.0,
        native_history_payload_invariant_lineage: bool = False,
        native_history_payload_blend_strength: float = 0.35,
        native_history_recent_entry_bridge: bool = False,
        native_history_motion_owner_dense_read: bool = False,
        native_history_entry_bridge_strength: float = 1.0,
        native_history_dual_evidence_arbitration: bool = False,
        native_history_token_atomic_payload: bool = False,
        native_history_persistent_residual_upsert: bool = False,
        native_history_last_trusted_appearance: bool = False,
        native_history_flow_indexed_residual: bool = False,
        native_history_decoupled_flow_trust: bool = False,
        native_history_multiframe_identity_sink: bool = False,
        native_history_multiframe_sink_topk_per_frame: int = 8,
        native_history_multiframe_sink_source_logit_bias: float = 1.0,
        native_history_multiframe_sink_strength: float = 1.0,
        native_history_timestep_counterfactual_memory: bool = False,
        native_history_tccm_flow_radius: float = 2.0,
        native_history_tccm_strength: float = 1.0,
        native_history_tccm_max_error_ratio: float = 1.0,
        native_history_flow_min_confidence: float = 0.10,
        native_history_residual_update_min_cosine: float = 0.50,
        native_history_residual_update_min_magnitude_ratio: float = 0.90,
        native_history_topology_complete_read: bool = False,
        native_history_min_payload_consistency: float = 0.15,
        native_history_dense_recent_min_residual_consensus: float = 0.05,
        native_history_owner_max_missing_frames: int = 1,
        native_history_verified_source_suppression: float = 0.35,
        paired_memory_transport_min_similarity: float = 0.10,
        paired_memory_transport_coordinate_radius: float = 0.60,
        paired_memory_transport_cycle_radius: float = 0.20,
        paired_memory_transport_min_confidence: float = 0.05,
        immutable_target_layers: Optional[Iterable] = None,
        immutable_target_num_prototypes: int = 4,
        immutable_target_value_mode: str = "residual",
        immutable_target_hard_owner: bool = False,
        factorized_orthogonal_geometry: bool = False,
        factorized_geometry_strength: float = 1.0,
        identity_correction_strength: float = 0.35,
        identity_visibility_lifecycle: bool = False,
        identity_max_occluded_blocks: int = 1,
        appearance_leakage_decomposition: bool = False,
        source_coordinate_identity: bool = False,
        identity_source_suppression: float = 0.35,
        identity_support_floor: float = 0.0,
        source_identity_residual_carry: bool = False,
        identity_residual_carry_strength: float = 0.25,
        source_owner_residual_constraint: bool = False,
        identity_residual_constraint_strength: float = 0.35,
        identity_residual_constraint_power: float = 2.0,
        source_owner_geometry_envelope: bool = False,
        source_geometry_strength: float = 0.35,
        source_geometry_power: float = 2.0,
        source_geometry_margin: int = 1,
        ignition_hand_exclusion_radius: int = 1,
        ignition_contact_radius: int = 3,
        oracle_source_owner_mask: Optional[torch.Tensor] = None,
        oracle_source_owner_full_mask: Optional[torch.Tensor] = None,
        source_owner_prepool_hand_exclusion: bool = False,
        causal_owner_consistent_kv_metadata: bool = False,
        factorized_source_coordinate_target_delta: bool = False,
        factorized_owner_complement_source: bool = False,
        factorized_owner_complement_margin: int = 1,
        factorized_owner_complement_min_preserve_confidence: float = 0.0,
        oracle_object_mask: Optional[torch.Tensor] = None,
        oracle_hand_mask: Optional[torch.Tensor] = None,
        hand_only_mask: Optional[torch.Tensor] = None,
        hand_occupancy_mask: Optional[torch.Tensor] = None,
        hand_persistent_mask: Optional[torch.Tensor] = None,
        hand_causal_evidence: bool = False,
        motion_geometry_owner: bool = False,
        source_flow_cache: Optional[SourceFlowCache] = None,
        source_flow_role_fusion: bool = False,
        source_flow_role_weight: float = 0.75,
        source_flow_verified_region: bool = False,
        source_flow_verified_owner_radius: int = 1,
        source_flow_background_veto_threshold: float = 0.55,
        source_flow_background_veto_min_confidence: float = 0.50,
        soft_region_modulation: bool = False,
        soft_region_blend_strength: float = 0.5,
        first_block_identity_anchor: bool = False,
        identity_anchor_scale: float = 1.5,
        suppress_source_bg_value: bool = False,
        projected_source_residual: bool = False,
        global_frame_indices: Optional[List[int]] = None,
        role_boundary_radius: int = 1,
        contact_target_weight: float = 0.7,
        posterior_flow_mode: str = "soft",
        posterior_flow_use_field: bool = False,
        hand_posterior_threshold: float = 0.20,
        hand_max_object_coverage: float = 0.18,
        hand_proximity_radius: int = 3,
        hand_propagation_steps: int = 2,
        hand_connected_hysteresis: bool = False,
        hand_connected_growth_steps: int = 3,
        hand_connected_candidate_ratio: float = 1.0,
        hand_visibility_ratio: float = 0.40,
        hand_temporal_weight: float = 0.45,
        hand_query_similarity_threshold: float = 0.65,
        hand_query_layers: Iterable = (8, 12, 16, 20),
        hand_field_quantile_low: float = 0.50,
        hand_field_quantile_high: float = 0.95,
        hand_field_power: float = 1.5,
        hand_field_weight: float = 0.65,
        hand_field_candidate_radius: int = 2,
        hand_field_update_mode: str = "diagnostic",
        identity_tokenprop_min_similarity: float = 0.55,
        identity_tokenprop_gate_strength: float = 0.85,
        identity_tokenprop_max_candidates: int = 512,
        committed_memory_feedback_strength: float = 0.75,
        contact_graph_mode: str = "no_graph",
        contact_graph_topk: int = 4,
        contact_graph_radius: float = 2.5,
        contact_graph_min_confidence: float = 0.05,
        contact_graph_strength: float = 0.25,
        contact_graph_layer_start: int = 10,
        contact_graph_layer_end: int = 20,
        contact_graph_seed: int = 0,
        save_role_dir: Optional[str] = None,
        _hand_role_inferencer: Optional[HandRoleInferencer] = None,
        _memory_consolidator: Optional[
            CausalMemoryConsolidator
        ] = None,
        _edit_commitment_controller: Optional[
            EditCommitmentController
        ] = None,
        _target_identity_memory: Optional[
            SlowTargetIdentityMemory
        ] = None,
        _identity_token_propagator: Optional[
            CausalObjectTokenPropagator
        ] = None,
        _identity_support_filter: Optional[
            CausalConnectedSupportFilter
        ] = None,
        _identity_owner_tracker: Optional[
            CausalIdentityOwnerTracker
        ] = None,
        _causal_ownership_tracker: Optional[
            CausalObjectOwnershipTracker | MotionAwareGeometryOwnerTracker
        ] = None,
        _native_owner_tracker: Optional[
            CausalReadOnlyOwnerTracker
            | AutomaticTransactionalOwnerTracker
        ] = None,
        _identity_residual_carry: Optional[
            SourceCoordinateResidualCarry
        ] = None,
        _causal_paired_edit_memory: Optional[
            CausalPairedEditMemory
        ] = None,
        _role_native_kv_history: Optional[
            RoleConditionedNativeKVHistory
        ] = None,
    ) -> torch.Tensor:
        assert not (independent_first_frame and triple_first_frame)
        independent_first_frame = independent_first_frame or self.independent_first_frame

        batch_size, num_frames, num_channels, height, width = src_video.shape
        if motion_geometry_owner:
            if source_flow_cache is None:
                raise ValueError(
                    "motion_geometry_owner requires source_flow_cache"
                )
            if global_frame_indices is None or (
                len(global_frame_indices) != num_frames
            ):
                raise ValueError(
                    "motion_geometry_owner requires one global index per "
                    "inference source frame"
                )
        oracle_role_enabled = routing_mode in {
            "oracle_role_flow",
            "oracle_role_flow_kv",
            "oracle_role_residual",
            "oracle_role_residual_kv",
        }
        oracle_kv_enabled = routing_mode in {
            "oracle_role_flow_kv",
            "oracle_role_residual_kv",
        }
        hand_role_enabled = routing_mode in {
            "hand_role_residual_kv",
            "hand_role_adaptive_kv",
            "hand_role_posterior_flow_kv",
            "hand_role_bayes_flow_kv",
            "hand_role_bayes_flow_dual_kv",
            "hand_role_bayes_flow_consolidated_kv",
            "hand_role_bayes_flow_commitment_kv",
            "hand_role_bayes_flow_identity_kv",
            "hand_role_bayes_flow_tokenprop_kv",
            "hand_role_bayes_flow_customized_kv",
            "hand_role_factorized_bayes_kv",
            "hand_role_factorized_causal_owner_kv",
        }
        if hand_only_mask is not None:
            if hand_occupancy_mask is None:
                hand_occupancy_mask = hand_only_mask.float()
            if hand_persistent_mask is None:
                hand_persistent_mask = hand_only_mask.bool()
        adaptive_role_enabled = routing_mode in {
            "hand_role_adaptive_kv",
            "hand_role_posterior_flow_kv",
            "hand_role_bayes_flow_kv",
            "hand_role_bayes_flow_dual_kv",
            "hand_role_bayes_flow_consolidated_kv",
            "hand_role_bayes_flow_commitment_kv",
            "hand_role_bayes_flow_identity_kv",
            "hand_role_bayes_flow_tokenprop_kv",
            "hand_role_bayes_flow_customized_kv",
            "hand_role_factorized_bayes_kv",
            "hand_role_factorized_causal_owner_kv",
        }
        posterior_flow_enabled = (
            routing_mode == "hand_role_posterior_flow_kv"
        )
        bayes_flow_enabled = (
            routing_mode in {
                "hand_role_bayes_flow_kv",
                "hand_role_bayes_flow_dual_kv",
                "hand_role_bayes_flow_consolidated_kv",
                "hand_role_bayes_flow_commitment_kv",
                "hand_role_bayes_flow_identity_kv",
                "hand_role_bayes_flow_tokenprop_kv",
                "hand_role_bayes_flow_customized_kv",
            }
        )
        factorized_bayes_enabled = routing_mode in {
            "hand_role_factorized_bayes_kv",
            "hand_role_factorized_causal_owner_kv",
        }
        causal_ownership_enabled = (
            routing_mode == "hand_role_factorized_causal_owner_kv"
        )
        if (
            causal_owner_consistent_kv_metadata
            and not causal_ownership_enabled
        ):
            raise ValueError(
                "Causal-owner consistent KV metadata requires "
                "factorized causal-owner routing"
            )
        if (
            factorized_source_coordinate_target_delta
            and not causal_ownership_enabled
        ):
            raise ValueError(
                "Source-coordinate target-delta routing requires "
                "factorized causal-owner routing"
            )
        if (
            factorized_owner_complement_source
            and not causal_ownership_enabled
        ):
            raise ValueError(
                "Owner-complement source routing requires factorized "
                "causal-owner routing"
            )
        if (
            factorized_owner_complement_source
            and oracle_source_owner_mask is None
            and not motion_geometry_owner
        ):
            raise ValueError(
                "Owner-complement source routing requires an explicit "
                "clean-source owner mask or the automatic motion owner"
            )
        if factorized_owner_complement_margin < 0:
            raise ValueError(
                "Owner-complement margin must be non-negative"
            )
        if not (
            0.0
            <= factorized_owner_complement_min_preserve_confidence
            <= 1.0
        ):
            raise ValueError(
                "Owner-complement minimum preserve confidence must lie "
                "in [0, 1]"
            )
        if (
            factorized_source_coordinate_target_delta
            and factorized_orthogonal_geometry
        ):
            raise ValueError(
                "Source-coordinate target-delta routing cannot be "
                "combined with orthogonal geometry"
            )
        if factorized_native_target_history and not causal_ownership_enabled:
            raise ValueError(
                "Native target-history ablation requires factorized "
                "causal-owner routing"
            )
        if role_fixed_native_history and not (
            factorized_native_target_history and causal_ownership_enabled
        ):
            raise ValueError(
                "Role-fixed native history requires factorized native "
                "target history with causal ownership"
            )
        if native_history_transactional_owner:
            if not role_fixed_native_history:
                raise ValueError(
                    "Transactional native owner requires role-fixed "
                    "native history"
                )
        if (
            native_history_consistent_transaction
            and not native_history_transactional_owner
        ):
            raise ValueError(
                "Consistent native transaction requires transactional "
                "owner reads"
            )
        if (
            native_history_verified_attention_authority
            and not native_history_consistent_transaction
        ):
            raise ValueError(
                "Verified attention authority requires a consistent native "
                "transaction"
            )
        if not (
            0.0 <= native_history_attention_authority_strength <= 1.0
        ):
            raise ValueError(
                "Native-history attention authority strength must lie in "
                "[0, 1]"
            )
        if (
            native_history_payload_invariant_lineage
            and not native_history_transactional_owner
        ):
            raise ValueError(
                "Payload-invariant native lineage requires "
                "transactional owner reads"
            )
        if not 0.0 <= native_history_payload_blend_strength <= 1.0:
            raise ValueError(
                "Native-history payload blend strength must lie in [0, 1]"
            )
        if native_history_recent_entry_bridge:
            if not native_history_consistent_transaction:
                raise ValueError(
                    "Recent-entry native history requires a consistent "
                    "transaction"
                )
            if native_history_payload_invariant_lineage:
                raise ValueError(
                    "Recent-entry native history cannot use payload-invariant "
                    "lineage"
                )
            validate_recent_entry_hand_only_contract(
                enabled=True,
                routing_mode=routing_mode,
                hand_only_mask=hand_only_mask,
                oracle_object_mask=oracle_object_mask,
                oracle_source_owner_mask=oracle_source_owner_mask,
                oracle_source_owner_full_mask=(
                    oracle_source_owner_full_mask
                ),
            )
        if native_history_motion_owner_dense_read and not (
            native_history_recent_entry_bridge
            and native_history_consistent_transaction
            and motion_geometry_owner
        ):
            raise ValueError(
                "Motion-owner dense native read requires the recent-entry "
                "consistent transaction and motion geometry owner"
            )
        if not 0.0 <= native_history_entry_bridge_strength <= 1.0:
            raise ValueError(
                "Native-history entry bridge strength must lie in [0, 1]"
            )
        if native_history_dual_evidence_arbitration and not (
            native_history_recent_entry_bridge
            and native_history_consistent_transaction
        ):
            raise ValueError(
                "Dual-evidence native history requires the consistent "
                "recent-entry bridge"
            )
        if native_history_token_atomic_payload and not (
            native_history_recent_entry_bridge
            and native_history_consistent_transaction
            and native_history_motion_owner_dense_read
        ):
            raise ValueError(
                "Token-atomic native payloads require the consistent "
                "dense motion-owner recent bridge"
            )
        if native_history_topology_complete_read and not (
            native_history_token_atomic_payload and motion_geometry_owner
        ):
            raise ValueError(
                "Topology-complete native reads require token-atomic "
                "payloads and motion geometry ownership"
            )
        if native_history_persistent_residual_upsert and not (
            native_history_token_atomic_payload and motion_geometry_owner
        ):
            raise ValueError(
                "Persistent residual native history requires token-atomic "
                "payloads and motion geometry ownership"
            )
        if native_history_last_trusted_appearance and not (
            native_history_persistent_residual_upsert
            and native_history_transactional_owner
        ):
            raise ValueError(
                "Last-trusted appearance requires persistent residual "
                "upserts and transactional owner arbitration"
            )
        if native_history_multiframe_identity_sink and not (
            native_history_decoupled_flow_trust
            and native_history_flow_indexed_residual
        ):
            raise ValueError(
                "Multi-frame identity sink requires decoupled flow-indexed "
                "trust"
            )
        if native_history_timestep_counterfactual_memory and not (
            native_history_multiframe_identity_sink
        ):
            raise ValueError(
                "Timestep counterfactual memory requires the multi-frame "
                "identity sink contract"
            )
        if native_history_timestep_counterfactual_memory and (
            oracle_object_mask is not None
            or oracle_source_owner_mask is not None
            or oracle_source_owner_full_mask is not None
        ):
            raise ValueError(
                "Timestep counterfactual memory forbids object/source-owner "
                "masks"
            )
        if native_history_tccm_flow_radius < 0.0:
            raise ValueError("TCCM flow radius must be non-negative")
        if not 0.0 <= native_history_tccm_strength <= 1.0:
            raise ValueError("TCCM strength must lie in [0, 1]")
        if native_history_tccm_max_error_ratio <= 0.0:
            raise ValueError(
                "TCCM maximum error ratio must be positive"
            )
        if native_history_multiframe_identity_sink and (
            oracle_object_mask is not None
            or oracle_source_owner_mask is not None
            or oracle_source_owner_full_mask is not None
        ):
            raise ValueError(
                "Multi-frame identity sink forbids object/source-owner "
                "masks"
            )
        if native_history_multiframe_sink_topk_per_frame <= 0:
            raise ValueError(
                "Multi-frame sink top-k per frame must be positive"
            )
        if not math.isfinite(
            native_history_multiframe_sink_source_logit_bias
        ):
            raise ValueError(
                "Multi-frame sink source logit bias must be finite"
            )
        if not 0.0 <= native_history_multiframe_sink_strength <= 1.0:
            raise ValueError(
                "Multi-frame sink strength must lie in [0, 1]"
            )
        if not (
            -1.0 <= native_history_residual_update_min_cosine <= 1.0
        ):
            raise ValueError(
                "Native-history residual update cosine must lie in [-1, 1]"
            )
        if not (
            0.0
            <= native_history_residual_update_min_magnitude_ratio
            <= 1.0
        ):
            raise ValueError(
                "Native-history residual magnitude ratio must lie in [0, 1]"
            )
        if not 0.0 <= native_history_min_payload_consistency <= 1.0:
            raise ValueError(
                "Native-history payload consistency must lie in [0, 1]"
            )
        if not (
            0.0
            <= native_history_dense_recent_min_residual_consensus
            <= 1.0
        ):
            raise ValueError(
                "Dense recent residual consensus must lie in [0, 1]"
            )
        if native_history_owner_max_missing_frames < 0:
            raise ValueError(
                "Native-history owner missing-frame limit must be "
                "non-negative"
            )
        if not (
            0.0 <= native_history_verified_source_suppression <= 1.0
        ):
            raise ValueError(
                "Native-history verified source suppression must lie "
                "in [0, 1]"
            )
        if (
            factorized_owner_source_block
            and not factorized_native_target_history
        ):
            raise ValueError(
                "Owner source blocking requires native target-history "
                "attention"
            )
        if (
            factorized_native_target_history
            and factorized_immutable_target_memory
        ):
            raise ValueError(
                "The 923 native target-history ablation must keep the "
                "clean target KV cache immutable-memory free"
            )
        paired_edit_memory_layers = tuple(
            hand_query_layers
            if paired_memory_layers is None
            else paired_memory_layers
        )
        role_native_history_layers = tuple(
            hand_query_layers
            if native_history_layers is None
            else native_history_layers
        )
        role_native_history_bypass_blocks = tuple(
            ()
            if native_history_bypass_blocks is None
            else native_history_bypass_blocks
        )
        if role_fixed_native_history:
            if not role_native_history_layers or len(set(
                role_native_history_layers
            )) != len(role_native_history_layers):
                raise ValueError(
                    "Native history layers must be unique and nonempty"
                )
            if any(
                layer < 0 or layer >= self.num_transformer_blocks
                for layer in role_native_history_layers
            ):
                raise ValueError(
                    "Native history layers must be valid transformer layers"
                )
            if native_history_max_tokens_per_frame <= 0:
                raise ValueError("Native history budget must be positive")
            if native_history_topk <= 0:
                raise ValueError("Native history topk must be positive")
            if not -1.0 < native_history_min_similarity < 1.0:
                raise ValueError(
                    "Native history similarity must lie in (-1, 1)"
                )
            if not (
                0.0 <= native_history_min_write_confidence <= 1.0
                and 0.0 <= native_history_min_query_confidence <= 1.0
            ):
                raise ValueError(
                    "Native history confidence thresholds must lie in [0, 1]"
                )
            if not math.isfinite(native_history_canonical_logit_bias):
                raise ValueError(
                    "Native history canonical logit bias must be finite"
                )
            if not -1.0 < native_history_min_part_similarity < 1.0:
                raise ValueError(
                    "Native history part similarity must lie in (-1, 1)"
                )
            if not (
                0.0 <= native_history_part_similarity_margin <= 2.0
            ):
                raise ValueError(
                    "Native history part similarity margin must lie in "
                    "[0, 2]"
                )
            if not 0.0 <= native_history_part_bias_strength <= 4.0:
                raise ValueError(
                    "Native history part bias strength must lie in [0, 4]"
                )
            if not 0.0 <= native_history_part_refinement_ratio <= 1.0:
                raise ValueError(
                    "Native history part refinement ratio must lie in "
                    "[0, 1]"
                )
            if (
                len(set(role_native_history_bypass_blocks))
                != len(role_native_history_bypass_blocks)
                or any(
                    block < 0
                    for block in role_native_history_bypass_blocks
                )
            ):
                raise ValueError(
                    "Native history bypass blocks must be unique and "
                    "non-negative"
                )
        if (
            paired_memory_value_projection
            or paired_memory_query_gated_projection
            or paired_memory_disable_persistent_projection
            or paired_memory_source_suppression != 0.0
            or paired_memory_interior_projection
            or paired_memory_coordinate_radius != 0.0
            or paired_memory_min_residual_consensus != 0.0
            or paired_memory_source_part_consistency
            or paired_memory_min_part_similarity != 0.45
            or paired_memory_part_similarity_margin != 0.08
            or paired_memory_source_transport
            or paired_memory_single_confidence
            or paired_memory_owner_attached_boundary
            or paired_memory_dual_timescale_anchor
            or paired_memory_canonical_key_anchor
            or paired_memory_transport_min_similarity != 0.10
            or paired_memory_transport_coordinate_radius != 0.60
            or paired_memory_transport_cycle_radius != 0.20
            or paired_memory_transport_min_confidence != 0.05
        ) and not causal_paired_edit_memory:
            raise ValueError(
                "Paired-memory projection and source suppression require "
                "causal paired edit memory"
            )
        if (
            paired_memory_query_gated_projection
            and not paired_memory_value_projection
        ):
            raise ValueError(
                "Query-gated projection requires paired value projection"
            )
        if (
            paired_memory_disable_persistent_projection
            and not paired_memory_value_projection
        ):
            raise ValueError(
                "Disabling persistent projection requires paired value "
                "projection"
            )
        if (
            paired_memory_query_gated_projection
            and not paired_memory_disable_persistent_projection
        ):
            raise ValueError(
                "Query-gated projection requires persistent cache "
                "projection to be disabled until historical owner "
                "metadata is available"
            )
        if (
            paired_memory_source_part_consistency
            and not paired_memory_query_gated_projection
        ):
            raise ValueError(
                "Source-part consistency requires query-gated projection"
            )
        if paired_memory_source_transport and not causal_paired_edit_memory:
            raise ValueError(
                "Source-transported memory requires causal paired edit "
                "memory"
            )
        if (
            paired_memory_owner_attached_boundary
            and not paired_memory_interior_projection
        ):
            raise ValueError(
                "Owner-attached boundary memory requires the paired "
                "projection gate"
            )
        if (
            paired_memory_single_confidence
            and not paired_memory_query_gated_projection
        ):
            raise ValueError(
                "Single-confidence memory requires query-gated "
                "projection"
            )
        if paired_memory_owner_attached_boundary and (
            not paired_memory_source_part_consistency
            or not paired_memory_source_transport
        ):
            raise ValueError(
                "Owner-attached boundary memory requires source-part "
                "consistency and source transport"
            )
        if paired_memory_dual_timescale_anchor and not (
            paired_memory_value_projection
            and paired_memory_query_gated_projection
            and paired_memory_disable_persistent_projection
            and paired_memory_single_confidence
            and paired_memory_source_transport
        ):
            raise ValueError(
                "Dual-timescale anchor requires transient, query-gated, "
                "single-confidence source-transported paired memory"
            )
        if paired_memory_canonical_key_anchor and not (
            paired_memory_dual_timescale_anchor
            and paired_memory_source_transport
        ):
            raise ValueError(
                "Immutable canonical-key anchor requires dual-timescale "
                "source-transported paired memory"
            )
        if causal_paired_edit_memory:
            if not causal_ownership_enabled:
                raise ValueError(
                    "Paired edit memory requires factorized "
                    "causal-owner routing"
                )
            if not factorized_native_target_history:
                raise ValueError(
                    "Paired edit memory requires native target-history "
                    "fallback"
                )
            if not paired_edit_memory_layers or any(
                layer < 0 or layer >= self.num_transformer_blocks
                for layer in paired_edit_memory_layers
            ):
                raise ValueError(
                    "Paired edit-memory layers must be valid transformer "
                    "indices"
                )
            if (
                len(set(paired_edit_memory_layers))
                != len(paired_edit_memory_layers)
            ):
                raise ValueError(
                    "Paired edit-memory layers must be unique"
                )
            if not 0.0 <= paired_memory_read_strength <= 1.0:
                raise ValueError(
                    "paired_memory_read_strength must lie in [0, 1]"
                )
            if paired_memory_coordinate_radius < 0.0:
                raise ValueError(
                    "paired_memory_coordinate_radius must be non-negative"
                )
            if not (
                0.0 <= paired_memory_min_residual_consensus < 1.0
            ):
                raise ValueError(
                    "paired_memory_min_residual_consensus must lie in "
                    "[0, 1)"
                )
            if not 0.0 <= paired_memory_source_suppression <= 1.0:
                raise ValueError(
                    "paired_memory_source_suppression must lie in [0, 1]"
                )
        aligned_belief_kv_enabled = (
            routing_mode == "hand_role_bayes_flow_dual_kv"
        )
        memory_consolidation_enabled = (
            routing_mode in {
                "hand_role_bayes_flow_consolidated_kv",
                "hand_role_bayes_flow_commitment_kv",
                "hand_role_bayes_flow_identity_kv",
                "hand_role_bayes_flow_tokenprop_kv",
                "hand_role_bayes_flow_customized_kv",
            }
        )
        edit_commitment_enabled = (
            routing_mode in {
                "hand_role_bayes_flow_commitment_kv",
                "hand_role_bayes_flow_identity_kv",
                "hand_role_bayes_flow_tokenprop_kv",
                "hand_role_bayes_flow_customized_kv",
            }
        )
        target_identity_enabled = (
            routing_mode in {
                "hand_role_bayes_flow_identity_kv",
                "hand_role_bayes_flow_tokenprop_kv",
                "hand_role_bayes_flow_customized_kv",
            }
            or factorized_immutable_target_memory
        )
        target_identity_tokenprop_enabled = (
            routing_mode == "hand_role_bayes_flow_tokenprop_kv"
        )
        reference_identity_enabled = (
            routing_mode == "hand_role_bayes_flow_customized_kv"
        )
        if (
            object_wise_anchor_reset
            and not identity_first_latent_bootstrap
        ):
            raise ValueError(
                "Object-wise anchor reset requires "
                "identity_first_latent_bootstrap"
            )
        if object_wise_anchor_reset and not target_identity_enabled:
            raise ValueError(
                "Object-wise anchor reset requires a target identity "
                "routing mode"
            )
        if object_wise_anchor_reset and reference_identity_enabled:
            raise ValueError(
                "Object-wise causal reset must not replace a reference "
                "identity anchor"
            )
        if (
            target_owned_object_handoff
            and not object_wise_anchor_reset
        ):
            raise ValueError(
                "Target-owned object handoff requires "
                "object_wise_anchor_reset"
            )
        if not -1.0 < target_owned_min_similarity < 1.0:
            raise ValueError(
                "target_owned_min_similarity must lie in (-1, 1)"
            )
        if not 0.0 <= identity_correction_strength <= 1.0:
            raise ValueError(
                "identity_correction_strength must lie in [0, 1]"
            )
        if identity_max_occluded_blocks < 0:
            raise ValueError(
                "identity_max_occluded_blocks must be non-negative"
            )
        if factorized_target_identity and not target_identity_enabled:
            raise ValueError(
                "Factorized target identity requires an identity "
                "routing mode"
            )
        if (
            identity_visibility_lifecycle
            and not factorized_target_identity
        ):
            raise ValueError(
                "Identity visibility lifecycle requires factorized "
                "target identity"
            )
        if (
            appearance_leakage_decomposition
            and not bayes_flow_enabled
            and not factorized_immutable_target_memory
        ):
            raise ValueError(
                "Appearance-leakage decomposition requires a Bayes "
                "residual routing mode"
            )
        if appearance_leakage_decomposition and factorized_target_identity:
            raise ValueError(
                "Appearance-leakage decomposition and factorized target "
                "identity are separate ablations and cannot be combined"
            )
        if source_coordinate_identity and not target_identity_enabled:
            raise ValueError(
                "Source-coordinate identity requires an identity "
                "routing mode"
            )
        if (
            source_coordinate_identity
            and not appearance_leakage_decomposition
        ):
            raise ValueError(
                "Source-coordinate identity requires appearance-leakage "
                "decomposition"
            )
        if (
            oracle_source_owner_mask is not None
            and not source_coordinate_identity
            and not causal_ownership_enabled
        ):
            raise ValueError(
                "Oracle source owner mask requires source-coordinate "
                "identity or factorized causal ownership"
            )
        if (
            factorized_immutable_target_memory
            and not causal_ownership_enabled
        ):
            raise ValueError(
                "Immutable factorized target memory requires "
                "causal-owner routing"
            )
        if not 0.0 <= identity_source_suppression <= 1.0:
            raise ValueError(
                "identity_source_suppression must lie in [0, 1]"
            )
        if not 0.0 <= identity_support_floor <= 1.0:
            raise ValueError(
                "identity_support_floor must lie in [0, 1]"
            )
        if (
            source_identity_residual_carry
            and not source_coordinate_identity
        ):
            raise ValueError(
                "Source identity residual carry requires "
                "source-coordinate identity"
            )
        if not 0.0 <= identity_residual_carry_strength <= 1.0:
            raise ValueError(
                "identity_residual_carry_strength must lie in [0, 1]"
            )
        if (
            source_owner_residual_constraint
            and not source_identity_residual_carry
        ):
            raise ValueError(
                "Source-owner residual constraint requires source "
                "identity residual carry"
            )
        if (
            source_owner_residual_constraint
            and oracle_source_owner_mask is None
        ):
            raise ValueError(
                "Source-owner residual constraint currently requires an "
                "oracle source owner mask for safe spatial isolation"
            )
        if not 0.0 <= identity_residual_constraint_strength <= 1.0:
            raise ValueError(
                "identity_residual_constraint_strength must lie in [0, 1]"
            )
        if identity_residual_constraint_power <= 0:
            raise ValueError(
                "identity_residual_constraint_power must be positive"
            )
        if (
            source_owner_geometry_envelope
            and oracle_source_owner_mask is None
        ):
            raise ValueError(
                "Source-owner geometry envelope requires an explicit "
                "clean-source owner mask"
            )
        if not 0.0 <= source_geometry_strength <= 1.0:
            raise ValueError(
                "source_geometry_strength must lie in [0, 1]"
            )
        if source_geometry_power <= 0:
            raise ValueError(
                "source_geometry_power must be positive"
            )
        if source_geometry_margin < 0:
            raise ValueError(
                "source_geometry_margin must be non-negative"
            )
        if ignition_hand_exclusion_radius < 0:
            raise ValueError(
                "ignition_hand_exclusion_radius must be non-negative"
            )
        if ignition_contact_radius <= ignition_hand_exclusion_radius:
            raise ValueError(
                "ignition_contact_radius must exceed "
                "ignition_hand_exclusion_radius"
            )
        if not 0.0 <= committed_memory_feedback_strength <= 1.0:
            raise ValueError(
                "committed_memory_feedback_strength must lie in [0, 1]"
            )
        belief_memory_enabled = (
            aligned_belief_kv_enabled
            or memory_consolidation_enabled
        )
        consistent_role_kv_enabled = (
            oracle_kv_enabled
            or (hand_role_enabled and not factorized_bayes_enabled)
        )
        if routing_mode not in {
            "dynamic_sog",
            "oracle_role_flow",
            "oracle_role_flow_kv",
            "oracle_role_residual",
            "oracle_role_residual_kv",
            "hand_role_residual_kv",
            "hand_role_adaptive_kv",
            "hand_role_posterior_flow_kv",
            "hand_role_bayes_flow_kv",
            "hand_role_bayes_flow_dual_kv",
            "hand_role_bayes_flow_consolidated_kv",
            "hand_role_bayes_flow_commitment_kv",
            "hand_role_bayes_flow_identity_kv",
            "hand_role_bayes_flow_tokenprop_kv",
            "hand_role_bayes_flow_customized_kv",
            "hand_role_factorized_bayes_kv",
            "hand_role_factorized_causal_owner_kv",
        }:
            raise ValueError(f"Unsupported routing_mode: {routing_mode}")
        reference_already_bootstrapped = (
            reference_identity_enabled
            and _target_identity_memory is not None
            and _target_identity_memory.reference_bootstrapped
        )
        if reference_identity_enabled and not reference_already_bootstrapped:
            if (
                src_initial_latent is None
                or trg_initial_latent is None
                or not independent_first_frame
                or src_initial_latent.shape[1] != 1
                or trg_initial_latent.shape[1] != 1
            ):
                raise ValueError(
                    "Customized identity mode requires one spatially "
                    "aligned edited reference through independent "
                    "src/trg initial latents"
                )
        if not 0.0 <= contact_target_weight <= 1.0:
            raise ValueError(
                "contact_target_weight must lie in [0, 1], got "
                f"{contact_target_weight}"
            )
        if posterior_flow_mode not in {"soft", "hard"}:
            raise ValueError(
                "posterior_flow_mode must be 'soft' or 'hard'"
            )
        if not 0.0 <= hand_posterior_threshold <= 1.0:
            raise ValueError("hand_posterior_threshold must be in [0, 1]")
        if not 0.0 < hand_max_object_coverage <= 1.0:
            raise ValueError("hand_max_object_coverage must be in (0, 1]")
        if hand_proximity_radius < 0:
            raise ValueError("hand_proximity_radius must be non-negative")
        if hand_propagation_steps < 0:
            raise ValueError("hand_propagation_steps must be non-negative")
        if not 0.0 <= hand_visibility_ratio <= 1.0:
            raise ValueError("hand_visibility_ratio must be in [0, 1]")
        if not 0.0 <= hand_temporal_weight <= 1.0:
            raise ValueError("hand_temporal_weight must be in [0, 1]")
        if not -1.0 < hand_query_similarity_threshold < 1.0:
            raise ValueError(
                "hand_query_similarity_threshold must be in (-1, 1)"
            )
        hand_query_layers = tuple(hand_query_layers)
        if not hand_query_layers:
            raise ValueError("hand_query_layers must not be empty")
        if any(
            layer < 0 or layer >= self.num_transformer_blocks
            for layer in hand_query_layers
        ):
            raise ValueError(
                "hand_query_layers must contain valid transformer layers"
            )
        identity_memory_layers = tuple(
            hand_query_layers
            if immutable_target_layers is None
            else immutable_target_layers
        )
        if not identity_memory_layers:
            raise ValueError("Immutable target layers must not be empty")
        if len(set(identity_memory_layers)) != len(identity_memory_layers):
            raise ValueError("Immutable target layers must be unique")
        if any(
            layer < 0 or layer >= self.num_transformer_blocks
            for layer in identity_memory_layers
        ):
            raise ValueError(
                "Immutable target layers must contain valid transformer layers"
            )
        if immutable_target_num_prototypes <= 0:
            raise ValueError(
                "immutable_target_num_prototypes must be positive"
            )
        if immutable_target_value_mode not in {
            "residual", "subspace", "absolute"
        }:
            raise ValueError(
                "immutable_target_value_mode must be 'residual', "
                "'subspace', or 'absolute'"
            )
        if not 0.0 <= factorized_geometry_strength <= 1.0:
            raise ValueError(
                "factorized_geometry_strength must lie in [0, 1]"
            )
        if (
            factorized_orthogonal_geometry
            and not factorized_immutable_target_memory
        ):
            raise ValueError(
                "Orthogonal geometry requires immutable target memory"
            )
        if not (
            0.0
            <= hand_field_quantile_low
            < hand_field_quantile_high
            <= 1.0
        ):
            raise ValueError(
                "hand field quantiles must satisfy 0 <= low < high <= 1"
            )
        if hand_field_power <= 0:
            raise ValueError("hand_field_power must be positive")
        if not 0.0 <= hand_field_weight <= 1.0:
            raise ValueError("hand_field_weight must be in [0, 1]")
        if hand_field_candidate_radius < 0:
            raise ValueError(
                "hand_field_candidate_radius must be non-negative"
            )
        if hand_field_update_mode not in {
            "off",
            "diagnostic",
            "posterior",
        }:
            raise ValueError(
                "hand_field_update_mode must be one of "
                "{'off', 'diagnostic', 'posterior'}"
            )
        if contact_graph_mode not in CONTACT_GRAPH_MODES:
            raise ValueError(
                f"Unsupported contact_graph_mode: {contact_graph_mode}"
            )
        if (
            contact_graph_mode != "no_graph"
            and routing_mode != "oracle_role_residual_kv"
        ):
            raise ValueError(
                "Contact graph modes require "
                "routing_mode=oracle_role_residual_kv"
            )
        if contact_graph_topk <= 0:
            raise ValueError("contact_graph_topk must be positive")
        if contact_graph_radius <= 0:
            raise ValueError("contact_graph_radius must be positive")
        if not 0.0 <= contact_graph_min_confidence <= 1.0:
            raise ValueError(
                "contact_graph_min_confidence must be in [0, 1]"
            )
        if contact_graph_strength < 0:
            raise ValueError("contact_graph_strength must be non-negative")
        if not (
            0
            <= contact_graph_layer_start
            < contact_graph_layer_end
            <= self.num_transformer_blocks
        ):
            raise ValueError(
                "Contact graph layer range must satisfy "
                f"0 <= start < end <= {self.num_transformer_blocks}"
            )
        if oracle_role_enabled:
            if oracle_object_mask is None or oracle_hand_mask is None:
                raise ValueError(
                    "Oracle role modes require oracle object and hand masks"
                )
            expected_role_shape = (batch_size, num_frames, height, width)
            if tuple(oracle_object_mask.shape) != expected_role_shape:
                raise ValueError(
                    f"oracle_object_mask must have shape {expected_role_shape}, "
                    f"got {tuple(oracle_object_mask.shape)}"
                )
            if tuple(oracle_hand_mask.shape) != expected_role_shape:
                raise ValueError(
                    f"oracle_hand_mask must have shape {expected_role_shape}, "
                    f"got {tuple(oracle_hand_mask.shape)}"
                )
            oracle_object_mask = oracle_object_mask.to(
                device=src_video.device, dtype=torch.bool
            )
            oracle_hand_mask = oracle_hand_mask.to(
                device=src_video.device, dtype=torch.bool
            )
            if save_role_dir is not None:
                os.makedirs(save_role_dir, exist_ok=True)
        if hand_role_enabled:
            expected_role_shape = (batch_size, num_frames, height, width)
            if hand_only_mask is None:
                raise ValueError(
                    f"{routing_mode} requires hand_only_mask"
                )
            if tuple(hand_only_mask.shape) != expected_role_shape:
                raise ValueError(
                    f"hand_only_mask must have shape {expected_role_shape}, "
                    f"got {tuple(hand_only_mask.shape)}"
                )
            hand_only_mask = hand_only_mask.to(
                device=src_video.device,
                dtype=torch.bool,
            )
            if tuple(hand_occupancy_mask.shape) != expected_role_shape:
                raise ValueError(
                    "hand_occupancy_mask must have shape "
                    f"{expected_role_shape}, got "
                    f"{tuple(hand_occupancy_mask.shape)}"
                )
            if tuple(hand_persistent_mask.shape) != expected_role_shape:
                raise ValueError(
                    "hand_persistent_mask must have shape "
                    f"{expected_role_shape}, got "
                    f"{tuple(hand_persistent_mask.shape)}"
                )
            hand_occupancy_mask = hand_occupancy_mask.to(
                device=src_video.device, dtype=torch.float32
            ).clamp(0.0, 1.0)
            hand_persistent_mask = hand_persistent_mask.to(
                device=src_video.device, dtype=torch.bool
            )
            if oracle_source_owner_mask is not None:
                if tuple(oracle_source_owner_mask.shape) != expected_role_shape:
                    raise ValueError(
                        "oracle_source_owner_mask must have shape "
                        f"{expected_role_shape}, got "
                        f"{tuple(oracle_source_owner_mask.shape)}"
                    )
                oracle_source_owner_mask = (
                    oracle_source_owner_mask.to(
                        device=src_video.device,
                        dtype=torch.bool,
                    )
                )
            if oracle_source_owner_full_mask is not None:
                if (
                    tuple(oracle_source_owner_full_mask.shape)
                    != expected_role_shape
                ):
                    raise ValueError(
                        "oracle_source_owner_full_mask must have shape "
                        f"{expected_role_shape}, got "
                        f"{tuple(oracle_source_owner_full_mask.shape)}"
                    )
                oracle_source_owner_full_mask = (
                    oracle_source_owner_full_mask.to(
                        device=src_video.device,
                        dtype=torch.bool,
                    )
                )
            if save_role_dir is not None:
                os.makedirs(save_role_dir, exist_ok=True)
        if routing_mode in {
            "oracle_role_residual",
            "oracle_role_residual_kv",
        }:
            print(
                "ORACLE_ROLE_RESIDUAL "
                f"mode={routing_mode} "
                f"contact_target_weight={contact_target_weight:.3f}"
            )
        if adaptive_role_enabled:
            print(
                "HAND_ROLE_ADAPTIVE "
                "online_radius=True online_visibility=True "
                "online_temporal=True online_threshold=True "
                "online_field_reliability=True"
            )
            if posterior_flow_enabled:
                print(
                    "POSTERIOR_RESIDUAL_FLOW "
                    f"mode={posterior_flow_mode} "
                    f"use_field={posterior_flow_use_field} "
                    "contact_split=posterior"
                )
            if factorized_bayes_enabled:
                print(
                    "FACTORIZED_BAYES_BRIDGE "
                    "roles=object,boundary,hand,background,unknown "
                    "provenance=source,target,unknown "
                    "operators=source_key,source_value,source_residual,target_memory "
                    "attention=asymmetric_qk_value "
                    "unknown=native_streamgve"
                )
                if causal_ownership_enabled:
                    print(
                        "CAUSAL_OBJECT_OWNERSHIP "
                        "ignition=text_hand_interaction "
                        + (
                            "transport=bidirectional_source_rgb_raft "
                            "camera_motion=background_median "
                            "velocity_role=counterfactual_edit_response "
                            if motion_geometry_owner
                            else "transport=clean_source_query "
                        )
                        +
                        "provenance=target_value_target_residual "
                        "absence=no_write_keep_reassociation_state"
                    )
                    if factorized_native_target_history:
                        print(
                            "NATIVE_TARGET_HISTORY_MODE "
                            "attention=native_streamgve_dense_clean_target "
                            "factorized_attention=counterfactual_only "
                            "immutable_kv_write=disabled "
                            "owner_source_block="
                            f"{int(factorized_owner_source_block)}"
                        )
                    if role_fixed_native_history:
                        print(
                            "ROLE_FIXED_NATIVE_KV_MODE "
                            + (
                                "payload=immutable_canonical_target_kv "
                                "mutable=source_address_lineage_only "
                                f"payload_blend={native_history_payload_blend_strength:.3f} "
                                if native_history_payload_invariant_lineage
                                else "payload=final_clean_native_target_kv "
                            )
                            + "address=clean_source_pre_rope_k "
                            + (
                                "timescales=canonical_target_plus_mutable_source_lineage "
                                if native_history_payload_invariant_lineage
                                else "timescales=recent_clean_target_entry_plus_canonical_fallback "
                                if native_history_recent_entry_bridge
                                else "timescales=immutable_ignition_plus_recent_block "
                            )
                            + "position=fixed_relative_3d_rope "
                            "fallback=exact_native_926 "
                            f"layers={role_native_history_layers} "
                            f"topk={native_history_topk} "
                            f"logit_bias={native_history_canonical_logit_bias:.3f} "
                            "bootstrap_time="
                            + (
                                "coalesced_same_commit"
                                if native_history_coalesce_bootstrap_time
                                else "legacy_sequential"
                            )
                            + " bypass_blocks="
                            + (
                                ",".join(
                                    str(block)
                                    for block in role_native_history_bypass_blocks
                                )
                                if role_native_history_bypass_blocks
                                else "none"
                            )
                            + " source_part_mode="
                            + str(
                                "soft_canonical_trust_region"
                                if native_history_source_part_consistency
                                else "disabled"
                            )
                            + " min_part_similarity="
                            + f"{native_history_min_part_similarity:.3f}"
                            + " part_margin="
                            + f"{native_history_part_similarity_margin:.3f}"
                            + " part_bias="
                            + f"{native_history_part_bias_strength:.3f}"
                            + " part_trust_ratio="
                            + f"{native_history_part_refinement_ratio:.3f}"
                        )
                        if native_history_transactional_owner:
                            print(
                                "TRANSACTIONAL_NATIVE_OWNER_MODE "
                                + (
                                    "source=automatic_hand_attention_source_transport_flow "
                                    if oracle_source_owner_full_mask is None
                                    else "source=oracle_complete_matte "
                                )
                                + (
                                    "external_object_mask=disabled "
                                    if oracle_source_owner_full_mask is None
                                    else "external_object_mask=enabled_oracle_ablation "
                                )
                                +
                                "read=core_plus_contact_plus_bounded_lifecycle "
                                "write=visible_non_hand_core_only "
                                "source_suppression=verified_kv_admission_only "
                                "max_missing_frames="
                                f"{native_history_owner_max_missing_frames} "
                                "suppression="
                                f"{native_history_verified_source_suppression:.3f}"
                            )
                            if native_history_consistent_transaction:
                                if native_history_payload_invariant_lineage:
                                    print(
                                        "CANONICAL_APPEARANCE_TRANSACTION "
                                        "target_payload=immutable_ignition_only "
                                        "mutable_payload=disabled "
                                        "recent=clean_source_address_lineage_only "
                                        "read=matched_canonical_target_minus_source_value "
                                        "geometry=current_native_stream "
                                        "write=address_only_after_ignition "
                                        "source=remove_antagonistic_appearance_only "
                                        "lifecycle=one_update_per_chunk"
                                    )
                                elif native_history_recent_entry_bridge:
                                    print(
                                        "RECENT_EDITED_ENTRY_BRIDGE "
                                        "input=hand_information_only "
                                        "recent=complete_previous_clean_target_kv "
                                        + (
                                            "scope=all_motion_owner_latent_frames "
                                            if native_history_motion_owner_dense_read
                                            else "scope=first_latent_frame_per_causal_block "
                                        )
                                        +
                                        "address=current_source_to_previous_source "
                                        + (
                                            "priority=authorized_recent_else_exact_native "
                                            if native_history_token_atomic_payload
                                            else "priority=recent_then_immutable_canonical_fallback "
                                        )
                                        +
                                        "write=uncertainty_gated_transaction "
                                        "abstention=hold_last_trusted_recent "
                                        f"strength={native_history_entry_bridge_strength:.3f} "
                                        "min_residual_consensus="
                                        f"{native_history_dense_recent_min_residual_consensus:.3f}"
                                        + (
                                            " dual_evidence=source_address_plus_"
                                            "canonical_payload_residual "
                                            "min_payload_consistency="
                                            f"{native_history_min_payload_consistency:.3f}"
                                            if native_history_dual_evidence_arbitration
                                            else " dual_evidence=disabled"
                                        )
                                        + (
                                            " payload_commit=token_atomic "
                                            + (
                                                "update=source_addressed_"
                                                + (
                                                    "last_trusted_residual_lineage "
                                                    if native_history_last_trusted_appearance
                                                    else "persistent_value_residual_upsert "
                                                )
                                                if native_history_persistent_residual_upsert
                                                else ""
                                            )
                                            + "unauthorized=exact_native"
                                            if native_history_token_atomic_payload
                                            else " payload_commit=block_atomic"
                                        )
                                    )
                                else:
                                    print(
                                        "CONSISTENT_NATIVE_KV_TRANSACTION "
                                        "recent=compact_write_approved "
                                        "empty_write=hold_last_commit "
                                        "read=soft_owner_x_source_match "
                                        "arbitration=same_retrieval_strength "
                                        "source=remove_antagonistic_appearance_only "
                                        "lifecycle=one_update_per_chunk"
                                    )
                        if native_history_verified_attention_authority:
                            print(
                                "VERIFIED_ATTENTION_AUTHORITY_MODE "
                                "scope=automatic_owner_and_successful_kv_read "
                                "native=motion_geometry_fallback "
                                "factorized=target_value_authority "
                                "start_layer="
                                f"{max(role_native_history_layers)} "
                                "strength="
                                f"{native_history_attention_authority_strength:.3f} "
                                "external_object_mask=disabled"
                            )
            if bayes_flow_enabled:
                print(
                    "BAYES_RESIDUAL_FLOW "
                    "beliefs=non_exclusive "
                    "precision=online_fp32 "
                    "field_role=precision_only"
                )
                if aligned_belief_kv_enabled:
                    print(
                        "BELIEF_DUAL_KV "
                        "target_memory=edit_belief "
                        "source_memory=preserve_belief "
                        "conflict=dual_active "
                        "memory=aligned_fusion "
                        "attention=native_single_pass "
                        "current_context=native"
                    )
                if memory_consolidation_enabled:
                    print(
                        "CAUSAL_MEMORY_CONSOLIDATION "
                        "transport=source_query_affinity "
                        "update=precision_filter "
                        "state=sufficient_statistics "
                        "materialization=aligned_kv"
                    )
                if edit_commitment_enabled:
                    print(
                        "PERSISTENT_EDIT_COMMITMENT "
                        "trigger=hand_interaction "
                        "transport=committed_token_forward_splat "
                        "presence=semantic_match "
                        "preserve_release=object_core"
                    )
                if target_identity_enabled:
                    print(
                        "SLOW_TARGET_IDENTITY_MEMORY "
                        f"layers={identity_memory_layers} "
                        f"slots={immutable_target_num_prototypes} "
                        "write=precision_statistics "
                        "read=value_only "
                        "belief_feedback=identity_match"
                        + (
                            " reference_prior=8"
                            if reference_identity_enabled
                            else ""
                        )
                    )
                    if factorized_target_identity:
                        print(
                            "FACTORIZED_TARGET_IDENTITY "
                            "localization=source_key "
                            "appearance=target_key_value "
                            f"strength={identity_correction_strength:.3f} "
                            "routing=soft_value_only"
                        )
                    if source_coordinate_identity:
                        print(
                            "SOURCE_COORDINATE_IDENTITY "
                            + (
                                "prediction=oracle_clean_source_mask "
                                "read=per_frame_oracle_owner "
                                "write=verified_oracle_owner "
                                if oracle_source_owner_mask is not None
                                else
                                "prediction=clean_source_prestep "
                                "read=transported_owner "
                                "write=verified_ignition_core "
                            )
                            +
                            "appearance=target_minus_source_value "
                            f"correction_strength="
                            f"{identity_correction_strength:.3f} "
                            f"source_suppression="
                            f"{identity_source_suppression:.3f} "
                            f"support_floor={identity_support_floor:.3f} "
                            "owner_update=clean_source_only "
                            "ignition=write_validator"
                            + (
                                " residual_carry=source_coordinate_frozen "
                                f"carry_strength="
                                f"{identity_residual_carry_strength:.3f}"
                                if source_identity_residual_carry
                                else " residual_carry=off"
                            )
                        )
                    if factorized_immutable_target_memory:
                        print(
                            "IMMUTABLE_TARGET_APPEARANCE "
                            f"layers={identity_memory_layers} "
                            f"slots={immutable_target_num_prototypes} "
                            f"value_mode={immutable_target_value_mode} "
                            f"hard_owner={int(immutable_target_hard_owner)} "
                            "writer=first_chunk_only"
                        )
                    if factorized_orthogonal_geometry:
                        print(
                            "FACTORIZED_ORTHOGONAL_GEOMETRY "
                            "source=reconstruction_residual "
                            "projection=remove_edit_antagonism "
                            f"strength={factorized_geometry_strength:.3f} "
                            "schedule=early_to_late_decay"
                        )
                    if source_owner_geometry_envelope:
                        print(
                            "SOURCE_OWNER_GEOMETRY_ENVELOPE "
                            "geometry=current_clean_source "
                            "appearance_region=source_owner "
                            f"strength={source_geometry_strength:.3f} "
                            f"power={source_geometry_power:.3f} "
                            f"margin={source_geometry_margin} "
                            "hand=source_preserved"
                        )
                    if identity_visibility_lifecycle:
                        print(
                            "IDENTITY_VISIBILITY_LIFECYCLE "
                            "states=visible,occluded,absent "
                            "absent_read=0 "
                            "absent_write=0 "
                            f"max_occluded_blocks="
                            f"{identity_max_occluded_blocks}"
                        )
                if appearance_leakage_decomposition:
                    print(
                        "TRAINING_FREE_EDIT_IGNITION "
                        "core=field_semantic_hand_contact "
                        "residual=remove_antagonistic_projection "
                        f"hand_exclusion_radius="
                        f"{ignition_hand_exclusion_radius} "
                        f"contact_radius={ignition_contact_radius}"
                    )
        elif hand_role_enabled:
            print(
                "HAND_ROLE_RESIDUAL "
                f"posterior_threshold={hand_posterior_threshold:.3f} "
                f"max_object_coverage={hand_max_object_coverage:.3f} "
                f"proximity_radius={hand_proximity_radius} "
                f"propagation_steps={hand_propagation_steps} "
                f"visibility_ratio={hand_visibility_ratio:.3f} "
                f"temporal_weight={hand_temporal_weight:.3f} "
                f"field_weight={hand_field_weight:.3f} "
                f"field_mode={hand_field_update_mode} "
                f"query_layers={hand_query_layers}"
            )
        # #region debug-point H1:network-reporter
        def _debug_report(hypothesis_id, location, msg, data):
            try:
                import json
                import time
                import urllib.request
                url = os.environ.get(
                    "DEBUG_SERVER_URL",
                    "http://10.254.206.67:7777/event",
                )
                payload = {
                    "sessionId": os.environ.get(
                        "DEBUG_SESSION_ID",
                        "reference-source-regression",
                    ),
                    "runId": os.environ.get("DEBUG_RUN_ID", "pre-fix"),
                    "hypothesisId": hypothesis_id,
                    "location": location,
                    "msg": f"[DEBUG] {msg}",
                    "data": data,
                    "ts": int(time.time() * 1000),
                }
                request = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(request, timeout=0.2).read()
            except Exception:
                pass
        # #endregion
        if not independent_first_frame or (independent_first_frame and trg_initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = trg_initial_latent.shape[1] if trg_initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames

        src_prompts, trg_prompts, src_trigger_words, trg_trigger_words = self._check_prompts(
            src_prompts, trg_prompts, src_trigger_words, trg_trigger_words
        )
        conditional_dict = self.text_encoder(
            text_prompts=src_prompts + trg_prompts
        )
        src_conditional_dict = self.text_encoder(
            text_prompts=src_prompts
        )
        trg_conditional_dict = self.text_encoder(
            text_prompts=trg_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=src_video.device,
            dtype=src_video.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache, trg_fg_mask cache, and crossattn cache
        kv_cache_src = self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=src_video.dtype,
            device=src_video.device
        )
        kv_cache_trg = self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=src_video.dtype,
            device=src_video.device
        )
        trg_fg_mask_cache = self._initialize_trg_fg_mask_cache(
            batch_size=batch_size,
            device=src_video.device
        )
        target_owned_mask_cache = (
            self._initialize_target_owned_mask_cache(
                batch_size=batch_size,
                device=src_video.device,
            )
            if target_owned_object_handoff
            else None
        )
        belief_kv_weight_cache = (
            self._initialize_belief_kv_weight_cache(
                batch_size=batch_size,
                device=src_video.device,
            )
            if belief_memory_enabled
            else None
        )
        factorized_operator_cache = (
            self._initialize_factorized_operator_cache(
                batch_size=batch_size,
                device=src_video.device,
            )
            if factorized_bayes_enabled
            else None
        )
        crossattn_cache_src = self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=src_video.dtype,
            device=src_video.device
        )
        crossattn_cache_trg = self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=src_video.dtype,
            device=src_video.device
        )
        crossattn_cache_target_semantic = (
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=src_video.dtype,
                device=src_video.device,
            )
            if target_semantic_competition
            else None
        )
        kv_cache_target_semantic = (
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=src_video.dtype,
                device=src_video.device,
            )
            if target_semantic_competition
            else None
        )
        crossattn_cache_dual = self._initialize_crossattn_cache(
            batch_size=batch_size * 2,
            dtype=src_video.dtype,
            device=src_video.device
        )
        # Initialize some helper
        self._initialize_noise_statistics(reuse_noise_temporal_mean)
        role_flow_router = RoleFlowRouter()
        residual_role_flow_router = ResidualRoleFlowRouter()
        posterior_residual_flow_router = PosteriorResidualFlowRouter()
        bayes_residual_flow_router = BayesResidualFlowRouter()
        factorized_operator_builder = FactorizedBayesOperatorBuilder()
        control_belief_builder = CausalControlBeliefBuilder()
        hand_role_inferencer = _hand_role_inferencer
        if hand_role_inferencer is None:
            hand_role_inferencer = HandRoleInferencer(
                hand_proximity_radius=hand_proximity_radius,
                propagation_steps=hand_propagation_steps,
                max_object_coverage=hand_max_object_coverage,
                visibility_ratio=hand_visibility_ratio,
                temporal_weight=hand_temporal_weight,
                query_similarity_threshold=(
                    hand_query_similarity_threshold
                ),
                field_quantile_low=hand_field_quantile_low,
                field_quantile_high=hand_field_quantile_high,
                field_power=hand_field_power,
                field_weight=hand_field_weight,
                field_candidate_radius=hand_field_candidate_radius,
                connected_hysteresis=hand_connected_hysteresis,
                connected_growth_steps=hand_connected_growth_steps,
                connected_candidate_ratio=(
                    hand_connected_candidate_ratio
                ),
                soft_hand_contact=hand_causal_evidence,
                adaptive=adaptive_role_enabled,
            )
        memory_consolidator = _memory_consolidator
        if memory_consolidation_enabled and memory_consolidator is None:
            memory_consolidator = CausalMemoryConsolidator()
        edit_commitment_controller = _edit_commitment_controller
        if edit_commitment_enabled and edit_commitment_controller is None:
            edit_commitment_controller = EditCommitmentController()
        target_identity_memory = _target_identity_memory
        if target_identity_enabled and target_identity_memory is None:
            target_identity_memory = SlowTargetIdentityMemory(
                layers=identity_memory_layers,
                num_prototypes=immutable_target_num_prototypes,
                store_value_residual=(
                    source_coordinate_identity
                    or (
                        factorized_immutable_target_memory
                        and immutable_target_value_mode in {
                            "residual", "subspace"
                        }
                    )
                ),
            )
        identity_owner_tracker = _identity_owner_tracker
        if source_coordinate_identity and identity_owner_tracker is None:
            identity_owner_tracker = CausalIdentityOwnerTracker(
                max_candidates=identity_tokenprop_max_candidates,
                max_area_fraction=hand_max_object_coverage,
                recover_visibility_from_source_match=(
                    source_identity_residual_carry
                ),
            )
        causal_ownership_tracker = _causal_ownership_tracker
        if causal_ownership_enabled and causal_ownership_tracker is None:
            causal_ownership_tracker = (
                MotionAwareGeometryOwnerTracker(
                    source_flow_cache,
                    bootstrap_frames=self.num_frame_per_block,
                    max_occluded_frames=(
                        identity_max_occluded_blocks
                        * self.num_frame_per_block
                    ),
                )
                if motion_geometry_owner
                else CausalObjectOwnershipTracker(
                max_candidates=identity_tokenprop_max_candidates,
                max_area_fraction=hand_max_object_coverage,
                min_similarity=identity_tokenprop_min_similarity,
                max_occluded_frames=(
                    identity_max_occluded_blocks
                    * self.num_frame_per_block
                ),
                )
            )
        native_owner_tracker = _native_owner_tracker
        if native_history_transactional_owner and native_owner_tracker is None:
            if oracle_source_owner_full_mask is None:
                native_owner_tracker = AutomaticTransactionalOwnerTracker(
                    max_missing_frames=(
                        native_history_owner_max_missing_frames
                    ),
                    blockwise_lifecycle=(
                        native_history_consistent_transaction
                    ),
                )
            else:
                native_owner_tracker = CausalReadOnlyOwnerTracker(
                    max_candidates=identity_tokenprop_max_candidates,
                    max_area_fraction=hand_max_object_coverage,
                    min_similarity=identity_tokenprop_min_similarity,
                    max_missing_frames=(
                        native_history_owner_max_missing_frames
                    ),
                )
        paired_edit_memory = _causal_paired_edit_memory
        role_native_kv_history = _role_native_kv_history
        if role_fixed_native_history and role_native_kv_history is None:
            role_native_kv_history = RoleConditionedNativeKVHistory(
                layers=role_native_history_layers,
                tokens_per_frame=self.frame_seq_length,
                max_tokens_per_frame=(
                    native_history_max_tokens_per_frame
                ),
                min_write_confidence=(
                    native_history_min_write_confidence
                ),
                payload_invariant_lineage=(
                    native_history_payload_invariant_lineage
                ),
                transactional_compact_recent=(
                    native_history_consistent_transaction
                    and not native_history_recent_entry_bridge
                ),
                transactional_dense_recent=(
                    native_history_recent_entry_bridge
                ),
                token_atomic_dense_recent=(
                    native_history_token_atomic_payload
                ),
                persistent_residual_upsert=(
                    native_history_persistent_residual_upsert
                ),
                last_trusted_residual_lineage=(
                    native_history_last_trusted_appearance
                ),
                flow_indexed_residual_ledger=(
                    native_history_flow_indexed_residual
                ),
                decoupled_flow_trust=(
                    native_history_decoupled_flow_trust
                ),
                multiframe_identity_sink=(
                    native_history_multiframe_identity_sink
                ),
                timestep_counterfactual_memory=(
                    native_history_timestep_counterfactual_memory
                ),
                source_flow_cache=source_flow_cache,
                flow_min_confidence=native_history_flow_min_confidence,
                residual_update_min_cosine=(
                    native_history_residual_update_min_cosine
                ),
                residual_update_min_magnitude_ratio=(
                    native_history_residual_update_min_magnitude_ratio
                ),
                dense_recent_min_residual_consensus=(
                    native_history_dense_recent_min_residual_consensus
                ),
                min_lineage_similarity=native_history_min_similarity,
            )
        if causal_paired_edit_memory and paired_edit_memory is None:
            paired_edit_memory = CausalPairedEditMemory(
                layers=paired_edit_memory_layers,
                max_tokens=paired_memory_max_tokens,
                max_tokens_per_block=(
                    paired_memory_max_tokens_per_block
                ),
                min_commit_confidence=(
                    paired_memory_min_commit_confidence
                ),
                min_similarity=paired_memory_min_similarity,
                coordinate_bias=paired_memory_coordinate_bias,
                coordinate_radius=paired_memory_coordinate_radius,
                min_residual_consensus=(
                    paired_memory_min_residual_consensus
                ),
                source_part_consistency=(
                    paired_memory_source_part_consistency
                ),
                min_part_similarity=paired_memory_min_part_similarity,
                part_similarity_margin=(
                    paired_memory_part_similarity_margin
                ),
                topk=paired_memory_topk,
                source_transport=paired_memory_source_transport,
                transport_min_similarity=(
                    paired_memory_transport_min_similarity
                ),
                transport_coordinate_radius=(
                    paired_memory_transport_coordinate_radius
                ),
                transport_cycle_radius=(
                    paired_memory_transport_cycle_radius
                ),
                transport_min_confidence=(
                    paired_memory_transport_min_confidence
                ),
                single_confidence=paired_memory_single_confidence,
                immutable_canonical_key_anchor=(
                    paired_memory_canonical_key_anchor
                ),
            )
        elif causal_paired_edit_memory:
            expected_paired_memory = CausalPairedEditMemory(
                layers=paired_edit_memory_layers,
                max_tokens=paired_memory_max_tokens,
                max_tokens_per_block=(
                    paired_memory_max_tokens_per_block
                ),
                min_commit_confidence=(
                    paired_memory_min_commit_confidence
                ),
                min_similarity=paired_memory_min_similarity,
                coordinate_bias=paired_memory_coordinate_bias,
                coordinate_radius=paired_memory_coordinate_radius,
                min_residual_consensus=(
                    paired_memory_min_residual_consensus
                ),
                source_part_consistency=(
                    paired_memory_source_part_consistency
                ),
                min_part_similarity=paired_memory_min_part_similarity,
                part_similarity_margin=(
                    paired_memory_part_similarity_margin
                ),
                topk=paired_memory_topk,
                source_transport=paired_memory_source_transport,
                transport_min_similarity=(
                    paired_memory_transport_min_similarity
                ),
                transport_coordinate_radius=(
                    paired_memory_transport_coordinate_radius
                ),
                transport_cycle_radius=(
                    paired_memory_transport_cycle_radius
                ),
                transport_min_confidence=(
                    paired_memory_transport_min_confidence
                ),
                single_confidence=paired_memory_single_confidence,
                immutable_canonical_key_anchor=(
                    paired_memory_canonical_key_anchor
                ),
            )
            if not paired_edit_memory.compatible_with(
                expected_paired_memory
            ):
                raise ValueError(
                    "Injected paired edit memory does not match the "
                    "current inference configuration"
                )
        identity_residual_carry = _identity_residual_carry
        if (
            source_identity_residual_carry
            and identity_residual_carry is None
        ):
            identity_residual_carry = SourceCoordinateResidualCarry(
                max_candidates=identity_tokenprop_max_candidates,
            )
        identity_token_propagator = _identity_token_propagator
        if (
            target_identity_tokenprop_enabled
            and identity_token_propagator is None
        ):
            identity_token_propagator = CausalObjectTokenPropagator(
                min_similarity=identity_tokenprop_min_similarity,
                gate_strength=identity_tokenprop_gate_strength,
                max_candidates=identity_tokenprop_max_candidates,
            )
        identity_support_filter = _identity_support_filter
        if (
            target_identity_tokenprop_enabled
            and identity_support_filter is None
        ):
            identity_support_filter = CausalConnectedSupportFilter()
        initial_target_owned_tokens = None
        if (
            target_owned_object_handoff
            and target_identity_memory.causal_edit_anchor_reset
            and num_input_frames > 0
        ):
            initial_target_owned_tokens = (
                target_identity_memory.recent_target_owned_tokens(
                    num_input_frames * self.frame_seq_length,
                    batch_size=batch_size,
                    device=src_video.device,
                )
            )

        # get trigger token indices
        trans_tokenizer = self.text_encoder.tokenizer.tokenizer
        tok_src = find_phrase_token_indices(trans_tokenizer, src_prompts, src_trigger_words)
        tok_trg = find_phrase_token_indices(trans_tokenizer, trg_prompts, trg_trigger_words)
        target_semantic_token_groups = None
        if target_semantic_competition:
            target_semantic_token_groups = {
                f"edit_{index:02d}": find_phrase_group_token_indices(
                    trans_tokenizer, trg_prompts, [phrase]
                )
                for index, phrase in enumerate(target_edit_phrases)
            }
            target_semantic_token_groups.update({
                f"preserve_{index:02d}": find_phrase_group_token_indices(
                    trans_tokenizer, trg_prompts, [phrase]
                )
                for index, phrase in enumerate(target_preserve_phrases)
            })
            print(
                "TARGET_SEMANTIC_TOKENS "
                f"edit_phrases={target_edit_phrases} "
                f"preserve_phrases={target_preserve_phrases}"
            )
        print(tok_src, tok_trg)

        # Step 2: Cache context feature
        current_start_frame = 0
        if trg_initial_latent is not None:
            # obtain both kv_cache and mask of both src and trg
            timestep = torch.zeros([batch_size, 1], device=src_video.device, dtype=torch.int64)
            if independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                slice_list = [(0, 1)] + [
                    (1 + idx * self.num_frame_per_block, 1 + (idx + 1) * self.num_frame_per_block)
                    for idx in range(num_input_blocks)
                ]
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block
                slice_list = [
                    (idx * self.num_frame_per_block, (idx + 1) * self.num_frame_per_block)
                    for idx in range(num_input_blocks)
                ]

            for left, right in slice_list:
                context_timestep = torch.ones(
                    [batch_size, right - left], device=src_video.device, dtype=torch.float32
                ) * self.args.context_noise
                #✨ src and src mask
                current_src_ref_latents = src_initial_latent[:, left: right]
                self._register_crossattn_mask_gatherer(crossattn_cache_src, tok_src, layers=mask_layers, fg_scale=fg_scale)
                if (
                    reference_identity_enabled
                    and not target_identity_memory.reference_bootstrapped
                ):
                    self._register_identity_key_capture(
                        kv_cache_src,
                        hand_query_layers,
                    )
                    self._register_query_capture(
                        kv_cache_src,
                        hand_query_layers,
                    )
                self.generator(
                    noisy_image_or_video=current_src_ref_latents,
                    conditional_dict=src_conditional_dict,
                    timestep=context_timestep,
                    kv_cache=kv_cache_src,
                    crossattn_cache=crossattn_cache_src,
                    current_start=left * self.frame_seq_length,
                )
                _, src_fg_mask_bin, _, _ = self._aggregate_crossattn_mask(crossattn_cache_src)
                reference_source_features = None
                if (
                    reference_identity_enabled
                    and not target_identity_memory.reference_bootstrapped
                ):
                    reference_source_features = (
                        self._aggregate_query_features(
                            kv_cache_src,
                            hand_query_layers,
                        )
                    )
                #✨ trg and trg mask
                current_trg_ref_latents = trg_initial_latent[:, left: right]
                self._register_crossattn_mask_gatherer(crossattn_cache_trg, tok_trg, layers=mask_layers, fg_scale=fg_scale)
                self.generator(
                    noisy_image_or_video=current_trg_ref_latents,
                    conditional_dict=trg_conditional_dict,
                    timestep=context_timestep,
                    kv_cache=kv_cache_trg,
                    crossattn_cache=crossattn_cache_trg,
                    current_start=left * self.frame_seq_length,
                )
                (
                    trg_fg_mask_soft,
                    trg_fg_mask_bin,
                    _,
                    _,
                ) = self._aggregate_crossattn_mask(
                    crossattn_cache_trg
                )
                if (
                    reference_identity_enabled
                    and not target_identity_memory.reference_bootstrapped
                ):
                    reference_hand_mask = hand_only_mask[
                        :, :right - left
                    ]
                    reference_bootstrap = (
                        build_reference_identity_bootstrap(
                            source_latent=current_src_ref_latents,
                            target_latent=current_trg_ref_latents,
                            target_attention=trg_fg_mask_soft,
                            hand_mask=reference_hand_mask,
                        )
                    )
                    reference_update = (
                        target_identity_memory.bootstrap_reference(
                            kv_cache=kv_cache_trg,
                            write_weight=(
                                reference_bootstrap.write_weight
                            ),
                            source_kv_cache=kv_cache_src,
                        )
                    )
                    (
                        reference_commitment,
                        reference_commitment_precision,
                    ) = edit_commitment_controller.bootstrap_reference(
                        source_features=reference_source_features,
                        edit_precision=(
                            reference_bootstrap.write_weight
                        ),
                    )
                    print(
                        "REFERENCE_IDENTITY_BOOTSTRAP "
                        "change="
                        f"{reference_bootstrap.change_score.mean().item():.4f} "
                        "semantic="
                        f"{reference_bootstrap.semantic_score.mean().item():.4f} "
                        "joint="
                        f"{reference_bootstrap.joint_score.mean().item():.4f} "
                        "hand_contact="
                        f"{reference_bootstrap.hand_contact_score.mean().item():.4f} "
                        "write="
                        f"{reference_bootstrap.write_weight.mean().item():.4f} "
                        "support="
                        f"{(reference_bootstrap.write_weight > 0).float().mean().item():.4f} "
                        "evidence="
                        f"{reference_update.accumulated_evidence.mean().item():.4f}"
                    )
                    print(
                        "REFERENCE_EDIT_COMMITMENT "
                        "commitment="
                        f"{reference_commitment.mean().item():.4f} "
                        "precision="
                        f"{reference_commitment_precision.mean().item():.4f} "
                        "anchor_score="
                        f"{edit_commitment_controller.anchor_score.mean().item():.4f}"
                    )
                    # #region debug-point H3:reference-bootstrap
                    _debug_selected = (
                        reference_bootstrap.write_weight > 0
                    )
                    _debug_selected_count = (
                        _debug_selected.sum().clamp_min(1)
                    )
                    _debug_contact = (
                        reference_bootstrap.hand_contact_score.reshape(
                            reference_bootstrap.write_weight.shape
                        )
                    )
                    _debug_report(
                        "H3",
                        "edit_causal_inference.py:reference-bootstrap",
                        "Reference bootstrap support and contact",
                        {
                            "support_coverage": (
                                _debug_selected.float().mean().item()
                            ),
                            "support_tokens": int(
                                _debug_selected.sum().item()
                            ),
                            "write_on_support": (
                                reference_bootstrap.write_weight[
                                    _debug_selected
                                ].sum()
                                / _debug_selected_count
                            ).item(),
                            "hand_contact_on_support": (
                                _debug_contact[_debug_selected].sum()
                                / _debug_selected_count
                            ).item(),
                            "reference_evidence": (
                                reference_update.accumulated_evidence
                                .mean().item()
                            ),
                        },
                    )
                    # #endregion
                    if save_role_dir is not None:
                        self._save_reference_identity_debug(
                            save_role_dir,
                            reference_bootstrap.as_debug_maps(),
                        )

                #✨ src & trg union
                current_trg_fg_mask = trg_fg_mask_bin | src_fg_mask_bin
                self._update_trg_fg_mask_cache(trg_fg_mask_cache, current_trg_fg_mask, kv_cache_trg)
                if target_owned_mask_cache is not None:
                    if initial_target_owned_tokens is None:
                        current_target_owned_mask = torch.zeros_like(
                            current_trg_fg_mask,
                            dtype=torch.bool,
                        )
                    else:
                        current_target_owned_mask = (
                            initial_target_owned_tokens[
                                :,
                                left * self.frame_seq_length:
                                right * self.frame_seq_length,
                            ]
                        )
                    self._update_target_owned_mask_cache(
                        target_owned_mask_cache,
                        current_target_owned_mask,
                        kv_cache_trg,
                    )
                if belief_memory_enabled:
                    self._update_belief_kv_weight_cache(
                        belief_kv_weight_cache,
                        current_preserve_action=torch.zeros_like(
                            current_trg_fg_mask,
                            dtype=torch.float32,
                        ),
                        kv_cache_trg=kv_cache_trg,
                    )
                if factorized_bayes_enabled:
                    initial_actions = {
                        "source_key_action": torch.zeros_like(
                            current_trg_fg_mask, dtype=torch.float32
                        ),
                        "source_value_action": torch.zeros_like(
                            current_trg_fg_mask, dtype=torch.float32
                        ),
                        "target_memory_action": torch.zeros_like(
                            current_trg_fg_mask, dtype=torch.float32
                        ),
                        "unknown_action": torch.ones_like(
                            current_trg_fg_mask, dtype=torch.float32
                        ),
                    }
                    self._update_factorized_operator_cache(
                        factorized_operator_cache,
                        current_actions=initial_actions,
                        kv_cache_trg=kv_cache_trg,
                    )

                output[:, left: right] = current_trg_ref_latents
                current_start_frame = right

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        denoising_step_list = self.denoising_step_list
        all_num_frames = [self.num_frame_per_block] * num_blocks
        persistent_identity_anchor_kv = None
        if independent_first_frame and trg_initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in tqdm(all_num_frames):
            if profile:
                block_start.record()

            src_input = src_video[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            denoised_pred = src_input

            context_timestep = torch.ones(
                [batch_size, current_num_frames],
                device=src_video.device,
                dtype=torch.float32
            ) * self.args.context_noise

            role_left = current_start_frame - num_input_frames
            role_right = role_left + current_num_frames
            current_source_frame_indices = (
                tuple(global_frame_indices[role_left:role_right])
                if global_frame_indices is not None
                else ()
            )
            if (
                native_history_flow_indexed_residual
                and role_native_kv_history is not None
            ):
                prepared_flow_read = role_native_kv_history.prepare_flow_read(
                    frame_indices=current_source_frame_indices,
                    spatial_shape=(height // 2, width // 2),
                    device=src_video.device,
                )
                flow_read_support = [
                    value.support.float().mean()
                    for value in prepared_flow_read.values()
                ]
                flow_read_confidence = [
                    value.confidence.float().mean()
                    for value in prepared_flow_read.values()
                ]
                print(
                    "FLOW_INDEXED_RESIDUAL_READ "
                    f"block={current_start_frame // self.num_frame_per_block} "
                    "frames="
                    + ",".join(str(value) for value in current_source_frame_indices)
                    + f" layers={len(prepared_flow_read)} "
                    + (
                        "state=empty support=0.0000 confidence=0.0000"
                        if not prepared_flow_read
                        else (
                            "state=transported support="
                            f"{torch.stack(flow_read_support).mean().item():.4f} "
                            "confidence="
                            f"{torch.stack(flow_read_confidence).mean().item():.4f}"
                        )
                    )
                )
                if native_history_timestep_counterfactual_memory:
                    prepared_tccm_correspondence = (
                        role_native_kv_history
                        .prepare_canonical_correspondence(
                            frame_indices=current_source_frame_indices,
                            spatial_shape=(height // 2, width // 2),
                            device=src_video.device,
                        )
                    )
                    tccm_support = [
                        value.support.float().mean()
                        for value in prepared_tccm_correspondence.values()
                    ]
                    tccm_confidence = [
                        value.confidence.float().sum()
                        / value.support.float().sum().clamp_min(1.0)
                        for value in prepared_tccm_correspondence.values()
                    ]
                    print(
                        "TCCM_CORRESPONDENCE_READ "
                        f"block={current_start_frame // self.num_frame_per_block} "
                        f"layers={len(prepared_tccm_correspondence)} "
                        "state=empty support=0.0000 confidence=0.0000"
                        if not prepared_tccm_correspondence
                        else (
                            "TCCM_CORRESPONDENCE_READ "
                            f"block={current_start_frame // self.num_frame_per_block} "
                            f"layers={len(prepared_tccm_correspondence)} "
                            "state=transported support="
                            f"{torch.stack(tccm_support).mean().item():.4f} "
                            "confidence="
                            f"{torch.stack(tccm_confidence).mean().item():.4f}"
                        )
                    )
            
            # obtain currently inprocessed kv_cache for dual branch
            shared_dict_dual = dict()
            if suppress_source_bg_value:
                shared_dict_dual[
                    "suppress_source_bg_value"
                ] = True
            if persistent_identity_anchor_kv is not None:
                shared_dict_dual[
                    "identity_anchor_kv"
                ] = persistent_identity_anchor_kv
                shared_dict_dual[
                    "identity_anchor_scale"
                ] = float(identity_anchor_scale)
            if role_fixed_native_history:
                if role_native_kv_history is None:
                    raise RuntimeError(
                        "Role-fixed native history was not initialized"
                    )
                native_history_block_index = (
                    current_start_frame // self.num_frame_per_block
                )
                shared_dict_dual.update({
                    "role_fixed_native_history": True,
                    "role_native_history_object": role_native_kv_history,
                    "role_native_history_reads": (
                        role_native_kv_history.read()
                    ),
                    "native_history_topk": int(native_history_topk),
                    "native_history_min_similarity": float(
                        native_history_min_similarity
                    ),
                    "native_history_min_query_confidence": float(
                        native_history_min_query_confidence
                    ),
                    "native_history_canonical_logit_bias": float(
                        native_history_canonical_logit_bias
                    ),
                    "native_history_coalesce_bootstrap_time": bool(
                        native_history_coalesce_bootstrap_time
                    ),
                    "role_native_history_bypass": (
                        native_history_block_index
                        in role_native_history_bypass_blocks
                    ),
                    "native_history_source_part_consistency": bool(
                        native_history_source_part_consistency
                    ),
                    "native_history_min_part_similarity": float(
                        native_history_min_part_similarity
                    ),
                    "native_history_part_similarity_margin": float(
                        native_history_part_similarity_margin
                    ),
                    "native_history_part_bias_strength": float(
                        native_history_part_bias_strength
                    ),
                    "native_history_part_refinement_ratio": float(
                        native_history_part_refinement_ratio
                    ),
                    "native_history_payload_invariant_lineage": bool(
                        native_history_payload_invariant_lineage
                    ),
                    "native_history_consistent_transaction": bool(
                        native_history_consistent_transaction
                    ),
                    "native_history_payload_blend_strength": float(
                        native_history_payload_blend_strength
                    ),
                    "native_history_recent_entry_bridge": bool(
                        native_history_recent_entry_bridge
                    ),
                    "native_history_motion_owner_dense_read": bool(
                        native_history_motion_owner_dense_read
                    ),
                    "native_history_entry_bridge_strength": float(
                        native_history_entry_bridge_strength
                    ),
                    "native_history_dual_evidence_arbitration": bool(
                        native_history_dual_evidence_arbitration
                    ),
                    "native_history_token_atomic_payload": bool(
                        native_history_token_atomic_payload
                    ),
                    "native_history_persistent_residual_upsert": bool(
                        native_history_persistent_residual_upsert
                    ),
                    "native_history_last_trusted_appearance": bool(
                        native_history_last_trusted_appearance
                    ),
                    "native_history_flow_indexed_residual": bool(
                        native_history_flow_indexed_residual
                    ),
                    "native_history_multiframe_identity_sink": bool(
                        native_history_multiframe_identity_sink
                    ),
                    "native_history_multiframe_sink_topk_per_frame": int(
                        native_history_multiframe_sink_topk_per_frame
                    ),
                    "native_history_multiframe_sink_source_logit_bias": float(
                        native_history_multiframe_sink_source_logit_bias
                    ),
                    "native_history_multiframe_sink_strength": float(
                        native_history_multiframe_sink_strength
                    ),
                    "native_history_timestep_counterfactual_memory": bool(
                        native_history_timestep_counterfactual_memory
                    ),
                    "native_history_tccm_flow_radius": float(
                        native_history_tccm_flow_radius
                    ),
                    "native_history_tccm_strength": float(
                        native_history_tccm_strength
                    ),
                    "native_history_tccm_max_error_ratio": float(
                        native_history_tccm_max_error_ratio
                    ),
                    "native_history_topology_complete_read": bool(
                        native_history_topology_complete_read
                    ),
                    "native_history_min_payload_consistency": float(
                        native_history_min_payload_consistency
                    ),
                })
            kv_cache_dual = self._concat_kv_cache(kv_cache_src, kv_cache_trg, shared_dict=shared_dict_dual)

            #✨ forward clean source video to get source mask, and store into kv_cache
            self._register_crossattn_mask_gatherer(crossattn_cache_src, tok_src, layers=mask_layers, fg_scale=fg_scale)
            if hand_role_enabled:
                self._register_query_capture(
                    kv_cache_src,
                    hand_query_layers,
                )
            if target_identity_enabled:
                self._register_identity_key_capture(
                    kv_cache_src,
                    identity_memory_layers,
                )
            self.generator(
                noisy_image_or_video=src_input,
                conditional_dict=src_conditional_dict,
                timestep=context_timestep,
                kv_cache=kv_cache_src,
                crossattn_cache=crossattn_cache_src,
                current_start=current_start_frame * self.frame_seq_length,
            )
            target_semantic_attention = None
            if target_semantic_competition:
                # Read-only target-text probe on the clean source block.  It
                # uses an isolated, source-only self-attention cache, so
                # neither source generation nor target/native-history state
                # is mutated. Its causal history contains only clean source.
                self._register_semantic_crossattn_gatherer(
                    crossattn_cache_target_semantic,
                    target_semantic_token_groups,
                    layers=mask_layers,
                )
                self.generator(
                    noisy_image_or_video=src_input,
                    conditional_dict=trg_conditional_dict,
                    timestep=context_timestep,
                    kv_cache=kv_cache_target_semantic,
                    crossattn_cache=crossattn_cache_target_semantic,
                    current_start=(
                        current_start_frame * self.frame_seq_length
                    ),
                )
                target_semantic_attention = (
                    self._aggregate_semantic_crossattn_masks(
                        crossattn_cache_target_semantic
                    )
                )
            if role_fixed_native_history:
                # The clean source forward above has now populated this
                # block. Expose its pre-RoPE K as a read-only address stream
                # so retrieval is independent of the denoising noise level.
                self._append_clean_src_kv_cache(
                    kv_cache_dual, kv_cache_src
                )
            src_fg_mask_soft, src_fg_mask_bin, _, _ = (
                self._aggregate_crossattn_mask(crossattn_cache_src)
            )
            source_query_features = (
                self._aggregate_query_features(
                    kv_cache_src,
                    hand_query_layers,
                )
                if hand_role_enabled
                else None
            )
            source_identity_keys = (
                self._collect_identity_keys(
                    kv_cache_src,
                    identity_memory_layers,
                )
                if target_identity_enabled
                else {}
            )
            current_roles = None
            role_edit_tokens = None
            contact_graphs = None
            hand_role_debug = None
            hand_role_inference = None
            current_control_belief = None
            current_belief_kv_weights = None
            current_factorized_operators: Optional[
                FactorizedBayesOperators
            ] = None
            current_causal_ownership = None
            flow_role_evidence = None
            current_transactional_owner = None
            current_memory_query_weight = None
            current_memory_plan: Optional[
                MemoryConsolidationPlan
            ] = None
            current_commitment: Optional[
                EditCommitmentResult
            ] = None
            current_identity_support = None
            identity_observation_belief = None
            identity_observation_tokens = None
            current_identity_update: Optional[
                TargetIdentityUpdate
            ] = None
            current_identity_propagation: Optional[
                TargetIdentityTokenPropagation
            ] = None
            current_connected_identity_support: Optional[
                ConnectedIdentitySupport
            ] = None
            current_causal_identity_bootstrap: Optional[
                TargetIdentityUpdate
            ] = None
            current_causal_identity_bootstrap_plan: Optional[
                FirstFrameIdentityBootstrap
            ] = None
            current_target_change_core = None
            current_edit_authority = None
            current_target_authority_tokens = None
            current_target_authority_support_tokens = None
            identity_tokenprop_support_tokens = None
            if oracle_role_enabled:
                current_roles = build_oracle_roles(
                    oracle_object_mask[:, role_left:role_right],
                    oracle_hand_mask[:, role_left:role_right],
                    boundary_radius=role_boundary_radius,
                )
                role_coverage = {
                    name: value.mean().item()
                    for name, value in current_roles.as_dict().items()
                }
                print(
                    "ORACLE_ROLE_FLOW "
                    f"block={current_start_frame // self.num_frame_per_block} "
                    + " ".join(
                        f"{name}={coverage:.4f}"
                        for name, coverage in role_coverage.items()
                    )
                )
                if contact_graph_mode != "no_graph":
                    block_index = (
                        current_start_frame // self.num_frame_per_block
                    )
                    contact_graphs = build_oracle_contact_graphs(
                        current_roles,
                        mode=contact_graph_mode,
                        topk=contact_graph_topk,
                        radius=contact_graph_radius,
                        min_confidence=contact_graph_min_confidence,
                        shuffle_seed=contact_graph_seed + block_index,
                    )
                    graph_stats = contact_graph_stats(contact_graphs)
                    print(
                        "ORACLE_CONTACT_GRAPH "
                        f"block={block_index} "
                        f"mode={contact_graph_mode} "
                        f"object_nodes={graph_stats['object_nodes']} "
                        f"valid_edges={graph_stats['valid_edges']}"
                    )
                # #region debug-point C:role-kv-mask-alignment
                role_edit_tokens = F.max_pool2d(
                    current_roles.edit_weight, kernel_size=2, stride=2
                ).bool().reshape(batch_size, -1)
                debug_src_tokens = src_fg_mask_bin.bool()
                debug_intersection = (
                    role_edit_tokens & debug_src_tokens
                ).float().sum()
                debug_union = (
                    role_edit_tokens | debug_src_tokens
                ).float().sum()
                _debug_report(
                    "C",
                    "edit_causal_inference.py:oracle-role-block",
                    "Oracle role and source KV mask alignment",
                    {
                        "block": current_start_frame // self.num_frame_per_block,
                        "role_edit_coverage": current_roles.edit_weight.mean().item(),
                        "kv_mask_coverage": debug_src_tokens.float().mean().item(),
                        "iou": (
                            debug_intersection / debug_union.clamp_min(1)
                        ).item(),
                        "role_recall": (
                            debug_intersection
                            / role_edit_tokens.float().sum().clamp_min(1)
                        ).item(),
                        "kv_precision": (
                            debug_intersection
                            / debug_src_tokens.float().sum().clamp_min(1)
                        ).item(),
                    },
                )
                # #endregion
                if save_role_dir is not None:
                    self._save_role_state(
                        save_role_dir,
                        current_start_frame // self.num_frame_per_block,
                        current_roles,
                    )
                    if contact_graphs is not None:
                        self._save_contact_graphs(
                            save_role_dir,
                            current_start_frame // self.num_frame_per_block,
                            contact_graphs,
                            contact_graph_mode,
                        )
            elif hand_role_enabled:
                hand_role_inference = hand_role_inferencer(
                    source_attention=src_fg_mask_soft,
                    hand_mask=hand_only_mask[:, role_left:role_right],
                    hand_occupancy=hand_occupancy_mask[
                        :, role_left:role_right
                    ],
                    source_features=source_query_features,
                )
                current_roles = hand_role_inference.roles
                hand_role_debug = hand_role_inference.debug
                hand_role_debug["hand_union"] = F.avg_pool2d(
                    hand_only_mask[
                        :, role_left:role_right
                    ].float().reshape(
                        batch_size * current_num_frames, 1, height, width
                    ),
                    kernel_size=2,
                    stride=2,
                ).reshape_as(
                    hand_role_debug["object_posterior"]
                ).clamp(0.0, 1.0)
                hand_role_debug["hand_occupancy"] = F.avg_pool2d(
                    hand_occupancy_mask[
                        :, role_left:role_right
                    ].float().reshape(
                        batch_size * current_num_frames, 1, height, width
                    ),
                    kernel_size=2,
                    stride=2,
                ).reshape_as(
                    hand_role_debug["object_posterior"]
                ).clamp(0.0, 1.0)
                hand_role_debug["hand_hard_exclusion"] = F.max_pool2d(
                    hand_persistent_mask[
                        :, role_left:role_right
                    ].float().reshape(
                        batch_size * current_num_frames, 1, height, width
                    ),
                    kernel_size=2,
                    stride=2,
                ).reshape_as(
                    hand_role_debug["object_posterior"]
                ).clamp(0.0, 1.0)
                if adaptive_role_enabled:
                    role_edit_tokens = (
                        hand_role_debug["object_posterior"]
                        >= hand_role_debug["posterior_threshold"]
                    ).reshape(batch_size, -1)
                else:
                    role_edit_tokens = (
                        hand_role_inference.token_edit_confidence
                        >= hand_posterior_threshold
                    )
                if causal_ownership_enabled:
                    owner_shape = hand_role_debug[
                        "object_posterior"
                    ].shape
                    owner_observation = (
                        hand_role_debug[
                            "object_posterior"
                        ].float()
                        * role_edit_tokens.reshape(
                            owner_shape
                        ).float()
                    ).reshape(batch_size, -1)
                    owner_hand = F.max_pool2d(
                        hand_persistent_mask[
                            :, role_left:role_right
                        ].float().reshape(
                            batch_size * current_num_frames,
                            1,
                            height,
                            width,
                        ),
                        kernel_size=2,
                        stride=2,
                    ).reshape(batch_size, -1) > 0.0
                    if oracle_source_owner_mask is not None:
                        current_causal_ownership = (
                            build_oracle_causal_ownership(
                                source_owner_mask=(
                                    oracle_source_owner_mask[
                                        :, role_left:role_right
                                    ]
                                ),
                                hand_mask=hand_only_mask[
                                    :, role_left:role_right
                                ],
                                spatial_shape=owner_shape[-2:],
                                hand_already_excluded=(
                                    source_owner_prepool_hand_exclusion
                                ),
                            )
                        )
                        owner_source = "oracle_clean_source"
                    else:
                        current_causal_ownership = causal_ownership_tracker(
                            source_features=source_query_features,
                            observation_weight=owner_observation,
                            source_semantic=src_fg_mask_soft,
                            hand_mask=owner_hand,
                            hand_proximity=hand_role_debug[
                                "hand_proximity"
                            ].reshape(batch_size, -1),
                            tokens_per_frame=self.frame_seq_length,
                            detector_visible=hand_role_debug[
                                "object_visible"
                            ],
                            spatial_shape=owner_shape[-2:],
                            **(
                                {
                                    "frame_indices": global_frame_indices[
                                        role_left:role_right
                                    ]
                                }
                                if motion_geometry_owner
                                else {}
                            ),
                            # In the automatic transactional path the source
                            # track is committed only after velocity evidence
                            # verifies the conservative write core.
                            # Motion geometry is the exception: its complete
                            # flow-transported state advances here and is
                            # intentionally independent of the later, sparse
                            # appearance-memory write transaction.
                            update_state=(
                                motion_geometry_owner
                                or not isinstance(
                                    native_owner_tracker,
                                    AutomaticTransactionalOwnerTracker,
                                )
                            ),
                        )
                        owner_source = (
                            "hand_conditioned_raft_geometry"
                            if motion_geometry_owner
                            else "tracked_source_features"
                        )
                    if source_flow_role_fusion:
                        if not motion_geometry_owner:
                            raise RuntimeError(
                                "Source-flow role fusion lost motion owner"
                            )
                        flow_role_evidence = build_flow_role_evidence(
                            current_causal_ownership,
                            shape=owner_shape,
                            hand_exclusion=hand_role_debug[
                                "hand_hard_exclusion"
                            ],
                        )
                        hand_role_inference = (
                            hand_role_inferencer.refine_with_source_flow(
                                hand_role_inference,
                                flow_role_evidence,
                                hand_occupancy=hand_occupancy_mask[
                                    :, role_left:role_right
                                ],
                                flow_weight=source_flow_role_weight,
                            )
                        )
                        if source_flow_verified_region:
                            hand_role_inference = (
                                hand_role_inferencer
                                .apply_source_flow_verified_region(
                                    hand_role_inference,
                                    flow_role_evidence,
                                    owner_support=(
                                        current_causal_ownership
                                        .owner_support
                                    ),
                                    hand_exclusion=hand_role_inference.debug[
                                        "hand_hard_exclusion"
                                    ],
                                    hand_occupancy=hand_occupancy_mask[
                                        :, role_left:role_right
                                    ],
                                    owner_radius=(
                                        source_flow_verified_owner_radius
                                    ),
                                    background_veto_threshold=(
                                        source_flow_background_veto_threshold
                                    ),
                                    background_veto_min_confidence=(
                                        source_flow_background_veto_min_confidence
                                    ),
                                )
                            )
                        current_roles = hand_role_inference.roles
                        hand_role_debug = hand_role_inference.debug
                        role_edit_tokens = (
                            hand_role_debug["object_posterior"]
                            >= hand_role_debug["posterior_threshold"]
                        ).reshape(batch_size, -1)
                        if not source_flow_verified_region:
                            role_edit_tokens = (
                                role_edit_tokens
                                | current_causal_ownership.owner_support
                            )
                        print(
                            "SOURCE_FLOW_ROLE_FUSION "
                            f"block={current_start_frame // self.num_frame_per_block} "
                            "external_object_mask=disabled "
                            "flow=clean_source_rgb_camera_compensated "
                            f"object={flow_role_evidence.object_likelihood.mean().item():.4f} "
                            f"background={flow_role_evidence.background_likelihood.mean().item():.4f} "
                            f"boundary={flow_role_evidence.boundary_likelihood.mean().item():.4f} "
                            f"unknown={flow_role_evidence.unknown_likelihood.mean().item():.4f} "
                            f"recovered={hand_role_debug['source_flow_recovered_support'].mean().item():.4f}"
                        )
                        if source_flow_verified_region:
                            print(
                                "SOURCE_FLOW_VERIFIED_REGION "
                                f"block={current_start_frame // self.num_frame_per_block} "
                                "external_object_mask=disabled "
                                f"radius={source_flow_verified_owner_radius} "
                                "proposal="
                                f"{hand_role_debug['source_flow_region_proposal'].mean().item():.4f} "
                                "verified="
                                f"{hand_role_debug['source_flow_verified_support'].mean().item():.4f} "
                                "background_veto="
                                f"{hand_role_debug['source_flow_region_background_veto'].mean().item():.4f} "
                                "hand_exclusion="
                                f"{hand_role_debug['source_flow_region_hand_exclusion'].mean().item():.4f}"
                            )
                    if native_history_transactional_owner:
                        if native_owner_tracker is None:
                            raise RuntimeError(
                                "Transactional native owner is missing "
                                "its tracker"
                            )
                        if isinstance(
                            native_owner_tracker,
                            AutomaticTransactionalOwnerTracker,
                        ):
                            owner_role_maps = F.adaptive_avg_pool2d(
                                torch.stack([
                                    current_roles.object,
                                    current_roles.boundary,
                                ], dim=2).flatten(0, 1),
                                output_size=owner_shape[-2:],
                            ).reshape(
                                batch_size, current_num_frames, 2,
                                *owner_shape[-2:],
                            )
                            current_transactional_owner = (
                                native_owner_tracker(
                                    ownership=current_causal_ownership,
                                    object_posterior=hand_role_debug[
                                        "object_posterior"
                                    ],
                                    posterior_threshold=hand_role_debug[
                                        "posterior_threshold"
                                    ],
                                    source_attention=hand_role_debug[
                                        "source_attention"
                                    ],
                                    hand_probability=hand_role_debug[
                                        "hand_hard_exclusion"
                                    ],
                                    hand_proximity=hand_role_debug[
                                        "hand_proximity"
                                    ],
                                    object_role=owner_role_maps[:, :, 0],
                                    boundary_role=owner_role_maps[:, :, 1],
                                    # Preview the transaction before the
                                    # first velocity observation.  Lifecycle
                                    # advances exactly once after step zero.
                                    update_state=False,
                                )
                            )
                            transaction_source = (
                                "hand_source_prior_pending_flow"
                            )
                        else:
                            if oracle_source_owner_full_mask is None:
                                raise RuntimeError(
                                    "Oracle transactional owner requires "
                                    "a complete source owner mask"
                                )
                            full_owner_map = (
                                build_oracle_source_owner_weight(
                                    source_owner_mask=(
                                        oracle_source_owner_full_mask[
                                            :, role_left:role_right
                                        ]
                                    ),
                                    hand_mask=hand_only_mask[
                                        :, role_left:role_right
                                    ],
                                    spatial_shape=owner_shape[-2:],
                                    hand_already_excluded=True,
                                )
                            )
                            current_transactional_owner = (
                                native_owner_tracker(
                                    source_features=source_query_features,
                                    full_owner_weight=(
                                        full_owner_map.reshape(
                                            batch_size, -1
                                        )
                                    ),
                                    core_owner_weight=(
                                        current_causal_ownership.owner_weight
                                    ),
                                    source_semantic=src_fg_mask_soft,
                                    hand_proximity=hand_role_debug[
                                        "hand_proximity"
                                    ].reshape(batch_size, -1),
                                    tokens_per_frame=self.frame_seq_length,
                                    spatial_shape=owner_shape[-2:],
                                )
                            )
                            transaction_source = "oracle_complete_matte"
                        hand_role_debug.update(
                            current_transactional_owner.as_debug_maps(
                                owner_shape
                            )
                        )
                        print(
                            "TRANSACTIONAL_NATIVE_OWNER "
                            f"block={current_start_frame // self.num_frame_per_block} "
                            f"source={transaction_source} "
                            "read="
                            f"{current_transactional_owner.read_weight.mean().item():.4f} "
                            "write="
                            f"{current_transactional_owner.write_weight.mean().item():.4f} "
                            "contact="
                            f"{current_transactional_owner.contact_weight.mean().item():.4f} "
                            "lifecycle="
                            f"{current_transactional_owner.lifecycle_weight.mean().item():.4f} "
                            "max_missing="
                            f"{int(current_transactional_owner.missing_observation_frames.max().item())}"
                        )
                    hand_role_debug.update(
                        current_causal_ownership.as_debug_maps(
                            owner_shape
                        )
                    )
                    # The same persistent owner must control text grounding,
                    # attention provenance, and velocity routing.  Keeping
                    # the old detector-only edit mask here would create
                    # contradictory experts on detector-dropout frames.
                    if not source_flow_verified_region:
                        role_edit_tokens = (
                            role_edit_tokens
                            | current_causal_ownership.owner_support
                        )
                    print(
                        "CAUSAL_OWNER_FLOW "
                        f"block={current_start_frame // self.num_frame_per_block} "
                        f"observation={current_causal_ownership.observation_weight.mean().item():.4f} "
                        f"transport={current_causal_ownership.transported_weight.mean().item():.4f} "
                        f"owner={current_causal_ownership.owner_weight.mean().item():.4f} "
                        f"support={current_causal_ownership.owner_support.float().mean().item():.4f} "
                        f"source={owner_source} "
                        "visible="
                        f"{(current_causal_ownership.state_code == 1).float().mean().item():.4f} "
                        "occluded="
                        f"{(current_causal_ownership.state_code == 2).float().mean().item():.4f} "
                        "absent="
                        f"{(current_causal_ownership.state_code == 3).float().mean().item():.4f} "
                        f"missing={int(current_causal_ownership.missing_frames.max().item())}"
                    )
                if target_semantic_competition:
                    if (
                        current_causal_ownership is None
                        or target_semantic_attention is None
                    ):
                        raise RuntimeError(
                            "Target semantic competition requires clean-"
                            "source target-text maps and causal ownership"
                        )
                    semantic_shape = hand_role_debug[
                        "object_posterior"
                    ].shape
                    # High-recall automatic support is the candidate prior.
                    # Causal owner transport remains separately available for
                    # whole-object geometry, while semantic competition can
                    # recover a cap token that the compact owner temporarily
                    # misses during a pose change.
                    semantic_candidate = role_edit_tokens.reshape(
                        semantic_shape
                    ).float() * (
                        1.0
                        - hand_role_debug[
                            "hand_hard_exclusion"
                        ].float().clamp(0.0, 1.0)
                    )
                    edit_attention = {
                        name: value.reshape(semantic_shape)
                        for name, value in target_semantic_attention.items()
                        if name.startswith("edit_")
                    }
                    preserve_attention = {
                        name: value.reshape(semantic_shape)
                        for name, value in target_semantic_attention.items()
                        if name.startswith("preserve_")
                    }
                    current_edit_authority = (
                        build_semantic_edit_authority(
                            edit_attention=edit_attention,
                            preserve_attention=preserve_attention,
                            owner_weight=(
                                semantic_candidate
                            ),
                            margin=target_semantic_margin,
                            min_confidence=(
                                target_semantic_min_confidence
                            ),
                        )
                    )
                    hand_role_debug.update(
                        current_edit_authority.as_debug_maps()
                    )
                    # The normalized score remains available for diagnosis,
                    # but Bayes provenance is categorical once the semantic
                    # competition has accepted a token.  Reusing the soft
                    # score as target-memory mass would mix source values
                    # back into an admitted cap token and weaken the edit a
                    # second time.
                    hand_role_debug["edit_authority"] = (
                        current_edit_authority.support.float()
                    )
                    current_target_authority_tokens = (
                        current_edit_authority.authority.reshape(
                            batch_size, -1
                        )
                    )
                    current_target_authority_support_tokens = (
                        current_edit_authority.support.reshape(
                            batch_size, -1
                        )
                    )
                    role_edit_tokens = (
                        current_target_authority_support_tokens
                    )
                    authority_owner = (
                        current_edit_authority.owner_weight > 0.0
                    )
                    authority_count = authority_owner.float().sum().clamp_min(
                        1.0
                    )
                    print(
                        "TARGET_SEMANTIC_AUTHORITY "
                        f"block={current_start_frame // self.num_frame_per_block} "
                        "probe=clean_source_target_text "
                        "external_object_mask=disabled "
                        f"edit={current_edit_authority.edit_likelihood.mean().item():.4f} "
                        f"preserve={current_edit_authority.preserve_likelihood.mean().item():.4f} "
                        f"owner={current_edit_authority.owner_weight.mean().item():.4f} "
                        f"authority={current_edit_authority.authority.mean().item():.4f} "
                        "authority_on_owner="
                        f"{(current_edit_authority.authority * authority_owner.float()).sum().div(authority_count).item():.4f} "
                        f"support={current_edit_authority.support.float().mean().item():.4f}"
                    )
                if current_causal_ownership is not None:
                    current_memory_query_weight = (
                        current_transactional_owner.read_weight.float()
                        if current_transactional_owner is not None
                        else current_causal_ownership.owner_support.float()
                    )
                    if (
                        native_history_motion_owner_dense_read
                        and current_transactional_owner is not None
                    ):
                        if native_history_topology_complete_read:
                            (
                                current_memory_query_weight,
                                motion_read_recovery,
                                topology_read_recovery,
                                topology_holes,
                            ) = build_topology_complete_motion_owner_read_weight(
                                current_causal_ownership,
                                current_transactional_owner,
                                shape=hand_role_debug[
                                    "object_posterior"
                                ].shape,
                                hand_exclusion=hand_role_debug[
                                    "hand_hard_exclusion"
                                ],
                            )
                            hand_role_debug[
                                "native_owner_topology_holes"
                            ] = topology_holes.reshape_as(
                                hand_role_debug["object_posterior"]
                            ).float()
                            hand_role_debug[
                                "native_owner_topology_read_recovery"
                            ] = topology_read_recovery.reshape_as(
                                hand_role_debug["object_posterior"]
                            )
                        else:
                            (
                                current_memory_query_weight,
                                motion_read_recovery,
                            ) = build_motion_owner_read_weight(
                                current_causal_ownership,
                                current_transactional_owner,
                            )
                        hand_role_debug[
                            "native_owner_motion_read_recovery"
                        ] = motion_read_recovery.reshape_as(
                            hand_role_debug["object_posterior"]
                        )
                        hand_role_debug[
                            "native_owner_effective_read"
                        ] = current_memory_query_weight.reshape_as(
                            hand_role_debug["object_posterior"]
                        )
                        print(
                            "MOTION_OWNER_DENSE_READ "
                            f"block={current_start_frame // self.num_frame_per_block} "
                            "scope=all_latent_frames "
                            "admission=source_address_match "
                            f"base={current_transactional_owner.read_weight.mean().item():.4f} "
                            f"recovered={motion_read_recovery.mean().item():.4f} "
                            + (
                                "topology_holes="
                                f"{topology_holes.float().mean().item():.4f} "
                                "topology_recovered="
                                f"{topology_read_recovery.mean().item():.4f} "
                                if native_history_topology_complete_read
                                else ""
                            )
                            +
                            f"effective={current_memory_query_weight.mean().item():.4f} "
                            "write=unchanged"
                        )
                    if (
                        current_target_authority_support_tokens is not None
                    ):
                        # Semantic confidence decides whether the query is
                        # allowed, while the transactional owner retains the
                        # strength of an admitted read.  Capping the read by
                        # the raw semantic score made the downstream 0.5 KV
                        # threshold reject otherwise valid cap queries.
                        current_memory_query_weight = (
                            apply_semantic_transaction_gate(
                                current_memory_query_weight,
                                current_target_authority_support_tokens,
                            )
                        )
                if factorized_bayes_enabled:
                    current_factorized_operators = (
                        factorized_operator_builder(
                            roles=current_roles,
                            evidence=hand_role_debug,
                            expected_token_length=(
                                role_edit_tokens.shape[1]
                            ),
                        )
                    )
                    hand_role_debug.update(
                        current_factorized_operators.as_debug_maps()
                    )
                role_coverage = {
                    name: value.mean().item()
                    for name, value in current_roles.as_dict().items()
                }
                adaptive_summary = ""
                if adaptive_role_enabled:
                    adaptive_summary = (
                        " radius="
                        f"{hand_role_debug['adaptive_hand_radius'].mean().item():.3f}"
                        " attention_reliability="
                        f"{hand_role_debug['adaptive_attention_reliability'].mean().item():.3f}"
                        " threshold="
                        f"{hand_role_debug['posterior_threshold'].mean().item():.3f}"
                        " budget="
                        f"{hand_role_debug['adaptive_coverage_budget'].mean().item():.3f}"
                    )
                print(
                    "HAND_ROLE_FLOW "
                    f"block={current_start_frame // self.num_frame_per_block} "
                    f"edit_tokens={role_edit_tokens.float().mean().item():.4f} "
                    f"visible={hand_role_debug['object_visible'].mean().item():.4f} "
                    f"temporal={hand_role_debug['temporal_posterior'].mean().item():.4f} "
                    + (
                        "adaptive=1 "
                        if adaptive_role_enabled
                        else "adaptive=0 "
                    )
                    + " ".join(
                        f"{name}={coverage:.4f}"
                        for name, coverage in role_coverage.items()
                    )
                    + adaptive_summary
                )
                if current_factorized_operators is not None:
                    print(
                        "FACTORIZED_OPERATOR "
                        f"block={current_start_frame // self.num_frame_per_block} "
                        "source_key="
                        f"{current_factorized_operators.source_key_action.mean().item():.4f} "
                        "source_value="
                        f"{current_factorized_operators.source_value_action.mean().item():.4f} "
                        "source_residual="
                        f"{current_factorized_operators.source_residual_action.mean().item():.4f} "
                        "target_memory="
                        f"{current_factorized_operators.target_memory_action.mean().item():.4f} "
                        "unknown="
                        f"{current_factorized_operators.unknown_action.mean().item():.4f}"
                    )
                    if current_causal_ownership is not None:
                        owner_support = (
                            current_causal_ownership.owner_support
                        )
                        owner_count = owner_support.float().sum().clamp_min(1)

                        def owner_mean(value):
                            return (
                                value.float()[owner_support].sum()
                                / owner_count
                            ).item()

                        print(
                            "CAUSAL_OWNER_OPERATOR "
                            f"block={current_start_frame // self.num_frame_per_block} "
                            f"source_key={owner_mean(current_factorized_operators.source_key_action):.4f} "
                            f"source_value={owner_mean(current_factorized_operators.source_value_action):.4f} "
                            f"source_residual={owner_mean(current_factorized_operators.source_residual_action):.4f} "
                            f"target_memory={owner_mean(current_factorized_operators.target_memory_action):.4f} "
                            f"unknown={owner_mean(current_factorized_operators.unknown_action):.4f}"
                        )
                if save_role_dir is not None:
                    block_index = (
                        current_start_frame // self.num_frame_per_block
                    )
                    self._save_role_state(
                        save_role_dir,
                        block_index,
                        current_roles,
                    )
                    self._save_hand_role_debug(
                        save_role_dir,
                        block_index,
                        hand_role_debug,
                    )
                if belief_memory_enabled:
                    current_control_belief = control_belief_builder(
                        debug=hand_role_debug,
                        hand_mask=hand_occupancy_mask[
                            :, role_left:role_right
                        ],
                    )
                    current_belief_kv_weights = build_belief_kv_weights(
                        current_control_belief,
                        expected_token_length=role_edit_tokens.shape[1],
                    )
            current_identity_read_mask = None
            current_identity_write_mask = None
            current_identity_lifecycle = None
            if (
                factorized_target_identity
                or factorized_immutable_target_memory
                or (
                    appearance_leakage_decomposition
                    and target_identity_enabled
                )
            ):
                if hand_role_debug is None or role_edit_tokens is None:
                    raise RuntimeError(
                        "Factorized identity requires current object core"
                    )
                if factorized_immutable_target_memory:
                    if current_causal_ownership is None:
                        raise RuntimeError(
                            "Immutable factorized memory requires a "
                            "current causal owner"
                        )
                    current_identity_read_mask = (
                        current_causal_ownership.owner_support.float()
                        if immutable_target_hard_owner
                        else current_causal_ownership.owner_weight
                    )
                    current_identity_write_mask = (
                        current_causal_ownership.owner_support
                    )
                    hand_role_debug.update({
                        "identity_owner_read": (
                            current_identity_read_mask.reshape_as(
                                hand_role_debug["object_posterior"]
                            )
                        ),
                        "identity_verified_write_core": (
                            current_identity_write_mask.reshape_as(
                                hand_role_debug["object_posterior"]
                            ).float()
                        ),
                    })
                    print(
                        "FACTORIZED_IMMUTABLE_OWNER "
                        f"block={current_start_frame // self.num_frame_per_block} "
                        f"read={current_identity_read_mask.mean().item():.4f} "
                        f"write={current_identity_write_mask.float().mean().item():.4f}"
                    )
                elif source_coordinate_identity:
                    if (
                        oracle_source_owner_mask is None
                        and identity_owner_tracker is None
                    ):
                        raise RuntimeError(
                            "Missing source-coordinate identity owner "
                            "tracker"
                        )
                    owner_shape = hand_role_debug[
                        "object_posterior"
                    ].shape
                    owner_observation = (
                        hand_role_debug[
                            "object_posterior"
                        ].float()
                        * role_edit_tokens.reshape(
                            owner_shape
                        ).float()
                    ).reshape(batch_size, -1)
                    owner_hand = F.max_pool2d(
                        hand_persistent_mask[
                            :, role_left:role_right
                        ].float().reshape(
                            batch_size * current_num_frames,
                            1,
                            height,
                            width,
                        ),
                        kernel_size=2,
                        stride=2,
                    ).reshape(batch_size, -1) > 0.0
                    if oracle_source_owner_mask is not None:
                        oracle_owner = build_oracle_source_owner_weight(
                            source_owner_mask=oracle_source_owner_mask[
                                :, role_left:role_right
                            ],
                            hand_mask=hand_only_mask[
                                :, role_left:role_right
                            ],
                            spatial_shape=owner_shape[-2:],
                            hand_already_excluded=(
                                source_owner_prepool_hand_exclusion
                            ),
                        )
                        current_identity_read_mask = oracle_owner.reshape(
                            batch_size, -1
                        )
                        current_identity_write_mask = (
                            current_identity_read_mask.clone()
                        )
                        hand_role_debug.update({
                            "identity_oracle_source_owner": oracle_owner,
                            "identity_owner_read": oracle_owner,
                            "identity_owner_transport": oracle_owner,
                            "identity_owner_observation": oracle_owner,
                            "identity_owner_similarity": oracle_owner,
                            "identity_owner_confidence": oracle_owner,
                            "identity_owner_semantic": oracle_owner,
                        })
                        owner_transport_mean = (
                            current_identity_read_mask.mean().item()
                        )
                        owner_observation_mean = owner_transport_mean
                        owner_source = "oracle_clean_source"
                    else:
                        current_identity_ownership = (
                            identity_owner_tracker(
                                source_features=source_query_features,
                                observation_weight=owner_observation,
                                source_semantic=src_fg_mask_soft,
                                hand_mask=owner_hand,
                                tokens_per_frame=self.frame_seq_length,
                                frame_visible=hand_role_debug[
                                    "object_visible"
                                ],
                                spatial_shape=owner_shape[-2:],
                            )
                        )
                        current_identity_read_mask = (
                            current_identity_ownership.read_weight
                        )
                        current_identity_write_mask = (
                            role_edit_tokens.clone().float()
                        )
                        hand_role_debug.update(
                            current_identity_ownership.as_debug_maps(
                                owner_shape
                            )
                        )
                        owner_transport_mean = (
                            current_identity_ownership.transported_weight
                            .mean().item()
                        )
                        owner_observation_mean = (
                            current_identity_ownership.observation_weight
                            .mean().item()
                        )
                        owner_source = "tracked_source_features"
                    if source_identity_residual_carry:
                        if identity_residual_carry is None:
                            raise RuntimeError(
                                "Missing source-coordinate residual carry"
                            )
                        carry_had_state = identity_residual_carry.has_state()
                        carry_result = identity_residual_carry.prepare(
                            source_features=source_query_features,
                            owner_weight=current_identity_read_mask,
                            source_latent=src_input,
                            tokens_per_frame=self.frame_seq_length,
                            spatial_shape=owner_shape[-2:],
                        )
                        carried_identity_residual = (
                            carry_result.target_residual
                        )
                        carried_identity_support = carry_result.support
                        denoised_pred = (
                            src_input
                            + identity_residual_carry_strength
                            * carry_result.residual
                        )
                        hand_role_debug[
                            "identity_residual_carry_support"
                        ] = carry_result.support.reshape(owner_shape)
                        carry_energy = carry_result.residual.float().square(
                        ).mean(dim=2).sqrt()
                        hand_role_debug[
                            "identity_residual_carry_energy"
                        ] = F.avg_pool2d(
                            carry_energy.reshape(
                                batch_size * current_num_frames,
                                1,
                                height,
                                width,
                            ),
                            kernel_size=2,
                            stride=2,
                        ).reshape(owner_shape)
                        print(
                            "SOURCE_IDENTITY_RESIDUAL_CARRY "
                            f"block="
                            f"{current_start_frame // self.num_frame_per_block} "
                            f"support={carry_result.support.mean().item():.4f} "
                            f"energy={carry_result.residual.float().square().mean().sqrt().item():.4f} "
                            f"strength={identity_residual_carry_strength:.3f}"
                        )
                    else:
                        carry_had_state = False
                        carried_identity_residual = None
                        carried_identity_support = None
                    print(
                        "PRESTEP_IDENTITY_OWNER "
                        f"block="
                        f"{current_start_frame // self.num_frame_per_block} "
                        f"read="
                        f"{current_identity_read_mask.mean().item():.4f} "
                        f"transport="
                        f"{owner_transport_mean:.4f} "
                        f"observation="
                        f"{owner_observation_mean:.4f} "
                        f"peak="
                        f"{current_identity_read_mask.max().item():.4f} "
                        f"source={owner_source}"
                    )
                else:
                    current_identity_read_mask = (
                        torch.zeros_like(
                            role_edit_tokens, dtype=torch.bool
                        )
                        if appearance_leakage_decomposition
                        else role_edit_tokens.clone().bool()
                    )
                    current_identity_write_mask = (
                        current_identity_read_mask
                    )
                if (
                    identity_visibility_lifecycle
                    and not appearance_leakage_decomposition
                ):
                    current_identity_lifecycle = (
                        target_identity_memory.update_visibility_lifecycle(
                            object_core=current_identity_read_mask,
                            frame_visible=hand_role_debug[
                                "object_visible"
                            ],
                            tokens_per_frame=self.frame_seq_length,
                            max_occluded_blocks=(
                                identity_max_occluded_blocks
                            ),
                        )
                    )
                    current_identity_read_mask = (
                        current_identity_lifecycle.read_mask
                    )
                    lifecycle_shape = hand_role_debug[
                        "object_posterior"
                    ].shape
                    hand_role_debug.update({
                        "identity_lifecycle_read_mask": (
                            current_identity_read_mask.reshape(
                                lifecycle_shape
                            ).float()
                        ),
                        "identity_lifecycle_state": (
                            current_identity_lifecycle.state_code[
                                :, None, None, None
                            ].float().expand(lifecycle_shape)
                        ),
                        "identity_lifecycle_missing_blocks": (
                            current_identity_lifecycle.missing_blocks[
                                :, None, None, None
                            ].float().expand(lifecycle_shape)
                        ),
                    })
                    print(
                        "IDENTITY_LIFECYCLE "
                        f"block="
                        f"{current_start_frame // self.num_frame_per_block} "
                        f"state="
                        f"{int(current_identity_lifecycle.state_code.max().item())} "
                        f"visible="
                        f"{current_identity_lifecycle.visible.float().mean().item():.4f} "
                        f"read_coverage="
                        f"{current_identity_read_mask.float().mean().item():.4f}"
                    )
            current_paired_memory_coordinate = None
            current_paired_memory_request = None
            current_paired_memory_proposal = None
            current_paired_memory_support = None
            current_paired_memory_interior = None
            if causal_paired_edit_memory:
                if (
                    paired_edit_memory is None
                    or current_causal_ownership is None
                    or current_factorized_operators is None
                    or hand_role_debug is None
                ):
                    raise RuntimeError(
                        "Paired edit memory requires causal ownership and "
                        "factorized operators"
                    )
                owner_weight = (
                    current_causal_ownership.owner_weight.float()
                )
                if paired_memory_interior_projection:
                    gate_builder = (
                        build_owner_attached_structure_gate
                        if paired_memory_owner_attached_boundary
                        else build_object_interior_gate
                    )
                    current_paired_memory_interior = gate_builder(
                        owner_weight,
                        # Use pre-Bayes roles so a target-owned posterior
                        # cannot relabel a hand/contact token as safe.
                        object_role=current_roles.object,
                        boundary_role=current_roles.boundary,
                        hand_role=current_roles.hand,
                        tokens_per_frame=self.frame_seq_length,
                        spatial_shape=hand_role_debug[
                            "object_posterior"
                        ].shape[-2:],
                    )
                else:
                    current_paired_memory_interior = torch.ones_like(
                        owner_weight
                    )
                current_paired_memory_coordinate = (
                    build_object_coordinates(
                        owner_weight,
                        tokens_per_frame=self.frame_seq_length,
                        spatial_shape=hand_role_debug[
                            "object_posterior"
                        ].shape[-2:],
                    )
                )
                # Role-specific read: source-addressed appearance is allowed
                # only where both causal ownership and the Bayes target-memory
                # expert agree.  Hand/background/unknown are exact abstentions.
                current_paired_memory_request = (
                    owner_weight
                    * current_factorized_operators.target_memory_action.float()
                    * current_paired_memory_interior
                ).clamp(0.0, 1.0)
                # The proposal contains only role/ownership uncertainty.
                # Layer-wise source matching and residual agreement are
                # independent checks applied atomically inside commit().
                has_canonical_state = paired_edit_memory.has_state()
                current_paired_memory_proposal = (
                    current_paired_memory_request
                )
                paired_reads = paired_edit_memory.read(
                    source_kv_cache=kv_cache_src,
                    current_coordinate=current_paired_memory_coordinate,
                    current_object_request=current_paired_memory_request,
                    current_transport_owner=owner_weight,
                )
                shared_dict_dual["causal_paired_edit_memory"] = (
                    paired_reads
                )
                shared_dict_dual["paired_memory_read_strength"] = float(
                    paired_memory_read_strength
                )
                shared_dict_dual[
                    "paired_memory_value_projection"
                ] = bool(paired_memory_value_projection)
                shared_dict_dual[
                    "paired_memory_query_gated_projection"
                ] = bool(paired_memory_query_gated_projection)
                shared_dict_dual[
                    "paired_memory_single_confidence"
                ] = bool(paired_memory_single_confidence)
                shared_dict_dual[
                    "paired_memory_dual_timescale_anchor"
                ] = bool(paired_memory_dual_timescale_anchor)
                shared_dict_dual[
                    "paired_memory_canonical_key_anchor"
                ] = bool(paired_memory_canonical_key_anchor)
                canonical_anchor_requests = (
                    paired_edit_memory.build_canonical_anchor_requests(
                        paired_reads,
                        current_paired_memory_coordinate,
                    )
                    if (
                        paired_memory_canonical_key_anchor
                        and paired_edit_memory.ignition_is_verified()
                    )
                    else {}
                )
                shared_dict_dual[
                    "paired_memory_canonical_anchors"
                ] = canonical_anchor_requests
                read_supports = [
                    read.support.float() for read in paired_reads.values()
                ]
                read_similarities = [
                    read.best_similarity.float()
                    for read in paired_reads.values()
                ]
                read_consensuses = [
                    (
                        read.residual_consensus.float()
                        if read.residual_consensus is not None
                        else torch.ones_like(read.support).float()
                    )
                    for read in paired_reads.values()
                ]
                read_part_similarities = [
                    (
                        read.part_similarity.float()
                        if read.part_similarity is not None
                        else torch.ones_like(read.support).float()
                    )
                    for read in paired_reads.values()
                ]
                read_part_confidences = [
                    (
                        read.part_confidence.float()
                        if read.part_confidence is not None
                        else torch.ones_like(read.support).float()
                    )
                    for read in paired_reads.values()
                ]
                read_canonical_supports = [
                    (
                        read.canonical_support.float()
                        if read.canonical_support is not None
                        else read.support.float()
                    )
                    for read in paired_reads.values()
                ]
                read_transport_supports = [
                    (
                        read.transported_support.float()
                        if read.transported_support is not None
                        else torch.zeros_like(read.support).float()
                    )
                    for read in paired_reads.values()
                ]
                read_transport_cycles = [
                    (
                        read.transport_cycle_confidence.float()
                        if read.transport_cycle_confidence is not None
                        else torch.zeros_like(read.support).float()
                    )
                    for read in paired_reads.values()
                ]
                paired_shape = hand_role_debug[
                    "object_posterior"
                ].shape
                mean_read_support = torch.stack(read_supports).mean(0)
                mean_read_similarity = torch.stack(
                    read_similarities
                ).mean(0)
                mean_read_consensus = torch.stack(
                    read_consensuses
                ).mean(0)
                mean_read_part_similarity = torch.stack(
                    read_part_similarities
                ).mean(0)
                mean_read_part_confidence = torch.stack(
                    read_part_confidences
                ).mean(0)
                mean_read_canonical_support = torch.stack(
                    read_canonical_supports
                ).mean(0)
                mean_read_transport_support = torch.stack(
                    read_transport_supports
                ).mean(0)
                mean_read_transport_cycle = torch.stack(
                    read_transport_cycles
                ).mean(0)
                current_paired_memory_support = (
                    mean_read_support.reshape(paired_shape)
                )
                if paired_memory_canonical_key_anchor:
                    canonical_query_supports = [
                        request.query_support.float()
                        for request in canonical_anchor_requests.values()
                    ]
                    mean_canonical_query_support = (
                        torch.stack(canonical_query_supports).mean(0)
                        if canonical_query_supports
                        else torch.zeros_like(mean_read_support)
                    )
                else:
                    mean_canonical_query_support = torch.zeros_like(
                        mean_read_support
                    )
                hand_role_debug.update({
                    "paired_memory_read_request": (
                        current_paired_memory_request.reshape(
                            paired_shape
                        )
                    ),
                    "paired_memory_read_support": (
                        mean_read_support.reshape(paired_shape)
                    ),
                    "paired_memory_read_similarity": (
                        (mean_read_similarity + 1.0).mul(0.5)
                        .reshape(paired_shape)
                    ),
                    "paired_memory_read_consensus": (
                        mean_read_consensus.reshape(paired_shape)
                    ),
                    "paired_memory_read_part_similarity": (
                        (mean_read_part_similarity + 1.0).mul(0.5)
                        .reshape(paired_shape)
                    ),
                    "paired_memory_read_part_confidence": (
                        mean_read_part_confidence.reshape(paired_shape)
                    ),
                    "paired_memory_canonical_support": (
                        mean_read_canonical_support.reshape(paired_shape)
                    ),
                    "paired_memory_transport_support": (
                        mean_read_transport_support.reshape(paired_shape)
                    ),
                    "paired_memory_transport_cycle": (
                        mean_read_transport_cycle.reshape(paired_shape)
                    ),
                    "paired_memory_interior_gate": (
                        current_paired_memory_interior.reshape(
                            paired_shape
                        )
                    ),
                    "paired_memory_canonical_query_support": (
                        mean_canonical_query_support.reshape(paired_shape)
                    ),
                })
                if paired_memory_owner_attached_boundary:
                    hand_role_debug[
                        "paired_memory_structure_gate"
                    ] = current_paired_memory_interior.reshape(paired_shape)
                requested = current_paired_memory_request > 0.0
                requested_count = requested.float().sum().clamp_min(1.0)
                requested_mass = current_paired_memory_request.sum().clamp_min(
                    paired_edit_memory.eps
                )
                print(
                    "PAIRED_EDIT_READ "
                    f"block={current_start_frame // self.num_frame_per_block} "
                    f"memory={int(has_canonical_state)} "
                    f"request={current_paired_memory_request.mean().item():.4f} "
                    f"support={mean_read_support.mean().item():.4f} "
                    "requested_similarity="
                    f"{(mean_read_similarity * requested.float()).sum().div(requested_count).item():.4f} "
                    "requested_consensus="
                    f"{(mean_read_consensus * requested.float()).sum().div(requested_count).item():.4f} "
                    "requested_part_similarity="
                    f"{(mean_read_part_similarity * requested.float()).sum().div(requested_count).item():.4f} "
                    "requested_part_confidence="
                    f"{(mean_read_part_confidence * requested.float()).sum().div(requested_count).item():.4f} "
                    "part_consistency="
                    f"{int(paired_memory_source_part_consistency)} "
                    f"interior={current_paired_memory_interior.mean().item():.4f} "
                    f"coordinate_radius={paired_memory_coordinate_radius:.3f} "
                    "min_consensus="
                    f"{paired_memory_min_residual_consensus:.3f} "
                    f"strength={paired_memory_read_strength:.3f}"
                    f" projection={int(paired_memory_value_projection)}"
                    " query_gate="
                    f"{int(paired_memory_query_gated_projection)}"
                    " single_confidence="
                    f"{int(paired_memory_single_confidence)}"
                    " dual_timescale_anchor="
                    f"{int(paired_memory_dual_timescale_anchor)}"
                    " canonical_key_anchor="
                    f"{int(paired_memory_canonical_key_anchor)}"
                    " ignition_verified="
                    f"{int(paired_edit_memory.ignition_is_verified())}"
                    " structure_boundary="
                    f"{int(paired_memory_owner_attached_boundary)}"
                    " persistent_projection="
                    f"{int(not paired_memory_disable_persistent_projection)}"
                    " source_suppression="
                    f"{paired_memory_source_suppression:.3f}"
                )
                if paired_memory_source_transport:
                    print(
                        "COUNTERFACTUAL_KV_TRANSPORT "
                        f"block={current_start_frame // self.num_frame_per_block} "
                        "address=clean_source payload=canonical_lineage "
                        "target_role=critic_only "
                        "canonical_mass_ratio="
                        f"{mean_read_canonical_support.sum().div(requested_mass).item():.4f} "
                        "transport_mass_ratio="
                        f"{mean_read_transport_support.sum().div(requested_mass).item():.4f} "
                        "effective_mass_ratio="
                        f"{mean_read_support.sum().div(requested_mass).item():.4f} "
                        "cycle="
                        f"{(mean_read_transport_cycle * requested.float()).sum().div(requested_count).item():.4f}"
                    )
                if paired_memory_canonical_key_anchor:
                    ignition_states = paired_edit_memory.export_ignition()
                    valid_slot_counts = [
                        (state.evidence > paired_edit_memory.eps)
                        .float().sum(dim=-1).mean()
                        for state in ignition_states.values()
                    ]
                    eligible_edges = [
                        request.query_key_mask.float().sum(dim=-1)
                        for request in canonical_anchor_requests.values()
                    ]
                    admitted = mean_canonical_query_support > 0.0
                    admitted_count = admitted.float().sum().clamp_min(1.0)
                    mean_edges = (
                        torch.stack(eligible_edges).mean(0)[admitted]
                        .sum().div(admitted_count).item()
                        if eligible_edges else 0.0
                    )
                    print(
                        "IMMUTABLE_CANONICAL_KV "
                        f"block={current_start_frame // self.num_frame_per_block} "
                        "address=ignition_pre_rope_source_k "
                        "payload=immutable_target_minus_source_delta_v "
                        f"verified={int(paired_edit_memory.ignition_is_verified())} "
                        "slots="
                        f"{torch.stack(valid_slot_counts).mean().item() if valid_slot_counts else 0.0:.1f} "
                        "query_coverage="
                        f"{mean_canonical_query_support.mean().item():.4f} "
                        f"keys_per_query={mean_edges:.2f} "
                        "confidence=key_logit_prior query_gate=binary"
                    )
            shared_dict_dual.update({
                "contact_graph_mode": contact_graph_mode,
                "contact_graphs": contact_graphs,
                "contact_graph_strength": contact_graph_strength,
                "contact_graph_layer_start": contact_graph_layer_start,
                "contact_graph_layer_end": contact_graph_layer_end,
                "target_identity_memory": (
                    target_identity_memory.export()
                    if target_identity_enabled
                    else {}
                ),
                "source_identity_keys": source_identity_keys,
                "factorized_target_identity": bool(
                    factorized_target_identity
                ),
                "appearance_leakage_decomposition": bool(
                    appearance_leakage_decomposition
                ),
                "factorized_immutable_target_memory": bool(
                    factorized_immutable_target_memory
                ),
                "factorized_native_target_history": bool(
                    factorized_native_target_history
                ),
                "role_fixed_native_history": bool(
                    role_fixed_native_history
                ),
                "native_history_transactional_owner": bool(
                    native_history_transactional_owner
                ),
                "native_history_consistent_transaction": bool(
                    native_history_consistent_transaction
                ),
                "native_history_verified_attention_authority": bool(
                    native_history_verified_attention_authority
                ),
                "native_history_attention_authority_strength": float(
                    native_history_attention_authority_strength
                ),
                "native_history_attention_authority_start_layer": (
                    max(role_native_history_layers)
                    if role_fixed_native_history
                    else self.num_transformer_blocks
                ),
                "native_history_payload_invariant_lineage": bool(
                    native_history_payload_invariant_lineage
                ),
                # Keep this explicit in the per-block shared state.  The flag
                # is initially installed before the clean-source probe, but
                # recording it again here prevents a future shared-state
                # rebuild from silently restoring entry-frame-only reads.
                "native_history_motion_owner_dense_read": bool(
                    native_history_motion_owner_dense_read
                ),
                "native_history_dual_evidence_arbitration": bool(
                    native_history_dual_evidence_arbitration
                ),
                "native_history_token_atomic_payload": bool(
                    native_history_token_atomic_payload
                ),
                "native_history_persistent_residual_upsert": bool(
                    native_history_persistent_residual_upsert
                ),
                "native_history_last_trusted_appearance": bool(
                    native_history_last_trusted_appearance
                ),
                "native_history_flow_indexed_residual": bool(
                    native_history_flow_indexed_residual
                ),
                "native_history_multiframe_identity_sink": bool(
                    native_history_multiframe_identity_sink
                ),
                "native_history_multiframe_sink_topk_per_frame": int(
                    native_history_multiframe_sink_topk_per_frame
                ),
                "native_history_multiframe_sink_source_logit_bias": float(
                    native_history_multiframe_sink_source_logit_bias
                ),
                "native_history_multiframe_sink_strength": float(
                    native_history_multiframe_sink_strength
                ),
                "native_history_timestep_counterfactual_memory": bool(
                    native_history_timestep_counterfactual_memory
                ),
                "native_history_tccm_flow_radius": float(
                    native_history_tccm_flow_radius
                ),
                "native_history_tccm_strength": float(
                    native_history_tccm_strength
                ),
                "native_history_tccm_max_error_ratio": float(
                    native_history_tccm_max_error_ratio
                ),
                "native_history_topology_complete_read": bool(
                    native_history_topology_complete_read
                ),
                "native_history_min_payload_consistency": float(
                    native_history_min_payload_consistency
                ),
                "factorized_owner_source_block": bool(
                    factorized_owner_source_block
                ),
                "immutable_target_residual_subspace": bool(
                    factorized_immutable_target_memory
                    and immutable_target_value_mode == "subspace"
                ),
                "identity_correction_strength": float(
                    identity_correction_strength
                ),
                "identity_support_floor": float(
                    identity_support_floor
                    if (
                        source_coordinate_identity
                        or factorized_immutable_target_memory
                    )
                    else 0.0
                ),
                "target_owned_object_handoff": bool(
                    target_owned_object_handoff
                    and target_identity_memory.causal_edit_anchor_reset
                ),
            })
            effective_src_fg_mask = (
                role_edit_tokens
                if (
                    consistent_role_kv_enabled
                    or causal_ownership_enabled
                )
                else src_fg_mask_bin
            )
            current_target_owned_mask = (
                target_identity_memory.match_target_owned_tokens(
                    source_features=source_query_features,
                    candidate_mask=(
                        src_fg_mask_bin.bool()
                        | role_edit_tokens.bool()
                    ),
                    min_similarity=target_owned_min_similarity,
                )
                if shared_dict_dual[
                    "target_owned_object_handoff"
                ]
                else torch.zeros_like(
                    src_fg_mask_bin, dtype=torch.bool
                )
                if target_owned_object_handoff
                else None
            )
            self._inject_masks_to_kv_cache(
                kv_cache_dual,
                trg_fg_mask_cache,
                effective_src_fg_mask,
                belief_kv_weight_cache=(
                    belief_kv_weight_cache
                    if belief_memory_enabled
                    else None
                ),
                factorized_operator_cache=(
                    factorized_operator_cache
                    if factorized_bayes_enabled
                    else None
                ),
                current_factorized_operators=(
                    current_factorized_operators
                ),
                target_owned_mask_cache=target_owned_mask_cache,
                current_target_owned_mask=current_target_owned_mask,
                current_identity_read_mask=current_identity_read_mask,
                current_causal_owner_mask=(
                    current_memory_query_weight
                ),
            )
            src_fg_mask_map = self._mask_reshape(
                effective_src_fg_mask,
                size=(current_num_frames, height, width),
            )
            inloop_trg_fg_mask = effective_src_fg_mask
            
            # Step 3.1: Spatial denoising loop
            noisy_pred_input = None
            identity_trace_steps = {
                0,
                min(1, len(denoising_step_list) - 1),
                min(7, len(denoising_step_list) - 1),
                len(denoising_step_list) - 1,
            }
            for index, current_timestep in tqdm(enumerate(denoising_step_list), total=len(denoising_step_list), leave=False):

                # set current timestep
                timestep = torch.ones(
                    [batch_size * 2, current_num_frames],
                    device=src_video.device,
                    dtype=torch.float32
                ) * current_timestep
                timestep_next = denoising_step_list[index + 1] / 1000 if (index < len(denoising_step_list) - 1) else 0
                shared_dict_dual['current_timestep_next'] = float(timestep_next)
                shared_dict_dual['current_timestep'] = float(current_timestep / 1000)
                shared_dict_dual['current_timestep_index'] = index
                shared_dict_dual['total_timestep'] = len(denoising_step_list)
                shared_dict_dual['blend_power'] = blend_power
                if target_identity_enabled:
                    shared_dict_dual["target_identity_support"] = {}
                    shared_dict_dual[
                        "target_identity_diagnostics"
                    ] = {}
                if factorized_native_target_history:
                    shared_dict_dual[
                        "native_target_history_diagnostics"
                    ] = {}
                if role_fixed_native_history:
                    shared_dict_dual[
                        "role_native_history_diagnostics"
                    ] = {}
                    shared_dict_dual[
                        "role_native_history_admissions"
                    ] = {}
                if native_history_verified_attention_authority:
                    # The consensus is local to this model evaluation. Never
                    # let read evidence leak across denoising timesteps.
                    shared_dict_dual[
                        "verified_attention_authority_state"
                    ] = {}
                    shared_dict_dual[
                        "verified_attention_authority_diagnostics"
                    ] = {}
                if causal_paired_edit_memory:
                    shared_dict_dual[
                        "paired_edit_memory_diagnostics"
                    ] = {}

                # use previous statistics on noise
                fwd_noise = torch.randn_like(src_input)
                fwd_noise = self._reuse_noise_statistics(fwd_noise, index, fg_mask=src_fg_mask_map)
                fwd_trg_noise = fwd_noise

                # update mask with trg mask at t^inj=0.5
                if index == len(denoising_step_list) // 2:
                    self._register_crossattn_mask_gatherer(crossattn_cache_dual, tok_src + tok_trg, layers=mask_layers, fg_scale=fg_scale)

                if fg_boost_factor != 1.0:
                    self._register_crossattn_enhancement(
                        crossattn_cache_dual, tok_src + tok_trg,
                        layers=enhance_layers, fg_boost_factor=fg_boost_factor,
                        current_src_fg_mask=inloop_trg_fg_mask,
                    )

                # add noise to both source video and generating video
                noisy_src_input = self.scheduler.add_noise(
                    src_input.flatten(0, 1),
                    fwd_noise.flatten(0, 1),
                    timestep[: batch_size],
                ).unflatten(0, src_input.shape[:2])
                noisy_pred_input = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    fwd_trg_noise.flatten(0, 1),
                    timestep[batch_size: ],
                ).unflatten(0, denoised_pred.shape[:2])
                noisy_input = torch.cat([noisy_src_input, noisy_pred_input], dim=0)

                # model forward
                if target_semantic_competition:
                    if current_memory_query_weight is None:
                        raise RuntimeError(
                            "Semantic KV authority requires a current "
                            "transactional read gate"
                        )
                    for layer_index, layer_cache in enumerate(kv_cache_dual):
                        cached_gate = layer_cache.get(
                            "current_causal_owner_mask"
                        )
                        if (
                            cached_gate is None
                            or cached_gate.shape
                            != current_memory_query_weight.shape
                            or not torch.equal(
                                cached_gate, current_memory_query_weight
                            )
                        ):
                            raise RuntimeError(
                                "Semantic KV authority was lost before "
                                "denoising at layer "
                                f"{layer_index}, step {index}"
                            )
                    if index == 0:
                        print(
                            "SEMANTIC_KV_AUTHORITY "
                            f"block={current_start_frame // self.num_frame_per_block} "
                            "contract=persistent_within_block "
                            "read_gate=transaction_x_semantic_support "
                            f"coverage={(current_memory_query_weight > 0).float().mean().item():.4f} "
                            f"strength={current_memory_query_weight.mean().item():.4f}"
                        )
                velocity_pred, _ = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=kv_cache_dual,
                    crossattn_cache=crossattn_cache_dual,
                    current_start=current_start_frame * self.frame_seq_length
                )
                if factorized_native_target_history and index == 0:
                    history_diagnostics = shared_dict_dual.get(
                        "native_target_history_diagnostics", {}
                    )
                    if history_diagnostics:
                        def diagnostic_mean(name):
                            values = [
                                item[name].float()
                                for item in history_diagnostics.values()
                                if name in item
                            ]
                            return (
                                torch.stack(values).mean().item()
                                if values else 0.0
                            )

                        print(
                            "NATIVE_TARGET_HISTORY "
                            f"block={current_start_frame // self.num_frame_per_block} "
                            "selected=1 "
                            "dense_read=1.0000 "
                            "counterfactual_factorized_read="
                            f"{diagnostic_mean('factorized_history_read_ratio'):.4f} "
                            "owner_output_gap="
                            f"{diagnostic_mean('owner_output_gap'):.4f} "
                            "owner_coverage="
                            f"{diagnostic_mean('owner_coverage'):.4f} "
                            "owner_source_block="
                            f"{int(factorized_owner_source_block)}"
                        )
                if causal_paired_edit_memory and index == 0:
                    paired_diagnostics = shared_dict_dual.get(
                        "paired_edit_memory_diagnostics", {}
                    )
                    if paired_diagnostics:
                        def paired_diagnostic_mean(name):
                            values = [
                                item[name].float()
                                for item in paired_diagnostics.values()
                                if name in item
                            ]
                            return (
                                torch.stack(values).mean().item()
                                if values else 0.0
                            )

                        print(
                            "PAIRED_EDIT_ATTENTION "
                            f"block={current_start_frame // self.num_frame_per_block} "
                            "native_fallback=1 "
                            "role=object_target_memory "
                            f"support={paired_diagnostic_mean('read_support'):.4f} "
                            f"similarity={paired_diagnostic_mean('read_similarity'):.4f} "
                            f"correction={paired_diagnostic_mean('correction'):.6f} "
                            "value_projection="
                            f"{paired_diagnostic_mean('value_projection'):.6f} "
                            "ungated_leakage="
                            f"{paired_diagnostic_mean('ungated_projection_leakage'):.6f} "
                            "anchor_delta="
                            f"{paired_diagnostic_mean('anchor_delta'):.6f} "
                            "anchor_query_coverage="
                            f"{paired_diagnostic_mean('anchor_query_coverage'):.4f} "
                            "anchor_keys="
                            f"{paired_diagnostic_mean('anchor_key_count'):.1f} "
                            "dual_timescale_anchor="
                            f"{int(paired_memory_dual_timescale_anchor)} "
                            "canonical_key_anchor="
                            f"{int(paired_memory_canonical_key_anchor)} "
                            "query_gate="
                            f"{int(paired_memory_query_gated_projection)}"
                        )
                if role_fixed_native_history and (
                    index == 0
                    or native_history_timestep_counterfactual_memory
                ):
                    native_role_diagnostics = shared_dict_dual.get(
                        "role_native_history_diagnostics", {}
                    )
                    if native_role_diagnostics:
                        def native_role_mean(name):
                            values = [
                                item[name].float().mean()
                                for item in native_role_diagnostics.values()
                                if name in item
                            ]
                            return (
                                torch.stack(values).mean().item()
                                if values else 0.0
                            )

                        def native_role_map(name):
                            values = [
                                item[name].float()
                                for item in native_role_diagnostics.values()
                                if name in item
                            ]
                            if not values:
                                return None
                            shape = values[0].shape
                            if any(value.shape != shape for value in values):
                                raise ValueError(
                                    f"Native-history diagnostic '{name}' "
                                    "must align across layers"
                                )
                            return torch.stack(values, dim=0).mean(
                                dim=0
                            ).reshape_as(
                                hand_role_debug["object_posterior"]
                            )

                        for diagnostic_name, debug_name in (
                            (
                                "applied_read_strength",
                                "native_history_applied_read_strength",
                            ),
                            (
                                "canonical_appearance_delta",
                                "native_history_canonical_appearance_delta",
                            ),
                            (
                                "canonical_payload_exclusive",
                                "native_history_canonical_payload_exclusive",
                            ),
                            (
                                "mutable_target_payload_enabled",
                                "native_history_mutable_target_payload_enabled",
                            ),
                            (
                                "recent_entry_admitted",
                                "native_history_recent_entry_admitted",
                            ),
                            (
                                "canonical_fallback_admitted",
                                "native_history_canonical_fallback_admitted",
                            ),
                            (
                                "recent_payload_consistency",
                                "native_history_recent_payload_consistency",
                            ),
                            (
                                "recent_payload_trust",
                                "native_history_recent_payload_trust",
                            ),
                            (
                                "recent_payload_rejected",
                                "native_history_recent_payload_rejected",
                            ),
                            (
                                "canonical_payload_weight",
                                "native_history_canonical_payload_weight",
                            ),
                            (
                                "request_strength",
                                "native_history_request_strength",
                            ),
                            (
                                "request_support",
                                "native_history_request_support",
                            ),
                            (
                                "read_scope",
                                "native_history_read_scope",
                            ),
                            (
                                "address_confidence",
                                "native_history_address_confidence",
                            ),
                            (
                                "flow_transport_confidence",
                                "native_history_flow_transport_confidence",
                            ),
                            (
                                "flow_appearance_trust",
                                "native_history_flow_appearance_trust",
                            ),
                            (
                                "flow_local_transport_confidence",
                                "native_history_flow_local_transport_confidence",
                            ),
                            (
                                "sink_admitted",
                                "native_history_sink_admission",
                            ),
                            (
                                "sink_selected_frame",
                                "native_history_sink_selected_frame",
                            ),
                            (
                                "sink_source_similarity",
                                "native_history_sink_source_similarity",
                            ),
                            (
                                "sink_attention_entropy",
                                "native_history_sink_attention_entropy",
                            ),
                            (
                                "sink_attention_peak",
                                "native_history_sink_attention_peak",
                            ),
                            (
                                "sink_coverage",
                                "native_history_sink_coverage",
                            ),
                            (
                                "sink_applied_strength",
                                "native_history_sink_applied_strength",
                            ),
                            (
                                "tccm_admitted",
                                f"tccm_step_{index:02d}_admission",
                            ),
                            (
                                "tccm_source_similarity",
                                f"tccm_step_{index:02d}_source_similarity",
                            ),
                            (
                                "tccm_correspondence_confidence",
                                f"tccm_step_{index:02d}_correspondence",
                            ),
                            (
                                "tccm_desired_norm",
                                f"tccm_step_{index:02d}_desired_norm",
                            ),
                            (
                                "tccm_current_norm",
                                f"tccm_step_{index:02d}_current_norm",
                            ),
                            (
                                "tccm_error_norm",
                                f"tccm_step_{index:02d}_error_norm",
                            ),
                            (
                                "tccm_residual_error_norm",
                                f"tccm_step_{index:02d}_residual_error_norm",
                            ),
                            (
                                "tccm_gain",
                                f"tccm_step_{index:02d}_gain",
                            ),
                            (
                                "tccm_clip_scale",
                                f"tccm_step_{index:02d}_clip_scale",
                            ),
                            (
                                "tccm_candidate_count",
                                f"tccm_step_{index:02d}_candidate_count",
                            ),
                            (
                                "tccm_attention_entropy",
                                f"tccm_step_{index:02d}_attention_entropy",
                            ),
                            (
                                "tccm_attention_peak",
                                f"tccm_step_{index:02d}_attention_peak",
                            ),
                        ):
                            diagnostic_map = native_role_map(diagnostic_name)
                            if diagnostic_map is not None:
                                hand_role_debug[debug_name] = diagnostic_map

                        if native_history_timestep_counterfactual_memory:
                            tccm_step_debug = {}
                            for diagnostic_name in (
                                "tccm_admitted",
                                "tccm_source_similarity",
                                "tccm_correspondence_confidence",
                                "tccm_desired_norm",
                                "tccm_current_norm",
                                "tccm_error_norm",
                                "tccm_residual_error_norm",
                                "tccm_gain",
                                "tccm_clip_scale",
                                "tccm_candidate_count",
                                "tccm_attention_entropy",
                                "tccm_attention_peak",
                            ):
                                value = native_role_map(diagnostic_name)
                                if value is not None:
                                    tccm_step_debug[diagnostic_name] = value
                            if save_role_dir is not None and tccm_step_debug:
                                self._save_hand_role_debug(
                                    save_role_dir,
                                    current_start_frame
                                    // self.num_frame_per_block,
                                    tccm_step_debug,
                                    artifact_suffix=(
                                        f"_tccm_step_{index:02d}"
                                    ),
                                )
                            print(
                                "TCCM_READ "
                                f"block={current_start_frame // self.num_frame_per_block} "
                                f"step={index} "
                                f"coverage={native_role_mean('tccm_admitted'):.4f} "
                                "correspondence="
                                f"{native_role_mean('tccm_correspondence_confidence'):.4f} "
                                "source_similarity="
                                f"{native_role_mean('tccm_source_similarity'):.4f} "
                                f"desired={native_role_mean('tccm_desired_norm'):.6f} "
                                f"current={native_role_mean('tccm_current_norm'):.6f} "
                                f"error={native_role_mean('tccm_error_norm'):.6f} "
                                "residual_error="
                                f"{native_role_mean('tccm_residual_error_norm'):.6f} "
                                f"gain={native_role_mean('tccm_gain'):.4f} "
                                f"clip={native_role_mean('tccm_clip_scale'):.4f}"
                            )
                        if index == 0:
                            print(
                            "ROLE_FIXED_NATIVE_KV_READ "
                            f"block={current_start_frame // self.num_frame_per_block} "
                            f"coverage={native_role_mean('admitted'):.4f} "
                            "lineage_coverage="
                            f"{native_role_mean('lineage_admitted'):.4f} "
                            "lineage_confidence="
                            f"{native_role_mean('lineage_confidence'):.4f} "
                            f"similarity={native_role_mean('best_similarity'):.4f} "
                            f"delta={native_role_mean('output_delta'):.6f} "
                            f"read_strength={native_role_mean('read_strength'):.4f} "
                            "applied_strength="
                            f"{native_role_mean('applied_read_strength'):.4f} "
                            "canonical_payload_exclusive="
                            f"{native_role_mean('canonical_payload_exclusive'):.4f} "
                            "mutable_target_payload_enabled="
                            f"{native_role_mean('mutable_target_payload_enabled'):.4f} "
                            "appearance_delta="
                            f"{native_role_mean('canonical_appearance_delta'):.6f} "
                            f"canonical={native_role_mean('canonical_candidates'):.1f} "
                            f"recent={native_role_mean('recent_tokens'):.1f} "
                            "recent_payload="
                            f"{native_role_mean('recent_payload_tokens'):.1f} "
                            "recent_entry="
                            f"{native_role_mean('recent_entry_admitted'):.4f} "
                            "canonical_fallback="
                            f"{native_role_mean('canonical_fallback_admitted'):.4f} "
                            "payload_consistency="
                            f"{native_role_mean('recent_payload_consistency_on_match'):.4f} "
                            "payload_trust="
                            f"{native_role_mean('recent_payload_trust_on_match'):.4f} "
                            "payload_rejection="
                            f"{native_role_mean('recent_payload_rejection_rate'):.4f} "
                            "canonical_weight="
                            f"{native_role_mean('canonical_payload_weight_on_read'):.4f} "
                            f"lineage={native_role_mean('lineage_tokens'):.1f} "
                            f"bypass={native_role_mean('bypass'):.0f} "
                            "bootstrap_alias="
                            f"{native_role_mean('bootstrap_alias'):.0f} "
                            "recent_origin="
                            f"{native_role_mean('recent_start_frame'):.0f} "
                            "current_origin="
                            f"{native_role_mean('current_start_frame'):.0f} "
                            "part_similarity="
                            f"{native_role_mean('admitted_part_similarity'):.4f} "
                            "part_candidates="
                            f"{native_role_mean('admitted_part_candidate_fraction'):.4f} "
                            "baseline_delta="
                            f"{native_role_mean('admitted_baseline_output_delta'):.6f} "
                            "part_refine_scale="
                            f"{native_role_mean('admitted_part_refinement_scale'):.4f} "
                            "flow_indexed="
                            f"{native_role_mean('flow_indexed_read'):.0f} "
                            "appearance_trust="
                            f"{native_role_mean('flow_appearance_trust_on_read'):.4f} "
                            "local_transport="
                            f"{native_role_mean('flow_local_transport_confidence_on_read'):.4f} "
                            "multiframe_sink="
                            f"{native_role_mean('multiframe_identity_sink'):.0f} "
                            "sink_coverage="
                            f"{native_role_mean('sink_admitted'):.4f} "
                            "sink_frame="
                            f"{native_role_mean('sink_selected_frame_on_read'):.3f} "
                            "sink_source_similarity="
                            f"{native_role_mean('sink_source_similarity_on_read'):.4f} "
                            "sink_entropy="
                            f"{native_role_mean('sink_attention_entropy_on_read'):.4f} "
                            "sink_peak="
                            f"{native_role_mean('sink_attention_peak_on_read'):.4f} "
                            "sink_frame_coverage="
                            f"{native_role_mean('sink_coverage_on_read'):.4f} "
                            "sink_strength="
                            f"{native_role_mean('sink_applied_strength_on_read'):.4f} "
                            + (
                                "payload=immutable_canonical_target_kv "
                                if native_history_payload_invariant_lineage
                                else "payload=native_target_kv "
                            )
                            + "fallback=exact_926"
                            )
                if (
                    native_history_verified_attention_authority
                    and index == 0
                ):
                    authority_diagnostics = shared_dict_dual.get(
                        "verified_attention_authority_diagnostics", {}
                    )
                    if authority_diagnostics:
                        authority_layers = [
                            item["gate"].float()
                            for item in authority_diagnostics.values()
                        ]
                        mean_authority_gate = torch.stack(
                            authority_layers, dim=0
                        ).mean(dim=0)
                        hand_role_debug[
                            "verified_attention_authority_gate"
                        ] = mean_authority_gate.reshape_as(
                            hand_role_debug["object_posterior"]
                        )

                        def authority_mean(name):
                            return torch.stack([
                                item[name].float()
                                for item in authority_diagnostics.values()
                            ]).mean().item()

                        print(
                            "VERIFIED_ATTENTION_AUTHORITY "
                            f"block={current_start_frame // self.num_frame_per_block} "
                            "source=native_vs_factorized_counterfactual "
                            "gate=automatic_owner_x_cross_layer_kv_read "
                            "start_layer="
                            f"{max(role_native_history_layers)} "
                            "layers="
                            f"{len(authority_diagnostics)} "
                            "coverage="
                            f"{(mean_authority_gate > 0).float().mean().item():.4f} "
                            "active_gate="
                            f"{authority_mean('active_gate'):.4f} "
                            "native_factorized_gap="
                            f"{authority_mean('active_native_factorized_gap'):.6f} "
                            "output_delta="
                            f"{authority_mean('active_output_delta'):.6f}"
                        )
                current_verified_native_history_support = None
                if (
                    native_history_transactional_owner
                    and role_fixed_native_history
                ):
                    native_role_admissions = shared_dict_dual.get(
                        "role_native_history_admissions", {}
                    )
                    admitted_layers = [
                        torch.cat(items, dim=0).float()
                        for layer, items in native_role_admissions.items()
                        if layer in role_native_history_layers and items
                    ]
                    if admitted_layers:
                        admitted_shape = admitted_layers[0].shape
                        if any(
                            value.shape != admitted_shape
                            for value in admitted_layers
                        ):
                            raise ValueError(
                                "Native-history admission maps must align "
                                "across layers"
                            )
                        layer_reads = torch.stack(admitted_layers, dim=0)
                        if native_history_consistent_transaction:
                            # The same soft retrieval confidence that injects
                            # target KV arbitrates source appearance. Geometric
                            # mean rewards cross-layer consensus without turning
                            # one weak layer into a binary all-or-nothing veto.
                            layer_agreement = (
                                layer_reads > 0.0
                            ).float().mean(dim=0)
                            current_verified_native_history_support = (
                                (
                                    layer_reads.mean(dim=0)
                                    * layer_agreement
                                ).reshape_as(
                                    hand_role_debug["object_posterior"]
                                )
                            )
                        else:
                            # Legacy cross-layer mean.
                            current_verified_native_history_support = (
                                layer_reads.mean(dim=0).reshape_as(
                                    hand_role_debug["object_posterior"]
                                )
                            )
                    else:
                        current_verified_native_history_support = (
                            torch.zeros_like(
                                hand_role_debug["object_posterior"],
                                dtype=torch.float32,
                            )
                        )
                    if index == 0:
                        hand_role_debug[
                            "native_history_verified_admission"
                        ] = current_verified_native_history_support
                        print(
                            "VERIFIED_NATIVE_KV_ARBITRATION "
                            f"block={current_start_frame // self.num_frame_per_block} "
                            f"layers={len(admitted_layers)} "
                            "admitted="
                            f"{current_verified_native_history_support.mean().item():.4f} "
                            "source_suppression="
                            f"{native_history_verified_source_suppression:.3f}"
                        )
                if source_coordinate_identity and index in identity_trace_steps:
                    identity_by_layer = shared_dict_dual.get(
                        "target_identity_support", {}
                    )
                    identity_diagnostics_by_layer = shared_dict_dual.get(
                        "target_identity_diagnostics", {}
                    )
                    identity_layers = [
                        identity_by_layer[layer]
                        for layer in identity_memory_layers
                        if layer in identity_by_layer
                    ]
                    if identity_layers:
                        step_support = torch.stack(
                            identity_layers, dim=0
                        ).mean(dim=0).reshape_as(
                            hand_role_debug["object_posterior"]
                        )
                        hand_role_debug[
                            f"identity_step_{index:02d}_support"
                        ] = step_support
                        step_diagnostics = {}
                        for diagnostic_name in (
                            "best_similarity",
                            "prototype_assignment_entropy",
                            "prototype_assignment_peak",
                            "prototype_assignment_margin",
                            "absolute_match",
                            "relative_match",
                            "support_before_mask",
                            "support_after_mask",
                            "correction_ratio",
                            "appearance_subspace_coherence",
                            "appearance_subspace_action",
                        ):
                            diagnostic_layers = [
                                identity_diagnostics_by_layer[layer][
                                    diagnostic_name
                                ]
                                for layer in identity_memory_layers
                                if (
                                    layer in identity_diagnostics_by_layer
                                    and diagnostic_name
                                    in identity_diagnostics_by_layer[layer]
                                )
                            ]
                            if diagnostic_layers:
                                diagnostic = torch.stack(
                                    diagnostic_layers, dim=0
                                ).mean(dim=0).reshape_as(step_support)
                                step_diagnostics[diagnostic_name] = diagnostic
                                hand_role_debug[
                                    f"identity_step_{index:02d}_"
                                    f"{diagnostic_name}"
                                ] = diagnostic
                        correction_ratio = step_diagnostics.get(
                            "correction_ratio",
                            torch.zeros_like(step_support),
                        )
                        similarity = step_diagnostics.get(
                            "best_similarity",
                            torch.zeros_like(step_support),
                        )
                        print(
                            "SOURCE_COORDINATE_IDENTITY_READ "
                            f"block="
                            f"{current_start_frame // self.num_frame_per_block} "
                            f"step={index} "
                            f"support={step_support.mean().item():.4f} "
                            f"peak={step_support.max().item():.4f} "
                            f"similarity={similarity.mean().item():.4f} "
                            "correction_ratio="
                            f"{correction_ratio.mean().item():.4f}"
                        )
                if (
                    current_causal_identity_bootstrap is not None
                    and index > 0
                ):
                    bootstrap_support_by_layer = (
                        shared_dict_dual.get(
                            "target_identity_support",
                            {},
                        )
                    )
                    bootstrap_support_layers = [
                        bootstrap_support_by_layer[layer]
                        for layer in identity_memory_layers
                        if layer in bootstrap_support_by_layer
                    ]
                    if bootstrap_support_layers:
                        hand_role_debug[
                            "identity_causal_bootstrap_read_support"
                        ] = torch.stack(
                            bootstrap_support_layers,
                            dim=0,
                        ).mean(dim=0).reshape_as(
                            hand_role_debug["object_posterior"]
                        )
                # for getting real output
                t_i = current_timestep / 1000
                v_src, v_trg = velocity_pred.chunk(2, dim=0)
                v_gt = fwd_noise - src_input
                if (
                    hand_role_enabled
                    and index == 0
                    and (
                        adaptive_role_enabled
                        or hand_field_update_mode != "off"
                    )
                ):
                    apply_field_update = (
                        isinstance(
                            native_owner_tracker,
                            AutomaticTransactionalOwnerTracker,
                        )
                        or
                        (
                            adaptive_role_enabled
                            and (
                                (
                                    not posterior_flow_enabled
                                    and not bayes_flow_enabled
                                    and not factorized_bayes_enabled
                                )
                                or (
                                    posterior_flow_enabled
                                    and posterior_flow_use_field
                                )
                            )
                        )
                        or (
                            not bayes_flow_enabled
                            and not factorized_bayes_enabled
                            and hand_field_update_mode == "posterior"
                        )
                    )
                    hand_role_inference = (
                        hand_role_inferencer.refine_with_field(
                            prior=hand_role_inference,
                            source_velocity=v_src.detach(),
                            target_velocity=v_trg.detach(),
                            hand_mask=hand_only_mask[
                                :, role_left:role_right
                            ],
                            hand_occupancy=hand_occupancy_mask[
                                :, role_left:role_right
                            ],
                            apply_update=apply_field_update,
                        )
                    )
                    if source_flow_verified_region:
                        if flow_role_evidence is None:
                            raise RuntimeError(
                                "Source-flow verified region lost flow "
                                "evidence during field refinement"
                            )
                        hand_role_inference = (
                            hand_role_inferencer
                            .apply_source_flow_verified_region(
                                hand_role_inference,
                                flow_role_evidence,
                                owner_support=(
                                    current_causal_ownership.owner_support
                                ),
                                hand_exclusion=hand_role_inference.debug[
                                    "hand_hard_exclusion"
                                ],
                                hand_occupancy=hand_occupancy_mask[
                                    :, role_left:role_right
                                ],
                                owner_radius=(
                                    source_flow_verified_owner_radius
                                ),
                                background_veto_threshold=(
                                    source_flow_background_veto_threshold
                                ),
                                background_veto_min_confidence=(
                                    source_flow_background_veto_min_confidence
                                ),
                            )
                        )
                    hand_role_debug = hand_role_inference.debug
                    if current_edit_authority is not None:
                        # The flow pass refines whole-object ownership only.
                        # Keep the target-text part authority immutable within
                        # this causal block so flow cannot re-authorize body
                        # or background tokens.
                        hand_role_debug.update(
                            current_edit_authority.as_debug_maps()
                        )
                        hand_role_debug["edit_authority"] = (
                            current_edit_authority.support.float()
                        )
                    if current_causal_ownership is not None:
                        hand_role_debug.update(
                            current_causal_ownership.as_debug_maps(
                                hand_role_debug[
                                    "object_posterior"
                                ].shape
                            )
                        )
                    if (
                        current_causal_ownership is not None
                        and isinstance(
                            native_owner_tracker,
                            AutomaticTransactionalOwnerTracker,
                        )
                    ):
                        refined_owner_shape = hand_role_debug[
                            "object_posterior"
                        ].shape
                        refined_role_maps = F.adaptive_avg_pool2d(
                            torch.stack([
                                hand_role_inference.roles.object,
                                hand_role_inference.roles.boundary,
                            ], dim=2).flatten(0, 1),
                            output_size=refined_owner_shape[-2:],
                        ).reshape(
                            batch_size, current_num_frames, 2,
                            *refined_owner_shape[-2:],
                        )
                        current_transactional_owner = (
                            native_owner_tracker(
                                ownership=current_causal_ownership,
                                object_posterior=hand_role_debug[
                                    "object_posterior"
                                ],
                                posterior_threshold=hand_role_debug[
                                    "posterior_threshold"
                                ],
                                source_attention=hand_role_debug[
                                    "source_attention"
                                ],
                                hand_probability=hand_role_debug[
                                    "hand_hard_exclusion"
                                ],
                                hand_proximity=hand_role_debug[
                                    "hand_proximity"
                                ],
                                object_role=refined_role_maps[:, :, 0],
                                boundary_role=refined_role_maps[:, :, 1],
                                field_likelihood=hand_role_debug.get(
                                    "adaptive_field_likelihood",
                                    hand_role_debug.get(
                                        "field_observation"
                                    ),
                                ),
                                field_reliability=hand_role_debug.get(
                                    "adaptive_field_reliability"
                                ),
                                update_state=True,
                            )
                        )
                        field_owner_observation = (
                            hand_role_debug[
                                "object_posterior"
                            ].float()
                            * (
                                hand_role_debug[
                                    "object_posterior"
                                ]
                                >= hand_role_debug[
                                    "posterior_threshold"
                                ]
                            ).float()
                            * (
                                1.0
                                - hand_role_debug[
                                    "hand_hard_exclusion"
                                ].float().clamp(0.0, 1.0)
                            )
                        ).reshape(batch_size, -1)
                        if motion_geometry_owner:
                            causal_ownership_tracker.correct_current_observation(
                                observation_weight=field_owner_observation,
                                tokens_per_frame=self.frame_seq_length,
                            )
                        # The causal source-address anchor advances only from
                        # the flow-verified write core.  Contact/boundary and
                        # lifecycle read support can never become next-block
                        # ownership.
                        causal_ownership_tracker.commit_verified(
                            source_features=source_query_features,
                            verified_weight=(
                                torch.where(
                                    current_transactional_owner.write_weight
                                    >= native_history_min_write_confidence,
                                    torch.minimum(
                                        current_transactional_owner.write_weight,
                                        field_owner_observation,
                                    ),
                                    torch.zeros_like(
                                        current_transactional_owner.write_weight
                                    ),
                                )
                            ),
                            tokens_per_frame=self.frame_seq_length,
                        )
                        hand_role_debug.update(
                            current_transactional_owner.as_debug_maps(
                                refined_owner_shape
                            )
                        )
                        current_memory_query_weight = (
                            current_transactional_owner.read_weight.float()
                        )
                        if native_history_motion_owner_dense_read:
                            if native_history_topology_complete_read:
                                (
                                    current_memory_query_weight,
                                    motion_read_recovery,
                                    topology_read_recovery,
                                    topology_holes,
                                ) = build_topology_complete_motion_owner_read_weight(
                                    current_causal_ownership,
                                    current_transactional_owner,
                                    shape=refined_owner_shape,
                                    hand_exclusion=hand_role_debug[
                                        "hand_hard_exclusion"
                                    ],
                                )
                                hand_role_debug[
                                    "native_owner_topology_holes"
                                ] = topology_holes.reshape_as(
                                    hand_role_debug["object_posterior"]
                                ).float()
                                hand_role_debug[
                                    "native_owner_topology_read_recovery"
                                ] = topology_read_recovery.reshape_as(
                                    hand_role_debug["object_posterior"]
                                )
                            else:
                                (
                                    current_memory_query_weight,
                                    motion_read_recovery,
                                ) = build_motion_owner_read_weight(
                                    current_causal_ownership,
                                    current_transactional_owner,
                                )
                            hand_role_debug[
                                "native_owner_motion_read_recovery"
                            ] = motion_read_recovery.reshape_as(
                                hand_role_debug["object_posterior"]
                            )
                            hand_role_debug[
                                "native_owner_effective_read"
                            ] = current_memory_query_weight.reshape_as(
                                hand_role_debug["object_posterior"]
                            )
                        if (
                            current_target_authority_support_tokens
                            is not None
                        ):
                            current_memory_query_weight = (
                                apply_semantic_transaction_gate(
                                    current_memory_query_weight,
                                    current_target_authority_support_tokens,
                                )
                            )
                        print(
                            "AUTOMATIC_OWNER_TRANSACTION "
                            f"block={current_start_frame // self.num_frame_per_block} "
                            "external_object_mask=disabled "
                            + (
                                "owner_source=hand_attention_plus_source_rgb_raft "
                                if motion_geometry_owner
                                else "owner_source=hand_attention_plus_source_features "
                            )
                            +
                            f"read={current_transactional_owner.read_weight.mean().item():.4f} "
                            f"write={current_transactional_owner.write_weight.mean().item():.4f} "
                            f"contact={current_transactional_owner.contact_weight.mean().item():.4f} "
                            f"lifecycle={current_transactional_owner.lifecycle_weight.mean().item():.4f}"
                        )
                    if (
                        adaptive_role_enabled
                        or hand_field_update_mode == "posterior"
                    ):
                        current_roles = hand_role_inference.roles
                        if adaptive_role_enabled:
                            role_edit_tokens = (
                                hand_role_debug["object_posterior"]
                                >= hand_role_debug[
                                    "posterior_threshold"
                                ]
                            ).reshape(batch_size, -1)
                        else:
                            role_edit_tokens = (
                                hand_role_inference
                                .token_edit_confidence
                                >= hand_posterior_threshold
                            )
                        if (
                            current_causal_ownership is not None
                            and not source_flow_verified_region
                        ):
                            role_edit_tokens = (
                                role_edit_tokens
                                | current_causal_ownership.owner_support
                            )
                        if current_target_authority_tokens is not None:
                            role_edit_tokens = (
                                current_target_authority_tokens
                                >= target_semantic_min_confidence
                            )
                        if factorized_bayes_enabled:
                            current_factorized_operators = (
                                factorized_operator_builder(
                                    roles=current_roles,
                                    evidence=hand_role_debug,
                                    expected_token_length=(
                                        role_edit_tokens.shape[1]
                                    ),
                                )
                            )
                            hand_role_debug.update(
                                current_factorized_operators.as_debug_maps()
                            )
                        if not soft_region_modulation:
                            inloop_trg_fg_mask = role_edit_tokens
                        src_fg_mask_map = self._mask_reshape(
                            role_edit_tokens
                            if not soft_region_modulation
                            else inloop_trg_fg_mask,
                            size=(current_num_frames, height, width),
                        )
                        if not soft_region_modulation:
                            self._inject_masks_to_kv_cache(
                                kv_cache_dual,
                                trg_fg_mask_cache,
                                role_edit_tokens,
                                factorized_operator_cache=(
                                    factorized_operator_cache
                                    if factorized_bayes_enabled
                                    else None
                                ),
                                current_factorized_operators=(
                                    current_factorized_operators
                                ),
                                target_owned_mask_cache=(
                                    target_owned_mask_cache
                                ),
                                current_target_owned_mask=(
                                    current_target_owned_mask
                                    if target_owned_object_handoff
                                    else None
                                ),
                                current_identity_read_mask=(
                                    current_identity_read_mask
                                ),
                                current_causal_owner_mask=(
                                    current_memory_query_weight
                                ),
                            )
                    print(
                        "HAND_ROLE_FIELD "
                        "block="
                        f"{current_start_frame // self.num_frame_per_block} "
                        "mode="
                        f"{'adaptive' if adaptive_role_enabled else hand_field_update_mode} "
                        f"applied={int(apply_field_update)} "
                        f"edit_tokens={role_edit_tokens.float().mean().item():.4f} "
                        f"field={hand_role_debug['field_score'].mean().item():.4f} "
                        f"observation={hand_role_debug['field_observation'].mean().item():.4f}"
                        + (
                            " reliability="
                            f"{hand_role_debug['adaptive_field_reliability'].mean().item():.4f}"
                            if adaptive_role_enabled
                            else ""
                        )
                    )
                    if save_role_dir is not None:
                        block_index = (
                            current_start_frame
                            // self.num_frame_per_block
                        )
                        self._save_role_state(
                            save_role_dir,
                            block_index,
                            current_roles,
                        )
                        self._save_hand_role_debug(
                            save_role_dir,
                            block_index,
                            hand_role_debug,
                        )
                if (
                    appearance_leakage_decomposition
                    and not factorized_immutable_target_memory
                    and index == 0
                ):
                    semantic_attention = src_fg_mask_soft.reshape_as(
                        hand_role_debug["object_posterior"]
                    )
                    current_target_change_core = build_target_change_core(
                        source_velocity=v_src.detach(),
                        target_velocity=v_trg.detach(),
                        source_semantic_attention=semantic_attention,
                        hand_mask=hand_only_mask[
                            :, role_left:role_right
                        ],
                        hand_exclusion_radius=(
                            ignition_hand_exclusion_radius
                        ),
                        contact_radius=ignition_contact_radius,
                    )
                    hand_role_debug.update(
                        current_target_change_core.as_debug_maps()
                    )
                    if (
                        source_coordinate_identity
                        and current_identity_read_mask is not None
                    ):
                        owner_routing_core = self._mask_reshape(
                            current_identity_read_mask > 0.05,
                            size=(current_num_frames, height, width),
                        )
                        hand_role_debug[
                            "ignition_owner_routing_core"
                        ] = F.avg_pool2d(
                            (
                                current_target_change_core.mask
                                & owner_routing_core
                            ).float().reshape(
                                batch_size * current_num_frames,
                                1,
                                height,
                                width,
                            ),
                            kernel_size=2,
                            stride=2,
                        ).reshape_as(
                            hand_role_debug["object_posterior"]
                        )
                    if target_identity_enabled:
                        core_mask = current_target_change_core.mask
                        core_tokens = F.avg_pool2d(
                            core_mask.float().reshape(
                                batch_size * current_num_frames,
                                1,
                                height,
                                width,
                            ),
                            kernel_size=2,
                            stride=2,
                        ).reshape_as(
                            hand_role_debug["object_posterior"]
                        ) >= 0.5
                        hand_tokens = F.max_pool2d(
                            hand_only_mask[
                                :, role_left:role_right
                            ].float().reshape(
                                batch_size * current_num_frames,
                                1,
                                height,
                                width,
                            ),
                            kernel_size=2,
                            stride=2,
                        ).reshape_as(core_tokens) > 0.0
                        verified_core_tokens = (
                            core_tokens & ~hand_tokens
                        ).reshape(batch_size, -1)
                        if source_coordinate_identity:
                            owner_write_tokens = (
                                current_identity_read_mask > 0.05
                            )
                            owner_has_support = owner_write_tokens.any(
                                dim=-1, keepdim=True
                            )
                            validated_owner = (
                                owner_write_tokens
                                & verified_core_tokens
                            )
                            validator_has_support = validated_owner.any(
                                dim=-1, keepdim=True
                            )
                            # Ignition is a block-level confidence check, not
                            # a pixel-wise locator. Once any overlap verifies
                            # the edit, write the full clean-source owner.
                            verified_write_tokens = torch.where(
                                validator_has_support,
                                owner_write_tokens,
                                torch.zeros_like(owner_write_tokens),
                            )
                            if not target_identity_memory.export():
                                # Bootstrap has no immutable identity yet.
                                # Ignition may seed it only when source owner
                                # bootstrap is unavailable.
                                verified_write_tokens = torch.where(
                                    owner_has_support,
                                    verified_write_tokens,
                                    verified_core_tokens,
                                )
                        else:
                            verified_write_tokens = verified_core_tokens
                        current_identity_write_mask = (
                            verified_write_tokens
                        )
                        if not source_coordinate_identity:
                            current_identity_read_mask = (
                                verified_core_tokens
                            )
                        if identity_visibility_lifecycle:
                            current_identity_lifecycle = (
                                target_identity_memory
                                .update_visibility_lifecycle(
                                    object_core=(
                                        current_identity_read_mask
                                    ),
                                    frame_visible=(
                                        current_identity_read_mask
                                        .reshape_as(core_tokens)
                                        .flatten(2)
                                        .any(dim=-1)[:, :, None, None]
                                    ),
                                    tokens_per_frame=(
                                        self.frame_seq_length
                                    ),
                                    max_occluded_blocks=(
                                        identity_max_occluded_blocks
                                    ),
                                )
                            )
                            current_identity_read_mask = (
                                current_identity_lifecycle.read_mask
                            )
                        hand_role_debug[
                            "ignition_identity_core"
                        ] = verified_core_tokens.reshape_as(
                            core_tokens
                        ).float()
                        hand_role_debug[
                            "identity_verified_write_core"
                        ] = verified_write_tokens.reshape_as(
                            core_tokens
                        ).float()
                    print(
                        "TARGET_CHANGE_IGNITION "
                        f"block="
                        f"{current_start_frame // self.num_frame_per_block} "
                        f"core="
                        f"{current_target_change_core.mask.float().mean().item():.4f} "
                        f"candidate="
                        f"{current_target_change_core.candidate_mask.float().mean().item():.4f} "
                        f"hand_overlap="
                        f"{(current_target_change_core.mask & current_target_change_core.hand_exclusion_mask).float().mean().item():.4f}"
                    )
                if (bayes_flow_enabled or factorized_bayes_enabled) and index == 0:
                    current_control_belief = control_belief_builder(
                        debug=hand_role_debug,
                        hand_mask=hand_occupancy_mask[
                            :, role_left:role_right
                        ],
                    )
                    if factorized_bayes_enabled:
                        current_factorized_operators = (
                            factorized_operator_builder(
                                roles=current_roles,
                                evidence=hand_role_debug,
                                expected_token_length=(
                                    role_edit_tokens.shape[1]
                                ),
                            )
                        )
                        hand_role_debug.update(
                            current_factorized_operators.as_debug_maps()
                        )
                        self._inject_masks_to_kv_cache(
                            kv_cache_dual,
                            trg_fg_mask_cache,
                            role_edit_tokens,
                            factorized_operator_cache=(
                                factorized_operator_cache
                            ),
                            current_factorized_operators=(
                                current_factorized_operators
                            ),
                            current_identity_read_mask=(
                                current_identity_read_mask
                            ),
                            current_causal_owner_mask=(
                                current_memory_query_weight
                            ),
                        )
                    if edit_commitment_enabled:
                        hand_role_debug.update({
                            f"precommit_control_{name}": value
                            for name, value
                            in current_control_belief.as_dict().items()
                        })
                        current_commitment = (
                            edit_commitment_controller(
                                belief=current_control_belief,
                                debug=hand_role_debug,
                                hand_mask=hand_only_mask[
                                    :, role_left:role_right
                                ],
                                source_features=source_query_features,
                            )
                        )
                        current_control_belief = (
                            current_commitment.belief
                        )
                        hand_role_debug.update(
                            current_commitment.as_debug_maps()
                        )
                        commitment_edit_tokens = (
                            current_commitment.edit_support.reshape(
                                batch_size,
                                -1,
                            )
                        )
                        role_edit_tokens = (
                            role_edit_tokens
                        )
                        if not soft_region_modulation:
                            inloop_trg_fg_mask = role_edit_tokens
                        src_fg_mask_map = self._mask_reshape(
                            role_edit_tokens
                            if not soft_region_modulation
                            else inloop_trg_fg_mask,
                            size=(
                                current_num_frames,
                                height,
                                width,
                            ),
                        )
                    if target_identity_enabled:
                        identity_observation_belief = (
                            current_control_belief
                        )
                        identity_observation_tokens = (
                            role_edit_tokens.clone()
                        )
                        identity_by_layer = shared_dict_dual.get(
                            "target_identity_support",
                            {},
                        )
                        identity_layers = [
                            identity_by_layer[layer]
                            for layer in identity_memory_layers
                            if layer in identity_by_layer
                        ]
                        if identity_layers:
                            current_identity_support = torch.stack(
                                identity_layers,
                                dim=0,
                            ).mean(dim=0).reshape_as(
                                hand_role_debug[
                                    "object_posterior"
                                ]
                            )
                        else:
                            current_identity_support = torch.zeros_like(
                                hand_role_debug[
                                    "object_posterior"
                                ]
                            )
                        hand_role_debug[
                            "identity_read_support"
                        ] = current_identity_support
                        identity_diagnostics_by_layer = (
                            shared_dict_dual.get(
                                "target_identity_diagnostics",
                                {},
                            )
                        )
                        for diagnostic_name in (
                            "best_similarity",
                            "prototype_assignment_entropy",
                            "prototype_assignment_peak",
                            "prototype_assignment_margin",
                            "absolute_match",
                            "relative_match",
                            "support_before_mask",
                            "support_after_mask",
                            "support_mask",
                            "correction_ratio",
                            "appearance_subspace_coherence",
                            "appearance_subspace_action",
                        ):
                            diagnostic_layers = [
                                identity_diagnostics_by_layer[layer][
                                    diagnostic_name
                                ]
                                for layer in identity_memory_layers
                                if (
                                    layer in identity_diagnostics_by_layer
                                    and diagnostic_name
                                    in identity_diagnostics_by_layer[layer]
                                )
                            ]
                            if diagnostic_layers:
                                hand_role_debug[
                                    f"identity_read_{diagnostic_name}"
                                ] = torch.stack(
                                    diagnostic_layers, dim=0
                                ).mean(dim=0).reshape_as(
                                    hand_role_debug[
                                        "object_posterior"
                                    ]
                                )
                        hand_role_debug.update({
                            f"preidentity_control_{name}": value
                            for name, value
                            in current_control_belief.as_dict().items()
                        })
                        current_control_belief = (
                            strengthen_belief_with_target_identity(
                                belief=current_control_belief,
                                identity_support=(
                                    current_identity_support
                                ),
                                hand_mask=hand_only_mask[
                                    :, role_left:role_right
                                ],
                            )
                        )
                        identity_flat = (
                            current_identity_support.flatten(2)
                        )
                        identity_threshold = torch.quantile(
                            identity_flat,
                            0.90,
                            dim=-1,
                            keepdim=True,
                        )
                        identity_edit_tokens = (
                            identity_flat >= identity_threshold
                        ) & (identity_flat > 0)
                        role_edit_tokens = (
                            role_edit_tokens
                            | identity_edit_tokens.reshape(
                                batch_size,
                                -1,
                            )
                        )
                        if not soft_region_modulation:
                            inloop_trg_fg_mask = role_edit_tokens
                        src_fg_mask_map = self._mask_reshape(
                            role_edit_tokens
                            if not soft_region_modulation
                            else inloop_trg_fg_mask,
                            size=(
                                current_num_frames,
                                height,
                                width,
                            ),
                        )
                    hand_role_debug.update({
                        f"control_{name}": value
                        for name, value
                        in current_control_belief.as_dict().items()
                    })
                    if belief_memory_enabled:
                        current_belief_kv_weights = (
                            build_belief_kv_weights(
                                current_control_belief,
                                expected_token_length=(
                                    role_edit_tokens.shape[1]
                                ),
                            )
                        )
                        if memory_consolidation_enabled:
                            current_memory_plan = memory_consolidator(
                                belief=current_control_belief,
                                weights=current_belief_kv_weights,
                                source_features=source_query_features,
                            )
                        if (
                            target_identity_tokenprop_enabled
                            and current_memory_plan is not None
                        ):
                            memory_edit_map = (
                                current_memory_plan
                                .materialized_edit_action.reshape_as(
                                    current_belief_kv_weights.edit_map
                                )
                            )
                            memory_precision_map = (
                                current_memory_plan
                                .consolidated_precision.reshape_as(
                                    current_belief_kv_weights.edit_map
                                )
                            )
                            committed_token_edit = memory_edit_map
                            committed_token_precision = memory_precision_map
                            if current_commitment is not None:
                                stable_commitment = (
                                    current_commitment
                                    .commitment.float()
                                )
                                committed_token_edit = torch.maximum(
                                    committed_token_edit,
                                    (
                                        stable_commitment
                                        * memory_precision_map
                                    ).clamp(0.0, 1.0),
                                )
                                committed_token_precision = torch.maximum(
                                    committed_token_precision,
                                    current_commitment
                                    .state_precision.float(),
                                )
                            if identity_support_filter is None:
                                raise RuntimeError(
                                    "Missing connected identity support filter"
                                )
                            raw_identity_support = torch.maximum(
                                committed_token_edit,
                                current_belief_kv_weights.edit_map,
                            ).clamp(0.0, 1.0)
                            object_posterior = hand_role_debug[
                                "object_posterior"
                            ].float()
                            posterior_threshold = hand_role_debug[
                                "posterior_threshold"
                            ].float()
                            posterior_likelihood = (
                                (object_posterior >= posterior_threshold)
                                & (object_posterior > 0)
                            )
                            identity_flat = (
                                current_identity_support.flatten(2)
                            )
                            identity_threshold = torch.quantile(
                                identity_flat,
                                0.90,
                                dim=-1,
                                keepdim=True,
                            )
                            identity_likelihood = (
                                identity_flat >= identity_threshold
                            ) & (identity_flat > 0)
                            object_likelihood_mask = (
                                posterior_likelihood
                                | identity_likelihood.reshape_as(
                                    object_posterior
                                )
                            ) & (
                                hand_role_debug[
                                    "hand_probability"
                                ].float() < 0.5
                            )
                            current_connected_identity_support = (
                                identity_support_filter(
                                    support_weight=raw_identity_support,
                                    anchor_mask=(
                                        identity_observation_tokens
                                        .reshape_as(raw_identity_support)
                                    ),
                                    object_likelihood_mask=(
                                        object_likelihood_mask
                                    ),
                                )
                            )
                            identity_tokenprop_support_tokens = (
                                current_connected_identity_support
                                .weight
                                .reshape(batch_size, -1)
                            )
                            committed_token_edit = (
                                committed_token_edit
                                * current_connected_identity_support
                                .keep_mask.float()
                            )
                            hand_role_debug.update({
                                "identity_support_raw": (
                                    raw_identity_support
                                ),
                                "identity_support_candidate": (
                                    current_connected_identity_support
                                    .candidate_mask.float()
                                ),
                                "identity_support_connected": (
                                    current_connected_identity_support
                                    .keep_mask.float()
                                ),
                                "identity_support_anchor": (
                                    current_connected_identity_support
                                    .anchor_mask.float()
                                ),
                                "identity_support_object_likelihood": (
                                    current_connected_identity_support
                                    .object_likelihood_mask.float()
                                ),
                                "identity_support_budget": (
                                    current_connected_identity_support
                                    .budget_fraction.expand_as(
                                        raw_identity_support
                                    )
                                ),
                            })
                        if (
                            target_identity_tokenprop_enabled
                            and current_memory_plan is not None
                            and committed_memory_feedback_strength > 0
                        ):
                            (
                                current_control_belief,
                                committed_memory_debug,
                            ) = inject_committed_memory_into_belief(
                                belief=current_control_belief,
                                committed_token_edit=committed_token_edit,
                                committed_token_precision=(
                                    committed_token_precision
                                ),
                                hand_mask=hand_only_mask[
                                    :, role_left:role_right
                                ],
                                feedback_strength=(
                                    committed_memory_feedback_strength
                                ),
                                identity_core_support=torch.maximum(
                                    current_connected_identity_support
                                    .weight,
                                    current_identity_support
                                    * current_connected_identity_support
                                    .keep_mask.float(),
                                ),
                            )
                            hand_role_debug.update(committed_memory_debug)
                            committed_memory_tokens = (
                                identity_tokenprop_support_tokens > 0.05
                            )
                            role_edit_tokens = (
                                role_edit_tokens | committed_memory_tokens
                            )
                            if not soft_region_modulation:
                                inloop_trg_fg_mask = role_edit_tokens
                            src_fg_mask_map = self._mask_reshape(
                                role_edit_tokens
                                if not soft_region_modulation
                                else inloop_trg_fg_mask,
                                size=(
                                    current_num_frames,
                                    height,
                                    width,
                                ),
                            )
                            current_belief_kv_weights = (
                                build_belief_kv_weights(
                                    current_control_belief,
                                    expected_token_length=(
                                        role_edit_tokens.shape[1]
                                    ),
                                )
                            )
                        hand_role_debug.update({
                            "dual_kv_edit_weight": (
                                current_belief_kv_weights.edit_map
                            ),
                            "dual_kv_preserve_weight": (
                                current_belief_kv_weights.preserve_map
                            ),
                            "dual_kv_edit_action": (
                                current_belief_kv_weights.edit_action_map
                            ),
                            "dual_kv_preserve_action": (
                                current_belief_kv_weights
                                .preserve_action_map
                            ),
                            "dual_kv_conflict_weight": (
                                current_belief_kv_weights.conflict_map
                            ),
                        })
                        if current_memory_plan is not None:
                            hand_role_debug.update(
                                current_memory_plan.as_debug_maps(
                                    height=(
                                        current_belief_kv_weights
                                        .edit_map.shape[-2]
                                    ),
                                    width=(
                                        current_belief_kv_weights
                                        .edit_map.shape[-1]
                                    ),
                                )
                            )
                        hand_role_debug.update({
                            f"control_{name}": value
                            for name, value
                            in current_control_belief.as_dict().items()
                        })
                        self._inject_masks_to_kv_cache(
                            kv_cache_dual,
                            trg_fg_mask_cache,
                            role_edit_tokens,
                            belief_kv_weight_cache=(
                                belief_kv_weight_cache
                            ),
                            target_owned_mask_cache=(
                                target_owned_mask_cache
                            ),
                            current_target_owned_mask=(
                                current_target_owned_mask
                                if target_owned_object_handoff
                                else None
                            ),
                            current_identity_read_mask=(
                                current_identity_read_mask
                            ),
                            current_causal_owner_mask=(
                                current_memory_query_weight
                            ),
                        )
                    if (
                        identity_first_latent_bootstrap
                        and target_identity_enabled
                        and not target_identity_memory.export()
                        and current_start_frame == 0
                        and current_num_frames > 1
                    ):
                        if (
                            identity_observation_belief is None
                            or identity_observation_tokens is None
                        ):
                            raise RuntimeError(
                                "Missing independent evidence for causal "
                                "identity bootstrap"
                            )
                        bootstrap_base_write_map = (
                            identity_observation_belief.edit_belief
                            * identity_observation_belief.edit_precision
                            * (
                                1.0
                                - identity_observation_belief.uncertainty
                            )
                            * identity_observation_belief.visibility
                        ).clamp(0.0, 1.0)
                        bootstrap_base_write_tokens = F.avg_pool2d(
                            bootstrap_base_write_map.reshape(
                                batch_size * current_num_frames,
                                1,
                                height,
                                width,
                            ),
                            kernel_size=2,
                            stride=2,
                        ).reshape_as(
                            hand_role_debug["object_posterior"]
                        )
                        bootstrap_object_likelihood = hand_role_debug[
                            "object_posterior"
                        ]
                        bootstrap_object_threshold = hand_role_debug[
                            "posterior_threshold"
                        ]
                        if current_target_change_core is not None:
                            bootstrap_object_likelihood = (
                                current_identity_write_mask.reshape_as(
                                    bootstrap_object_likelihood
                                ).float()
                            )
                            bootstrap_object_threshold = (
                                bootstrap_object_likelihood.new_full(
                                    (batch_size, current_num_frames, 1, 1),
                                    0.5,
                                )
                            )
                        current_causal_identity_bootstrap_plan = (
                            build_first_frame_object_core_bootstrap(
                                base_write_weight=(
                                    bootstrap_base_write_tokens
                                ),
                                object_likelihood=(
                                    bootstrap_object_likelihood
                                ),
                                object_threshold=(
                                    bootstrap_object_threshold
                                ),
                                hand_probability=hand_role_debug[
                                    "hand_probability"
                                ],
                            )
                        )
                        hand_role_debug.update(
                            current_causal_identity_bootstrap_plan
                            .as_debug_maps()
                        )
                    # #region debug-point H1-H3:commitment-to-belief
                    if current_commitment is not None:
                        _debug_pre_edit = (
                            hand_role_debug[
                                "precommit_control_edit_belief"
                            ]
                            * hand_role_debug[
                                "precommit_control_edit_precision"
                            ]
                        )
                        _debug_pre_preserve = (
                            hand_role_debug[
                                "precommit_control_preserve_belief"
                            ]
                            * hand_role_debug[
                                "precommit_control_preserve_precision"
                            ]
                        )
                        _debug_post_edit = (
                            current_control_belief.edit_belief
                            * current_control_belief.edit_precision
                        )
                        _debug_post_preserve = (
                            current_control_belief.preserve_belief
                            * current_control_belief.preserve_precision
                        )
                        _debug_effective = F.interpolate(
                            current_commitment.effective_commitment.reshape(
                                batch_size * current_num_frames,
                                1,
                                *current_commitment
                                .effective_commitment.shape[-2:],
                            ),
                            size=_debug_post_edit.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ).reshape_as(_debug_post_edit)
                        _debug_active = _debug_effective > 0.05
                        _debug_active_count = (
                            _debug_active.sum().clamp_min(1)
                        )
                        _debug_pre_action = (
                            _debug_pre_preserve
                            / (
                                _debug_pre_edit
                                + _debug_pre_preserve
                            ).clamp_min(1e-6)
                        )
                        _debug_post_action = (
                            _debug_post_preserve
                            / (
                                _debug_post_edit
                                + _debug_post_preserve
                            ).clamp_min(1e-6)
                        )
                        _debug_report(
                            "H1-H3",
                            "edit_causal_inference.py:commitment-belief",
                            "Commitment effect on final control belief",
                            {
                                "block": (
                                    current_start_frame
                                    // self.num_frame_per_block
                                ),
                                "active_coverage": (
                                    _debug_active.float().mean().item()
                                ),
                                "effective_on_active": (
                                    _debug_effective[
                                        _debug_active
                                    ].sum()
                                    / _debug_active_count
                                ).item(),
                                "pre_preserve_action_active": (
                                    _debug_pre_action[
                                        _debug_active
                                    ].sum()
                                    / _debug_active_count
                                ).item(),
                                "post_preserve_action_active": (
                                    _debug_post_action[
                                        _debug_active
                                    ].sum()
                                    / _debug_active_count
                                ).item(),
                                "post_edit_action_active": (
                                    1.0
                                    - _debug_post_action[
                                        _debug_active
                                    ].sum()
                                    / _debug_active_count
                                ).item(),
                                "post_preserve_action_global": (
                                    _debug_post_action.mean().item()
                                ),
                            },
                        )
                    # #endregion
                    print(
                        "CAUSAL_CONTROL_BELIEF "
                        "block="
                        f"{current_start_frame // self.num_frame_per_block} "
                        "edit="
                        f"{current_control_belief.edit_belief.mean().item():.4f} "
                        "preserve="
                        f"{current_control_belief.preserve_belief.mean().item():.4f} "
                        "edit_precision="
                        f"{current_control_belief.edit_precision.mean().item():.4f} "
                        "preserve_precision="
                        f"{current_control_belief.preserve_precision.mean().item():.4f} "
                        "conflict="
                        f"{current_control_belief.conflict.mean().item():.4f} "
                        "uncertainty="
                        f"{current_control_belief.uncertainty.mean().item():.4f}"
                    )
                    if current_commitment is not None:
                        print(
                            "EDIT_COMMITMENT "
                            "block="
                            f"{current_start_frame // self.num_frame_per_block} "
                            "trigger="
                            f"{current_commitment.trigger.mean().item():.4f} "
                            "transport="
                            f"{current_commitment.transported.mean().item():.4f} "
                            "anchor="
                            f"{current_commitment.anchor_transport.mean().item():.4f} "
                            "presence="
                            f"{current_commitment.semantic_presence.mean().item():.4f} "
                            "absence="
                            f"{current_commitment.semantic_absence.mean().item():.4f} "
                            "precision="
                            f"{current_commitment.commitment_precision.mean().item():.4f} "
                            "state_precision="
                            f"{current_commitment.state_precision.mean().item():.4f} "
                            "effective="
                            f"{current_commitment.effective_commitment.mean().item():.4f} "
                            "edit_support="
                            f"{current_commitment.edit_support.float().mean().item():.4f}"
                        )
                        if reference_identity_enabled:
                            print(
                                "REFERENCE_LOCAL_TRANSPORT "
                                "block="
                                f"{current_start_frame // self.num_frame_per_block} "
                                "radius="
                                f"{edit_commitment_controller.last_spatial_radius} "
                                "budget="
                                f"{edit_commitment_controller.reference_support_budget.float().mean().item():.1f} "
                                "precision_scale="
                                f"{edit_commitment_controller.last_reference_precision_scale.mean().item():.4f} "
                                "mode=previous_only_local_splat"
                            )
                    if current_identity_support is not None:
                        print(
                            "TARGET_IDENTITY_READ "
                            "block="
                            f"{current_start_frame // self.num_frame_per_block} "
                            "support="
                            f"{current_identity_support.mean().item():.4f} "
                            "peak="
                            f"{current_identity_support.max().item():.4f} "
                            "edit_tokens="
                            f"{identity_edit_tokens.float().mean().item():.4f}"
                        )
                    if aligned_belief_kv_enabled:
                        print(
                            "BELIEF_DUAL_KV "
                            "block="
                            f"{current_start_frame // self.num_frame_per_block} "
                            "target="
                            f"{current_belief_kv_weights.edit.mean().item():.4f} "
                            "source="
                            f"{current_belief_kv_weights.preserve.mean().item():.4f} "
                            "edit_action="
                            f"{current_belief_kv_weights.edit_action.mean().item():.4f} "
                            "cache_preserve="
                            f"{current_belief_kv_weights.preserve_action.mean().item():.4f} "
                            "conflict="
                            f"{current_belief_kv_weights.conflict_map.mean().item():.4f}"
                        )
                    elif memory_consolidation_enabled:
                        print(
                            "BELIEF_MEMORY_OBSERVATION "
                            "block="
                            f"{current_start_frame // self.num_frame_per_block} "
                            "edit_action="
                            f"{current_belief_kv_weights.edit_action.mean().item():.4f} "
                            "preserve_action="
                            f"{current_belief_kv_weights.preserve_action.mean().item():.4f} "
                            "conflict="
                            f"{current_belief_kv_weights.conflict_map.mean().item():.4f}"
                        )
                    if current_memory_plan is not None:
                        print(
                            "CAUSAL_MEMORY_WRITE "
                            "block="
                            f"{current_start_frame // self.num_frame_per_block} "
                            "observation_precision="
                            f"{current_memory_plan.observation_precision.mean().item():.4f} "
                            "transport_precision="
                            f"{current_memory_plan.transported_precision.mean().item():.4f} "
                            "observation_gain="
                            f"{current_memory_plan.observation_gain.mean().item():.4f} "
                            "consolidated_edit="
                            f"{current_memory_plan.consolidated_edit_action.mean().item():.4f} "
                            "consolidated_precision="
                            f"{current_memory_plan.consolidated_precision.mean().item():.4f} "
                            "materialized_edit="
                            f"{current_memory_plan.materialized_edit_action.mean().item():.4f}"
                        )
                    if current_connected_identity_support is not None:
                        print(
                            "CONNECTED_IDENTITY_SUPPORT "
                            "block="
                            f"{current_start_frame // self.num_frame_per_block} "
                            "candidate="
                            f"{current_connected_identity_support.candidate_mask.float().mean().item():.4f} "
                            "kept="
                            f"{current_connected_identity_support.keep_mask.float().mean().item():.4f} "
                            "anchor="
                            f"{current_connected_identity_support.anchor_mask.float().mean().item():.4f} "
                            "likelihood="
                            f"{current_connected_identity_support.object_likelihood_mask.float().mean().item():.4f} "
                            "budget="
                            f"{current_connected_identity_support.budget_fraction.mean().item():.4f}"
                        )
                    if save_role_dir is not None:
                        self._save_hand_role_debug(
                            save_role_dir,
                            current_start_frame
                            // self.num_frame_per_block,
                            hand_role_debug,
                        )
                # #region debug-point B:velocity-collapse
                if current_roles is not None and index in {
                    0,
                    len(denoising_step_list) // 2,
                    len(denoising_step_list) - 1,
                }:
                    debug_edit_mask = current_roles.edit_weight.bool().unsqueeze(2)
                    debug_edit_mask = debug_edit_mask.expand_as(v_trg)
                    debug_edit_count = debug_edit_mask.sum().clamp_min(1)
                    _debug_report(
                        "B",
                        "edit_causal_inference.py:denoising-velocity",
                        "Target velocity separation inside oracle edit roles",
                        {
                            "block": current_start_frame // self.num_frame_per_block,
                            "step": index,
                            "timestep": float(current_timestep),
                            "target_source_gap": (
                                (v_trg - v_src).abs()[debug_edit_mask].sum()
                                / debug_edit_count
                            ).item(),
                            "target_exact_source_gap": (
                                (v_trg - v_gt).abs()[debug_edit_mask].sum()
                                / debug_edit_count
                            ).item(),
                            "target_velocity_abs": (
                                v_trg.abs()[debug_edit_mask].sum()
                                / debug_edit_count
                            ).item(),
                        },
                    )
                # #endregion

                if factorized_bayes_enabled:
                    if current_factorized_operators is None:
                        raise RuntimeError(
                            "Missing operators for factorized Bayes routing"
                        )
                    native_change = (v_trg - v_src).abs().mean(
                        dim=2, keepdim=True
                    )
                    native_dims = list(range(native_change.ndim))[1:]
                    native_change = (
                        native_change
                        - native_change.amin(
                            dim=native_dims, keepdim=True
                        )
                    ) / (
                        native_change.amax(
                            dim=native_dims, keepdim=True
                        )
                        - native_change.amin(
                            dim=native_dims, keepdim=True
                        )
                        + 1e-7
                    )
                    native_background_action = 1.0 - native_change
                    if soft_region_modulation:
                        bg_mask = native_background_action
                        raw_posterior = hand_role_debug.get(
                            "object_posterior_pre_source_flow",
                            hand_role_debug.get("object_posterior"),
                        )
                        flow_boost = hand_role_debug.get(
                            "source_flow_verified_support",
                        )
                        hand_excl = hand_role_debug.get(
                            "hand_hard_exclusion",
                        )
                        if raw_posterior is not None:
                            region_posterior = raw_posterior.float(
                            ).clamp(0.0, 1.0)
                            if flow_boost is not None:
                                flow_support = flow_boost.float(
                                ).clamp(0.0, 1.0)
                                region_posterior = torch.maximum(
                                    region_posterior, flow_support
                                )
                            if hand_excl is not None:
                                region_posterior = region_posterior * (
                                    1.0 - hand_excl.float().clamp(
                                        0.0, 1.0
                                    )
                                )
                            region_confidence = F.interpolate(
                                region_posterior.float()
                                .reshape(
                                    batch_size * current_num_frames,
                                    1,
                                    *region_posterior.shape[-2:],
                                ),
                                size=v_trg.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            ).reshape(
                                batch_size,
                                current_num_frames,
                                1,
                                *v_trg.shape[-2:],
                            ).clamp(0.0, 1.0)
                            source_suppression = (
                                soft_region_blend_strength
                                * region_confidence
                            )
                        else:
                            source_suppression = torch.zeros_like(bg_mask)
                        source_residual = (v_gt - v_src)
                        v_t = (
                            v_trg
                            + bg_mask
                            * (1.0 - source_suppression)
                            * source_residual
                        ).to(v_trg.dtype)
                        factorized_flow_debug = {
                            "source_residual_action": bg_mask,
                            "unknown_action": torch.zeros_like(bg_mask),
                            "native_fallback_action": native_background_action,
                            "effective_source_residual_action": (
                                bg_mask * (1.0 - source_suppression)
                            ),
                            "source_suppression": source_suppression,
                            "paired_memory_source_suppression_action": torch.zeros_like(bg_mask),
                            "verified_native_history_source_suppression_action": torch.zeros_like(bg_mask),
                            "orthogonal_geometry_action": torch.zeros_like(bg_mask),
                            "owner_complement_source_action": torch.zeros_like(bg_mask),
                            "owner_complement_abstain_action": torch.zeros_like(bg_mask),
                        }
                        if factorized_native_target_history:
                            factorized_flow_debug[
                                "target_owned_native_fallback_action"
                            ] = torch.zeros_like(bg_mask)
                        if index == 0:
                            region_coverage = (
                                region_confidence.mean().item()
                                if region_posterior is not None
                                else 0.0
                            )
                            suppression_mean = (
                                source_suppression.mean().item()
                            )
                            effective_residual = (
                                bg_mask * (1.0 - source_suppression)
                            ).mean().item()
                            print(
                                "SOFT_REGION_MODULATION "
                                f"block={current_start_frame // self.num_frame_per_block} "
                                f"blend_strength={soft_region_blend_strength:.2f} "
                                f"region_coverage={region_coverage:.4f} "
                                f"bg_mask_mean={bg_mask.mean().item():.4f} "
                                f"source_suppression={suppression_mean:.4f} "
                                f"effective_residual={effective_residual:.4f}"
                            )
                        if soft_region_modulation and region_posterior is not None:
                            blender_rate_scalar = (
                                1.0
                                - float(timestep_next)
                                ** blend_power
                            )
                            region_flat = region_posterior.reshape(
                                batch_size, -1
                            ).clamp(0.0, 1.0)
                            spatial_blender = (
                                blender_rate_scalar
                                + (1.0 - blender_rate_scalar)
                                * soft_region_blend_strength
                                * region_flat
                            ).clamp(0.0, 1.0)
                            shared_dict_dual[
                                "spatial_blender_rate"
                            ] = spatial_blender
                        elif soft_region_modulation:
                            shared_dict_dual.pop(
                                "spatial_blender_rate", None
                            )
                    else:
                        v_t, factorized_flow_debug = (
                            route_factorized_velocity(
                                target_velocity=v_trg,
                                source_velocity=v_src,
                                source_reconstruction_velocity=v_gt,
                                operators=current_factorized_operators,
                                native_fallback_action=(
                                    native_background_action
                                ),
                            target_owned_weight=(
                                (
                                    current_edit_authority.support.float()
                                    if current_edit_authority is not None
                                    else current_causal_ownership.owner_support
                                    .reshape_as(
                                        hand_role_debug["object_posterior"]
                                    )
                                )
                                if (
                                    factorized_native_target_history
                                    and current_causal_ownership is not None
                                )
                                else None
                            ),
                            block_target_owned_source=(
                                factorized_owner_source_block
                                or not factorized_native_target_history
                                or target_semantic_competition
                            ),
                            geometry_owner_weight=(
                                current_identity_read_mask.reshape_as(
                                    hand_role_debug["object_posterior"]
                                )
                                if factorized_orthogonal_geometry
                                else None
                            ),
                            geometry_strength=(
                                factorized_geometry_strength
                                if factorized_orthogonal_geometry
                                else 0.0
                            ),
                            denoising_fraction=(
                                1.0
                                - float(index)
                                / max(len(denoising_step_list) - 1, 1)
                            ),
                            source_coordinate_target_delta=(
                                factorized_source_coordinate_target_delta
                            ),
                            owner_complement_source_weight=(
                                current_causal_ownership.owner_support
                                .reshape_as(
                                    hand_role_debug["object_posterior"]
                                )
                                if (
                                    factorized_owner_complement_source
                                    and current_causal_ownership is not None
                                )
                                else None
                            ),
                            owner_complement_margin=(
                                factorized_owner_complement_margin
                            ),
                            owner_complement_min_preserve_confidence=(
                                factorized_owner_complement_min_preserve_confidence
                            ),
                            paired_memory_support_weight=(
                                current_paired_memory_support
                                if (
                                    causal_paired_edit_memory
                                    and paired_memory_source_suppression > 0.0
                                )
                                else None
                            ),
                            paired_memory_source_suppression=(
                                paired_memory_source_suppression
                            ),
                            verified_native_history_support_weight=(
                                current_verified_native_history_support
                                if native_history_transactional_owner
                                else None
                            ),
                            verified_native_history_source_suppression=(
                                native_history_verified_source_suppression
                                if native_history_transactional_owner
                                else 0.0
                            ),
                            verified_native_history_appearance_projection=(
                                native_history_consistent_transaction
                            ),
                            edit_authority_weight=(
                                current_edit_authority.support.float()
                                if current_edit_authority is not None
                                else None
                            ),
                        )
                    )
                    if index == 0:
                        hand_role_debug.update({
                            f"factorized_flow_{name}": value.squeeze(2)
                            for name, value in (
                                factorized_flow_debug.items()
                            )
                        })
                        print(
                            "FACTORIZED_FLOW "
                            f"block={current_start_frame // self.num_frame_per_block} "
                            "source_residual="
                            f"{factorized_flow_debug['source_residual_action'].mean().item():.4f} "
                            "unknown="
                            f"{factorized_flow_debug['unknown_action'].mean().item():.4f} "
                            "native_fallback="
                            f"{factorized_flow_debug['native_fallback_action'].mean().item():.4f} "
                            "effective="
                            f"{factorized_flow_debug['effective_source_residual_action'].mean().item():.4f} "
                            "paired_suppression="
                            f"{factorized_flow_debug['paired_memory_source_suppression_action'].mean().item():.4f} "
                            "verified_native_suppression="
                            f"{factorized_flow_debug['verified_native_history_source_suppression_action'].mean().item():.4f} "
                            "orthogonal_geometry="
                            f"{factorized_flow_debug['orthogonal_geometry_action'].mean().item():.4f}"
                            + (
                                " owner_source_fallback="
                                f"{factorized_flow_debug['target_owned_native_fallback_action'].mean().item():.4f}"
                                if factorized_native_target_history
                                else ""
                            )
                        )
                        if factorized_source_coordinate_target_delta:
                            target_delta_action = factorized_flow_debug[
                                "source_coordinate_target_delta_action"
                            ]
                            print(
                                "SOURCE_COORDINATE_TARGET_DELTA "
                                f"block="
                                f"{current_start_frame // self.num_frame_per_block} "
                                "base=clean_source_reconstruction "
                                "edit=target_minus_source "
                                f"coverage="
                                f"{target_delta_action.mean().item():.4f} "
                                "background_delta=0"
                            )
                        if factorized_owner_complement_source:
                            complement_action = factorized_flow_debug[
                                "owner_complement_source_action"
                            ]
                            abstain_action = factorized_flow_debug[
                                "owner_complement_abstain_action"
                            ]
                            print(
                                "OWNER_COMPLEMENT_SOURCE_FLOW "
                                f"block={current_start_frame // self.num_frame_per_block} "
                                "background=clean_source_reconstruction "
                                "owner=role_fixed_native_kv "
                                f"margin={factorized_owner_complement_margin} "
                                "min_preserve_confidence="
                                f"{factorized_owner_complement_min_preserve_confidence:.3f} "
                                f"source_coverage={complement_action.mean().item():.4f} "
                                f"abstain_coverage={abstain_action.mean().item():.4f}"
                            )
                        if (
                            factorized_native_target_history
                            and current_causal_ownership is not None
                        ):
                            owner_velocity = F.interpolate(
                                current_causal_ownership.owner_support
                                .reshape_as(
                                    hand_role_debug["object_posterior"]
                                ).float().flatten(0, 1).unsqueeze(1),
                                size=v_trg.shape[-2:],
                                mode="nearest",
                            ).reshape(
                                batch_size,
                                current_num_frames,
                                1,
                                *v_trg.shape[-2:],
                            ) > 0.0
                            owner_count = owner_velocity.sum().clamp_min(1)
                            owner_source_fallback = factorized_flow_debug[
                                "target_owned_native_fallback_action"
                            ]
                            print(
                                "OWNER_SOURCE_APPEARANCE "
                                f"block={current_start_frame // self.num_frame_per_block} "
                                f"blocked={int(factorized_owner_source_block)} "
                                "fallback="
                                f"{owner_source_fallback[owner_velocity].sum().div(owner_count).item():.4f}"
                            )
                        if factorized_orthogonal_geometry:
                            owner_velocity = F.interpolate(
                                current_identity_read_mask.reshape_as(
                                    hand_role_debug["object_posterior"]
                                ).float().flatten(0, 1).unsqueeze(1),
                                size=v_trg.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            ).reshape(
                                batch_size,
                                current_num_frames,
                                1,
                                *v_trg.shape[-2:],
                            ) > 0.0
                            owner_count = owner_velocity.sum().clamp_min(1)
                            geometry_action = factorized_flow_debug[
                                "orthogonal_geometry_action"
                            ]
                            print(
                                "ORTHOGONAL_GEOMETRY_FLOW "
                                f"block={current_start_frame // self.num_frame_per_block} "
                                f"step={index} "
                                "owner_action="
                                f"{geometry_action[owner_velocity].sum().div(owner_count).item():.4f} "
                                "residual="
                                f"{factorized_flow_debug['orthogonal_geometry_residual_abs'][owner_velocity].sum().div(owner_count).item():.4f}"
                            )
                        if save_role_dir is not None:
                            self._save_hand_role_debug(
                                save_role_dir,
                                current_start_frame
                                // self.num_frame_per_block,
                                hand_role_debug,
                            )
                elif bayes_flow_enabled:
                    if current_control_belief is None:
                        raise RuntimeError(
                            "Missing causal control belief for Bayes routing"
                        )
                    v_t, bayes_flow_debug = bayes_residual_flow_router(
                        target_velocity=v_trg,
                        source_velocity=v_src,
                        source_reconstruction_velocity=v_gt,
                        belief=current_control_belief,
                        target_owned_mask=(
                            current_target_owned_mask.reshape_as(
                                hand_role_debug["object_posterior"]
                            )
                            if (
                                target_owned_object_handoff
                                and current_target_owned_mask is not None
                            )
                            else None
                        ),
                        target_change_core=(
                            (
                                current_target_change_core.mask
                                & self._mask_reshape(
                                    current_identity_read_mask > 0.05,
                                    size=(
                                        current_num_frames,
                                        height,
                                        width,
                                    ),
                                )
                                if (
                                    source_coordinate_identity
                                    and current_identity_read_mask is not None
                                )
                                else current_target_change_core.mask
                            )
                            if current_target_change_core is not None
                            else None
                        ),
                        protect_mask=(
                            hand_only_mask[:, role_left:role_right]
                            if current_target_change_core is not None
                            else None
                        ),
                        identity_owner_weight=(
                            current_identity_read_mask.reshape_as(
                                hand_role_debug[
                                    "object_posterior"
                                ]
                            )
                            if (
                                source_coordinate_identity
                                and current_identity_read_mask is not None
                            )
                            else None
                        ),
                        identity_source_suppression=(
                            identity_source_suppression
                        ),
                        denoising_fraction=(
                            1.0
                            - float(index)
                            / max(len(denoising_step_list) - 1, 1)
                        ),
                    )
                    if index == 0:
                        # #region debug-point H2-H4:velocity-routing
                        if current_commitment is not None:
                            _debug_action = bayes_flow_debug[
                                "preserve_action_weight"
                            ].float()
                            _debug_effective = F.interpolate(
                                current_commitment
                                .effective_commitment.reshape(
                                    batch_size * current_num_frames,
                                    1,
                                    *current_commitment
                                    .effective_commitment.shape[-2:],
                                ),
                                size=v_trg.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            ).reshape(
                                batch_size,
                                current_num_frames,
                                1,
                                *v_trg.shape[-2:],
                            )
                            _debug_active = _debug_effective > 0.05
                            _debug_active_channels = (
                                _debug_active.expand_as(v_trg)
                            )
                            _debug_map_count = (
                                _debug_active.sum().clamp_min(1)
                            )
                            _debug_value_count = (
                                _debug_active_channels.sum().clamp_min(1)
                            )
                            _debug_residual = (
                                v_gt.float() - v_src.float()
                            )
                            _debug_contribution = (
                                _debug_action * _debug_residual
                            )
                            hand_role_debug.update({
                                "velocity_target_abs": (
                                    v_trg.float().abs().mean(dim=2)
                                ),
                                "velocity_source_residual_abs": (
                                    _debug_residual.abs().mean(dim=2)
                                ),
                                "velocity_source_contribution_abs": (
                                    _debug_contribution.abs().mean(dim=2)
                                ),
                                "velocity_routed_abs": (
                                    v_t.float().abs().mean(dim=2)
                                ),
                                "velocity_target_source_gap_abs": (
                                    (v_trg.float() - v_src.float())
                                    .abs().mean(dim=2)
                                ),
                                "velocity_target_exact_source_gap_abs": (
                                    (v_trg.float() - v_gt.float())
                                    .abs().mean(dim=2)
                                ),
                            })
                            _debug_target_abs = (
                                v_trg.float().abs()[
                                    _debug_active_channels
                                ].sum()
                                / _debug_value_count
                            )
                            _debug_contribution_abs = (
                                _debug_contribution.abs()[
                                    _debug_active_channels
                                ].sum()
                                / _debug_value_count
                            )
                            _debug_report(
                                "H2-H4",
                                "edit_causal_inference.py:bayes-router",
                                "Final action and velocity terms on commitment",
                                {
                                    "block": (
                                        current_start_frame
                                        // self.num_frame_per_block
                                    ),
                                    "preserve_action_active": (
                                        _debug_action[
                                            _debug_active
                                        ].sum()
                                        / _debug_map_count
                                    ).item(),
                                    "preserve_action_global": (
                                        _debug_action.mean().item()
                                    ),
                                    "target_velocity_abs": (
                                        _debug_target_abs.item()
                                    ),
                                    "source_residual_abs": (
                                        _debug_residual.abs()[
                                            _debug_active_channels
                                        ].sum()
                                        / _debug_value_count
                                    ).item(),
                                    "source_contribution_abs": (
                                        _debug_contribution_abs.item()
                                    ),
                                    "contribution_target_ratio": (
                                        _debug_contribution_abs
                                        / _debug_target_abs.clamp_min(
                                            1e-6
                                        )
                                    ).item(),
                                    "routed_delta_error": (
                                        (
                                            v_t.float()
                                            - v_trg.float()
                                            - _debug_contribution
                                        ).abs().max().item()
                                    ),
                                },
                            )
                        # #endregion
                        hand_role_debug.update({
                            f"bayes_{name}": value.squeeze(2)
                            for name, value in bayes_flow_debug.items()
                        })
                        print(
                            "BAYES_FLOW "
                            "block="
                            f"{current_start_frame // self.num_frame_per_block} "
                            "edit_action="
                            f"{bayes_flow_debug['edit_action_weight'].mean().item():.4f} "
                            "preserve_action="
                            f"{bayes_flow_debug['preserve_action_weight'].mean().item():.4f} "
                            "sum_error="
                            f"{bayes_flow_debug['action_sum_error'].max().item():.2e} "
                            "no_evidence="
                            f"{bayes_flow_debug['no_evidence'].mean().item():.4f}"
                        )
                        if current_target_change_core is not None:
                            leakage_core = bayes_flow_debug[
                                "appearance_leakage_core"
                            ]
                            source_energy = bayes_flow_debug[
                                "appearance_leakage_source_energy"
                            ]
                            removed_energy = bayes_flow_debug[
                                "appearance_leakage_removed_energy"
                            ]
                            preserved_energy = bayes_flow_debug[
                                "appearance_leakage_preserved_energy"
                            ]
                            core_source_energy = (
                                source_energy * leakage_core
                            ).sum().clamp_min(1e-6)
                            print(
                                "APPEARANCE_LEAKAGE_DECOMPOSITION "
                                f"block="
                                f"{current_start_frame // self.num_frame_per_block} "
                                f"core={leakage_core.mean().item():.4f} "
                                f"removed_energy_ratio="
                                f"{((removed_energy * leakage_core).sum() / core_source_energy).item():.4f} "
                                f"preserved_energy_ratio="
                                f"{((preserved_energy * leakage_core).sum() / core_source_energy).item():.4f}"
                            )
                        if save_role_dir is not None:
                            self._save_hand_role_debug(
                                save_role_dir,
                                current_start_frame
                                // self.num_frame_per_block,
                                hand_role_debug,
                            )
                elif posterior_flow_enabled:
                    v_t, posterior_flow_debug = (
                        posterior_residual_flow_router(
                            target_velocity=v_trg,
                            source_velocity=v_src,
                            source_reconstruction_velocity=v_gt,
                            roles=current_roles,
                            hard_roles=(
                                posterior_flow_mode == "hard"
                            ),
                        )
                    )
                    if index == 0:
                        role_probabilities = posterior_flow_debug[
                            "role_probabilities"
                        ]
                        hand_role_debug.update({
                            "flow_target_expert_weight": (
                                posterior_flow_debug[
                                    "target_expert_weight"
                                ].squeeze(2)
                            ),
                            "flow_residual_expert_weight": (
                                posterior_flow_debug[
                                    "residual_expert_weight"
                                ].squeeze(2)
                            ),
                            "flow_contact_target_weight": (
                                posterior_flow_debug[
                                    "contact_target_weight"
                                ].squeeze(2)
                            ),
                            "flow_contact_residual_weight": (
                                posterior_flow_debug[
                                    "contact_residual_weight"
                                ].squeeze(2)
                            ),
                            "flow_role_entropy": (
                                posterior_flow_debug[
                                    "role_entropy"
                                ].squeeze(2)
                            ),
                            "flow_object_probability": (
                                role_probabilities[:, :, 0]
                            ),
                            "flow_contact_probability": (
                                role_probabilities[:, :, 1]
                            ),
                            "flow_hand_probability": (
                                role_probabilities[:, :, 2]
                            ),
                            "flow_background_probability": (
                                role_probabilities[:, :, 3]
                            ),
                        })
                        expert_sum_error = (
                            posterior_flow_debug[
                                "target_expert_weight"
                            ]
                            + posterior_flow_debug[
                                "residual_expert_weight"
                            ]
                            - 1.0
                        ).abs().max()
                        contact_mass = role_probabilities[
                            :, :, 1:2
                        ].float()
                        contact_target_mean = (
                            posterior_flow_debug[
                                "contact_target_weight"
                            ].float()
                            * contact_mass
                        ).sum() / contact_mass.sum().clamp_min(1e-6)
                        print(
                            "POSTERIOR_FLOW "
                            f"block={current_start_frame // self.num_frame_per_block} "
                            f"mode={posterior_flow_mode} "
                            "target="
                            f"{posterior_flow_debug['target_expert_weight'].mean().item():.4f} "
                            "residual="
                            f"{posterior_flow_debug['residual_expert_weight'].mean().item():.4f} "
                            "contact_target="
                            f"{contact_target_mean.item():.4f} "
                            "entropy="
                            f"{posterior_flow_debug['role_entropy'].mean().item():.4f} "
                            f"sum_error={expert_sum_error.item():.2e}"
                        )
                        if save_role_dir is not None:
                            self._save_hand_role_debug(
                                save_role_dir,
                                current_start_frame
                                // self.num_frame_per_block,
                                hand_role_debug,
                            )
                elif routing_mode in {
                    "oracle_role_residual",
                    "oracle_role_residual_kv",
                    "hand_role_residual_kv",
                    "hand_role_adaptive_kv",
                }:
                    v_t, _ = residual_role_flow_router(
                        target_velocity=v_trg,
                        source_velocity=v_src,
                        source_reconstruction_velocity=v_gt,
                        roles=current_roles,
                        contact_target_weight=contact_target_weight,
                    )
                elif oracle_role_enabled:
                    v_t, _, _ = role_flow_router(
                        target_velocity=v_trg,
                        source_reconstruction_velocity=v_gt,
                        roles=current_roles,
                    )
                else:
                    # Original source-oriented guidance.
                    fg_mask = (v_trg - v_src).abs().mean(
                        dim=2, keepdim=True
                    )
                    data_dims = list(range(fg_mask.ndim))[1:]
                    fg_mask = (
                        fg_mask - fg_mask.amin(dim=data_dims, keepdim=True)
                    ) / (
                        fg_mask.amax(dim=data_dims, keepdim=True)
                        - fg_mask.amin(dim=data_dims, keepdim=True)
                        + 1e-7
                    )
                    bg_mask = 1 - fg_mask
                    source_residual = v_gt - v_src
                    if projected_source_residual:
                        edit_direction = (
                            v_trg.float() - v_src.float()
                        )
                        valid_core = (
                            edit_direction.square()
                            .sum(dim=2)
                            > 1e-6
                        )
                        safe_residual, proj_diag = (
                            remove_antagonistic_source_residual(
                                source_residual=source_residual,
                                edit_direction=edit_direction,
                                target_change_core=valid_core,
                            )
                        )
                        v_t = v_trg + bg_mask * safe_residual
                        if index == 0:
                            removed_frac = (
                                proj_diag[
                                    "appearance_leakage_removed_energy"
                                ].sum()
                                / (
                                    source_residual.float()
                                    .square()
                                    .sum(dim=2, keepdim=True)
                                    .sum()
                                    + 1e-8
                                )
                            ).item()
                            print(
                                "PROJECTED_RESIDUAL "
                                f"block={current_start_frame // self.num_frame_per_block} "
                                f"removed_frac={removed_frac:.4f} "
                                f"core_coverage={valid_core.float().mean().item():.4f}"
                            )
                    else:
                        v_t = v_trg + bg_mask * source_residual
                denoised_pred = noisy_pred_input - t_i * v_t
                if (
                    source_owner_residual_constraint
                    and carry_had_state
                ):
                    if (
                        carried_identity_residual is None
                        or carried_identity_support is None
                    ):
                        raise RuntimeError(
                            "Per-step residual constraint requires a "
                            "prepared source-coordinate residual"
                        )
                    constraint_progress = float(index + 1) / float(
                        len(denoising_step_list)
                    )
                    constraint_result = (
                        apply_source_owner_residual_constraint(
                            current_latent=denoised_pred,
                            source_latent=src_input,
                            carried_residual=(
                                carried_identity_residual
                            ),
                            support=carried_identity_support,
                            spatial_shape=(
                                hand_role_debug[
                                    "object_posterior"
                                ].shape[-2:]
                            ),
                            strength=(
                                identity_residual_constraint_strength
                            ),
                            denoising_progress=constraint_progress,
                            schedule_power=(
                                identity_residual_constraint_power
                            ),
                            protect_mask=hand_only_mask[
                                :, role_left:role_right
                            ],
                        )
                    )
                    denoised_pred = constraint_result.latent
                    hand_role_debug.update({
                        "identity_residual_constraint_weight": (
                            constraint_result.weight
                        ),
                        "identity_residual_constraint_correction": (
                            constraint_result.correction.float().square()
                            .mean(dim=2).sqrt()
                        ),
                        "identity_residual_constraint_gap_before": (
                            constraint_result.target_gap_before
                        ),
                        "identity_residual_constraint_gap_after": (
                            constraint_result.target_gap_after
                        ),
                    })
                    if index in identity_trace_steps:
                        constraint_debug = {
                            "identity_residual_constraint_weight": (
                                constraint_result.weight
                            ),
                            "identity_residual_constraint_correction": (
                                constraint_result.correction.float().square()
                                .mean(dim=2).sqrt()
                            ),
                            "identity_residual_constraint_gap_before": (
                                constraint_result.target_gap_before
                            ),
                            "identity_residual_constraint_gap_after": (
                                constraint_result.target_gap_after
                            ),
                        }
                        active_constraint = (
                            constraint_result.weight > 0
                        )
                        active_count = active_constraint.sum().clamp_min(1)
                        before_on_owner = (
                            constraint_result.target_gap_before[
                                active_constraint
                            ].sum()
                            / active_count
                        )
                        after_on_owner = (
                            constraint_result.target_gap_after[
                                active_constraint
                            ].sum()
                            / active_count
                        )
                        print(
                            "SOURCE_OWNER_RESIDUAL_CONSTRAINT "
                            f"block="
                            f"{current_start_frame // self.num_frame_per_block} "
                            f"step={index} "
                            f"progress={constraint_progress:.3f} "
                            f"weight={constraint_result.weight.mean().item():.4f} "
                            f"owner_gap_before={before_on_owner.item():.4f} "
                            f"owner_gap_after={after_on_owner.item():.4f}"
                        )
                        if save_role_dir is not None:
                            self._save_hand_role_debug(
                                save_role_dir,
                                current_start_frame
                                // self.num_frame_per_block,
                                constraint_debug,
                                artifact_suffix=(
                                    f"_constraint_step_{index:02d}"
                                ),
                            )
                if source_owner_geometry_envelope:
                    if oracle_source_owner_mask is None:
                        raise RuntimeError(
                            "Geometry envelope requires the clean-source "
                            "owner mask"
                        )
                    geometry_progress = float(index + 1) / float(
                        len(denoising_step_list)
                    )
                    geometry_result = apply_source_owner_geometry_envelope(
                        current_latent=denoised_pred,
                        source_latent=src_input,
                        source_owner_mask=oracle_source_owner_mask[
                            :, role_left:role_right
                        ],
                        strength=source_geometry_strength,
                        denoising_progress=geometry_progress,
                        schedule_power=source_geometry_power,
                        margin=source_geometry_margin,
                        protect_mask=hand_only_mask[
                            :, role_left:role_right
                        ],
                    )
                    denoised_pred = geometry_result.latent
                    geometry_debug = {
                        "source_geometry_envelope_weight": (
                            geometry_result.weight
                        ),
                        "source_geometry_envelope_correction": (
                            geometry_result.correction.float().square()
                            .mean(dim=2).sqrt()
                        ),
                        "source_geometry_gap_before": (
                            geometry_result.source_gap_before
                        ),
                        "source_geometry_gap_after": (
                            geometry_result.source_gap_after
                        ),
                    }
                    hand_role_debug.update(geometry_debug)
                    if index in identity_trace_steps:
                        active_geometry = geometry_result.weight > 0
                        active_count = active_geometry.sum().clamp_min(1)
                        before_on_preserve = (
                            geometry_result.source_gap_before[
                                active_geometry
                            ].sum()
                            / active_count
                        )
                        after_on_preserve = (
                            geometry_result.source_gap_after[
                                active_geometry
                            ].sum()
                            / active_count
                        )
                        print(
                            "SOURCE_OWNER_GEOMETRY_STEP "
                            f"block="
                            f"{current_start_frame // self.num_frame_per_block} "
                            f"step={index} "
                            f"progress={geometry_progress:.3f} "
                            f"preserve={active_geometry.float().mean().item():.4f} "
                            f"weight={geometry_result.weight.mean().item():.4f} "
                            f"source_gap_before={before_on_preserve.item():.4f} "
                            f"source_gap_after={after_on_preserve.item():.4f}"
                        )
                        if save_role_dir is not None:
                            self._save_hand_role_debug(
                                save_role_dir,
                                current_start_frame
                                // self.num_frame_per_block,
                                geometry_debug,
                                artifact_suffix=(
                                    f"_geometry_step_{index:02d}"
                                ),
                            )
                if (
                    current_causal_identity_bootstrap_plan is not None
                    and current_causal_identity_bootstrap is None
                ):
                    # Pair clean source correspondence keys with low-noise
                    # target values, then keep that identity anchor frozen.
                    self.generator(
                        noisy_image_or_video=denoised_pred.detach(),
                        conditional_dict=trg_conditional_dict,
                        timestep=context_timestep,
                        kv_cache=kv_cache_trg,
                        crossattn_cache=crossattn_cache_trg,
                        current_start=0,
                    )
                    current_causal_identity_bootstrap = (
                        target_identity_memory
                        .bootstrap_causal_first_frame(
                            kv_cache=kv_cache_trg,
                            write_weight=(
                                current_causal_identity_bootstrap_plan
                                .write_weight
                            ),
                            num_frames=current_num_frames,
                            target_batch_start=0,
                            source_kv_cache=kv_cache_src,
                        )
                    )
                    shared_dict_dual["target_identity_memory"] = (
                        target_identity_memory.export()
                    )
                    bootstrap_shape = hand_role_debug[
                        "object_posterior"
                    ]
                    hand_role_debug[
                        "identity_causal_bootstrap_evidence"
                    ] = (
                        current_causal_identity_bootstrap
                        .accumulated_evidence.mean(
                            dim=(0, 2),
                        )[:, None, None, None].expand_as(
                            bootstrap_shape
                        )
                    )
                    print(
                        "CAUSAL_IDENTITY_BOOTSTRAP "
                        "block=0 "
                        "source=clean_source_key_target_x0_value "
                        "immutable=1 "
                        "applies_from_step=1 "
                        "weight="
                        f"{current_causal_identity_bootstrap.write_weight.mean().item():.4f} "
                        "first_frame_weight="
                        f"{current_causal_identity_bootstrap.write_weight.reshape(batch_size, current_num_frames, -1)[:, 0].mean().item():.4f} "
                        "support="
                        f"{(current_causal_identity_bootstrap.write_weight > 0).float().mean().item():.4f} "
                        "evidence="
                        f"{current_causal_identity_bootstrap.accumulated_evidence.mean().item():.4f}"
                    )

                #✨ target mask grounding
                if index == len(denoising_step_list) // 2:
                    _, inloop_src_trg_fg_mask_bin, mask_soft_vis, mask_bin_vis = self._aggregate_crossattn_mask(
                        crossattn_cache_dual, size=(current_num_frames, height, width), scale_factor=16
                    )
                    inloop_trg_fg_mask_bin = inloop_src_trg_fg_mask_bin.chunk(2, dim=0)[1]
                    # inject union of origin src and in-processing trg masks to kv_cache
                    inloop_trg_fg_mask = inloop_trg_fg_mask_bin | src_fg_mask_bin
                    # #region debug-point D:target-mask-union
                    if current_roles is not None:
                        debug_target_tokens = inloop_trg_fg_mask_bin.bool()
                        debug_union_tokens = inloop_trg_fg_mask.bool()
                        debug_target_intersection = (
                            debug_target_tokens & role_edit_tokens
                        ).float().sum()
                        _debug_report(
                            "D",
                            "edit_causal_inference.py:target-mask-union",
                            "Mid-step target mask union",
                            {
                                "block": current_start_frame // self.num_frame_per_block,
                                "target_coverage": debug_target_tokens.float().mean().item(),
                                "union_coverage": debug_union_tokens.float().mean().item(),
                                "target_role_iou": (
                                    debug_target_intersection
                                    / (
                                        debug_target_tokens | role_edit_tokens
                                    ).float().sum().clamp_min(1)
                                ).item(),
                            },
                        )
                    # #endregion
                    if (
                        not soft_region_modulation
                        and (
                            consistent_role_kv_enabled
                            or causal_ownership_enabled
                        )
                    ):
                        inloop_trg_fg_mask = role_edit_tokens
                    self._inject_masks_to_kv_cache(
                        kv_cache_dual, trg_fg_mask_cache, inloop_trg_fg_mask,
                        factorized_operator_cache=(
                            factorized_operator_cache
                            if factorized_bayes_enabled
                            else None
                        ),
                        current_factorized_operators=(
                            current_factorized_operators
                        ),
                        target_owned_mask_cache=target_owned_mask_cache,
                        current_target_owned_mask=(
                            current_target_owned_mask
                            if target_owned_object_handoff
                            else None
                        ),
                        current_identity_read_mask=(
                            current_identity_read_mask
                        ),
                        current_causal_owner_mask=(
                            current_memory_query_weight
                        ),
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
            if source_identity_residual_carry:
                if (
                    identity_residual_carry is None
                    or current_identity_read_mask is None
                    or source_query_features is None
                ):
                    raise RuntimeError(
                        "Residual carry commit requires source ownership"
                    )
                # Freeze the short-term residual after the first edited
                # source-owned block.  Rewriting it from its own carried
                # output would recursively reinforce errors and drift.
                if not carry_had_state:
                    identity_residual_carry.commit(
                        source_features=source_query_features,
                        owner_weight=current_identity_read_mask,
                        source_latent=src_input,
                        target_latent=denoised_pred,
                        tokens_per_frame=self.frame_seq_length,
                        spatial_shape=(
                            hand_role_debug["object_posterior"].shape[-2:]
                        ),
                    )

            del kv_cache_dual
            self._kv_cache_to(kv_cache_trg, 'cuda', low_memory)
            self._register_crossattn_mask_gatherer(crossattn_cache_trg, tok_trg, layers=mask_layers, fg_scale=fg_scale)
            if target_identity_enabled:
                self._register_identity_key_capture(
                    kv_cache_trg,
                    identity_memory_layers,
                )
            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=trg_conditional_dict,
                timestep=context_timestep,
                kv_cache=kv_cache_trg,
                crossattn_cache=crossattn_cache_trg,
                current_start=current_start_frame * self.frame_seq_length,
            )
            block_index = current_start_frame // self.num_frame_per_block
            if first_block_identity_anchor and block_index == 0:
                anchor_num_tokens = (
                    current_num_frames * self.frame_seq_length
                )
                identity_anchor_kv = []
                for layer_idx in range(self.num_transformer_blocks):
                    layer_cache = kv_cache_trg[layer_idx]
                    end_idx = int(
                        layer_cache["local_end_index"].item()
                    )
                    start_idx = max(0, end_idx - anchor_num_tokens)
                    identity_anchor_kv.append({
                        "k": layer_cache["k"][
                            :, start_idx:end_idx
                        ].clone().detach(),
                        "v": layer_cache["v"][
                            :, start_idx:end_idx
                        ].clone().detach(),
                    })
                persistent_identity_anchor_kv = identity_anchor_kv
                shared_dict_dual[
                    "identity_anchor_kv"
                ] = identity_anchor_kv
                shared_dict_dual[
                    "identity_anchor_scale"
                ] = float(identity_anchor_scale)
                print(
                    "IDENTITY_ANCHOR frozen "
                    f"block={block_index} "
                    f"tokens={anchor_num_tokens} "
                    f"layers={len(identity_anchor_kv)} "
                    f"scale={identity_anchor_scale:.2f}"
                )
            if (
                first_block_identity_anchor
                and block_index > 0
                and persistent_identity_anchor_kv is not None
            ):
                anchor_kv = persistent_identity_anchor_kv
                anchor_blend = 0.3
                corrected_layers = 0
                for layer_idx in range(self.num_transformer_blocks):
                    layer_cache = kv_cache_trg[layer_idx]
                    end_idx = int(
                        layer_cache["local_end_index"].item()
                    )
                    current_tokens = (
                        current_num_frames * self.frame_seq_length
                    )
                    start_idx = max(0, end_idx - current_tokens)
                    current_v = layer_cache["v"][
                        :, start_idx:end_idx
                    ]
                    anchor_v = anchor_kv[layer_idx]["v"]
                    if current_v.shape[1] == anchor_v.shape[1]:
                        layer_cache["v"][
                            :, start_idx:end_idx
                        ] = (
                            (1.0 - anchor_blend) * current_v
                            + anchor_blend * anchor_v
                        )
                        corrected_layers += 1
                print(
                    "ANCHOR_WRITE_CORRECTION "
                    f"block={block_index} "
                    f"blend={anchor_blend:.2f} "
                    f"tokens={current_tokens} "
                    f"layers={corrected_layers}/{self.num_transformer_blocks}"
                )
                shared_dict_dual[
                    "identity_anchor_kv"
                ] = persistent_identity_anchor_kv
                shared_dict_dual[
                    "identity_anchor_scale"
                ] = float(identity_anchor_scale)
            if role_fixed_native_history:
                if (
                    role_native_kv_history is None
                    or current_causal_ownership is None
                    or current_factorized_operators is None
                ):
                    raise RuntimeError(
                        "Role-fixed native history requires owner and "
                        "factorized write evidence"
                    )
                native_write_base = (
                    current_transactional_owner.write_weight.float()
                    if current_transactional_owner is not None
                    else current_causal_ownership.owner_weight.float()
                )
                native_write_gate = (
                    current_target_authority_support_tokens.float()
                    if current_target_authority_support_tokens is not None
                    else current_factorized_operators
                    .target_memory_action.float()
                )
                # As with reads, semantics is a hard permission boundary and
                # the automatic transaction supplies confidence. Multiplying
                # by the soft semantic score twice would suppress legitimate
                # writes below the native-history admission threshold.
                native_write_confidence = (
                    apply_semantic_transaction_gate(
                        native_write_base, native_write_gate
                    )
                    if current_target_authority_support_tokens is not None
                    else (native_write_base * native_write_gate).clamp(
                        0.0, 1.0
                    )
                )
                if current_edit_authority is not None:
                    semantic_shape = hand_role_debug[
                        "object_posterior"
                    ].shape
                    hand_role_debug[
                        "target_semantic_kv_read_gate"
                    ] = current_memory_query_weight.reshape(semantic_shape)
                    hand_role_debug[
                        "target_semantic_kv_write_gate"
                    ] = native_write_confidence.reshape(semantic_shape)
                    print(
                        "SEMANTIC_KV_TRANSACTION "
                        f"block={current_start_frame // self.num_frame_per_block} "
                        f"read={current_memory_query_weight.mean().item():.4f} "
                        f"write={native_write_confidence.mean().item():.4f} "
                        "external_object_mask=disabled"
                    )
                native_commit = role_native_kv_history.commit(
                    source_kv_cache=kv_cache_src,
                    target_kv_cache=kv_cache_trg,
                    write_confidence=native_write_confidence,
                    lineage_confidence=(
                        current_transactional_owner.read_weight.float()
                        if native_history_payload_invariant_lineage
                        else None
                    ),
                    retention_confidence=(
                        current_memory_query_weight.float()
                        if native_history_persistent_residual_upsert
                        else None
                    ),
                    frame_indices=(
                        current_source_frame_indices
                        if native_history_flow_indexed_residual
                        else None
                    ),
                    spatial_shape=(
                        hand_role_debug["object_posterior"].shape[-2:]
                        if native_history_flow_indexed_residual
                        else None
                    ),
                )
                written = torch.stack([
                    value["written"].float()
                    for value in native_commit.values()
                ]).mean()
                lineage_tokens = torch.stack([
                    value["lineage_tokens"].float()
                    for value in native_commit.values()
                ]).mean()
                lineage_held_tokens = torch.stack([
                    value["lineage_held_tokens"].float()
                    for value in native_commit.values()
                ]).mean()
                recent_held_tokens = torch.stack([
                    value["recent_held_tokens"].float()
                    for value in native_commit.values()
                ]).mean()
                dense_recent_accepted = torch.stack([
                    value["dense_recent_accepted"].float()
                    for value in native_commit.values()
                ]).mean()
                dense_recent_residual_consensus = torch.stack([
                    value["dense_recent_residual_consensus"].float()
                    for value in native_commit.values()
                ]).mean()
                candidate_target_source_similarity = torch.stack([
                    value["candidate_target_source_similarity"].float()
                    for value in native_commit.values()
                ]).mean()
                candidate_target_canonical_similarity = torch.stack([
                    value["candidate_target_canonical_similarity"].float()
                    for value in native_commit.values()
                ]).mean()
                mutable_target_payload_written = torch.stack([
                    value["mutable_target_payload_written"].float()
                    for value in native_commit.values()
                ]).mean()
                mutable_target_payload_authorized = torch.stack([
                    value["mutable_target_payload_authorized"].float()
                    for value in native_commit.values()
                ]).mean()
                persistent_residual_transport_tokens = torch.stack([
                    value["persistent_residual_transport_tokens"].float()
                    for value in native_commit.values()
                ]).mean()
                persistent_residual_transport_similarity = torch.stack([
                    value[
                        "persistent_residual_transport_similarity"
                    ].float()
                    for value in native_commit.values()
                ]).mean()
                persistent_guarded_update_tokens = torch.stack([
                    value["persistent_guarded_update_tokens"].float()
                    for value in native_commit.values()
                ]).mean()
                persistent_residual_consistency = torch.stack([
                    value[
                        "persistent_residual_consistency_on_match"
                    ].float()
                    for value in native_commit.values()
                ]).mean()
                flow_appearance_trust_on_support = torch.stack([
                    value[
                        "flow_indexed_appearance_trust_on_support"
                    ].float()
                    for value in native_commit.values()
                ]).mean() if native_history_flow_indexed_residual else (
                    written.new_zeros(())
                )
                flow_local_transport_on_support = torch.stack([
                    value[
                        "flow_indexed_local_transport_on_support"
                    ].float()
                    for value in native_commit.values()
                ]).mean() if native_history_flow_indexed_residual else (
                    written.new_zeros(())
                )
                print(
                    "ROLE_FIXED_NATIVE_KV_COMMIT "
                    f"block={current_start_frame // self.num_frame_per_block} "
                    f"layers={len(native_commit)} "
                    f"canonical={int(role_native_kv_history.has_canonical())} "
                    f"tokens={written.item():.1f} "
                    f"lineage_tokens={lineage_tokens.item():.1f} "
                    f"lineage_held={lineage_held_tokens.item():.1f} "
                    f"recent_held={recent_held_tokens.item():.1f} "
                    f"dense_recent_accepted={dense_recent_accepted.item():.4f} "
                    "dense_recent_residual_consensus="
                    f"{dense_recent_residual_consensus.item():.4f} "
                    "candidate_target_source_similarity="
                    f"{candidate_target_source_similarity.item():.4f} "
                    "candidate_target_canonical_similarity="
                    f"{candidate_target_canonical_similarity.item():.4f} "
                    "mutable_target_payload_written="
                    f"{mutable_target_payload_written.item():.1f} "
                    "mutable_target_payload_authorized="
                    f"{mutable_target_payload_authorized.item():.1f} "
                    "residual_transport_tokens="
                    f"{persistent_residual_transport_tokens.item():.1f} "
                    "residual_transport_similarity="
                    f"{persistent_residual_transport_similarity.item():.4f} "
                    "guarded_updates="
                    f"{persistent_guarded_update_tokens.item():.1f} "
                    "residual_consistency="
                    f"{persistent_residual_consistency.item():.4f} "
                    "appearance_trust="
                    f"{flow_appearance_trust_on_support.item():.4f} "
                    "local_transport_reset="
                    f"{flow_local_transport_on_support.item():.4f} "
                    + (
                        "address=clean_source_flow_indexed "
                        "target_key_identity=disabled "
                        if native_history_flow_indexed_residual
                        else ""
                    )
                    + (
                        "payload=immutable_canonical_target_kv "
                        "mutable=source_address_lineage_only"
                        if native_history_payload_invariant_lineage
                        else (
                            (
                                (
                                    "payload=persistent_rebased_target_residual "
                                    "update=token_upsert "
                                    "canonical=address_only_no_payload_fallback"
                                )
                                if native_history_persistent_residual_upsert
                                else (
                                    "payload=token_atomic_recent_clean_target_kv "
                                    "canonical=address_only_no_payload_fallback"
                                )
                            )
                            if native_history_token_atomic_payload
                            else (
                                "payload=dense_recent_clean_target_kv "
                                "canonical=immutable_fallback"
                            )
                        )
                        if native_history_recent_entry_bridge
                        else "payload=transactional_compact_target_kv"
                        if native_history_consistent_transaction
                        else "payload=final_clean_native_target_kv"
                    )
                )
                if native_history_timestep_counterfactual_memory:
                    tccm_bank = (
                        role_native_kv_history.timestep_bank_statistics()
                    )
                    print(
                        "TCCM_BANK "
                        f"block={current_start_frame // self.num_frame_per_block} "
                        f"frozen={tccm_bank['frozen']} "
                        f"timesteps={tccm_bank['timesteps']} "
                        f"layers={tccm_bank['layers']} "
                        f"entries={tccm_bank['frozen_entries']} "
                        f"support={tccm_bank['supported_slots']}/"
                        f"{tccm_bank['total_slots']} "
                        "storage=cpu_detached_compact"
                    )
                if (
                    native_history_persistent_residual_upsert
                    and save_role_dir is not None
                ):
                    persistent_shape = hand_role_debug[
                        "object_posterior"
                    ].shape

                    def persistent_map(name):
                        values = [
                            item[name].float()
                            for item in native_commit.values()
                        ]
                        return torch.stack(values, dim=0).mean(
                            dim=0
                        ).reshape(persistent_shape)

                    self._save_hand_role_debug(
                        save_role_dir,
                        current_start_frame // self.num_frame_per_block,
                        {
                            "persistent_kv_direct_write": persistent_map(
                                "persistent_direct_support"
                            ),
                            "persistent_kv_retained_residual": persistent_map(
                                "persistent_retained_support"
                            ),
                            "persistent_kv_payload_support": persistent_map(
                                "persistent_payload_support"
                            ),
                            "persistent_kv_guarded_update": persistent_map(
                                "persistent_guarded_update_support"
                            ),
                            "persistent_kv_residual_consistency": persistent_map(
                                "persistent_residual_consistency"
                            ),
                            **(
                                {
                                    "flow_indexed_state_support": persistent_map(
                                        "flow_indexed_state_support"
                                    ),
                                    "flow_indexed_state_confidence": persistent_map(
                                        "flow_indexed_state_confidence"
                                    ),
                                    "flow_indexed_appearance_trust": persistent_map(
                                        "flow_indexed_appearance_trust"
                                    ),
                                    "flow_indexed_local_transport_confidence": persistent_map(
                                        "flow_indexed_local_transport_confidence"
                                    ),
                                }
                                if native_history_flow_indexed_residual
                                else {}
                            ),
                        },
                        artifact_suffix="_persistent_kv_transaction",
                    )
                if (
                    current_edit_authority is not None
                    and save_role_dir is not None
                ):
                    # This save occurs after the native-history transaction;
                    # earlier debug snapshots cannot contain the final write
                    # gate.  The suffix keeps it distinct from step-zero role
                    # diagnostics and makes read-vs-write failures observable.
                    self._save_hand_role_debug(
                        save_role_dir,
                        current_start_frame
                        // self.num_frame_per_block,
                        {
                            "target_semantic_kv_read_gate": (
                                hand_role_debug[
                                    "target_semantic_kv_read_gate"
                                ]
                            ),
                            "target_semantic_kv_write_gate": (
                                hand_role_debug[
                                    "target_semantic_kv_write_gate"
                                ]
                            ),
                        },
                        artifact_suffix="_semantic_kv_transaction",
                    )
            if causal_paired_edit_memory:
                if (
                    paired_edit_memory is None
                    or current_paired_memory_coordinate is None
                    or current_paired_memory_proposal is None
                ):
                    raise RuntimeError(
                        "Missing paired edit-memory transaction state"
                    )
                paired_memory_had_state = (
                    paired_edit_memory.has_state()
                )
                paired_commit = paired_edit_memory.commit(
                    source_kv_cache=kv_cache_src,
                    target_kv_cache=kv_cache_trg,
                    proposal_weight=current_paired_memory_proposal,
                    object_coordinate=current_paired_memory_coordinate,
                    transactional=paired_memory_had_state,
                    preserve_canonical_payload=(
                        paired_memory_value_projection
                    ),
                )
                paired_shape = hand_role_debug[
                    "object_posterior"
                ].shape
                hand_role_debug.update({
                    "paired_memory_proposal": paired_commit[
                        "proposal"
                    ].reshape(paired_shape),
                    "paired_memory_write": paired_commit[
                        "write"
                    ].reshape(paired_shape),
                    "paired_memory_source_match": paired_commit[
                        "source_match"
                    ].reshape(paired_shape),
                    "paired_memory_residual_agreement": paired_commit[
                        "residual_agreement"
                    ].reshape(paired_shape),
                    "paired_memory_accepted": paired_commit[
                        "accepted"
                    ].reshape(paired_shape),
                })
                memory_valid = [
                    (state.evidence > paired_edit_memory.eps)
                    .float().sum(dim=-1).mean()
                    for state in paired_edit_memory.export().values()
                ]
                print(
                    "PAIRED_EDIT_COMMIT "
                    f"block={current_start_frame // self.num_frame_per_block} "
                    f"transactional={int(paired_memory_had_state)} "
                    f"proposal={paired_commit['proposal'].mean().item():.4f} "
                    f"source_match={paired_commit['source_match'].mean().item():.4f} "
                    f"residual_agreement={paired_commit['residual_agreement'].mean().item():.4f} "
                    f"accepted={paired_commit['accepted'].mean().item():.4f} "
                    f"write={paired_commit['write'].mean().item():.4f} "
                    f"slots={torch.stack(memory_valid).mean().item():.1f}"
                )
                if (
                    paired_memory_value_projection
                    and not paired_memory_disable_persistent_projection
                ):
                    paired_projection = (
                        paired_edit_memory.project_target_cache(
                            source_kv_cache=kv_cache_src,
                            target_kv_cache=kv_cache_trg,
                            reads=paired_reads,
                            strength=paired_memory_read_strength,
                        )
                    )
                    hand_role_debug[
                        "paired_memory_cache_projection"
                    ] = paired_projection["correction"].reshape(
                        paired_shape
                    )
                    print(
                        "PAIRED_EDIT_KV_PROJECTION "
                        f"block="
                        f"{current_start_frame // self.num_frame_per_block} "
                        "mode=source_plus_canonical_residual "
                        f"support={paired_projection['support'].mean().item():.4f} "
                        f"correction={paired_projection['correction'].mean().item():.6f} "
                        f"strength={paired_memory_read_strength:.3f}"
                    )
                elif paired_memory_value_projection:
                    print(
                        "PAIRED_EDIT_KV_PROJECTION "
                        f"block="
                        f"{current_start_frame // self.num_frame_per_block} "
                        "mode=disabled_transient_only "
                        "query_gate="
                        f"{int(paired_memory_query_gated_projection)}"
                    )
                if low_memory:
                    paired_edit_memory.to("cpu")
                if save_role_dir is not None:
                    self._save_hand_role_debug(
                        save_role_dir,
                        current_start_frame
                        // self.num_frame_per_block,
                        hand_role_debug,
                    )
            if (
                factorized_immutable_target_memory
                and target_identity_memory.anchor_states
            ):
                immutable_diagnostics = (
                    self._materialize_immutable_target_kv(
                        kv_cache_trg=kv_cache_trg,
                        kv_cache_src=kv_cache_src,
                        source_identity_keys=source_identity_keys,
                        target_identity_memory=target_identity_memory,
                        owner_weight=current_identity_read_mask,
                        tokens_per_frame=self.frame_seq_length,
                        correction_strength=(
                            identity_correction_strength
                        ),
                        support_floor=identity_support_floor,
                        residual_subspace=(
                            immutable_target_value_mode == "subspace"
                        ),
                    )
                )
                immutable_shape = hand_role_debug[
                    "object_posterior"
                ]
                hand_role_debug.update({
                    "immutable_target_memory_support": (
                        immutable_diagnostics["support"].reshape_as(
                            immutable_shape
                        )
                    ),
                    "immutable_target_memory_correction_ratio": (
                        immutable_diagnostics[
                            "correction_ratio"
                        ].reshape_as(immutable_shape)
                    ),
                })
                for diagnostic_name in (
                    "appearance_subspace_coherence",
                    "prototype_assignment_entropy",
                    "prototype_assignment_peak",
                    "prototype_assignment_margin",
                ):
                    if diagnostic_name in immutable_diagnostics:
                        hand_role_debug[
                            f"immutable_target_memory_{diagnostic_name}"
                        ] = immutable_diagnostics[
                            diagnostic_name
                        ].reshape_as(immutable_shape)
                for diagnostic_name, diagnostic in immutable_diagnostics.items():
                    if not diagnostic_name.startswith(
                        "selected_prototype_layer_"
                    ):
                        continue
                    hand_role_debug[
                        f"immutable_target_memory_{diagnostic_name}"
                    ] = diagnostic.reshape_as(immutable_shape)
                print(
                    "IMMUTABLE_TARGET_MEMORY "
                    f"block={current_start_frame // self.num_frame_per_block} "
                    "mode=first_chunk_"
                    f"{immutable_target_value_mode}_read_only "
                    f"support={immutable_diagnostics['support'].mean().item():.4f} "
                    f"correction_ratio={immutable_diagnostics['correction_ratio'].mean().item():.4f}"
                    + (
                        " subspace_coherence="
                        f"{immutable_diagnostics['appearance_subspace_coherence'].mean().item():.4f}"
                        if "appearance_subspace_coherence"
                        in immutable_diagnostics
                        else ""
                    )
                    + (
                        " assignment_entropy="
                        f"{immutable_diagnostics['prototype_assignment_entropy'].mean().item():.4f}"
                        " assignment_peak="
                        f"{immutable_diagnostics['prototype_assignment_peak'].mean().item():.4f}"
                        " assignment_margin="
                        f"{immutable_diagnostics['prototype_assignment_margin'].mean().item():.4f}"
                        if "prototype_assignment_entropy"
                        in immutable_diagnostics
                        else ""
                    )
                )
            # #region debug-point B:target-kv-write
            if current_roles is not None:
                debug_num_tokens = kv_cache_trg[0]["num_new_tokens"]
                debug_src_end = kv_cache_src[0]["local_end_index"].item()
                debug_trg_end = kv_cache_trg[0]["local_end_index"].item()
                debug_src_key = kv_cache_src[0]["k"][
                    :, debug_src_end - debug_num_tokens:debug_src_end
                ]
                debug_trg_key = kv_cache_trg[0]["k"][
                    :, debug_trg_end - debug_num_tokens:debug_trg_end
                ]
                debug_src_value = kv_cache_src[0]["v"][
                    :, debug_src_end - debug_num_tokens:debug_src_end
                ]
                debug_trg_value = kv_cache_trg[0]["v"][
                    :, debug_trg_end - debug_num_tokens:debug_trg_end
                ]
                _debug_report(
                    "B",
                    "edit_causal_inference.py:target-kv-write",
                    "Source and target KV similarity after target write",
                    {
                        "block": current_start_frame // self.num_frame_per_block,
                        "key_cosine": F.cosine_similarity(
                            debug_src_key.float(),
                            debug_trg_key.float(),
                            dim=-1,
                        ).mean().item(),
                        "value_cosine": F.cosine_similarity(
                            debug_src_value.float(),
                            debug_trg_value.float(),
                            dim=-1,
                        ).mean().item(),
                    },
                )
                if (
                    factorized_native_target_history
                    and current_causal_ownership is not None
                ):
                    debug_owner = (
                        current_causal_ownership.owner_support.float()
                    )
                    debug_owner_count = debug_owner.sum().clamp_min(1.0)
                    owner_key_cosine = (
                        F.cosine_similarity(
                            debug_src_key.float(),
                            debug_trg_key.float(),
                            dim=-1,
                        ).mean(dim=-1)
                        * debug_owner
                    ).sum() / debug_owner_count
                    owner_value_cosine = (
                        F.cosine_similarity(
                            debug_src_value.float(),
                            debug_trg_value.float(),
                            dim=-1,
                        ).mean(dim=-1)
                        * debug_owner
                    ).sum() / debug_owner_count
                    print(
                        "NATIVE_TARGET_KV_COMMIT "
                        f"block={current_start_frame // self.num_frame_per_block} "
                        f"owner_tokens={int(debug_owner.sum().item())} "
                        f"owner_key_cosine={owner_key_cosine.item():.4f} "
                        f"owner_value_cosine={owner_value_cosine.item():.4f} "
                        "immutable_write=0"
                    )
            # #endregion
            if target_identity_enabled:
                if (
                    identity_observation_belief is None
                    or identity_observation_tokens is None
                ):
                    raise RuntimeError(
                        "Missing independent evidence for identity write"
                    )
                identity_write_map = (
                    identity_observation_belief.edit_belief
                    * identity_observation_belief.edit_precision
                    * (
                        1.0
                        - identity_observation_belief.uncertainty
                    )
                    * identity_observation_belief.visibility
                ).clamp(0.0, 1.0)
                identity_write_tokens = F.avg_pool2d(
                    identity_write_map.reshape(
                        batch_size * current_num_frames,
                        1,
                        height,
                        width,
                    ),
                    kernel_size=2,
                    stride=2,
                ).reshape(batch_size, -1)
                identity_write_tokens = (
                    identity_write_tokens
                    * identity_observation_tokens.float()
                )
                if (
                    factorized_target_identity
                    or factorized_immutable_target_memory
                    or appearance_leakage_decomposition
                ):
                    if current_identity_write_mask is None:
                        raise RuntimeError(
                            "Factorized identity write requires a current "
                            "object core"
                        )
                    identity_write_tokens = (
                        identity_write_tokens
                        * current_identity_write_mask.float()
                    )
                    if oracle_source_owner_mask is not None:
                        # The oracle experiment tests whether localization
                        # is the bottleneck. Once target-change evidence has
                        # validated the block, write the complete visible
                        # source owner instead of cropping it again with the
                        # learned belief/token-propagation gates.
                        identity_write_tokens = (
                            current_identity_write_mask.float()
                        )
                if (
                    factorized_immutable_target_memory
                    and target_identity_memory.anchor_states
                ):
                    # The proposal pass is the sole writer.  Replay and all
                    # later blocks may read the immutable anchor but cannot
                    # contaminate it with their generated output.
                    identity_write_tokens = torch.zeros_like(
                        identity_write_tokens
                    )
                if (
                    target_identity_tokenprop_enabled
                    and oracle_source_owner_mask is None
                ):
                    if identity_token_propagator is None:
                        raise RuntimeError(
                            "Missing identity token propagator"
                        )
                    if source_query_features is None:
                        raise RuntimeError(
                            "Missing source features for token propagation"
                        )
                    current_identity_propagation = (
                        identity_token_propagator(
                            source_features=source_query_features,
                            base_write_weight=identity_write_tokens,
                            support_weight=(
                                identity_tokenprop_support_tokens
                                * current_identity_write_mask.float()
                                if (
                                    factorized_target_identity
                                    or appearance_leakage_decomposition
                                )
                                else identity_tokenprop_support_tokens
                            ),
                        )
                    )
                    identity_write_tokens = (
                        current_identity_propagation.write_weight
                    )
                    if (
                        factorized_target_identity
                        or appearance_leakage_decomposition
                    ):
                        identity_write_tokens = (
                            identity_write_tokens
                            * current_identity_write_mask.float()
                        )
                if (
                    object_wise_anchor_reset
                    and current_causal_identity_bootstrap is not None
                    and not target_identity_memory.causal_edit_anchor_reset
                ):
                    if current_causal_identity_bootstrap_plan is None:
                        raise RuntimeError(
                            "Missing object core for anchor reset"
                        )
                    object_anchor_reset_weight = (
                        current_causal_identity_bootstrap_plan
                        .write_weight
                        * identity_write_tokens
                    )
                    reset_has_support = (
                        object_anchor_reset_weight
                        > target_identity_memory.eps
                    ).any(dim=-1)
                    reset_committed = bool(
                        reset_has_support.all().item()
                    )
                    reset_key_cosine = torch.tensor(
                        1.0,
                        device=identity_write_tokens.device,
                    )
                    reset_value_cosine = torch.tensor(
                        1.0,
                        device=identity_write_tokens.device,
                    )
                    reset_evidence = torch.tensor(
                        0.0,
                        device=identity_write_tokens.device,
                    )
                    if reset_committed:
                        provisional_states = {
                            layer: state
                            for layer, state in
                            target_identity_memory.export().items()
                        }
                        object_anchor_reset_update = (
                            target_identity_memory
                            .reset_causal_edit_anchor(
                                kv_cache=kv_cache_trg,
                                write_weight=(
                                    object_anchor_reset_weight
                                ),
                                num_frames=current_num_frames,
                                source_kv_cache=kv_cache_src,
                            )
                        )
                        reset_evidence = (
                            object_anchor_reset_update
                            .accumulated_evidence.mean()
                        )
                        if target_owned_object_handoff:
                            if source_query_features is None:
                                raise RuntimeError(
                                    "Missing source features for target-"
                                    "owned anchor commit"
                                )
                            target_identity_memory.commit_target_owned_anchor(
                                anchor_mask=(
                                    object_anchor_reset_weight >
                                    target_identity_memory.eps
                                ),
                                anchor_features=source_query_features,
                            )
                        reset_key_scores = []
                        reset_value_scores = []
                        for layer, committed_state in (
                            target_identity_memory.export().items()
                        ):
                            provisional_state = (
                                provisional_states[layer]
                            )
                            valid = (
                                provisional_state.evidence
                                > target_identity_memory.eps
                            ) & (
                                committed_state.evidence
                                > target_identity_memory.eps
                            )
                            valid_count = valid.sum().clamp_min(1)
                            reset_key_scores.append(
                                (
                                    F.cosine_similarity(
                                        provisional_state.key.float(),
                                        committed_state.key.float(),
                                        dim=-1,
                                    ).mean(dim=-1)
                                    * valid
                                ).sum() / valid_count
                            )
                            reset_value_scores.append(
                                (
                                    F.cosine_similarity(
                                        provisional_state.value.float(),
                                        committed_state.value.float(),
                                        dim=-1,
                                    ).mean(dim=-1)
                                    * valid
                                ).sum() / valid_count
                            )
                        reset_key_cosine = torch.stack(
                            reset_key_scores
                        ).mean()
                        reset_value_cosine = torch.stack(
                            reset_value_scores
                        ).mean()
                    reset_shape = hand_role_debug[
                        "object_posterior"
                    ]
                    hand_role_debug.update({
                        "identity_object_anchor_reset_weight": (
                            object_anchor_reset_weight.reshape_as(
                                reset_shape
                            )
                        ),
                        "identity_object_anchor_reset_committed": (
                            torch.ones_like(reset_shape)
                            * float(reset_committed)
                        ),
                        "identity_object_anchor_reset_key_cosine": (
                            torch.ones_like(reset_shape)
                            * reset_key_cosine
                        ),
                        "identity_object_anchor_reset_value_cosine": (
                            torch.ones_like(reset_shape)
                            * reset_value_cosine
                        ),
                    })
                    print(
                        "OBJECT_EDIT_ANCHOR_RESET "
                        "block=0 "
                        "object_only=1 background_source=1 "
                        f"committed={int(reset_committed)} "
                        "weight="
                        f"{object_anchor_reset_weight.mean().item():.4f} "
                        "support="
                        f"{(object_anchor_reset_weight > 0).float().mean().item():.4f} "
                        "key_cosine_to_provisional="
                        f"{reset_key_cosine.item():.4f} "
                        "value_cosine_to_provisional="
                        f"{reset_value_cosine.item():.4f} "
                        f"evidence={reset_evidence.item():.4f}"
                    )
                current_identity_update = (
                    target_identity_memory.update(
                        kv_cache=kv_cache_trg,
                        write_weight=identity_write_tokens,
                        source_kv_cache=kv_cache_src,
                    )
                )
                # #region debug-point H17:identity-prototype-drift
                _debug_anchor_states = getattr(
                    target_identity_memory,
                    "_debug_online_anchor_states",
                    None,
                )
                if _debug_anchor_states is None:
                    _debug_anchor_states = {
                        layer: (
                            state.key.detach().clone(),
                            state.value.detach().clone(),
                            state.evidence.detach().clone(),
                        )
                        for layer, state
                        in target_identity_memory.export().items()
                    }
                    target_identity_memory._debug_online_anchor_states = (
                        _debug_anchor_states
                    )
                _debug_key_cosines = []
                _debug_value_cosines = []
                for layer, state in (
                    target_identity_memory.export().items()
                ):
                    (
                        _debug_anchor_key,
                        _debug_anchor_value,
                        _debug_anchor_evidence,
                    ) = _debug_anchor_states[layer]
                    _debug_valid = (
                        _debug_anchor_evidence.float()
                        > target_identity_memory.eps
                    )
                    _debug_valid_count = (
                        _debug_valid.sum().clamp_min(1)
                    )
                    _debug_key_cosines.append(
                        (
                            F.cosine_similarity(
                                state.key.float(),
                                _debug_anchor_key.float(),
                                dim=-1,
                            ).mean(dim=-1)
                            * _debug_valid
                        ).sum()
                        / _debug_valid_count
                    )
                    _debug_value_cosines.append(
                        (
                            F.cosine_similarity(
                                state.value.float(),
                                _debug_anchor_value.float(),
                                dim=-1,
                            ).mean(dim=-1)
                            * _debug_valid
                        ).sum()
                        / _debug_valid_count
                    )
                _debug_key_cosine = torch.stack(
                    _debug_key_cosines
                ).mean()
                _debug_value_cosine = torch.stack(
                    _debug_value_cosines
                ).mean()
                hand_role_debug[
                    "identity_anchor_key_cosine"
                ] = (
                    torch.ones_like(
                        hand_role_debug["object_posterior"]
                    )
                    * _debug_key_cosine
                )
                hand_role_debug[
                    "identity_anchor_value_cosine"
                ] = (
                    torch.ones_like(
                        hand_role_debug["object_posterior"]
                    )
                    * _debug_value_cosine
                )
                # #endregion
                hand_role_debug["identity_write_weight"] = (
                    identity_write_tokens.reshape_as(
                        hand_role_debug["object_posterior"]
                    )
                )
                if current_identity_propagation is not None:
                    _debug_shape = hand_role_debug["object_posterior"]
                    hand_role_debug.update({
                        "identity_tokenprop_base_write": (
                            current_identity_propagation
                            .base_write_weight.reshape_as(_debug_shape)
                        ),
                        "identity_tokenprop_support_weight": (
                            current_identity_propagation
                            .support_weight.reshape_as(_debug_shape)
                        ),
                        "identity_tokenprop_match_confidence": (
                            current_identity_propagation
                            .match_confidence.reshape_as(_debug_shape)
                        ),
                        "identity_tokenprop_similarity": (
                            (
                                current_identity_propagation
                                .best_similarity.reshape_as(_debug_shape)
                                + 1.0
                            )
                            * 0.5
                        ),
                        "identity_tokenprop_previous_weight": (
                            current_identity_propagation
                            .matched_previous_weight.reshape_as(
                                _debug_shape
                            )
                        ),
                        "identity_tokenprop_has_previous": (
                            torch.ones_like(_debug_shape)
                            * current_identity_propagation
                            .has_previous.float().mean()
                        ),
                    })
                print(
                    "TARGET_IDENTITY_WRITE "
                    "block="
                    f"{current_start_frame // self.num_frame_per_block} "
                    "weight="
                    f"{identity_write_tokens.mean().item():.4f} "
                    "observation_evidence="
                    f"{current_identity_update.observation_evidence.mean().item():.4f} "
                    "gain="
                    f"{current_identity_update.update_gain.mean().item():.4f} "
                    "accumulated_evidence="
                    f"{current_identity_update.accumulated_evidence.mean().item():.4f}"
                )
                if current_identity_propagation is not None:
                    print(
                        "TARGET_IDENTITY_TOKENPROP "
                        "block="
                        f"{current_start_frame // self.num_frame_per_block} "
                        "has_previous="
                        f"{current_identity_propagation.has_previous.float().mean().item():.4f} "
                        "base_weight="
                        f"{current_identity_propagation.base_write_weight.mean().item():.4f} "
                        "support="
                        f"{current_identity_propagation.support_weight.mean().item():.4f} "
                        "gated_weight="
                        f"{identity_write_tokens.mean().item():.4f} "
                        "match="
                        f"{current_identity_propagation.match_confidence.mean().item():.4f} "
                        "previous="
                        f"{current_identity_propagation.matched_previous_weight.mean().item():.4f} "
                        "similarity="
                        f"{current_identity_propagation.best_similarity.mean().item():.4f}"
                    )
                if save_role_dir is not None:
                    self._save_hand_role_debug(
                        save_role_dir,
                        current_start_frame
                        // self.num_frame_per_block,
                        hand_role_debug,
                    )
            #✨ store clean target kv cache, and obtain clean target mask
            _, trg_fg_mask_bin, _, _ = self._aggregate_crossattn_mask(crossattn_cache_trg)
            legacy_trg_fg_mask = trg_fg_mask_bin | src_fg_mask_bin
            current_trg_fg_mask = (
                legacy_trg_fg_mask
                if soft_region_modulation
                else (
                    role_edit_tokens
                    if (
                        consistent_role_kv_enabled
                        or (
                            causal_owner_consistent_kv_metadata
                            and causal_ownership_enabled
                        )
                    )
                    else legacy_trg_fg_mask
                )
            )
            if (
                causal_owner_consistent_kv_metadata
                and causal_ownership_enabled
                and not soft_region_modulation
            ):
                metadata_xor = (
                    role_edit_tokens.bool()
                    ^ legacy_trg_fg_mask.bool()
                )
                print(
                    "CAUSAL_OWNER_KV_METADATA "
                    f"block="
                    f"{current_start_frame // self.num_frame_per_block} "
                    "mode=role_edit_tokens "
                    f"role_coverage="
                    f"{role_edit_tokens.float().mean().item():.6f} "
                    f"legacy_coverage="
                    f"{legacy_trg_fg_mask.float().mean().item():.6f} "
                    f"xor={metadata_xor.float().mean().item():.6f}"
                )
            self._update_trg_fg_mask_cache(trg_fg_mask_cache, current_trg_fg_mask, kv_cache_trg)
            if target_owned_mask_cache is not None:
                handoff_active = bool(
                    target_identity_memory.causal_edit_anchor_reset
                )
                current_target_owned_mask = (
                    target_identity_memory.match_target_owned_tokens(
                        source_features=source_query_features,
                        candidate_mask=(
                            src_fg_mask_bin.bool()
                            | role_edit_tokens.bool()
                        ),
                        min_similarity=target_owned_min_similarity,
                    )
                    if handoff_active
                    else torch.zeros_like(
                        current_trg_fg_mask, dtype=torch.bool
                    )
                )
                self._update_target_owned_mask_cache(
                    target_owned_mask_cache,
                    current_target_owned_mask,
                    kv_cache_trg,
                )
                if handoff_active:
                    target_identity_memory.record_target_owned_tokens(
                        current_target_owned_mask
                    )
                if hand_role_debug is not None:
                    target_owned_shape = hand_role_debug[
                        "object_posterior"
                    ].shape
                    hand_role_debug.update({
                        "target_owned_object_mask": (
                            current_target_owned_mask.reshape(
                                target_owned_shape
                            ).float()
                        ),
                        "target_owned_object_active": (
                            torch.ones_like(
                                hand_role_debug[
                                    "object_posterior"
                                ]
                            ) * float(handoff_active)
                        ),
                    })
                print(
                    "TARGET_OWNED_OBJECT_HANDOFF "
                    f"block={current_start_frame // self.num_frame_per_block} "
                    f"active={int(handoff_active)} "
                    "owned_qk_source_blend=0 "
                    "owned_history_source_preserve=0 "
                    "owned_coverage="
                    f"{current_target_owned_mask.float().mean().item():.4f}"
                )
                if save_role_dir is not None:
                    self._save_hand_role_debug(
                        save_role_dir,
                        current_start_frame
                        // self.num_frame_per_block,
                        hand_role_debug,
                    )
            if belief_memory_enabled:
                if current_belief_kv_weights is None:
                    raise RuntimeError(
                        "Missing belief weights for memory cache update"
                    )
                if (
                    memory_consolidation_enabled
                    and current_memory_plan is None
                ):
                    raise RuntimeError(
                        "Missing consolidated memory write plan"
                    )
                self._update_belief_kv_weight_cache(
                    belief_kv_weight_cache,
                    current_preserve_action=(
                        1.0
                        - current_memory_plan
                        .materialized_edit_action.reshape(
                            current_belief_kv_weights
                            .preserve_action.shape
                        )
                        if current_memory_plan is not None
                        else current_belief_kv_weights
                        .preserve_action
                    ),
                    kv_cache_trg=kv_cache_trg,
                )
            if factorized_bayes_enabled:
                if current_factorized_operators is None:
                    raise RuntimeError(
                        "Missing factorized operators for cache commit"
                    )
                self._update_factorized_operator_cache(
                    factorized_operator_cache,
                    current_actions={
                        "source_key_action": (
                            current_factorized_operators
                            .source_key_action
                        ),
                        "source_value_action": (
                            current_factorized_operators
                            .source_value_action
                        ),
                        "target_memory_action": (
                            current_factorized_operators
                            .target_memory_action
                        ),
                        "unknown_action": (
                            current_factorized_operators.unknown_action
                        ),
                    },
                    kv_cache_trg=kv_cache_trg,
                )
            self._kv_cache_to(kv_cache_trg, 'cpu', low_memory)

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        if independent_first_frame:
            output = output[:, 1: ]
        if triple_first_frame:
            output = output[:, 3: ]
        if wo_video_decode:
            video = None
        else:
            dec_latent = output
            video = self.vae.decode_to_pixel(dec_latent, use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    @staticmethod
    def _save_role_state(save_dir, block_index, roles):
        arrays = {
            **roles.as_dict(),
            "edit_weight": roles.edit_weight,
            "preserve_weight": roles.preserve_weight,
        }
        np.savez_compressed(
            os.path.join(save_dir, f"block_{block_index:03d}_roles.npz"),
            **{
                name: value.detach().float().cpu().numpy()
                for name, value in arrays.items()
            },
        )
        for name, value in arrays.items():
            strip = value[0].detach().float().cpu().numpy()
            strip = np.concatenate(list((strip * 255).astype(np.uint8)), axis=0)
            Image.fromarray(strip, mode="L").resize(
                (832, strip.shape[0] * 8),
                Image.Resampling.NEAREST,
            ).save(
                os.path.join(
                    save_dir, f"block_{block_index:03d}_{name}.png"
                )
            )

    @staticmethod
    def _save_hand_role_debug(
        save_dir,
        block_index,
        debug,
        artifact_suffix="",
    ):
        np.savez_compressed(
            os.path.join(
                save_dir,
                "block_"
                f"{block_index:03d}{artifact_suffix}"
                "_hand_role_debug.npz",
            ),
            **{
                name: value.detach().float().cpu().numpy()
                for name, value in debug.items()
            },
        )
        for name, value in debug.items():
            strip = value[0].detach().float().cpu().numpy()
            strip = np.concatenate(
                list((strip * 255).clip(0, 255).astype(np.uint8)),
                axis=0,
            )
            Image.fromarray(strip, mode="L").resize(
                (832, strip.shape[0] * 16),
                Image.Resampling.NEAREST,
            ).save(
                os.path.join(
                    save_dir,
                    "block_"
                    f"{block_index:03d}{artifact_suffix}_{name}.png",
                )
            )

    @staticmethod
    def _save_reference_identity_debug(save_dir, debug):
        np.savez_compressed(
            os.path.join(
                save_dir,
                "reference_identity_bootstrap.npz",
            ),
            **{
                name: value.detach().float().cpu().numpy()
                for name, value in debug.items()
            },
        )
        for name, value in debug.items():
            strip = value[0].detach().float().cpu().numpy()
            strip = np.concatenate(
                list(
                    (strip * 255)
                    .clip(0, 255)
                    .astype(np.uint8)
                ),
                axis=0,
            )
            Image.fromarray(strip, mode="L").resize(
                (832, strip.shape[0] * 16),
                Image.Resampling.NEAREST,
            ).save(
                os.path.join(save_dir, f"{name}.png")
            )

    @staticmethod
    def _save_contact_graphs(
        save_dir,
        block_index,
        graphs,
        mode,
    ):
        arrays = {"mode": np.array(mode)}
        for batch_index, graph in enumerate(graphs):
            for name, value in graph.items():
                tensor = value.detach()
                if torch.is_floating_point(tensor):
                    tensor = tensor.float()
                arrays[f"batch_{batch_index}_{name}"] = (
                    tensor.cpu().numpy()
                )
        np.savez_compressed(
            os.path.join(
                save_dir,
                f"block_{block_index:03d}_contact_graph.npz",
            ),
            **arrays,
        )


    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        return kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        return crossattn_cache

    @staticmethod
    @torch.no_grad()
    def _materialize_immutable_target_kv(
        *,
        kv_cache_trg,
        kv_cache_src,
        source_identity_keys,
        target_identity_memory,
        owner_weight,
        tokens_per_frame,
        correction_strength,
        support_floor,
        residual_subspace=False,
    ):
        """Correct freshly written target KV from a frozen appearance bank.

        The normal target cache is still written for background and hand
        context.  Only visible owner values are projected back to the
        immutable first-chunk appearance prototypes, preventing later
        generated chunks from recursively redefining object identity.  In
        subspace mode only the coherent residual direction is constrained.
        """
        if owner_weight is None:
            raise RuntimeError(
                "Immutable target KV requires current owner weights"
            )
        states = target_identity_memory.export()
        if not states:
            raise RuntimeError(
                "Immutable target KV requires a frozen appearance bank"
            )
        support_by_layer = []
        correction_by_layer = []
        subspace_coherence_by_layer = []
        retrieval_diagnostics_by_name = {
            "prototype_assignment_entropy": [],
            "prototype_assignment_peak": [],
            "prototype_assignment_margin": [],
        }
        selected_prototype_by_layer = {}
        for layer_index, state in states.items():
            target_cache = kv_cache_trg[layer_index]
            source_cache = kv_cache_src[layer_index]
            num_new_tokens = target_cache.get("num_new_tokens")
            if num_new_tokens != owner_weight.shape[1]:
                raise ValueError(
                    "Immutable owner and target KV write must align"
                )
            target_end = target_cache["local_end_index"].item()
            source_end = source_cache["local_end_index"].item()
            current_target_value = target_cache["v"][
                :, target_end - num_new_tokens:target_end
            ]
            current_source_value = source_cache["v"][
                :, source_end - num_new_tokens:source_end
            ]
            correspondence_key = source_identity_keys.get(layer_index)
            if correspondence_key is None:
                raise RuntimeError(
                    "Missing source correspondence key for immutable "
                    f"target memory at layer {layer_index}"
                )
            corrected, support, diagnostics = (
                materialize_immutable_target_value(
                    correspondence_key=correspondence_key,
                    target_value=current_target_value,
                    source_value=current_source_value,
                    prototype_key=state.key.to(
                        device=correspondence_key.device,
                        dtype=correspondence_key.dtype,
                    ),
                    prototype_value=state.value.to(
                        device=current_target_value.device,
                        dtype=current_target_value.dtype,
                    ),
                    prototype_evidence=state.evidence.to(
                        device=correspondence_key.device
                    ),
                    owner_weight=owner_weight,
                    tokens_per_frame=tokens_per_frame,
                    prototype_value_is_residual=state.value_is_residual,
                    residual_subspace=residual_subspace,
                    support_floor=support_floor,
                    correction_strength=correction_strength,
                )
            )
            target_cache["v"][
                :, target_end - num_new_tokens:target_end
            ] = corrected
            support_by_layer.append(support.float())
            correction_by_layer.append(
                diagnostics["correction_ratio"].float()
            )
            if "appearance_subspace_coherence" in diagnostics:
                subspace_coherence_by_layer.append(
                    diagnostics[
                        "appearance_subspace_coherence"
                    ].float()
                )
            for diagnostic_name in retrieval_diagnostics_by_name:
                if diagnostic_name not in diagnostics:
                    continue
                diagnostic = diagnostics[diagnostic_name].float()
                retrieval_diagnostics_by_name[diagnostic_name].append(
                    diagnostic
                )
            if "selected_prototype" in diagnostics:
                # Preserve per-layer top-1 IDs in the NPZ. Averaging IDs
                # across layers would hide layer-local prototype switches.
                selected_prototype_by_layer[
                    f"selected_prototype_layer_{layer_index:02d}"
                ] = torch.where(
                    owner_weight > 0.0,
                    diagnostics["selected_prototype"].float(),
                    torch.full_like(
                        diagnostics["selected_prototype"].float(),
                        -1.0,
                    ),
                )
        result = {
            "support": torch.stack(
                support_by_layer, dim=0
            ).mean(dim=0),
            "correction_ratio": torch.stack(
                correction_by_layer, dim=0
            ).mean(dim=0),
        }
        if subspace_coherence_by_layer:
            result["appearance_subspace_coherence"] = torch.stack(
                subspace_coherence_by_layer, dim=0
            ).mean(dim=0)
        for diagnostic_name, diagnostics in (
            retrieval_diagnostics_by_name.items()
        ):
            if isinstance(diagnostics, list):
                if diagnostics:
                    result[diagnostic_name] = torch.stack(
                        diagnostics, dim=0
                    ).mean(dim=0)
        result.update(selected_prototype_by_layer)
        return result


    def _initialize_trg_fg_mask_cache(self, batch_size, device):
        '''
        ✨ initialize target mask as ones
        '''
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760
        trg_fg_mask_cache = {
            "trg_fg_mask": torch.ones([batch_size, kv_cache_size], dtype=torch.bool, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
        }
        return trg_fg_mask_cache

    def _initialize_belief_kv_weight_cache(self, batch_size, device):
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 32760
        return {
            "preserve_action": torch.ones(
                [batch_size, kv_cache_size],
                dtype=torch.float32,
                device=device,
            ),
            "global_end_index": torch.tensor(
                [0],
                dtype=torch.long,
                device=device,
            ),
            "local_end_index": torch.tensor(
                [0],
                dtype=torch.long,
                device=device,
            ),
        }

    def _initialize_factorized_operator_cache(
        self, batch_size, device
    ):
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 32760
        cache = {
            name: torch.zeros(
                [batch_size, kv_cache_size],
                dtype=torch.float32,
                device=device,
            )
            for name in (
                "source_key_action",
                "source_value_action",
                "target_memory_action",
                "unknown_action",
            )
        }
        cache.update({
            "global_end_index": torch.tensor(
                [0], dtype=torch.long, device=device
            ),
            "local_end_index": torch.tensor(
                [0], dtype=torch.long, device=device
            ),
        })
        return cache

    def _update_factorized_operator_cache(
        self,
        cache_state,
        current_actions,
        kv_cache_trg,
    ):
        action_names = (
            "source_key_action",
            "source_value_action",
            "target_memory_action",
            "unknown_action",
        )
        if cache_state is None:
            raise ValueError(
                "Factorized operator cache has not been initialized"
            )
        if set(current_actions) != set(action_names):
            raise ValueError(
                "Factorized cache update requires all provenance actions"
            )
        action_shapes = {
            tuple(current_actions[name].shape) for name in action_names
        }
        if len(action_shapes) != 1 or len(next(iter(action_shapes))) != 2:
            raise ValueError(
                "Factorized cache actions must share shape [B,L]"
            )

        current_end = kv_cache_trg[0]["global_end_index"].item()
        sink_tokens = kv_cache_trg[0]["sink_tokens"]
        num_new_tokens = current_actions[action_names[0]].shape[1]
        if num_new_tokens != kv_cache_trg[0]["num_new_tokens"]:
            raise ValueError(
                "Factorized actions and target cache write must align"
            )
        kv_cache_size = cache_state[action_names[0]].shape[1]
        cache_end = cache_state["local_end_index"].item()
        cache_global_end = cache_state["global_end_index"].item()
        if (
            self.local_attn_size != -1
            and current_end > cache_global_end
            and num_new_tokens + cache_end > kv_cache_size
        ):
            num_evicted_tokens = (
                num_new_tokens + cache_end - kv_cache_size
            )
            num_rolled_tokens = (
                cache_end - num_evicted_tokens - sink_tokens
            )
            for name in action_names:
                cache = cache_state[name]
                cache[
                    :, sink_tokens:sink_tokens + num_rolled_tokens
                ] = cache[
                    :,
                    sink_tokens + num_evicted_tokens:
                    sink_tokens + num_evicted_tokens + num_rolled_tokens,
                ].clone()
            local_end_index = (
                cache_end + current_end - cache_global_end
                - num_evicted_tokens
            )
        else:
            local_end_index = (
                cache_end + current_end - cache_global_end
            )
        local_start_index = local_end_index - num_new_tokens
        for name in action_names:
            value = current_actions[name].float().clamp(0.0, 1.0)
            cache_state[name][
                :, local_start_index:local_end_index
            ] = value
        provenance_sum = (
            current_actions["source_value_action"].float()
            + current_actions["target_memory_action"].float()
            + current_actions["unknown_action"].float()
        )
        if not torch.allclose(
            provenance_sum,
            torch.ones_like(provenance_sum),
            atol=1e-5,
        ):
            raise ValueError(
                "Committed factorized provenance must sum to one"
            )
        cache_state["global_end_index"].fill_(current_end)
        cache_state["local_end_index"].fill_(local_end_index)

    def _initialize_target_owned_mask_cache(
        self,
        batch_size,
        device,
    ):
        if self.local_attn_size != -1:
            kv_cache_size = (
                self.local_attn_size * self.frame_seq_length
            )
        else:
            kv_cache_size = 32760
        return {
            "target_owned_mask": torch.zeros(
                [batch_size, kv_cache_size],
                dtype=torch.bool,
                device=device,
            ),
            "global_end_index": torch.tensor(
                [0], dtype=torch.long, device=device
            ),
            "local_end_index": torch.tensor(
                [0], dtype=torch.long, device=device
            ),
        }

    def _update_trg_fg_mask_cache(self, trg_fg_mask_cache, current_trg_fg_mask, kv_cache_trg):
        '''
        ✨ update trg_fg_mask similar to kv cache update
        '''
        current_end = kv_cache_trg[0]["global_end_index"].item()
        sink_tokens = kv_cache_trg[0]["sink_tokens"]
        kv_cache_size = trg_fg_mask_cache["trg_fg_mask"].shape[1]
        num_new_tokens = current_trg_fg_mask.shape[1]
        assert num_new_tokens == kv_cache_trg[0]["num_new_tokens"], '%d != %d' % (num_new_tokens, kv_cache_trg[0]["num_new_tokens"])
        if self.local_attn_size != -1 and (current_end > trg_fg_mask_cache["global_end_index"].item()) and (
                num_new_tokens + trg_fg_mask_cache["local_end_index"].item() > kv_cache_size):
            num_evicted_tokens = num_new_tokens + trg_fg_mask_cache["local_end_index"].item() - kv_cache_size
            num_rolled_tokens = trg_fg_mask_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
            trg_fg_mask_cache["trg_fg_mask"][:, sink_tokens: sink_tokens + num_rolled_tokens] = \
                trg_fg_mask_cache["trg_fg_mask"][:, sink_tokens + num_evicted_tokens: sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            # Insert the new keys/values at the end
            local_end_index = trg_fg_mask_cache["local_end_index"].item() + current_end - \
                trg_fg_mask_cache["global_end_index"].item() - num_evicted_tokens
        else:
            # Assign new keys/values directly up to current_end
            local_end_index = trg_fg_mask_cache["local_end_index"].item() + current_end - trg_fg_mask_cache["global_end_index"].item()
        local_start_index = local_end_index - num_new_tokens
        trg_fg_mask_cache["trg_fg_mask"][:, local_start_index:local_end_index] = current_trg_fg_mask
        trg_fg_mask_cache["global_end_index"].fill_(current_end)
        trg_fg_mask_cache["local_end_index"].fill_(local_end_index)

    def _update_belief_kv_weight_cache(
        self,
        belief_kv_weight_cache,
        current_preserve_action,
        kv_cache_trg,
    ):
        if current_preserve_action.ndim != 2:
            raise ValueError(
                "Current belief KV action must have shape [B,L]"
            )
        current_end = kv_cache_trg[0]["global_end_index"].item()
        sink_tokens = kv_cache_trg[0]["sink_tokens"]
        kv_cache_size = belief_kv_weight_cache[
            "preserve_action"
        ].shape[1]
        num_new_tokens = current_preserve_action.shape[1]
        if num_new_tokens != kv_cache_trg[0]["num_new_tokens"]:
            raise ValueError(
                "Belief KV weights and target cache write must align: "
                f"{num_new_tokens} != "
                f"{kv_cache_trg[0]['num_new_tokens']}"
            )
        cache_end = belief_kv_weight_cache["local_end_index"].item()
        cache_global_end = belief_kv_weight_cache[
            "global_end_index"
        ].item()
        if (
            self.local_attn_size != -1
            and current_end > cache_global_end
            and num_new_tokens + cache_end > kv_cache_size
        ):
            num_evicted_tokens = (
                num_new_tokens + cache_end - kv_cache_size
            )
            num_rolled_tokens = (
                cache_end - num_evicted_tokens - sink_tokens
            )
            cache = belief_kv_weight_cache["preserve_action"]
            cache[
                :,
                sink_tokens:sink_tokens + num_rolled_tokens,
            ] = cache[
                :,
                sink_tokens + num_evicted_tokens:
                sink_tokens + num_evicted_tokens + num_rolled_tokens,
            ].clone()
            local_end_index = (
                cache_end
                + current_end
                - cache_global_end
                - num_evicted_tokens
            )
        else:
            local_end_index = (
                cache_end + current_end - cache_global_end
            )
        local_start_index = local_end_index - num_new_tokens
        belief_kv_weight_cache["preserve_action"][
            :, local_start_index:local_end_index
        ] = current_preserve_action.float().clamp(0.0, 1.0)
        belief_kv_weight_cache["global_end_index"].fill_(current_end)
        belief_kv_weight_cache["local_end_index"].fill_(local_end_index)

    def _update_target_owned_mask_cache(
        self,
        target_owned_mask_cache,
        current_target_owned_mask,
        kv_cache_trg,
    ):
        self._update_aligned_token_cache(
            cache_state=target_owned_mask_cache,
            cache_name="target_owned_mask",
            current_tokens=current_target_owned_mask.detach().bool(),
            kv_cache_trg=kv_cache_trg,
        )

    def _update_aligned_token_cache(
        self,
        cache_state,
        cache_name,
        current_tokens,
        kv_cache_trg,
    ):
        if current_tokens.ndim != 2:
            raise ValueError(
                "Aligned cache tokens must have shape [B,L]"
            )
        current_end = kv_cache_trg[0]["global_end_index"].item()
        sink_tokens = kv_cache_trg[0]["sink_tokens"]
        cache = cache_state[cache_name]
        kv_cache_size = cache.shape[1]
        num_new_tokens = current_tokens.shape[1]
        if num_new_tokens != kv_cache_trg[0]["num_new_tokens"]:
            raise ValueError(
                "Aligned cache and target KV write must have the "
                "same number of tokens"
            )
        cache_end = cache_state["local_end_index"].item()
        cache_global_end = cache_state["global_end_index"].item()
        if (
            self.local_attn_size != -1
            and current_end > cache_global_end
            and num_new_tokens + cache_end > kv_cache_size
        ):
            num_evicted_tokens = (
                num_new_tokens + cache_end - kv_cache_size
            )
            num_rolled_tokens = (
                cache_end - num_evicted_tokens - sink_tokens
            )
            cache[
                :, sink_tokens:sink_tokens + num_rolled_tokens
            ] = cache[
                :,
                sink_tokens + num_evicted_tokens:
                sink_tokens + num_evicted_tokens + num_rolled_tokens,
            ].clone()
            local_end_index = (
                cache_end + current_end
                - cache_global_end - num_evicted_tokens
            )
        else:
            local_end_index = (
                cache_end + current_end - cache_global_end
            )
        local_start_index = local_end_index - num_new_tokens
        cache[:, local_start_index:local_end_index] = current_tokens
        cache_state["global_end_index"].fill_(current_end)
        cache_state["local_end_index"].fill_(local_end_index)


    def _concat_kv_cache(self, kvc_1, kvc_2, index_select=-1, shared_dict=None):
        '''
        ✨ concat source and target kv cache at batch dim for dual branch sampling
        '''
        kv_cache1 = []
        if index_select == -1:
            index_kvc = kvc_2
        else:
            index_kvc = kvc_1
        for b_idx in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.cat((kvc_1[b_idx]["k"], kvc_2[b_idx]["k"]), dim=0).clone(),
                "v": torch.cat((kvc_1[b_idx]["v"], kvc_2[b_idx]["v"]), dim=0).clone(),
                "global_end_index": index_kvc[b_idx]["global_end_index"].clone(),
                "local_end_index": index_kvc[b_idx]["local_end_index"].clone(),
                "shared_dict": shared_dict,
                "layer_index": b_idx,
            })
        return kv_cache1

    def _append_clean_src_kv_cache(self, kvc_dual, kvc_src):
        '''
        ✨ add clean src kv cache to dual cache dict
        '''
        for b_idx in range(self.num_transformer_blocks):
            kvc_dual[b_idx].update({
                'k_src_clean': kvc_src[b_idx]['k'],
                'v_src_clean': kvc_src[b_idx]['v'],
            })

    def _inject_masks_to_kv_cache(
        self, kv_cache, 
        trg_fg_mask_cache=None, current_src_fg_mask=None,
        belief_kv_weight_cache=None,
        factorized_operator_cache=None,
        current_factorized_operators=None,
        target_owned_mask_cache=None,
        current_target_owned_mask=None,
        current_identity_read_mask=None,
        current_causal_owner_mask=None,
    ):
        '''
        ✨
        trg_fg_mask: [B, kv_cache_size], previous chunks' foreground mask.
        current_src_fg_mask: [B, lq], current chunk's foreground mask.
        '''
        for b_idx in range(self.num_transformer_blocks):
            kv_cache[b_idx].update({
                "trg_fg_mask": trg_fg_mask_cache['trg_fg_mask'],
                "current_src_fg_mask": current_src_fg_mask,
            })
            if belief_kv_weight_cache is not None:
                kv_cache[b_idx].update({
                    "cached_preserve_kv_action": (
                        belief_kv_weight_cache["preserve_action"]
                    ),
                })
            factorized_names = (
                "source_key_action",
                "source_value_action",
                "target_memory_action",
                "unknown_action",
            )
            if factorized_operator_cache is not None:
                if current_factorized_operators is None:
                    raise ValueError(
                        "Current factorized operators are required with "
                        "their history cache"
                    )
                for name in factorized_names:
                    kv_cache[b_idx][f"cached_{name}"] = (
                        factorized_operator_cache[name]
                    )
                    kv_cache[b_idx][f"current_{name}"] = getattr(
                        current_factorized_operators, name
                    )
            if current_identity_read_mask is not None:
                kv_cache[b_idx][
                    "current_identity_read_mask"
                ] = current_identity_read_mask
            else:
                kv_cache[b_idx].pop(
                    "current_identity_read_mask",
                    None,
                )
            if current_causal_owner_mask is not None:
                kv_cache[b_idx][
                    "current_causal_owner_mask"
                ] = current_causal_owner_mask
            else:
                kv_cache[b_idx].pop(
                    "current_causal_owner_mask",
                    None,
                )
            if target_owned_mask_cache is not None:
                if current_target_owned_mask is None:
                    raise ValueError(
                        "Current ownership mask is required with its "
                        "history cache"
                    )
                kv_cache[b_idx].update({
                    "target_owned_history_mask": (
                        target_owned_mask_cache[
                            "target_owned_mask"
                        ]
                    ),
                    "current_target_owned_mask": (
                        current_target_owned_mask
                    ),
                })

    @staticmethod
    def _register_identity_key_capture(kv_cache, layers):
        for layer_index in layers:
            kv_cache[layer_index][
                "capture_current_identity_key"
            ] = True

    @staticmethod
    def _collect_identity_keys(kv_cache, layers):
        keys = {}
        for layer_index in layers:
            key = kv_cache[layer_index].get(
                "current_identity_key"
            )
            if key is None:
                raise RuntimeError(
                    "Missing captured source identity key at layer "
                    f"{layer_index}"
                )
            keys[layer_index] = key
        return keys

    @staticmethod
    def _register_query_capture(kv_cache, layers):
        for layer_index in layers:
            kv_cache[layer_index]["capture_current_query"] = True

    @staticmethod
    def _aggregate_query_features(kv_cache, layers):
        features = []
        for layer_index in layers:
            feature = kv_cache[layer_index].get("current_query_feature")
            if feature is None:
                raise RuntimeError(
                    f"Missing captured source query at layer {layer_index}"
                )
            features.append(F.normalize(feature.float(), dim=-1))
        return F.normalize(torch.stack(features).mean(dim=0), dim=-1)
    
    def _kv_cache_to(self, kvc, device, low_memory):
        if not low_memory:
            return
        for itm in kvc:
            for k, v in itm.items():
                if isinstance(v, torch.Tensor):
                    v.to(device)


    def _register_crossattn_enhancement(self, crossattn_cache, fg_indices, fg_boost_factor=1.0, layers=range(30), current_src_fg_mask=None):
        '''
        ✨ default [src, trg] for multiple batches
        '''
        if layers is None:
            # all layers
            layers = range(self.num_transformer_blocks)
        for l_idx in layers:
            crossattn_cache[l_idx]["fg_indices"] = fg_indices
            crossattn_cache[l_idx]["fg_boost_factor"] = fg_boost_factor
            crossattn_cache[l_idx]["current_src_fg_mask"] = current_src_fg_mask
            crossattn_cache[l_idx]["apply_enhance"] = True
        
    def _register_crossattn_mask_gatherer(self, crossattn_cache, fg_indices, fg_scale=1.0, layers=range(20)):
        '''
        ✨ fg_indices will be poped in blocks to avoid repeating
        '''
        if layers is None:
            # all layers
            layers = range(self.num_transformer_blocks)
        for l_idx in layers:
            crossattn_cache[l_idx]["fg_indices"] = fg_indices
            crossattn_cache[l_idx]["fg_scale"] = fg_scale
            crossattn_cache[l_idx]["obtain_mask"] = True

    def _register_semantic_crossattn_gatherer(
        self, crossattn_cache, semantic_group_indices, layers=range(20)
    ):
        """Request independent target-prompt phrase maps in one pass."""
        if crossattn_cache is None:
            raise ValueError("Semantic cross-attention cache is required")
        if not semantic_group_indices:
            raise ValueError("Semantic token-index groups are required")
        if layers is None:
            layers = range(self.num_transformer_blocks)
        for layer_index in layers:
            state = crossattn_cache[layer_index]
            state["semantic_group_indices"] = semantic_group_indices
            state["obtain_semantic_masks"] = True

    def _aggregate_semantic_crossattn_masks(self, crossattn_cache):
        """Average each named semantic map over requested layers."""
        grouped = {}
        for layer_index in range(self.num_transformer_blocks):
            semantic_masks = crossattn_cache[layer_index].get(
                "semantic_masks_soft"
            )
            if semantic_masks is None:
                continue
            for name, value in semantic_masks.items():
                grouped.setdefault(name, []).append(
                    value.squeeze(-1).squeeze(-1).float()
                )
        if not grouped:
            raise RuntimeError(
                "No target semantic cross-attention maps were captured"
            )
        return {
            name: torch.stack(values, dim=0).mean(dim=0)
            for name, values in grouped.items()
        }

    def _aggregate_crossattn_mask(self, crossattn_cache, size=None, patch=(1, 2, 2), scale_factor=1):
        '''
        ✨
        size: (Ttok, Htok, Wtok), for visualization only. \\
        patch: patchify kernel size. \\
        
        crossattn_cache[l_idx]["fg_mask_soft"]: [B, Lq, 1, 1] \\
        return:
            mask_soft, mask_bin: [B, Lq]
            mask_soft_vis, mask_bin_vis: [B, Ttok, Htok, Wtok]
        '''
        total_mask = 0
        account = 0
        for l_idx in range(self.num_transformer_blocks):
            if "fg_mask_soft" in crossattn_cache[l_idx]:
                total_mask += crossattn_cache[l_idx]["fg_mask_soft"].squeeze(-1).squeeze(-1)
                account += 1
        mask_soft = total_mask / account
        mask_bin = mask_soft > 0
        if size is None:
            mask_soft_vis = None
            mask_bin_vis = None
        else:
            view_size = (total_mask.size(0), *map(lambda s, p: s // p, size, patch))
            mask_soft_vis = mask_soft.view(view_size)
            mask_bin_vis = mask_bin.view(view_size)
            if scale_factor != 1:
                mask_soft_vis = F.interpolate(mask_soft_vis, scale_factor=scale_factor).to(mask_soft_vis)
                mask_bin_vis = F.interpolate(mask_bin_vis.float(), scale_factor=scale_factor) > 0.5
        return mask_soft, mask_bin, mask_soft_vis, mask_bin_vis

    def _mask_reshape(self, mask_seq, size, patch=(1, 2, 2), scale_factor=2):
        '''
        ✨
        mask_seq: [B, Lq]
        mask_map: [B, Ttok, Htok, Wtok]
        '''
        view_size = (mask_seq.size(0), *map(lambda s, p: s // p, size, patch))
        mask_map = mask_seq.view(view_size)
        if scale_factor != 1:
            mask_map = F.interpolate(mask_map.float(), scale_factor=scale_factor) > 0.5
        return mask_map


    def _initialize_noise_statistics(
        self, reuse_noise_temporal_mean=False
    ):
        if reuse_noise_temporal_mean:
            self.noise_temporal_mean = dict()
            self.noise_temporal_mean_fg = dict()
            self.noise_temporal_mean_bg = dict()
        else:
            self.noise_temporal_mean = None
            self.noise_temporal_mean_fg = None
            self.noise_temporal_mean_bg = None

    def _reuse_noise_statistics(
        self, noise: torch.Tensor, step_idx: int, 
        ema_factor: float = 0.5, fg_mask=None,
        alpha_prog=2, alpha_mixed=1,
    ):
        if self.noise_temporal_mean is not None:
            if step_idx not in self.noise_temporal_mean.keys():
                self.noise_temporal_mean[step_idx] = noise
            else:
                previous_noise = self.noise_temporal_mean[step_idx]
                if previous_noise.shape[1] != noise.shape[1]:
                    previous_noise = previous_noise.mean(
                        dim=1,
                        keepdim=True,
                    ).expand(
                        -1,
                        noise.shape[1],
                        -1,
                        -1,
                        -1,
                    )
                noise_normalizer = (1 + alpha_prog ** 2) ** 0.5
                noise = (
                    previous_noise.flip(1)
                    * alpha_prog
                    / noise_normalizer
                    + noise / noise_normalizer
                )
                self.noise_temporal_mean[step_idx] = noise
        
        return noise
