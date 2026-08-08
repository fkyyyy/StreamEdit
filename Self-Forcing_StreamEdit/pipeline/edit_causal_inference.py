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
from .utils import find_phrase_token_indices
from .contact_graph import (
    CONTACT_GRAPH_MODES,
    build_oracle_contact_graphs,
    contact_graph_stats,
)
from .belief_kv import build_belief_kv_weights
from .control_belief import CausalControlBeliefBuilder
from .hand_role_inference import HandRoleInferencer
from .role_router import (
    BayesResidualFlowRouter,
    PosteriorResidualFlowRouter,
    ResidualRoleFlowRouter,
    RoleFlowRouter,
    build_oracle_roles,
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
        oracle_object_mask: Optional[torch.Tensor] = None,
        oracle_hand_mask: Optional[torch.Tensor] = None,
        hand_only_mask: Optional[torch.Tensor] = None,
        role_boundary_radius: int = 1,
        contact_target_weight: float = 0.7,
        posterior_flow_mode: str = "soft",
        posterior_flow_use_field: bool = False,
        hand_posterior_threshold: float = 0.20,
        hand_max_object_coverage: float = 0.18,
        hand_proximity_radius: int = 3,
        hand_propagation_steps: int = 2,
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
    ) -> torch.Tensor:
        expected_role_shape = (
            src_video.shape[0], src_video.shape[1],
            src_video.shape[-2], src_video.shape[-1],
        )
        for name, mask in (
            ("oracle_object_mask", oracle_object_mask),
            ("oracle_hand_mask", oracle_hand_mask),
            ("hand_only_mask", hand_only_mask),
        ):
            if mask is not None and tuple(mask.shape) != expected_role_shape:
                raise ValueError(
                    f"{name} must have shape {expected_role_shape}, "
                    f"got {tuple(mask.shape)}"
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
                oracle_object_mask=oracle_object_mask,
                oracle_hand_mask=oracle_hand_mask,
                hand_only_mask=hand_only_mask,
                role_boundary_radius=role_boundary_radius,
                contact_target_weight=contact_target_weight,
                posterior_flow_mode=posterior_flow_mode,
                posterior_flow_use_field=posterior_flow_use_field,
                hand_posterior_threshold=hand_posterior_threshold,
                hand_max_object_coverage=hand_max_object_coverage,
                hand_proximity_radius=hand_proximity_radius,
                hand_propagation_steps=hand_propagation_steps,
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
                contact_graph_mode=contact_graph_mode,
                contact_graph_topk=contact_graph_topk,
                contact_graph_radius=contact_graph_radius,
                contact_graph_min_confidence=contact_graph_min_confidence,
                contact_graph_strength=contact_graph_strength,
                contact_graph_layer_start=contact_graph_layer_start,
                contact_graph_layer_end=contact_graph_layer_end,
                contact_graph_seed=contact_graph_seed,
                save_role_dir=save_role_dir,
            )

        rollout_overlap = rollout_overlap_block_num * self.num_frame_per_block
        rollout_hand_role_inferencer = _hand_role_inferencer
        if (
            rollout_hand_role_inferencer is None
            and routing_mode in {
                "hand_role_adaptive_kv",
                "hand_role_posterior_flow_kv",
                "hand_role_bayes_flow_kv",
                "hand_role_bayes_flow_dual_kv",
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
                adaptive=(
                    routing_mode in {
                        "hand_role_adaptive_kv",
                        "hand_role_posterior_flow_kv",
                        "hand_role_bayes_flow_kv",
                        "hand_role_bayes_flow_dual_kv",
                    }
                ),
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
                rollout_src_video = src_video[:, start_idx: chunk_right_idx]
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
            else:
                rollout_src_video = src_video[:, start_idx + rollout_overlap: chunk_right_idx]
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
                oracle_object_mask=rollout_object_mask,
                oracle_hand_mask=rollout_hand_mask,
                hand_only_mask=rollout_hand_only_mask,
                role_boundary_radius=role_boundary_radius,
                contact_target_weight=contact_target_weight,
                posterior_flow_mode=posterior_flow_mode,
                posterior_flow_use_field=posterior_flow_use_field,
                hand_posterior_threshold=hand_posterior_threshold,
                hand_max_object_coverage=hand_max_object_coverage,
                hand_proximity_radius=hand_proximity_radius,
                hand_propagation_steps=hand_propagation_steps,
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
        oracle_object_mask: Optional[torch.Tensor] = None,
        oracle_hand_mask: Optional[torch.Tensor] = None,
        hand_only_mask: Optional[torch.Tensor] = None,
        role_boundary_radius: int = 1,
        contact_target_weight: float = 0.7,
        posterior_flow_mode: str = "soft",
        posterior_flow_use_field: bool = False,
        hand_posterior_threshold: float = 0.20,
        hand_max_object_coverage: float = 0.18,
        hand_proximity_radius: int = 3,
        hand_propagation_steps: int = 2,
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
    ) -> torch.Tensor:
        assert not (independent_first_frame and triple_first_frame)
        independent_first_frame = independent_first_frame or self.independent_first_frame
        
        batch_size, num_frames, num_channels, height, width = src_video.shape
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
        }
        adaptive_role_enabled = routing_mode in {
            "hand_role_adaptive_kv",
            "hand_role_posterior_flow_kv",
            "hand_role_bayes_flow_kv",
            "hand_role_bayes_flow_dual_kv",
        }
        posterior_flow_enabled = (
            routing_mode == "hand_role_posterior_flow_kv"
        )
        bayes_flow_enabled = (
            routing_mode in {
                "hand_role_bayes_flow_kv",
                "hand_role_bayes_flow_dual_kv",
            }
        )
        belief_dual_kv_enabled = (
            routing_mode == "hand_role_bayes_flow_dual_kv"
        )
        consistent_role_kv_enabled = oracle_kv_enabled or hand_role_enabled
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
        }:
            raise ValueError(f"Unsupported routing_mode: {routing_mode}")
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
            if bayes_flow_enabled:
                print(
                    "BAYES_RESIDUAL_FLOW "
                    "beliefs=non_exclusive "
                    "precision=online_fp32 "
                    "field_role=precision_only"
                )
                if belief_dual_kv_enabled:
                    print(
                        "BELIEF_DUAL_KV "
                        "target_memory=edit_belief "
                        "source_memory=preserve_belief "
                        "conflict=dual_active"
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
        # #region debug-point B:network-reporter
        def _debug_report(hypothesis_id, location, msg, data):
            try:
                import json
                import time
                import urllib.request
                url = os.environ.get(
                    "DEBUG_SERVER_URL",
                    "http://10.74.55.101:7777/event",
                )
                payload = {
                    "sessionId": os.environ.get(
                        "DEBUG_SESSION_ID", "oracle-edit-collapse"
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
        belief_kv_weight_cache = (
            self._initialize_belief_kv_weight_cache(
                batch_size=batch_size,
                device=src_video.device,
            )
            if belief_dual_kv_enabled
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
                adaptive=adaptive_role_enabled,
            )

        # get trigger token indices
        trans_tokenizer = self.text_encoder.tokenizer.tokenizer
        tok_src = find_phrase_token_indices(trans_tokenizer, src_prompts, src_trigger_words)
        tok_trg = find_phrase_token_indices(trans_tokenizer, trg_prompts, trg_trigger_words)
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
                self.generator(
                    noisy_image_or_video=current_src_ref_latents,
                    conditional_dict=src_conditional_dict,
                    timestep=context_timestep,
                    kv_cache=kv_cache_src,
                    crossattn_cache=crossattn_cache_src,
                    current_start=left * self.frame_seq_length,
                )
                _, src_fg_mask_bin, _, _ = self._aggregate_crossattn_mask(crossattn_cache_src)
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
                _, trg_fg_mask_bin, _, _ = self._aggregate_crossattn_mask(crossattn_cache_trg)

                #✨ src & trg union
                current_trg_fg_mask = trg_fg_mask_bin | src_fg_mask_bin
                self._update_trg_fg_mask_cache(trg_fg_mask_cache, current_trg_fg_mask, kv_cache_trg)
                if belief_dual_kv_enabled:
                    self._update_belief_kv_weight_cache(
                        belief_kv_weight_cache,
                        current_edit_weight=torch.ones_like(
                            current_trg_fg_mask,
                            dtype=torch.float32,
                        ),
                        current_preserve_weight=torch.ones_like(
                            current_trg_fg_mask,
                            dtype=torch.float32,
                        ),
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
            
            # obtain currently inprocessed kv_cache for dual branch
            shared_dict_dual = dict()
            kv_cache_dual = self._concat_kv_cache(kv_cache_src, kv_cache_trg, shared_dict=shared_dict_dual)

            #✨ forward clean source video to get source mask, and store into kv_cache
            self._register_crossattn_mask_gatherer(crossattn_cache_src, tok_src, layers=mask_layers, fg_scale=fg_scale)
            if hand_role_enabled:
                self._register_query_capture(
                    kv_cache_src,
                    hand_query_layers,
                )
            self.generator(
                noisy_image_or_video=src_input,
                conditional_dict=src_conditional_dict,
                timestep=context_timestep,
                kv_cache=kv_cache_src,
                crossattn_cache=crossattn_cache_src,
                current_start=current_start_frame * self.frame_seq_length,
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
            current_roles = None
            role_edit_tokens = None
            contact_graphs = None
            hand_role_debug = None
            hand_role_inference = None
            current_control_belief = None
            current_belief_kv_weights = None
            if oracle_role_enabled:
                role_left = current_start_frame - num_input_frames
                role_right = role_left + current_num_frames
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
                role_left = current_start_frame - num_input_frames
                role_right = role_left + current_num_frames
                hand_role_inference = hand_role_inferencer(
                    source_attention=src_fg_mask_soft,
                    hand_mask=hand_only_mask[:, role_left:role_right],
                    source_features=source_query_features,
                )
                current_roles = hand_role_inference.roles
                hand_role_debug = hand_role_inference.debug
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
                if belief_dual_kv_enabled:
                    current_control_belief = control_belief_builder(
                        debug=hand_role_debug,
                        hand_mask=hand_only_mask[
                            :, role_left:role_right
                        ],
                    )
                    current_belief_kv_weights = build_belief_kv_weights(
                        current_control_belief,
                        expected_token_length=role_edit_tokens.shape[1],
                    )
            shared_dict_dual.update({
                "contact_graph_mode": contact_graph_mode,
                "contact_graphs": contact_graphs,
                "contact_graph_strength": contact_graph_strength,
                "contact_graph_layer_start": contact_graph_layer_start,
                "contact_graph_layer_end": contact_graph_layer_end,
            })
            effective_src_fg_mask = (
                role_edit_tokens
                if consistent_role_kv_enabled
                else src_fg_mask_bin
            )
            self._inject_masks_to_kv_cache(
                kv_cache_dual,
                trg_fg_mask_cache,
                effective_src_fg_mask,
                belief_kv_weight_cache=(
                    belief_kv_weight_cache
                    if belief_dual_kv_enabled
                    else None
                ),
                current_belief_kv_weights=(
                    current_belief_kv_weights
                    if belief_dual_kv_enabled
                    else None
                ),
            )
            src_fg_mask_map = self._mask_reshape(
                effective_src_fg_mask,
                size=(current_num_frames, height, width),
            )
            inloop_trg_fg_mask = effective_src_fg_mask
            
            # Step 3.1: Spatial denoising loop
            noisy_pred_input = None
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
                velocity_pred, _ = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=kv_cache_dual,
                    crossattn_cache=crossattn_cache_dual,
                    current_start=current_start_frame * self.frame_seq_length
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
                        (
                            adaptive_role_enabled
                            and (
                                (
                                    not posterior_flow_enabled
                                    and not bayes_flow_enabled
                                )
                                or (
                                    posterior_flow_enabled
                                    and posterior_flow_use_field
                                )
                            )
                        )
                        or (
                            not bayes_flow_enabled
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
                            apply_update=apply_field_update,
                        )
                    )
                    hand_role_debug = hand_role_inference.debug
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
                        inloop_trg_fg_mask = role_edit_tokens
                        src_fg_mask_map = self._mask_reshape(
                            role_edit_tokens,
                            size=(current_num_frames, height, width),
                        )
                        self._inject_masks_to_kv_cache(
                            kv_cache_dual,
                            trg_fg_mask_cache,
                            role_edit_tokens,
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
                if bayes_flow_enabled and index == 0:
                    current_control_belief = control_belief_builder(
                        debug=hand_role_debug,
                        hand_mask=hand_only_mask[
                            :, role_left:role_right
                        ],
                    )
                    hand_role_debug.update({
                        f"control_{name}": value
                        for name, value
                        in current_control_belief.as_dict().items()
                    })
                    if belief_dual_kv_enabled:
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
                        self._inject_masks_to_kv_cache(
                            kv_cache_dual,
                            trg_fg_mask_cache,
                            role_edit_tokens,
                            belief_kv_weight_cache=(
                                belief_kv_weight_cache
                            ),
                            current_belief_kv_weights=(
                                current_belief_kv_weights
                            ),
                        )
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
                    if belief_dual_kv_enabled:
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
                            "conflict="
                            f"{current_belief_kv_weights.conflict_map.mean().item():.4f}"
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
                
                if bayes_flow_enabled:
                    if current_control_belief is None:
                        raise RuntimeError(
                            "Missing causal control belief for Bayes routing"
                        )
                    v_t, bayes_flow_debug = bayes_residual_flow_router(
                        target_velocity=v_trg,
                        source_velocity=v_src,
                        source_reconstruction_velocity=v_gt,
                        belief=current_control_belief,
                    )
                    if index == 0:
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
                    v_t = v_trg + bg_mask * (v_gt - v_src)
                denoised_pred = noisy_pred_input - t_i * v_t

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
                    if consistent_role_kv_enabled:
                        inloop_trg_fg_mask = role_edit_tokens
                    self._inject_masks_to_kv_cache(
                        kv_cache_dual, trg_fg_mask_cache, inloop_trg_fg_mask, 
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            del kv_cache_dual
            self._kv_cache_to(kv_cache_trg, 'cuda', low_memory)
            self._register_crossattn_mask_gatherer(crossattn_cache_trg, tok_trg, layers=mask_layers, fg_scale=fg_scale)
            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=trg_conditional_dict,
                timestep=context_timestep,
                kv_cache=kv_cache_trg,
                crossattn_cache=crossattn_cache_trg,
                current_start=current_start_frame * self.frame_seq_length,
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
            # #endregion
            #✨ store clean target kv cache, and obtain clean target mask
            _, trg_fg_mask_bin, _, _ = self._aggregate_crossattn_mask(crossattn_cache_trg)
            current_trg_fg_mask = (
                role_edit_tokens
                if consistent_role_kv_enabled
                else trg_fg_mask_bin | src_fg_mask_bin
            )
            self._update_trg_fg_mask_cache(trg_fg_mask_cache, current_trg_fg_mask, kv_cache_trg)
            if belief_dual_kv_enabled:
                if current_belief_kv_weights is None:
                    raise RuntimeError(
                        "Missing belief KV weights for dual-KV cache update"
                    )
                self._update_belief_kv_weight_cache(
                    belief_kv_weight_cache,
                    current_edit_weight=(
                        current_belief_kv_weights.edit
                    ),
                    current_preserve_weight=(
                        current_belief_kv_weights.preserve
                    ),
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
    ):
        np.savez_compressed(
            os.path.join(
                save_dir,
                f"block_{block_index:03d}_hand_role_debug.npz",
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
                    f"block_{block_index:03d}_{name}.png",
                )
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
            "edit": torch.zeros(
                [batch_size, kv_cache_size],
                dtype=torch.float32,
                device=device,
            ),
            "preserve": torch.zeros(
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
        current_edit_weight,
        current_preserve_weight,
        kv_cache_trg,
    ):
        if current_edit_weight.shape != current_preserve_weight.shape:
            raise ValueError(
                "Current edit and preserve KV weights must share shape"
            )
        if current_edit_weight.ndim != 2:
            raise ValueError(
                "Current belief KV weights must have shape [B,L]"
            )
        current_end = kv_cache_trg[0]["global_end_index"].item()
        sink_tokens = kv_cache_trg[0]["sink_tokens"]
        kv_cache_size = belief_kv_weight_cache["edit"].shape[1]
        num_new_tokens = current_edit_weight.shape[1]
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
            for name in ("edit", "preserve"):
                cache = belief_kv_weight_cache[name]
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
        belief_kv_weight_cache["edit"][
            :, local_start_index:local_end_index
        ] = current_edit_weight.float().clamp(0.0, 1.0)
        belief_kv_weight_cache["preserve"][
            :, local_start_index:local_end_index
        ] = current_preserve_weight.float().clamp(0.0, 1.0)
        belief_kv_weight_cache["global_end_index"].fill_(current_end)
        belief_kv_weight_cache["local_end_index"].fill_(local_end_index)


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
        current_belief_kv_weights=None,
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
            if (
                belief_kv_weight_cache is not None
                and current_belief_kv_weights is not None
            ):
                kv_cache[b_idx].update({
                    "cached_edit_kv_weight": (
                        belief_kv_weight_cache["edit"]
                    ),
                    "cached_preserve_kv_weight": (
                        belief_kv_weight_cache["preserve"]
                    ),
                    "current_edit_kv_weight": (
                        current_belief_kv_weights.edit
                    ),
                    "current_preserve_kv_weight": (
                        current_belief_kv_weights.preserve
                    ),
                    "current_edit_kv_action": (
                        current_belief_kv_weights.edit_action
                    ),
                    "current_preserve_kv_action": (
                        current_belief_kv_weights.preserve_action
                    ),
                })

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
                noise = self.noise_temporal_mean[step_idx].flip(1) * alpha_prog / (1 + alpha_prog ** 2) ** 0.5 + \
                    noise * 1 / (1 + alpha_prog ** 2) ** 0.5
                self.noise_temporal_mean[step_idx] = noise
        
        return noise
