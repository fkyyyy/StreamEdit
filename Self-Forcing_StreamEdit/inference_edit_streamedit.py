import argparse
import math
import torch
import torch.nn.functional as F
import os
from pathlib import Path

import json
from collections import OrderedDict
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from scipy import ndimage
from einops import rearrange
import torch.distributed as dist
from torchvision import transforms
from torchvision.io import write_video

from pipeline import (
    EditCausalInferencePipeline
)
from pipeline.mask_alignment import (
    causal_vae_frame_groups,
    project_hand_evidence_to_causal_latents,
    project_visible_owner_to_causal_latents,
)
from pipeline.motion.causal_motion_owner import SourceFlowCache
from utils.misc import set_seed
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

from diffusers.utils import load_video


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt', encoding='utf-8') as handle:
        return json.load(handle, object_hook=OrderedDict)

def find_closest_num_frame(x, a=4, b=3):
    max_m = (x + a - 1) // (a * b)
    while max_m > 0:
        y = a * b * max_m - a + 1
        if y <= x:
            return y
        max_m -= 1


def build_white_mask(
    mask_video_path,
    source_frames,
    latent_shape,
    threshold,
    min_latent_coverage=0.0,
    mode="white",
    overlay_diff_threshold=24.0,
    recover_overlay_components=False,
):
    """Resample a white matte to source time and the Wan latent grid.

    ``mode="white"`` expects a normal binary/white matte. ``overlay_white``
    handles videos where the hand is painted white over the source frame by
    keeping only white pixels that differ from the source frame.
    """
    if mode not in {"white", "overlay_white"}:
        raise ValueError(f"Unsupported mask mode: {mode}")
    mask_video = load_video(mask_video_path)
    if not mask_video:
        raise RuntimeError(f"No frames decoded from mask video: {mask_video_path}")
    frame_indices = np.rint(
        np.linspace(0, len(mask_video) - 1, len(source_frames))
    ).astype(int)
    raw_masks = []
    resample_bilinear = getattr(Image, "Resampling", Image).BILINEAR
    for source_index, index in enumerate(frame_indices):
        rgb = np.asarray(mask_video[int(index)].convert("RGB"))
        white_mask = np.all(rgb >= threshold, axis=-1)
        if mode == "overlay_white":
            source_rgb_image = source_frames[source_index].convert("RGB")
            if source_rgb_image.size != (rgb.shape[1], rgb.shape[0]):
                source_rgb_image = source_rgb_image.resize(
                    (rgb.shape[1], rgb.shape[0]),
                    resample=resample_bilinear,
                )
            source_rgb = np.asarray(source_rgb_image)
            source_difference = np.abs(
                rgb.astype(np.int16) - source_rgb.astype(np.int16)
            ).mean(axis=-1)
            changed_white = white_mask & (
                source_difference >= overlay_diff_threshold
            )
            if recover_overlay_components and changed_white.any():
                # A white overlay can be identical to an already-white
                # object interior. Use changed pixels as seeds, then retain
                # their complete 8-connected white components. This recovers
                # the white bottle body without admitting unrelated white
                # scene regions such as the plastic bag or stove highlights.
                labels, _ = ndimage.label(
                    white_mask,
                    structure=np.ones((3, 3), dtype=np.uint8),
                )
                seeded_labels = []
                for label_index in np.unique(labels[changed_white]):
                    if label_index == 0:
                        continue
                    component = labels == label_index
                    component_size = int(component.sum())
                    changed_fraction = float(
                        changed_white[component].mean()
                    )
                    # Temporal resampling and compression make static white
                    # background regions acquire a few changed edge pixels.
                    # The painted object, by contrast, changes nearly its
                    # entire connected component. Reject small and weakly
                    # seeded components so the oracle cannot leak onto the
                    # window, plastic bag, or subtitles.
                    if (
                        component_size >= 32
                        and changed_fraction >= 0.85
                    ):
                        seeded_labels.append(label_index)
                white_mask = np.isin(labels, seeded_labels)
            else:
                white_mask = changed_white
        raw_masks.append(white_mask)
    pixel_mask = torch.from_numpy(np.stack(raw_masks)).unsqueeze(1).float()
    pixel_mask = F.interpolate(
        pixel_mask, size=(480, 832), mode="nearest"
    ) > 0.5

    _, latent_frames, _, latent_height, latent_width = latent_shape
    frame_groups = []
    temporal_groups = causal_vae_frame_groups(
        len(source_frames),
        latent_frames,
    )
    for left, right in temporal_groups:
        frame_groups.append(pixel_mask[left:right].amax(dim=0))
    latent_mask = torch.stack(frame_groups).float()
    latent_mask = F.interpolate(
        latent_mask,
        size=(latent_height, latent_width),
        mode="nearest",
    ) > 0.5
    if min_latent_coverage > 0:
        frame_coverage = latent_mask.float().mean(dim=(1, 2, 3))
        latent_mask[frame_coverage < min_latent_coverage] = False
    return pixel_mask.bool(), latent_mask.squeeze(1).bool()

def load_pipe(args):
    
    # Initialize distributed inference
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        set_seed(args.seed + local_rank)
    else:
        device = torch.device("cuda")
        local_rank = 0
        world_size = 1
        set_seed(args.seed)

    print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
    low_memory = get_cuda_free_memory_gb(gpu) < 40

    torch.set_grad_enabled(False)

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    # settings for editing
    config['guidance_scale'] = 1.0
    config['timestep_shift'] = args.flow_shift
    config['model_kwargs']['timestep_shift'] = args.flow_shift
    config['model_kwargs']['sink_size'] = getattr(args, 'sink_size', 0)
    config['denoising_step_list'] = np.arange(1000, 0, -1000 / args.step).astype(int).tolist()

    # Initialize pipeline, few-step method is unimplemented
    pipeline = EditCausalInferencePipeline(config, device=device)

    if args.checkpoint_path:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        pipeline.generator.load_state_dict(state_dict['generator' if not args.use_ema else 'generator_ema'])

    pipeline = pipeline.to(dtype=torch.bfloat16)
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)

    return pipeline, low_memory, device, local_rank


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--src_prompt", type=str, required=True)
    parser.add_argument("--trg_prompt", type=str, required=True)
    parser.add_argument("--src_word", type=str, required=True)
    parser.add_argument("--trg_word", type=str, required=True)
    
    # first frame condition, triple_first_frame=False for Self Forcing
    parser.add_argument("--first_frame_edit", type=str, default=None)
    parser.add_argument("--triple_first_frame", action="store_true", default=False)

    # hyper-parameters
    parser.add_argument("--fg_boost_factor", type=float, default=4.0, help='CrossAttn Boosting')
    parser.add_argument("--blend_power", type=float, default=2.0, help='rho')

    # model settings
    parser.add_argument("--step", type=int, default=15, help='1~1000')
    parser.add_argument("--flow_shift", type=float, default=1.0)

    # for Self-forcing rollout long video sampling
    parser.add_argument("--rollout_chunk_size", type=int, default=21)
    parser.add_argument("--rollout_overlap_block_num", type=int, default=1)

    parser.add_argument("--config_path", type=str, default='configs/self_forcing_dmd.yaml')
    parser.add_argument("--checkpoint_path", type=str, default='checkpoints/self_forcing_dmd.pt')
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--routing_mode",
        choices=[
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
        ],
        default="dynamic_sog",
    )
    parser.add_argument(
        "--identity_first_latent_bootstrap",
        action="store_true",
        default=False,
        help=(
            "Freeze target appearance from the first frame's low-noise "
            "object core and retrieve it across blocks with source keys."
        ),
    )
    parser.add_argument(
        "--object_wise_anchor_reset",
        action="store_true",
        default=False,
        help=(
            "After the first edited block, replace the provisional "
            "identity anchor with final clean target values only on the "
            "verified object core; background remains source-anchored."
        ),
    )
    parser.add_argument(
        "--target_owned_object_handoff",
        action="store_true",
        default=False,
        help=(
            "After object anchor reset, stop source Q/K/history "
            "injection on target-owned object tokens while retaining "
            "legacy source preservation on the background."
        ),
    )
    parser.add_argument(
        "--target_owned_min_similarity",
        type=float,
        default=0.55,
        help=(
            "Minimum cosine match to the committed ignition object "
            "core before a current token becomes target-owned."
        ),
    )
    parser.add_argument(
        "--first_chunk_identity_replay",
        action="store_true",
        default=False,
        help=(
            "Generate the first rollout once as a clean target identity "
            "proposal, freeze its multi-frame appearance bank, then "
            "regenerate the rollout from the same random state."
        ),
    )
    parser.add_argument(
        "--factorized_target_identity",
        action="store_true",
        default=False,
        help=(
            "Use source keys only for localization and target keys/values "
            "for soft appearance correction inside the current object core."
        ),
    )
    parser.add_argument(
        "--factorized_immutable_target_memory",
        action="store_true",
        default=False,
        help=(
            "For factorized causal ownership, freeze first-block target "
            "appearance statistics and use them to constrain later owner "
            "KV reads and writes."
        ),
    )
    parser.add_argument(
        "--factorized_native_target_history",
        action="store_true",
        default=False,
        help=(
            "Keep factorized Bayes role/velocity routing, but make the "
            "main self-attention output use StreamGVE's complete clean "
            "target KV history instead of provenance-pruned history."
        ),
    )
    parser.add_argument(
        "--factorized_owner_source_block",
        action="store_true",
        default=False,
        help=(
            "With native target-history attention, prohibit source "
            "appearance residuals on target-owned object tokens."
        ),
    )
    parser.add_argument(
        "--target_semantic_competition",
        action="store_true",
        default=False,
        help=(
            "Probe the clean source with target text, compete local edit "
            "phrases against preserve phrases, and use the result as "
            "part-level target velocity/KV authority inside the automatic "
            "hand-flow owner."
        ),
    )
    parser.add_argument(
        "--target_edit_phrases",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--target_preserve_phrases",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--target_semantic_margin", type=float, default=0.10
    )
    parser.add_argument(
        "--target_semantic_min_confidence", type=float, default=0.20
    )
    parser.add_argument(
        "--causal_paired_edit_memory",
        action="store_true",
        default=False,
        help=(
            "Read a sparse canonical target-minus-source value memory "
            "with clean-source keys on causal object tokens, and update "
            "it only through uncertainty-gated transactional commits."
        ),
    )
    parser.add_argument(
        "--paired_memory_layers",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Transformer layers carrying paired edit memory. Defaults "
            "to --hand_query_layers."
        ),
    )
    parser.add_argument(
        "--paired_memory_max_tokens", type=int, default=1536
    )
    parser.add_argument(
        "--paired_memory_max_tokens_per_block", type=int, default=192
    )
    parser.add_argument(
        "--paired_memory_topk", type=int, default=8
    )
    parser.add_argument(
        "--paired_memory_min_similarity", type=float, default=0.35
    )
    parser.add_argument(
        "--paired_memory_min_commit_confidence",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--paired_memory_coordinate_bias", type=float, default=1.0
    )
    parser.add_argument(
        "--paired_memory_coordinate_radius", type=float, default=0.0,
        help=(
            "Maximum normalized object-coordinate distance for paired "
            "retrieval. Zero preserves the unrestricted 928 behavior."
        ),
    )
    parser.add_argument(
        "--paired_memory_min_residual_consensus",
        type=float,
        default=0.0,
        help=(
            "Abstain from paired reads whose local top-k canonical "
            "residuals disagree in direction or magnitude."
        ),
    )
    parser.add_argument(
        "--paired_memory_source_part_consistency",
        action="store_true",
        default=False,
        help=(
            "Store a normalized clean-source value signature per "
            "canonical slot and require part-compatible retrieval. This "
            "separates object-internal parts without semantic labels."
        ),
    )
    parser.add_argument(
        "--paired_memory_min_part_similarity",
        type=float,
        default=0.45,
        help=(
            "Minimum clean-source value-signature cosine similarity for "
            "a canonical slot to represent the same latent object part."
        ),
    )
    parser.add_argument(
        "--paired_memory_part_similarity_margin",
        type=float,
        default=0.08,
        help=(
            "Keep only part candidates within this cosine margin of the "
            "best local source-part match."
        ),
    )
    parser.add_argument(
        "--paired_memory_read_strength", type=float, default=0.35
    )
    parser.add_argument(
        "--paired_memory_value_projection",
        action="store_true",
        default=False,
        help=(
            "Materialize paired target-minus-source residuals as current "
            "source values plus the canonical edit before attention and "
            "in the persistent clean target KV cache."
        ),
    )
    parser.add_argument(
        "--paired_memory_query_gated_projection",
        action="store_true",
        default=False,
        help=(
            "Compute native and paired-projected self-attention "
            "separately, then admit the projected-output difference only "
            "on queries with a successful paired-memory read."
        ),
    )
    parser.add_argument(
        "--paired_memory_disable_persistent_projection",
        action="store_true",
        default=False,
        help=(
            "Do not rewrite the persistent clean target KV cache with "
            "paired values. This isolates current-block query-gated reads "
            "from historical projected-value propagation."
        ),
    )
    parser.add_argument(
        "--paired_memory_source_suppression",
        type=float,
        default=0.0,
        help=(
            "Suppress competing source-reconstruction residuals only in "
            "proportion to successful paired-memory reads."
        ),
    )
    parser.add_argument(
        "--paired_memory_interior_projection",
        action="store_true",
        default=False,
        help=(
            "Restrict paired reads and persistent KV projection to the "
            "role-pure local object interior; boundary/contact tokens use "
            "the exact native path."
        ),
    )
    parser.add_argument(
        "--paired_memory_first_block_replay",
        action="store_true",
        default=False,
        help=(
            "Run the first model block once to build paired memory, then "
            "restore RNG state and regenerate it through the same read path "
            "used by later blocks."
        ),
    )
    parser.add_argument(
        "--paired_memory_source_transport",
        action="store_true",
        default=False,
        help=(
            "Move immutable edit-residual lineages through adjacent "
            "clean-source KV states before reading the canonical bank."
        ),
    )
    parser.add_argument(
        "--paired_memory_single_confidence",
        action="store_true",
        default=False,
        help=(
            "Apply paired-memory confidence exactly once: continuous "
            "confidence controls value projection, while query arbitration "
            "is a binary exact-native access policy. Transactional source-"
            "transport writes likewise avoid squaring the role proposal."
        ),
    )
    parser.add_argument(
        "--paired_memory_owner_attached_boundary",
        action="store_true",
        default=False,
        help=(
            "Extend the paired-memory projection gate from the strict "
            "interior to owner-attached object structure. Hand-dominant, "
            "background, unknown, and failed-correspondence tokens remain "
            "exact native fallback."
        ),
    )
    parser.add_argument(
        "--paired_memory_dual_timescale_anchor",
        action="store_true",
        default=False,
        help=(
            "Keep dense native target history as the motion branch, and "
            "read the immutable source-addressed appearance residual in a "
            "separate owner-only current-source attention branch."
        ),
    )
    parser.add_argument(
        "--paired_memory_canonical_key_anchor",
        action="store_true",
        default=False,
        help=(
            "Use replay-verified first-block pre-RoPE source keys and "
            "immutable target-minus-source values for the separate "
            "appearance cross-attention. Later blocks can transport "
            "lineage/admission but cannot append keys or overwrite values."
        ),
    )
    parser.add_argument(
        "--role_fixed_native_history",
        action="store_true",
        default=False,
        help=(
            "Read immutable ignition and recent final-clean native target "
            "K/V at fixed relative RoPE positions for source-matched object "
            "queries. Unsupported roles retain exact native history."
        ),
    )
    parser.add_argument(
        "--native_history_layers", type=int, nargs="+", default=None
    )
    parser.add_argument(
        "--native_history_max_tokens_per_frame", type=int, default=256
    )
    parser.add_argument("--native_history_topk", type=int, default=8)
    parser.add_argument(
        "--native_history_min_similarity", type=float, default=0.35
    )
    parser.add_argument(
        "--native_history_min_write_confidence",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--native_history_min_query_confidence",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--native_history_canonical_logit_bias",
        type=float,
        default=1.0,
        help=(
            "Finite logit prior for admitted immutable native target keys; "
            "it changes selection probability but never scales K or V."
        ),
    )
    parser.add_argument(
        "--native_history_coalesce_bootstrap_time",
        action="store_true",
        default=False,
        help=(
            "When canonical and recent native-KV tiers come from the same "
            "ignition commit, give them the same temporal RoPE origin and "
            "place the current block immediately after that real block."
        ),
    )
    parser.add_argument(
        "--native_history_bypass_blocks",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Zero-based causal block indices that keep exact native "
            "attention instead of applying the role-fixed KV replacement."
        ),
    )
    parser.add_argument(
        "--native_history_source_part_consistency",
        action="store_true",
        default=False,
        help=(
            "Use centered clean-source value signatures to softly "
            "reweight only canonical target-KV candidates. Recent and "
            "current keys remain intact, and the refinement is bounded "
            "relative to the native role-fixed read."
        ),
    )
    parser.add_argument(
        "--native_history_min_part_similarity",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--native_history_part_similarity_margin",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--native_history_part_bias_strength",
        type=float,
        default=0.5,
        help="Maximum signed logit bias applied to canonical candidates.",
    )
    parser.add_argument(
        "--native_history_part_refinement_ratio",
        type=float,
        default=0.25,
        help=(
            "Maximum RMS size of the part-aware refinement as a "
            "fraction of the original role-fixed memory residual."
        ),
    )
    parser.add_argument(
        "--native_history_transactional_owner",
        action="store_true",
        default=False,
        help=(
            "Use separate source-addressed native-KV read and write "
            "supports. Without a source-owner matte they are inferred "
            "from hand-conditioned roles, clean-source transport, and "
            "online velocity-field agreement; supplying a matte selects "
            "the oracle ablation."
        ),
    )
    parser.add_argument(
        "--hand_flow_transactional_owner",
        action="store_true",
        default=False,
        help=(
            "Enforce the deployable hand-only transactional-KV path. "
            "This mode rejects object masks and source-owner masks."
        ),
    )
    parser.add_argument(
        "--native_history_consistent_transaction",
        action="store_true",
        default=False,
        help=(
            "Make native target history obey one end-to-end transaction: "
            "compact write-approved recent payload, continuous source-"
            "addressed read strength, retrieval-coupled appearance "
            "arbitration, and block-wise lifecycle accounting."
        ),
    )
    parser.add_argument(
        "--native_history_verified_attention_authority",
        action="store_true",
        default=False,
        help=(
            "After all configured native-history layers agree on a "
            "source-addressed target-KV read, transfer only those query "
            "outputs from source-mixed native attention to factorized "
            "target-value attention. Unverified queries remain exact native "
            "fallback."
        ),
    )
    parser.add_argument(
        "--native_history_attention_authority_strength",
        type=float,
        default=1.0,
        help=(
            "Maximum native-to-factorized attention interpolation after a "
            "cross-layer verified target-KV read."
        ),
    )
    parser.add_argument(
        "--native_history_payload_invariant_lineage",
        action="store_true",
        default=False,
        help=(
            "Keep mutable native history source-address-only: propagate "
            "canonical slot lineage through clean-source K while reading "
            "target appearance exclusively from the immutable ignition "
            "payload."
        ),
    )
    parser.add_argument(
        "--native_history_payload_blend_strength",
        type=float,
        default=0.35,
        help=(
            "Bounded convex weight of immutable canonical target "
            "appearance on source-lineage-admitted queries."
        ),
    )
    parser.add_argument(
        "--native_history_recent_entry_bridge",
        action="store_true",
        default=False,
        help=(
            "Keep the complete previous timestep-zero clean target K/V as "
            "short-term state, read it only for the first latent frame of "
            "each new causal block through the automatic hand-flow owner, "
            "and use the immutable ignition tier only as fallback."
        ),
    )
    parser.add_argument(
        "--native_history_motion_owner_dense_read",
        action="store_true",
        default=False,
        help=(
            "Extend the source-addressed recent-target transaction from "
            "the chunk entry to every latent frame inside the automatic "
            "motion owner. This changes read authority only; KV writes "
            "remain gated by the conservative transactional core."
        ),
    )
    parser.add_argument(
        "--native_history_entry_bridge_strength",
        type=float,
        default=1.0,
        help=(
            "Maximum interpolation from native attention to the verified "
            "recent edited-state read at a causal-block entry."
        ),
    )
    parser.add_argument(
        "--native_history_dual_evidence_arbitration",
        action="store_true",
        default=False,
        help=(
            "Require both clean-source address correspondence and immutable-"
            "canonical edit-residual consistency before trusting mutable "
            "recent target K/V. Rejected recent payload falls back to the "
            "canonical target branch."
        ),
    )
    parser.add_argument(
        "--native_history_token_atomic_payload",
        action="store_true",
        default=False,
        help=(
            "Keep clean-source recent K as a dense address table, but "
            "authorize recent target K/V independently at each token using "
            "the conservative transactional write core. Queries without "
            "an authorized recent payload stay on exact native attention."
        ),
    )
    parser.add_argument(
        "--native_history_persistent_residual_upsert",
        action="store_true",
        default=False,
        help=(
            "Retain trusted recent target-minus-source KV residuals with a "
            "tokenwise source-address transaction. Current high-confidence "
            "writes replace individual tokens; flow-owner-supported matched "
            "tokens retain the previous value residual rebased on current "
            "source V while current target K keeps geometry. No object mask "
            "is consumed."
        ),
    )
    parser.add_argument(
        "--native_history_last_trusted_appearance",
        action="store_true",
        default=False,
        help=(
            "Keep a one-to-one last-trusted target-minus-source residual "
            "lineage inside the automatic hand-flow owner. A new direct "
            "write may replace an old residual only when direction and "
            "magnitude do not regress; verified reads protect the same "
            "token from source-appearance fallback."
        ),
    )
    parser.add_argument(
        "--native_history_flow_indexed_residual",
        action="store_true",
        default=False,
        help=(
            "Transport the last-trusted target-minus-source V residual "
            "with clean-source bidirectional optical flow and read it at "
            "the aligned current token, without target-K addressing."
        ),
    )
    parser.add_argument(
        "--native_history_decoupled_flow_trust",
        action="store_true",
        default=False,
        help=(
            "Keep target-appearance trust persistent across blocks and use "
            "bidirectional source-flow confidence only as block-local "
            "transport reliability. Prevents exponential confidence decay."
        ),
    )
    parser.add_argument(
        "--native_history_multiframe_identity_sink",
        action="store_true",
        default=False,
        help=(
            "Read identity only from the frozen multi-frame ignition "
            "canonical. Clean-source keys restrict candidates per frame; "
            "the current target query selects target-minus-source value "
            "evidence only inside those candidates."
        ),
    )
    parser.add_argument(
        "--native_history_multiframe_sink_topk_per_frame",
        type=int,
        default=8,
        help=(
            "Maximum source-key-authorized identity candidates retained "
            "from each immutable ignition frame per current query."
        ),
    )
    parser.add_argument(
        "--native_history_multiframe_sink_source_logit_bias",
        type=float,
        default=1.0,
        help=(
            "Finite clean-source similarity prior used after per-frame "
            "candidate restriction in the immutable identity sink."
        ),
    )
    parser.add_argument(
        "--native_history_multiframe_sink_strength",
        type=float,
        default=1.0,
        help=(
            "Maximum frozen multi-frame identity residual strength before "
            "automatic owner, appearance, and local transport gating."
        ),
    )
    parser.add_argument(
        "--native_history_timestep_counterfactual_memory",
        action="store_true",
        default=False,
        help=(
            "Capture immutable paired source/target K/V from B0 at every "
            "denoising timestep, then apply a source-flow-addressed "
            "desired-minus-current counterfactual feedback correction."
        ),
    )
    parser.add_argument(
        "--native_history_tccm_flow_radius",
        type=float,
        default=2.0,
        help=(
            "Maximum token-grid distance from the transported B0 "
            "coordinate for a canonical candidate."
        ),
    )
    parser.add_argument(
        "--native_history_tccm_strength",
        type=float,
        default=1.0,
        help="Maximum closed-loop counterfactual feedback gain.",
    )
    parser.add_argument(
        "--native_history_tccm_max_error_ratio",
        type=float,
        default=1.0,
        help=(
            "Clip the feedback-error norm to this multiple of the larger "
            "desired/current counterfactual-response norm."
        ),
    )
    parser.add_argument(
        "--native_history_flow_min_confidence",
        type=float,
        default=0.10,
        help=(
            "Minimum bidirectional source-flow confidence for a transported "
            "identity residual to remain readable."
        ),
    )
    parser.add_argument(
        "--native_history_residual_update_min_cosine",
        type=float,
        default=0.50,
        help=(
            "Minimum cosine agreement with a matched last-trusted residual "
            "required before a current direct write can replace it."
        ),
    )
    parser.add_argument(
        "--native_history_residual_update_min_magnitude_ratio",
        type=float,
        default=0.90,
        help=(
            "Minimum current/last-trusted residual norm ratio required "
            "before replacing the stored target appearance."
        ),
    )
    parser.add_argument(
        "--native_history_topology_complete_read",
        action="store_true",
        default=False,
        help=(
            "Add only closed interior holes of the transported automatic "
            "owner to native-KV read requests, gated by clean-source motion "
            "affinity and source semantics and excluding the hand mask. "
            "The write transaction is unchanged."
        ),
    )
    parser.add_argument(
        "--native_history_min_payload_consistency",
        type=float,
        default=0.15,
        help=(
            "Minimum tokenwise agreement between recent and immutable "
            "canonical target-minus-source value residuals."
        ),
    )
    parser.add_argument(
        "--native_history_dense_recent_min_residual_consensus",
        type=float,
        default=0.05,
        help=(
            "Minimum source-aligned edit-residual agreement with the "
            "immutable ignition block required to commit a new dense recent "
            "target state. Failed transactions hold the previous state."
        ),
    )
    parser.add_argument(
        "--native_history_owner_max_missing_frames",
        type=int,
        default=1,
        help=(
            "Maximum missing observations for read-only owner transport. "
            "The consistent-transaction mode counts causal chunks; legacy "
            "modes count latent frames."
        ),
    )
    parser.add_argument(
        "--native_history_verified_source_suppression",
        type=float,
        default=0.35,
        help=(
            "Maximum source-residual suppression at queries admitted by "
            "the source-addressed native-KV read."
        ),
    )
    parser.add_argument(
        "--paired_memory_transport_min_similarity",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--paired_memory_transport_coordinate_radius",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--paired_memory_transport_cycle_radius",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--paired_memory_transport_min_confidence",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--immutable_target_layers",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Transformer layers carrying immutable target appearance. "
            "Defaults to --hand_query_layers for backward compatibility."
        ),
    )
    parser.add_argument(
        "--immutable_target_num_prototypes",
        type=int,
        default=4,
        help=(
            "Number of first-chunk appearance prototypes per immutable "
            "target-memory layer."
        ),
    )
    parser.add_argument(
        "--immutable_target_value_mode",
        choices=("residual", "subspace", "absolute"),
        default="residual",
        help=(
            "Store target-minus-source values (legacy 920), constrain only "
            "their cross-prototype coherent appearance subspace, or store "
            "absolute edited target values (failed 921 ablation)."
        ),
    )
    parser.add_argument(
        "--immutable_target_hard_owner",
        action="store_true",
        default=False,
        help=(
            "Use complete owner-token support, rather than fractional mask "
            "coverage, as the immutable appearance read gate."
        ),
    )
    parser.add_argument(
        "--factorized_orthogonal_geometry",
        action="store_true",
        default=False,
        help=(
            "Inside causal object ownership, restore the source "
            "reconstruction residual after removing only the component "
            "opposing the target edit direction."
        ),
    )
    parser.add_argument(
        "--factorized_geometry_strength",
        type=float,
        default=1.0,
        help=(
            "Maximum early-denoising strength of orthogonal source geometry."
        ),
    )
    parser.add_argument(
        "--identity_correction_strength",
        type=float,
        default=0.35,
        help=(
            "Maximum target-appearance value correction in the verified "
            "object core; also controls source-coordinate residual reads."
        ),
    )
    parser.add_argument(
        "--identity_visibility_lifecycle",
        action="store_true",
        default=False,
        help=(
            "Gate factorized identity reads and writes with visible, "
            "occluded, and absent object states."
        ),
    )
    parser.add_argument(
        "--identity_max_occluded_blocks",
        type=int,
        default=1,
        help=(
            "Number of missing blocks retained as occluded before the "
            "object enters the absent state. Reads remain disabled while "
            "the object core is missing."
        ),
    )
    parser.add_argument(
        "--appearance_leakage_decomposition",
        action="store_true",
        default=False,
        help=(
            "Build a training-free interaction-conditioned target-change "
            "core and remove only source residual components that oppose "
            "the target edit direction."
        ),
    )
    parser.add_argument(
        "--source_coordinate_identity",
        action="store_true",
        default=False,
        help=(
            "Predict object ownership from clean source features before "
            "denoising and transport an immutable target-minus-source "
            "attention-value residual. Requires first-chunk replay and "
            "appearance-leakage decomposition."
        ),
    )
    parser.add_argument(
        "--identity_source_suppression",
        type=float,
        default=0.35,
        help=(
            "Maximum early-denoising source-residual suppression inside "
            "the transported identity owner; hand and background remain "
            "source-preserved."
        ),
    )
    parser.add_argument(
        "--identity_support_floor",
        type=float,
        default=0.0,
        help=(
            "Minimum identity correction confidence inside the clean-"
            "source owner. Attention similarity remains a soft modulation "
            "instead of a second spatial gate."
        ),
    )
    parser.add_argument(
        "--source_identity_residual_carry",
        action="store_true",
        default=False,
        help=(
            "Initialize each new block with a source-coordinate transport "
            "of the frozen first edited block's target-minus-source latent "
            "residual, restricted to the object owner."
        ),
    )
    parser.add_argument(
        "--identity_residual_carry_strength",
        type=float,
        default=0.25,
        help=(
            "Strength of object-only target-minus-source residual used for "
            "new-block initialization."
        ),
    )
    parser.add_argument(
        "--source_owner_residual_constraint",
        action="store_true",
        default=False,
        help=(
            "Apply the frozen source-coordinate target-minus-source "
            "residual as a late-rising proximal constraint after every "
            "denoising step. Requires residual carry and an explicit "
            "source owner mask."
        ),
    )
    parser.add_argument(
        "--identity_residual_constraint_strength",
        type=float,
        default=0.35,
        help=(
            "Final-step proximal strength for the object-only residual "
            "constraint."
        ),
    )
    parser.add_argument(
        "--identity_residual_constraint_power",
        type=float,
        default=2.0,
        help=(
            "Power of the late-denoising residual constraint schedule; "
            "larger values delay the constraint."
        ),
    )
    parser.add_argument(
        "--source_owner_geometry_envelope",
        action="store_true",
        default=False,
        help=(
            "For appearance-only editing, preserve the current clean-source "
            "latent outside the explicit source owner after each denoising "
            "step. The owner interior remains free for target appearance."
        ),
    )
    parser.add_argument(
        "--source_geometry_strength",
        type=float,
        default=0.35,
        help=(
            "Final-step source proximal strength outside the owner "
            "geometry envelope."
        ),
    )
    parser.add_argument(
        "--source_geometry_power",
        type=float,
        default=2.0,
        help=(
            "Power of the late-denoising source geometry schedule."
        ),
    )
    parser.add_argument(
        "--source_geometry_margin",
        type=int,
        default=1,
        help=(
            "Latent-pixel margin around the source owner left free for "
            "appearance boundary blending."
        ),
    )
    parser.add_argument(
        "--ignition_hand_exclusion_radius",
        type=int,
        default=1,
        help=(
            "Latent-grid radius protected around the hand when forming "
            "the target-change core."
        ),
    )
    parser.add_argument(
        "--ignition_contact_radius",
        type=int,
        default=3,
        help=(
            "Latent-grid radius used only to verify that a target-change "
            "component belongs to the hand-held object."
        ),
    )
    parser.add_argument(
        "--identity_tokenprop_min_similarity",
        type=float,
        default=0.55,
        help=(
            "Minimum source-query token cosine used by "
            "hand_role_bayes_flow_tokenprop_kv identity writes."
        ),
    )
    parser.add_argument(
        "--identity_tokenprop_gate_strength",
        type=float,
        default=0.85,
        help=(
            "How strongly causal token matching gates identity-memory writes."
        ),
    )
    parser.add_argument(
        "--identity_tokenprop_max_candidates",
        type=int,
        default=512,
        help=(
            "Maximum previous committed tokens used for online token matching."
        ),
    )
    parser.add_argument(
        "--committed_memory_feedback_strength",
        type=float,
        default=0.75,
        help=(
            "Strength for feeding transported committed edit memory back "
            "into the current control belief."
        ),
    )
    parser.add_argument("--object_mask_video", type=str, default=None)
    parser.add_argument(
        "--source_owner_mask_video",
        type=str,
        default=None,
        help=(
            "Clean-source object mask used only as the source-coordinate "
            "identity owner. This does not enable oracle role routing."
        ),
    )
    parser.add_argument(
        "--source_owner_mask_mode",
        choices=["white", "overlay_white"],
        default="overlay_white",
        help=(
            "How to parse --source_owner_mask_video. overlay_white "
            "removes white pixels already present in the clean source."
        ),
    )
    parser.add_argument(
        "--source_owner_overlay_diff_threshold",
        type=float,
        default=24.0,
        help=(
            "Mean RGB source difference required for an overlay-white "
            "source owner pixel."
        ),
    )
    parser.add_argument(
        "--source_owner_prepool_hand_exclusion",
        action="store_true",
        default=False,
        help=(
            "Remove the hand from the source-owner mask per pixel frame "
            "before causal VAE temporal pooling. This avoids eroding the "
            "object by the union of a moving hand trajectory."
        ),
    )
    parser.add_argument(
        "--causal_owner_consistent_kv_metadata",
        action="store_true",
        default=False,
        help=(
            "Use the same causal-owner role mask for target KV "
            "foreground metadata during denoising and after the clean "
            "target KV commit."
        ),
    )
    parser.add_argument(
        "--factorized_source_coordinate_target_delta",
        action="store_true",
        default=False,
        help=(
            "Reconstruct each frame from its clean-source velocity and "
            "apply the target-minus-source velocity only on factorized "
            "object/boundary support. This preserves source-coordinate "
            "geometry without fixing an object's absolute size."
        ),
    )
    parser.add_argument(
        "--factorized_owner_complement_source",
        action="store_true",
        default=False,
        help=(
            "Use clean-source reconstruction velocity strictly outside "
            "a dilated causal source-owner mask, while leaving the owner "
            "and its safety boundary on the normal factorized/native-KV "
            "path. This closes target-prompt leakage into background."
        ),
    )
    parser.add_argument(
        "--factorized_owner_complement_margin",
        type=int,
        default=1,
        help=(
            "Causal-owner-grid dilation radius retained on the normal "
            "editing path by --factorized_owner_complement_source."
        ),
    )
    parser.add_argument(
        "--factorized_owner_complement_min_preserve_confidence",
        type=float,
        default=0.0,
        help=(
            "Close an owner-complement pixel to clean source only when "
            "its factorized hand/background confidence reaches this "
            "threshold. Positive values preserve the native target/KV "
            "path for uncertain or occluded pixels."
        ),
    )
    parser.add_argument("--hand_mask_video", type=str, default=None)
    parser.add_argument(
        "--source_flow_cache",
        type=str,
        default=None,
        help=(
            "Read-only bidirectional RAFT cache computed from the clean "
            "source RGB video. No generated frame or object mask is used."
        ),
    )
    parser.add_argument(
        "--motion_geometry_owner",
        action="store_true",
        default=False,
        help=(
            "Transport the full hand-conditioned geometry owner with "
            "source RGB optical flow, independently of conservative KV writes."
        ),
    )
    parser.add_argument(
        "--source_flow_role_fusion",
        action="store_true",
        default=False,
        help=(
            "Fuse camera-compensated clean-source flow confidence and "
            "hand-relative motion into automatic object/background/"
            "boundary/unknown token-role evidence."
        ),
    )
    parser.add_argument(
        "--source_flow_role_weight",
        type=float,
        default=0.75,
        help="Maximum positive object recovery from source-flow evidence.",
    )
    parser.add_argument(
        "--source_flow_verified_region",
        action="store_true",
        default=False,
        help=(
            "Use token semantics as a proposal, verify it within a local "
            "clean-source flow-owner neighborhood, veto reliable flow "
            "background, and apply hard exclusion last."
        ),
    )
    parser.add_argument(
        "--source_flow_verified_owner_radius",
        type=int,
        default=1,
        help="Token-grid radius in which owner may verify semantic extent.",
    )
    parser.add_argument(
        "--source_flow_background_veto_threshold",
        type=float,
        default=0.55,
        help="Minimum clean-source background likelihood for a veto.",
    )
    parser.add_argument(
        "--source_flow_background_veto_min_confidence",
        type=float,
        default=0.50,
        help="Minimum flow confidence required for a background veto.",
    )
    parser.add_argument(
        "--soft_region_modulation",
        action="store_true",
        default=False,
        help=(
            "Use the flow-verified region as a continuous [0,1] spatial "
            "modulation signal on top of the native StreamEdit velocity "
            "blend, instead of hard-switching velocity/KV/attention "
            "routing. KV metadata falls back to the legacy cross-attention "
            "union mask."
        ),
    )
    parser.add_argument(
        "--soft_region_blend_strength",
        type=float,
        default=0.5,
        help=(
            "How strongly the flow-verified region confidence suppresses "
            "the background velocity correction inside the detected edit "
            "region. 0 = no effect, 1 = full suppression."
        ),
    )
    parser.add_argument(
        "--first_block_identity_anchor",
        action="store_true",
        default=False,
        help=(
            "Freeze the first generation block's clean target KV as a "
            "persistent identity anchor. All subsequent blocks attend to "
            "this anchor in addition to the mutable KV history, preventing "
            "identity drift."
        ),
    )
    parser.add_argument(
        "--identity_anchor_scale",
        type=float,
        default=1.5,
        help=(
            "Multiplicative scaling applied to anchor keys before "
            "attention. Values > 1 amplify anchor influence, counteracting "
            "the natural dilution as more blocks are committed. Analogous "
            "to Helios Guidance Attention but training-free."
        ),
    )
    parser.add_argument(
        "--suppress_source_bg_value",
        action="store_true",
        default=False,
        help=(
            "Replace source background values with target background "
            "values in the late injection (after t^inj=0.5). Source keys "
            "remain unchanged so attention addressing is preserved, but "
            "source appearance no longer leaks through the value path."
        ),
    )
    parser.add_argument("--mask_white_threshold", type=int, default=245)
    parser.add_argument(
        "--hand_mask_mode",
        choices=["white", "overlay_white"],
        default="white",
        help=(
            "How to parse --hand_mask_video. Use 'white' for a binary "
            "white matte; use 'overlay_white' when the hand is painted "
            "white over the source video."
        ),
    )
    parser.add_argument(
        "--hand_mask_overlay_diff_threshold",
        type=float,
        default=24.0,
        help=(
            "Mean RGB difference from the source frame required by "
            "--hand_mask_mode overlay_white."
        ),
    )
    parser.add_argument(
        "--hand_causal_evidence",
        action="store_true",
        default=False,
        help=(
            "Keep causal hand union, temporal occupancy, and persistent "
            "support separate. Union is used only for interaction "
            "proximity, occupancy for soft contact, and persistent support "
            "for hard owner exclusion."
        ),
    )
    parser.add_argument(
        "--hand_persistent_occupancy",
        type=float,
        default=1.0,
        help=(
            "Minimum within-group hand occupancy used for hard owner "
            "exclusion when --hand_causal_evidence is enabled."
        ),
    )
    parser.add_argument(
        "--object_min_latent_coverage", type=float, default=0.001
    )
    parser.add_argument(
        "--hand_posterior_threshold",
        type=float,
        default=0.20,
        help="Boolean KV threshold for the soft hand-only object posterior.",
    )
    parser.add_argument(
        "--hand_max_object_coverage",
        type=float,
        default=0.18,
        help="Maximum per-frame active-object posterior coverage.",
    )
    parser.add_argument(
        "--hand_proximity_radius",
        type=int,
        default=3,
        help="Hand-proximity radius on the transformer token grid.",
    )
    parser.add_argument(
        "--hand_propagation_steps",
        type=int,
        default=2,
        help="Spatial propagation steps for hand-only object discovery.",
    )
    parser.add_argument(
        "--hand_connected_hysteresis",
        action="store_true",
        default=False,
        help=(
            "Recover low-confidence object extent only when it is connected "
            "to a high-confidence hand-conditioned seed."
        ),
    )
    parser.add_argument(
        "--hand_connected_growth_steps",
        type=int,
        default=3,
        help=(
            "Maximum token-grid geodesic growth distance for connected "
            "hysteresis."
        ),
    )
    parser.add_argument(
        "--hand_connected_candidate_ratio",
        type=float,
        default=1.0,
        help=(
            "Relative low threshold for connected object-extent growth. "
            "Only candidates connected to a high-confidence seed can be "
            "admitted; 1.0 preserves the legacy candidate threshold."
        ),
    )
    parser.add_argument(
        "--hand_visibility_ratio",
        type=float,
        default=0.40,
        help="Relative interaction-support threshold for object visibility.",
    )
    parser.add_argument(
        "--hand_temporal_weight",
        type=float,
        default=0.45,
        help="Weight of source-query temporal posterior propagation.",
    )
    parser.add_argument(
        "--hand_query_similarity_threshold",
        type=float,
        default=0.65,
        help="Minimum cosine confidence for source-query propagation.",
    )
    parser.add_argument(
        "--hand_query_layers",
        type=int,
        nargs="+",
        default=[8, 12, 16, 20],
        help="Transformer layers used to form clean-source query features.",
    )
    parser.add_argument(
        "--hand_field_quantile_low",
        type=float,
        default=0.50,
        help="Lower per-frame quantile for field-disagreement normalization.",
    )
    parser.add_argument(
        "--hand_field_quantile_high",
        type=float,
        default=0.95,
        help="Upper per-frame quantile for field-disagreement normalization.",
    )
    parser.add_argument(
        "--hand_field_power",
        type=float,
        default=1.5,
        help="Power applied to normalized vector-field disagreement.",
    )
    parser.add_argument(
        "--hand_field_weight",
        type=float,
        default=0.65,
        help="Weight of the post-forward field observation; zero disables it.",
    )
    parser.add_argument(
        "--hand_field_candidate_radius",
        type=int,
        default=2,
        help="Prior-neighborhood radius allowed for field posterior expansion.",
    )
    parser.add_argument(
        "--hand_field_update_mode",
        choices=["off", "diagnostic", "posterior"],
        default="diagnostic",
        help=(
            "Use field disagreement only for diagnostics by default; "
            "'posterior' enables the experimental posterior expansion."
        ),
    )
    parser.add_argument("--role_boundary_radius", type=int, default=1)
    parser.add_argument(
        "--contact_target_weight",
        type=float,
        default=0.7,
        help=(
            "Target-field weight for contact tokens in role residual modes; "
            "must be in [0, 1]."
        ),
    )
    parser.add_argument(
        "--posterior_flow_mode",
        choices=["soft", "hard"],
        default="soft",
        help=(
            "Use soft role probabilities or hard argmax roles for "
            "posterior-routed residual vector fields."
        ),
    )
    parser.add_argument(
        "--posterior_flow_use_field",
        action="store_true",
        help=(
            "Allow target-dependent field evidence to update the adaptive "
            "posterior in posterior-flow mode. Disabled for clean routing "
            "ablations."
        ),
    )
    parser.add_argument(
        "--contact_graph_mode",
        choices=[
            "no_graph",
            "distance_only",
            "shuffled",
            "source_qk",
        ],
        default="no_graph",
        help="Oracle contact-relation ablation applied in self-attention.",
    )
    parser.add_argument("--contact_graph_topk", type=int, default=4)
    parser.add_argument("--contact_graph_radius", type=float, default=2.5)
    parser.add_argument(
        "--contact_graph_min_confidence",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--contact_graph_strength",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--contact_graph_layer_start",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--contact_graph_layer_end",
        type=int,
        default=20,
    )
    parser.add_argument("--contact_graph_seed", type=int, default=0)
    parser.add_argument("--save_role_dir", type=str, default=None)
    args = parser.parse_args()
    oracle_role_enabled = args.routing_mode in {
        "oracle_role_flow",
        "oracle_role_flow_kv",
        "oracle_role_residual",
        "oracle_role_residual_kv",
    }
    hand_role_enabled = args.routing_mode in {
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
    customized_reference_enabled = (
        args.routing_mode == "hand_role_bayes_flow_customized_kv"
    )
    if oracle_role_enabled and (
        args.object_mask_video is None or args.hand_mask_video is None
    ):
        parser.error(
            "Oracle role modes require --object_mask_video and "
            "--hand_mask_video"
        )
    if hand_role_enabled and args.hand_mask_video is None:
        parser.error(
            f"{args.routing_mode} requires --hand_mask_video"
        )
    if args.motion_geometry_owner:
        if args.routing_mode != "hand_role_factorized_causal_owner_kv":
            parser.error(
                "--motion_geometry_owner requires "
                "--routing_mode hand_role_factorized_causal_owner_kv"
            )
        if args.source_flow_cache is None:
            parser.error(
                "--motion_geometry_owner requires --source_flow_cache"
            )
        if args.object_mask_video is not None:
            parser.error(
                "Motion geometry inference forbids --object_mask_video"
            )
        if args.source_owner_mask_video is not None:
            parser.error(
                "Motion geometry inference forbids --source_owner_mask_video"
            )
    if customized_reference_enabled and args.first_frame_edit is None:
        parser.error(
            "hand_role_bayes_flow_customized_kv requires "
            "--first_frame_edit"
        )
    if customized_reference_enabled and args.triple_first_frame:
        parser.error(
            "Customized reference mode requires one independent "
            "reference frame; do not use --triple_first_frame"
        )
    if not -1.0 < args.identity_tokenprop_min_similarity < 1.0:
        parser.error(
            "--identity_tokenprop_min_similarity must be in (-1, 1)"
        )
    if not 0.0 <= args.identity_tokenprop_gate_strength <= 1.0:
        parser.error(
            "--identity_tokenprop_gate_strength must be in [0, 1]"
        )
    if args.identity_tokenprop_max_candidates <= 0:
        parser.error("--identity_tokenprop_max_candidates must be positive")
    if args.source_coordinate_identity and not args.first_chunk_identity_replay:
        parser.error(
            "--source_coordinate_identity requires "
            "--first_chunk_identity_replay"
        )
    if args.factorized_immutable_target_memory and (
        args.routing_mode != "hand_role_factorized_causal_owner_kv"
    ):
        parser.error(
            "--factorized_immutable_target_memory requires "
            "--routing_mode hand_role_factorized_causal_owner_kv"
        )
    if args.factorized_native_target_history and (
        args.routing_mode != "hand_role_factorized_causal_owner_kv"
    ):
        parser.error(
            "--factorized_native_target_history requires "
            "--routing_mode hand_role_factorized_causal_owner_kv"
        )
    if (
        args.factorized_owner_source_block
        and not args.factorized_native_target_history
    ):
        parser.error(
            "--factorized_owner_source_block requires "
            "--factorized_native_target_history"
        )
    if args.target_semantic_competition:
        if (
            args.routing_mode
            != "hand_role_factorized_causal_owner_kv"
        ):
            parser.error(
                "--target_semantic_competition requires hand-role "
                "factorized causal-owner routing"
            )
        if not args.target_edit_phrases:
            parser.error(
                "--target_semantic_competition requires "
                "--target_edit_phrases"
            )
        if not args.target_preserve_phrases:
            parser.error(
                "--target_semantic_competition requires "
                "--target_preserve_phrases"
            )
        if not 0.0 <= args.target_semantic_margin < 1.0:
            parser.error(
                "--target_semantic_margin must lie in [0, 1)"
            )
        if not (
            0.0 <= args.target_semantic_min_confidence <= 1.0
        ):
            parser.error(
                "--target_semantic_min_confidence must lie in [0, 1]"
            )
    if args.causal_paired_edit_memory and (
        args.routing_mode != "hand_role_factorized_causal_owner_kv"
    ):
        parser.error(
            "--causal_paired_edit_memory requires "
            "--routing_mode hand_role_factorized_causal_owner_kv"
        )
    if (
        args.causal_paired_edit_memory
        and not args.factorized_native_target_history
    ):
        parser.error(
            "--causal_paired_edit_memory requires "
            "--factorized_native_target_history"
        )
    if args.paired_memory_layers is not None and (
        not args.paired_memory_layers
        or any(
            layer < 0 or layer >= 30
            for layer in args.paired_memory_layers
        )
        or len(set(args.paired_memory_layers))
        != len(args.paired_memory_layers)
    ):
        parser.error(
            "--paired_memory_layers must contain unique values in [0, 29]"
        )
    if args.paired_memory_max_tokens <= 0:
        parser.error("--paired_memory_max_tokens must be positive")
    if args.paired_memory_max_tokens_per_block <= 0:
        parser.error(
            "--paired_memory_max_tokens_per_block must be positive"
        )
    if args.paired_memory_topk <= 0:
        parser.error("--paired_memory_topk must be positive")
    if not -1.0 < args.paired_memory_min_similarity < 1.0:
        parser.error(
            "--paired_memory_min_similarity must lie in (-1, 1)"
        )
    if not 0.0 <= args.paired_memory_min_commit_confidence <= 1.0:
        parser.error(
            "--paired_memory_min_commit_confidence must lie in [0, 1]"
        )
    if args.paired_memory_coordinate_bias < 0.0:
        parser.error(
            "--paired_memory_coordinate_bias must be non-negative"
        )
    if args.paired_memory_coordinate_radius < 0.0:
        parser.error(
            "--paired_memory_coordinate_radius must be non-negative"
        )
    if not 0.0 <= args.paired_memory_min_residual_consensus < 1.0:
        parser.error(
            "--paired_memory_min_residual_consensus must lie in [0, 1)"
        )
    if not -1.0 < args.paired_memory_min_part_similarity < 1.0:
        parser.error(
            "--paired_memory_min_part_similarity must lie in (-1, 1)"
        )
    if not 0.0 <= args.paired_memory_part_similarity_margin <= 2.0:
        parser.error(
            "--paired_memory_part_similarity_margin must lie in [0, 2]"
        )
    if not 0.0 <= args.paired_memory_read_strength <= 1.0:
        parser.error(
            "--paired_memory_read_strength must lie in [0, 1]"
        )
    if not 0.0 <= args.paired_memory_source_suppression <= 1.0:
        parser.error(
            "--paired_memory_source_suppression must lie in [0, 1]"
        )
    if not -1.0 < args.paired_memory_transport_min_similarity < 1.0:
        parser.error(
            "--paired_memory_transport_min_similarity must lie in (-1, 1)"
        )
    if args.paired_memory_transport_coordinate_radius <= 0.0:
        parser.error(
            "--paired_memory_transport_coordinate_radius must be positive"
        )
    if args.paired_memory_transport_cycle_radius <= 0.0:
        parser.error(
            "--paired_memory_transport_cycle_radius must be positive"
        )
    if not (
        0.0 <= args.paired_memory_transport_min_confidence <= 1.0
    ):
        parser.error(
            "--paired_memory_transport_min_confidence must lie in [0, 1]"
        )
    if (
        args.paired_memory_source_transport
        and not args.causal_paired_edit_memory
    ):
        parser.error(
            "--paired_memory_source_transport requires "
            "--causal_paired_edit_memory"
        )
    if (
        args.paired_memory_owner_attached_boundary
        and not args.paired_memory_interior_projection
    ):
        parser.error(
            "--paired_memory_owner_attached_boundary requires "
            "--paired_memory_interior_projection"
        )
    if (
        args.paired_memory_single_confidence
        and not args.paired_memory_query_gated_projection
    ):
        parser.error(
            "--paired_memory_single_confidence requires "
            "--paired_memory_query_gated_projection"
        )
    if args.paired_memory_owner_attached_boundary and (
        not args.paired_memory_source_part_consistency
        or not args.paired_memory_source_transport
    ):
        parser.error(
            "--paired_memory_owner_attached_boundary requires both "
            "--paired_memory_source_part_consistency and "
            "--paired_memory_source_transport"
        )
    if args.paired_memory_dual_timescale_anchor and not (
        args.paired_memory_value_projection
        and args.paired_memory_query_gated_projection
        and args.paired_memory_disable_persistent_projection
        and args.paired_memory_single_confidence
        and args.paired_memory_source_transport
    ):
        parser.error(
            "--paired_memory_dual_timescale_anchor requires transient, "
            "query-gated, single-confidence source-transported paired "
            "memory"
        )
    if args.paired_memory_canonical_key_anchor and not (
        args.paired_memory_dual_timescale_anchor
        and args.paired_memory_first_block_replay
    ):
        parser.error(
            "--paired_memory_canonical_key_anchor requires "
            "--paired_memory_dual_timescale_anchor and "
            "--paired_memory_first_block_replay"
        )
    if args.role_fixed_native_history and not (
        args.factorized_native_target_history
        and args.routing_mode == "hand_role_factorized_causal_owner_kv"
    ):
        parser.error(
            "--role_fixed_native_history requires factorized causal-owner "
            "routing and --factorized_native_target_history"
        )
    if args.native_history_layers is not None and (
        not args.native_history_layers
        or len(set(args.native_history_layers))
        != len(args.native_history_layers)
        or any(layer < 0 or layer >= 30 for layer in args.native_history_layers)
    ):
        parser.error(
            "--native_history_layers must contain unique values in [0, 29]"
        )
    if args.native_history_max_tokens_per_frame <= 0:
        parser.error(
            "--native_history_max_tokens_per_frame must be positive"
        )
    if args.native_history_topk <= 0:
        parser.error("--native_history_topk must be positive")
    if not -1.0 < args.native_history_min_similarity < 1.0:
        parser.error(
            "--native_history_min_similarity must lie in (-1, 1)"
        )
    for name in (
        "native_history_min_write_confidence",
        "native_history_min_query_confidence",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name} must lie in [0, 1]")
    if not math.isfinite(args.native_history_canonical_logit_bias):
        parser.error(
            "--native_history_canonical_logit_bias must be finite"
        )
    if args.native_history_bypass_blocks is not None and (
        len(set(args.native_history_bypass_blocks))
        != len(args.native_history_bypass_blocks)
        or any(block < 0 for block in args.native_history_bypass_blocks)
    ):
        parser.error(
            "--native_history_bypass_blocks must contain unique "
            "non-negative block indices"
        )
    if (
        args.native_history_coalesce_bootstrap_time
        or args.native_history_bypass_blocks is not None
        or args.native_history_source_part_consistency
    ) and not args.role_fixed_native_history:
        parser.error(
            "Native-history bootstrap controls require "
            "--role_fixed_native_history"
        )
    if not -1.0 < args.native_history_min_part_similarity < 1.0:
        parser.error(
            "--native_history_min_part_similarity must lie in (-1, 1)"
        )
    if not (
        0.0 <= args.native_history_part_similarity_margin <= 2.0
    ):
        parser.error(
            "--native_history_part_similarity_margin must lie in [0, 2]"
        )
    if not 0.0 <= args.native_history_part_bias_strength <= 4.0:
        parser.error(
            "--native_history_part_bias_strength must lie in [0, 4]"
        )
    if not 0.0 <= args.native_history_part_refinement_ratio <= 1.0:
        parser.error(
            "--native_history_part_refinement_ratio must lie in [0, 1]"
        )
    if (
        args.native_history_transactional_owner
        and not args.role_fixed_native_history
    ):
        parser.error(
            "--native_history_transactional_owner requires "
            "--role_fixed_native_history"
        )
    if args.hand_flow_transactional_owner:
        if not args.native_history_transactional_owner:
            parser.error(
                "--hand_flow_transactional_owner requires "
                "--native_history_transactional_owner"
            )
        if args.routing_mode != "hand_role_factorized_causal_owner_kv":
            parser.error(
                "--hand_flow_transactional_owner requires "
                "--routing_mode hand_role_factorized_causal_owner_kv"
            )
        if args.hand_mask_video is None:
            parser.error(
                "--hand_flow_transactional_owner requires "
                "--hand_mask_video"
            )
        if args.object_mask_video is not None:
            parser.error(
                "--hand_flow_transactional_owner forbids "
                "--object_mask_video"
            )
        if args.source_owner_mask_video is not None:
            parser.error(
                "--hand_flow_transactional_owner forbids "
                "--source_owner_mask_video"
            )
    if (
        args.native_history_consistent_transaction
        and not args.hand_flow_transactional_owner
    ):
        parser.error(
            "--native_history_consistent_transaction requires "
            "--hand_flow_transactional_owner"
        )
    if (
        args.native_history_verified_attention_authority
        and not args.native_history_consistent_transaction
    ):
        parser.error(
            "--native_history_verified_attention_authority requires "
            "--native_history_consistent_transaction"
        )
    if not (
        0.0 <= args.native_history_attention_authority_strength <= 1.0
    ):
        parser.error(
            "--native_history_attention_authority_strength must lie in "
            "[0, 1]"
        )
    if (
        args.native_history_payload_invariant_lineage
        and not args.native_history_transactional_owner
    ):
        parser.error(
            "--native_history_payload_invariant_lineage requires "
            "--native_history_transactional_owner"
        )
    if not 0.0 <= args.native_history_payload_blend_strength <= 1.0:
        parser.error(
            "--native_history_payload_blend_strength must lie in [0, 1]"
        )
    if args.native_history_recent_entry_bridge:
        if not args.native_history_consistent_transaction:
            parser.error(
                "--native_history_recent_entry_bridge requires "
                "--native_history_consistent_transaction"
            )
        if args.native_history_payload_invariant_lineage:
            parser.error(
                "--native_history_recent_entry_bridge is incompatible with "
                "--native_history_payload_invariant_lineage"
            )
        if not args.hand_flow_transactional_owner:
            parser.error(
                "--native_history_recent_entry_bridge requires the "
                "hand-only transactional owner"
            )
    if args.native_history_motion_owner_dense_read and not (
        args.native_history_recent_entry_bridge
        and args.native_history_consistent_transaction
        and args.motion_geometry_owner
        and args.hand_flow_transactional_owner
    ):
        parser.error(
            "--native_history_motion_owner_dense_read requires "
            "--native_history_recent_entry_bridge, "
            "--native_history_consistent_transaction, "
            "--motion_geometry_owner, and the hand-only transactional "
            "owner"
        )
    if not 0.0 <= args.native_history_entry_bridge_strength <= 1.0:
        parser.error(
            "--native_history_entry_bridge_strength must lie in [0, 1]"
        )
    if not (
        0.0
        <= args.native_history_dense_recent_min_residual_consensus
        <= 1.0
    ):
        parser.error(
            "--native_history_dense_recent_min_residual_consensus must lie "
            "in [0, 1]"
        )
    if args.native_history_owner_max_missing_frames < 0:
        parser.error(
            "--native_history_owner_max_missing_frames must be "
            "non-negative"
        )
    if not (
        0.0 <= args.native_history_verified_source_suppression <= 1.0
    ):
        parser.error(
            "--native_history_verified_source_suppression must lie in "
            "[0, 1]"
        )
    if args.native_history_dual_evidence_arbitration and not (
        args.native_history_recent_entry_bridge
        and args.native_history_consistent_transaction
    ):
        parser.error(
            "--native_history_dual_evidence_arbitration requires the "
            "consistent recent-entry bridge"
        )
    if args.native_history_token_atomic_payload and not (
        args.native_history_recent_entry_bridge
        and args.native_history_consistent_transaction
        and args.native_history_motion_owner_dense_read
    ):
        parser.error(
            "--native_history_token_atomic_payload requires the "
            "consistent dense motion-owner recent bridge"
        )
    if args.native_history_topology_complete_read and not (
        args.native_history_token_atomic_payload
        and args.motion_geometry_owner
    ):
        parser.error(
            "--native_history_topology_complete_read requires token-atomic "
            "payloads and motion geometry ownership"
        )
    if args.native_history_persistent_residual_upsert and not (
        args.native_history_token_atomic_payload
        and args.motion_geometry_owner
    ):
        parser.error(
            "--native_history_persistent_residual_upsert requires "
            "token-atomic payloads and motion geometry ownership"
        )
    if args.native_history_last_trusted_appearance and not (
        args.native_history_persistent_residual_upsert
        and args.native_history_transactional_owner
    ):
        parser.error(
            "--native_history_last_trusted_appearance requires persistent "
            "residual upserts and transactional owner arbitration"
        )
    if args.native_history_flow_indexed_residual and not (
        args.native_history_last_trusted_appearance
        and args.motion_geometry_owner
    ):
        parser.error(
            "--native_history_flow_indexed_residual requires last-trusted "
            "appearance and motion geometry ownership"
        )
    if args.native_history_decoupled_flow_trust and not (
        args.native_history_flow_indexed_residual
    ):
        parser.error(
            "--native_history_decoupled_flow_trust requires "
            "--native_history_flow_indexed_residual"
        )
    if args.native_history_multiframe_identity_sink and not (
        args.native_history_decoupled_flow_trust
    ):
        parser.error(
            "--native_history_multiframe_identity_sink requires "
            "--native_history_decoupled_flow_trust"
        )
    if args.native_history_multiframe_identity_sink and (
        args.object_mask_video is not None
        or args.source_owner_mask_video is not None
    ):
        parser.error(
            "--native_history_multiframe_identity_sink forbids object and "
            "source-owner masks"
        )
    if args.native_history_timestep_counterfactual_memory and not (
        args.native_history_multiframe_identity_sink
    ):
        parser.error(
            "--native_history_timestep_counterfactual_memory requires "
            "--native_history_multiframe_identity_sink"
        )
    if args.native_history_timestep_counterfactual_memory and (
        args.object_mask_video is not None
        or args.source_owner_mask_video is not None
    ):
        parser.error(
            "--native_history_timestep_counterfactual_memory forbids "
            "object and source-owner masks"
        )
    if args.native_history_tccm_flow_radius < 0.0:
        parser.error(
            "--native_history_tccm_flow_radius must be non-negative"
        )
    if not 0.0 <= args.native_history_tccm_strength <= 1.0:
        parser.error(
            "--native_history_tccm_strength must lie in [0, 1]"
        )
    if args.native_history_tccm_max_error_ratio <= 0.0:
        parser.error(
            "--native_history_tccm_max_error_ratio must be positive"
        )
    if args.native_history_multiframe_sink_topk_per_frame <= 0:
        parser.error(
            "--native_history_multiframe_sink_topk_per_frame must be "
            "positive"
        )
    if not math.isfinite(
        args.native_history_multiframe_sink_source_logit_bias
    ):
        parser.error(
            "--native_history_multiframe_sink_source_logit_bias must be "
            "finite"
        )
    if not 0.0 <= args.native_history_multiframe_sink_strength <= 1.0:
        parser.error(
            "--native_history_multiframe_sink_strength must lie in [0, 1]"
        )
    if not 0.0 <= args.native_history_flow_min_confidence <= 1.0:
        parser.error(
            "--native_history_flow_min_confidence must lie in [0, 1]"
        )
    if args.source_flow_role_fusion and not args.motion_geometry_owner:
        parser.error(
            "--source_flow_role_fusion requires --motion_geometry_owner"
        )
    if args.source_flow_verified_region and not args.source_flow_role_fusion:
        parser.error(
            "--source_flow_verified_region requires "
            "--source_flow_role_fusion"
        )
    if not 0.0 <= args.source_flow_role_weight <= 1.0:
        parser.error("--source_flow_role_weight must lie in [0, 1]")
    if args.source_flow_verified_owner_radius < 0:
        parser.error(
            "--source_flow_verified_owner_radius must be non-negative"
        )
    if not 0.0 <= args.source_flow_background_veto_threshold <= 1.0:
        parser.error(
            "--source_flow_background_veto_threshold must lie in [0, 1]"
        )
    if not (
        0.0 <= args.source_flow_background_veto_min_confidence <= 1.0
    ):
        parser.error(
            "--source_flow_background_veto_min_confidence must lie in "
            "[0, 1]"
        )
    if not (
        -1.0 <= args.native_history_residual_update_min_cosine <= 1.0
    ):
        parser.error(
            "--native_history_residual_update_min_cosine must lie in [-1, 1]"
        )
    if not (
        0.0
        <= args.native_history_residual_update_min_magnitude_ratio
        <= 1.0
    ):
        parser.error(
            "--native_history_residual_update_min_magnitude_ratio must lie "
            "in [0, 1]"
        )
    if not 0.0 <= args.native_history_min_payload_consistency <= 1.0:
        parser.error(
            "--native_history_min_payload_consistency must lie in [0, 1]"
        )
    if (
        args.paired_memory_first_block_replay
        and not args.causal_paired_edit_memory
    ):
        parser.error(
            "--paired_memory_first_block_replay requires "
            "--causal_paired_edit_memory"
        )
    if (
        args.paired_memory_query_gated_projection
        and not args.paired_memory_value_projection
    ):
        parser.error(
            "--paired_memory_query_gated_projection requires "
            "--paired_memory_value_projection"
        )
    if (
        args.paired_memory_disable_persistent_projection
        and not args.paired_memory_value_projection
    ):
        parser.error(
            "--paired_memory_disable_persistent_projection requires "
            "--paired_memory_value_projection"
        )
    if (
        args.paired_memory_query_gated_projection
        and not args.paired_memory_disable_persistent_projection
    ):
        parser.error(
            "Query-gated projection currently requires "
            "--paired_memory_disable_persistent_projection because "
            "historical projected values do not yet carry query-readable "
            "owner metadata"
        )
    if (
        args.paired_memory_source_part_consistency
        and not args.paired_memory_query_gated_projection
    ):
        parser.error(
            "--paired_memory_source_part_consistency requires "
            "--paired_memory_query_gated_projection"
        )
    if (
        (
            args.paired_memory_layers is not None
            or args.paired_memory_max_tokens != 1536
            or args.paired_memory_max_tokens_per_block != 192
            or args.paired_memory_topk != 8
            or args.paired_memory_min_similarity != 0.35
            or args.paired_memory_min_commit_confidence != 0.20
            or args.paired_memory_coordinate_bias != 1.0
            or args.paired_memory_coordinate_radius != 0.0
            or args.paired_memory_min_residual_consensus != 0.0
            or args.paired_memory_source_part_consistency
            or args.paired_memory_min_part_similarity != 0.45
            or args.paired_memory_part_similarity_margin != 0.08
            or args.paired_memory_read_strength != 0.35
            or args.paired_memory_value_projection
            or args.paired_memory_query_gated_projection
            or args.paired_memory_disable_persistent_projection
            or args.paired_memory_source_suppression != 0.0
            or args.paired_memory_interior_projection
            or args.paired_memory_first_block_replay
            or args.paired_memory_source_transport
            or args.paired_memory_single_confidence
            or args.paired_memory_owner_attached_boundary
            or args.paired_memory_dual_timescale_anchor
            or args.paired_memory_canonical_key_anchor
            or args.paired_memory_transport_min_similarity != 0.10
            or args.paired_memory_transport_coordinate_radius != 0.60
            or args.paired_memory_transport_cycle_radius != 0.20
            or args.paired_memory_transport_min_confidence != 0.05
        )
        and not args.causal_paired_edit_memory
    ):
        parser.error(
            "Paired-memory configuration requires "
            "--causal_paired_edit_memory"
        )
    if (
        args.factorized_native_target_history
        and args.factorized_immutable_target_memory
    ):
        parser.error(
            "--factorized_native_target_history is a clean-cache "
            "ablation and cannot be combined with "
            "--factorized_immutable_target_memory"
        )
    if (
        args.factorized_immutable_target_memory
        and not args.first_chunk_identity_replay
    ):
        parser.error(
            "--factorized_immutable_target_memory requires "
            "--first_chunk_identity_replay"
        )
    if args.immutable_target_layers is not None and (
        not args.immutable_target_layers
        or any(layer < 0 or layer >= 30 for layer in args.immutable_target_layers)
        or len(set(args.immutable_target_layers))
        != len(args.immutable_target_layers)
    ):
        parser.error(
            "--immutable_target_layers must contain unique values in [0, 29]"
        )
    if args.immutable_target_num_prototypes <= 0:
        parser.error("--immutable_target_num_prototypes must be positive")
    if (
        (
            args.immutable_target_layers is not None
            or args.immutable_target_num_prototypes != 4
            or args.immutable_target_value_mode != "residual"
            or args.immutable_target_hard_owner
        )
        and not args.factorized_immutable_target_memory
    ):
        parser.error(
            "Immutable target-memory configuration requires "
            "--factorized_immutable_target_memory"
        )
    if (
        args.factorized_orthogonal_geometry
        and not args.factorized_immutable_target_memory
    ):
        parser.error(
            "--factorized_orthogonal_geometry requires "
            "--factorized_immutable_target_memory"
        )
    if not 0.0 <= args.factorized_geometry_strength <= 1.0:
        parser.error("--factorized_geometry_strength must be in [0, 1]")
    if (
        args.source_coordinate_identity
        and not args.appearance_leakage_decomposition
    ):
        parser.error(
            "--source_coordinate_identity requires "
            "--appearance_leakage_decomposition"
        )
    if not 0.0 <= args.identity_source_suppression <= 1.0:
        parser.error(
            "--identity_source_suppression must be in [0, 1]"
        )
    if not 0.0 <= args.identity_support_floor <= 1.0:
        parser.error(
            "--identity_support_floor must be in [0, 1]"
        )
    if (
        args.source_identity_residual_carry
        and not args.source_coordinate_identity
    ):
        parser.error(
            "--source_identity_residual_carry requires "
            "--source_coordinate_identity"
        )
    if (
        args.source_owner_mask_video is not None
        and not args.source_coordinate_identity
        and args.routing_mode
        != "hand_role_factorized_causal_owner_kv"
    ):
        parser.error(
            "--source_owner_mask_video requires source-coordinate "
            "identity or factorized causal-owner routing"
        )
    if args.source_owner_overlay_diff_threshold < 0:
        parser.error(
            "--source_owner_overlay_diff_threshold must be non-negative"
        )
    if (
        args.source_owner_prepool_hand_exclusion
        and args.source_owner_mask_video is None
    ):
        parser.error(
            "--source_owner_prepool_hand_exclusion requires "
            "--source_owner_mask_video"
        )
    if (
        args.source_owner_prepool_hand_exclusion
        and args.hand_mask_video is None
    ):
        parser.error(
            "--source_owner_prepool_hand_exclusion requires "
            "--hand_mask_video"
        )
    if (
        args.causal_owner_consistent_kv_metadata
        and args.routing_mode
        != "hand_role_factorized_causal_owner_kv"
    ):
        parser.error(
            "--causal_owner_consistent_kv_metadata requires "
            "--routing_mode hand_role_factorized_causal_owner_kv"
        )
    if (
        args.factorized_source_coordinate_target_delta
        and args.routing_mode
        != "hand_role_factorized_causal_owner_kv"
    ):
        parser.error(
            "--factorized_source_coordinate_target_delta requires "
            "--routing_mode hand_role_factorized_causal_owner_kv"
        )
    if (
        args.factorized_source_coordinate_target_delta
        and args.factorized_orthogonal_geometry
    ):
        parser.error(
            "--factorized_source_coordinate_target_delta cannot be "
            "combined with --factorized_orthogonal_geometry"
        )
    if (
        args.factorized_owner_complement_source
        and args.routing_mode
        != "hand_role_factorized_causal_owner_kv"
    ):
        parser.error(
            "--factorized_owner_complement_source requires "
            "--routing_mode hand_role_factorized_causal_owner_kv"
        )
    if (
        args.factorized_owner_complement_source
        and args.source_owner_mask_video is None
        and not args.motion_geometry_owner
    ):
        parser.error(
            "--factorized_owner_complement_source requires "
            "--source_owner_mask_video or --motion_geometry_owner"
        )
    if args.factorized_owner_complement_margin < 0:
        parser.error(
            "--factorized_owner_complement_margin must be non-negative"
        )
    if not (
        0.0
        <= args.factorized_owner_complement_min_preserve_confidence
        <= 1.0
    ):
        parser.error(
            "--factorized_owner_complement_min_preserve_confidence "
            "must lie in [0, 1]"
        )
    if not 0.0 <= args.identity_residual_carry_strength <= 1.0:
        parser.error(
            "--identity_residual_carry_strength must be in [0, 1]"
        )
    if (
        args.source_owner_residual_constraint
        and not args.source_identity_residual_carry
    ):
        parser.error(
            "--source_owner_residual_constraint requires "
            "--source_identity_residual_carry"
        )
    if (
        args.source_owner_residual_constraint
        and args.source_owner_mask_video is None
    ):
        parser.error(
            "--source_owner_residual_constraint requires "
            "--source_owner_mask_video"
        )
    if not 0.0 <= args.identity_residual_constraint_strength <= 1.0:
        parser.error(
            "--identity_residual_constraint_strength must be in [0, 1]"
        )
    if args.identity_residual_constraint_power <= 0:
        parser.error(
            "--identity_residual_constraint_power must be positive"
        )
    if (
        args.source_owner_geometry_envelope
        and args.source_owner_mask_video is None
    ):
        parser.error(
            "--source_owner_geometry_envelope requires "
            "--source_owner_mask_video"
        )
    if not 0.0 <= args.source_geometry_strength <= 1.0:
        parser.error(
            "--source_geometry_strength must be in [0, 1]"
        )
    if args.source_geometry_power <= 0:
        parser.error(
            "--source_geometry_power must be positive"
        )
    if args.source_geometry_margin < 0:
        parser.error(
            "--source_geometry_margin must be non-negative"
        )
    if not 0.0 <= args.committed_memory_feedback_strength <= 1.0:
        parser.error(
            "--committed_memory_feedback_strength must be in [0, 1]"
        )
    if not 0 <= args.mask_white_threshold <= 255:
        parser.error("--mask_white_threshold must be in [0, 255]")
    if args.hand_mask_overlay_diff_threshold < 0:
        parser.error("--hand_mask_overlay_diff_threshold must be non-negative")
    if not 0.0 < args.hand_persistent_occupancy <= 1.0:
        parser.error(
            "--hand_persistent_occupancy must be in (0, 1]"
        )
    if not 0.0 <= args.contact_target_weight <= 1.0:
        parser.error("--contact_target_weight must be in [0, 1]")
    if not 0.0 <= args.hand_posterior_threshold <= 1.0:
        parser.error("--hand_posterior_threshold must be in [0, 1]")
    if not 0.0 < args.hand_max_object_coverage <= 1.0:
        parser.error("--hand_max_object_coverage must be in (0, 1]")
    if args.hand_proximity_radius < 0:
        parser.error("--hand_proximity_radius must be non-negative")
    if args.hand_propagation_steps < 0:
        parser.error("--hand_propagation_steps must be non-negative")
    if args.hand_connected_growth_steps < 0:
        parser.error(
            "--hand_connected_growth_steps must be non-negative"
        )
    if not 0.0 < args.hand_connected_candidate_ratio <= 1.0:
        parser.error(
            "--hand_connected_candidate_ratio must be in (0, 1]"
        )
    if not 0.0 <= args.hand_visibility_ratio <= 1.0:
        parser.error("--hand_visibility_ratio must be in [0, 1]")
    if not 0.0 <= args.hand_temporal_weight <= 1.0:
        parser.error("--hand_temporal_weight must be in [0, 1]")
    if not -1.0 < args.hand_query_similarity_threshold < 1.0:
        parser.error(
            "--hand_query_similarity_threshold must be in (-1, 1)"
        )
    if not args.hand_query_layers or any(
        layer < 0 or layer >= 30 for layer in args.hand_query_layers
    ):
        parser.error(
            "--hand_query_layers must contain values in [0, 29]"
        )
    if not (
        0.0
        <= args.hand_field_quantile_low
        < args.hand_field_quantile_high
        <= 1.0
    ):
        parser.error(
            "hand field quantiles must satisfy 0 <= low < high <= 1"
        )
    if args.hand_field_power <= 0:
        parser.error("--hand_field_power must be positive")
    if not 0.0 <= args.hand_field_weight <= 1.0:
        parser.error("--hand_field_weight must be in [0, 1]")
    if args.hand_field_candidate_radius < 0:
        parser.error(
            "--hand_field_candidate_radius must be non-negative"
        )
    if (
        args.contact_graph_mode != "no_graph"
        and args.routing_mode != "oracle_role_residual_kv"
    ):
        parser.error(
            "Contact graph modes require "
            "--routing_mode oracle_role_residual_kv"
        )
    if args.contact_graph_topk <= 0:
        parser.error("--contact_graph_topk must be positive")
    if args.contact_graph_radius <= 0:
        parser.error("--contact_graph_radius must be positive")
    if not 0.0 <= args.contact_graph_min_confidence <= 1.0:
        parser.error(
            "--contact_graph_min_confidence must be in [0, 1]"
        )
    if args.contact_graph_strength < 0:
        parser.error("--contact_graph_strength must be non-negative")
    if not (
        0
        <= args.contact_graph_layer_start
        < args.contact_graph_layer_end
        <= 30
    ):
        parser.error(
            "Contact graph layer range must satisfy "
            "0 <= start < end <= 30"
        )

    if args.hand_flow_transactional_owner:
        print(
            "HAND_FLOW_INPUT_CONTRACT "
            "external_object_mask=disabled "
            "external_source_owner_mask=disabled "
            "external_hand_mask=enabled "
            + (
                "owner_source=hand_attention_plus_source_rgb_raft "
                "diffusion_velocity=counterfactual_edit_response_only"
                if args.motion_geometry_owner
                else "owner_source=hand_attention_source_feature_transport"
            )
        )

    pipeline, low_memory, device, local_rank = load_pipe(args)

    # Create output directory (only on main process to avoid race conditions)
    if local_rank == 0:
        os.makedirs(Path(args.save_path).parent, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    # load video
    src_video = load_video(args.data_path)
    if args.first_frame_edit is not None:
        src_first_frame = src_video[0]
        trg_first_frame = Image.open(args.first_frame_edit).convert('RGB')
        if (
            customized_reference_enabled
            and trg_first_frame.size != src_first_frame.size
        ):
            parser.error(
                "Customized reference image must have the same spatial "
                "size as the source first frame before resizing: "
                f"{trg_first_frame.size} != {src_first_frame.size}"
            )
    else:
        src_first_frame = None
        trg_first_frame = None

    height = src_video[0].size[1]
    width = src_video[0].size[0]
    num_frames = len(src_video)
    new_len = find_closest_num_frame(num_frames)
    src_video = src_video[: new_len]
    num_frames = len(src_video)
    print(num_frames, height, width)

    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # AE
    src_video_tensor = torch.stack([transform(img) for img in src_video], dim=1).unsqueeze(0)
    video_latents = pipeline.vae.encode_to_latent(
        src_video_tensor.to(device=device, dtype=torch.bfloat16)
    ).to(device=device, dtype=torch.bfloat16)
    source_flow_cache = (
        SourceFlowCache.load(
            args.source_flow_cache,
            latent_frame_count=video_latents.shape[1],
            pixel_frame_count=num_frames,
            source_video_path=args.data_path,
        )
        if args.motion_geometry_owner
        else None
    )
    if source_flow_cache is not None:
        print(
            "MOTION_GEOMETRY_INPUT_CONTRACT "
            "clean_source_rgb_flow=enabled external_hand_mask=enabled "
            "external_object_mask=disabled external_source_owner_mask=disabled "
            f"latent_frames={source_flow_cache.latent_frame_count} "
            f"cache={Path(args.source_flow_cache).expanduser().resolve()}"
        )
    mask_temporal_groups = np.asarray(
        causal_vae_frame_groups(
            num_frames,
            video_latents.shape[1],
        ),
        dtype=np.int64,
    )
    mask_temporal_stride = (
        1
        if video_latents.shape[1] == 1
        else (num_frames - 1) // (video_latents.shape[1] - 1)
    )
    print(
        "MASK_TEMPORAL_ALIGNMENT "
        "mode=causal_first_frame_then_stride "
        f"pixel_frames={num_frames} "
        f"latent_frames={video_latents.shape[1]} "
        f"temporal_stride={mask_temporal_stride}"
    )
    object_pixel_mask = None
    object_latent_mask = None
    source_owner_pixel_mask = None
    source_owner_latent_mask = None
    source_owner_full_latent_mask = None
    legacy_visible_source_owner_latent_mask = None
    hand_pixel_mask = None
    hand_latent_mask = None
    hand_occupancy_latent = None
    hand_persistent_latent_mask = None
    if oracle_role_enabled:
        object_pixel_mask, object_latent_mask = build_white_mask(
            args.object_mask_video,
            src_video,
            tuple(video_latents.shape),
            args.mask_white_threshold,
            min_latent_coverage=args.object_min_latent_coverage,
        )
    if args.source_owner_mask_video is not None:
        (
            source_owner_pixel_mask,
            source_owner_latent_mask,
        ) = build_white_mask(
            args.source_owner_mask_video,
            src_video,
            tuple(video_latents.shape),
            args.mask_white_threshold,
            min_latent_coverage=args.object_min_latent_coverage,
            mode=args.source_owner_mask_mode,
            overlay_diff_threshold=(
                args.source_owner_overlay_diff_threshold
            ),
            recover_overlay_components=True,
        )
        source_owner_full_latent_mask = source_owner_latent_mask.clone()
    if oracle_role_enabled or hand_role_enabled:
        hand_pixel_mask, hand_latent_mask = build_white_mask(
            args.hand_mask_video,
            src_video,
            tuple(video_latents.shape),
            args.mask_white_threshold,
            mode=args.hand_mask_mode,
            overlay_diff_threshold=args.hand_mask_overlay_diff_threshold,
        )
        if hand_role_enabled and args.hand_causal_evidence:
            hand_evidence = project_hand_evidence_to_causal_latents(
                hand_pixel_mask,
                latent_frames=video_latents.shape[1],
                latent_spatial_shape=video_latents.shape[-2:],
                persistent_occupancy=args.hand_persistent_occupancy,
            )
            hand_latent_mask = hand_evidence.union
            hand_occupancy_latent = hand_evidence.occupancy
            hand_persistent_latent_mask = hand_evidence.persistent
    if args.source_owner_prepool_hand_exclusion:
        legacy_visible_source_owner_latent_mask = (
            source_owner_latent_mask & ~hand_latent_mask
        )
        source_owner_latent_mask = (
            project_visible_owner_to_causal_latents(
                source_owner_pixel_mask,
                hand_pixel_mask,
                latent_frames=video_latents.shape[1],
                latent_spatial_shape=video_latents.shape[-2:],
                min_latent_coverage=args.object_min_latent_coverage,
            )
        )
        recovered_owner = (
            source_owner_latent_mask
            & ~legacy_visible_source_owner_latent_mask
        )
        removed_owner = (
            legacy_visible_source_owner_latent_mask
            & ~source_owner_latent_mask
        )
        print(
            "SOURCE_OWNER_PREPOOL_HAND_EXCLUSION "
            "order=pixel_subtract_then_causal_pool "
            f"legacy_visible_cells="
            f"{int(legacy_visible_source_owner_latent_mask.sum().item())} "
            f"corrected_visible_cells="
            f"{int(source_owner_latent_mask.sum().item())} "
            f"recovered_cells={int(recovered_owner.sum().item())} "
            f"removed_cells={int(removed_owner.sum().item())}"
        )
    if source_owner_pixel_mask is not None:
        owner_input_path = Path(args.save_path).with_suffix(
            ".source_owner_input.npz"
        )
        owner_artifact = {
            "source_owner_pixel_mask": (
                source_owner_pixel_mask.float().cpu().numpy()
            ),
            "source_owner_latent_mask": (
                source_owner_latent_mask.float().cpu().numpy()
            ),
            "source_owner_full_latent_mask": (
                source_owner_full_latent_mask.float().cpu().numpy()
            ),
            "white_threshold": np.array(args.mask_white_threshold),
            "source_owner_mask_mode": np.array(
                args.source_owner_mask_mode
            ),
            "source_owner_overlay_diff_threshold": np.array(
                args.source_owner_overlay_diff_threshold
            ),
            "source_owner_prepool_hand_exclusion": np.array(
                args.source_owner_prepool_hand_exclusion
            ),
            "causal_temporal_groups": mask_temporal_groups,
            "causal_temporal_stride": np.array(mask_temporal_stride),
        }
        if legacy_visible_source_owner_latent_mask is not None:
            owner_artifact[
                "legacy_visible_source_owner_latent_mask"
            ] = (
                legacy_visible_source_owner_latent_mask.float().cpu().numpy()
            )
        np.savez_compressed(owner_input_path, **owner_artifact)
        print(
            "SOURCE_OWNER_INPUT "
            f"pixel={source_owner_pixel_mask.float().mean().item():.4f} "
            f"latent={source_owner_latent_mask.float().mean().item():.4f} "
            f"prepool_hand_exclusion="
            f"{int(args.source_owner_prepool_hand_exclusion)} "
            f"mode={args.source_owner_mask_mode} "
            f"artifact={owner_input_path}"
        )
    if oracle_role_enabled:
        role_input_path = Path(args.save_path).with_suffix(
            ".oracle_role_inputs.npz"
        )
        np.savez_compressed(
            role_input_path,
            object_pixel_mask=object_pixel_mask.numpy(),
            object_latent_mask=object_latent_mask.numpy(),
            hand_pixel_mask=hand_pixel_mask.numpy(),
            hand_latent_mask=hand_latent_mask.numpy(),
            white_threshold=np.array(args.mask_white_threshold),
            hand_mask_mode=np.array(args.hand_mask_mode),
            hand_mask_overlay_diff_threshold=np.array(
                args.hand_mask_overlay_diff_threshold
            ),
            causal_temporal_groups=mask_temporal_groups,
            causal_temporal_stride=np.array(mask_temporal_stride),
        )
        print(
            "ORACLE_ROLE_INPUT "
            f"object={object_latent_mask.float().mean().item():.4f} "
            f"hand_pixel={hand_pixel_mask.float().mean().item():.4f} "
            f"hand_latent={hand_latent_mask.float().mean().item():.4f} "
            f"hand_mode={args.hand_mask_mode} "
            f"artifact={role_input_path}"
        )
    elif hand_role_enabled:
        hand_input_path = Path(args.save_path).with_suffix(
            ".hand_role_input.npz"
        )
        np.savez_compressed(
            hand_input_path,
            hand_pixel_mask=hand_pixel_mask.float().cpu().numpy(),
            hand_latent_mask=hand_latent_mask.float().cpu().numpy(),
            hand_occupancy_latent=(
                hand_latent_mask.float().cpu().numpy()
                if hand_occupancy_latent is None
                else hand_occupancy_latent.float().cpu().numpy()
            ),
            hand_persistent_latent_mask=(
                hand_latent_mask.float().cpu().numpy()
                if hand_persistent_latent_mask is None
                else hand_persistent_latent_mask.float().cpu().numpy()
            ),
            hand_causal_evidence=np.array(args.hand_causal_evidence),
            hand_persistent_occupancy=np.array(
                args.hand_persistent_occupancy
            ),
            causal_temporal_groups=mask_temporal_groups,
            causal_temporal_stride=np.array(mask_temporal_stride),
            white_threshold=np.array(args.mask_white_threshold),
            hand_mask_mode=np.array(args.hand_mask_mode),
            hand_mask_overlay_diff_threshold=np.array(
                args.hand_mask_overlay_diff_threshold
            ),
        )
        print(
            "HAND_ROLE_INPUT "
            f"hand_pixel={hand_pixel_mask.float().mean().item():.4f} "
            f"hand_latent={hand_latent_mask.float().mean().item():.4f} "
            + (
                "causal_evidence=1 "
                f"occupancy={hand_occupancy_latent.mean().item():.4f} "
                f"persistent={hand_persistent_latent_mask.float().mean().item():.4f} "
                if hand_occupancy_latent is not None
                else "causal_evidence=0 "
            )
            + f"hand_mode={args.hand_mask_mode} "
            f"artifact={hand_input_path}"
        )

    # first frame condition
    independent_first_frame = False
    triple_first_frame = False
    if args.first_frame_edit is not None:
        independent_first_frame = True
        triple_first_frame = False
        src_first_frame = pipeline.vae.encode_to_latent(
            transform(src_first_frame).unsqueeze(0).unsqueeze(2).to(video_latents)
        ).to(video_latents)
        trg_first_frame = pipeline.vae.encode_to_latent(
            transform(trg_first_frame).unsqueeze(0).unsqueeze(2).to(video_latents)
        ).to(video_latents)
        if args.triple_first_frame:
            independent_first_frame = False
            triple_first_frame = True
            src_first_frame = src_first_frame.repeat_interleave(3, dim=1)   # [B, F, C, H, W]
            trg_first_frame = trg_first_frame.repeat_interleave(3, dim=1)   # [B, F, C, H, W]

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    edit_video = pipeline.rollout_inference(
        src_video=video_latents,
        src_prompts=args.src_prompt,
        trg_prompts=args.trg_prompt,
        src_trigger_words=args.src_word,
        trg_trigger_words=args.trg_word,
        return_latents=False,
        wo_video_decode=False,
        profile=False,
        low_memory=low_memory,

        independent_first_frame=independent_first_frame,
        triple_first_frame=triple_first_frame,
        src_initial_latent=src_first_frame,
        trg_initial_latent=trg_first_frame,

        fg_boost_factor=args.fg_boost_factor,
        blend_power=args.blend_power,

        rollout_chunk_size=args.rollout_chunk_size,
        rollout_overlap_block_num=args.rollout_overlap_block_num,
        routing_mode=args.routing_mode,
        identity_first_latent_bootstrap=(
            args.identity_first_latent_bootstrap
        ),
        object_wise_anchor_reset=args.object_wise_anchor_reset,
        target_owned_object_handoff=(
            args.target_owned_object_handoff
        ),
        target_owned_min_similarity=(
            args.target_owned_min_similarity
        ),
        first_chunk_identity_replay=(
            args.first_chunk_identity_replay
        ),
        factorized_target_identity=(
            args.factorized_target_identity
        ),
        factorized_immutable_target_memory=(
            args.factorized_immutable_target_memory
        ),
        factorized_native_target_history=(
            args.factorized_native_target_history
        ),
        factorized_owner_source_block=(
            args.factorized_owner_source_block
        ),
        target_semantic_competition=(
            args.target_semantic_competition
        ),
        target_edit_phrases=args.target_edit_phrases,
        target_preserve_phrases=args.target_preserve_phrases,
        target_semantic_margin=args.target_semantic_margin,
        target_semantic_min_confidence=(
            args.target_semantic_min_confidence
        ),
        causal_paired_edit_memory=args.causal_paired_edit_memory,
        paired_memory_layers=args.paired_memory_layers,
        paired_memory_max_tokens=args.paired_memory_max_tokens,
        paired_memory_max_tokens_per_block=(
            args.paired_memory_max_tokens_per_block
        ),
        paired_memory_topk=args.paired_memory_topk,
        paired_memory_min_similarity=(
            args.paired_memory_min_similarity
        ),
        paired_memory_min_commit_confidence=(
            args.paired_memory_min_commit_confidence
        ),
        paired_memory_coordinate_bias=(
            args.paired_memory_coordinate_bias
        ),
        paired_memory_coordinate_radius=(
            args.paired_memory_coordinate_radius
        ),
        paired_memory_min_residual_consensus=(
            args.paired_memory_min_residual_consensus
        ),
        paired_memory_source_part_consistency=(
            args.paired_memory_source_part_consistency
        ),
        paired_memory_min_part_similarity=(
            args.paired_memory_min_part_similarity
        ),
        paired_memory_part_similarity_margin=(
            args.paired_memory_part_similarity_margin
        ),
        paired_memory_read_strength=args.paired_memory_read_strength,
        paired_memory_value_projection=(
            args.paired_memory_value_projection
        ),
        paired_memory_query_gated_projection=(
            args.paired_memory_query_gated_projection
        ),
        paired_memory_disable_persistent_projection=(
            args.paired_memory_disable_persistent_projection
        ),
        paired_memory_source_suppression=(
            args.paired_memory_source_suppression
        ),
        paired_memory_interior_projection=(
            args.paired_memory_interior_projection
        ),
        paired_memory_first_block_replay=(
            args.paired_memory_first_block_replay
        ),
        paired_memory_source_transport=(
            args.paired_memory_source_transport
        ),
        paired_memory_single_confidence=(
            args.paired_memory_single_confidence
        ),
        paired_memory_owner_attached_boundary=(
            args.paired_memory_owner_attached_boundary
        ),
        paired_memory_dual_timescale_anchor=(
            args.paired_memory_dual_timescale_anchor
        ),
        paired_memory_canonical_key_anchor=(
            args.paired_memory_canonical_key_anchor
        ),
        role_fixed_native_history=args.role_fixed_native_history,
        native_history_layers=args.native_history_layers,
        native_history_max_tokens_per_frame=(
            args.native_history_max_tokens_per_frame
        ),
        native_history_topk=args.native_history_topk,
        native_history_min_similarity=(
            args.native_history_min_similarity
        ),
        native_history_min_write_confidence=(
            args.native_history_min_write_confidence
        ),
        native_history_min_query_confidence=(
            args.native_history_min_query_confidence
        ),
        native_history_canonical_logit_bias=(
            args.native_history_canonical_logit_bias
        ),
        native_history_coalesce_bootstrap_time=(
            args.native_history_coalesce_bootstrap_time
        ),
        native_history_bypass_blocks=(
            args.native_history_bypass_blocks
        ),
        native_history_source_part_consistency=(
            args.native_history_source_part_consistency
        ),
        native_history_min_part_similarity=(
            args.native_history_min_part_similarity
        ),
        native_history_part_similarity_margin=(
            args.native_history_part_similarity_margin
        ),
        native_history_part_bias_strength=(
            args.native_history_part_bias_strength
        ),
        native_history_part_refinement_ratio=(
            args.native_history_part_refinement_ratio
        ),
        native_history_transactional_owner=(
            args.native_history_transactional_owner
        ),
        native_history_consistent_transaction=(
            args.native_history_consistent_transaction
        ),
        native_history_verified_attention_authority=(
            args.native_history_verified_attention_authority
        ),
        native_history_attention_authority_strength=(
            args.native_history_attention_authority_strength
        ),
        native_history_payload_invariant_lineage=(
            args.native_history_payload_invariant_lineage
        ),
        native_history_payload_blend_strength=(
            args.native_history_payload_blend_strength
        ),
        native_history_recent_entry_bridge=(
            args.native_history_recent_entry_bridge
        ),
        native_history_motion_owner_dense_read=(
            args.native_history_motion_owner_dense_read
        ),
        native_history_entry_bridge_strength=(
            args.native_history_entry_bridge_strength
        ),
        native_history_dual_evidence_arbitration=(
            args.native_history_dual_evidence_arbitration
        ),
        native_history_token_atomic_payload=(
            args.native_history_token_atomic_payload
        ),
        native_history_persistent_residual_upsert=(
            args.native_history_persistent_residual_upsert
        ),
        native_history_last_trusted_appearance=(
            args.native_history_last_trusted_appearance
        ),
        native_history_flow_indexed_residual=(
            args.native_history_flow_indexed_residual
        ),
        native_history_decoupled_flow_trust=(
            args.native_history_decoupled_flow_trust
        ),
        native_history_multiframe_identity_sink=(
            args.native_history_multiframe_identity_sink
        ),
        native_history_multiframe_sink_topk_per_frame=(
            args.native_history_multiframe_sink_topk_per_frame
        ),
        native_history_multiframe_sink_source_logit_bias=(
            args.native_history_multiframe_sink_source_logit_bias
        ),
        native_history_multiframe_sink_strength=(
            args.native_history_multiframe_sink_strength
        ),
        native_history_timestep_counterfactual_memory=(
            args.native_history_timestep_counterfactual_memory
        ),
        native_history_tccm_flow_radius=(
            args.native_history_tccm_flow_radius
        ),
        native_history_tccm_strength=(
            args.native_history_tccm_strength
        ),
        native_history_tccm_max_error_ratio=(
            args.native_history_tccm_max_error_ratio
        ),
        native_history_flow_min_confidence=(
            args.native_history_flow_min_confidence
        ),
        native_history_residual_update_min_cosine=(
            args.native_history_residual_update_min_cosine
        ),
        native_history_residual_update_min_magnitude_ratio=(
            args.native_history_residual_update_min_magnitude_ratio
        ),
        native_history_topology_complete_read=(
            args.native_history_topology_complete_read
        ),
        native_history_min_payload_consistency=(
            args.native_history_min_payload_consistency
        ),
        native_history_dense_recent_min_residual_consensus=(
            args.native_history_dense_recent_min_residual_consensus
        ),
        native_history_owner_max_missing_frames=(
            args.native_history_owner_max_missing_frames
        ),
        native_history_verified_source_suppression=(
            args.native_history_verified_source_suppression
        ),
        paired_memory_transport_min_similarity=(
            args.paired_memory_transport_min_similarity
        ),
        paired_memory_transport_coordinate_radius=(
            args.paired_memory_transport_coordinate_radius
        ),
        paired_memory_transport_cycle_radius=(
            args.paired_memory_transport_cycle_radius
        ),
        paired_memory_transport_min_confidence=(
            args.paired_memory_transport_min_confidence
        ),
        immutable_target_layers=args.immutable_target_layers,
        immutable_target_num_prototypes=(
            args.immutable_target_num_prototypes
        ),
        immutable_target_value_mode=args.immutable_target_value_mode,
        immutable_target_hard_owner=args.immutable_target_hard_owner,
        factorized_orthogonal_geometry=(
            args.factorized_orthogonal_geometry
        ),
        factorized_geometry_strength=(
            args.factorized_geometry_strength
        ),
        identity_correction_strength=(
            args.identity_correction_strength
        ),
        identity_visibility_lifecycle=(
            args.identity_visibility_lifecycle
        ),
        identity_max_occluded_blocks=(
            args.identity_max_occluded_blocks
        ),
        appearance_leakage_decomposition=(
            args.appearance_leakage_decomposition
        ),
        source_coordinate_identity=args.source_coordinate_identity,
        identity_source_suppression=(
            args.identity_source_suppression
        ),
        identity_support_floor=args.identity_support_floor,
        source_identity_residual_carry=(
            args.source_identity_residual_carry
        ),
        identity_residual_carry_strength=(
            args.identity_residual_carry_strength
        ),
        source_owner_residual_constraint=(
            args.source_owner_residual_constraint
        ),
        identity_residual_constraint_strength=(
            args.identity_residual_constraint_strength
        ),
        identity_residual_constraint_power=(
            args.identity_residual_constraint_power
        ),
        source_owner_geometry_envelope=(
            args.source_owner_geometry_envelope
        ),
        source_geometry_strength=args.source_geometry_strength,
        source_geometry_power=args.source_geometry_power,
        source_geometry_margin=args.source_geometry_margin,
        ignition_hand_exclusion_radius=(
            args.ignition_hand_exclusion_radius
        ),
        ignition_contact_radius=args.ignition_contact_radius,
        oracle_source_owner_mask=(
            None
            if source_owner_latent_mask is None
            else source_owner_latent_mask.unsqueeze(0).to(
                device=device
            )
        ),
        oracle_source_owner_full_mask=(
            None
            if source_owner_full_latent_mask is None
            else source_owner_full_latent_mask.unsqueeze(0).to(
                device=device
            )
        ),
        source_owner_prepool_hand_exclusion=(
            args.source_owner_prepool_hand_exclusion
        ),
        causal_owner_consistent_kv_metadata=(
            args.causal_owner_consistent_kv_metadata
        ),
        factorized_source_coordinate_target_delta=(
            args.factorized_source_coordinate_target_delta
        ),
        factorized_owner_complement_source=(
            args.factorized_owner_complement_source
        ),
        factorized_owner_complement_margin=(
            args.factorized_owner_complement_margin
        ),
        factorized_owner_complement_min_preserve_confidence=(
            args.factorized_owner_complement_min_preserve_confidence
        ),
        oracle_object_mask=(
            None
            if object_latent_mask is None
            else object_latent_mask.unsqueeze(0).to(device=device)
        ),
        oracle_hand_mask=(
            None
            if not oracle_role_enabled
            else hand_latent_mask.unsqueeze(0).to(device=device)
        ),
        hand_only_mask=(
            None
            if not hand_role_enabled
            else hand_latent_mask.unsqueeze(0).to(device=device)
        ),
        hand_occupancy_mask=(
            None
            if hand_occupancy_latent is None
            else hand_occupancy_latent.unsqueeze(0).to(device=device)
        ),
        hand_persistent_mask=(
            None
            if hand_persistent_latent_mask is None
            else hand_persistent_latent_mask.unsqueeze(0).to(
                device=device
            )
        ),
        hand_causal_evidence=args.hand_causal_evidence,
        motion_geometry_owner=args.motion_geometry_owner,
        source_flow_cache=source_flow_cache,
        source_flow_role_fusion=args.source_flow_role_fusion,
        source_flow_role_weight=args.source_flow_role_weight,
        source_flow_verified_region=args.source_flow_verified_region,
        source_flow_verified_owner_radius=(
            args.source_flow_verified_owner_radius
        ),
        source_flow_background_veto_threshold=(
            args.source_flow_background_veto_threshold
        ),
        source_flow_background_veto_min_confidence=(
            args.source_flow_background_veto_min_confidence
        ),
        soft_region_modulation=args.soft_region_modulation,
        soft_region_blend_strength=args.soft_region_blend_strength,
        first_block_identity_anchor=args.first_block_identity_anchor,
        identity_anchor_scale=args.identity_anchor_scale,
        suppress_source_bg_value=args.suppress_source_bg_value,
        role_boundary_radius=args.role_boundary_radius,
        contact_target_weight=args.contact_target_weight,
        posterior_flow_mode=args.posterior_flow_mode,
        posterior_flow_use_field=args.posterior_flow_use_field,
        hand_posterior_threshold=args.hand_posterior_threshold,
        hand_max_object_coverage=args.hand_max_object_coverage,
        hand_proximity_radius=args.hand_proximity_radius,
        hand_propagation_steps=args.hand_propagation_steps,
        hand_connected_hysteresis=args.hand_connected_hysteresis,
        hand_connected_growth_steps=args.hand_connected_growth_steps,
        hand_connected_candidate_ratio=(
            args.hand_connected_candidate_ratio
        ),
        hand_visibility_ratio=args.hand_visibility_ratio,
        hand_temporal_weight=args.hand_temporal_weight,
        hand_query_similarity_threshold=(
            args.hand_query_similarity_threshold
        ),
        hand_query_layers=args.hand_query_layers,
        hand_field_quantile_low=args.hand_field_quantile_low,
        hand_field_quantile_high=args.hand_field_quantile_high,
        hand_field_power=args.hand_field_power,
        hand_field_weight=args.hand_field_weight,
        hand_field_candidate_radius=args.hand_field_candidate_radius,
        hand_field_update_mode=args.hand_field_update_mode,
        identity_tokenprop_min_similarity=(
            args.identity_tokenprop_min_similarity
        ),
        identity_tokenprop_gate_strength=(
            args.identity_tokenprop_gate_strength
        ),
        identity_tokenprop_max_candidates=(
            args.identity_tokenprop_max_candidates
        ),
        committed_memory_feedback_strength=(
            args.committed_memory_feedback_strength
        ),
        contact_graph_mode=args.contact_graph_mode,
        contact_graph_topk=args.contact_graph_topk,
        contact_graph_radius=args.contact_graph_radius,
        contact_graph_min_confidence=(
            args.contact_graph_min_confidence
        ),
        contact_graph_strength=args.contact_graph_strength,
        contact_graph_layer_start=args.contact_graph_layer_start,
        contact_graph_layer_end=args.contact_graph_layer_end,
        contact_graph_seed=args.contact_graph_seed,
        save_role_dir=args.save_role_dir,
    )

    # Clear VAE cache
    pipeline.vae.model.clear_cache()
    write_video(
        args.save_path, 
        rearrange(edit_video[0], 't c h w -> t h w c').cpu() * 255, 
        fps=16
    )
