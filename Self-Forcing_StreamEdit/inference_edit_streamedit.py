import argparse
import torch
import torch.nn.functional as F
import os
from pathlib import Path 

import json
from collections import OrderedDict
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from einops import rearrange
import torch.distributed as dist
from torchvision import transforms
from torchvision.io import write_video

from pipeline import (
    EditCausalInferencePipeline
)
from utils.misc import set_seed
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

from diffusers.utils import load_video


BACKBONE_PRESETS = {
    "self_forcing": {
        "config": "configs/self_forcing_dmd.yaml",
        "checkpoint": "checkpoints/self_forcing_dmd.pt",
    },
    "causal_forcing_framewise": {
        "config": "configs/causal_forcing_dmd_framewise.yaml",
        "checkpoint": "checkpoints/framewise/causal_forcing.pt",
    },
    "causal_forcing_chunkwise": {
        "config": "configs/causal_forcing_dmd_chunkwise.yaml",
        "checkpoint": "checkpoints/chunkwise/causal_forcing.pt",
    },
    "causal_forcing_plus_plus_2step": {
        "config": (
            "configs/"
            "causal_forcing_dmd_framewise_2step.yaml"
        ),
        "checkpoint": (
            "checkpoints/"
            "causal-forcing++/framewise-2step.pt"
        ),
    },
}


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
):
    """Resample a white matte to source time and the Wan latent grid."""
    mask_video = load_video(mask_video_path)
    if not mask_video:
        raise RuntimeError(f"No frames decoded from mask video: {mask_video_path}")
    frame_indices = np.rint(
        np.linspace(0, len(mask_video) - 1, len(source_frames))
    ).astype(int)
    raw_masks = []
    for index in frame_indices:
        rgb = np.asarray(mask_video[int(index)].convert("RGB"))
        raw_masks.append(np.all(rgb >= threshold, axis=-1))
    pixel_mask = torch.from_numpy(np.stack(raw_masks)).unsqueeze(1).float()
    pixel_mask = F.interpolate(
        pixel_mask, size=(480, 832), mode="nearest"
    ) > 0.5

    _, latent_frames, _, latent_height, latent_width = latent_shape
    frame_groups = []
    for index in range(latent_frames):
        left = int(np.floor(index * len(source_frames) / latent_frames))
        right = max(
            left + 1,
            int(np.floor((index + 1) * len(source_frames) / latent_frames)),
        )
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

    preset = BACKBONE_PRESETS[args.backbone]
    config_path = args.config_path or preset["config"]
    checkpoint_path = (
        args.checkpoint_path or preset["checkpoint"]
    )
    config = OmegaConf.load(config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config["backbone_name"] = args.backbone

    # settings for editing
    config['guidance_scale'] = 1.0
    config['model_kwargs']['sink_size'] = getattr(args, 'sink_size', 0)
    if args.backbone == "self_forcing":
        config['timestep_shift'] = args.flow_shift
        config['model_kwargs']['timestep_shift'] = args.flow_shift
        config['denoising_step_list'] = np.arange(
            1000,
            0,
            -1000 / args.step,
        ).astype(int).tolist()
    else:
        config['model_kwargs']['absolute_kv_rope'] = True
        expected_frames_per_block = (
            3
            if args.backbone == "causal_forcing_chunkwise"
            else 1
        )
        if (
            config.num_frame_per_block
            != expected_frames_per_block
        ):
            raise ValueError(
                f"{args.backbone} requires "
                "num_frame_per_block="
                f"{expected_frames_per_block}, got "
                f"{config.num_frame_per_block}"
            )
    print(
        "BACKBONE_CONFIG "
        f"name={args.backbone} "
        f"config={config_path} "
        f"checkpoint={checkpoint_path}"
    )

    # Initialize the editing pipeline with the selected AR backbone.
    pipeline = EditCausalInferencePipeline(config, device=device)

    if checkpoint_path:
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
        )
        checkpoint_key = (
            "generator_ema" if args.use_ema else "generator"
        )
        if checkpoint_key not in state_dict:
            raise KeyError(
                f"Checkpoint {checkpoint_path} has no "
                f"'{checkpoint_key}' weights"
            )
        generator_state = {}
        for name, value in state_dict[checkpoint_key].items():
            if name.startswith(
                "model._fsdp_wrapped_module."
            ):
                name = name.replace(
                    "model._fsdp_wrapped_module.",
                    "model.",
                    1,
                )
            elif name.startswith("_fsdp_wrapped_module."):
                name = name.replace(
                    "_fsdp_wrapped_module.",
                    "",
                    1,
                )
            generator_state[name] = value
        pipeline.generator.load_state_dict(
            generator_state,
            strict=True,
        )
        print(
            "BACKBONE_CHECKPOINT_LOADED "
            f"name={args.backbone} key={checkpoint_key}"
        )

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
    parser.add_argument(
        "--step",
        type=int,
        default=15,
        help=(
            "Self-Forcing denoising steps. Causal Forcing presets "
            "use their checkpoint schedule."
        ),
    )
    parser.add_argument("--flow_shift", type=float, default=1.0)
    parser.add_argument(
        "--backbone",
        choices=tuple(BACKBONE_PRESETS),
        default="self_forcing",
    )

    # for Self-forcing rollout long video sampling
    parser.add_argument("--rollout_chunk_size", type=int, default=21)
    parser.add_argument("--rollout_overlap_block_num", type=int, default=1)

    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
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
            "hand_role_bayes_flow_customized_kv",
        ],
        default="dynamic_sog",
    )
    parser.add_argument("--object_mask_video", type=str, default=None)
    parser.add_argument("--hand_mask_video", type=str, default=None)
    parser.add_argument("--mask_white_threshold", type=int, default=245)
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
        "hand_role_bayes_flow_customized_kv",
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
    new_len = find_closest_num_frame(
        num_frames,
        b=pipeline.num_frame_per_block,
    )
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
    object_latent_mask = None
    hand_latent_mask = None
    if oracle_role_enabled:
        _, object_latent_mask = build_white_mask(
            args.object_mask_video,
            src_video,
            tuple(video_latents.shape),
            args.mask_white_threshold,
            min_latent_coverage=args.object_min_latent_coverage,
        )
    if oracle_role_enabled or hand_role_enabled:
        _, hand_latent_mask = build_white_mask(
            args.hand_mask_video,
            src_video,
            tuple(video_latents.shape),
            args.mask_white_threshold,
        )
    if oracle_role_enabled:
        role_input_path = Path(args.save_path).with_suffix(
            ".oracle_role_inputs.npz"
        )
        np.savez_compressed(
            role_input_path,
            object_latent_mask=object_latent_mask.numpy(),
            hand_latent_mask=hand_latent_mask.numpy(),
            white_threshold=np.array(args.mask_white_threshold),
        )
        print(
            "ORACLE_ROLE_INPUT "
            f"object={object_latent_mask.float().mean().item():.4f} "
            f"hand={hand_latent_mask.float().mean().item():.4f} "
            f"artifact={role_input_path}"
        )
    elif hand_role_enabled:
        hand_input_path = Path(args.save_path).with_suffix(
            ".hand_role_input.npz"
        )
        np.savez_compressed(
            hand_input_path,
            hand_latent_mask=hand_latent_mask.float().cpu().numpy(),
            white_threshold=np.array(args.mask_white_threshold),
        )
        print(
            "HAND_ROLE_INPUT "
            f"hand={hand_latent_mask.float().mean().item():.4f} "
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
        role_boundary_radius=args.role_boundary_radius,
        contact_target_weight=args.contact_target_weight,
        posterior_flow_mode=args.posterior_flow_mode,
        posterior_flow_use_field=args.posterior_flow_use_field,
        hand_posterior_threshold=args.hand_posterior_threshold,
        hand_max_object_coverage=args.hand_max_object_coverage,
        hand_proximity_radius=args.hand_proximity_radius,
        hand_propagation_steps=args.hand_propagation_steps,
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
