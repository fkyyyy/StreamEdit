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
        ],
        default="dynamic_sog",
    )
    parser.add_argument("--object_mask_video", type=str, default=None)
    parser.add_argument("--hand_mask_video", type=str, default=None)
    parser.add_argument("--mask_white_threshold", type=int, default=245)
    parser.add_argument(
        "--object_min_latent_coverage", type=float, default=0.001
    )
    parser.add_argument("--role_boundary_radius", type=int, default=1)
    parser.add_argument(
        "--contact_target_weight",
        type=float,
        default=0.7,
        help=(
            "Target-field weight for Oracle contact tokens in "
            "Oracle residual modes; must be in [0, 1]."
        ),
    )
    parser.add_argument("--save_role_dir", type=str, default=None)
    args = parser.parse_args()
    oracle_role_enabled = args.routing_mode in {
        "oracle_role_flow",
        "oracle_role_flow_kv",
        "oracle_role_residual",
        "oracle_role_residual_kv",
    }
    if oracle_role_enabled and (
        args.object_mask_video is None or args.hand_mask_video is None
    ):
        parser.error(
            "Oracle role modes require --object_mask_video and "
            "--hand_mask_video"
        )
    if not 0.0 <= args.contact_target_weight <= 1.0:
        parser.error("--contact_target_weight must be in [0, 1]")

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
        _, hand_latent_mask = build_white_mask(
            args.hand_mask_video,
            src_video,
            tuple(video_latents.shape),
            args.mask_white_threshold,
        )
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
            if hand_latent_mask is None
            else hand_latent_mask.unsqueeze(0).to(device=device)
        ),
        role_boundary_radius=args.role_boundary_radius,
        contact_target_weight=args.contact_target_weight,
        save_role_dir=args.save_role_dir,
    )

    # Clear VAE cache
    pipeline.vae.model.clear_cache()
    write_video(
        args.save_path, 
        rearrange(edit_video[0], 't c h w -> t h w c').cpu() * 255, 
        fps=16
    )
