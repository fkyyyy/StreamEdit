#!/usr/bin/env python3
"""Precompute bidirectional RAFT flow on clean source video frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import types

import av
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.transforms import functional as TF

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# ``pipeline.__init__`` eagerly imports the full Wan stack and initializes
# CUDA.  Flow extraction is intentionally standalone, so expose only the
# package search path and import the lightweight motion modules below.
if "pipeline" not in sys.modules:
    pipeline_package = types.ModuleType("pipeline")
    pipeline_package.__path__ = [str(REPO_ROOT / "pipeline")]
    sys.modules["pipeline"] = pipeline_package

from pipeline.motion.flow_geometry import forward_backward_confidence
from pipeline.motion.raft_backend import TorchvisionRAFT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract source-only bidirectional RAFT flow for causal region "
            "tracking. No hand or object mask is consumed."
        )
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--fb-alpha", type=float, default=0.01)
    parser.add_argument("--fb-beta", type=float, default=0.5)
    parser.add_argument("--preview-pairs", type=int, default=8)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and replace an existing cache",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_video(
    path: Path,
    *,
    height: int,
    width: int,
) -> tuple[torch.Tensor, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Source video not found: {path}")
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        average_rate = float(stream.average_rate) if stream.average_rate else 0.0
        original_width = int(stream.width)
        original_height = int(stream.height)
        for decoded in container.decode(stream):
            image = decoded.to_image().convert("RGB")
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            frames.append(TF.pil_to_tensor(image))
    if len(frames) < 2:
        raise RuntimeError("At least two decoded video frames are required")
    return torch.stack(frames), {
        "fps": average_rate,
        "original_height": original_height,
        "original_width": original_width,
    }


def flow_to_rgb(flow: torch.Tensor, scale: float) -> np.ndarray:
    flow = flow.float().cpu()
    magnitude = torch.linalg.vector_norm(flow, dim=0)
    angle = torch.atan2(flow[1], flow[0])
    hue = ((angle + math.pi) / (2.0 * math.pi) * 255.0).byte().numpy()
    saturation = np.full_like(hue, 255, dtype=np.uint8)
    value = (magnitude / max(scale, 1e-6)).clamp(0.0, 1.0)
    value = (255.0 * torch.sqrt(value)).byte().numpy()
    hsv = np.stack((hue, saturation, value), axis=-1)
    return np.asarray(Image.fromarray(hsv).convert("RGB"))


def sampled_quantile(
    value: torch.Tensor,
    quantile: float,
    *,
    max_samples: int = 1_000_000,
) -> torch.Tensor:
    """Compute a deterministic approximate quantile for large tensors.

    ``torch.quantile`` rejects tensors with more than 2**24 elements on the
    current PyTorch build. Dense video flow exceeds that limit, while a fixed
    uniform sample is sufficient for metadata and visualization scaling.
    """

    flat = value.float().reshape(-1)
    if flat.numel() > max_samples:
        stride = math.ceil(flat.numel() / max_samples)
        flat = flat[::stride]
    return torch.quantile(flat, quantile)


def save_contact_sheet(
    path: Path,
    frames: torch.Tensor,
    forward: torch.Tensor,
    confidence: torch.Tensor,
    occlusion: torch.Tensor,
    count: int,
) -> None:
    pair_count = forward.shape[0]
    indices = np.unique(
        np.rint(np.linspace(0, pair_count - 1, min(count, pair_count))).astype(int)
    )
    preview_width = 260
    preview_height = 150
    label_height = 22
    columns = 5
    sheet = Image.new(
        "RGB",
        (columns * preview_width, len(indices) * (preview_height + label_height)),
        "black",
    )
    draw = ImageDraw.Draw(sheet)
    magnitude = torch.linalg.vector_norm(forward.float(), dim=1)
    active = magnitude[magnitude > 0]
    scale = float(sampled_quantile(active, 0.95)) if active.numel() else 1.0
    headings = ("frame t", "frame t+1", "forward flow", "FB confidence", "occlusion")
    for row, pair_index in enumerate(indices.tolist()):
        arrays = [
            frames[pair_index].permute(1, 2, 0).numpy(),
            frames[pair_index + 1].permute(1, 2, 0).numpy(),
            flow_to_rgb(forward[pair_index], scale),
            np.repeat(
                (confidence[pair_index, 0].float().numpy() * 255.0)
                .clip(0, 255).astype(np.uint8)[..., None],
                3,
                axis=-1,
            ),
            np.repeat(
                (occlusion[pair_index, 0].numpy().astype(np.uint8) * 255)[..., None],
                3,
                axis=-1,
            ),
        ]
        top = row * (preview_height + label_height)
        for column, (heading, array) in enumerate(zip(headings, arrays)):
            image = Image.fromarray(array).resize(
                (preview_width, preview_height), Image.Resampling.BILINEAR
            )
            left = column * preview_width
            sheet.paste(image, (left, top + label_height))
            draw.text(
                (left + 4, top + 4),
                f"{heading} [{pair_index}->{pair_index + 1}]",
                fill="white",
            )
    sheet.save(path)


def main() -> None:
    args = parse_args()
    if args.height <= 0 or args.width <= 0:
        raise ValueError("height and width must be positive")
    if args.height % 8 or args.width % 8:
        raise ValueError("RAFT height and width must be divisible by 8")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    video_path = args.video.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_digest = sha256(video_path)
    checkpoint_digest = sha256(checkpoint_path)

    frames, video_metadata = decode_video(
        video_path, height=args.height, width=args.width
    )
    print(
        "SOURCE_FLOW_INPUT "
        f"frames={frames.shape[0]} resized={args.width}x{args.height} "
        "external_hand_mask=disabled external_object_mask=disabled"
    )
    pair_count = frames.shape[0] - 1
    cache_path = output_dir / "raft_large_bidirectional_flow.pt"
    metadata_path = output_dir / "metadata.json"
    if cache_path.is_file() and not args.force:
        if not metadata_path.is_file():
            raise RuntimeError(
                "Existing flow cache has no metadata; use --force to "
                "replace it safely"
            )
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_identity = {
            "video_sha256": video_digest,
            "checkpoint_sha256": checkpoint_digest,
            "resized_size": [args.height, args.width],
            "fb_alpha": args.fb_alpha,
            "fb_beta": args.fb_beta,
        }
        mismatched = {
            name: (existing_metadata.get(name), expected)
            for name, expected in expected_identity.items()
            if existing_metadata.get(name) != expected
        }
        if mismatched:
            raise RuntimeError(
                "Existing flow cache belongs to a different input/config; "
                f"use another output directory or --force: {mismatched}"
            )
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        required = {
            "forward_flow",
            "backward_flow",
            "forward_confidence",
            "backward_confidence",
            "forward_occlusion",
            "backward_occlusion",
            "forward_consistency_error",
            "backward_consistency_error",
        }
        missing = required.difference(cache)
        if missing:
            raise RuntimeError(
                f"Existing flow cache is incomplete: {sorted(missing)}"
            )
        forward = cache["forward_flow"]
        backward = cache["backward_flow"]
        confidence_forward = cache["forward_confidence"]
        confidence_backward = cache["backward_confidence"]
        occlusion_forward = cache["forward_occlusion"]
        occlusion_backward = cache["backward_occlusion"]
        error_forward = cache["forward_consistency_error"]
        error_backward = cache["backward_consistency_error"]
        expected_flow_shape = (pair_count, 2, args.height, args.width)
        if tuple(forward.shape) != expected_flow_shape:
            raise RuntimeError(
                "Existing flow cache does not match this video/config: "
                f"{tuple(forward.shape)} != {expected_flow_shape}"
            )
        print(f"SOURCE_FLOW_REUSE cache={cache_path}")
    else:
        estimator = TorchvisionRAFT(checkpoint_path, device=args.device)
        forward_parts = []
        backward_parts = []
        confidence_forward_parts = []
        confidence_backward_parts = []
        occlusion_forward_parts = []
        occlusion_backward_parts = []
        error_forward_parts = []
        error_backward_parts = []
        for left in range(0, pair_count, args.batch_size):
            right = min(left + args.batch_size, pair_count)
            forward_batch, backward_batch = estimator.estimate_bidirectional(
                frames[left:right], frames[left + 1:right + 1]
            )
            confidence_forward_batch, occlusion_forward_batch, error_forward_batch = (
                forward_backward_confidence(
                    forward_batch, backward_batch,
                    alpha=args.fb_alpha, beta=args.fb_beta,
                )
            )
            confidence_backward_batch, occlusion_backward_batch, error_backward_batch = (
                forward_backward_confidence(
                    backward_batch, forward_batch,
                    alpha=args.fb_alpha, beta=args.fb_beta,
                )
            )
            forward_parts.append(forward_batch.cpu().half())
            backward_parts.append(backward_batch.cpu().half())
            confidence_forward_parts.append(confidence_forward_batch.cpu().half())
            confidence_backward_parts.append(confidence_backward_batch.cpu().half())
            occlusion_forward_parts.append(occlusion_forward_batch.cpu())
            occlusion_backward_parts.append(occlusion_backward_batch.cpu())
            error_forward_parts.append(error_forward_batch.cpu().half())
            error_backward_parts.append(error_backward_batch.cpu().half())
            print(f"SOURCE_FLOW_PROGRESS pairs={right}/{pair_count}", flush=True)

        forward = torch.cat(forward_parts)
        backward = torch.cat(backward_parts)
        confidence_forward = torch.cat(confidence_forward_parts)
        confidence_backward = torch.cat(confidence_backward_parts)
        occlusion_forward = torch.cat(occlusion_forward_parts)
        occlusion_backward = torch.cat(occlusion_backward_parts)
        error_forward = torch.cat(error_forward_parts)
        error_backward = torch.cat(error_backward_parts)
        torch.save(
            {
                "forward_flow": forward,
                "backward_flow": backward,
                "forward_confidence": confidence_forward,
                "backward_confidence": confidence_backward,
                "forward_occlusion": occlusion_forward,
                "backward_occlusion": occlusion_backward,
                "forward_consistency_error": error_forward,
                "backward_consistency_error": error_backward,
            },
            cache_path,
        )

    flow_magnitude = torch.linalg.vector_norm(forward.float(), dim=1)
    metadata = {
        "format_version": 1,
        "video": str(video_path),
        "video_sha256": video_digest,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "model": "torchvision_raft_large_C_T_SKHT_V2",
        "frame_count": int(frames.shape[0]),
        "pair_count": int(pair_count),
        "fps": video_metadata["fps"],
        "original_size": [
            video_metadata["original_height"],
            video_metadata["original_width"],
        ],
        "resized_size": [args.height, args.width],
        "resize_mode": "exact_bilinear_matching_inference",
        "flow_convention": "pixel_dx_dy_defined_on_origin_frame",
        "tensor_file": cache_path.name,
        "tensor_dtype": "float16_flow_confidence_error_bool_occlusion",
        "forward_flow_shape": list(forward.shape),
        "fb_alpha": args.fb_alpha,
        "fb_beta": args.fb_beta,
        "mean_flow_magnitude_px": float(flow_magnitude.mean()),
        "p95_flow_magnitude_px": float(
            sampled_quantile(flow_magnitude, 0.95)
        ),
        "mean_forward_confidence": float(confidence_forward.float().mean()),
        "forward_occlusion_fraction": float(occlusion_forward.float().mean()),
        "input_contract": {
            "clean_source_rgb": True,
            "external_hand_mask": False,
            "external_object_mask": False,
            "generated_video": False,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    preview_path = output_dir / "flow_contact_sheet.png"
    save_contact_sheet(
        preview_path,
        frames,
        forward,
        confidence_forward,
        occlusion_forward,
        args.preview_pairs,
    )
    print(
        "SOURCE_FLOW_COMPLETE "
        f"cache={cache_path} metadata={metadata_path} preview={preview_path} "
        f"mean_magnitude={metadata['mean_flow_magnitude_px']:.4f} "
        f"p95_magnitude={metadata['p95_flow_magnitude_px']:.4f} "
        f"mean_confidence={metadata['mean_forward_confidence']:.4f} "
        f"occlusion={metadata['forward_occlusion_fraction']:.4f}"
    )


if __name__ == "__main__":
    main()
