#!/usr/bin/env python3
"""Create mask-free frame and owner diagnostics for cook F1V.

This is an offline visualization utility. It consumes the rendered output,
the source video, and the owner debug maps written by inference. It never
uses or reconstructs an object mask.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BOUNDARIES = (9, 21, 33, 45, 57, 69)
KEY_FRAMES = (0, 4, 8, 9, 12, 20, 21, 24, 32, 33, 36, 44, 45, 48, 56, 57, 60, 68, 69, 72, 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roles-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def probe(path: Path) -> tuple[int, int]:
    value = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x", str(path),
    ], text=True).strip()
    return tuple(int(item) for item in value.split("x"))


def decode(path: Path, *, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vf", f"scale={width}:{height}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ])
    frame_bytes = width * height * 3
    if len(raw) % frame_bytes:
        raise RuntimeError(f"Unexpected decoded byte count for {path}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return (
        ImageFont.truetype(str(path), size)
        if path.exists()
        else ImageFont.load_default()
    )


def labelled(frame: np.ndarray, label: str, size=(208, 120)) -> Image.Image:
    image = Image.fromarray(frame).resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline="#555555")
    box = draw.textbbox((0, 0), label, font=font(14))
    draw.rectangle((3, 3, box[2] + 9, box[3] + 8), fill="black")
    draw.text((6, 4), label, fill="white", font=font(14))
    return image


def save_frame_grid(source: np.ndarray, output: np.ndarray, path: Path) -> None:
    frames = [index for index in KEY_FRAMES if index < len(output)]
    columns = 7
    tile_size = (208, 120)
    rows = (len(frames) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * tile_size[0], rows * 2 * tile_size[1]),
        "white",
    )
    for position, index in enumerate(frames):
        group = position // columns
        column = position % columns
        top = group * 2 * tile_size[1]
        canvas.paste(
            labelled(source[index], f"source F{index:02d}", tile_size),
            (column * tile_size[0], top),
        )
        canvas.paste(
            labelled(output[index], f"F1V F{index:02d}", tile_size),
            (column * tile_size[0], top + tile_size[1]),
        )
    canvas.save(path, quality=95)


def save_boundary_grid(source: np.ndarray, output: np.ndarray, path: Path) -> None:
    boundaries = [
        index for index in BOUNDARIES
        if 0 < index < min(len(source), len(output)) - 1
    ]
    tile_size = (208, 120)
    canvas = Image.new(
        "RGB",
        (6 * tile_size[0], len(boundaries) * tile_size[1]),
        "white",
    )
    for row, boundary in enumerate(boundaries):
        column = 0
        for name, video in (("source", source), ("F1V", output)):
            for index in range(boundary - 1, boundary + 2):
                canvas.paste(
                    labelled(video[index], f"{name} F{index:02d}", tile_size),
                    (column * tile_size[0], row * tile_size[1]),
                )
                column += 1
    canvas.save(path, quality=95)


def load_owner_rows(roles_dir: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    names = {
        "causal_owner_weight",
        "causal_owner_support",
        "causal_owner_transport",
        "causal_owner_observation",
        "causal_owner_confidence",
        "causal_owner_query_cycle_confidence",
        "velocity_owner_response_likelihood",
        "velocity_owner_signature_similarity",
        "velocity_owner_verification",
        "velocity_owner_ignition",
        "velocity_owner_verified_transport",
        "factorized_source_value_action",
        "factorized_source_residual_action",
        "factorized_target_memory_action",
    }
    for path in sorted(roles_dir.glob("block_*_hand_role_debug.npz")):
        block = int(path.name.split("_")[1])
        with np.load(path) as data:
            owner = data.get("causal_owner_support")
            for name in sorted(names.intersection(data.files)):
                value = data[name].astype(np.float64)
                if value.ndim < 2:
                    continue
                frame_values = value.reshape(value.shape[0], value.shape[1], -1).mean(axis=(0, 2))
                # Debug tensors may come from different transformer resolutions
                # (for example 1,560 versus 6,240 tokens).  Owner-conditioned
                # statistics are meaningful only when the full layout matches;
                # never pretend those flattened token grids are aligned.
                if owner is not None and owner.shape == value.shape:
                    owner_flat = owner.reshape(owner.shape[0], owner.shape[1], -1) > 0
                    value_flat = value.reshape(value.shape[0], value.shape[1], -1)
                    for frame_index in range(value.shape[1]):
                        support = owner_flat[:, frame_index]
                        owner_mean = (
                            float(value_flat[:, frame_index][support].mean())
                            if support.any()
                            else 0.0
                        )
                        rows.append({
                            "block": block,
                            "latent_frame": frame_index,
                            "metric": name,
                            "mean": float(frame_values[frame_index]),
                            "owner_mean": owner_mean,
                        })
                else:
                    for frame_index, mean in enumerate(frame_values):
                        rows.append({
                            "block": block,
                            "latent_frame": frame_index,
                            "metric": name,
                            "mean": float(mean),
                            "owner_mean": float("nan"),
                        })
    return rows


def main() -> None:
    args = parse_args()
    for dependency in ("ffmpeg", "ffprobe"):
        if shutil.which(dependency) is None:
            raise RuntimeError(f"{dependency} is required")
    if not args.source.is_file() or not args.output.is_file():
        raise FileNotFoundError("Source and rendered output must exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    size = probe(args.output)
    source = decode(args.source, size=size)
    output = decode(args.output, size=size)
    frame_count = min(len(source), len(output))
    source, output = source[:frame_count], output[:frame_count]
    save_frame_grid(source, output, args.output_dir / "f1v_keyframes.jpg")
    save_boundary_grid(source, output, args.output_dir / "f1v_boundaries.jpg")

    frame_mae = np.abs(
        output.astype(np.float32) - source.astype(np.float32)
    ).mean(axis=(1, 2, 3))
    temporal_l1 = np.zeros(frame_count, dtype=np.float64)
    if frame_count > 1:
        temporal_l1[1:] = np.abs(
            output[1:].astype(np.float32)
            - output[:-1].astype(np.float32)
        ).mean(axis=(1, 2, 3))
    frame_rows = [{
        "frame": index,
        "is_chunk_boundary": int(index in BOUNDARIES),
        "source_output_mae": float(frame_mae[index]),
        "output_temporal_l1": float(temporal_l1[index]),
    } for index in range(frame_count)]
    with (args.output_dir / "f1v_frame_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=frame_rows[0].keys())
        writer.writeheader()
        writer.writerows(frame_rows)

    owner_rows = load_owner_rows(args.roles_dir)
    if owner_rows:
        with (args.output_dir / "f1v_owner_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=owner_rows[0].keys())
            writer.writeheader()
            writer.writerows(owner_rows)
    summary = {
        "frame_count": frame_count,
        "frame_mae_mean": float(frame_mae.mean()),
        "temporal_l1_mean": float(temporal_l1[1:].mean()) if frame_count > 1 else 0.0,
        "boundary_temporal_l1": {
            str(index): float(temporal_l1[index])
            for index in BOUNDARIES if index < frame_count
        },
        "owner_metric_rows": len(owner_rows),
        "uses_object_mask": False,
    }
    (args.output_dir / "f1v_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
