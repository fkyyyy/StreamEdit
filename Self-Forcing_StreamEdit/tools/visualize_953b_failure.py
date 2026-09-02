#!/usr/bin/env python3
"""Build an aligned video/role diagnostic sheet for a 953b run."""

from __future__ import annotations

import argparse
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_video(path: Path) -> list[Image.Image]:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_image().convert("RGB"))
    return frames


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = image.copy()
    result.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "black")
    canvas.paste(
        result, ((size[0] - result.width) // 2, (size[1] - result.height) // 2)
    )
    return canvas


def overlay_heat(image: Image.Image, heat_path: Path) -> Image.Image:
    heat = Image.open(heat_path).convert("L").resize(
        image.size, Image.Resampling.NEAREST
    )
    alpha = np.asarray(heat, dtype=np.float32) / 255.0 * 0.65
    rgb = np.asarray(image, dtype=np.float32)
    color = np.zeros_like(rgb)
    color[..., 0] = 255.0
    color[..., 1] = 60.0
    merged = rgb * (1.0 - alpha[..., None]) + color * alpha[..., None]
    return Image.fromarray(np.clip(merged, 0, 255).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    args = parser.parse_args()
    source = read_video(args.source)
    output = read_video(args.output)
    if len(source) != len(output):
        raise ValueError("Source and output frame counts differ")

    # Each three-latent block corresponds to 12 decoded frames except the
    # causal first group. Use representative frame indices at block starts,
    # centers and ends.
    frame_indices = [0, 4, 8, 12, 20, 32, 44, 56, 68, 76, 80]
    cell = (240, 180)
    label_h = 24
    columns = len(frame_indices)
    rows = 4
    sheet = Image.new(
        "RGB", (columns * cell[0], rows * (cell[1] + label_h)), "white"
    )
    draw = ImageDraw.Draw(sheet)

    for col, frame_index in enumerate(frame_indices):
        latent_index = 0 if frame_index == 0 else (frame_index + 3) // 4
        block_index = min(latent_index // 3, 6)
        role_paths = {
            "owner": args.roles / f"block_{block_index:03d}_motion_geometry_state.png",
            "target_kv": args.roles / f"block_{block_index:03d}_factorized_target_memory_action.png",
        }
        images = [source[frame_index], output[frame_index]]
        for role_name in ("owner", "target_kv"):
            role_path = role_paths[role_name]
            images.append(
                overlay_heat(output[frame_index], role_path)
                if role_path.is_file() else output[frame_index]
            )
        labels = (
            f"source f{frame_index}",
            f"output f{frame_index}",
            f"owner overlay b{block_index}",
            f"target-memory action b{block_index}",
        )
        for row, (image, label) in enumerate(zip(images, labels)):
            x = col * cell[0]
            y = row * (cell[1] + label_h)
            sheet.paste(fit(image, cell), (x, y + label_h))
            draw.text((x + 3, y + 4), label, fill="black")
    args.save.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.save)
    print(args.save)


if __name__ == "__main__":
    main()
