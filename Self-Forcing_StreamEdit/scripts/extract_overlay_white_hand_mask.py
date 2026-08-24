#!/usr/bin/env python3
"""Extract a binary hand mask from a white-overlay mask video.

The expected input is a source video plus a mask video where the hand/forearm
has been painted white over the original RGB frame. Pure white thresholding on
that overlay also selects naturally white background regions, so this script
keeps only white pixels that differ from the aligned source frame.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def _read_video(path: Path) -> tuple[list[np.ndarray], dict]:
    reader = imageio.get_reader(str(path))
    try:
        meta = reader.get_meta_data()
        frames = [np.asarray(frame)[..., :3].copy() for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames, meta


def _resize_rgb(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(frame.astype(np.uint8))
    image = image.resize(size, resample=Image.Resampling.BILINEAR)
    return np.asarray(image)


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    image = image.resize(size, resample=Image.Resampling.NEAREST)
    return np.asarray(image) >= 128


def _filter_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    keep = np.zeros_like(mask, dtype=bool)
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            component = []
            queue = deque([(y, x)])
            visited[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                component.append((cy, cx))
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            if len(component) >= min_area:
                ys, xs = zip(*component)
                keep[np.array(ys), np.array(xs)] = True
    return keep


def _make_contact_sheet(
    source_frames: list[np.ndarray],
    mask_frames: list[np.ndarray],
    frame_indices: np.ndarray,
    binary_masks: list[np.ndarray],
    out_path: Path,
    columns: int = 8,
) -> None:
    selected = np.rint(
        np.linspace(0, len(source_frames) - 1, min(columns, len(source_frames)))
    ).astype(int)
    tile_width, tile_height, label_height = 240, 180, 24
    rows = [
        ("source", "source"),
        ("overlay", "overlay"),
        ("extracted", "mask"),
    ]
    sheet = Image.new(
        "RGB",
        (tile_width * len(selected), (tile_height + label_height) * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row_index, (label, mode) in enumerate(rows):
        for col_index, source_index in enumerate(selected):
            mask_index = int(frame_indices[source_index])
            if mode == "source":
                tile = source_frames[source_index]
            elif mode == "overlay":
                tile = mask_frames[mask_index]
            else:
                tile = np.repeat(
                    (binary_masks[source_index].astype(np.uint8) * 255)[..., None],
                    3,
                    axis=-1,
                )
            image = Image.fromarray(tile.astype(np.uint8)).resize(
                (tile_width, tile_height),
                resample=Image.Resampling.NEAREST,
            )
            x = col_index * tile_width
            y = row_index * (tile_height + label_height)
            sheet.paste(image, (x, y + label_height))
            draw.rectangle((x, y, x + tile_width, y + label_height), fill=(30, 32, 38))
            header = f"{label} f{source_index}" if col_index == 0 else f"f{source_index}"
            draw.text((x + 6, y + 4), header, fill=(245, 245, 247))
    sheet.save(out_path, quality=92)


def _write_video(path: Path, masks: list[np.ndarray], fps: float) -> bool:
    try:
        with imageio.get_writer(str(path), fps=fps, macro_block_size=1) as writer:
            for mask in masks:
                frame = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=-1)
                writer.append_data(frame)
    except Exception as exc:
        print(f"WARNING failed to write mp4 {path}: {exc}")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_video", type=Path, required=True)
    parser.add_argument("--mask_video", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--white_threshold", type=int, default=245)
    parser.add_argument("--diff_threshold", type=float, default=24.0)
    parser.add_argument("--min_component_area", type=int, default=0)
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Optionally resize extracted masks before saving.",
    )
    parser.add_argument("--preview_columns", type=int, default=8)
    parser.add_argument("--no_video", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.white_threshold <= 255:
        raise ValueError("--white_threshold must be in [0, 255]")
    if args.diff_threshold < 0:
        raise ValueError("--diff_threshold must be non-negative")
    if args.min_component_area < 0:
        raise ValueError("--min_component_area must be non-negative")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    source_frames, source_meta = _read_video(args.source_video)
    mask_frames, mask_meta = _read_video(args.mask_video)
    frame_indices = np.rint(
        np.linspace(0, len(mask_frames) - 1, len(source_frames))
    ).astype(int)

    binary_masks = []
    white_coverages = []
    extracted_coverages = []
    removed_coverages = []
    for source_index, mask_index in enumerate(frame_indices):
        source = source_frames[source_index]
        overlay = mask_frames[int(mask_index)]
        if source.shape[:2] != overlay.shape[:2]:
            source = _resize_rgb(source, (overlay.shape[1], overlay.shape[0]))
        white = np.all(overlay >= args.white_threshold, axis=-1)
        difference = np.abs(
            overlay.astype(np.int16) - source.astype(np.int16)
        ).mean(axis=-1)
        extracted = white & (difference >= args.diff_threshold)
        extracted = _filter_small_components(extracted, args.min_component_area)
        if args.resize is not None:
            out_height, out_width = args.resize
            extracted = _resize_mask(extracted, (out_width, out_height))

        binary_masks.append(extracted)
        white_coverages.append(float(white.mean()))
        extracted_coverages.append(float(extracted.mean()))
        removed_coverages.append(float((white & ~extracted).mean()))

        image = Image.fromarray(extracted.astype(np.uint8) * 255)
        image.save(frame_dir / f"mask_{source_index:06d}.png")

    preview_path = args.out_dir / "preview_source_overlay_extracted.jpg"
    _make_contact_sheet(
        source_frames,
        mask_frames,
        frame_indices,
        binary_masks,
        preview_path,
        columns=args.preview_columns,
    )

    video_path = args.out_dir / "extracted_hand_mask.mp4"
    video_written = False
    if not args.no_video:
        fps = float(mask_meta.get("fps") or source_meta.get("fps") or 15)
        video_written = _write_video(video_path, binary_masks, fps=fps)

    stats = {
        "source_video": str(args.source_video),
        "mask_video": str(args.mask_video),
        "num_source_frames": len(source_frames),
        "num_mask_frames": len(mask_frames),
        "white_threshold": args.white_threshold,
        "diff_threshold": args.diff_threshold,
        "min_component_area": args.min_component_area,
        "resize": args.resize,
        "white_coverage": {
            "mean": float(np.mean(white_coverages)),
            "min": float(np.min(white_coverages)),
            "max": float(np.max(white_coverages)),
        },
        "extracted_coverage": {
            "mean": float(np.mean(extracted_coverages)),
            "min": float(np.min(extracted_coverages)),
            "max": float(np.max(extracted_coverages)),
        },
        "removed_white_background_coverage": {
            "mean": float(np.mean(removed_coverages)),
            "min": float(np.min(removed_coverages)),
            "max": float(np.max(removed_coverages)),
        },
        "relative_removed_of_white": float(
            np.sum(removed_coverages) / max(np.sum(white_coverages), 1e-12)
        ),
        "preview": str(preview_path),
        "frames_dir": str(frame_dir),
        "video": str(video_path) if video_written else None,
    }
    stats_path = args.out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("EXTRACT_OVERLAY_WHITE_HAND_MASK")
    print(f"frames={len(source_frames)} mask_frames={len(mask_frames)}")
    print(
        "coverage "
        f"white={stats['white_coverage']['mean']:.4f} "
        f"extracted={stats['extracted_coverage']['mean']:.4f} "
        f"removed={stats['removed_white_background_coverage']['mean']:.4f} "
        f"relative_removed={stats['relative_removed_of_white']:.4f}"
    )
    print(f"preview={preview_path}")
    print(f"frames_dir={frame_dir}")
    if video_written:
        print(f"video={video_path}")
    else:
        print("video=None")
    print(f"stats={stats_path}")


if __name__ == "__main__":
    main()
