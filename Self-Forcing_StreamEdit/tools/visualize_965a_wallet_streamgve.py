#!/usr/bin/env python3
"""Offline appearance audit for the pure-StreamGVE wallet baseline.

The phone mask is used only to define a moving evaluation support. It is not
an inference input. The audit separates stable brown-shell appearance from a
cool/bright source-like spot so that a local leak is not mistaken for global
darkening.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


BOUNDARIES = (9, 21, 33, 45, 57, 69)
KEY_FRAMES = (0, 16, 32, 48, 64, 80)
EVERY4 = tuple(range(0, 81, 4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--phone-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="965a pure StreamGVE wallet")
    parser.add_argument("--prefix", default="965a")
    return parser.parse_args()


def probe_size(path: Path) -> tuple[int, int]:
    output = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path),
    ], text=True).strip()
    width, height = output.split("x")
    return int(width), int(height)


def decode(path: Path, *, filter_graph: str | None = None) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")
    width, height = probe_size(path)
    if filter_graph:
        if filter_graph == "generated":
            width, height = 832, 480
            vf = "scale=832:480"
        elif filter_graph == "source":
            width, height = 832, 480
            vf = "scale=832:624,crop=832:480:0:72"
        else:
            raise ValueError(filter_graph)
    else:
        vf = None
    command = ["ffmpeg", "-v", "error", "-i", str(path)]
    if vf:
        command.extend(["-vf", vf])
    command.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    raw = subprocess.check_output(command)
    frame_bytes = width * height * 3
    if len(raw) % frame_bytes:
        raise RuntimeError(f"Unexpected decoded byte count for {path}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def central_white_component(frame: np.ndarray) -> np.ndarray:
    """Recover the white phone overlay while rejecting wall/window whites."""
    height, width = frame.shape[:2]
    white = np.all(frame >= 245, axis=-1)
    allowed = np.zeros_like(white)
    allowed[int(0.25 * height):, int(0.20 * width):int(0.78 * width)] = True
    labels, count = ndimage.label(white & allowed)
    if count == 0:
        raise RuntimeError("No phone-mask component found")
    best_label, best_score = 0, -1.0
    yy, xx = np.indices(white.shape)
    for label in range(1, count + 1):
        component = labels == label
        area = int(component.sum())
        if area < 300:
            continue
        cx = float(xx[component].mean()) / width
        cy = float(yy[component].mean()) / height
        center_penalty = 1.0 + 8.0 * abs(cx - 0.50) + 3.0 * abs(cy - 0.68)
        score = area / center_penalty
        if score > best_score:
            best_label, best_score = label, score
    if best_label == 0:
        raise RuntimeError("No sufficiently large phone-mask component found")
    return labels == best_label


def evaluation_masks(mask_video: np.ndarray, frame_count: int, output_size: tuple[int, int]) -> np.ndarray:
    output_width, output_height = output_size
    components = [central_white_component(frame) for frame in mask_video]
    mapped: list[np.ndarray] = []
    for frame_index in range(frame_count):
        mask_index = int(round(frame_index * (len(components) - 1) / max(frame_count - 1, 1)))
        image = Image.fromarray(components[mask_index].astype(np.uint8) * 255)
        resized = np.asarray(image.resize((output_width, output_height), Image.Resampling.NEAREST)) > 0
        resized = ndimage.binary_closing(resized, iterations=4)
        resized = ndimage.binary_dilation(resized, iterations=8)
        mapped.append(resized)
    return np.stack(mapped)


def crop_box(masks: np.ndarray, margin: int = 28) -> tuple[int, int, int, int]:
    occupancy = masks.mean(axis=0) >= 0.08
    ys, xs = np.nonzero(occupancy)
    if not len(xs):
        raise RuntimeError("Empty evaluation masks")
    height, width = occupancy.shape
    return (
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(width, int(xs.max()) + margin + 1),
        min(height, int(ys.max()) + margin + 1),
    )


def classify(frame: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = frame.astype(np.float32)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    brown = (
        support
        & (red >= 1.12 * green)
        & (red >= 1.04 * blue)
        & ((np.maximum.reduce((red, green, blue)) - np.minimum.reduce((red, green, blue))) >= 18.0)
        & (red >= 45.0)
    )
    cool = support & (blue >= red + 8.0) & (blue >= green + 4.0) & (luma >= 55.0)
    return luma, brown, cool


def frame_metrics(video: np.ndarray, masks: np.ndarray) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    previous_luma: np.ndarray | None = None
    for index, (frame, support) in enumerate(zip(video, masks)):
        luma, brown, cool = classify(frame, support)
        brown_values = luma[brown]
        support_values = luma[support]
        background = ~ndimage.binary_dilation(support, iterations=32)
        background_values = luma[background]
        if not len(brown_values) or not len(support_values):
            raise RuntimeError(f"Empty color class at frame {index}")
        temporal_l1 = 0.0 if previous_luma is None else float(np.mean(np.abs(luma[support] - previous_luma[support])))
        rows.append({
            "frame": index,
            "is_boundary": int(index in BOUNDARIES),
            "support_area": int(support.sum()),
            "object_luma_median": float(np.median(support_values)),
            "brown_shell_luma_median": float(np.median(brown_values)),
            "brown_shell_luma_p25": float(np.percentile(brown_values, 25)),
            "brown_shell_fraction": float(brown.sum() / support.sum()),
            "cool_spot_fraction": float(cool.sum() / support.sum()),
            "background_luma_median": float(np.median(background_values)),
            "brown_to_background_ratio": float(
                np.median(brown_values) / max(np.median(background_values), 1e-6)
            ),
            "temporal_luma_l1": temporal_l1,
        })
        previous_luma = luma
    return rows


def tile(frame: np.ndarray, size: tuple[int, int], label: str, border: str | None = None) -> Image.Image:
    image = Image.fromarray(frame).resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=border or "#333333", width=4 if border else 1)
    label_font = font(16)
    box = draw.textbbox((0, 0), label, font=label_font)
    draw.rectangle((4, 4, box[2] + 12, box[3] + 10), fill=(0, 0, 0, 180))
    draw.text((8, 6), label, fill="white", font=label_font)
    return image


def grid(
    rows: list[tuple[str, list[tuple[int, np.ndarray, str | None]]]],
    output: Path,
    *,
    columns: int,
    tile_size: tuple[int, int],
    title: str,
) -> None:
    left, top = 145, 52
    cell_width, cell_height = tile_size
    total_rows = sum((len(items) + columns - 1) // columns for _, items in rows)
    canvas = Image.new("RGB", (left + columns * cell_width, top + total_rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), title, fill="black", font=font(21))
    row_cursor = 0
    for row_name, items in rows:
        draw.text((8, top + row_cursor * cell_height + 10), row_name, fill="black", font=font(16))
        for local_index, (frame_index, frame, border) in enumerate(items):
            row_offset, column = divmod(local_index, columns)
            canvas.paste(
                tile(frame, tile_size, f"F{frame_index:02d}", border),
                (left + column * cell_width, top + (row_cursor + row_offset) * cell_height),
            )
        row_cursor += (len(items) + columns - 1) // columns
    canvas.save(output, quality=95)


def classification_overlay(frame: np.ndarray, support: np.ndarray) -> np.ndarray:
    _, brown, cool = classify(frame, support)
    result = frame.astype(np.float32).copy()
    result[brown] = 0.62 * result[brown] + 0.38 * np.asarray((255, 155, 0), dtype=np.float32)
    result[cool] = 0.50 * result[cool] + 0.50 * np.asarray((0, 210, 255), dtype=np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def plot_metrics(
    rows: list[dict[str, float | int]], output: Path, label: str
) -> None:
    frames = np.asarray([int(row["frame"]) for row in rows])
    figure, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True)
    series = (
        ("brown_shell_luma_median", "Brown-shell median luma", "#8c2d1c"),
        ("object_luma_median", "Whole support median luma", "#555555"),
        ("cool_spot_fraction", "Cool source-like spot fraction", "#1f77b4"),
        ("brown_shell_fraction", "Brown-shell fraction", "#d95f02"),
        ("brown_to_background_ratio", "Brown / background luma", "#2ca02c"),
    )
    for axis, (key, label, color) in zip(axes, series):
        axis.plot(frames, [float(row[key]) for row in rows], color=color, linewidth=2)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        for boundary in BOUNDARIES:
            axis.axvline(boundary, color="#d62728", linestyle="--", alpha=0.45)
    axes[-1].set_xlabel("RGB frame; dashed lines are causal-block boundaries")
    figure.suptitle(f"{label}: global color versus local source-like spot")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def period_summary(rows: list[dict[str, float | int]], start: int, end: int) -> dict[str, float | list[int]]:
    selected = rows[start:end + 1]
    keys = (
        "object_luma_median", "brown_shell_luma_median",
        "brown_shell_fraction", "cool_spot_fraction",
        "background_luma_median", "brown_to_background_ratio",
    )
    result: dict[str, float | list[int]] = {"frames": [start, end]}
    for key in keys:
        result[f"{key}_mean"] = float(np.mean([float(row[key]) for row in selected]))
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video = decode(args.video, filter_graph="generated")
    source = decode(args.source, filter_graph="source")
    phone_mask = decode(args.phone_mask)
    if len(video) != 81 or len(source) < 81:
        raise RuntimeError(f"Expected 81 generated/source frames, got {len(video)}/{len(source)}")
    source = source[:81]
    masks = evaluation_masks(phone_mask, len(video), (video.shape[2], video.shape[1]))
    crop = crop_box(masks)
    x0, y0, x1, y1 = crop
    metrics = frame_metrics(video, masks)
    prefix = args.prefix

    grid(
        [("965a output", [(index, video[index], None) for index in EVERY4])],
        args.output_dir / f"{prefix}_timeline_every4.jpg", columns=7, tile_size=(280, 162),
        title=f"{args.label} - every fourth RGB frame",
    )
    grid(
        [("wallet crop", [(index, video[index, y0:y1, x0:x1], None) for index in EVERY4])],
        args.output_dir / f"{prefix}_wallet_crop_every4.jpg", columns=7, tile_size=(230, 230),
        title=f"Fixed offline phone/wallet crop {crop}; no per-frame recentering",
    )
    grid(
        [("wallet crop", [(index, video[index, y0:y1, x0:x1], None) for index in range(81)])],
        args.output_dir / f"{prefix}_wallet_crop_all_frames.jpg", columns=9, tile_size=(180, 180),
        title=f"{args.label} - all 81 RGB frames",
    )
    grid(
        [("B6 F69-F80", [(index, video[index, y0:y1, x0:x1], None) for index in range(69, 81)])],
        args.output_dir / f"{prefix}_b6_all_frames_focus.jpg", columns=6, tile_size=(290, 290),
        title=f"{args.label} - final causal block, fixed crop",
    )
    grid(
        [("F67-F80", [(index, video[index, y0:y1, x0:x1], "#d62728" if index == 69 else None) for index in range(67, 81)])],
        args.output_dir / f"{prefix}_b5_b6_transition_focus.jpg", columns=7, tile_size=(250, 250),
        title="Final block transition; red border marks B6 first frame F69",
    )
    grid(
        [("B6 full frame", [(index, video[index], None) for index in range(69, 81)])],
        args.output_dir / f"{prefix}_b6_full_frames.jpg", columns=4, tile_size=(416, 240),
        title=f"{args.label} - final causal block, full frame",
    )
    boundary_rows = []
    for boundary in BOUNDARIES:
        items = []
        for index in range(boundary - 2, boundary + 3):
            border = "#d62728" if index == boundary else None
            items.append((index, video[index, y0:y1, x0:x1], border))
        boundary_rows.append((f"enter F{boundary:02d}", items))
    grid(
        boundary_rows, args.output_dir / f"{prefix}_block_boundaries_pm2.jpg", columns=5, tile_size=(245, 245),
        title="Wallet around causal-block boundaries; red border is first new-block frame",
    )
    grid(
        [
            ("source phone", [(index, source[index, y0:y1, x0:x1], None) for index in KEY_FRAMES]),
            ("965a wallet", [(index, video[index, y0:y1, x0:x1], None) for index in KEY_FRAMES]),
            ("color audit", [(index, classification_overlay(video[index], masks[index])[y0:y1, x0:x1], None) for index in KEY_FRAMES]),
        ],
        args.output_dir / f"{prefix}_source_wallet_color_audit.jpg", columns=6, tile_size=(230, 230),
        title="Offline audit: orange=brown shell, cyan=cool source-like spot",
    )
    plot_metrics(metrics, args.output_dir / f"{prefix}_appearance_metrics.png", args.label)

    with (args.output_dir / f"{prefix}_appearance_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(metrics[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(metrics)

    brown_luma = np.asarray([float(row["brown_shell_luma_median"]) for row in metrics])
    cool_fraction = np.asarray([float(row["cool_spot_fraction"]) for row in metrics])
    early_threshold = float(cool_fraction[:21].mean() + max(0.02, 3.0 * cool_fraction[:21].std()))
    candidates = np.nonzero(cool_fraction > early_threshold)[0]
    boundary_jumps = {
        str(boundary): {
            "brown_shell_luma_delta": float(brown_luma[boundary] - brown_luma[boundary - 1]),
            "cool_spot_fraction_delta": float(cool_fraction[boundary] - cool_fraction[boundary - 1]),
            "brown_luma_post4_minus_pre4": float(
                brown_luma[boundary:boundary + 4].mean()
                - brown_luma[boundary - 4:boundary].mean()
            ),
        } for boundary in BOUNDARIES
    }
    block_ranges = ((0, 8), (9, 20), (21, 32), (33, 44), (45, 56), (57, 68), (69, 80))
    block_summaries = {}
    for block_index, (start, end) in enumerate(block_ranges):
        block = metrics[start:end + 1]
        block_summaries[str(block_index)] = {
            **period_summary(metrics, start, end),
            "brown_shell_luma_first": float(block[0]["brown_shell_luma_median"]),
            "brown_shell_luma_last": float(block[-1]["brown_shell_luma_median"]),
            "brown_shell_luma_within_block_delta": float(
                block[-1]["brown_shell_luma_median"]
                - block[0]["brown_shell_luma_median"]
            ),
        }
    summary = {
        "video_frames": len(video),
        "offline_mask_frames": len(phone_mask),
        "fixed_crop_xyxy": list(crop),
        "block_boundaries": list(BOUNDARIES),
        "periods": {
            "early": period_summary(metrics, 0, 20),
            "middle": period_summary(metrics, 32, 52),
            "late": period_summary(metrics, 60, 80),
        },
        "brown_shell_luma_late_minus_early": float(brown_luma[60:].mean() - brown_luma[:21].mean()),
        "brown_shell_luma_relative_change": float(brown_luma[60:].mean() / brown_luma[:21].mean() - 1.0),
        "cool_spot_late_minus_early": float(cool_fraction[60:].mean() - cool_fraction[:21].mean()),
        "cool_spot_detection_threshold": early_threshold,
        "cool_spot_first_detected_frame": int(candidates[0]) if len(candidates) else None,
        "boundary_jumps": boundary_jumps,
        "causal_blocks": block_summaries,
        "interpretation_note": "phone mask used offline only; brown and cool classes are color diagnostics, not semantic segmentation",
    }
    with (args.output_dir / f"{prefix}_appearance_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
