#!/usr/bin/env python3
"""Compare P0 brightness, source-like glare, and geometry stability.

The phone matte is strictly an offline evaluation support.  P0 itself uses no
automatic or external object region during inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import stats

import analyze_F0R_jitter as jitter
import visualize_965a_wallet_streamgve as appearance


METHODS = ("L0", "F0R", "P0")
COLORS = {"L0": "#2ca02c", "F0R": "#d62728", "P0": "#1f77b4"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--phone-mask", type=Path, required=True)
    parser.add_argument("--l0", type=Path, required=True)
    parser.add_argument("--f0r", type=Path, required=True)
    parser.add_argument("--p0", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def transition_rows(
    videos: dict[str, np.ndarray],
    gt: np.ndarray,
    binary: dict[str, np.ndarray],
    geometry: dict[str, list[dict[str, float]]],
) -> list[dict[str, float | int]]:
    rows = []
    for frame in range(1, jitter.FRAME_COUNT):
        row: dict[str, float | int] = {
            "target_frame": frame,
            "is_block_boundary": int(frame in jitter.BOUNDARIES),
        }
        for name in METHODS:
            previous, current = geometry[name][frame - 1], geometry[name][frame]
            dx = current["hard_offset_x"] - previous["hard_offset_x"]
            dy = current["hard_offset_y"] - previous["hard_offset_y"]
            union = binary[name][frame - 1] | binary[name][frame]
            intersection = binary[name][frame - 1] & binary[name][frame]
            row[f"{name}_hard_relative_step_px"] = float(np.hypot(dx, dy))
            row[f"{name}_temporal_iou"] = float(intersection.sum() / max(union.sum(), 1))
            row[f"{name}_angle_step_deg"] = float(jitter.angle_delta_degrees(
                previous["hard_angle"], current["hard_angle"]
            ))
        rows.append(row)
    return rows


def summarize_transition(rows: list[dict[str, float | int]], name: str) -> dict[str, object]:
    boundary = np.asarray([bool(row["is_block_boundary"]) for row in rows])
    step = np.asarray([float(row[f"{name}_hard_relative_step_px"]) for row in rows])
    iou = np.asarray([float(row[f"{name}_temporal_iou"]) for row in rows])
    angle = np.asarray([float(row[f"{name}_angle_step_deg"]) for row in rows])
    return {
        "mean_step_px": float(step.mean()),
        "median_step_px": float(np.median(step)),
        "p90_step_px": float(np.percentile(step, 90)),
        "boundary_step_px": float(step[boundary].mean()),
        "nonboundary_step_px": float(step[~boundary].mean()),
        "boundary_to_nonboundary_ratio": float(step[boundary].mean() / step[~boundary].mean()),
        "mean_temporal_iou": float(iou.mean()),
        "mean_angle_step_deg": float(angle.mean()),
        "largest_step_frames": [int(index + 1) for index in np.argsort(step)[-8:][::-1]],
    }


def correlations(
    appearance_rows: list[dict[str, float | int]],
    temporal_rows: list[dict[str, float | int]],
    name: str,
) -> dict[str, float]:
    # Transition into frame f is paired with the mean appearance of f-1 and f.
    luma = np.asarray([float(row["brown_shell_luma_median"]) for row in appearance_rows])
    cool = np.asarray([float(row["cool_spot_fraction"]) for row in appearance_rows])
    transition_luma = 0.5 * (luma[:-1] + luma[1:])
    transition_cool = 0.5 * (cool[:-1] + cool[1:])
    step = np.asarray([float(row[f"{name}_hard_relative_step_px"]) for row in temporal_rows])
    luma_spearman = stats.spearmanr(transition_luma, step)
    cool_spearman = stats.spearmanr(transition_cool, step)
    return {
        "luma_vs_geometry_step_spearman_r": float(luma_spearman.statistic),
        "luma_vs_geometry_step_p": float(luma_spearman.pvalue),
        "cool_spot_vs_geometry_step_spearman_r": float(cool_spearman.statistic),
        "cool_spot_vs_geometry_step_p": float(cool_spearman.pvalue),
    }


def make_all_frames(
    videos: dict[str, np.ndarray], gt: np.ndarray, output: Path
) -> None:
    crop = jitter.crop_box(gt, margin=32)
    x0, y0, x1, y1 = crop
    columns, tile = 9, 150
    rows_per_method = 9
    left, top = 88, 40
    canvas = Image.new(
        "RGB",
        (left + columns * tile, top + len(METHODS) * rows_per_method * tile),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "All 81 frames, fixed crop; rows grouped by method", fill="black", font=font(18))
    for method_index, name in enumerate(METHODS):
        base_row = method_index * rows_per_method
        draw.text((8, top + base_row * tile + 8), name, fill=COLORS[name], font=font(17))
        for frame in range(jitter.FRAME_COUNT):
            local_row, column = divmod(frame, columns)
            image = Image.fromarray(videos[name][frame, y0:y1, x0:x1]).resize(
                (tile, tile), Image.Resampling.LANCZOS
            )
            painter = ImageDraw.Draw(image)
            painter.rectangle((2, 2, 39, 23), fill=(0, 0, 0))
            painter.text((5, 4), f"F{frame:02d}", fill="white", font=font(12))
            if frame in jitter.BOUNDARIES:
                painter.rectangle((1, 1, tile - 2, tile - 2), outline=(220, 0, 0), width=4)
            canvas.paste(image, (left + column * tile, top + (base_row + local_row) * tile))
    canvas.save(output, quality=94)


def plot(
    temporal_rows: list[dict[str, float | int]],
    appearance_rows: dict[str, list[dict[str, float | int]]],
    output: Path,
) -> None:
    frames = np.arange(1, jitter.FRAME_COUNT)
    full_frames = np.arange(jitter.FRAME_COUNT)
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for name in METHODS:
        axes[0].plot(full_frames, [row["brown_shell_luma_median"] for row in appearance_rows[name]], label=name, color=COLORS[name])
        axes[1].plot(full_frames, [row["cool_spot_fraction"] for row in appearance_rows[name]], label=name, color=COLORS[name])
        axes[2].plot(frames, [row[f"{name}_hard_relative_step_px"] for row in temporal_rows], label=name, color=COLORS[name])
    for axis in axes:
        for boundary in jitter.BOUNDARIES:
            axis.axvline(boundary, color="black", linestyle="--", alpha=0.22)
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")
    axes[0].set_ylabel("brown-shell luma")
    axes[1].set_ylabel("cool source-like fraction")
    axes[2].set_ylabel("GT-relative silhouette step (px)")
    axes[2].set_xlabel("RGB frame; dashed = causal-block boundary")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = jitter.decode(args.source)
    videos = {
        "L0": jitter.decode(args.l0),
        "F0R": jitter.decode(args.f0r),
        "P0": jitter.decode(args.p0),
    }
    if len(source) != jitter.FRAME_COUNT or any(len(value) != jitter.FRAME_COUNT for value in videos.values()):
        raise RuntimeError("Expected exactly 81 frames for every input")
    mask_video = jitter.decode(args.phone_mask, resize=False)
    gt = jitter.evaluation_masks(mask_video)
    binary, geometry, thresholds = {}, {}, {}
    appearance_rows = {}
    _, _, common_threshold = jitter.edit_maps(videos["L0"], source, gt)
    for name, video in videos.items():
        binary[name], weights, thresholds[name] = jitter.edit_maps(
            video, source, gt, threshold=common_threshold
        )
        geometry[name] = jitter.per_frame_geometry(binary[name], weights, gt)
        appearance_rows[name] = appearance.frame_metrics(video, gt)
    temporal = transition_rows(videos, gt, binary, geometry)
    summary = {
        "offline_phone_mask_only": True,
        "common_edit_support_threshold_from": "L0",
        "edit_support_thresholds": thresholds,
        "methods": {},
    }
    for name in METHODS:
        summary["methods"][name] = {
            **summarize_transition(temporal, name),
            **correlations(appearance_rows[name], temporal, name),
            "early_luma_mean": float(np.mean([row["brown_shell_luma_median"] for row in appearance_rows[name][:21]])),
            "late_luma_mean": float(np.mean([row["brown_shell_luma_median"] for row in appearance_rows[name][60:]])),
            "late_cool_spot_mean": float(np.mean([row["cool_spot_fraction"] for row in appearance_rows[name][60:]])),
        }
    with (args.output_dir / "P0_comparison_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (args.output_dir / "P0_comparison_temporal.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(temporal[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(temporal)
    make_all_frames(videos, gt, args.output_dir / "P0_L0_F0R_all_81_frames.jpg")
    plot(temporal, appearance_rows, args.output_dir / "P0_L0_F0R_metrics.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
