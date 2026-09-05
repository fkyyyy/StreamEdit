#!/usr/bin/env python3
"""Offline framewise comparison of P0 and P1.

The phone matte is evaluation-only and is never an inference input.
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


METHODS = ("P0", "P1")
COLORS = {"P0": "#1f77b4", "P1": "#ff7f0e"}
BLOCKS = ((0, 8), (9, 20), (21, 32), (33, 44), (45, 56), (57, 68), (69, 80))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--phone-mask", type=Path, required=True)
    parser.add_argument("--p0", type=Path, required=True)
    parser.add_argument("--p1", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def frame_rows(videos, masks, appearance_rows, binary, geometry):
    rows = []
    for frame in range(jitter.FRAME_COUNT):
        support = masks[frame]
        difference = np.abs(
            videos["P1"][frame].astype(np.float32)
            - videos["P0"][frame].astype(np.float32)
        )
        row = {
            "frame": frame,
            "block": next(index for index, (left, right) in enumerate(BLOCKS) if left <= frame <= right),
            "is_boundary": int(frame in jitter.BOUNDARIES),
            "P0_P1_support_rgb_mae": float(difference[support].mean()),
        }
        for name in METHODS:
            appearance_row = appearance_rows[name][frame]
            row[f"{name}_brown_luma"] = float(appearance_row["brown_shell_luma_median"])
            row[f"{name}_cool_fraction"] = float(appearance_row["cool_spot_fraction"])
            row[f"{name}_brown_fraction"] = float(appearance_row["brown_shell_fraction"])
            if frame == 0:
                row[f"{name}_geometry_step_px"] = 0.0
            else:
                previous, current = geometry[name][frame - 1], geometry[name][frame]
                row[f"{name}_geometry_step_px"] = float(np.hypot(
                    current["hard_offset_x"] - previous["hard_offset_x"],
                    current["hard_offset_y"] - previous["hard_offset_y"],
                ))
        row["P1_minus_P0_brown_luma"] = row["P1_brown_luma"] - row["P0_brown_luma"]
        row["P1_minus_P0_cool_fraction"] = row["P1_cool_fraction"] - row["P0_cool_fraction"]
        rows.append(row)
    return rows


def method_summary(rows, name):
    steps = np.asarray([row[f"{name}_geometry_step_px"] for row in rows[1:]])
    return {
        "mean_geometry_step_px": float(steps.mean()),
        "p90_geometry_step_px": float(np.percentile(steps, 90)),
        "early_brown_luma": float(np.mean([row[f"{name}_brown_luma"] for row in rows[:21]])),
        "late_brown_luma": float(np.mean([row[f"{name}_brown_luma"] for row in rows[60:]])),
        "late_cool_fraction": float(np.mean([row[f"{name}_cool_fraction"] for row in rows[60:]])),
    }


def block_summaries(rows):
    summaries = {}
    for block, (left, right) in enumerate(BLOCKS):
        selected = rows[left:right + 1]
        summaries[str(block)] = {
            "frames": [left, right],
            "P0_brown_luma": float(np.mean([row["P0_brown_luma"] for row in selected])),
            "P1_brown_luma": float(np.mean([row["P1_brown_luma"] for row in selected])),
            "P1_minus_P0_brown_luma": float(np.mean([row["P1_minus_P0_brown_luma"] for row in selected])),
            "P0_cool_fraction": float(np.mean([row["P0_cool_fraction"] for row in selected])),
            "P1_cool_fraction": float(np.mean([row["P1_cool_fraction"] for row in selected])),
            "P1_minus_P0_cool_fraction": float(np.mean([row["P1_minus_P0_cool_fraction"] for row in selected])),
            "P0_geometry_step_px": float(np.mean([row["P0_geometry_step_px"] for row in selected if row["frame"] > left])),
            "P1_geometry_step_px": float(np.mean([row["P1_geometry_step_px"] for row in selected if row["frame"] > left])),
            "P0_P1_support_rgb_mae": float(np.mean([row["P0_P1_support_rgb_mae"] for row in selected])),
        }
    return summaries


def all_frames(videos, masks, output):
    x0, y0, x1, y1 = jitter.crop_box(masks, margin=32)
    columns, tile, top, left = 9, 150, 38, 75
    rows_per_method = 9
    canvas = Image.new("RGB", (left + columns * tile, top + 18 * tile), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), "P0 vs P1: all 81 frames; red border = block start", fill="black", font=font(18))
    for method_index, name in enumerate(METHODS):
        base = method_index * rows_per_method
        draw.text((8, top + base * tile + 8), name, fill=COLORS[name], font=font(17))
        for frame in range(jitter.FRAME_COUNT):
            row, column = divmod(frame, columns)
            image = Image.fromarray(videos[name][frame, y0:y1, x0:x1]).resize((tile, tile), Image.Resampling.LANCZOS)
            painter = ImageDraw.Draw(image)
            painter.rectangle((2, 2, 39, 23), fill=(0, 0, 0))
            painter.text((5, 4), f"F{frame:02d}", fill="white", font=font(12))
            if frame in jitter.BOUNDARIES:
                painter.rectangle((1, 1, tile - 2, tile - 2), outline=(220, 0, 0), width=4)
            canvas.paste(image, (left + column * tile, top + (base + row) * tile))
    canvas.save(output, quality=94)


def plot(rows, output):
    frames = np.arange(jitter.FRAME_COUNT)
    figure, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fields = ("brown_luma", "cool_fraction", "geometry_step_px")
    labels = ("brown-shell luma", "cool source-like fraction", "GT-relative silhouette step (px)")
    for name in METHODS:
        for axis, field in zip(axes[:3], fields):
            axis.plot(frames, [row[f"{name}_{field}"] for row in rows], label=name, color=COLORS[name])
    axes[3].plot(frames, [row["P1_minus_P0_brown_luma"] for row in rows], label="P1-P0 luma", color="#9467bd")
    axes[3].plot(frames, [100.0 * row["P1_minus_P0_cool_fraction"] for row in rows], label="100*(P1-P0 cool fraction)", color="#17becf")
    for axis in axes:
        for boundary in jitter.BOUNDARIES:
            axis.axvline(boundary, color="black", linestyle="--", alpha=0.22)
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")
    for axis, label in zip(axes[:3], labels):
        axis.set_ylabel(label)
    axes[3].set_ylabel("P1 - P0 delta")
    axes[3].set_xlabel("RGB frame; dashed = causal-block boundary")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = jitter.decode(args.source)
    videos = {"P0": jitter.decode(args.p0), "P1": jitter.decode(args.p1)}
    mask_video = jitter.decode(args.phone_mask, resize=False)
    masks = jitter.evaluation_masks(mask_video)
    _, _, threshold = jitter.edit_maps(videos["P0"], source, masks)
    binary, geometry, appearance_rows = {}, {}, {}
    for name in METHODS:
        binary[name], weights, _ = jitter.edit_maps(videos[name], source, masks, threshold=threshold)
        geometry[name] = jitter.per_frame_geometry(binary[name], weights, masks)
        appearance_rows[name] = appearance.frame_metrics(videos[name], masks)
    rows = frame_rows(videos, masks, appearance_rows, binary, geometry)
    summary = {
        "offline_phone_mask_only": True,
        "common_edit_support_threshold_from": "P0",
        "common_edit_support_threshold": threshold,
        "methods": {name: method_summary(rows, name) for name in METHODS},
        "blocks": block_summaries(rows),
        "P1_luma_gain_vs_cool_gain_spearman": {
            "r": float(stats.spearmanr(
                [row["P1_minus_P0_brown_luma"] for row in rows],
                [row["P1_minus_P0_cool_fraction"] for row in rows],
            ).statistic),
            "p": float(stats.spearmanr(
                [row["P1_minus_P0_brown_luma"] for row in rows],
                [row["P1_minus_P0_cool_fraction"] for row in rows],
            ).pvalue),
        },
    }
    with (args.output_dir / "P0_P1_framewise.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "P0_P1_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    all_frames(videos, masks, args.output_dir / "P0_P1_all_81_frames.jpg")
    plot(rows, args.output_dir / "P0_P1_metrics.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
