#!/usr/bin/env python3
"""Offline M1/L0/StreamGVE appearance and motion comparison.

The phone mask is evaluation-only. It is never passed to inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from visualize_965a_wallet_streamgve import (
    BOUNDARIES,
    classify,
    crop_box,
    decode,
    evaluation_masks,
)


METHODS = ("L0", "965a", "M1")
COLORS = {"L0": "#4d4d4d", "965a": "#2ca02c", "M1": "#d62728"}
PERIODS = {"early": (0, 20), "middle": (32, 52), "late": (60, 80)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1", type=Path, required=True)
    parser.add_argument("--l0", type=Path, required=True)
    parser.add_argument("--streamgve", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--phone-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def centroid(weight: np.ndarray) -> tuple[float, float]:
    total = float(weight.sum())
    if total <= 1e-8:
        return float("nan"), float("nan")
    yy, xx = np.indices(weight.shape)
    return float((weight * xx).sum() / total), float((weight * yy).sum() / total)


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def appearance(video: np.ndarray, masks: np.ndarray) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {
        "brown_luma": [],
        "brown_fraction": [],
        "cool_fraction": [],
        "object_luma": [],
    }
    for frame, support in zip(video, masks):
        luma, brown, cool = classify(frame, support)
        values["brown_luma"].append(float(np.median(luma[brown])))
        values["brown_fraction"].append(float(brown.sum() / support.sum()))
        values["cool_fraction"].append(float(cool.sum() / support.sum()))
        values["object_luma"].append(float(np.median(luma[support])))
    return {key: np.asarray(value) for key, value in values.items()}


def motion(
    video: np.ndarray, source: np.ndarray, masks: np.ndarray, roi: np.ndarray
) -> dict[str, np.ndarray]:
    frames = video.astype(np.float32) / 255.0
    src = source.astype(np.float32) / 255.0
    gt_xy = np.asarray([centroid(mask.astype(np.float32)) for mask in masks])
    diff_xy: list[tuple[float, float]] = []
    brown_xy: list[tuple[float, float]] = []
    temporal = np.zeros(len(video), dtype=np.float64)

    for index, (frame, src_frame, support, region) in enumerate(zip(frames, src, masks, roi)):
        difference = np.abs(frame - src_frame).mean(axis=-1)
        floor = float(np.quantile(difference[region], 0.35))
        weight = np.maximum(difference - floor, 0.0) * region
        diff_xy.append(centroid(weight))

        _, brown, _ = classify(video[index], support)
        brown = ndimage.binary_closing(brown, iterations=2)
        brown_xy.append(centroid(brown.astype(np.float32)))

        if index:
            union = roi[index - 1] | roi[index]
            delta = np.abs(frames[index] - frames[index - 1]).mean(axis=-1)
            temporal[index] = float(delta[union].mean())

    diff_xy_a = np.asarray(diff_xy)
    brown_xy_a = np.asarray(brown_xy)
    diff_offset = diff_xy_a - gt_xy
    brown_offset = brown_xy_a - gt_xy

    def steps(offset: np.ndarray) -> np.ndarray:
        result = np.zeros(len(offset), dtype=np.float64)
        result[1:] = np.linalg.norm(np.diff(offset, axis=0), axis=-1)
        return result

    def x_acceleration(offset: np.ndarray) -> np.ndarray:
        result = np.zeros(len(offset), dtype=np.float64)
        result[2:] = np.abs(np.diff(offset[:, 0], n=2))
        return result

    return {
        "temporal_l1": temporal,
        "diff_offset_x": diff_offset[:, 0],
        "diff_offset_y": diff_offset[:, 1],
        "diff_offset_step": steps(diff_offset),
        "diff_offset_x_accel": x_acceleration(diff_offset),
        "brown_offset_x": brown_offset[:, 0],
        "brown_offset_y": brown_offset[:, 1],
        "brown_offset_step": steps(brown_offset),
        "brown_offset_x_accel": x_acceleration(brown_offset),
    }


def pairwise(
    reference: np.ndarray, candidate: np.ndarray, roi: np.ndarray, masks: np.ndarray
) -> dict[str, np.ndarray]:
    ref = reference.astype(np.float32)
    cand = candidate.astype(np.float32)
    delta = np.abs(ref - cand).mean(axis=-1)
    squared = np.square(ref - cand).mean(axis=-1)
    roi_mae, full_mae, psnr = [], [], []
    gt_xy = np.asarray([centroid(mask.astype(np.float32)) for mask in masks])
    delta_xy: list[tuple[float, float]] = []
    for index in range(len(ref)):
        roi_mae.append(float(delta[index][roi[index]].mean()))
        full_mae.append(float(delta[index].mean()))
        mse = float(squared[index].mean())
        psnr.append(float(20.0 * math.log10(255.0 / math.sqrt(max(mse, 1e-12)))))
        floor = float(np.quantile(delta[index][roi[index]], 0.35))
        weight = np.maximum(delta[index] - floor, 0.0) * roi[index]
        location = centroid(weight)
        if not np.isfinite(location).all():
            # Identical frames have no pairwise residual centroid. Reuse the
            # previous location so the diagnostic does not invent a jump.
            location = delta_xy[-1] if delta_xy else tuple(gt_xy[index])
        delta_xy.append(location)
    delta_offset = np.asarray(delta_xy) - gt_xy
    delta_step = np.zeros(len(reference), dtype=np.float64)
    delta_step[1:] = np.linalg.norm(np.diff(delta_offset, axis=0), axis=-1)
    return {
        "roi_mae": np.asarray(roi_mae),
        "full_mae": np.asarray(full_mae),
        "full_psnr": np.asarray(psnr),
        "delta_offset_x": delta_offset[:, 0],
        "delta_offset_y": delta_offset[:, 1],
        "delta_offset_step": delta_step,
    }


def mean_at(values: np.ndarray, indices: list[int]) -> float:
    return float(values[np.asarray(indices, dtype=np.int64)].mean())


def summarize_motion(values: dict[str, np.ndarray], source_temporal: np.ndarray) -> dict[str, float | int]:
    boundary = list(BOUNDARIES)
    ordinary = [index for index in range(1, 81) if index not in BOUNDARIES]
    temporal = values["temporal_l1"]
    excess = temporal - source_temporal
    result: dict[str, float | int] = {
        "boundary_temporal_l1": mean_at(temporal, boundary),
        "nonboundary_temporal_l1": mean_at(temporal, ordinary),
        "boundary_excess_over_source": mean_at(excess, boundary),
        "nonboundary_excess_over_source": mean_at(excess, ordinary),
        "boundary_diff_centroid_step_px": mean_at(values["diff_offset_step"], boundary),
        "nonboundary_diff_centroid_step_px": mean_at(values["diff_offset_step"], ordinary),
        "boundary_brown_centroid_step_px": mean_at(values["brown_offset_step"], boundary),
        "nonboundary_brown_centroid_step_px": mean_at(values["brown_offset_step"], ordinary),
        "diff_x_accel_mean": float(values["diff_offset_x_accel"][2:].mean()),
        "brown_x_accel_mean": float(values["brown_offset_x_accel"][2:].mean()),
    }
    for key in ("diff_offset_step", "brown_offset_step", "diff_offset_x_accel", "brown_offset_x_accel"):
        ranked = np.argsort(values[key])[::-1][:8]
        result[f"top_{key}_frames"] = [int(index) for index in ranked]
    return result


def tile(frame: np.ndarray, label: str, size: tuple[int, int], border: str | None = None) -> Image.Image:
    image = Image.fromarray(frame).resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=border or "#555555", width=4 if border else 1)
    box = draw.textbbox((0, 0), label, font=font(14))
    draw.rectangle((3, 3, box[2] + 9, box[3] + 8), fill="black")
    draw.text((6, 4), label, fill="white", font=font(14))
    return image


def boundary_grid(videos: dict[str, np.ndarray], crop: tuple[int, int, int, int], output: Path) -> None:
    x0, y0, x1, y1 = crop
    size = (145, 160)
    left, top = 92, 46
    columns = len(METHODS) * 3
    canvas = Image.new("RGB", (left + columns * size[0], top + len(BOUNDARIES) * size[1]), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Same-frame comparison around block entries; red = first new-block frame", fill="black", font=font(19))
    for row, boundary in enumerate(BOUNDARIES):
        draw.text((8, top + row * size[1] + 6), f"F{boundary:02d}", fill="black", font=font(16))
        column = 0
        for method in METHODS:
            for index in range(boundary - 1, boundary + 2):
                crop_frame = videos[method][index, y0:y1, x0:x1]
                canvas.paste(
                    tile(crop_frame, f"{method} F{index:02d}", size, "#d62728" if index == boundary else None),
                    (left + column * size[0], top + row * size[1]),
                )
                column += 1
    canvas.save(output, quality=95)


def difference_grid(videos: dict[str, np.ndarray], crop: tuple[int, int, int, int], output: Path) -> None:
    x0, y0, x1, y1 = crop
    frames = (0, 16, 22, 32, 33, 44, 45, 56, 57, 68, 69, 80)
    rows = ("M1", "L0", "965a", "M1-L0 x6", "M1-965a x6")
    size = (145, 160)
    left, top = 120, 50
    canvas = Image.new("RGB", (left + len(frames) * size[0], top + len(rows) * size[1]), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "M1 against L0 and 965a; amplified absolute RGB difference", fill="black", font=font(19))
    for column, index in enumerate(frames):
        for row, name in enumerate(rows):
            if name in videos:
                frame = videos[name][index, y0:y1, x0:x1]
            elif name == "M1-L0 x6":
                frame = np.clip(np.abs(videos["M1"][index].astype(np.int16) - videos["L0"][index].astype(np.int16)) * 6, 0, 255).astype(np.uint8)[y0:y1, x0:x1]
            else:
                frame = np.clip(np.abs(videos["M1"][index].astype(np.int16) - videos["965a"][index].astype(np.int16)) * 6, 0, 255).astype(np.uint8)[y0:y1, x0:x1]
            canvas.paste(tile(frame, f"F{index:02d}", size), (left + column * size[0], top + row * size[1]))
        if column == 0:
            for row, name in enumerate(rows):
                draw.text((5, top + row * size[1] + 8), name, fill="black", font=font(14))
    canvas.save(output, quality=95)


def top_windows(
    videos: dict[str, np.ndarray], values: dict[str, dict[str, np.ndarray]], crop: tuple[int, int, int, int], output: Path
) -> None:
    score = values["M1"]["diff_offset_step"]
    chosen = [int(index) for index in np.argsort(score)[::-1] if index > 0][:8]
    x0, y0, x1, y1 = crop
    size = (175, 190)
    left, top = 105, 52
    canvas = Image.new("RGB", (left + len(chosen) * size[0], top + len(METHODS) * size[1]), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Largest M1 source-relative edit-centroid steps", fill="black", font=font(19))
    for row, method in enumerate(METHODS):
        draw.text((8, top + row * size[1] + 8), method, fill="black", font=font(15))
        for column, index in enumerate(chosen):
            frame = videos[method][index, y0:y1, x0:x1]
            label = f"F{index:02d} {values[method]['diff_offset_step'][index]:.1f}px"
            canvas.paste(tile(frame, label, size, "#d62728" if index in BOUNDARIES else None), (left + column * size[0], top + row * size[1]))
    canvas.save(output, quality=95)


def plot_all(
    appearance_values: dict[str, dict[str, np.ndarray]],
    motion_values: dict[str, dict[str, np.ndarray]],
    source_temporal: np.ndarray,
    pairwise_values: dict[str, dict[str, np.ndarray]],
    output: Path,
) -> None:
    frames = np.arange(81)
    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)
    for method in METHODS:
        color = COLORS[method]
        axes[0].plot(frames, appearance_values[method]["brown_luma"], label=method, color=color)
        axes[1].plot(frames, appearance_values[method]["cool_fraction"], label=method, color=color)
        axes[2].plot(frames, motion_values[method]["diff_offset_x"], label=method, color=color)
        axes[3].plot(frames, motion_values[method]["diff_offset_step"], label=method, color=color)
        axes[4].plot(frames, motion_values[method]["temporal_l1"] - source_temporal, label=method, color=color)
    axes[5].plot(frames, pairwise_values["M1-L0"]["roi_mae"], label="M1-L0 ROI MAE", color="#9467bd")
    axes[5].plot(frames, pairwise_values["M1-965a"]["roi_mae"], label="M1-965a ROI MAE", color="#ff7f0e")
    labels = (
        "brown-shell median luma",
        "cool source-like fraction",
        "edit centroid x offset vs GT (px)",
        "edit centroid source-relative step (px)",
        "ROI temporal L1 minus source",
        "same-frame ROI MAE (0-255)",
    )
    for axis, label in zip(axes, labels):
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
        for boundary in BOUNDARIES:
            axis.axvline(boundary, color="#d62728", linestyle="--", alpha=0.35)
    axes[-1].set_xlabel("RGB frame; dashed lines are causal-block entries")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos = {
        "M1": decode(args.m1, filter_graph="generated"),
        "L0": decode(args.l0, filter_graph="generated"),
        "965a": decode(args.streamgve, filter_graph="generated"),
    }
    source = decode(args.source, filter_graph="source")[:81]
    mask_video = decode(args.phone_mask)
    if any(len(video) != 81 for video in videos.values()) or len(source) != 81:
        raise RuntimeError("Expected exactly 81 frames")
    masks = evaluation_masks(mask_video, 81, (832, 480))
    roi = np.stack([ndimage.binary_dilation(mask, iterations=24) for mask in masks])
    crop = crop_box(masks)

    appearance_values = {method: appearance(video, masks) for method, video in videos.items()}
    source_motion = motion(source, source, masks, roi)
    motion_values = {method: motion(video, source, masks, roi) for method, video in videos.items()}
    pairwise_values = {
        "M1-L0": pairwise(videos["L0"], videos["M1"], roi, masks),
        "M1-965a": pairwise(videos["965a"], videos["M1"], roi, masks),
        "L0-965a": pairwise(videos["965a"], videos["L0"], roi, masks),
    }

    rows: list[dict[str, float | int]] = []
    for index in range(81):
        row: dict[str, float | int] = {"frame": index, "is_boundary": int(index in BOUNDARIES)}
        for method in METHODS:
            for key, values in appearance_values[method].items():
                row[f"{method}_{key}"] = float(values[index])
            for key, values in motion_values[method].items():
                row[f"{method}_{key}"] = float(values[index])
        for pair, metrics in pairwise_values.items():
            for key, values in metrics.items():
                row[f"{pair}_{key}"] = float(values[index])
        rows.append(row)

    with (args.output_dir / "M1_framewise_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "crop_xyxy": list(crop),
        "boundaries": list(BOUNDARIES),
        "appearance": {},
        "motion": {},
        "pairwise": {},
        "metric_caveat": "phone mask is offline-only; color masks and centroids are diagnostic proxies, not semantic segmentation",
    }
    for method in METHODS:
        method_periods: dict[str, dict[str, float | list[int]]] = {}
        for period, (start, end) in PERIODS.items():
            method_periods[period] = {
                "frames": [start, end],
                **{key: float(values[start:end + 1].mean()) for key, values in appearance_values[method].items()},
            }
        summary["appearance"][method] = method_periods  # type: ignore[index]
        summary["motion"][method] = summarize_motion(motion_values[method], source_motion["temporal_l1"])  # type: ignore[index]
    for pair, metrics in pairwise_values.items():
        summary["pairwise"][pair] = {  # type: ignore[index]
            "full_mae_mean": float(metrics["full_mae"].mean()),
            "roi_mae_mean": float(metrics["roi_mae"].mean()),
            "full_psnr_mean": float(metrics["full_psnr"].mean()),
            "roi_mae_early": float(metrics["roi_mae"][:21].mean()),
            "roi_mae_late": float(metrics["roi_mae"][60:].mean()),
            "top_roi_mae_frames": [int(index) for index in np.argsort(metrics["roi_mae"])[::-1][:10]],
            "boundary_delta_centroid_step_px": mean_at(metrics["delta_offset_step"], list(BOUNDARIES)),
            "nonboundary_delta_centroid_step_px": mean_at(metrics["delta_offset_step"], [index for index in range(1, 81) if index not in BOUNDARIES]),
            "top_delta_centroid_step_frames": [int(index) for index in np.argsort(metrics["delta_offset_step"])[::-1][:10]],
        }

    with (args.output_dir / "M1_framewise_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    boundary_grid(videos, crop, args.output_dir / "M1_vs_baselines_boundaries.jpg")
    difference_grid(videos, crop, args.output_dir / "M1_vs_baselines_differences.jpg")
    top_windows(videos, motion_values, crop, args.output_dir / "M1_top_motion_windows.jpg")
    plot_all(appearance_values, motion_values, source_motion["temporal_l1"], pairwise_values, args.output_dir / "M1_framewise_timelines.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
