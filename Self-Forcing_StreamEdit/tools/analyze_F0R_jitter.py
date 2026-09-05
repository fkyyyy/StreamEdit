#!/usr/bin/env python3
"""Offline jitter audit for F0-R versus the local StreamGVE baseline.

The phone mask is evaluation-only.  It is time-resampled exactly like the
inference mask loader (linspace over the complete mask video), but is never
passed into generation.  The audit separates three effects that can all look
like "shaking":

* binary edit-support motion (geometry / silhouette),
* weighted edit-energy motion (appearance redistribution), and
* motion-compensated temporal change (texture / illumination flicker).

It also compares those output signals with the automatically inferred region
maps saved by F0-R at the 21 causal latent time groups.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage, stats


WIDTH = 832
HEIGHT = 480
FRAME_COUNT = 81
BOUNDARIES = (9, 21, 33, 45, 57, 69)
LATENT_BOUNDARIES = (3, 6, 9, 12, 15, 18)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--phone-mask", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--f0r", type=Path, required=True)
    parser.add_argument("--roles-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def decode(path: Path, *, resize: bool = True) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")
    command = ["ffmpeg", "-v", "error", "-i", str(path)]
    if resize:
        # This is the spatial transform used by inference_edit_streamedit.py.
        command.extend(["-vf", f"scale={WIDTH}:{HEIGHT}"])
        width, height = WIDTH, HEIGHT
    else:
        probe = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x", str(path),
        ], text=True).strip()
        width, height = (int(value) for value in probe.split("x"))
    command.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    raw = subprocess.check_output(command)
    frame_bytes = width * height * 3
    if len(raw) % frame_bytes:
        raise RuntimeError(f"Unexpected decoded byte count for {path}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)


def central_white_component(frame: np.ndarray) -> np.ndarray:
    """Recover the painted phone component while rejecting scene whites."""
    height, width = frame.shape[:2]
    white = np.all(frame >= 245, axis=-1)
    allowed = np.zeros_like(white)
    allowed[int(0.25 * height):, int(0.20 * width):int(0.78 * width)] = True
    labels, count = ndimage.label(white & allowed)
    if count == 0:
        raise RuntimeError("No white phone component found")
    yy, xx = np.indices(white.shape)
    best_label, best_score = 0, -1.0
    for label in range(1, count + 1):
        component = labels == label
        area = int(component.sum())
        if area < 300:
            continue
        cx = float(xx[component].mean()) / width
        cy = float(yy[component].mean()) / height
        score = area / (1.0 + 8.0 * abs(cx - 0.50) + 3.0 * abs(cy - 0.68))
        if score > best_score:
            best_label, best_score = label, score
    if best_label == 0:
        raise RuntimeError("No sufficiently large phone component found")
    return labels == best_label


def evaluation_masks(mask_video: np.ndarray) -> np.ndarray:
    components = [central_white_component(frame) for frame in mask_video]
    indices = np.rint(np.linspace(0, len(components) - 1, FRAME_COUNT)).astype(int)
    mapped = []
    for index in indices:
        image = Image.fromarray(components[int(index)].astype(np.uint8) * 255)
        value = np.asarray(
            image.resize((WIDTH, HEIGHT), Image.Resampling.NEAREST)
        ) > 0
        mapped.append(ndimage.binary_closing(value, iterations=2))
    return np.stack(mapped)


def centroid(weight: np.ndarray) -> tuple[float, float]:
    total = float(weight.sum())
    if total <= 1e-8:
        return float("nan"), float("nan")
    yy, xx = np.indices(weight.shape)
    return (
        float((weight * xx).sum() / total),
        float((weight * yy).sum() / total),
    )


def primary_component(mask: np.ndarray, reference: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros_like(mask)
    overlap = ndimage.sum(reference, labels, range(1, count + 1))
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    score = np.asarray(overlap) * 10.0 + np.asarray(sizes)
    return labels == int(np.argmax(score)) + 1


def angle_delta_degrees(left: float, right: float) -> float:
    # Principal-axis orientation is periodic over 180 degrees.
    return abs((right - left + 90.0) % 180.0 - 90.0)


def weighted_moments(weight: np.ndarray) -> dict[str, float]:
    cx, cy = centroid(weight)
    total = float(weight.sum())
    if total <= 1e-8 or not np.isfinite(cx + cy):
        return {
            "x": cx, "y": cy, "major": float("nan"),
            "minor": float("nan"), "angle": float("nan"),
        }
    yy, xx = np.indices(weight.shape)
    dx, dy = xx - cx, yy - cy
    covariance = np.asarray([
        [(weight * dx * dx).sum(), (weight * dx * dy).sum()],
        [(weight * dx * dy).sum(), (weight * dy * dy).sum()],
    ], dtype=np.float64) / total
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    return {
        "x": cx,
        "y": cy,
        "major": float(math.sqrt(max(values[0], 0.0))),
        "minor": float(math.sqrt(max(values[1], 0.0))),
        "angle": float(math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))),
    }


def edit_maps(
    video: np.ndarray,
    source: np.ndarray,
    gt: np.ndarray,
    *,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    roi = np.stack([ndimage.binary_dilation(frame, iterations=10) for frame in gt])
    raw = np.abs(video.astype(np.float32) - source.astype(np.float32)).mean(axis=-1) / 255.0
    smooth = np.stack([ndimage.gaussian_filter(frame, sigma=1.4) for frame in raw])
    values = np.concatenate([smooth[index][roi[index]] for index in range(FRAME_COUNT)])
    # Use a low threshold for the hard map so it follows the complete edited
    # silhouette.  A mid/high quantile instead follows changing highlights
    # inside the wallet and makes appearance flicker look like a 90-degree
    # geometry rotation.
    if threshold is None:
        threshold = float(max(0.035, np.quantile(values, 0.15)))
    else:
        threshold = float(threshold)
    binary, weights = [], []
    for index in range(FRAME_COUNT):
        candidate = (smooth[index] >= threshold) & roi[index]
        candidate = ndimage.binary_closing(candidate, iterations=4)
        candidate = ndimage.binary_opening(candidate, iterations=1)
        candidate = primary_component(candidate, gt[index])
        candidate = ndimage.binary_fill_holes(candidate)
        # Retain a broad, clipped appearance-energy map.  Clipping prevents a
        # single bright source leak from dominating the centroid.
        floor = float(np.quantile(smooth[index][roi[index]], 0.25))
        weight = np.maximum(smooth[index] - floor, 0.0) * roi[index]
        positive = weight[weight > 0]
        if len(positive):
            weight = np.minimum(weight, float(np.quantile(positive, 0.90)))
        binary.append(candidate)
        weights.append(weight)
    return np.stack(binary), np.stack(weights), threshold


def gray(video: np.ndarray) -> np.ndarray:
    rgb = video.astype(np.float32) / 255.0
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def edge_magnitude(value: np.ndarray) -> np.ndarray:
    dx = ndimage.sobel(value, axis=1) / 8.0
    dy = ndimage.sobel(value, axis=0) / 8.0
    return np.hypot(dx, dy)


def per_frame_geometry(binary: np.ndarray, weights: np.ndarray, gt: np.ndarray) -> list[dict[str, float]]:
    result = []
    for binary_frame, weight, gt_frame in zip(binary, weights, gt):
        gx, gy = centroid(gt_frame.astype(np.float32))
        hard = weighted_moments(binary_frame.astype(np.float32))
        soft = weighted_moments(weight)
        result.append({
            "gt_x": gx, "gt_y": gy,
            "hard_x": hard["x"], "hard_y": hard["y"],
            "hard_offset_x": hard["x"] - gx,
            "hard_offset_y": hard["y"] - gy,
            "hard_area": float(binary_frame.sum()),
            "hard_major": hard["major"], "hard_minor": hard["minor"],
            "hard_angle": hard["angle"],
            "soft_x": soft["x"], "soft_y": soft["y"],
            "soft_offset_x": soft["x"] - gx,
            "soft_offset_y": soft["y"] - gy,
            "soft_major": soft["major"], "soft_minor": soft["minor"],
            "soft_angle": soft["angle"],
        })
    return result


def transition_metrics(
    videos: dict[str, np.ndarray],
    binary: dict[str, np.ndarray],
    geometry: dict[str, list[dict[str, float]]],
    gt: np.ndarray,
) -> list[dict[str, float | int]]:
    grayscale = {name: gray(video) for name, video in videos.items()}
    edges = {name: np.stack([edge_magnitude(frame) for frame in value]) for name, value in grayscale.items()}
    rows: list[dict[str, float | int]] = []
    for frame in range(1, FRAME_COUNT):
        previous_gt = geometry["L0"][frame - 1]
        current_gt = geometry["L0"][frame]
        dx = current_gt["gt_x"] - previous_gt["gt_x"]
        dy = current_gt["gt_y"] - previous_gt["gt_y"]
        support = ndimage.binary_dilation(gt[frame], iterations=12)
        row: dict[str, float | int] = {
            "target_frame": frame,
            "is_block_boundary": int(frame in BOUNDARIES),
            "gt_center_step_px": float(math.hypot(dx, dy)),
        }
        for name in ("source", "L0", "F0R"):
            previous_aligned = ndimage.shift(
                grayscale[name][frame - 1], (dy, dx), order=1, mode="nearest"
            )
            previous_edge_aligned = ndimage.shift(
                edges[name][frame - 1], (dy, dx), order=1, mode="nearest"
            )
            row[f"{name}_aligned_l1"] = float(
                np.abs(grayscale[name][frame] - previous_aligned)[support].mean()
            )
            row[f"{name}_aligned_edge_l1"] = float(
                np.abs(edges[name][frame] - previous_edge_aligned)[support].mean()
            )
        for name in ("L0", "F0R"):
            prev, curr = geometry[name][frame - 1], geometry[name][frame]
            hard_dx = curr["hard_offset_x"] - prev["hard_offset_x"]
            hard_dy = curr["hard_offset_y"] - prev["hard_offset_y"]
            soft_dx = curr["soft_offset_x"] - prev["soft_offset_x"]
            soft_dy = curr["soft_offset_y"] - prev["soft_offset_y"]
            union = binary[name][frame - 1] | binary[name][frame]
            intersection = binary[name][frame - 1] & binary[name][frame]
            row.update({
                f"{name}_hard_relative_dx": float(hard_dx),
                f"{name}_hard_relative_dy": float(hard_dy),
                f"{name}_hard_relative_step_px": float(math.hypot(hard_dx, hard_dy)),
                f"{name}_soft_relative_step_px": float(math.hypot(soft_dx, soft_dy)),
                f"{name}_binary_temporal_iou": float(intersection.sum() / max(union.sum(), 1)),
                f"{name}_area_log_step": float(abs(math.log(max(curr["hard_area"], 1.0) / max(prev["hard_area"], 1.0)))),
                f"{name}_angle_step_deg": float(angle_delta_degrees(prev["hard_angle"], curr["hard_angle"])),
                f"{name}_aligned_l1_excess_source": float(row[f"{name}_aligned_l1"] - row["source_aligned_l1"]),
                f"{name}_aligned_edge_l1_excess_source": float(row[f"{name}_aligned_edge_l1"] - row["source_aligned_edge_l1"]),
            })
        rows.append(row)
    return rows


def safe_mean(values: np.ndarray) -> float:
    return float(np.nanmean(values))


def method_summary(rows: list[dict[str, float | int]], name: str) -> dict[str, object]:
    boundary = np.asarray([bool(row["is_block_boundary"]) for row in rows])
    result: dict[str, object] = {}
    metrics = (
        "hard_relative_step_px",
        "soft_relative_step_px",
        "binary_temporal_iou",
        "area_log_step",
        "angle_step_deg",
        "aligned_l1_excess_source",
        "aligned_edge_l1_excess_source",
    )
    for metric in metrics:
        values = np.asarray([float(row[f"{name}_{metric}"]) for row in rows])
        boundary_mean = safe_mean(values[boundary])
        ordinary_mean = safe_mean(values[~boundary])
        result[metric] = {
            "all_mean": safe_mean(values),
            "all_median": float(np.nanmedian(values)),
            "p90": float(np.nanpercentile(values, 90)),
            "boundary_mean": boundary_mean,
            "nonboundary_mean": ordinary_mean,
            "boundary_to_nonboundary_ratio": float(boundary_mean / max(abs(ordinary_mean), 1e-8)),
        }
    hard = np.asarray([float(row[f"{name}_hard_relative_step_px"]) for row in rows])
    soft = np.asarray([float(row[f"{name}_soft_relative_step_px"]) for row in rows])
    aligned = np.asarray([float(row[f"{name}_aligned_l1_excess_source"]) for row in rows])
    result["largest_hard_motion_frames"] = [int(index + 1) for index in np.argsort(hard)[-8:][::-1]]
    result["largest_soft_motion_frames"] = [int(index + 1) for index in np.argsort(soft)[-8:][::-1]]
    result["largest_aligned_change_frames"] = [int(index + 1) for index in np.argsort(aligned)[-8:][::-1]]
    local_boundary = {}
    for boundary_frame in BOUNDARIES:
        index = boundary_frame - 1
        neighbors = [value for value in (index - 2, index - 1, index + 1, index + 2) if 0 <= value < len(rows)]
        local_boundary[str(boundary_frame)] = {
            "hard_step_px": float(hard[index]),
            "hard_minus_local_mean_px": float(hard[index] - hard[neighbors].mean()),
            "soft_step_px": float(soft[index]),
            "aligned_change_excess": float(aligned[index]),
        }
    result["boundaries"] = local_boundary
    return result


def causal_groups() -> list[tuple[int, int]]:
    groups = [(0, 1)]
    groups.extend((1 + 4 * (index - 1), 1 + 4 * index) for index in range(1, 21))
    return groups


def load_debug(roles_dir: Path) -> dict[str, np.ndarray]:
    keys = (
        "source_flow_verified_support",
        "source_flow_verified_posterior",
        "source_flow_region_proposal",
        "causal_owner_weight",
    )
    parts = {key: [] for key in keys}
    paths = sorted(roles_dir.glob("block_*_hand_role_debug.npz"))
    if len(paths) != 7:
        raise RuntimeError(f"Expected 7 debug blocks, found {len(paths)}")
    for path in paths:
        with np.load(path) as data:
            for key in keys:
                parts[key].append(data[key][0].astype(np.float32))
    return {key: np.concatenate(value, axis=0) for key, value in parts.items()}


def resize_binary(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    image = Image.fromarray(value.astype(np.uint8) * 255)
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST)) > 0


def region_metrics(
    debug: dict[str, np.ndarray],
    gt: np.ndarray,
    geometry: list[dict[str, float]],
) -> tuple[list[dict[str, float | int]], dict[str, object]]:
    groups = causal_groups()
    rows: list[dict[str, float | int]] = []
    previous: dict[str, tuple[float, float]] = {}
    for latent, (left, right) in enumerate(groups):
        shape = debug["source_flow_verified_support"][latent].shape
        gt_group = gt[left:right].any(axis=0)
        gt_token = resize_binary(gt_group, shape)
        gx, gy = centroid(gt_token.astype(np.float32))
        row: dict[str, float | int] = {
            "latent_index": latent,
            "rgb_start": left,
            "rgb_end": right - 1,
            "is_block_start": int(latent in LATENT_BOUNDARIES),
        }
        maps = {
            "verified": debug["source_flow_verified_support"][latent] > 0.5,
            "posterior": debug["source_flow_verified_posterior"][latent],
            "owner": debug["causal_owner_weight"][latent],
            "proposal": debug["source_flow_region_proposal"][latent] > 0.5,
        }
        for name, value in maps.items():
            cx, cy = centroid(value.astype(np.float32))
            offset = (cx - gx, cy - gy)
            row[f"{name}_coverage"] = float(np.mean(value))
            row[f"{name}_offset_x_token"] = float(offset[0])
            row[f"{name}_offset_y_token"] = float(offset[1])
            if name in previous:
                row[f"{name}_relative_step_token_px"] = float(math.hypot(
                    offset[0] - previous[name][0], offset[1] - previous[name][1]
                ))
            else:
                row[f"{name}_relative_step_token_px"] = 0.0
            previous[name] = offset
        verified = maps["verified"]
        row["verified_gt_iou"] = float(
            (verified & gt_token).sum() / max((verified | gt_token).sum(), 1)
        )
        # Average the output edit offset within the exact RGB group represented
        # by this causal latent. Convert pixels to 30x52 token coordinates.
        out_x = float(np.mean([geometry[index]["hard_offset_x"] for index in range(left, right)])) / 16.0
        out_y = float(np.mean([geometry[index]["hard_offset_y"] for index in range(left, right)])) / 16.0
        row["output_offset_x_token"] = out_x
        row["output_offset_y_token"] = out_y
        if rows:
            row["output_relative_step_token_px"] = float(math.hypot(
                out_x - float(rows[-1]["output_offset_x_token"]),
                out_y - float(rows[-1]["output_offset_y_token"]),
            ))
        else:
            row["output_relative_step_token_px"] = 0.0
        rows.append(row)

    transitions = rows[1:]
    block_start = np.asarray([bool(row["is_block_start"]) for row in transitions])
    output_step = np.asarray([float(row["output_relative_step_token_px"]) for row in transitions])
    summary: dict[str, object] = {
        "verified_gt_iou_mean": safe_mean(np.asarray([float(row["verified_gt_iou"]) for row in rows])),
        "verified_gt_iou_min": float(min(float(row["verified_gt_iou"]) for row in rows)),
        "output_group_step_block_start_mean_token_px": safe_mean(output_step[block_start]),
        "output_group_step_nonblock_mean_token_px": safe_mean(output_step[~block_start]),
    }
    for name in ("verified", "posterior", "owner", "proposal"):
        region_step = np.asarray([float(row[f"{name}_relative_step_token_px"]) for row in transitions])
        pearson = stats.pearsonr(region_step, output_step)
        spearman = stats.spearmanr(region_step, output_step)
        summary[name] = {
            "step_mean_token_px": safe_mean(region_step),
            "step_block_start_mean_token_px": safe_mean(region_step[block_start]),
            "step_nonblock_mean_token_px": safe_mean(region_step[~block_start]),
            "step_block_to_nonblock_ratio": float(
                safe_mean(region_step[block_start]) / max(safe_mean(region_step[~block_start]), 1e-8)
            ),
            "output_step_pearson_r": float(pearson.statistic),
            "output_step_pearson_p": float(pearson.pvalue),
            "output_step_spearman_r": float(spearman.statistic),
            "output_step_spearman_p": float(spearman.pvalue),
            "largest_step_latents": [int(index + 1) for index in np.argsort(region_step)[-5:][::-1]],
        }
    return rows, summary


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def crop_box(gt: np.ndarray, margin: int = 45) -> tuple[int, int, int, int]:
    occupancy = gt.mean(axis=0) >= 0.04
    ys, xs = np.nonzero(occupancy)
    return (
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(WIDTH, int(xs.max()) + margin + 1),
        min(HEIGHT, int(ys.max()) + margin + 1),
    )


def annotated_crop(
    frame: np.ndarray,
    gt_frame: np.ndarray,
    edit_binary: np.ndarray | None,
    geom: dict[str, float] | None,
    crop: tuple[int, int, int, int],
    label: str,
) -> Image.Image:
    x0, y0, x1, y1 = crop
    image = Image.fromarray(frame[y0:y1, x0:x1]).resize((220, 220), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    sx, sy = 220.0 / (x1 - x0), 220.0 / (y1 - y0)
    # Draw GT outline in green.
    outline = gt_frame ^ ndimage.binary_erosion(gt_frame, iterations=2)
    yy, xx = np.nonzero(outline[y0:y1, x0:x1])
    for x, y in zip(xx[::2], yy[::2]):
        draw.point((int(x * sx), int(y * sy)), fill=(0, 255, 80))
    if edit_binary is not None:
        edit_outline = edit_binary ^ ndimage.binary_erosion(edit_binary, iterations=2)
        yy, xx = np.nonzero(edit_outline[y0:y1, x0:x1])
        for x, y in zip(xx[::2], yy[::2]):
            draw.point((int(x * sx), int(y * sy)), fill=(255, 180, 0))
    if geom is not None:
        gx, gy = geom["gt_x"], geom["gt_y"]
        ex, ey = geom["hard_x"], geom["hard_y"]
        points = [((gx - x0) * sx, (gy - y0) * sy), ((ex - x0) * sx, (ey - y0) * sy)]
        draw.line(points, fill=(255, 0, 0), width=3)
        for px, py in points:
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(255, 0, 0))
    draw.rectangle((0, 0, 219, 219), outline=(30, 30, 30), width=1)
    draw.rectangle((3, 3, 3 + 8 * len(label), 24), fill=(0, 0, 0))
    draw.text((7, 5), label, fill="white", font=font(14))
    return image


def make_peak_sheet(
    videos: dict[str, np.ndarray],
    gt: np.ndarray,
    binary: dict[str, np.ndarray],
    geometry: dict[str, list[dict[str, float]]],
    rows: list[dict[str, float | int]],
    output: Path,
) -> list[int]:
    values = np.asarray([float(row["F0R_hard_relative_step_px"]) for row in rows])
    peaks: list[int] = []
    for index in np.argsort(values)[::-1]:
        frame = int(index + 1)
        if frame < 2 or frame > 78:
            continue
        if all(abs(frame - selected) > 4 for selected in peaks):
            peaks.append(frame)
        if len(peaks) == 4:
            break
    crop = crop_box(gt)
    panel_width = 116 + 5 * 220
    panel_height = 44 + 3 * 220
    canvas = Image.new("RGB", (panel_width, panel_height * len(peaks)), "white")
    for panel_index, peak in enumerate(peaks):
        top = panel_index * panel_height
        draw = ImageDraw.Draw(canvas)
        draw.text((8, top + 8), f"F0-R geometry peak entering F{peak:02d}", fill="black", font=font(18))
        frames = list(range(peak - 2, peak + 3))
        for column, frame in enumerate(frames):
            draw.text((116 + column * 220 + 5, top + 25), f"F{frame:02d}", fill="black", font=font(13))
        for row_index, name in enumerate(("source", "L0", "F0R")):
            draw.text((8, top + 44 + row_index * 220 + 8), name, fill="black", font=font(16))
            for column, frame in enumerate(frames):
                tile = annotated_crop(
                    videos[name][frame], gt[frame],
                    None if name == "source" else binary[name][frame],
                    None if name == "source" else geometry[name][frame],
                    crop, f"F{frame:02d}",
                )
                if frame == peak:
                    ImageDraw.Draw(tile).rectangle((2, 2, 217, 217), outline=(220, 0, 0), width=5)
                canvas.paste(tile, (116 + column * 220, top + 44 + row_index * 220))
    canvas.save(output, quality=95)
    return peaks


def make_boundary_sheet(
    videos: dict[str, np.ndarray],
    gt: np.ndarray,
    output: Path,
) -> None:
    crop = crop_box(gt)
    cell_w, cell_h = 154, 154
    left, top = 105, 38
    canvas = Image.new("RGB", (left + 5 * cell_w, (top + 2 * cell_h) * len(BOUNDARIES)), "white")
    draw = ImageDraw.Draw(canvas)
    for block_row, boundary in enumerate(BOUNDARIES):
        panel_y = block_row * (top + 2 * cell_h)
        draw.text((5, panel_y + 7), f"enter F{boundary}", fill="black", font=font(16))
        frames = range(boundary - 2, boundary + 3)
        for column, frame in enumerate(frames):
            draw.text((left + column * cell_w + 4, panel_y + 20), f"F{frame:02d}", fill="black", font=font(12))
        for row_index, name in enumerate(("L0", "F0R")):
            draw.text((5, panel_y + top + row_index * cell_h + 8), name, fill="black", font=font(15))
            for column, frame in enumerate(frames):
                x0, y0, x1, y1 = crop
                tile = Image.fromarray(videos[name][frame, y0:y1, x0:x1]).resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                if frame == boundary:
                    ImageDraw.Draw(tile).rectangle((2, 2, cell_w - 3, cell_h - 3), outline=(220, 0, 0), width=4)
                canvas.paste(tile, (left + column * cell_w, panel_y + top + row_index * cell_h))
    canvas.save(output, quality=95)


def plot_timelines(
    rows: list[dict[str, float | int]],
    region_rows: list[dict[str, float | int]],
    output: Path,
) -> None:
    frames = np.asarray([int(row["target_frame"]) for row in rows])
    figure, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
    colors = {"L0": "#2ca02c", "F0R": "#d62728"}
    for name in ("L0", "F0R"):
        axes[0].plot(frames, [row[f"{name}_hard_relative_step_px"] for row in rows], label=f"{name} binary support", color=colors[name])
        axes[0].plot(frames, [row[f"{name}_soft_relative_step_px"] for row in rows], label=f"{name} weighted appearance", color=colors[name], alpha=0.42)
        axes[1].plot(frames, [row[f"{name}_aligned_l1_excess_source"] for row in rows], label=name, color=colors[name])
        axes[2].plot(frames, [row[f"{name}_angle_step_deg"] for row in rows], label=f"{name} angle", color=colors[name])
    for axis in axes[:3]:
        for boundary in BOUNDARIES:
            axis.axvline(boundary, color="black", alpha=0.22, linestyle="--")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")
    axes[0].set_ylabel("GT-relative center step (px)")
    axes[0].set_title("Geometry versus appearance-center motion")
    axes[1].set_ylabel("Motion-compensated L1\nminus source")
    axes[1].set_title("Texture / illumination change after phone-trajectory compensation")
    axes[2].set_ylabel("binary support angle step (deg)")
    axes[2].set_xlabel("RGB target frame; dashed = first frame of a new causal block")

    latent = np.asarray([int(row["latent_index"]) for row in region_rows[1:]])
    axes[3].plot(latent, [row["output_relative_step_token_px"] for row in region_rows[1:]], label="F0R output edit support", color="#d62728", linewidth=2)
    axes[3].plot(latent, [row["verified_relative_step_token_px"] for row in region_rows[1:]], label="verified hard region", color="#1f77b4")
    axes[3].plot(latent, [row["owner_relative_step_token_px"] for row in region_rows[1:]], label="soft causal owner", color="#9467bd")
    for boundary in LATENT_BOUNDARIES:
        axes[3].axvline(boundary, color="black", alpha=0.22, linestyle="--")
    axes[3].grid(alpha=0.2)
    axes[3].legend(loc="upper right")
    axes[3].set_ylabel("GT-relative step (token px)")
    axes[3].set_xlabel("causal latent index; dashed = block start")
    axes[3].set_title("Automatic-region motion versus grouped output motion")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos = {
        "source": decode(args.source),
        "L0": decode(args.baseline),
        "F0R": decode(args.f0r),
    }
    lengths = {name: len(video) for name, video in videos.items()}
    if lengths != {"source": FRAME_COUNT, "L0": FRAME_COUNT, "F0R": FRAME_COUNT}:
        raise RuntimeError(f"Expected 81 frames, got {lengths}")
    mask_video = decode(args.phone_mask, resize=False)
    gt = evaluation_masks(mask_video)

    binary: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    geometry: dict[str, list[dict[str, float]]] = {}
    for name in ("L0", "F0R"):
        binary[name], weights[name], thresholds[name] = edit_maps(videos[name], videos["source"], gt)
        geometry[name] = per_frame_geometry(binary[name], weights[name], gt)

    transition_rows = transition_metrics(videos, binary, geometry, gt)
    debug = load_debug(args.roles_dir)
    region_rows, region_summary = region_metrics(debug, gt, geometry["F0R"])
    peak_frames = make_peak_sheet(
        videos, gt, binary, geometry, transition_rows,
        args.output_dir / "F0R_largest_jitter_windows.jpg",
    )
    make_boundary_sheet(videos, gt, args.output_dir / "F0R_block_boundary_windows.jpg")
    plot_timelines(transition_rows, region_rows, args.output_dir / "F0R_jitter_region_timeline.png")
    write_csv(args.output_dir / "F0R_jitter_metrics.csv", transition_rows)
    write_csv(args.output_dir / "F0R_region_motion_metrics.csv", region_rows)

    summary = {
        "inputs": {
            "source_frames": len(videos["source"]),
            "mask_source_frames": len(mask_video),
            "mask_resampling": "linspace complete mask video to 81 frames",
            "spatial_transform": f"direct resize to {WIDTH}x{HEIGHT}, matching inference",
            "phone_mask_usage": "offline evaluation only",
        },
        "edit_support_thresholds": thresholds,
        "F0R_peak_geometry_frames": peak_frames,
        "methods": {
            "L0": method_summary(transition_rows, "L0"),
            "F0R": method_summary(transition_rows, "F0R"),
        },
        "automatic_region": region_summary,
    }
    with (args.output_dir / "F0R_jitter_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
