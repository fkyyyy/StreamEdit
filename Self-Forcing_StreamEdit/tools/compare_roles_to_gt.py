#!/usr/bin/env python3
"""Offline GT audit for hand-inferred editable regions.

The bottle mask is read only by this post-hoc script. It is never passed to
the editing pipeline, role inferencer, ownership tracker, or KV cache.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import av
import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles_dir", type=Path, required=True)
    parser.add_argument("--gt_mask_video", type=Path, required=True)
    parser.add_argument("--source_video", type=Path, required=True)
    parser.add_argument("--hand_input_npz", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--white_threshold", type=int, default=245)
    parser.add_argument("--overlay_diff_threshold", type=float, default=24.0)
    parser.add_argument("--tiles_per_page", type=int, default=7)
    return parser.parse_args()


def decode_video(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        return [
            frame.to_ndarray(format="rgb24")
            for frame in container.decode(video=0)
        ]


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(image.resize(size, Image.Resampling.NEAREST)) > 127


def resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(image, mode="RGB").resize(
            size, Image.Resampling.BILINEAR
        )
    )


def extract_overlay_mask(
    overlay: np.ndarray,
    source: np.ndarray,
    *,
    threshold: int,
    diff_threshold: float,
) -> np.ndarray:
    source = resize_rgb(source, (overlay.shape[1], overlay.shape[0]))
    white = np.all(overlay >= threshold, axis=-1)
    difference = np.abs(
        overlay.astype(np.int16) - source.astype(np.int16)
    ).mean(axis=-1)
    return white & (difference >= diff_threshold)


def causal_groups(pixel_frames: int, latent_frames: int):
    if latent_frames == 1:
        return [(0, 1)]
    stride = (pixel_frames - 1) // (latent_frames - 1)
    if 1 + (pixel_frames - 1) // stride != latent_frames:
        raise ValueError(
            "Source and debug frames do not define the causal VAE mapping"
        )
    return [(0, 1)] + [
        (1 + stride * (i - 1), 1 + stride * i)
        for i in range(1, latent_frames)
    ]


def map_video_masks_to_source(
    mask_frames: list[np.ndarray],
    source_frames: list[np.ndarray],
    *,
    threshold: int,
    diff_threshold: float,
) -> list[np.ndarray]:
    indices = np.rint(
        np.linspace(0, len(mask_frames) - 1, len(source_frames))
    ).astype(int)
    return [
        extract_overlay_mask(
            mask_frames[int(mask_index)],
            source_frames[source_index],
            threshold=threshold,
            diff_threshold=diff_threshold,
        )
        for source_index, mask_index in enumerate(indices)
    ]


def project_gt(
    pixel_masks: list[np.ndarray],
    *,
    latent_frames: int,
    size: tuple[int, int],
) -> np.ndarray:
    groups = causal_groups(len(pixel_masks), latent_frames)
    return np.stack([
        resize_mask(
            np.maximum.reduce(pixel_masks[left:right]), size
        )
        for left, right in groups
    ])


def metric(prediction: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    prediction = prediction.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(prediction, gt).sum())
    fp = int(np.logical_and(prediction, ~gt).sum())
    fn = int(np.logical_and(~prediction, gt).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "pred_tokens": int(prediction.sum()),
        "gt_tokens": int(gt.sum()),
    }


def overlay(
    prediction: np.ndarray,
    gt: np.ndarray,
    source: np.ndarray,
) -> np.ndarray:
    source = resize_rgb(source, (prediction.shape[1], prediction.shape[0]))
    canvas = (0.30 * source).astype(np.uint8)
    true_positive = prediction & gt
    false_positive = prediction & ~gt
    false_negative = ~prediction & gt
    canvas[true_positive] = (40, 210, 70)
    canvas[false_positive] = (235, 65, 50)
    canvas[false_negative] = (45, 100, 245)
    return canvas


def add_title(image: Image.Image, title: str) -> Image.Image:
    result = Image.new("RGB", (image.width, image.height + 28), "white")
    result.paste(image, (0, 28))
    ImageDraw.Draw(result).text((6, 7), title, fill="black")
    return result


def save_token_count_plot(
    rows: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    """Draw dependency-free per-latent extent/recall diagnostics."""
    stages = list(dict.fromkeys(str(row["stage"]) for row in rows))
    latent_indices = sorted({int(row["latent_frame"]) for row in rows})
    width = max(1120, 70 * len(latent_indices))
    panel_height = 280
    margin_left, margin_right = 68, 20
    margin_top, margin_bottom = 42, 52
    canvas = Image.new(
        "RGB", (width, 2 * panel_height + 38), "white"
    )
    draw = ImageDraw.Draw(canvas)
    palette = [
        (31, 119, 180), (255, 127, 14), (44, 160, 44),
        (214, 39, 40), (148, 103, 189), (140, 86, 75),
        (227, 119, 194),
    ]
    by_stage = {
        stage: {
            int(row["latent_frame"]): row
            for row in rows if row["stage"] == stage
        }
        for stage in stages
    }
    by_latent = {
        latent_index: next(
            row for row in rows
            if int(row["latent_frame"]) == latent_index
        )
        for latent_index in latent_indices
    }
    x_left, x_right = margin_left, width - margin_right

    def x_coord(index: int) -> float:
        if len(latent_indices) == 1:
            return (x_left + x_right) / 2
        return x_left + index * (x_right - x_left) / (len(latent_indices) - 1)

    def draw_panel(top: int, key: str, label: str, maximum: float) -> None:
        plot_top = top + margin_top
        plot_bottom = top + panel_height - margin_bottom
        draw.line((x_left, plot_top, x_left, plot_bottom), fill="black", width=1)
        draw.line((x_left, plot_bottom, x_right, plot_bottom), fill="black", width=1)
        draw.text((8, top + 8), label, fill="black")
        for tick in range(5):
            value = maximum * tick / 4
            y = plot_bottom - tick * (plot_bottom - plot_top) / 4
            draw.line((x_left - 4, y, x_right, y), fill=(225, 225, 225), width=1)
            draw.text((4, y - 7), f"{value:.2f}" if maximum <= 1 else f"{value:.0f}", fill="black")
        for position, latent_index in enumerate(latent_indices):
            if position == 0:
                continue
            previous = by_latent[latent_indices[position - 1]]
            current = by_latent[latent_index]
            if int(current["block"]) != int(previous["block"]):
                x = x_coord(position)
                draw.line(
                    (x, plot_top, x, plot_bottom),
                    fill=(150, 150, 150),
                    width=1,
                )
                draw.text(
                    (x + 3, plot_top + 3),
                    f"B{int(current['block'])}",
                    fill=(90, 90, 90),
                )
        for stage_index, stage in enumerate(stages):
            color = palette[stage_index % len(palette)]
            points = []
            for position, latent_index in enumerate(latent_indices):
                row = by_stage[stage].get(latent_index)
                if row is None:
                    continue
                value = float(row[key])
                y = plot_bottom - min(value / max(maximum, 1e-6), 1.0) * (plot_bottom - plot_top)
                points.append((x_coord(position), y))
            if len(points) > 1:
                draw.line(points, fill=color, width=2)
            for point in points:
                draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=color)
        for position, latent_index in enumerate(latent_indices):
            draw.text((x_coord(position) - 7, plot_bottom + 8), f"L{latent_index}", fill="black")

    draw_panel(0, "recall", "GT recall by latent frame", 1.0)
    max_tokens = max(
        float(row["pred_tokens"]) for row in rows
    )
    max_tokens = max(
        max_tokens, max(float(row["gt_tokens"]) for row in rows), 1.0
    )
    draw_panel(panel_height + 38, "pred_tokens", "Selected token count by latent frame", max_tokens)
    token_top = panel_height + 38 + margin_top
    token_bottom = 2 * panel_height + 38 - margin_bottom
    gt_points = [
        (
            x_coord(position),
            token_bottom
            - float(by_latent[latent_index]["gt_tokens"])
            / max_tokens
            * (token_bottom - token_top),
        )
        for position, latent_index in enumerate(latent_indices)
    ]
    if len(gt_points) > 1:
        draw.line(gt_points, fill=(0, 0, 0), width=3)
    for point in gt_points:
        draw.rectangle(
            (point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2),
            fill=(0, 0, 0),
        )

    legend_x = margin_left
    for index, stage in enumerate(stages):
        color = palette[index % len(palette)]
        draw.rectangle((legend_x, 24, legend_x + 10, 34), fill=color)
        draw.text((legend_x + 14, 22), stage, fill="black")
        legend_x += 14 + 7 * len(stage) + 20
    draw.rectangle((legend_x, 24, legend_x + 10, 34), fill=(0, 0, 0))
    draw.text((legend_x + 14, 22), "GT tokens", fill="black")
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    debug_pattern = re.compile(r"block_(\d+)_hand_role_debug\.npz")
    debug_paths = sorted(
        (
            path for path in args.roles_dir.iterdir()
            if debug_pattern.fullmatch(path.name)
        ),
        key=lambda path: int(debug_pattern.fullmatch(path.name).group(1)),
    )
    if not debug_paths:
        raise FileNotFoundError(
            f"No block_*_hand_role_debug.npz in {args.roles_dir}"
        )
    source_frames = decode_video(args.source_video)
    gt_video_frames = decode_video(args.gt_mask_video)
    gt_pixel = map_video_masks_to_source(
        gt_video_frames,
        source_frames,
        threshold=args.white_threshold,
        diff_threshold=args.overlay_diff_threshold,
    )

    first_debug = np.load(debug_paths[0])
    frames_per_block = int(first_debug["object_posterior"].shape[1])
    token_height, token_width = first_debug["object_posterior"].shape[-2:]
    latent_frames = len(debug_paths) * frames_per_block
    expected_latent_frames = 1 + (len(source_frames) - 1) // 4
    if latent_frames != expected_latent_frames:
        raise ValueError(
            "Role debug does not cover the complete source video: "
            f"debug_latents={latent_frames}, "
            f"source_latents={expected_latent_frames}"
        )
    gt = project_gt(
        gt_pixel,
        latent_frames=latent_frames,
        size=(token_width, token_height),
    )
    source_indices = [left for left, _ in causal_groups(
        len(source_frames), latent_frames
    )]
    pixel_groups = causal_groups(len(source_frames), latent_frames)

    hand_evidence = None
    if args.hand_input_npz is not None:
        hand_evidence = np.load(args.hand_input_npz)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    tiles = []
    stage_names = (
        "object_seed",
        "soft_edit_region",
        "selected_posterior",
        "causal_owner_observation",
        "causal_owner_weight",
        "native_owner_read",
        "native_owner_write",
    )
    for block_index, debug_path in enumerate(debug_paths):
        role_path = args.roles_dir / debug_path.name.replace(
            "_hand_role_debug.npz", "_roles.npz"
        )
        with np.load(debug_path) as debug:
            posterior = debug["object_posterior"][0]
            threshold = debug["posterior_threshold"][0]
            available_stages = {
                "selected_posterior": posterior >= threshold,
            }
            if role_path.exists():
                with np.load(role_path) as roles:
                    edit_weight = roles["edit_weight"][0]
                available_stages["soft_edit_region"] = np.stack([
                    resize_mask(frame > 0.0, (token_width, token_height))
                    for frame in edit_weight
                ])
            for name in stage_names:
                if name in debug:
                    available_stages[name] = debug[name][0] > 0.0
            if "connected_hysteresis_support" in debug:
                available_stages["connected_hysteresis_support"] = (
                    debug["connected_hysteresis_support"][0] > 0.0
                )
            ordered_names = [
                "object_seed",
                "connected_hysteresis_support",
                "selected_posterior",
                "soft_edit_region",
                "causal_owner_observation",
                "causal_owner_weight",
                "native_owner_read",
                "native_owner_write",
            ]
            stages = {
                name: available_stages[name]
                for name in ordered_names
                if name in available_stages
            }

        for local_frame in range(frames_per_block):
            latent_index = block_index * frames_per_block + local_frame
            if latent_index >= latent_frames:
                continue
            frame_tiles = []
            for stage_name, stage in stages.items():
                values = metric(stage[local_frame], gt[latent_index])
                rows.append({
                    "latent_frame": latent_index,
                    "block": block_index,
                    "block_frame": local_frame,
                    "pixel_frame_start": pixel_groups[latent_index][0],
                    "pixel_frame_end": pixel_groups[latent_index][1] - 1,
                    "stage": stage_name,
                    **values,
                })
                view = overlay(
                    stage[local_frame],
                    gt[latent_index],
                    source_frames[source_indices[latent_index]],
                )
                view_image = Image.fromarray(view).resize(
                    (token_width * 8, token_height * 8),
                    Image.Resampling.NEAREST,
                )
                frame_tiles.append(add_title(
                    view_image,
                    f"L{latent_index} B{block_index} {stage_name} "
                    f"px={pixel_groups[latent_index][0]}-"
                    f"{pixel_groups[latent_index][1] - 1} "
                    f"P={values['precision']:.3f} "
                    f"R={values['recall']:.3f} "
                    f"IoU={values['iou']:.3f} "
                    f"pred={values['pred_tokens']} "
                    f"gt={values['gt_tokens']}",
                ))
            tile = Image.new(
                "RGB",
                (frame_tiles[0].width, sum(x.height for x in frame_tiles)),
                "white",
            )
            top = 0
            for frame_tile in frame_tiles:
                tile.paste(frame_tile, (0, top))
                top += frame_tile.height
            tiles.append((latent_index, tile))

    csv_path = args.output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_lines = []
    for stage_name in sorted({row["stage"] for row in rows}):
        stage_rows = [row for row in rows if row["stage"] == stage_name]
        tp = sum(row["tp"] for row in stage_rows)
        fp = sum(row["fp"] for row in stage_rows)
        fn = sum(row["fn"] for row in stage_rows)
        summary_lines.append(
            f"{stage_name:32s} "
            f"precision={tp / max(tp + fp, 1):.4f} "
            f"recall={tp / max(tp + fn, 1):.4f} "
            f"iou={tp / max(tp + fp + fn, 1):.4f} "
            f"pred={tp + fp} gt={tp + fn}"
        )

    if hand_evidence is not None:
        if "hand_latent_mask" in hand_evidence:
            union = hand_evidence["hand_latent_mask"].astype(bool)
            occupancy = hand_evidence.get(
                "hand_occupancy_latent", union.astype(np.float32)
            )
            persistent = hand_evidence.get(
                "hand_persistent_latent_mask", union
            ).astype(bool)
            summary_lines.append("")
            summary_lines.append(
                "hand_evidence_mean "
                f"union={union.mean():.4f} "
                f"occupancy={occupancy.mean():.4f} "
                f"persistent={persistent.mean():.4f}"
            )
            for name, evidence in (
                ("union_hard_exclusion", union),
                ("occupancy_ge_0.5", occupancy >= 0.5),
                ("persistent_hard_exclusion", persistent),
            ):
                if evidence.shape[0] != latent_frames:
                    continue
                evidence_token = np.stack([
                    resize_mask(frame, (token_width, token_height))
                    for frame in evidence[:latent_frames]
                ])
                overlap = np.logical_and(evidence_token, gt).sum()
                summary_lines.append(
                    f"{name:32s} gt_removed="
                    f"{overlap / max(gt.sum(), 1):.4f}"
                )

    summary_path = args.output_dir / "summary.txt"
    summary_path.write_text(
        "LEGEND green=true-positive red=false-positive "
        "blue=false-negative\n"
        "GT_USAGE=offline_evaluation_only\n"
        + "\n".join(summary_lines)
        + "\n",
        encoding="utf-8",
    )
    save_token_count_plot(rows, args.output_dir / "token_count_recall.png")

    for page_index in range(0, len(tiles), args.tiles_per_page):
        page_tiles = tiles[page_index:page_index + args.tiles_per_page]
        legend_height = 30
        page = Image.new(
            "RGB",
            (
                sum(tile.width for _, tile in page_tiles),
                page_tiles[0][1].height + legend_height,
            ),
            "white",
        )
        ImageDraw.Draw(page).text(
            (8, 8),
            "GT comparison: green=TP, red=FP, blue=FN (GT is evaluation only)",
            fill="black",
        )
        left = 0
        for _, tile in page_tiles:
            page.paste(tile, (left, legend_height))
            left += tile.width
        first = page_tiles[0][0]
        last = page_tiles[-1][0]
        page.save(args.output_dir / f"gt_compare_L{first:02d}_L{last:02d}.png")

    print(summary_path.read_text(encoding="utf-8"), end="")
    print(f"metrics={csv_path}")
    print(f"curves={args.output_dir}/token_count_recall.png")
    print(f"visualizations={args.output_dir}/gt_compare_L*_L*.png")


if __name__ == "__main__":
    main()
