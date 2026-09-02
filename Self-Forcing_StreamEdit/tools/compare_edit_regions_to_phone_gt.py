#!/usr/bin/env python3
"""Offline-only comparison of 965c runtime edit regions to a phone GT mask.

The GT mask is used exclusively after inference. It is never imported by the
generation pipeline or any run script. Alignment intentionally mirrors the
existing inference preprocessing: uniform frame-index resampling, white-overlay
extraction, resize to 480x832, causal-VAE temporal grouping, and projection to
the 30x52 role-token grid.
"""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path

import av
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--phone-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--white-threshold", type=int, default=245)
    parser.add_argument("--overlay-diff-threshold", type=float, default=24.0)
    return parser.parse_args()


def decode_video(path: Path) -> tuple[list[np.ndarray], Fraction]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = Fraction(stream.average_rate or 16)
        frames = [
            frame.to_ndarray(format="rgb24")
            for frame in container.decode(video=0)
        ]
    if not frames:
        raise ValueError(f"No video frames decoded from {path}")
    return frames, rate


def resize_rgb(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(frame).resize(size, Image.Resampling.BILINEAR)
    )


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(
            size, Image.Resampling.NEAREST
        )
    ) > 127


def causal_groups(pixel_frames: int, latent_frames: int) -> list[tuple[int, int]]:
    if latent_frames == 1:
        return [(0, pixel_frames)]
    stride = (pixel_frames - 1) // (latent_frames - 1)
    if stride <= 0 or 1 + (pixel_frames - 1) // stride != latent_frames:
        raise ValueError(
            "Pixel and latent frame counts do not define causal grouping: "
            f"pixel={pixel_frames}, latent={latent_frames}"
        )
    return [(0, 1)] + [
        (1 + stride * (index - 1), 1 + stride * index)
        for index in range(1, latent_frames)
    ]


def extract_phone_overlay(
    mask_frames: list[np.ndarray],
    source_frames: list[np.ndarray],
    *,
    white_threshold: int,
    overlay_diff_threshold: float,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-source-frame masks at overlay and model resolution."""

    indices = np.rint(
        np.linspace(0, len(mask_frames) - 1, len(source_frames))
    ).astype(int)
    raw_masks = []
    model_masks = []
    for source_index, mask_index in enumerate(indices):
        overlay = mask_frames[int(mask_index)]
        source = resize_rgb(
            source_frames[source_index],
            (overlay.shape[1], overlay.shape[0]),
        )
        white = np.all(overlay >= white_threshold, axis=-1)
        difference = np.abs(
            overlay.astype(np.int16) - source.astype(np.int16)
        ).mean(axis=-1)
        phone = white & (difference >= overlay_diff_threshold)
        raw_masks.append(phone)
        model_masks.append(resize_mask(phone, output_size))
    return np.stack(raw_masks), np.stack(model_masks)


def max_pool_2x(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape[-2:]
    if height % 2 or width % 2:
        raise ValueError(f"Cannot pool odd mask shape {(height, width)}")
    return mask.reshape(
        *mask.shape[:-2], height // 2, 2, width // 2, 2
    ).max(axis=(-3, -1))


def load_runtime_regions(roles_dir: Path):
    paths = sorted(roles_dir.glob("block_*_hand_role_debug.npz"))
    if not paths:
        raise FileNotFoundError(f"No role debug NPZ files in {roles_dir}")
    selected_all = []
    owner_all = []
    posterior_all = []
    threshold_all = []
    source_attention_all = []
    flow_object_all = []
    hand_all = []
    for expected, path in enumerate(paths):
        if path.name != f"block_{expected:03d}_hand_role_debug.npz":
            raise ValueError(f"Role blocks are not contiguous at {path}")
        with np.load(path) as debug:
            posterior = debug["object_posterior"][0].astype(np.float32)
            threshold = debug["posterior_threshold"][0].astype(np.float32)
            owner = debug["causal_owner_support"][0] > 0.5
            selected = posterior >= threshold
            selected_all.append(selected)
            owner_all.append(owner)
            posterior_all.append(posterior)
            threshold_all.append(threshold.reshape(-1))
            source_attention_all.append(
                debug["source_attention"][0].astype(np.float32)
            )
            flow_object_all.append(
                debug["flow_object_likelihood"][0].astype(np.float32)
            )
            hand_all.append(debug["hand_hard_exclusion"][0] > 0)
    selected = np.concatenate(selected_all, axis=0)
    owner = np.concatenate(owner_all, axis=0)
    return {
        "token_selected": selected,
        "flow_owner": owner,
        "runtime_union": selected | owner,
        "posterior": np.concatenate(posterior_all, axis=0),
        "threshold": np.concatenate(threshold_all, axis=0),
        "source_attention": np.concatenate(source_attention_all, axis=0),
        "flow_object": np.concatenate(flow_object_all, axis=0),
        "hand": np.concatenate(hand_all, axis=0),
    }


def centroid(mask: np.ndarray, width: int = 832, height: int = 480):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return float("nan"), float("nan")
    return (
        float((xs.mean() + 0.5) * width / mask.shape[1]),
        float((ys.mean() + 0.5) * height / mask.shape[0]),
    )


def mask_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    pred_count = int(prediction.sum())
    target_count = int(target.sum())
    px, py = centroid(prediction)
    gx, gy = centroid(target)
    center_error = float(np.hypot(px - gx, py - gy))
    return {
        "iou": intersection / max(union, 1),
        "precision": intersection / max(pred_count, 1),
        "recall": intersection / max(target_count, 1),
        "coverage": pred_count / prediction.size,
        "gt_coverage": target_count / target.size,
        "area_ratio": pred_count / max(target_count, 1),
        "centroid_x": px,
        "centroid_y": py,
        "gt_centroid_x": gx,
        "gt_centroid_y": gy,
        "centroid_error_px": center_error,
        "false_positive_coverage": (
            np.logical_and(prediction, ~target).sum() / prediction.size
        ),
        "false_negative_coverage": (
            np.logical_and(~prediction, target).sum() / prediction.size
        ),
    }


def outline(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1)
    eroded = np.ones_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return mask.astype(bool) & ~eroded


def overlay(
    frame: np.ndarray,
    layers: list[tuple[np.ndarray, tuple[int, int, int], float]],
    *,
    dim: float = 0.62,
) -> np.ndarray:
    canvas = frame.astype(np.float32) * dim
    size = (frame.shape[1], frame.shape[0])
    for mask, color, alpha in layers:
        resized = resize_mask(mask, size)
        blend = resized[..., None].astype(np.float32) * alpha
        canvas = canvas * (1 - blend) + np.asarray(color) * blend
    return np.clip(canvas, 0, 255).astype(np.uint8)


def error_overlay(frame: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return overlay(
        frame,
        [
            (pred & gt, (40, 220, 80), 0.78),
            (pred & ~gt, (255, 45, 45), 0.82),
            (~pred & gt, (40, 220, 255), 0.85),
        ],
    )


def label(frame: np.ndarray, title: str, subtitle: str = "") -> Image.Image:
    header = 42
    image = Image.new("RGB", (frame.shape[1], frame.shape[0] + header), "white")
    image.paste(Image.fromarray(frame), (0, header))
    draw = ImageDraw.Draw(image)
    draw.text((5, 4), title, fill="black")
    draw.text((5, 22), subtitle, fill=(45, 45, 45))
    return image


def write_contact_sheets(
    output_dir: Path,
    source_frames: list[np.ndarray],
    generated_frames: list[np.ndarray],
    groups: list[tuple[int, int]],
    gt_role: np.ndarray,
    regions: dict[str, np.ndarray],
    rows: list[dict[str, float]],
) -> None:
    tile_size = (320, 185)
    for page_start in range(0, len(groups), 7):
        page_end = min(page_start + 7, len(groups))
        page = Image.new("RGB", (tile_size[0] * 7, (tile_size[1] + 42) * 5), "white")
        for column, latent_index in enumerate(range(page_start, page_end)):
            left, right = groups[latent_index]
            pixel_index = right - 1
            source = resize_rgb(source_frames[pixel_index], (832, 480))
            generated = generated_frames[pixel_index]
            gt = gt_role[latent_index]
            selected = regions["token_selected"][latent_index]
            owner = regions["flow_owner"][latent_index]
            runtime = regions["runtime_union"][latent_index]
            metric = rows[latent_index]
            panels = [
                label(
                    overlay(source, [(gt, (45, 220, 70), 0.72)]),
                    f"L{latent_index:02d} GT phone: green",
                    f"px={left}-{right - 1} coverage={metric['gt_coverage']:.3f}",
                ),
                label(
                    generated,
                    "generated 965c",
                    "visual object pose/shape",
                ),
                label(
                    overlay(
                        source,
                        [
                            (selected, (255, 50, 50), 0.62),
                            (owner, (40, 210, 255), 0.58),
                            (outline(gt), (40, 255, 70), 1.0),
                        ],
                    ),
                    "token red | flow owner cyan | GT outline green",
                    f"thr={metric['threshold']:.3f} union={metric['runtime_union_coverage']:.3f}",
                ),
                label(
                    error_overlay(source, runtime, gt),
                    "runtime vs GT: TP green | FP red | FN cyan",
                    f"IoU={metric['runtime_union_iou']:.3f} P={metric['runtime_union_precision']:.3f} R={metric['runtime_union_recall']:.3f}",
                ),
                label(
                    overlay(
                        source,
                        [
                            (regions["hand"][latent_index], (0, 210, 255), 0.72),
                            (runtime, (255, 45, 45), 0.48),
                            (outline(gt), (40, 255, 70), 1.0),
                        ],
                    ),
                    "runtime red | hand cyan | GT outline green",
                    f"center err={metric['runtime_union_centroid_error_px']:.1f}px area={metric['runtime_union_area_ratio']:.2f}x",
                ),
            ]
            for row_index, panel in enumerate(panels):
                panel = panel.resize(
                    (tile_size[0], tile_size[1] + 42), Image.Resampling.LANCZOS
                )
                page.paste(panel, (column * tile_size[0], row_index * (tile_size[1] + 42)))
        page.save(
            output_dir / f"phone_gt_comparison_L{page_start:02d}_L{page_end - 1:02d}.jpg",
            quality=92,
        )


def write_timeline(
    output_dir: Path,
    rows: list[dict[str, float]],
    regions: dict[str, np.ndarray],
) -> None:
    latent = np.arange(len(rows))
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    axes[0].plot(latent, [row["gt_coverage"] for row in rows], "k-o", label="GT phone")
    for name, color in (("token_selected", "tab:red"), ("flow_owner", "tab:blue"), ("runtime_union", "tab:orange")):
        axes[0].plot(latent, [row[f"{name}_coverage"] for row in rows], "-o", color=color, label=name)
    axes[0].set_ylabel("token coverage")
    axes[0].legend(ncol=4)
    for name, color in (("token_selected", "tab:red"), ("flow_owner", "tab:blue"), ("runtime_union", "tab:orange")):
        axes[1].plot(latent, [row[f"{name}_iou"] for row in rows], "-o", color=color, label=name)
    axes[1].set_ylabel("IoU with phone GT")
    axes[1].set_ylim(0, 1)
    for name, color in (("token_selected", "tab:red"), ("flow_owner", "tab:blue"), ("runtime_union", "tab:orange")):
        axes[2].plot(latent, [row[f"{name}_centroid_error_px"] for row in rows], "-o", color=color, label=name)
    axes[2].set_ylabel("centroid error (px)")
    axes[3].plot(latent, regions["threshold"], "m-o", label="adaptive posterior threshold")
    axes[3].set_ylabel("threshold")
    axes[3].set_xlabel("latent frame (vertical lines: 3-frame block boundaries)")
    for axis in axes:
        axis.grid(alpha=0.25)
        for boundary in range(3, len(rows), 3):
            axis.axvline(boundary - 0.5, color="gray", linestyle="--", alpha=0.4)
    fig.suptitle("965c automatic edit region vs offline phone GT")
    fig.tight_layout()
    fig.savefig(output_dir / "phone_gt_metrics_timeline.png", dpi=180)
    plt.close(fig)


def write_video(
    output_path: Path,
    source_frames: list[np.ndarray],
    generated_frames: list[np.ndarray],
    pixel_gt: np.ndarray,
    groups: list[tuple[int, int]],
    regions: dict[str, np.ndarray],
    rate: Fraction,
) -> None:
    latent_for_pixel = np.zeros(len(source_frames), dtype=np.int64)
    for latent_index, (left, right) in enumerate(groups):
        latent_for_pixel[left:right] = latent_index
    output = av.open(str(output_path), mode="w")
    stream = output.add_stream("libx264", rate=rate)
    stream.width = 832 * 2
    stream.height = 480 * 2
    stream.pix_fmt = "yuv420p"
    for pixel_index, (source_raw, generated) in enumerate(
        zip(source_frames, generated_frames)
    ):
        source = resize_rgb(source_raw, (832, 480))
        latent_index = int(latent_for_pixel[pixel_index])
        gt_pixel = pixel_gt[pixel_index]
        gt_role = resize_mask(
            pixel_gt[pixel_index], (52, 30)
        )
        runtime = regions["runtime_union"][latent_index]
        top_left = overlay(source, [(gt_pixel, (40, 230, 70), 0.72)])
        top_right = generated
        bottom_left = overlay(
            source,
            [
                (runtime, (255, 45, 45), 0.62),
                (regions["flow_owner"][latent_index], (40, 210, 255), 0.55),
                (outline(gt_role), (40, 255, 70), 1.0),
            ],
        )
        bottom_right = error_overlay(source, runtime, gt_role)
        canvas = np.concatenate(
            [
                np.concatenate([top_left, top_right], axis=1),
                np.concatenate([bottom_left, bottom_right], axis=1),
            ],
            axis=0,
        )
        frame = av.VideoFrame.from_ndarray(canvas, format="rgb24")
        for packet in stream.encode(frame):
            output.mux(packet)
    for packet in stream.encode():
        output.mux(packet)
    output.close()


def motion_summary(rows: list[dict[str, float]], name: str) -> dict[str, float]:
    pred = np.asarray([
        [row[f"{name}_centroid_x"], row[f"{name}_centroid_y"]]
        for row in rows
    ])
    gt = np.asarray([
        [row["gt_centroid_x"], row["gt_centroid_y"]]
        for row in rows
    ])
    residual = pred - gt
    displacement_error = np.linalg.norm(
        np.diff(pred, axis=0) - np.diff(gt, axis=0), axis=1
    )
    boundary_indices = np.arange(3, len(rows), 3)
    boundary_error = displacement_error[boundary_indices - 1]
    return {
        "mean_iou": float(np.mean([row[f"{name}_iou"] for row in rows])),
        "mean_precision": float(np.mean([row[f"{name}_precision"] for row in rows])),
        "mean_recall": float(np.mean([row[f"{name}_recall"] for row in rows])),
        "mean_area_ratio": float(np.mean([row[f"{name}_area_ratio"] for row in rows])),
        "area_ratio_std": float(np.std([row[f"{name}_area_ratio"] for row in rows])),
        "mean_centroid_error_px": float(np.mean(np.linalg.norm(residual, axis=1))),
        "residual_centroid_std_x_px": float(np.std(residual[:, 0])),
        "residual_centroid_std_y_px": float(np.std(residual[:, 1])),
        "mean_motion_error_px": float(np.mean(displacement_error)),
        "mean_block_boundary_motion_error_px": float(np.mean(boundary_error)),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "phone_gt_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    video_paths = [
        path for path in sorted(run_dir.glob("*.mp4"))
        if "inference-edit-regions" not in path.name
    ]
    if len(video_paths) != 1:
        raise ValueError(f"Expected one generated video, got {video_paths}")

    source_frames, source_rate = decode_video(args.source_video)
    generated_frames, generated_rate = decode_video(video_paths[0])
    mask_frames, mask_rate = decode_video(args.phone_mask)
    if len(generated_frames) != len(source_frames):
        raise ValueError(
            f"Source/generated frame mismatch: {len(source_frames)} vs {len(generated_frames)}"
        )
    generated_frames = [resize_rgb(frame, (832, 480)) for frame in generated_frames]
    _, pixel_gt = extract_phone_overlay(
        mask_frames,
        source_frames,
        white_threshold=args.white_threshold,
        overlay_diff_threshold=args.overlay_diff_threshold,
        output_size=(832, 480),
    )
    regions = load_runtime_regions(run_dir / "roles")
    latent_frames = regions["runtime_union"].shape[0]
    groups = causal_groups(len(source_frames), latent_frames)
    latent_gt_60x104 = np.stack([
        resize_mask(pixel_gt[left:right].max(axis=0), (104, 60))
        for left, right in groups
    ])
    gt_role = max_pool_2x(latent_gt_60x104)
    if gt_role.shape != regions["runtime_union"].shape:
        raise ValueError(
            f"GT/runtime role shape mismatch: {gt_role.shape} vs "
            f"{regions['runtime_union'].shape}"
        )

    rows = []
    names = ("token_selected", "flow_owner", "runtime_union")
    for latent_index, (left, right) in enumerate(groups):
        row: dict[str, float | int] = {
            "latent_frame": latent_index,
            "block": latent_index // 3,
            "frame_in_block": latent_index % 3,
            "pixel_left": left,
            "pixel_right_exclusive": right,
            "threshold": float(regions["threshold"][latent_index]),
        }
        gt_x, gt_y = centroid(gt_role[latent_index])
        row.update({
            "gt_coverage": float(gt_role[latent_index].mean()),
            "gt_centroid_x": gt_x,
            "gt_centroid_y": gt_y,
        })
        for name in names:
            metric = mask_metrics(regions[name][latent_index], gt_role[latent_index])
            for key, value in metric.items():
                if key.startswith("gt_"):
                    continue
                row[f"{name}_{key}"] = value
        rows.append(row)

    csv_path = output_dir / "phone_gt_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "contract": "offline_evaluation_only_phone_mask_never_enters_inference",
        "alignment": {
            "source_frames": len(source_frames),
            "source_fps": float(source_rate),
            "generated_frames": len(generated_frames),
            "generated_fps": float(generated_rate),
            "phone_mask_frames": len(mask_frames),
            "phone_mask_fps": float(mask_rate),
            "latent_frames": latent_frames,
            "role_grid": list(gt_role.shape[1:]),
            "white_threshold": args.white_threshold,
            "overlay_diff_threshold": args.overlay_diff_threshold,
        },
        "gt": {
            "mean_coverage": float(gt_role.mean()),
            "coverage_std": float(gt_role.mean(axis=(1, 2)).std()),
        },
        **{name: motion_summary(rows, name) for name in names},
    }
    (output_dir / "phone_gt_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_contact_sheets(
        output_dir,
        source_frames,
        generated_frames,
        groups,
        gt_role,
        regions,
        rows,
    )
    write_timeline(output_dir, rows, regions)
    write_video(
        output_dir / "phone_gt_region_comparison.mp4",
        source_frames,
        generated_frames,
        pixel_gt,
        groups,
        regions,
        generated_rate,
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote offline GT comparison to {output_dir}")


if __name__ == "__main__":
    main()
