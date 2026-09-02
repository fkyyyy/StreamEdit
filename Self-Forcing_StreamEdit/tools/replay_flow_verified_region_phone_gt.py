#!/usr/bin/env python3
"""Replay the generic flow-verified region against offline phone GT.

The phone mask is decoded only by this post-inference tool.  It is never used
to construct ownership, choose parameters, or influence generated latents.
The candidate region is rebuilt exclusively from role diagnostics that the
inference pipeline already produced without an object mask.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import av
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from compare_edit_regions_to_phone_gt import (
    causal_groups,
    centroid,
    decode_video,
    error_overlay,
    extract_phone_overlay,
    label,
    mask_metrics,
    max_pool_2x,
    motion_summary,
    outline,
    overlay,
    resize_mask,
    resize_rgb,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_region_builder():
    path = REPO_ROOT / "pipeline" / "source_flow_verified_region.py"
    module_name = "_offline_source_flow_verified_region"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load verified-region implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.build_source_flow_verified_region


build_source_flow_verified_region = _load_region_builder()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--phone-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--owner-radius", type=int, default=1)
    parser.add_argument(
        "--background-veto-threshold", type=float, default=0.55
    )
    parser.add_argument(
        "--background-veto-min-confidence", type=float, default=0.50
    )
    parser.add_argument("--white-threshold", type=int, default=245)
    parser.add_argument(
        "--overlay-diff-threshold", type=float, default=24.0
    )
    return parser.parse_args()


def _concat_debug(
    paths: list[Path], name: str, *, required: bool = True
) -> np.ndarray | None:
    values = []
    for path in paths:
        with np.load(path) as debug:
            if name not in debug:
                if required:
                    raise KeyError(f"Missing '{name}' in {path}")
                return None
            value = debug[name].astype(np.float32)
            if value.shape[0] != 1:
                raise ValueError(f"Expected batch size one for {name}: {path}")
            values.append(value)
    return np.concatenate(values, axis=1)


def load_and_replay_regions(
    roles_dir: Path,
    *,
    owner_radius: int,
    background_veto_threshold: float,
    background_veto_min_confidence: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    paths = sorted(roles_dir.glob("block_*_hand_role_debug.npz"))
    if not paths:
        raise FileNotFoundError(f"No role debug NPZ files in {roles_dir}")
    for expected, path in enumerate(paths):
        if path.name != f"block_{expected:03d}_hand_role_debug.npz":
            raise ValueError(f"Role blocks are not contiguous at {path}")

    source_key = "object_posterior"
    with np.load(paths[0]) as first:
        if "object_posterior_pre_latest_flow_verification" in first:
            source_key = "object_posterior_pre_latest_flow_verification"
        elif "object_posterior_pre_flow_verification" in first:
            source_key = "object_posterior_pre_flow_verification"
        elif "object_posterior_pre_source_flow" in first:
            source_key = "object_posterior_pre_source_flow"

    posterior = _concat_debug(paths, source_key)
    threshold = _concat_debug(paths, "posterior_threshold")
    owner = _concat_debug(paths, "causal_owner_support")
    hand = _concat_debug(paths, "hand_hard_exclusion")
    background = _concat_debug(paths, "flow_background_likelihood")
    confidence = _concat_debug(paths, "flow_cycle_confidence")
    saved_verified = _concat_debug(
        paths, "source_flow_verified_support", required=False
    )
    assert posterior is not None
    assert threshold is not None
    assert owner is not None
    assert hand is not None
    assert background is not None
    assert confidence is not None

    replay = build_source_flow_verified_region(
        object_posterior=torch.from_numpy(posterior),
        posterior_threshold=torch.from_numpy(threshold),
        owner_support=torch.from_numpy(owner),
        hand_exclusion=torch.from_numpy(hand),
        background_likelihood=torch.from_numpy(background),
        flow_confidence=torch.from_numpy(confidence),
        owner_radius=owner_radius,
        background_veto_threshold=background_veto_threshold,
        background_veto_min_confidence=(
            background_veto_min_confidence
        ),
    )
    proposal = posterior >= threshold
    flow_owner = owner > 0.5
    replayed = replay.support.numpy()[0]
    regions = {
        "token_proposal": proposal[0],
        "flow_owner": flow_owner[0],
        "old_runtime": (proposal | flow_owner)[0],
        "verified_semantic": replay.verified_semantic.numpy()[0],
        "background_veto": replay.reliable_background_veto.numpy()[0],
        "hand": hand[0] > 0,
        "verified_region": replayed,
        "threshold": threshold[0, :, 0, 0],
    }
    replay_meta: dict[str, object] = {
        "proposal_posterior_key": source_key,
        "saved_verified_region_present": saved_verified is not None,
    }
    if saved_verified is not None:
        saved = saved_verified[0] > 0.5
        regions["saved_verified_region"] = saved
        replay_meta.update({
            "saved_replay_exact_match": bool(np.array_equal(saved, replayed)),
            "saved_replay_disagreement_fraction": float(
                np.not_equal(saved, replayed).mean()
            ),
        })
    return regions, replay_meta


def _find_generated_video(run_dir: Path) -> Path:
    paths = [
        path
        for path in sorted(run_dir.glob("*.mp4"))
        if "inference-edit-regions" not in path.name
        and "comparison" not in path.name
    ]
    if len(paths) != 1:
        raise ValueError(f"Expected one generated video, got {paths}")
    return paths[0]


def _write_contact_sheets(
    output_dir: Path,
    source_frames: list[np.ndarray],
    generated_frames: list[np.ndarray],
    groups: list[tuple[int, int]],
    gt: np.ndarray,
    regions: dict[str, np.ndarray],
    rows: list[dict[str, float]],
) -> None:
    tile_width, tile_height = 320, 185
    row_height = tile_height + 42
    for page_start in range(0, len(groups), 7):
        page_end = min(page_start + 7, len(groups))
        page = Image.new(
            "RGB", (tile_width * 7, row_height * 6), "white"
        )
        for column, latent_index in enumerate(range(page_start, page_end)):
            left, right = groups[latent_index]
            pixel_index = right - 1
            source = resize_rgb(source_frames[pixel_index], (832, 480))
            generated = generated_frames[pixel_index]
            target = gt[latent_index]
            old = regions["old_runtime"][latent_index]
            verified = regions["verified_region"][latent_index]
            metric = rows[latent_index]
            panels = [
                label(
                    overlay(source, [(target, (45, 220, 70), 0.72)]),
                    f"L{latent_index:02d} offline phone GT: green",
                    f"source pixels {left}-{right - 1}",
                ),
                label(generated, "generated video", "visual pose/shape check"),
                label(
                    overlay(
                        source,
                        [
                            (regions["token_proposal"][latent_index], (255, 45, 45), 0.55),
                            (regions["flow_owner"][latent_index], (30, 200, 255), 0.58),
                            (outline(target), (40, 255, 70), 1.0),
                        ],
                    ),
                    "proposal red | owner cyan | GT outline",
                    f"old area={metric['old_runtime_area_ratio']:.2f}x GT",
                ),
                label(
                    error_overlay(source, old, target),
                    "old 965c union: TP green | FP red | FN cyan",
                    f"IoU={metric['old_runtime_iou']:.3f} P={metric['old_runtime_precision']:.3f} R={metric['old_runtime_recall']:.3f}",
                ),
                label(
                    overlay(
                        source,
                        [
                            (regions["verified_region"][latent_index], (255, 45, 45), 0.58),
                            (regions["background_veto"][latent_index], (255, 210, 30), 0.40),
                            (regions["hand"][latent_index], (30, 200, 255), 0.50),
                            (outline(target), (40, 255, 70), 1.0),
                        ],
                    ),
                    "verified red | veto yellow | hand cyan | GT",
                    f"new area={metric['verified_region_area_ratio']:.2f}x GT",
                ),
                label(
                    error_overlay(source, verified, target),
                    "verified vs GT: TP green | FP red | FN cyan",
                    f"IoU={metric['verified_region_iou']:.3f} P={metric['verified_region_precision']:.3f} R={metric['verified_region_recall']:.3f}",
                ),
            ]
            for row_index, panel in enumerate(panels):
                panel = panel.resize(
                    (tile_width, row_height), Image.Resampling.LANCZOS
                )
                page.paste(
                    panel, (column * tile_width, row_index * row_height)
                )
        page.save(
            output_dir
            / f"phone_verified_L{page_start:02d}_L{page_end - 1:02d}.jpg",
            quality=92,
        )


def _write_timeline(
    output_dir: Path, rows: list[dict[str, float]]
) -> None:
    latent = np.arange(len(rows))
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    for name, color in (
        ("old_runtime", "tab:orange"),
        ("verified_region", "tab:green"),
    ):
        axes[0].plot(
            latent, [row[f"{name}_iou"] for row in rows], "-o",
            color=color, label=name,
        )
        axes[1].plot(
            latent, [row[f"{name}_precision"] for row in rows], "-o",
            color=color, label=f"{name} precision",
        )
        axes[1].plot(
            latent, [row[f"{name}_recall"] for row in rows], "--o",
            color=color, alpha=0.65, label=f"{name} recall",
        )
        axes[2].plot(
            latent, [row[f"{name}_area_ratio"] for row in rows], "-o",
            color=color, label=name,
        )
        axes[3].plot(
            latent, [row[f"{name}_centroid_error_px"] for row in rows],
            "-o", color=color, label=name,
        )
    axes[0].set_ylabel("IoU with phone GT")
    axes[1].set_ylabel("precision / recall")
    axes[2].set_ylabel("area / GT area")
    axes[3].set_ylabel("centroid error (px)")
    axes[3].set_xlabel("latent frame; dashed lines are chunk boundaries")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=2)
        for boundary in range(3, len(rows), 3):
            axis.axvline(boundary - 0.5, color="gray", linestyle="--", alpha=0.4)
    fig.suptitle("965c union vs source-flow-verified region on offline phone GT")
    fig.tight_layout()
    fig.savefig(output_dir / "phone_verified_metrics_timeline.png", dpi=180)
    plt.close(fig)


def _write_video(
    output_path: Path,
    source_frames: list[np.ndarray],
    generated_frames: list[np.ndarray],
    pixel_gt: np.ndarray,
    groups: list[tuple[int, int]],
    regions: dict[str, np.ndarray],
    rate,
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
        target = resize_mask(pixel_gt[pixel_index], (52, 30))
        gt_panel = overlay(source, [(pixel_gt[pixel_index], (40, 230, 70), 0.72)])
        old_panel = error_overlay(
            source, regions["old_runtime"][latent_index], target
        )
        verified_panel = error_overlay(
            source, regions["verified_region"][latent_index], target
        )
        canvas = np.concatenate(
            [
                np.concatenate([gt_panel, generated], axis=1),
                np.concatenate([old_panel, verified_panel], axis=1),
            ],
            axis=0,
        )
        frame = av.VideoFrame.from_ndarray(canvas, format="rgb24")
        for packet in stream.encode(frame):
            output.mux(packet)
    for packet in stream.encode():
        output.mux(packet)
    output.close()


def _false_positive_hand_fraction(
    prediction: np.ndarray, target: np.ndarray, hand: np.ndarray
) -> float:
    false_positive = prediction & ~target
    return float((false_positive & hand).sum() / max(false_positive.sum(), 1))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "phone_gt_flow_verified_replay"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_frames, source_rate = decode_video(args.source_video)
    generated_frames, generated_rate = decode_video(
        _find_generated_video(run_dir)
    )
    mask_frames, mask_rate = decode_video(args.phone_mask)
    if len(generated_frames) != len(source_frames):
        raise ValueError(
            "Source/generated frame mismatch: "
            f"{len(source_frames)} vs {len(generated_frames)}"
        )
    generated_frames = [
        resize_rgb(frame, (832, 480)) for frame in generated_frames
    ]
    _, pixel_gt = extract_phone_overlay(
        mask_frames,
        source_frames,
        white_threshold=args.white_threshold,
        overlay_diff_threshold=args.overlay_diff_threshold,
        output_size=(832, 480),
    )
    regions, replay_meta = load_and_replay_regions(
        run_dir / "roles",
        owner_radius=args.owner_radius,
        background_veto_threshold=args.background_veto_threshold,
        background_veto_min_confidence=(
            args.background_veto_min_confidence
        ),
    )
    latent_frames = regions["old_runtime"].shape[0]
    groups = causal_groups(len(source_frames), latent_frames)
    latent_gt_60x104 = np.stack([
        resize_mask(pixel_gt[left:right].max(axis=0), (104, 60))
        for left, right in groups
    ])
    gt_role = max_pool_2x(latent_gt_60x104)
    if gt_role.shape != regions["old_runtime"].shape:
        raise ValueError(
            f"GT/runtime shape mismatch: {gt_role.shape} vs "
            f"{regions['old_runtime'].shape}"
        )

    names = ("token_proposal", "flow_owner", "old_runtime", "verified_region")
    rows: list[dict[str, float]] = []
    for latent_index, (left, right) in enumerate(groups):
        gt_x, gt_y = centroid(gt_role[latent_index])
        row: dict[str, float | int] = {
            "latent_frame": latent_index,
            "block": latent_index // 3,
            "frame_in_block": latent_index % 3,
            "pixel_left": left,
            "pixel_right_exclusive": right,
            "threshold": float(regions["threshold"][latent_index]),
            "gt_coverage": float(gt_role[latent_index].mean()),
            "gt_centroid_x": gt_x,
            "gt_centroid_y": gt_y,
        }
        for name in names:
            for key, value in mask_metrics(
                regions[name][latent_index], gt_role[latent_index]
            ).items():
                if not key.startswith("gt_"):
                    row[f"{name}_{key}"] = value
        rows.append(row)

    with (output_dir / "phone_verified_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries = {name: motion_summary(rows, name) for name in names}
    old_summary = summaries["old_runtime"]
    new_summary = summaries["verified_region"]
    summary = {
        "contract": (
            "offline_evaluation_only_phone_mask_never_enters_inference"
        ),
        "method": {
            "proposal": "token posterior above adaptive threshold",
            "verification": "clean-source flow owner neighborhood",
            "owner_radius": args.owner_radius,
            "background_veto_threshold": (
                args.background_veto_threshold
            ),
            "background_veto_min_confidence": (
                args.background_veto_min_confidence
            ),
            "hard_hand_exclusion": "applied_last",
            **replay_meta,
        },
        "alignment": {
            "source_frames": len(source_frames),
            "source_fps": float(source_rate),
            "generated_frames": len(generated_frames),
            "generated_fps": float(generated_rate),
            "phone_mask_frames": len(mask_frames),
            "phone_mask_fps": float(mask_rate),
            "latent_frames": latent_frames,
            "role_grid": list(gt_role.shape[1:]),
        },
        "gt_mean_coverage": float(gt_role.mean()),
        **summaries,
        "verified_minus_old": {
            "mean_iou": new_summary["mean_iou"] - old_summary["mean_iou"],
            "mean_precision": (
                new_summary["mean_precision"]
                - old_summary["mean_precision"]
            ),
            "mean_recall": (
                new_summary["mean_recall"] - old_summary["mean_recall"]
            ),
            "mean_area_ratio": (
                new_summary["mean_area_ratio"]
                - old_summary["mean_area_ratio"]
            ),
            "mean_centroid_error_px": (
                new_summary["mean_centroid_error_px"]
                - old_summary["mean_centroid_error_px"]
            ),
            "mean_block_boundary_motion_error_px": (
                new_summary["mean_block_boundary_motion_error_px"]
                - old_summary["mean_block_boundary_motion_error_px"]
            ),
        },
        "false_positive_hand_fraction": {
            name: _false_positive_hand_fraction(
                regions[name], gt_role, regions["hand"]
            )
            for name in ("old_runtime", "verified_region")
        },
    }
    (output_dir / "phone_verified_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_contact_sheets(
        output_dir, source_frames, generated_frames, groups, gt_role,
        regions, rows,
    )
    _write_timeline(output_dir, rows)
    _write_video(
        output_dir / "phone_verified_comparison.mp4",
        source_frames, generated_frames, pixel_gt, groups, regions,
        generated_rate,
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote phone-only offline replay to {output_dir}")


if __name__ == "__main__":
    main()
