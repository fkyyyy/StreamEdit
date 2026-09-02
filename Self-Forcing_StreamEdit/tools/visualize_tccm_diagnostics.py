#!/usr/bin/env python3
"""Render per-timestep TCCM desired/current/error/gain diagnostics.

This is post-hoc visualization only. It reads inference artifacts and never
feeds a spatial mask or other signal back into generation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw


FIELDS = (
    "tccm_admitted",
    "tccm_correspondence_confidence",
    "tccm_source_similarity",
    "tccm_desired_norm",
    "tccm_current_norm",
    "tccm_error_norm",
    "tccm_residual_error_norm",
    "tccm_gain",
    "tccm_clip_scale",
    "tccm_candidate_count",
    "tccm_attention_entropy",
    "tccm_attention_peak",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--roles-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def normalize(value: np.ndarray, *, signed: bool = False) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    finite = np.isfinite(value)
    if not finite.any():
        return np.zeros_like(value)
    if signed:
        value = (value + 1.0) * 0.5
        return np.clip(value, 0.0, 1.0)
    positive = value[finite & (value > 0.0)]
    scale = float(np.percentile(positive, 95)) if positive.size else 1.0
    return np.clip(value / max(scale, 1e-6), 0.0, 1.0)


def heatmap(value: np.ndarray, *, signed: bool = False) -> Image.Image:
    value = normalize(value, signed=signed)
    red = (255.0 * value).astype(np.uint8)
    green = (255.0 * np.minimum(value * 2.0, 1.0)).astype(np.uint8)
    blue = (255.0 * (1.0 - value)).astype(np.uint8)
    return Image.fromarray(np.stack((red, green, blue), axis=-1))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    roles_dir = (args.roles_dir or run_dir / "roles").resolve()
    output_dir = (
        args.output_dir or run_dir / "tccm_diagnostics"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"block_(\d+)_tccm_step_(\d+)_hand_role_debug\.npz")
    paths = sorted(
        (path for path in roles_dir.iterdir() if pattern.fullmatch(path.name)),
        key=lambda path: tuple(
            int(value) for value in pattern.fullmatch(path.name).groups()
        ),
    )
    if not paths:
        raise FileNotFoundError(
            f"No TCCM timestep diagnostics found in {roles_dir}"
        )

    rows = []
    for path in paths:
        block, step = (
            int(value) for value in pattern.fullmatch(path.name).groups()
        )
        with np.load(path) as debug:
            available = [field for field in FIELDS if field in debug]
            if not available:
                continue
            maps = {field: debug[field][0] for field in available}
        frames = maps[available[0]].shape[0]
        cell = (240, 144)
        sheet = Image.new(
            "RGB", (frames * cell[0], len(available) * (cell[1] + 24)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for row_index, field in enumerate(available):
            for frame_index in range(frames):
                image = heatmap(
                    maps[field][frame_index],
                    signed=(field == "tccm_source_similarity"),
                ).resize(cell, Image.Resampling.NEAREST)
                x = frame_index * cell[0]
                y = row_index * (cell[1] + 24)
                sheet.paste(image, (x, y + 24))
                draw.text(
                    (x + 4, y + 5),
                    f"{field} | B{block} S{step:02d} F{frame_index}",
                    fill="black",
                )
        sheet.save(output_dir / f"block_{block:03d}_step_{step:02d}.png")
        admitted = maps.get(
            "tccm_admitted", np.zeros_like(maps[available[0]])
        ) > 0
        for frame_index in range(frames):
            support = admitted[frame_index]
            count = max(int(support.sum()), 1)
            row = {
                "block": block, "step": step,
                "block_frame": frame_index,
                "admission_fraction": float(support.mean()),
            }
            for field in available:
                value = maps[field][frame_index]
                row[f"{field}_mean"] = float(
                    value[support].sum() / count if support.any() else 0.0
                )
            rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "tccm_stats.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "legend.txt").write_text(
        "TCCM diagnostics are post-hoc and use no object mask.\n"
        "desired/current are paired target-minus-source attention response norms.\n"
        "error is desired-current before gain; residual_error is after feedback.\n"
        "correspondence comes from complete clean-source RGB optical flow.\n",
        encoding="utf-8",
    )
    print(f"diagnostics={output_dir}")
    print(f"stats={output_dir / 'tccm_stats.csv'}")


if __name__ == "__main__":
    main()
