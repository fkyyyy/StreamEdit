#!/usr/bin/env python3
"""Offline-only final-block comparison for the wallet ablation."""

from pathlib import Path
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from visualize_965a_wallet_streamgve import decode, font


ROOT = Path(__file__).parents[1]
CROP = (331, 212, 573, 480)
RUNS = (
    (
        "967h source-K / target-V",
        ROOT / "outputs/967h_source_bg_attention_diagnostics/967h-source-bg-attention-diagnostics.mp4",
    ),
    (
        "967j drop source-bg K/V",
        ROOT / "outputs/967j_drop_source_bg_kv/967j-drop-source-bg-kv.mp4",
    ),
)
OUTPUT = (
    ROOT
    / "outputs/967j_drop_source_bg_kv/analysis/967h_967j_b6_comparison.jpg"
)


def main() -> None:
    frames = [(label, decode(path, filter_graph="generated")) for label, path in RUNS]
    tile_size = (255, 282)
    left = 220
    top = 48
    canvas = Image.new(
        "RGB",
        (left + 6 * tile_size[0], top + 4 * tile_size[1]),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 8),
        "Final causal block F69-F80; same fixed crop and display scale",
        fill="black",
        font=font(21),
    )
    for run_index, (label, video) in enumerate(frames):
        base_row = run_index * 2
        draw.text(
            (8, top + base_row * tile_size[1] + 12),
            label,
            fill="black",
            font=font(16),
        )
        for offset, frame_index in enumerate(range(69, 81)):
            local_row, column = divmod(offset, 6)
            crop = Image.fromarray(video[frame_index]).crop(CROP)
            crop = crop.resize(tile_size, Image.Resampling.LANCZOS)
            crop_draw = ImageDraw.Draw(crop)
            crop_draw.rectangle((0, 0, 52, 25), fill="black")
            crop_draw.text(
                (5, 3), f"F{frame_index}", fill="white", font=font(15)
            )
            canvas.paste(
                crop,
                (
                    left + column * tile_size[0],
                    top + (base_row + local_row) * tile_size[1],
                ),
            )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, quality=95)
    print(OUTPUT)


if __name__ == "__main__":
    main()
