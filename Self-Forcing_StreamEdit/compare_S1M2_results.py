#!/usr/bin/env python3
"""
Compare S1+M2 full system results against baselines.

Analyzes:
1. Identity preservation: CLIP-I temporal consistency
2. Wobble metrics: centroid displacement at block boundaries
3. Brightness: luminance statistics over time
4. Role posterior coverage: owner_gate statistics vs binary fg_mask
"""

import torch
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def load_video_frames(video_dir):
    """Load all frames from a directory."""
    frame_paths = sorted(Path(video_dir).glob("*.png"))
    frames = [np.array(Image.open(p)) for p in frame_paths]
    return np.stack(frames)


def compute_clip_i_consistency(frames, clip_model, preprocess):
    """Compute frame-to-frame CLIP image similarity."""
    from torchvision import transforms
    device = "cuda" if torch.cuda.is_available() else "cpu"

    similarities = []
    for i in range(len(frames) - 1):
        img1 = preprocess(Image.fromarray(frames[i])).unsqueeze(0).to(device)
        img2 = preprocess(Image.fromarray(frames[i+1])).unsqueeze(0).to(device)

        with torch.no_grad():
            feat1 = clip_model.encode_image(img1)
            feat2 = clip_model.encode_image(img2)
            feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
            feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
            sim = (feat1 @ feat2.T).item()
        similarities.append(sim)

    return np.array(similarities)


def compute_block_boundary_wobble(frames, block_size=20):
    """Detect motion jitter at block boundaries."""
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]

    displacements = []
    for i in range(len(gray_frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i], gray_frames[i+1], None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        displacements.append(magnitude.mean())

    displacements = np.array(displacements)

    # Compare boundary vs non-boundary frames
    boundary_indices = [i for i in range(len(displacements))
                       if (i + 1) % block_size == 0]
    non_boundary_indices = [i for i in range(len(displacements))
                           if (i + 1) % block_size != 0]

    boundary_wobble = displacements[boundary_indices].mean()
    non_boundary_wobble = displacements[non_boundary_indices].mean()

    return {
        "boundary_mean": boundary_wobble,
        "non_boundary_mean": non_boundary_wobble,
        "wobble_ratio": boundary_wobble / (non_boundary_wobble + 1e-8),
        "all_displacements": displacements
    }


def compute_brightness_decay(frames, mask_region=None):
    """Compute luminance statistics over time."""
    luminances = []
    for frame in frames:
        if mask_region is not None:
            frame = frame * mask_region[..., None]

        # Convert to LAB color space
        lab = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2LAB)
        luminance = lab[..., 0].mean()
        luminances.append(luminance)

    luminances = np.array(luminances)

    # Compute decay: linear fit slope
    x = np.arange(len(luminances))
    slope, _ = np.polyfit(x, luminances, 1)

    return {
        "mean_luminance": luminances.mean(),
        "decay_slope": slope,
        "luminance_curve": luminances
    }


def compare_experiments(baseline_dir, s1m2_dir, output_dir):
    """Compare baseline vs S1+M2 full system."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Loading frames...")
    baseline_frames = load_video_frames(baseline_dir)
    s1m2_frames = load_video_frames(s1m2_dir)

    print("Computing wobble metrics...")
    baseline_wobble = compute_block_boundary_wobble(baseline_frames)
    s1m2_wobble = compute_block_boundary_wobble(s1m2_frames)

    print("Computing brightness decay...")
    baseline_brightness = compute_brightness_decay(baseline_frames)
    s1m2_brightness = compute_brightness_decay(s1m2_frames)

    # Print summary
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    print("\n1. WOBBLE METRICS:")
    print(f"  Baseline boundary wobble:     {baseline_wobble['boundary_mean']:.4f}")
    print(f"  S1+M2 boundary wobble:        {s1m2_wobble['boundary_mean']:.4f}")
    print(f"  Improvement:                  {(1 - s1m2_wobble['boundary_mean']/baseline_wobble['boundary_mean'])*100:.2f}%")
    print(f"  Baseline wobble ratio:        {baseline_wobble['wobble_ratio']:.4f}")
    print(f"  S1+M2 wobble ratio:           {s1m2_wobble['wobble_ratio']:.4f}")

    print("\n2. BRIGHTNESS DECAY:")
    print(f"  Baseline mean luminance:      {baseline_brightness['mean_luminance']:.2f}")
    print(f"  S1+M2 mean luminance:         {s1m2_brightness['mean_luminance']:.2f}")
    print(f"  Baseline decay slope:         {baseline_brightness['decay_slope']:.4f}")
    print(f"  S1+M2 decay slope:            {s1m2_brightness['decay_slope']:.4f}")
    print(f"  Decay reduction:              {(1 - abs(s1m2_brightness['decay_slope'])/abs(baseline_brightness['decay_slope']))*100:.2f}%")

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Wobble over time
    axes[0].plot(baseline_wobble['all_displacements'], label='Baseline', alpha=0.7)
    axes[0].plot(s1m2_wobble['all_displacements'], label='S1+M2', alpha=0.7)
    axes[0].axhline(baseline_wobble['boundary_mean'], color='C0', linestyle='--', alpha=0.5)
    axes[0].axhline(s1m2_wobble['boundary_mean'], color='C1', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Optical Flow Magnitude')
    axes[0].set_xlabel('Frame')
    axes[0].legend()
    axes[0].set_title('Motion Jitter Over Time')
    axes[0].grid(True, alpha=0.3)

    # Brightness over time
    axes[1].plot(baseline_brightness['luminance_curve'], label='Baseline', alpha=0.7)
    axes[1].plot(s1m2_brightness['luminance_curve'], label='S1+M2', alpha=0.7)
    axes[1].set_ylabel('Luminance (LAB L-channel)')
    axes[1].set_xlabel('Frame')
    axes[1].legend()
    axes[1].set_title('Brightness Over Time')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=150)
    print(f"\nPlot saved to {output_dir / 'comparison.png'}")

    # Save numerical results
    results = {
        "wobble": {
            "baseline": baseline_wobble,
            "s1m2": s1m2_wobble
        },
        "brightness": {
            "baseline": baseline_brightness,
            "s1m2": s1m2_brightness
        }
    }

    import json
    with open(output_dir / "results.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        json.dump(convert(results), f, indent=2)

    print(f"Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", type=str, required=True,
                       help="Directory with baseline output frames")
    parser.add_argument("--s1m2_dir", type=str, required=True,
                       help="Directory with S1+M2 output frames")
    parser.add_argument("--output_dir", type=str, default="analysis/S1M2_comparison",
                       help="Directory to save analysis results")

    args = parser.parse_args()
    compare_experiments(args.baseline_dir, args.s1m2_dir, args.output_dir)
