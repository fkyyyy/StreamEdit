#!/usr/bin/env python3
"""Render three-frame diagnostics from an existing single-case capture.

This is an offline renderer.  It never invokes the model.  Cross-attention,
responsibilities, permissions, velocity fields, and anchor maps are sliced at
each latent frame stored in the selected responsibility block.  The existing
self-attention capture contains queries from one frame only; those fixed
queries are therefore shown against each current-chunk key frame and labelled
accordingly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import visualize_single_case_diagnostics as single


SCHEMA_VERSION = "single-case-multiframe-diagnostics-v1"
DISPLAY_LAYER = 15
DISPLAY_STEP = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render all frames available in an existing single-case diagnostics block."
    )
    parser.add_argument("--responsibility-dir", type=Path, required=True)
    parser.add_argument("--attention-dir", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--edited-video", type=Path, required=True)
    parser.add_argument("--baseline-video", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--hand-role-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=DISPLAY_LAYER)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 1, 7])
    return parser.parse_args()


def _select_interface(block: Any, block_position: int, local_frame: int) -> Any:
    candidate = (
        block.maps["valid_token"][local_frame]
        & block.maps["connected_object_support"][local_frame]
        & (block.maps[block.hand_field][local_frame] > 0)
    )
    if not bool(candidate.any()):
        raise single.SchemaError(
            f"latent {int(block.latent_frame_indices[local_frame])}: no valid connected hand-supported token"
        )
    scores = np.where(candidate, block.maps["q_interface"][local_frame], -np.inf)
    row, column = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return single.responsibility_viz.TokenSelection(
        block_position, local_frame, int(row), int(column), float(scores[row, column])
    )


def _attention_subset(
    artifacts: list[single.AttentionArtifact], layer: int, steps: list[int]
) -> dict[tuple[str, int], single.AttentionArtifact]:
    result = {
        (artifact.kind, artifact.step): artifact
        for artifact in artifacts
        if artifact.layer == layer and artifact.step in steps
    }
    expected = {(kind, step) for kind in single.ATTENTION_KINDS for step in steps}
    missing = expected - set(result)
    if missing:
        raise single.SchemaError(f"Missing requested attention captures: {sorted(missing)}")
    return result


def _render_attention_frames(
    subset: dict[tuple[str, int], single.AttentionArtifact],
    latent_frames: list[int],
    output_dir: Path,
    layer: int,
    steps: list[int],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"cross": [], "self_fixed_queries": []}
    for step in steps:
        cross = subset[("cross", step)]
        cross_maps = {
            latent: single.cross_attention_maps(cross, latent) for latent in latent_frames
        }
        cross_scale = (
            min(float(value.min()) for value in cross_maps.values()),
            max(float(value.max()) for value in cross_maps.values()),
        )
        for latent, maps in cross_maps.items():
            destination = (
                output_dir / f"latent_{latent:03d}" / "attention" /
                f"cross_step_{step:03d}_layer_{layer:02d}"
            )
            rendered = single._render_attention_bundle(
                destination,
                maps,
                f"cross S{step} L{layer} latent {latent}",
                cross_scale,
            )
            metadata["cross"].append({
                "latent_frame": latent,
                "step": step,
                "layer": layer,
                "source": str(cross.path.resolve()),
                **rendered,
            })

        self_artifact = subset[("self", step)]
        views = single.self_attention_views(self_artifact)
        shared_values = [view.current_maps for view in views]
        self_scale = (
            min(float(value.min()) for value in shared_values),
            max(float(value.max()) for value in shared_values),
        )
        artifact_latents = [int(value) for value in self_artifact.metadata["latent_frame_indices"]]
        for local_frame, latent in enumerate(artifact_latents):
            if latent not in latent_frames:
                continue
            for view in views:
                maps = view.current_maps[:, local_frame]
                destination = (
                    output_dir / f"latent_{latent:03d}" / "attention" /
                    f"self_fixed_query_step_{step:03d}_layer_{layer:02d}" /
                    f"query_{single._safe_name(view.name)}"
                )
                rendered = single._render_attention_bundle(
                    destination,
                    maps,
                    f"self S{step} L{layer} fixed {view.name} -> key latent {latent}",
                    self_scale,
                )
                single._write_json(destination / "history_mass.json", view.mass)
                metadata["self_fixed_queries"].append({
                    "key_latent_frame": latent,
                    "step": step,
                    "layer": layer,
                    "query_name": view.name,
                    "query_original_index": view.metadata["original_query_index"],
                    "query_frame_policy": (
                        "fixed query captured at latent 19; only the current-chunk key frame is varied"
                    ),
                    "source": str(self_artifact.path.resolve()),
                    **rendered,
                })
    return metadata


def _image_panel(axis: Any, image: np.ndarray, title: str) -> None:
    axis.imshow(image)
    axis.set_title(title, fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])


def _map_panel(
    figure: Any,
    axis: Any,
    value: np.ndarray,
    title: str,
    *,
    cmap: Any = "turbo",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    image = axis.imshow(value, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axis.set_title(title, fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.025)


def _save_overviews(
    output_dir: Path,
    block: Any,
    local_frames: list[int],
    pixel_frames: dict[int, int],
    source: Any,
    edited: Any,
    cross_tm: dict[int, np.ndarray],
    self_tm: dict[str, np.ndarray],
    velocity_tm: dict[int, dict[str, np.ndarray]],
) -> dict[str, str]:
    rows = len(local_frames)
    retention_scale = max(
        float(np.quantile(block.raw_velocity_maps["rho_role_magnitude"][frame], 0.99))
        for frame in local_frames
    )
    retention_scale = retention_scale if retention_scale > 0 else 1.0
    cross_scale = max(float(value.mean(axis=0).max()) for value in cross_tm.values())
    cross_scale = cross_scale if cross_scale > 0 else 1.0

    figure, axes = plt.subplots(rows, 8, figsize=(25, 3.4 * rows), squeeze=False)
    for row, local_frame in enumerate(local_frames):
        latent = int(block.latent_frame_indices[local_frame])
        pixel = pixel_frames[latent]
        _image_panel(axes[row, 0], source.frames[pixel], f"Source\nlatent {latent} / pixel {pixel}")
        _image_panel(axes[row, 1], edited.frames[pixel], "Edited result")
        _map_panel(
            figure, axes[row, 2], cross_tm[latent].mean(axis=0),
            "Cross-attn head mean\nTm layer 15", vmax=cross_scale,
        )
        _image_panel(
            axes[row, 3], single._soft_role_composite(block, local_frame), "Composite soft roles"
        )
        _map_panel(
            figure, axes[row, 4], block.raw_velocity_maps["rho_role_magnitude"][local_frame],
            "Source retention\n||rho_role||", cmap="Blues", vmax=retention_scale,
        )
        _map_panel(
            figure, axes[row, 5], block.maps["appearance_access"][local_frame],
            "Appearance access", cmap="Oranges",
        )
        _map_panel(
            figure, axes[row, 6], block.maps["reference_write_gate"][local_frame],
            "Reference write", cmap="Purples",
        )
        _map_panel(
            figure, axes[row, 7], block.maps["reference_read_gate"][local_frame],
            "Anchor read", cmap="Purples",
        )
    figure.suptitle("Three-frame mechanism overview (same captured Full run)", fontsize=14)
    figure.tight_layout()
    overview = output_dir / "multiframe_overview.png"
    figure.savefig(overview, dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(rows, 5, figsize=(18, 3.4 * rows), squeeze=False)
    self_scale = max(float(value.max()) for value in self_tm.values())
    self_scale = self_scale if self_scale > 0 else 1.0
    for row, local_frame in enumerate(local_frames):
        latent = int(block.latent_frame_indices[local_frame])
        pixel = pixel_frames[latent]
        _image_panel(axes[row, 0], source.frames[pixel], f"Source latent {latent}\npixel {pixel}")
        _map_panel(
            figure, axes[row, 1], cross_tm[latent].mean(axis=0),
            "Cross mean", vmax=cross_scale,
        )
        for column, name in enumerate(("interface_query", "object_query", "hand_query"), start=2):
            _map_panel(
                figure, axes[row, column], self_tm[name][:, local_frame].mean(axis=0),
                f"Fixed frame-19 {name.replace('_query', '')}\nattention to key frame {latent}",
                vmax=self_scale,
            )
    figure.suptitle(
        "Three-frame attention view; Self queries are fixed at latent 19 (no model rerun)", fontsize=14
    )
    figure.tight_layout()
    attention = output_dir / "multiframe_attention.png"
    figure.savefig(attention, dpi=180, bbox_inches="tight")
    plt.close(figure)

    update_fields = (
        ("conflict_score", "Conflict", 1.0),
        ("removed_component", "Removed conflict", None),
        ("safe_residual_norm", "Safe residual", None),
        ("role_allocated_correction", "Role correction", None),
        ("controlled_velocity_change", "Controlled change", None),
    )
    magnitude_scale = max(
        float(np.quantile(value[field].reshape(-1), 0.99))
        for value in velocity_tm.values()
        for field, _, fixed in update_fields
        if fixed is None
    )
    magnitude_scale = magnitude_scale if magnitude_scale > 0 else 1.0
    figure, axes = plt.subplots(rows, len(update_fields), figsize=(18, 3.4 * rows), squeeze=False)
    for row, local_frame in enumerate(local_frames):
        latent = int(block.latent_frame_indices[local_frame])
        for column, (field, label, fixed) in enumerate(update_fields):
            _map_panel(
                figure, axes[row, column], velocity_tm[latent][field],
                f"latent {latent}\n{label}", cmap="inferno",
                vmax=fixed if fixed is not None else magnitude_scale,
            )
    figure.suptitle("Three-frame Role-Guided Update at Tm; shared magnitude scale", fontsize=14)
    figure.tight_layout()
    update = output_dir / "multiframe_velocity_update.png"
    figure.savefig(update, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {
        "multiframe_overview": str(overview.resolve()),
        "multiframe_attention": str(attention.resolve()),
        "multiframe_velocity_update": str(update.resolve()),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    selection = single.load_selection(args.selection)
    blocks = single.responsibility_viz.load_blocks(args.responsibility_dir)
    positions = [index for index, value in enumerate(blocks) if value.index == selection["block_index"]]
    if len(positions) != 1:
        raise single.SchemaError("Selected responsibility block must exist exactly once")
    block_position = positions[0]
    block = blocks[block_position]
    local_frames = list(range(len(block.latent_frame_indices)))
    latent_frames = [int(value) for value in block.latent_frame_indices]
    raw_block, raw_block_path = single._raw_responsibility_block(
        args.responsibility_dir, block.index
    )
    groups = single.load_temporal_groups(args.hand_role_input)
    source = single.read_video(args.source_video)
    edited = single.read_video(args.edited_video)
    baseline = single.read_video(args.baseline_video)
    videos = {"source": source, "edited": edited, "baseline": baseline}
    pixel_mapping = {
        latent: single.selected_pixel_frame(groups, latent, videos) for latent in latent_frames
    }

    artifacts = single.discover_attention_artifacts(args.attention_dir)
    subset = _attention_subset(artifacts, args.layer, args.steps)
    attention_metadata = _render_attention_frames(
        subset, latent_frames, output_dir, args.layer, args.steps
    )

    frame_metadata = []
    velocity_tm: dict[int, dict[str, np.ndarray]] = {}
    for local_frame, latent in zip(local_frames, latent_frames):
        frame_dir = output_dir / f"latent_{latent:03d}"
        frame_dir.mkdir(exist_ok=True)
        interface = _select_interface(block, block_position, local_frame)
        pixel_frame, pixel_group = pixel_mapping[latent]
        Image.fromarray(source.frames[pixel_frame]).save(frame_dir / "source_frame.png")
        Image.fromarray(edited.frames[pixel_frame]).save(frame_dir / "edited_frame.png")
        Image.fromarray(baseline.frames[pixel_frame]).save(frame_dir / "baseline_frame.png")
        single.render_responsibility(block, local_frame, frame_dir / "responsibility")
        single.render_permissions(block, local_frame, frame_dir / "permissions")
        card, _ = single.interface_card(block, local_frame, interface, frame_dir)
        velocity_metrics, _ = single.render_velocity(
            raw_block, block, local_frame, interface, raw_block_path, frame_dir
        )
        anchor_metadata, _ = single.render_anchor(
            raw_block, block, local_frame, raw_block_path, frame_dir / "anchor"
        )
        updates, _ = single._velocity_updates(raw_block, local_frame, raw_block_path)
        velocity_tm[latent] = single._derived_velocity_maps(updates["prediction_007"])
        frame_metadata.append({
            "latent_frame": latent,
            "local_frame": local_frame,
            "pixel_frame_range": list(pixel_group),
            "representative_pixel_frame": pixel_frame,
            "interface_token": card,
            "velocity": velocity_metrics,
            "anchor": anchor_metadata,
        })

    cross_tm = {
        latent: single.cross_attention_maps(subset[("cross", DISPLAY_STEP)], latent)
        for latent in latent_frames
    }
    self_views = single.self_attention_views(subset[("self", DISPLAY_STEP)])
    self_tm = {view.name: view.current_maps for view in self_views}
    overview_outputs = _save_overviews(
        output_dir, block, local_frames,
        {latent: value[0] for latent, value in pixel_mapping.items()},
        source, edited, cross_tm, self_tm, velocity_tm,
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model_rerun": False,
        "source_capture": str(args.attention_dir.resolve()),
        "block_index": block.index,
        "latent_frames": latent_frames,
        "attention_display_layer": args.layer,
        "attention_steps": args.steps,
        "self_attention_limitation": (
            "The three saved Self-Attention queries originate at latent frame 19. "
            "Each row varies only the current-chunk key frame; it is not a newly selected query per frame."
        ),
        "frames": frame_metadata,
        "attention": attention_metadata,
        "overview_outputs": overview_outputs,
    }
    single._write_json(output_dir / "metadata.json", metadata)
    print(json.dumps({
        "output_dir": str(output_dir),
        "latent_frames": latent_frames,
        **overview_outputs,
    }, indent=2))


if __name__ == "__main__":
    main()
