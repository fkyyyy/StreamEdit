#!/usr/bin/env python3
"""Visualize inference-time edit routing without an object GT mask.

The script consumes only artifacts already emitted by inference. It never
feeds a mask back into generation. In particular, ``runtime_edit_support`` is
reconstructed from the exact hand-role path used by the pipeline:

    (object_posterior >= posterior_threshold) OR causal_owner_support

Separate panels expose source-flow role evidence, target-flow ownership,
exact-source closure, native-KV reads, the conservative KV write core, and the
source-flow-indexed residual ledger, and the frozen multi-frame identity sink.
"""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
from pathlib import Path
import re

import av
import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--roles-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tiles-per-page", type=int, default=7)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    return parser.parse_args()


def decode_video(path: Path) -> tuple[list[np.ndarray], Fraction]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or Fraction(16, 1)
        frames = [
            frame.to_ndarray(format="rgb24")
            for frame in container.decode(video=0)
        ]
    if not frames:
        raise ValueError(f"Video contains no frames: {path}")
    return frames, Fraction(rate)


def find_video(run_dir: Path) -> Path:
    candidates = [
        path for path in sorted(run_dir.glob("*.mp4"))
        if "inference-edit-regions" not in path.name
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one generated mp4 in {run_dir}, got {candidates}"
        )
    return candidates[0]


def causal_groups(pixel_frames: int, latent_frames: int):
    if latent_frames == 1:
        return [(0, pixel_frames)]
    stride = (pixel_frames - 1) // (latent_frames - 1)
    if stride <= 0 or 1 + (pixel_frames - 1) // stride != latent_frames:
        raise ValueError(
            "Video and debug frames do not define the causal VAE mapping: "
            f"pixel_frames={pixel_frames}, latent_frames={latent_frames}"
        )
    return [(0, 1)] + [
        (1 + stride * (index - 1), 1 + stride * index)
        for index in range(1, latent_frames)
    ]


def resize_map(value: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(value, dtype=np.float32), mode="F")
    return np.asarray(
        image.resize(size, Image.Resampling.NEAREST), dtype=np.float32
    )


def downsample_2x(value: np.ndarray) -> np.ndarray:
    height, width = value.shape[-2:]
    if height % 2 or width % 2:
        raise ValueError(
            f"Expected an even factorized role grid, got {(height, width)}"
        )
    return value.reshape(
        *value.shape[:-2], height // 2, 2, width // 2, 2
    ).mean(axis=(-3, -1))


def color_overlay(
    frame: np.ndarray,
    layers: list[tuple[np.ndarray, tuple[int, int, int], float]],
    *,
    dim: float = 0.58,
) -> np.ndarray:
    height, width = frame.shape[:2]
    canvas = frame.astype(np.float32) * dim
    for values, color, base_alpha in layers:
        resized = np.clip(resize_map(values, (width, height)), 0.0, 1.0)
        support = resized > 0.0
        alpha = np.where(
            support, base_alpha * (0.35 + 0.65 * resized), 0.0
        )[..., None]
        color_array = np.asarray(color, dtype=np.float32)[None, None]
        canvas = canvas * (1.0 - alpha) + color_array * alpha
    return np.clip(canvas, 0, 255).astype(np.uint8)


def labeled(image: np.ndarray, title: str, subtitle: str = "") -> Image.Image:
    header = 40 if subtitle else 25
    result = Image.new(
        "RGB", (image.shape[1], image.shape[0] + header), "white"
    )
    result.paste(Image.fromarray(image), (0, header))
    draw = ImageDraw.Draw(result)
    draw.text((5, 4), title, fill="black")
    if subtitle:
        draw.text((5, 21), subtitle, fill=(55, 55, 55))
    return result


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.resize(size, Image.Resampling.LANCZOS)


def parse_logged_edit_coverage(log_path: Path) -> dict[int, float]:
    if not log_path.is_file():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    # HAND_ROLE_FIELD is emitted after the online field update and therefore
    # supersedes the earlier HAND_ROLE_FLOW summary for the same block.
    result = {
        int(block): float(value)
        for block, value in re.findall(
            r"HAND_ROLE_FLOW block=(\d+).*?edit_tokens=([0-9.]+)", text
        )
    }
    result.update({
        int(block): float(value)
        for block, value in re.findall(
            r"HAND_ROLE_FIELD block=(\d+).*?edit_tokens=([0-9.]+)", text
        )
    })
    return result


def load_maps(
    roles_dir: Path, log_path: Path, write_threshold: float
) -> tuple[dict[str, np.ndarray], int]:
    pattern = re.compile(r"block_(\d+)_hand_role_debug\.npz")
    paths = sorted(
        (path for path in roles_dir.iterdir() if pattern.fullmatch(path.name)),
        key=lambda path: int(pattern.fullmatch(path.name).group(1)),
    )
    if not paths:
        raise FileNotFoundError(
            f"No block_*_hand_role_debug.npz files in {roles_dir}"
        )
    expected_blocks = list(range(len(paths)))
    actual_blocks = [int(pattern.fullmatch(path.name).group(1)) for path in paths]
    if actual_blocks != expected_blocks:
        raise ValueError(f"Debug blocks are not contiguous: {actual_blocks}")

    log_text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.is_file() else ""
    )
    token_atomic = "payload_commit=token_atomic" in log_text
    logged_coverage = parse_logged_edit_coverage(log_path)
    collected: dict[str, list[np.ndarray]] = {}

    for block_index, path in enumerate(paths):
        with np.load(path) as debug:
            posterior = debug["object_posterior"][0]
            threshold = debug["posterior_threshold"][0]
            selected = posterior >= threshold
            owner = debug["causal_owner_support"][0] > 0.5
            runtime_edit = selected | owner
            grid_shape = runtime_edit.shape
            zeros = np.zeros(grid_shape, dtype=np.float32)

            def debug_map(name: str) -> np.ndarray:
                return debug[name][0] if name in debug else zeros

            hand = debug_map("hand_hard_exclusion") > 0.0

            logged = logged_coverage.get(block_index)
            if logged is not None and abs(float(runtime_edit.mean()) - logged) > 7e-5:
                raise ValueError(
                    "Reconstructed runtime edit support disagrees with the "
                    f"inference log at block {block_index}: "
                    f"derived={runtime_edit.mean():.6f}, logged={logged:.6f}"
                )

            def factorized_debug_map(name: str) -> np.ndarray:
                if name not in debug:
                    return zeros
                value = debug[name][0]
                if value.shape != grid_shape:
                    value = downsample_2x(value)
                return value

            target_action = factorized_debug_map(
                "factorized_flow_target_owned_action"
            )
            if not target_action.any():
                target_action = factorized_debug_map(
                    "factorized_target_memory_action"
                )
            source_closure = factorized_debug_map(
                "factorized_flow_owner_complement_source_action"
            )
            kv_read = (
                debug["native_history_applied_read_strength"][0]
                if "native_history_applied_read_strength" in debug
                else zeros
            )
            write_gate = factorized_debug_map(
                "factorized_target_memory_action"
            )
            if "native_owner_write" in debug:
                write_confidence = (
                    debug["native_owner_write"][0] * write_gate
                ).clip(0.0, 1.0)
            else:
                # Immutable-memory replay is deliberately read-only. Its
                # proposal write lives under roles/proposal; the main rollout
                # exposes an all-zero identity_write_weight after freezing.
                write_confidence = debug_map(
                    "identity_write_weight"
                ).clip(0.0, 1.0)
            write_core = write_confidence >= write_threshold
            # 956a's block-atomic bug committed all target payload whenever a
            # block had any valid write.  Token-atomic runs commit exactly the
            # per-token core. This visualization reflects the implementation,
            # not a desired or GT region.
            payload_commit = (
                write_core
                if token_atomic
                else np.ones_like(write_core, dtype=bool)
                if write_core.any()
                else np.zeros_like(write_core, dtype=bool)
            )
            topology_holes = (
                debug["native_owner_topology_holes"][0]
                if "native_owner_topology_holes" in debug else zeros
            )
            topology_recovery = (
                debug["native_owner_topology_read_recovery"][0]
                if "native_owner_topology_read_recovery" in debug else zeros
            )
            persistent_path = roles_dir / (
                f"block_{block_index:03d}_persistent_kv_transaction"
                "_hand_role_debug.npz"
            )
            persistent_direct = zeros
            persistent_retained = zeros
            persistent_guarded = zeros
            persistent_payload = payload_commit.astype(np.float32)
            flow_ledger_support = zeros
            flow_ledger_confidence = zeros
            flow_ledger_appearance_trust = zeros
            flow_ledger_local_transport = zeros
            if persistent_path.is_file():
                with np.load(persistent_path) as persistent_debug:
                    persistent_direct = persistent_debug[
                        "persistent_kv_direct_write"
                    ][0]
                    persistent_retained = persistent_debug[
                        "persistent_kv_retained_residual"
                    ][0]
                    persistent_payload = persistent_debug[
                        "persistent_kv_payload_support"
                    ][0]
                    if "persistent_kv_guarded_update" in persistent_debug:
                        persistent_guarded = persistent_debug[
                            "persistent_kv_guarded_update"
                        ][0]
                    if "flow_indexed_state_support" in persistent_debug:
                        flow_ledger_support = persistent_debug[
                            "flow_indexed_state_support"
                        ][0]
                    if "flow_indexed_state_confidence" in persistent_debug:
                        flow_ledger_confidence = persistent_debug[
                            "flow_indexed_state_confidence"
                        ][0]
                    if "flow_indexed_appearance_trust" in persistent_debug:
                        flow_ledger_appearance_trust = persistent_debug[
                            "flow_indexed_appearance_trust"
                        ][0]
                    if (
                        "flow_indexed_local_transport_confidence"
                        in persistent_debug
                    ):
                        flow_ledger_local_transport = persistent_debug[
                            "flow_indexed_local_transport_confidence"
                        ][0]

            values = {
                "runtime_edit": runtime_edit.astype(np.float32),
                "selected_posterior": selected.astype(np.float32),
                "causal_owner": owner.astype(np.float32),
                "hand": hand.astype(np.float32),
                "target_action": target_action.astype(np.float32),
                "source_closure": source_closure.astype(np.float32),
                "kv_read": kv_read.astype(np.float32),
                "kv_write_core": write_core.astype(np.float32),
                "payload_commit": payload_commit.astype(np.float32),
                "topology_holes": topology_holes.astype(np.float32),
                "topology_recovery": topology_recovery.astype(np.float32),
                "persistent_direct": persistent_direct.astype(np.float32),
                "persistent_retained": (
                    persistent_retained.astype(np.float32)
                ),
                "persistent_payload": persistent_payload.astype(np.float32),
                "persistent_guarded": persistent_guarded.astype(np.float32),
                "flow_object": debug_map(
                    "flow_object_likelihood"
                ).astype(np.float32),
                "flow_background": debug_map(
                    "flow_background_likelihood"
                ).astype(np.float32),
                "flow_boundary": debug_map(
                    "flow_boundary_likelihood"
                ).astype(np.float32),
                "flow_unknown": debug_map(
                    "flow_unknown_likelihood"
                ).astype(np.float32),
                "flow_recovered": debug_map(
                    "source_flow_recovered_support"
                ).astype(np.float32),
                "flow_transport_confidence": debug_map(
                    "native_history_flow_transport_confidence"
                ).astype(np.float32),
                "flow_appearance_trust": debug_map(
                    "native_history_flow_appearance_trust"
                ).astype(np.float32),
                "flow_local_transport_confidence": debug_map(
                    "native_history_flow_local_transport_confidence"
                ).astype(np.float32),
                "sink_admission": debug_map(
                    "native_history_sink_admission"
                ).astype(np.float32),
                "sink_selected_frame": debug_map(
                    "native_history_sink_selected_frame"
                ).astype(np.float32),
                "sink_source_similarity": debug_map(
                    "native_history_sink_source_similarity"
                ).astype(np.float32),
                "sink_attention_entropy": debug_map(
                    "native_history_sink_attention_entropy"
                ).astype(np.float32),
                "sink_attention_peak": debug_map(
                    "native_history_sink_attention_peak"
                ).astype(np.float32),
                "sink_coverage": debug_map(
                    "native_history_sink_coverage"
                ).astype(np.float32),
                "sink_applied_strength": debug_map(
                    "native_history_sink_applied_strength"
                ).astype(np.float32),
                "immutable_owner": debug_map(
                    "identity_owner_read"
                ).astype(np.float32),
                "immutable_support": debug_map(
                    "immutable_target_memory_support"
                ).astype(np.float32),
                "immutable_correction": debug_map(
                    "immutable_target_memory_correction_ratio"
                ).astype(np.float32),
                "immutable_subspace_coherence": (
                    debug_map(
                        "immutable_target_memory_appearance_subspace_coherence"
                    ) * debug_map("identity_owner_read")
                ).astype(np.float32),
                "immutable_assignment_entropy": (
                    debug_map(
                        "immutable_target_memory_prototype_assignment_entropy"
                    ) * debug_map("identity_owner_read")
                ).astype(np.float32),
                "immutable_assignment_peak": (
                    debug_map(
                        "immutable_target_memory_prototype_assignment_peak"
                    ) * debug_map("identity_owner_read")
                ).astype(np.float32),
                "immutable_assignment_margin": (
                    debug_map(
                        "immutable_target_memory_prototype_assignment_margin"
                    ) * debug_map("identity_owner_read")
                ).astype(np.float32),
                "flow_ledger_support": (
                    flow_ledger_support.astype(np.float32)
                ),
                "flow_ledger_confidence": (
                    flow_ledger_confidence.astype(np.float32)
                ),
                "flow_ledger_appearance_trust": (
                    flow_ledger_appearance_trust.astype(np.float32)
                ),
                "flow_ledger_local_transport": (
                    flow_ledger_local_transport.astype(np.float32)
                ),
            }
            for name, value in values.items():
                collected.setdefault(name, []).append(value)

    maps = {name: np.concatenate(values, axis=0) for name, values in collected.items()}
    return maps, len(paths)


def render_views(frame: np.ndarray, maps: dict[str, np.ndarray], index: int):
    runtime = color_overlay(
        frame,
        [
            (maps["runtime_edit"][index], (245, 55, 45), 0.76),
            (maps["hand"][index], (35, 190, 245), 0.68),
        ],
    )
    flow = color_overlay(
        frame,
        [
            (maps["source_closure"][index], (45, 105, 245), 0.68),
            (maps["target_action"][index], (255, 175, 25), 0.80),
        ],
    )
    source_flow = color_overlay(
        frame,
        [
            (maps["flow_background"][index], (45, 105, 245), 0.56),
            (maps["flow_object"][index], (35, 205, 80), 0.78),
            (maps["flow_boundary"][index], (255, 205, 25), 0.82),
            (maps["flow_recovered"][index], (245, 55, 45), 0.92),
        ],
    )
    immutable_present = bool((maps["immutable_owner"] > 0).any())
    kv = color_overlay(
        frame,
        (
            [
                (maps["immutable_owner"][index],
                 (40, 205, 85), 0.72),
                (maps["immutable_support"][index],
                 (255, 120, 20), 0.82),
                (maps["immutable_correction"][index],
                 (175, 65, 235), 0.88),
            ]
            if immutable_present
            else
            [
                (maps["sink_admission"][index],
                 (40, 205, 85), 0.76),
                (maps["sink_applied_strength"][index],
                 (255, 120, 20), 0.80),
                (maps["sink_coverage"][index],
                 (175, 65, 235), 0.86),
            ]
            if (maps["sink_admission"] > 0).any()
            else [
                (maps["flow_transport_confidence"][index],
                 (40, 205, 85), 0.76),
                (maps["kv_write_core"][index], (255, 120, 20), 0.80),
                (maps["flow_ledger_support"][index],
                 (175, 65, 235), 0.86),
            ]
        ),
    )
    return runtime, source_flow, flow, kv


def save_video(
    frames: list[np.ndarray],
    rate: Fraction,
    maps: dict[str, np.ndarray],
    groups: list[tuple[int, int]],
    output_path: Path,
) -> None:
    pixel_to_latent = np.empty(len(frames), dtype=np.int64)
    for latent_index, (left, right) in enumerate(groups):
        pixel_to_latent[left:right] = latent_index
    immutable_present = bool((maps["immutable_owner"] > 0).any())
    panel_size = (416, 240)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output_path), mode="w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = panel_size[0] * 5
        stream.height = panel_size[1] + 40
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        for pixel_index, frame in enumerate(frames):
            latent_index = int(pixel_to_latent[pixel_index])
            runtime, source_flow, flow, kv = render_views(
                frame, maps, latent_index
            )
            views = (frame, runtime, source_flow, flow, kv)
            labels = (
                "generated output",
                "runtime edit red | hand cyan",
                "source flow: obj green | bg blue | boundary yellow",
                "target route orange | source closure blue",
                (
                    "immutable KV: owner green | support orange | correction violet"
                    if immutable_present
                    else (
                        "962 sink: admit green | strength orange | views violet"
                        if (maps["sink_admission"] > 0).any()
                        else "flow read green | write orange | ledger violet"
                    )
                ),
            )
            canvas = Image.new(
                "RGB", (panel_size[0] * 5, panel_size[1] + 40), "white"
            )
            draw = ImageDraw.Draw(canvas)
            for column, (view, label) in enumerate(zip(views, labels)):
                x = column * panel_size[0]
                canvas.paste(
                    Image.fromarray(view).resize(
                        panel_size, Image.Resampling.LANCZOS
                    ),
                    (x, 40),
                )
                draw.text((x + 5, 5), label, fill="black")
            draw.text(
                (5, 22),
                f"pixel={pixel_index:02d} latent={latent_index:02d} "
                f"block={latent_index // 3}",
                fill=(60, 60, 60),
            )
            packet_frame = av.VideoFrame.from_ndarray(
                np.asarray(canvas), format="rgb24"
            )
            for packet in stream.encode(packet_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def save_contact_sheets(
    frames: list[np.ndarray],
    maps: dict[str, np.ndarray],
    groups: list[tuple[int, int]],
    output_dir: Path,
    tiles_per_page: int,
) -> None:
    cell = (260, 150)
    topology_present = bool((maps["topology_holes"] > 0).any())
    stage_names = [
        "runtime edit: red | hand: cyan",
        "source flow: object green | background blue",
        "source flow: boundary yellow | unknown magenta",
        "source-flow recovered support: red",
        "target route: orange",
        "exact-source closure: blue",
    ]
    immutable_present = bool((maps["immutable_owner"] > 0).any())
    if immutable_present:
        stage_names.extend((
            "immutable target-KV owner: green",
            "immutable target-KV retrieval support: orange",
            "immutable target-KV correction ratio: violet",
        ))
        if (maps["immutable_subspace_coherence"] > 0).any():
            stage_names.append(
                "pose-preserving subspace coherence: cyan"
            )
        if (maps["immutable_assignment_peak"] > 0).any():
            stage_names.append(
                "prototype retrieval: peak cyan | entropy red"
            )
    else:
        stage_names.extend((
            "KV applied read: green",
            "KV write core: orange",
            "committed target payload: violet",
        ))
    if topology_present:
        stage_names.append("topology holes: yellow | admitted read: magenta")
    persistent_present = bool(
        (maps["persistent_direct"] > 0).any()
        or (maps["persistent_retained"] > 0).any()
    )
    if persistent_present:
        stage_names.append(
            "persistent KV: direct orange | retained cyan | guarded red"
        )
    flow_ledger_present = bool(
        (maps["flow_transport_confidence"] > 0).any()
        or (maps["flow_ledger_support"] > 0).any()
    )
    if flow_ledger_present:
        stage_names.extend((
            "flow read: effective green | local transport orange",
            "flow ledger: appearance trust cyan | support violet",
        ))
    sink_present = bool((maps["sink_admission"] > 0).any())
    if sink_present:
        stage_names.extend((
            "962 sink: admission green | strength orange",
            "962 source similarity green | frame coverage violet",
            "962 target attention: peak cyan | entropy red",
            "962 selected ignition frame: 0 blue | 1 green | 2 orange",
        ))

    for first in range(0, len(groups), tiles_per_page):
        indices = list(range(first, min(first + tiles_per_page, len(groups))))
        page = Image.new(
            "RGB",
            (len(indices) * cell[0], len(stage_names) * (cell[1] + 38)),
            "white",
        )
        for column, latent_index in enumerate(indices):
            left, right = groups[latent_index]
            frame = frames[left]
            runtime, _, _, _ = render_views(frame, maps, latent_index)
            views = [
                runtime,
                color_overlay(frame, [
                    (maps["flow_background"][latent_index],
                     (45, 105, 245), 0.62),
                    (maps["flow_object"][latent_index],
                     (35, 205, 80), 0.82),
                ]),
                color_overlay(frame, [
                    (maps["flow_unknown"][latent_index],
                     (215, 55, 245), 0.62),
                    (maps["flow_boundary"][latent_index],
                     (255, 205, 25), 0.86),
                ]),
                color_overlay(frame, [(maps["flow_recovered"][latent_index],
                                       (245, 55, 45), 0.90)]),
                color_overlay(frame, [(maps["target_action"][latent_index],
                                       (255, 175, 25), 0.82)]),
                color_overlay(frame, [(maps["source_closure"][latent_index],
                                       (45, 105, 245), 0.75)]),
            ]
            if immutable_present:
                views.extend((
                    color_overlay(frame, [(
                        maps["immutable_owner"][latent_index],
                        (40, 205, 85), 0.76,
                    )]),
                    color_overlay(frame, [(
                        maps["immutable_support"][latent_index],
                        (255, 120, 20), 0.82,
                    )]),
                    color_overlay(frame, [(
                        maps["immutable_correction"][latent_index],
                        (155, 70, 230), 0.86,
                    )]),
                ))
                if (maps["immutable_subspace_coherence"] > 0).any():
                    views.append(color_overlay(frame, [(
                        maps["immutable_subspace_coherence"][latent_index],
                        (35, 210, 220), 0.82,
                    )]))
                if (maps["immutable_assignment_peak"] > 0).any():
                    views.append(color_overlay(frame, [
                        (maps["immutable_assignment_peak"][latent_index],
                         (35, 210, 220), 0.72),
                        (maps["immutable_assignment_entropy"][latent_index],
                         (240, 45, 50), 0.72),
                    ]))
            else:
                views.extend((
                    color_overlay(frame, [((
                        maps["kv_read"][latent_index] > 0
                    ).astype(np.float32), (40, 205, 85), 0.76)]),
                    color_overlay(frame, [(
                        maps["kv_write_core"][latent_index],
                        (255, 120, 20), 0.82,
                    )]),
                    color_overlay(frame, [(
                        maps["payload_commit"][latent_index],
                        (155, 70, 230), 0.74,
                    )]),
                ))
            if topology_present:
                views.append(color_overlay(
                    frame,
                    [
                        (maps["topology_holes"][latent_index],
                         (255, 220, 30), 0.72),
                        ((maps["topology_recovery"][latent_index] > 0).astype(np.float32),
                         (215, 55, 245), 0.92),
                    ],
                ))
            if persistent_present:
                views.append(color_overlay(
                    frame,
                    [
                        (maps["persistent_payload"][latent_index],
                         (155, 70, 230), 0.62),
                        (maps["persistent_direct"][latent_index],
                         (255, 120, 20), 0.82),
                        (maps["persistent_retained"][latent_index],
                         (35, 215, 230), 0.90),
                        (maps["persistent_guarded"][latent_index],
                         (240, 45, 50), 0.92),
                    ],
                ))
            if flow_ledger_present:
                views.append(color_overlay(
                    frame,
                    [
                        (maps["flow_transport_confidence"][latent_index],
                         (35, 205, 80), 0.78),
                        (maps["flow_local_transport_confidence"][latent_index],
                         (255, 135, 25), 0.76),
                    ],
                ))
            if sink_present:
                views.extend((
                    color_overlay(frame, [
                        (maps["sink_admission"][latent_index],
                         (35, 205, 80), 0.78),
                        (maps["sink_applied_strength"][latent_index],
                         (255, 135, 25), 0.80),
                    ]),
                    color_overlay(frame, [
                        ((maps["sink_source_similarity"][latent_index]
                          + 1.0) * 0.5, (35, 205, 80), 0.76),
                        (maps["sink_coverage"][latent_index],
                         (175, 65, 235), 0.78),
                    ]),
                    color_overlay(frame, [
                        (maps["sink_attention_peak"][latent_index],
                         (35, 210, 220), 0.78),
                        (maps["sink_attention_entropy"][latent_index],
                         (240, 45, 50), 0.74),
                    ]),
                    color_overlay(frame, [
                        ((maps["sink_selected_frame"][latent_index] == 0)
                         * maps["sink_admission"][latent_index],
                         (45, 105, 245), 0.82),
                        ((maps["sink_selected_frame"][latent_index] == 1)
                         * maps["sink_admission"][latent_index],
                         (35, 205, 80), 0.82),
                        ((maps["sink_selected_frame"][latent_index] == 2)
                         * maps["sink_admission"][latent_index],
                         (255, 135, 25), 0.82),
                    ]),
                ))
                views.append(color_overlay(
                    frame,
                    [
                        (maps["flow_ledger_appearance_trust"][latent_index],
                         (35, 210, 220), 0.78),
                        (maps["flow_ledger_support"][latent_index],
                         (175, 65, 235), 0.78),
                    ],
                ))
            subtitle = (
                f"L{latent_index:02d} B{latent_index // 3} "
                f"px={left}-{right - 1}"
            )
            for row, (view, stage_name) in enumerate(zip(views, stage_names)):
                tile = labeled(
                    np.asarray(fit(Image.fromarray(view), cell)),
                    stage_name, subtitle,
                )
                page.paste(
                    tile, (column * cell[0], row * (cell[1] + 38))
                )
        page.save(
            output_dir
            / f"inference_edit_regions_L{indices[0]:02d}_L{indices[-1]:02d}.png"
        )


def save_stats(
    maps: dict[str, np.ndarray],
    groups: list[tuple[int, int]],
    output_path: Path,
) -> None:
    fields = [
        "latent_frame", "block", "block_frame", "pixel_start",
        "pixel_end", "runtime_edit_fraction", "selected_fraction",
        "owner_fraction", "hand_fraction", "edit_hand_overlap_fraction",
        "target_action_mean", "source_closure_mean",
        "kv_read_fraction", "kv_read_mean", "kv_write_core_fraction",
        "payload_commit_fraction", "topology_hole_fraction",
        "topology_read_fraction", "topology_read_mean",
        "persistent_direct_fraction", "persistent_retained_fraction",
        "persistent_payload_fraction", "persistent_guarded_fraction",
        "flow_object_mean", "flow_background_mean",
        "flow_boundary_mean", "flow_unknown_mean",
        "flow_recovered_fraction", "flow_transport_confidence_mean",
        "flow_appearance_trust_mean",
        "flow_local_transport_confidence_mean",
        "flow_ledger_support_fraction", "flow_ledger_confidence_mean",
        "flow_ledger_appearance_trust_mean",
        "flow_ledger_local_transport_mean",
        "sink_admission_fraction", "sink_selected_frame_mean",
        "sink_source_similarity_mean", "sink_attention_entropy_mean",
        "sink_attention_peak_mean", "sink_coverage_mean",
        "sink_applied_strength_mean",
        "immutable_owner_mean", "immutable_support_mean",
        "immutable_correction_mean",
        "immutable_subspace_coherence_mean",
        "immutable_assignment_entropy_mean",
        "immutable_assignment_peak_mean",
        "immutable_assignment_margin_mean",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (left, right) in enumerate(groups):
            edit = maps["runtime_edit"][index] > 0
            hand = maps["hand"][index] > 0
            topology = maps["topology_recovery"][index]
            sink_admitted = maps["sink_admission"][index] > 0
            immutable_owned = maps["immutable_owner"][index] > 0

            def sink_read_mean(name: str, empty: float = 0.0) -> float:
                return (
                    float(maps[name][index][sink_admitted].mean())
                    if sink_admitted.any() else empty
                )

            def immutable_read_mean(
                name: str, empty: float = 0.0
            ) -> float:
                return (
                    float(maps[name][index][immutable_owned].mean())
                    if immutable_owned.any() else empty
                )

            writer.writerow({
                "latent_frame": index,
                "block": index // 3,
                "block_frame": index % 3,
                "pixel_start": left,
                "pixel_end": right - 1,
                "runtime_edit_fraction": float(edit.mean()),
                "selected_fraction": float(
                    maps["selected_posterior"][index].mean()
                ),
                "owner_fraction": float(maps["causal_owner"][index].mean()),
                "hand_fraction": float(hand.mean()),
                "edit_hand_overlap_fraction": float((edit & hand).mean()),
                "target_action_mean": float(maps["target_action"][index].mean()),
                "source_closure_mean": float(maps["source_closure"][index].mean()),
                "kv_read_fraction": float((maps["kv_read"][index] > 0).mean()),
                "kv_read_mean": float(maps["kv_read"][index].mean()),
                "kv_write_core_fraction": float(maps["kv_write_core"][index].mean()),
                "payload_commit_fraction": float(maps["payload_commit"][index].mean()),
                "topology_hole_fraction": float((maps["topology_holes"][index] > 0).mean()),
                "topology_read_fraction": float((topology > 0).mean()),
                "topology_read_mean": float(topology.mean()),
                "persistent_direct_fraction": float(
                    (maps["persistent_direct"][index] > 0).mean()
                ),
                "persistent_retained_fraction": float(
                    (maps["persistent_retained"][index] > 0).mean()
                ),
                "persistent_payload_fraction": float(
                    (maps["persistent_payload"][index] > 0).mean()
                ),
                "persistent_guarded_fraction": float(
                    (maps["persistent_guarded"][index] > 0).mean()
                ),
                "flow_object_mean": float(
                    maps["flow_object"][index].mean()
                ),
                "flow_background_mean": float(
                    maps["flow_background"][index].mean()
                ),
                "flow_boundary_mean": float(
                    maps["flow_boundary"][index].mean()
                ),
                "flow_unknown_mean": float(
                    maps["flow_unknown"][index].mean()
                ),
                "flow_recovered_fraction": float(
                    (maps["flow_recovered"][index] > 0).mean()
                ),
                "flow_transport_confidence_mean": float(
                    maps["flow_transport_confidence"][index].mean()
                ),
                "flow_appearance_trust_mean": float(
                    maps["flow_appearance_trust"][index].mean()
                ),
                "flow_local_transport_confidence_mean": float(
                    maps["flow_local_transport_confidence"][index].mean()
                ),
                "flow_ledger_support_fraction": float(
                    (maps["flow_ledger_support"][index] > 0).mean()
                ),
                "flow_ledger_confidence_mean": float(
                    maps["flow_ledger_confidence"][index].mean()
                ),
                "flow_ledger_appearance_trust_mean": float(
                    maps["flow_ledger_appearance_trust"][index].mean()
                ),
                "flow_ledger_local_transport_mean": float(
                    maps["flow_ledger_local_transport"][index].mean()
                ),
                "sink_admission_fraction": float(
                    (maps["sink_admission"][index] > 0).mean()
                ),
                "sink_selected_frame_mean": float(
                    sink_read_mean("sink_selected_frame", -1.0)
                ),
                "sink_source_similarity_mean": sink_read_mean(
                    "sink_source_similarity", -1.0
                ),
                "sink_attention_entropy_mean": sink_read_mean(
                    "sink_attention_entropy"
                ),
                "sink_attention_peak_mean": sink_read_mean(
                    "sink_attention_peak"
                ),
                "sink_coverage_mean": sink_read_mean(
                    "sink_coverage"
                ),
                "sink_applied_strength_mean": sink_read_mean(
                    "sink_applied_strength"
                ),
                "immutable_owner_mean": float(
                    maps["immutable_owner"][index].mean()
                ),
                "immutable_support_mean": float(
                    maps["immutable_support"][index].mean()
                ),
                "immutable_correction_mean": float(
                    maps["immutable_correction"][index].mean()
                ),
                "immutable_subspace_coherence_mean": float(
                    immutable_read_mean("immutable_subspace_coherence")
                ),
                "immutable_assignment_entropy_mean": float(
                    immutable_read_mean("immutable_assignment_entropy")
                ),
                "immutable_assignment_peak_mean": float(
                    immutable_read_mean("immutable_assignment_peak")
                ),
                "immutable_assignment_margin_mean": float(
                    immutable_read_mean("immutable_assignment_margin")
                ),
            })


def main() -> None:
    args = parse_args()
    if args.tiles_per_page <= 0:
        raise ValueError("tiles_per_page must be positive")
    run_dir = args.run_dir.resolve()
    video_path = args.video.resolve() if args.video else find_video(run_dir)
    roles_dir = (args.roles_dir or run_dir / "roles").resolve()
    output_dir = (
        args.output_dir or run_dir / "inference_edit_region_viz"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, rate = decode_video(video_path)
    maps, blocks = load_maps(
        roles_dir, run_dir / "run.log", args.write_threshold
    )
    groups = causal_groups(len(frames), maps["runtime_edit"].shape[0])
    save_video(
        frames, rate, maps, groups,
        output_dir / "inference-edit-regions.mp4",
    )
    save_contact_sheets(
        frames, maps, groups, output_dir, args.tiles_per_page
    )
    save_stats(maps, groups, output_dir / "inference_edit_regions.csv")
    (output_dir / "legend.txt").write_text(
        "No GT object mask is used.\n"
        "runtime edit = selected posterior OR causal owner support\n"
        "target route = factorized flow target-owned action\n"
        "source closure = locations forced to clean-source reconstruction\n"
        "immutable target-KV owner = automatic clean-source RGB flow owner; "
        "retrieval support and correction are read-only first-block target "
        "appearance diagnostics\n"
        "pose-preserving immutable mode constrains only the residual "
        "direction coherent across first-block prototypes; current chunk "
        "retains orthogonal pose, view, boundary, and occlusion structure\n"
        "prototype entropy/peak diagnose unstable soft retrieval and are "
        "never fed back into inference\n"
        "KV read = native history applied read strength > 0\n"
        "KV write core = transactional write confidence >= "
        f"{args.write_threshold:g}\n"
        "topology completion is read-only and never enlarges KV write\n"
        "persistent direct = current high-confidence target write\n"
        "persistent retained = previous trusted target-source residual "
        "rebased at a mutually matched current source address\n"
        "persistent total = direct OR retained; retained tokens are not "
        "counted as new writes\n"
        "persistent guarded = a current direct proposal was rejected "
        "because it regressed relative to the last trusted residual\n"
        "source-flow role evidence uses clean-source RGB flow; flow "
        "magnitude alone never defines an object\n"
        "flow recovered = weak counterfactual role support recovered only "
        "inside the transported hand-conditioned owner\n"
        "962 sink admission = automatic owner AND clean-source candidate "
        "AND flow-supported appearance/local-transport trust\n"
        "962 target attention is evaluated only inside per-frame "
        "clean-source top-k candidates from the immutable first block\n"
        "962 selected frame reports which frozen ignition view received "
        "the largest candidate-restricted target-query attention mass\n"
        "flow transport confidence = confidence of the residual injected "
        "at the current clean-source coordinate\n"
        "flow ledger support/confidence = last-frame state committed for "
        "transport into the next causal block\n",
        encoding="utf-8",
    )
    print(f"run={run_dir.name} blocks={blocks} latent_frames={len(groups)}")
    print(f"video={output_dir / 'inference-edit-regions.mp4'}")
    print(f"sheets={output_dir / 'inference_edit_regions_L*_L*.png'}")
    print(f"stats={output_dir / 'inference_edit_regions.csv'}")


if __name__ == "__main__":
    main()
