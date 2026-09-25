#!/usr/bin/env python3
"""Render strict, offline diagnostics for one StreamEdit case.

The tool consumes two existing, tensor-native diagnostic contracts:

* ``ours-responsibility-v1`` ``block_*.pt`` files, parsed by
  :mod:`visualize_ours_responsibility` without reconstructing missing fields.
* attention files below ``raw_attention/{cross,self}/step_*/layer_*.pt``.
  Each attention file is a dictionary containing ``raw_attention`` (or
  ``attention``) and a ``metadata`` dictionary.  Step/layer identifiers and
  all selections are read from metadata, never inferred from a heatmap.

Canonical cross-attention metadata uses ``tensor_layout='heads_query_key'``,
``query_spatial_shape=[T,H,W]``, ``latent_frame_indices``, and
``selected_key_index``.  A preselected-key tensor may instead use
``tensor_layout='heads_frames_height_width'``.

Canonical self-attention metadata uses ``tensor_layout='heads_query_key'``,
exactly three ``queries`` entries (each with ``name`` and ``index``),
``current_key_range=[start,end]``, and
``current_key_spatial_shape=[T,H,W]``.  Only current-chunk keys are reshaped;
all remaining/history keys are represented by mass statistics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


SCHEMA_VERSION = "single-case-diagnostics-v1"
ATTENTION_KINDS = ("cross", "self")
VELOCITY_VECTOR_FIELDS = (
    "v_src", "v_trg", "rho", "rho_safe", "rho_role", "v_controlled",
)
VELOCITY_UPDATES = (
    ("prediction_000", "T0"),
    ("prediction_001", "T1"),
    ("prediction_007", "Tm"),
)
STEP_LABELS = {0: "T0", 1: "T1", 7: "Tm"}
ROLE_COLORS = {
    "q_object": "#4c78a8",
    "q_interface": "#9c6ade",
    "q_hand": "#f5857a",
    "q_background": "#9d9da1",
}
ROLE_CMAPS = {
    name: LinearSegmentedColormap.from_list(
        f"single_case_{name}", ("#05070a", color)
    )
    for name, color in ROLE_COLORS.items()
}
PERMISSION_CMAPS = {
    "safe_residual_coefficient": "Blues",
    "full_residual_coefficient": "Blues",
    "effective_background_gated_coefficient": "Blues",
    "appearance_access": "Oranges",
    "reference_read_gate": "Purples",
    "reference_write_gate": "Purples",
}
PERMISSION_FIELDS = (
    "safe_residual_coefficient",
    "full_residual_coefficient",
    "effective_background_gated_coefficient",
    "appearance_access",
    "reference_read_gate",
    "reference_write_gate",
)


class SchemaError(ValueError):
    """Raised when an input cannot satisfy the explicit diagnostic contract."""


@dataclass(frozen=True)
class AttentionArtifact:
    kind: str
    step: int
    layer: int
    path: Path
    raw: torch.Tensor
    metadata: dict[str, Any]


@dataclass(frozen=True)
class QueryView:
    name: str
    index: int
    metadata: dict[str, Any]
    current_maps: np.ndarray  # [heads, time, height, width]
    mass: dict[str, Any]


def _load_responsibility_module():
    path = Path(__file__).with_name("visualize_ours_responsibility.py")
    if not path.is_file():
        raise ImportError(f"Required parser is missing: {path}")
    module_name = "_single_case_responsibility_parser"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import responsibility parser: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


responsibility_viz = _load_responsibility_module()
read_video = responsibility_viz.read_video


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one case from real responsibility and raw-attention tensors."
    )
    parser.add_argument("--responsibility-dir", type=Path, required=True)
    parser.add_argument("--attention-dir", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--edited-video", type=Path, required=True)
    parser.add_argument("--baseline-video", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--hand-role-input", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with torch < 2
        return torch.load(path, map_location="cpu")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _required(mapping: Any, key: str, scope: str) -> Any:
    if not isinstance(mapping, dict) or key not in mapping:
        raise SchemaError(f"{scope}: missing required field {key!r}")
    return mapping[key]


def _metadata_int(metadata: dict[str, Any], names: Iterable[str], scope: str) -> int:
    for name in names:
        if name in metadata:
            value = np.asarray(_jsonable(metadata[name]))
            if value.size != 1:
                raise SchemaError(f"{scope}.{name} must be scalar")
            number = value.reshape(-1)[0]
            if isinstance(number, (bool, np.bool_)) or int(number) != float(number):
                raise SchemaError(f"{scope}.{name} must be an integer")
            return int(number)
    raise SchemaError(f"{scope}: missing one of {tuple(names)}")


def _shape(value: Any, name: str, dimensions: tuple[int, ...]) -> tuple[int, ...]:
    array = np.asarray(_jsonable(value))
    if array.ndim != 1 or len(array) not in dimensions:
        raise SchemaError(f"{name} must have length in {dimensions}, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise SchemaError(f"{name} must be numeric")
    result = tuple(int(item) for item in array)
    if any(item <= 0 for item in result) or any(float(a) != b for a, b in zip(array, result)):
        raise SchemaError(f"{name} must contain positive integers")
    return result


def _as_attention_tensor(value: Any, path: Path) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise SchemaError(f"{path}: raw_attention must be a torch.Tensor")
    tensor = value.detach().cpu()
    if not tensor.is_floating_point():
        raise SchemaError(f"{path}: raw_attention must be floating point")
    tensor = tensor.float()
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
        raise SchemaError(f"{path}: raw_attention must be non-empty and finite")
    if bool((tensor < -1e-7).any()):
        raise SchemaError(f"{path}: attention weights must be non-negative")
    return tensor.clamp_min(0.0)


def discover_attention_artifacts(attention_dir: Path) -> list[AttentionArtifact]:
    base = attention_dir / "raw_attention" if (attention_dir / "raw_attention").is_dir() else attention_dir
    if not base.is_dir():
        raise FileNotFoundError(f"Attention directory does not exist: {attention_dir}")
    artifacts: list[AttentionArtifact] = []
    seen: set[tuple[str, int, int]] = set()
    for kind in ATTENTION_KINDS:
        kind_dir = base / kind
        if not kind_dir.is_dir():
            raise FileNotFoundError(f"Missing raw attention directory: {kind_dir}")
        files = sorted(kind_dir.glob("**/*.pt"))
        if not files:
            raise FileNotFoundError(f"No .pt attention artifacts under {kind_dir}")
        for path in files:
            payload = _torch_load(path)
            if not isinstance(payload, dict):
                raise SchemaError(f"{path}: attention artifact must be a dictionary")
            metadata = _required(payload, "metadata", str(path))
            if not isinstance(metadata, dict):
                raise SchemaError(f"{path}: metadata must be a dictionary")
            step = _metadata_int(metadata, ("step_index",), f"{path}:metadata")
            layer = _metadata_int(metadata, ("layer_index",), f"{path}:metadata")
            if step < 0 or layer < 0:
                raise SchemaError(f"{path}: step and layer must be non-negative")
            key = (kind, step, layer)
            if key in seen:
                raise SchemaError(f"Duplicate attention artifact for {key}")
            seen.add(key)
            tensor_value = _required(payload, "raw_attention", str(path))
            artifacts.append(
                AttentionArtifact(kind, step, layer, path.resolve(), _as_attention_tensor(tensor_value, path), metadata)
            )
    return sorted(artifacts, key=lambda item: (item.step, item.kind, item.layer))


def _layout(metadata: dict[str, Any], path: Path) -> str:
    value = metadata.get("tensor_layout")
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{path}: metadata.tensor_layout is required")
    return value.lower().replace("-", "_")


def _select_batch(raw: torch.Tensor, metadata: dict[str, Any], path: Path) -> torch.Tensor:
    batch = _metadata_int(metadata, ("batch_index",), f"{path}:metadata") if "batch_index" in metadata else 0
    if batch < 0 or batch >= raw.shape[0]:
        raise SchemaError(f"{path}: batch_index={batch} outside [0,{raw.shape[0]})")
    return raw[batch]


def _latent_position(metadata: dict[str, Any], temporal_size: int, selected: int, path: Path) -> int:
    indices = np.asarray(_jsonable(_required(metadata, "latent_frame_indices", str(path))))
    if indices.ndim != 1 or len(indices) != temporal_size:
        raise SchemaError(f"{path}: latent_frame_indices must have length {temporal_size}")
    if not np.issubdtype(indices.dtype, np.integer):
        raise SchemaError(f"{path}: latent_frame_indices must be integers")
    matches = np.flatnonzero(indices.astype(np.int64) == selected)
    if len(matches) != 1:
        raise SchemaError(f"{path}: selected latent frame {selected} is not represented exactly once")
    return int(matches[0])


def cross_attention_maps(artifact: AttentionArtifact, selected_latent_frame: int) -> np.ndarray:
    raw, metadata, path = artifact.raw, artifact.metadata, artifact.path
    layout = _layout(metadata, path)
    if layout in {"batch_heads_query_key", "bhqk"}:
        if raw.ndim != 4:
            raise SchemaError(f"{path}: {layout} requires [B,H,Q,K], got {tuple(raw.shape)}")
        raw = _select_batch(raw, metadata, path)
        layout = "heads_query_key"
    if layout in {"heads_query_key", "hqk"}:
        if raw.ndim != 3:
            raise SchemaError(f"{path}: {layout} requires [H,Q,K], got {tuple(raw.shape)}")
        key_index = _metadata_int(metadata, ("selected_key_index", "key_index"), f"{path}:metadata")
        if key_index < 0 or key_index >= raw.shape[2]:
            raise SchemaError(f"{path}: selected_key_index={key_index} outside key dimension")
        spatial = _shape(_required(metadata, "query_spatial_shape", str(path)), "query_spatial_shape", (3,))
        if math.prod(spatial) != raw.shape[1]:
            raise SchemaError(f"{path}: query_spatial_shape product does not equal Q={raw.shape[1]}")
        maps = raw[:, :, key_index].reshape(raw.shape[0], *spatial)
    elif layout in {"heads_frames_height_width", "hthw"}:
        if raw.ndim != 4:
            raise SchemaError(f"{path}: {layout} requires [H,T,Y,X], got {tuple(raw.shape)}")
        maps = raw
        spatial = tuple(int(item) for item in maps.shape[1:])
    elif layout in {"heads_height_width", "hhw"}:
        if raw.ndim != 3:
            raise SchemaError(f"{path}: {layout} requires [H,Y,X], got {tuple(raw.shape)}")
        maps = raw[:, None]
        spatial = (1, int(raw.shape[1]), int(raw.shape[2]))
    else:
        raise SchemaError(f"{path}: unsupported cross tensor_layout={layout!r}")
    position = _latent_position(metadata, spatial[0], selected_latent_frame, path)
    result = maps[:, position].numpy()
    if result.ndim != 3:
        raise SchemaError(f"{path}: cross map did not resolve to [heads,height,width]")
    return result


def _query_entries(metadata: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    queries = _required(metadata, "queries", str(path))
    if isinstance(queries, (list, tuple)):
        entries = list(queries)
    else:
        raise SchemaError(f"{path}: metadata.queries must be a list")
    if len(entries) != 3 or not all(isinstance(entry, dict) for entry in entries):
        raise SchemaError(f"{path}: metadata.queries must contain exactly three entries")
    names: set[str] = set()
    result = []
    for entry in entries:
        name = str(_required(entry, "name", f"{path}:query"))
        index = _metadata_int(entry, ("index", "query_index"), f"{path}:query:{name}")
        if not name or name in names:
            raise SchemaError(f"{path}: query names must be unique and non-empty")
        names.add(name)
        result.append({**entry, "name": name, "index": index})
    return result


def _range(value: Any, name: str, upper: int) -> tuple[int, int]:
    array = np.asarray(_jsonable(value))
    if array.shape != (2,):
        raise SchemaError(f"{name} must be [start,end]")
    start, end = int(array[0]), int(array[1])
    if float(array[0]) != start or float(array[1]) != end or start < 0 or end <= start or end > upper:
        raise SchemaError(f"{name}={array.tolist()} is outside [0,{upper}] or empty")
    return start, end


def _history_ranges(metadata: dict[str, Any], current: tuple[int, int], keys: int, path: Path) -> list[tuple[str, int, int]]:
    value = metadata.get("history_key_ranges")
    if value is None:
        raise SchemaError(f"{path}: metadata.history_key_ranges is required")
    if not isinstance(value, (list, tuple)):
        raise SchemaError(f"{path}: history_key_ranges must be a list")
    ranges = []
    occupied = np.zeros(keys, dtype=bool)
    occupied[current[0]:current[1]] = True
    for position, entry in enumerate(value):
        if isinstance(entry, dict):
            name = str(entry.get("name", f"history_{position}"))
            pair = entry.get("range", [entry.get("start"), entry.get("end")])
        else:
            name, pair = f"history_{position}", entry
        start, end = _range(pair, f"{path}:history_key_ranges[{position}]", keys)
        if occupied[start:end].any():
            raise SchemaError(f"{path}: history range {name!r} overlaps current/another history range")
        occupied[start:end] = True
        ranges.append((name, start, end))
    outside = np.ones(keys, dtype=bool)
    outside[current[0]:current[1]] = False
    if not np.array_equal(occupied, np.ones(keys, dtype=bool)):
        missing = np.flatnonzero(~occupied)
        raise SchemaError(f"{path}: key ranges do not cover all keys; first uncovered={int(missing[0])}")
    return ranges


def _named_ranges(
    metadata: dict[str, Any], field: str, keys: int, path: Path
) -> list[tuple[str, int, int]] | None:
    """Parse an optional, explicitly recorded range collection.

    These ranges are reporting coordinates only.  In particular, this helper
    never derives a range from tensor lengths or reshapes history keys.
    """
    if field not in metadata:
        return None
    value = metadata[field]
    if isinstance(value, dict):
        entries = [
            ({"name": name, **item} if isinstance(item, dict) else {"name": name, "range": item})
            for name, item in value.items()
        ]
    elif isinstance(value, (list, tuple)):
        entries = list(value)
    else:
        raise SchemaError(f"{path}: metadata.{field} must be a list or dictionary")
    ranges = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SchemaError(f"{path}: metadata.{field}[{position}] must be a dictionary")
        name = str(entry.get("name", f"{field}_{position:03d}"))
        pair = entry.get("range", [entry.get("start"), entry.get("end")])
        start, end = _range(pair, f"{path}:metadata.{field}[{position}]", keys)
        ranges.append((name, start, end))
    return ranges


def _mass_report(
    values: torch.Tensor,
    ranges: list[tuple[str, int, int]] | None,
    *,
    unavailable_reason: str,
) -> dict[str, Any]:
    if not ranges:
        return {"available": False, "reason": unavailable_reason}
    entries = []
    totals = torch.zeros(values.shape[0], dtype=values.dtype)
    for name, start, end in ranges:
        mass = values[:, start:end].sum(dim=-1)
        totals += mass
        entries.append({
            "name": name,
            "range": [start, end],
            "mass_per_head": mass.numpy().tolist(),
        })
    return {
        "available": True,
        "ranges": entries,
        "total_mass_per_head": totals.numpy().tolist(),
        "mean_total_mass": float(totals.mean()),
    }


def self_attention_views(artifact: AttentionArtifact) -> list[QueryView]:
    raw, metadata, path = artifact.raw, artifact.metadata, artifact.path
    layout = _layout(metadata, path)
    if layout in {"batch_heads_query_key", "bhqk"}:
        if raw.ndim != 4:
            raise SchemaError(f"{path}: {layout} requires [B,H,Q,K], got {tuple(raw.shape)}")
        raw = _select_batch(raw, metadata, path)
        layout = "heads_query_key"
    if layout not in {"heads_query_key", "hqk"} or raw.ndim != 3:
        raise SchemaError(f"{path}: self attention requires tensor_layout=heads_query_key and [H,Q,K]")
    range_value = _required(metadata, "current_key_range", str(path))
    current = _range(range_value, f"{path}:current_key_range", raw.shape[2])
    shape_value = _required(metadata, "current_key_spatial_shape", str(path))
    spatial = _shape(shape_value, "current_key_spatial_shape", (3,))
    if math.prod(spatial) != current[1] - current[0]:
        raise SchemaError(f"{path}: current_key_spatial_shape product does not match current key range")
    history = _history_ranges(metadata, current, raw.shape[2], path)
    frame_ranges = _named_ranges(metadata, "history_frame_ranges", raw.shape[2], path)
    chunk_ranges = _named_ranges(metadata, "history_chunk_ranges", raw.shape[2], path)
    views = []
    for entry in _query_entries(metadata, path):
        query_index = int(entry["index"])
        if query_index < 0 or query_index >= raw.shape[1]:
            raise SchemaError(f"{path}: query index {query_index} outside Q={raw.shape[1]}")
        values = raw[:, query_index]
        current_maps = values[:, current[0]:current[1]].reshape(raw.shape[0], *spatial).numpy()
        target_ranges = [item for item in history if "target_history" in item[0].lower()]
        source_ranges = [item for item in history if "source_history" in item[0].lower()]
        mass = {
            "current_chunk": _mass_report(
                values, [("current_chunk", *current)],
                unavailable_reason="current_key_range is unavailable",
            ),
            "target_history": _mass_report(
                values, target_ranges,
                unavailable_reason="no target_history segment exists in metadata.history_key_ranges",
            ),
            "source_history": _mass_report(
                values, source_ranges,
                unavailable_reason="no source_history segment exists in metadata.history_key_ranges",
            ),
            "previous_chunks": _mass_report(
                values, chunk_ranges,
                unavailable_reason="metadata.history_chunk_ranges is absent or empty",
            ),
            "history_frames": _mass_report(
                values, frame_ranges,
                unavailable_reason="metadata.history_frame_ranges is absent or empty",
            ),
            "all_declared_history_segments": _mass_report(
                values, history,
                unavailable_reason="metadata.history_key_ranges is empty",
            ),
            "spatial_visualization_generated": False,
            "reason": "history keys are reported only as exact range sums and are never spatially reshaped",
        }
        views.append(QueryView(str(entry["name"]), query_index, _jsonable(entry), current_maps, mass))
    return views


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "query"


def _temporal_strip(value: np.ndarray) -> np.ndarray:
    if value.ndim == 2:
        return value
    if value.ndim != 3:
        raise ValueError(f"Expected [T,H,W] or [H,W], got {value.shape}")
    return np.concatenate([frame for frame in value], axis=1)


def _normalize_map(value: np.ndarray) -> np.ndarray:
    minimum, maximum = float(value.min()), float(value.max())
    if maximum <= minimum:
        return np.zeros_like(value, dtype=np.float32)
    return ((value - minimum) / (maximum - minimum)).astype(np.float32)


def _colorize(value: np.ndarray, vmin: float, vmax: float, cmap: Any) -> np.ndarray:
    if vmax <= vmin:
        vmax = vmin + 1e-12
    normalized = np.clip((value.astype(np.float64) - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = matplotlib.colormaps.get_cmap(cmap)(normalized, bytes=True)
    return np.asarray(rgba[..., :3], dtype=np.uint8)


def _fit_nearest(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    shape = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    return image.resize(shape, Image.Resampling.NEAREST)


def _save_heatmap(
    path: Path,
    value: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    *,
    cmap: Any = "turbo",
) -> None:
    """Write an RGB heatmap with a colorbar without creating a pyplot figure.

    Thousands of per-head figures are required for this diagnostic.  Pillow
    keeps that export tractable while the color values still come directly
    from Matplotlib's named scientific colormap and the declared scale.
    """
    strip = _temporal_strip(value)
    mapped = Image.fromarray(_colorize(strip, vmin, vmax, cmap), mode="RGB")
    mapped = _fit_nearest(mapped, 600, 410)
    canvas = Image.new("RGB", (760, 500), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(17)
    label_font = _font(13)
    left = 22 + (600 - mapped.width) // 2
    top = 55 + (410 - mapped.height) // 2
    canvas.paste(mapped, (left, top))
    draw.text((22, 18), title, fill="black", font=title_font)
    gradient = np.linspace(vmax, vmin, 256, dtype=np.float32)[:, None]
    bar = Image.fromarray(_colorize(gradient, vmin, vmax, cmap), mode="RGB")
    bar = bar.resize((28, 360), Image.Resampling.NEAREST)
    canvas.paste(bar, (645, 78))
    draw.rectangle((644, 77, 673, 438), outline="black", width=1)
    draw.text((682, 69), f"{vmax:.4g}", fill="black", font=label_font)
    draw.text((682, 425), f"{vmin:.4g}", fill="black", font=label_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _save_montage(
    path: Path,
    maps: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    *,
    cmap: Any = "turbo",
) -> None:
    values = [maps[index] for index in range(len(maps))]
    labels = [f"head {index:02d}" for index in range(len(maps))]
    values.extend((maps.mean(axis=0), maps.std(axis=0)))
    labels.extend(("mean", "std"))
    columns = min(4, len(values))
    rows = int(math.ceil(len(values) / columns))
    tile_width, tile_height = 300, 225
    header, footer, bar_width = 52, 28, 92
    canvas = Image.new(
        "RGB",
        (columns * tile_width + bar_width, header + rows * tile_height + footer),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 14), title, fill="black", font=_font(17))
    for position, (value, label) in enumerate(zip(values, labels)):
        row, column = divmod(position, columns)
        tile = Image.fromarray(
            _colorize(_temporal_strip(value), vmin, vmax, cmap), mode="RGB"
        )
        tile = _fit_nearest(tile, tile_width - 16, tile_height - 38)
        x0 = column * tile_width + (tile_width - tile.width) // 2
        y0 = header + row * tile_height + 30 + (tile_height - 38 - tile.height) // 2
        canvas.paste(tile, (x0, y0))
        draw.text(
            (column * tile_width + 10, header + row * tile_height + 5),
            label,
            fill="black",
            font=_font(13),
        )
    gradient = np.linspace(vmax, vmin, 256, dtype=np.float32)[:, None]
    bar_height = min(420, rows * tile_height - 30)
    bar = Image.fromarray(_colorize(gradient, vmin, vmax, cmap), mode="RGB")
    bar = bar.resize((28, bar_height), Image.Resampling.NEAREST)
    bar_x, bar_y = columns * tile_width + 18, header + 20
    canvas.paste(bar, (bar_x, bar_y))
    draw.rectangle((bar_x - 1, bar_y - 1, bar_x + 28, bar_y + bar_height), outline="black", width=1)
    draw.text((bar_x + 34, bar_y - 8), f"{vmax:.4g}", fill="black", font=_font(12))
    draw.text((bar_x + 34, bar_y + bar_height - 12), f"{vmin:.4g}", fill="black", font=_font(12))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _render_attention_bundle(
    output_dir: Path,
    maps: np.ndarray,
    title: str,
    raw_scale: tuple[float, float],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_scale"
    normalized_dir = output_dir / "normalized_for_display"
    raw_dir.mkdir(exist_ok=True)
    normalized_dir.mkdir(exist_ok=True)
    summaries = {"mean": maps.mean(axis=0), "std": maps.std(axis=0)}
    raw_min, raw_max = raw_scale
    # Mean and std are first-class raw panels, so the layer-shared scale must
    # cover them too (std can be below every individual attention weight).
    raw_min = min(raw_min, *(float(value.min()) for value in summaries.values()))
    raw_max = max(raw_max, *(float(value.max()) for value in summaries.values()))
    if raw_max <= raw_min:
        raw_max = raw_min + 1e-12
    for index, value in enumerate(maps):
        _save_heatmap(raw_dir / f"head_{index:02d}.png", value, f"{title} head {index}", raw_min, raw_max)
        _save_heatmap(normalized_dir / f"head_{index:02d}.png", _normalize_map(value), f"{title} head {index} normalized", 0.0, 1.0)
    for name, value in summaries.items():
        filename = f"head_{name}.png"
        _save_heatmap(raw_dir / filename, value, f"{title} head {name}", raw_min, raw_max)
        _save_heatmap(normalized_dir / filename, _normalize_map(value), f"{title} head {name} normalized", 0.0, 1.0)
    _save_montage(raw_dir / "montage_all_heads.png", maps, f"{title} raw scale", raw_min, raw_max)
    normalized = np.stack([_normalize_map(value) for value in maps])
    _save_montage(normalized_dir / "montage_all_heads.png", normalized, f"{title} normalized", 0.0, 1.0)
    return {
        "head_count": int(len(maps)),
        "map_shape": list(maps.shape[1:]),
        "raw_shared_scale": [raw_min, raw_max],
        "normalized_color_scale": [0.0, 1.0],
        "raw_statistics": _distribution(maps),
    }


def render_attention(
    artifacts: list[AttentionArtifact], output_dir: Path, selected_case: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_latent_frame = int(selected_case["global_latent_frame"])
    metadata: dict[str, list[dict[str, Any]]] = {kind: [] for kind in ATTENTION_KINDS}
    metrics: dict[str, list[dict[str, Any]]] = {kind: [] for kind in ATTENTION_KINDS}
    for artifact in artifacts:
        if artifact.step not in STEP_LABELS:
            raise SchemaError(
                f"{artifact.path}: unexpected diagnostic step {artifact.step}; expected 0, 1, or 7"
            )
        artifact_block = artifact.metadata.get("block_index")
        if artifact_block is not None and int(artifact_block) != int(selected_case["block_index"]):
            raise SchemaError(f"{artifact.path}: attention block_index does not match selection")
        layer_dir = output_dir / f"{artifact.kind}_attention" / f"step_{artifact.step:03d}" / f"layer_{artifact.layer:02d}"
        layer_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(artifact.path, layer_dir / "raw_attention.pt")
        if artifact.kind == "cross":
            maps = cross_attention_maps(artifact, selected_latent_frame)
            scale = (float(maps.min()), float(maps.max()))
            rendered = _render_attention_bundle(layer_dir, maps, f"cross S{artifact.step} L{artifact.layer}", scale)
            entry = {
                "step": artifact.step,
                "update_label": STEP_LABELS[artifact.step],
                "layer": artifact.layer,
                "source": str(artifact.path),
                "artifact_metadata": _jsonable(artifact.metadata),
                **rendered,
            }
            metadata["cross"].append(entry)
            metrics["cross"].append({"step": artifact.step, "layer": artifact.layer, **rendered["raw_statistics"]})
        else:
            views = self_attention_views(artifact)
            expected_queries = selected_case["queries"]
            if {view.name for view in views} != set(expected_queries):
                raise SchemaError(
                    f"{artifact.path}: query names must be interface_query, object_query, and hand_query"
                )
            for view in views:
                original = _required(
                    view.metadata, "original_query_index", f"{artifact.path}:query:{view.name}"
                )
                if int(original) != int(expected_queries[view.name]["query_index"]):
                    raise SchemaError(
                        f"{artifact.path}: {view.name} original_query_index does not match selection"
                    )
            scale_values = []
            for view in views:
                scale_values.extend((
                    view.current_maps,
                    view.current_maps.mean(axis=0),
                    view.current_maps.std(axis=0),
                ))
            raw_min = min(float(value.min()) for value in scale_values)
            raw_max = max(float(value.max()) for value in scale_values)
            query_entries = []
            for view in views:
                query_dir = layer_dir / f"query_{_safe_name(view.name)}"
                rendered = _render_attention_bundle(
                    query_dir,
                    view.current_maps,
                    f"self S{artifact.step} L{artifact.layer} {view.name}",
                    (raw_min, raw_max),
                )
                history = view.mass
                _write_json(query_dir / "history_mass.json", history)
                query_entries.append({
                    "name": view.name,
                    "index": view.index,
                    "query_metadata": view.metadata,
                    "history": history,
                    **rendered,
                })
                metrics["self"].append({
                    "step": artifact.step,
                    "layer": artifact.layer,
                    "query": view.name,
                    "query_index": view.index,
                    "current_attention": rendered["raw_statistics"],
                    "current_chunk_mass_mean": history["current_chunk"]["mean_total_mass"],
                    "target_history_mass_mean": (
                        history["target_history"].get("mean_total_mass")
                    ),
                    "source_history_mass_mean": (
                        history["source_history"].get("mean_total_mass")
                    ),
                    "previous_chunk_mass_mean": (
                        history["previous_chunks"].get("mean_total_mass")
                    ),
                })
            entry = {
                "step": artifact.step,
                "update_label": STEP_LABELS[artifact.step],
                "layer": artifact.layer,
                "source": str(artifact.path),
                "artifact_metadata": _jsonable(artifact.metadata),
                "raw_shared_scale_across_three_queries": [raw_min, raw_max],
                "queries": query_entries,
            }
            metadata["self"].append(entry)
        _write_json(layer_dir / "visualization_metadata.json", entry)
    return metadata, metrics


def _distribution(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not len(array):
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def load_selection(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing selection JSON: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SchemaError(f"{path}: invalid selection JSON: {error}") from error
    if not isinstance(payload, dict):
        raise SchemaError(f"{path}: selection must be a JSON object")
    for name in ("block_index", "global_latent_frame", "local_frame"):
        value = _required(payload, name, str(path))
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SchemaError(f"{path}: {name} must be a non-negative integer")
    token_grid = _shape(_required(payload, "token_grid", str(path)), "selection.token_grid", (2,))
    queries = _required(payload, "queries", str(path))
    if not isinstance(queries, dict):
        raise SchemaError(f"{path}: queries must be a dictionary")
    for name in ("interface_query", "object_query", "hand_query"):
        query = _required(queries, name, f"{path}:queries")
        if not isinstance(query, dict):
            raise SchemaError(f"{path}: queries.{name} must be a dictionary")
        for field in ("query_index", "local_frame", "row", "column"):
            value = _required(query, field, f"{path}:queries.{name}")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SchemaError(f"{path}: queries.{name}.{field} must be a non-negative integer")
        row, column = query["row"], query["column"]
        if row >= token_grid[0] or column >= token_grid[1]:
            raise SchemaError(f"{path}: queries.{name} coordinate lies outside token_grid")
        expected = query["local_frame"] * math.prod(token_grid) + row * token_grid[1] + column
        if query["query_index"] != expected:
            raise SchemaError(
                f"{path}: queries.{name}.query_index={query['query_index']} does not match "
                f"the declared local_frame/row/column ({expected})"
            )
    return payload


def load_run_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing run manifest: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SchemaError(f"{path}: invalid run manifest JSON: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != "ours-single-case-run-manifest-v1":
        raise SchemaError(f"{path}: unsupported run manifest schema")
    commit = _required(payload, "git_commit", str(path))
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", commit):
        raise SchemaError(f"{path}: git_commit must be a full hexadecimal commit id")
    case = _required(payload, "case", str(path))
    if not isinstance(case, dict):
        raise SchemaError(f"{path}: case must be a dictionary")
    for name in (
        "source_video", "hand_mask_video", "source_prompt", "target_prompt",
        "source_phrase", "target_phrase", "checkpoint_path", "config_path",
        "wan_models_root",
    ):
        if not isinstance(_required(case, name, f"{path}:case"), str) or not case[name]:
            raise SchemaError(f"{path}: case.{name} must be a non-empty string")
    for name, expected in (("seed", 0), ("steps", 15), ("rollout_chunk_size", 21)):
        value = _required(case, name, f"{path}:case")
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise SchemaError(f"{path}: case.{name} must be {expected}, got {value!r}")
    for phase in ("baseline", "capture"):
        record = _required(payload, phase, str(path))
        if not isinstance(record, dict):
            raise SchemaError(f"{path}: {phase} must be a dictionary")
        for name in ("resolved_config", "command"):
            if not isinstance(_required(record, name, f"{path}:{phase}"), str) or not record[name]:
                raise SchemaError(f"{path}: {phase}.{name} must be a non-empty string")
    expected_updates = {
        label: int(prediction.removeprefix("prediction_"))
        for prediction, label in VELOCITY_UPDATES
    }
    if payload.get("diagnostic_updates") != expected_updates:
        raise SchemaError(
            f"{path}: diagnostic_updates must be {expected_updates}, got {payload.get('diagnostic_updates')!r}"
        )
    return payload


def load_temporal_groups(path: Path) -> np.ndarray:
    groups = responsibility_viz.load_temporal_groups(path)
    if len(groups) == 0 or int(groups[0, 0]) != 0:
        raise SchemaError("causal_temporal_groups must start at pixel frame 0")
    if len(groups) > 1 and not np.array_equal(groups[1:, 0], groups[:-1, 1]):
        raise SchemaError("causal_temporal_groups must be contiguous and non-overlapping")
    return groups


def selected_pixel_frame(
    groups: np.ndarray, latent_frame: int, videos: dict[str, Any]
) -> tuple[int, tuple[int, int]]:
    if latent_frame < 0 or latent_frame >= len(groups):
        raise SchemaError(
            f"Selected latent frame {latent_frame} is outside {len(groups)} causal_temporal_groups"
        )
    expected_frames = int(groups[-1, 1])
    for name, video in videos.items():
        if len(video.frames) != expected_frames:
            raise SchemaError(
                f"{name} has {len(video.frames)} frames, but causal_temporal_groups exactly cover "
                f"{expected_frames} frames"
            )
    start, end = (int(value) for value in groups[latent_frame])
    return responsibility_viz.representative_pixel_frame(groups, latent_frame), (start, end)


def select_responsibility(
    responsibility_dir: Path, selected_case: dict[str, Any]
) -> tuple[list[Any], Any, int, int, Any]:
    selected_block = int(selected_case["block_index"])
    selected_latent_frame = int(selected_case["global_latent_frame"])
    blocks = responsibility_viz.load_blocks(responsibility_dir)
    positions = [index for index, block in enumerate(blocks) if block.index == selected_block]
    if len(positions) != 1:
        raise SchemaError(f"selected block {selected_block} is not represented exactly once")
    block_position = positions[0]
    block = blocks[block_position]
    matches = np.flatnonzero(block.latent_frame_indices == selected_latent_frame)
    if len(matches) != 1:
        raise SchemaError(
            f"selected latent frame {selected_latent_frame} is not represented exactly once in block {selected_block}"
        )
    local_frame = int(matches[0])
    if local_frame != int(selected_case["local_frame"]):
        raise SchemaError(
            f"selection local_frame={selected_case['local_frame']} does not match responsibility artifact {local_frame}"
        )
    if tuple(selected_case["token_grid"]) != tuple(block.spatial_shape):
        raise SchemaError(
            f"selection token_grid={selected_case['token_grid']} does not match responsibility grid {block.spatial_shape}"
        )
    interface = selected_case["queries"]["interface_query"]
    if int(interface["local_frame"]) != local_frame:
        raise SchemaError("selection interface_query.local_frame does not match selected frame")
    row, column = int(interface["row"]), int(interface["column"])
    candidate = (
        block.maps["valid_token"][local_frame]
        & block.maps["connected_object_support"][local_frame]
        & (block.maps[block.hand_field][local_frame] > 0)
    )
    if not bool(candidate[row, column]):
        raise SchemaError("selection interface token is not supported by real valid/connected/hand evidence")
    score = float(block.maps["q_interface"][local_frame, row, column])
    selection = responsibility_viz.TokenSelection(block_position, local_frame, row, column, score)
    return blocks, block, block_position, local_frame, selection


def _map_grid(path: Path, panels: list[tuple[str, np.ndarray]], title: str, *, fixed_probability: bool = True) -> None:
    columns = min(4, len(panels))
    rows = int(math.ceil(len(panels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.25 * rows), squeeze=False)
    for axis, (label, value) in zip(axes.flat, panels):
        vmin, vmax = (0.0, 1.0) if fixed_probability else (float(value.min()), float(value.max()))
        if vmax <= vmin:
            vmax = vmin + 1e-12
        image = axis.imshow(
            value,
            cmap=PERMISSION_CMAPS.get(label, "turbo"),
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        axis.set_title(label, fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes.flat[len(panels):]:
        axis.axis("off")
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _soft_role_composite(block: Any, local_frame: int) -> np.ndarray:
    roles = np.stack([
        block.maps[name][local_frame] for name in responsibility_viz.ROLE_FIELDS
    ], axis=-1)
    dominant = roles.argmax(axis=-1)
    confidence = roles.max(axis=-1)
    palette = np.asarray([
        tuple(int(ROLE_COLORS[name][offset:offset + 2], 16) / 255.0 for offset in (1, 3, 5))
        for name in responsibility_viz.ROLE_FIELDS
    ], dtype=np.float32)
    brightness = 0.25 + 0.75 * confidence
    composite = palette[dominant] * brightness[..., None]
    composite[~block.maps["valid_token"][local_frame]] = 0.0
    return np.clip(composite, 0.0, 1.0)


def render_responsibility(block: Any, local_frame: int, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    composite = _soft_role_composite(block, local_frame)
    panels = [("role composite", composite)] + [
        (name, block.maps[name][local_frame]) for name in responsibility_viz.ROLE_FIELDS
    ]
    figure, axes = plt.subplots(1, 5, figsize=(17, 3.3))
    axes[0].imshow(composite, interpolation="nearest")
    axes[0].set_title("role composite")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for axis, (name, value) in zip(axes[1:], panels[1:]):
        image = axis.imshow(value, cmap=ROLE_CMAPS[name], vmin=0.0, vmax=1.0, interpolation="nearest")
        axis.set_title(name)
        axis.set_xticks([]); axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"Responsibility B{block.index} latent {int(block.latent_frame_indices[local_frame])}")
    figure.tight_layout()
    responsibility_png = output_dir / "responsibility_overview.png"
    figure.savefig(responsibility_png, dpi=170, bbox_inches="tight")
    plt.close(figure)
    composite_png = output_dir / "role_composite.png"
    Image.fromarray(np.clip(composite * 255.0, 0, 255).astype(np.uint8)).save(composite_png)
    composite_named_png = output_dir / "composite_soft_roles.png"
    Image.fromarray(np.clip(composite * 255.0, 0, 255).astype(np.uint8)).save(composite_named_png)
    evidence_panels = [
        ("Semantic attention", "source_attention", block.maps["source_attention"][local_frame]),
        ("Hand occupancy", "hand_occupancy", block.maps[block.hand_field][local_frame]),
        ("Hand proximity", "hand_proximity", block.maps["hand_proximity"][local_frame]),
        ("Temporal match confidence", "temporal_confidence", block.maps["temporal_confidence"][local_frame]),
        ("Temporal propagation", "temporal_posterior", block.maps["temporal_posterior"][local_frame]),
        ("Source-grounded prior p_src", "object_posterior_prior", block.maps["object_posterior_prior"][local_frame]),
        ("First branch response", "field_observation", block.maps["field_observation"][local_frame]),
        ("Refined object support p", "object_posterior", block.maps["object_posterior"][local_frame]),
        ("Connected support S", "connected_object_support", block.maps["connected_object_support"][local_frame]),
        ("Role entropy", "role_entropy", block.maps["role_entropy"][local_frame]),
    ]
    evidence_png = output_dir / "evidence_pipeline.png"
    _map_grid(
        evidence_png,
        [(label, value) for label, _, value in evidence_panels],
        "Responsibility evidence pipeline; real aligned tensors",
        fixed_probability=True,
    )
    refinement_png = output_dir / "initial_vs_refined_support.png"
    figure, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    refinement_panels = [
        ("Initial Object Support", block.maps["initial_object_posterior"][local_frame]),
        ("Branch-Response Evidence", block.maps["field_observation"][local_frame]),
        ("Refined Object Support", block.maps["object_posterior"][local_frame]),
    ]
    for axis, (label, value) in zip(axes[:3], refinement_panels):
        image = axis.imshow(value, cmap="turbo", vmin=0.0, vmax=1.0, interpolation="nearest")
        axis.set_title(label)
        axis.set_xticks([]); axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axes[3].imshow(composite, interpolation="nearest")
    axes[3].set_title("Final Soft Roles")
    axes[3].set_xticks([]); axes[3].set_yticks([])
    figure.tight_layout()
    figure.savefig(refinement_png, dpi=170, bbox_inches="tight")
    plt.close(figure)
    detail_outputs = {
        "responsibility_role_composite": composite_png,
        "responsibility_composite_soft_roles": composite_named_png,
        "responsibility_evidence_pipeline": evidence_png,
        "responsibility_initial_vs_refined": refinement_png,
    }
    for label, field, value in evidence_panels:
        evidence_detail = output_dir / f"{field}.png"
        _save_heatmap(
            evidence_detail,
            value,
            label,
            0.0,
            1.0,
            cmap="turbo",
        )
        detail_outputs[f"responsibility_{field}"] = evidence_detail
    for name in responsibility_viz.ROLE_FIELDS:
        path = output_dir / f"{name}.png"
        _save_heatmap(
            path,
            block.maps[name][local_frame],
            name,
            0.0,
            1.0,
            cmap=ROLE_CMAPS[name],
        )
        detail_outputs[f"responsibility_{name}"] = path
    raw_roles = output_dir / "raw_roles.pt"
    torch.save({
        "schema_version": responsibility_viz.SCHEMA_VERSION,
        "block_index": block.index,
        "latent_frame_index": int(block.latent_frame_indices[local_frame]),
        "common_spatial_shape": block.spatial_shape,
        "native_spatial_shape": block.raw_role_spatial_shape,
        "common_grid": {name: torch.from_numpy(block.maps[name][local_frame].copy()) for name in responsibility_viz.ROLE_FIELDS},
        "native_grid": {name: torch.from_numpy(block.raw_role_maps[name][local_frame].copy()) for name in responsibility_viz.ROLE_FIELDS},
        "role_alignment": block.role_alignment,
    }, raw_roles)
    return {
        "responsibility_overview": responsibility_png,
        "raw_roles": raw_roles,
        **detail_outputs,
    }


def render_permissions(block: Any, local_frame: int, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = [(name, block.maps[name][local_frame]) for name in PERMISSION_FIELDS]
    png = output_dir / "permissions_overview.png"
    _map_grid(png, panels, "Operation-specific permissions; fixed [0,1]", fixed_probability=True)
    montage_png = output_dir / "permissions_montage.png"
    _map_grid(montage_png, panels, "Operation-specific permissions; fixed [0,1]", fixed_probability=True)
    detail_outputs = {}
    for name, value in panels:
        path = output_dir / f"{name}.png"
        _save_heatmap(
            path, value, name, 0.0, 1.0,
            cmap=PERMISSION_CMAPS[name],
        )
        detail_outputs[f"permission_{name}"] = path
    raw = output_dir / "raw_permissions.pt"
    torch.save({
        "schema_version": responsibility_viz.SCHEMA_VERSION,
        "block_index": block.index,
        "latent_frame_index": int(block.latent_frame_indices[local_frame]),
        "spatial_shape": block.spatial_shape,
        "permissions": {name: torch.from_numpy(value.copy()) for name, value in panels},
    }, raw)
    source_retention = output_dir / "source_retention.png"
    retention = block.raw_velocity_maps["rho_role_magnitude"][local_frame]
    retention_max = float(np.quantile(retention, 0.99))
    if retention_max <= 0:
        retention_max = 1.0
    _save_heatmap(
        source_retention,
        retention,
        "Role-Allocated Source Velocity Correction ||rho_role||",
        0.0,
        retention_max,
        cmap="Blues",
    )
    write_eligibility = output_dir / "reference_write_eligibility.png"
    _save_heatmap(
        write_eligibility,
        block.maps["reference_write_gate"][local_frame],
        "Reference Write Eligibility",
        0.0,
        1.0,
        cmap="Purples",
    )
    read_access = output_dir / "anchor_read_access.png"
    _save_heatmap(
        read_access,
        block.maps["reference_read_gate"][local_frame],
        "Anchor Read Access",
        0.0,
        1.0,
        cmap="Purples",
    )
    return {
        "permissions_overview": png,
        "permissions_montage": montage_png,
        "source_retention": source_retention,
        "reference_write_eligibility": write_eligibility,
        "anchor_read_access": read_access,
        "raw_permissions": raw,
        **detail_outputs,
    }


def interface_card(block: Any, local_frame: int, selection: Any, output_dir: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    index = (local_frame, selection.row, selection.column)
    fields = (
        (block.hand_field,)
        + responsibility_viz.ROLE_FIELDS
        + responsibility_viz.AUX_PROBABILITY_FIELDS
        + responsibility_viz.COEFFICIENT_FIELDS
        + responsibility_viz.PERMISSION_FIELDS
        + responsibility_viz.VELOCITY_FIELDS
    )
    card = {
        "selection_rule": "argmax q_interface within selected block/frame and valid_token & connected_object_support & hand_evidence>0",
        "block_index": block.index,
        "latent_frame_index": int(block.latent_frame_indices[local_frame]),
        "local_frame_index": local_frame,
        "token_row": selection.row,
        "token_column": selection.column,
        "spatial_shape": list(block.spatial_shape),
        "values": {name: float(block.maps[name][index]) for name in fields},
    }
    json_path = output_dir / "interface_token_card.json"
    _write_json(json_path, card)
    values_path = output_dir / "interface_token_values.json"
    _write_json(values_path, card)
    lines = [
        "Automatically selected interface token (real artifact values)",
        f"block={block.index} latent={card['latent_frame_index']} row={selection.row} col={selection.column}",
        "",
    ] + [f"{name}: {value:.6g}" for name, value in card["values"].items()]
    font = _font(20)
    image = Image.new("RGB", (1050, 90 + 29 * len(lines)), "white")
    draw = ImageDraw.Draw(image)
    for line_index, line in enumerate(lines):
        draw.text((28, 25 + line_index * 29), line, fill="black", font=font)
    png_path = output_dir / "interface_token_card.png"
    image.save(png_path)
    return card, {
        "interface_token_card_png": png_path,
        "interface_token_card_json": json_path,
        "interface_token_values_json": values_path,
    }


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _raw_responsibility_block(responsibility_dir: Path, block_index: int) -> tuple[dict[str, Any], Path]:
    matches = []
    for path in responsibility_viz.discover_artifacts(responsibility_dir):
        payload = _torch_load(path)
        for raw in responsibility_viz._extract_blocks(payload, path):
            if "block_index" in raw:
                index = int(np.asarray(_jsonable(raw["block_index"])).reshape(-1)[0])
            else:
                match = re.search(r"block[_-]?(\d+)", path.stem)
                if match is None:
                    raise SchemaError(f"{path}: nested responsibility block needs block index in filename")
                index = int(match.group(1))
            if index == block_index:
                matches.append((raw, path.resolve()))
    if len(matches) != 1:
        raise SchemaError(f"Raw responsibility block {block_index} is represented {len(matches)} times")
    return matches[0]


def _velocity_updates(
    raw_block: dict[str, Any], local_frame: int, path: Path
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float]]:
    predictions = _required(raw_block, "velocity_predictions", str(path))
    if not isinstance(predictions, dict):
        raise SchemaError(f"{path}: velocity_predictions must be a dictionary")
    expected = {name for name, _ in VELOCITY_UPDATES}
    if set(predictions) != expected:
        raise SchemaError(
            f"{path}: velocity_predictions keys must be exactly {sorted(expected)}, got {sorted(predictions)}"
        )
    updates = {}
    noise_levels = {}
    for prediction_name, _ in VELOCITY_UPDATES:
        prediction = predictions[prediction_name]
        if not isinstance(prediction, dict):
            raise SchemaError(f"{path}: velocity_predictions.{prediction_name} must be a dictionary")
        noise = _required(prediction, "noise_level", f"{path}:velocity_predictions.{prediction_name}")
        if not isinstance(noise, torch.Tensor) or noise.numel() != 1:
            raise SchemaError(
                f"{path}: velocity_predictions.{prediction_name}.noise_level must be a scalar tensor"
            )
        noise_level = float(noise.detach().cpu().item())
        if not math.isfinite(noise_level):
            raise SchemaError(
                f"{path}: velocity_predictions.{prediction_name}.noise_level must be finite"
            )
        noise_levels[prediction_name] = noise_level
        vectors = {}
        expected_shape = None
        for name in VELOCITY_VECTOR_FIELDS:
            value = _required(prediction, name, f"{path}:velocity_predictions.{prediction_name}")
            if not isinstance(value, torch.Tensor):
                raise SchemaError(
                    f"{path}: velocity_predictions.{prediction_name}.{name} must be a torch.Tensor"
                )
            tensor = value.detach().cpu().float()
            if tensor.ndim != 5 or tensor.shape[0] != 1:
                raise SchemaError(
                    f"{path}: velocity_predictions.{prediction_name}.{name} must be [1,T,C,H,W]"
                )
            if local_frame >= tensor.shape[1]:
                raise SchemaError(
                    f"{path}: velocity_predictions.{prediction_name}.{name} has no local frame {local_frame}"
                )
            vector = tensor[0, local_frame].numpy()
            if not np.isfinite(vector).all():
                raise SchemaError(
                    f"{path}: velocity_predictions.{prediction_name}.{name} contains non-finite values"
                )
            if expected_shape is None:
                expected_shape = vector.shape
            elif vector.shape != expected_shape:
                raise SchemaError(
                    f"{path}: {prediction_name} vectors do not share [C,H,W]: "
                    f"{vector.shape} versus {expected_shape}"
                )
            vectors[name] = vector
        updates[prediction_name] = vectors
    return updates, noise_levels


def _joint_velocity_pca(vectors: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    channel_count = next(iter(vectors.values())).shape[0]
    samples = [value.transpose(1, 2, 0).reshape(-1, channel_count).astype(np.float64) for value in vectors.values()]
    matrix = np.concatenate(samples, axis=0)
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    component_count = min(3, vh.shape[0])
    if component_count < 2:
        raise SchemaError("Velocity joint PCA requires at least two channel dimensions")
    components = vh[:component_count].copy()
    for index in range(component_count):
        pivot = int(np.argmax(np.abs(components[index])))
        if components[index, pivot] < 0:
            components[index] *= -1
    denominator = float(np.square(singular).sum())
    explained = np.square(singular[:component_count]) / denominator if denominator > 0 else np.zeros(component_count)
    projections = {}
    for name, value in vectors.items():
        flat = value.transpose(1, 2, 0).reshape(-1, channel_count).astype(np.float64)
        projected = (flat - mean) @ components.T
        projections[name] = projected.reshape(value.shape[1], value.shape[2], component_count).astype(np.float32)
    metadata = {
        "fit_input": "concatenated per-token raw velocity_predictions vectors for all six fields",
        "input_fields": list(VELOCITY_VECTOR_FIELDS),
        "excluded_inputs": [
            "role probabilities", "permission maps", "attention maps",
            "source_residual_norm", "conflict_score", "removed_component",
            "safe_residual_norm", "role_allocated_correction",
            "controlled_velocity_change",
        ],
        "sample_count": int(len(matrix)),
        "channel_count": channel_count,
        "component_count": component_count,
        "mean": mean.tolist(),
        "components": components.tolist(),
        "explained_variance_ratio": explained.tolist(),
    }
    return projections, metadata


def _velocity_role_colors(block: Any, local_frame: int, spatial_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    roles = np.stack([
        block.raw_role_maps[name][local_frame] for name in responsibility_viz.ROLE_FIELDS
    ], axis=-1)
    if tuple(roles.shape[:2]) != spatial_shape:
        raise SchemaError(
            f"Native role grid {roles.shape[:2]} does not exactly match velocity grid {spatial_shape}; "
            "refusing to interpolate role labels"
        )
    if not np.isfinite(roles).all() or (roles < 0).any():
        raise SchemaError("Native soft role probabilities must be finite and non-negative")
    dominant = roles.argmax(axis=-1).reshape(-1)
    names = np.asarray(responsibility_viz.ROLE_FIELDS, dtype=object)[dominant]
    colors = np.asarray([ROLE_COLORS[str(name)] for name in names], dtype=object)
    return names, colors


def _derived_velocity_maps(vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rho = vectors["rho"].astype(np.float64)
    direction = vectors["v_trg"].astype(np.float64) - vectors["v_src"].astype(np.float64)
    rho_safe = vectors["rho_safe"].astype(np.float64)
    rho_role = vectors["rho_role"].astype(np.float64)
    controlled_change = (
        vectors["v_controlled"].astype(np.float64)
        - vectors["v_trg"].astype(np.float64)
    )
    rho_norm = np.linalg.norm(rho, axis=0)
    direction_norm = np.linalg.norm(direction, axis=0)
    denominator = rho_norm * direction_norm
    cosine = np.zeros_like(denominator)
    np.divide(
        np.sum(rho * direction, axis=0), denominator,
        out=cosine, where=denominator > 0,
    )
    return {
        "source_residual_norm": rho_norm.astype(np.float32),
        "conflict_score": np.maximum(-cosine, 0.0).astype(np.float32),
        "removed_component": np.linalg.norm(rho - rho_safe, axis=0).astype(np.float32),
        "safe_residual_norm": np.linalg.norm(rho_safe, axis=0).astype(np.float32),
        "role_allocated_correction": np.linalg.norm(rho_role, axis=0).astype(np.float32),
        "controlled_velocity_change": np.linalg.norm(controlled_change, axis=0).astype(np.float32),
    }


DERIVED_VELOCITY_FORMULAS = {
    "source_residual_norm": "||rho||_2 over velocity channels",
    "conflict_score": "max(-cosine_similarity(rho, v_trg - v_src), 0); zero when either norm is zero",
    "removed_component": "||rho - rho_safe||_2 over velocity channels",
    "safe_residual_norm": "||rho_safe||_2 over velocity channels",
    "role_allocated_correction": "||rho_role||_2 over velocity channels",
    "controlled_velocity_change": "||v_controlled - v_trg||_2 over velocity channels",
}


def _interface_velocity_indices(
    selection: Any, common_shape: tuple[int, int], velocity_shape: tuple[int, int]
) -> list[int]:
    common_h, common_w = common_shape
    velocity_h, velocity_w = velocity_shape
    if velocity_h % common_h or velocity_w % common_w:
        raise SchemaError(
            f"Velocity grid {velocity_shape} is not an exact integer subdivision of token grid {common_shape}"
        )
    row_scale, column_scale = velocity_h // common_h, velocity_w // common_w
    return [
        row * velocity_w + column
        for row in range(selection.row * row_scale, (selection.row + 1) * row_scale)
        for column in range(selection.column * column_scale, (selection.column + 1) * column_scale)
    ]


def _pca_figure(
    path: Path,
    projections: dict[str, np.ndarray],
    role_names: np.ndarray,
    role_colors: np.ndarray,
    interface_indices: list[int],
    title: str,
    *,
    interface_only: bool,
) -> None:
    flat = {name: value.reshape(-1, value.shape[-1]) for name, value in projections.items()}
    figure, axis = plt.subplots(figsize=(10.5, 8.5))
    if interface_only:
        indices = np.asarray(interface_indices, dtype=np.int64)
        alpha, size = 0.85, 52
    else:
        indices = np.arange(len(role_names), dtype=np.int64)
        alpha, size = 0.22, 8
    markers = {
        "v_src": "o", "v_trg": "s", "rho": "^", "rho_safe": "v",
        "rho_role": "D", "v_controlled": "P",
    }
    for name in VELOCITY_VECTOR_FIELDS:
        points = flat[name][indices, :2]
        axis.scatter(
            points[:, 0], points[:, 1], c=role_colors[indices].tolist(),
            marker=markers[name], s=size, alpha=alpha, linewidths=0,
        )
    arrow_indices = indices
    segments = np.stack(
        [flat["v_trg"][arrow_indices, :2], flat["v_controlled"][arrow_indices, :2]],
        axis=1,
    )
    axis.add_collection(LineCollection(
        segments,
        colors=role_colors[arrow_indices].tolist(),
        linewidths=1.4 if interface_only else 0.35,
        alpha=0.8 if interface_only else 0.12,
    ))
    interface = np.asarray(interface_indices, dtype=np.int64)
    for name in VELOCITY_VECTOR_FIELDS:
        points = flat[name][interface, :2]
        axis.scatter(points[:, 0], points[:, 1], marker="*", s=170, facecolors="none", edgecolors="black", linewidths=1.3)
    for index in interface:
        start, end = flat["v_trg"][index, :2], flat["v_controlled"][index, :2]
        axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.8})
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=name, markersize=7)
        for name, color in ROLE_COLORS.items()
    ] + [
        Line2D([0], [0], marker=markers[name], linestyle="", color="black", label=name, markersize=6)
        for name in VELOCITY_VECTOR_FIELDS
    ]
    axis.legend(handles=handles, ncol=2, fontsize=7, loc="best")
    axis.set_xlabel("joint PC1")
    axis.set_ylabel("joint PC2")
    axis.set_title(title)
    axis.grid(alpha=0.15)
    axis.autoscale_view()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_velocity(
    raw_block: dict[str, Any], block: Any, local_frame: int, selection: Any,
    source_path: Path, output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    updates, noise_levels = _velocity_updates(raw_block, local_frame, source_path)
    outputs: dict[str, Path] = {}
    metrics: dict[str, Any] = {"updates": {}}
    all_scales = []
    for prediction_name, label in VELOCITY_UPDATES:
        vectors = updates[prediction_name]
        update_dir = output_dir / f"update_{label}"
        update_dir.mkdir(parents=True, exist_ok=False)
        spatial_shape = tuple(next(iter(vectors.values())).shape[1:])
        role_names, role_colors = _velocity_role_colors(block, local_frame, spatial_shape)
        interface_indices = _interface_velocity_indices(selection, block.spatial_shape, spatial_shape)
        derived = _derived_velocity_maps(vectors)
        norm_fields = [name for name in derived if name != "conflict_score"]
        pooled = np.concatenate([derived[name].reshape(-1) for name in norm_fields])
        scale = float(np.quantile(pooled, 0.99))
        if scale <= 0:
            scale = 1.0
        all_scales.append(scale)
        overview = update_dir / "update_diagnostics.png"
        figure, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
        for axis, (name, value) in zip(axes.flat, derived.items()):
            vmax = 1.0 if name == "conflict_score" else scale
            image = axis.imshow(value, cmap="inferno", vmin=0.0, vmax=vmax, interpolation="nearest")
            axis.set_title(name)
            axis.set_xticks([]); axis.set_yticks([])
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.suptitle(f"{label} / {prediction_name}; derived from exact raw vectors")
        figure.tight_layout()
        figure.savefig(overview, dpi=170, bbox_inches="tight")
        plt.close(figure)
        outputs[f"update_{label}_diagnostics"] = overview
        for name, value in derived.items():
            detail = update_dir / f"{name}.png"
            vmax = 1.0 if name == "conflict_score" else scale
            _save_heatmap(detail, value, f"{label} {name}", 0.0, vmax)
            outputs[f"update_{label}_{name}"] = detail
        derived_pt = update_dir / "derived_maps.pt"
        torch.save({
            "prediction": prediction_name,
            "update_label": label,
            "formulas": DERIVED_VELOCITY_FORMULAS,
            "maps": {name: torch.from_numpy(value.copy()) for name, value in derived.items()},
        }, derived_pt)
        derived_json = update_dir / "derived_maps_metadata.json"
        _write_json(derived_json, {
            "prediction": prediction_name,
            "update_label": label,
            "source": f"{source_path}:velocity_predictions.{prediction_name}",
            "formulas": DERIVED_VELOCITY_FORMULAS,
            "pca_inclusion": False,
        })
        outputs[f"update_{label}_derived_maps_raw"] = derived_pt
        outputs[f"update_{label}_derived_maps_metadata"] = derived_json

        projections, pca_metadata = _joint_velocity_pca(vectors)
        joint_png = update_dir / "velocity_joint_pca.png"
        zoom_png = update_dir / "velocity_pca_interface_zoom.png"
        _pca_figure(
            joint_png, projections, role_names, role_colors, interface_indices,
            f"{label}: six-field joint PCA; arrows v_trg to v_controlled", interface_only=False,
        )
        _pca_figure(
            zoom_png, projections, role_names, role_colors, interface_indices,
            f"{label}: exact interface-token velocity cells", interface_only=True,
        )
        role_counts = {
            name: int(np.count_nonzero(role_names == name))
            for name in responsibility_viz.ROLE_FIELDS
        }
        pca_metadata.update({
            "prediction": prediction_name,
            "update_label": label,
            "noise_level": noise_levels[prediction_name],
            "block_index": block.index,
            "latent_frame_index": int(block.latent_frame_indices[local_frame]),
            "spatial_shape": list(spatial_shape),
            "role_source": "first_response soft role probabilities on the exact native velocity grid",
            "role_coloring": "argmax dominant role; no interpolation",
            "role_colors": ROLE_COLORS,
            "dominant_role_counts": role_counts,
            "arrows": "per-token joint-PCA v_trg to v_controlled",
            "interface_velocity_flat_indices": interface_indices,
            "interface_mapping": "exact integer subdivision of the selected attention token; all covered velocity cells highlighted",
            "magnitude_shared_p99": scale,
        })
        pca_json = update_dir / "pca_metadata.json"
        _write_json(pca_json, pca_metadata)
        pca_pt = update_dir / "velocity_pca.pt"
        torch.save({
            "metadata": pca_metadata,
            "vectors": {name: torch.from_numpy(value.copy()) for name, value in vectors.items()},
            "projections": {name: torch.from_numpy(value.copy()) for name, value in projections.items()},
            "dominant_role": role_names.tolist(),
        }, pca_pt)
        outputs.update({
            f"update_{label}_velocity_joint_pca": joint_png,
            f"update_{label}_velocity_pca_interface_zoom": zoom_png,
            f"update_{label}_pca_metadata": pca_json,
            f"update_{label}_velocity_pca_raw": pca_pt,
        })
        metrics["updates"][label] = {
            "prediction": prediction_name,
            "noise_level": noise_levels[prediction_name],
            "magnitude_shared_p99": scale,
            "derived_map_statistics": {name: _distribution(value) for name, value in derived.items()},
            "derived_map_formulas": DERIVED_VELOCITY_FORMULAS,
            "joint_pca_explained_variance_ratio": pca_metadata["explained_variance_ratio"],
            "dominant_role_counts": role_counts,
        }
    metrics["magnitude_shared_p99"] = max(all_scales)
    return metrics, outputs


def _compact_anchor_summary(
    compact: Any, raw_block: dict[str, Any], path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(compact, dict) or not compact:
        raise SchemaError(f"{path}: m2.compact_anchor_state must be a non-empty dictionary")
    spatial = _shape(
        _required(raw_block, "token_spatial_shape", str(path)),
        "token_spatial_shape", (2,),
    )
    tokens_per_chunk = math.prod(spatial)
    raw_state = {}
    summary = {
        "available": True,
        "projection": (
            "High-dimensional key/value/query tensors are not spatially reconstructed. "
            "Reported values are exact RMS reductions over their trailing head/channel dimensions."
        ),
        "tokens_per_spatial_chunk": tokens_per_chunk,
        "layers": {},
    }
    for layer_name in sorted(compact):
        layer = compact[layer_name]
        if not isinstance(layer, dict):
            raise SchemaError(f"{path}: compact_anchor_state.{layer_name} must be a dictionary")
        tensors = {}
        for field in (
            "stored_token_indices", "stored_source_keys",
            "stored_target_minus_source_values", "current_clean_source_queries",
        ):
            value = _required(layer, field, f"{path}:compact_anchor_state.{layer_name}")
            if not isinstance(value, torch.Tensor):
                raise SchemaError(f"{path}: compact_anchor_state.{layer_name}.{field} must be a tensor")
            tensors[field] = value.detach().cpu().clone()
        indices = tensors["stored_token_indices"]
        if indices.ndim != 1 or indices.dtype == torch.bool or indices.is_floating_point():
            raise SchemaError(f"{path}: {layer_name}.stored_token_indices must be a 1D integer tensor")
        indices = indices.to(torch.int64)
        if bool((indices < 0).any()):
            raise SchemaError(f"{path}: {layer_name}.stored_token_indices must be non-negative")
        source_keys = tensors["stored_source_keys"]
        corrections = tensors["stored_target_minus_source_values"]
        queries = tensors["current_clean_source_queries"]
        if source_keys.ndim < 2 or corrections.ndim < 2 or source_keys.shape[0] != len(indices) or corrections.shape[0] != len(indices):
            raise SchemaError(
                f"{path}: {layer_name} stored key/value tensors must start with stored_token_indices length"
            )
        if queries.numel() and queries.ndim < 2:
            raise SchemaError(f"{path}: {layer_name}.current_clean_source_queries must be empty or at least 2D")
        for field, tensor in (("stored_source_keys", source_keys), ("stored_target_minus_source_values", corrections), ("current_clean_source_queries", queries)):
            if not tensor.is_floating_point() or not bool(torch.isfinite(tensor.float()).all()):
                raise SchemaError(f"{path}: {layer_name}.{field} must be finite floating point")

        def rms_rows(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.float().reshape(tensor.shape[0], -1).square().mean(dim=1).sqrt()

        key_rms = rms_rows(source_keys)
        correction_rms = rms_rows(corrections)
        query_rms = rms_rows(queries) if queries.numel() else None
        chunk_ids = torch.div(indices, tokens_per_chunk, rounding_mode="floor")
        chunks = []
        for chunk_id in torch.unique(chunk_ids, sorted=True).tolist():
            mask = chunk_ids == int(chunk_id)
            chunk_values = corrections[mask].float()
            chunks.append({
                "stored_token_chunk_index": int(chunk_id),
                "stored_token_count": int(mask.sum()),
                "correction_rms": float(chunk_values.square().mean().sqrt()),
                "stored_token_index_min": int(indices[mask].min()),
                "stored_token_index_max": int(indices[mask].max()),
            })
        summary["layers"][layer_name] = {
            "tensor_shapes": {name: list(value.shape) for name, value in tensors.items()},
            "tensor_dtypes": {name: str(value.dtype) for name, value in tensors.items()},
            "stored_source_key_rms_per_token": key_rms.tolist(),
            "stored_correction_rms_per_token": correction_rms.tolist(),
            "current_clean_source_query_rms": (
                {"available": True, "values": query_rms.tolist()}
                if query_rms is not None
                else {"available": False, "reason": "raw tensor is explicitly empty"}
            ),
            "correction_by_chunk": chunks,
        }
        raw_state[layer_name] = tensors
    return raw_state, summary


def _render_correction_by_chunk(path: Path, summary: dict[str, Any]) -> None:
    layers = list(summary["layers"])
    figure, axes = plt.subplots(len(layers), 1, figsize=(8, max(3.2, 2.8 * len(layers))), squeeze=False)
    for axis, layer_name in zip(axes.flat, layers):
        chunks = summary["layers"][layer_name]["correction_by_chunk"]
        positions = [entry["stored_token_chunk_index"] for entry in chunks]
        values = [entry["correction_rms"] for entry in chunks]
        axis.bar(positions, values, color=ROLE_COLORS["q_interface"])
        axis.set_title(f"{layer_name}: stored target-minus-source value RMS by token chunk")
        axis.set_xlabel("stored token chunk index")
        axis.set_ylabel("RMS")
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def render_anchor(
    raw_block: dict[str, Any], block: Any, local_frame: int, source_path: Path, output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    retrieval = block.hook_extras.get("retrieval_mean_maps", {}) if block.hook_extras else {}
    raw_m2 = _required(raw_block, "m2", str(source_path))
    compact = raw_m2.get("compact_anchor_state") if isinstance(raw_m2, dict) else None
    if not retrieval and compact is None:
        return {
            "available": False,
            "generated": False,
            "reason": "selected responsibility block contains neither real retrieval maps nor compact anchor state",
        }, {}
    raw_payload = {}
    raw_topk_payload = {}
    temporal_size = len(block.latent_frame_indices)
    token_count = temporal_size * math.prod(block.spatial_shape)
    for prediction in sorted(retrieval):
        entry = retrieval[prediction]
        maps = {}
        topk_tables = {}
        if not isinstance(entry, dict) or not entry:
            raise SchemaError(f"retrieval.{prediction} must be a non-empty dictionary of real maps")
        for field in sorted(entry):
            value = _required(entry, field, f"retrieval.{prediction}")
            array = np.asarray(value)
            if field.endswith("_topk_index") or field.endswith("_topk_similarity"):
                while array.ndim > 2 and array.shape[0] == 1:
                    array = array[0]
                if array.ndim != 2 or array.shape[0] != token_count:
                    raise SchemaError(
                        f"retrieval.{prediction}.{field} must be [T*H*W,K], got {array.shape}"
                    )
                if field.endswith("_topk_index"):
                    if not np.issubdtype(array.dtype, np.integer):
                        raise SchemaError(
                            f"retrieval.{prediction}.{field} must contain integer key indices"
                        )
                elif not np.isfinite(array).all():
                    raise SchemaError(
                        f"retrieval.{prediction}.{field} contains non-finite similarities"
                    )
                topk_tables[field] = array.copy()
                if field.endswith("_topk_similarity"):
                    # Each query has K real matches.  Preserve that K-axis in
                    # raw_anchor.pt and expose explicitly labelled reductions
                    # for spatial display; indices themselves are never drawn
                    # as scalar heatmaps.
                    shaped = array.reshape(
                        temporal_size, *block.spatial_shape, array.shape[-1]
                    )
                    maps[f"{field}_mean_over_k"] = shaped.mean(axis=-1)[local_frame]
                    maps[f"{field}_max_over_k"] = shaped.max(axis=-1)[local_frame]
                continue
            full = responsibility_viz._flattened_or_spatial_map(
                value,
                name=f"retrieval.{prediction}.{field}",
                temporal_size=temporal_size,
                candidate_shapes=(block.spatial_shape,),
            )
            maps[field] = full[local_frame]
        raw_payload[prediction] = maps
        raw_topk_payload[prediction] = topk_tables
    preferred = ("prediction_000", "prediction_007", "prediction_014")
    displayed_predictions = [name for name in preferred if name in raw_payload]
    if not displayed_predictions:
        displayed_predictions = sorted(raw_payload)[:3]
    rows = [
        (f"{prediction} {field}", value)
        for prediction in displayed_predictions
        for field, value in raw_payload[prediction].items()
    ]
    detail_outputs = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    if rows:
        png = output_dir / "anchor.png"
        _map_grid(png, rows, "Real anchor retrieval maps", fixed_probability=False)
        outputs["anchor_png"] = png
    for prediction in displayed_predictions:
        maps = raw_payload[prediction]
        for field, value in maps.items():
            detail = output_dir / f"retrieval_{prediction}_{_safe_name(field)}.png"
            minimum, maximum = float(value.min()), float(value.max())
            if maximum <= minimum:
                maximum = minimum + 1e-12
            _save_heatmap(detail, value, f"{prediction} {field}", minimum, maximum)
            detail_outputs[f"anchor_{prediction}_{field}"] = detail
    summary_prediction = (
        "prediction_007"
        if "prediction_007" in raw_payload
        else displayed_predictions[-1] if displayed_predictions else None
    )
    if summary_prediction is not None:
        summary_maps = raw_payload[summary_prediction]
        summary_fields = {
            "retrieval_matches.png": "joint_similarity",
            "reference_response.png": "reference_response_rms",
            "current_response.png": "current_response_rms",
            "response_discrepancy.png": "response_discrepancy_rms",
            "correction_magnitude.png": "correction_rms",
        }
        for filename, field in summary_fields.items():
            if field not in summary_maps:
                continue
            value = summary_maps[field]
            minimum, maximum = float(value.min()), float(value.max())
            if maximum <= minimum:
                maximum = minimum + 1e-12
            destination = output_dir / filename
            _save_heatmap(
                destination,
                value,
                f"Anchor {field} ({summary_prediction})",
                minimum,
                maximum,
                cmap="Purples",
            )
            detail_outputs[f"anchor_summary_{field}"] = destination
    selected_write = block.hook_extras.get("selected_write_tokens") if block.hook_extras else None
    if selected_write is not None:
        full_write = responsibility_viz._flattened_or_spatial_map(
            selected_write,
            name="m2.selected_write_tokens",
            temporal_size=temporal_size,
            candidate_shapes=(block.spatial_shape,),
        )
        write_tokens_png = output_dir / "write_tokens.png"
        _save_heatmap(
            write_tokens_png,
            full_write[local_frame],
            "Selected Anchor Write Tokens",
            0.0,
            1.0,
            cmap="Purples",
        )
        detail_outputs["anchor_write_tokens"] = write_tokens_png
    compact_raw = {}
    compact_summary = {"available": False, "reason": "m2.compact_anchor_state is absent"}
    if compact is not None:
        compact_raw, compact_summary = _compact_anchor_summary(compact, raw_block, source_path)
        compact_json = output_dir / "compact_anchor_state_summary.json"
        _write_json(compact_json, compact_summary)
        correction_json = output_dir / "correction_by_chunk.json"
        _write_json(correction_json, {
            "projection": compact_summary["projection"],
            "layers": {
                name: value["correction_by_chunk"]
                for name, value in compact_summary["layers"].items()
            },
        })
        correction_png = output_dir / "correction_by_chunk.png"
        _render_correction_by_chunk(correction_png, compact_summary)
        outputs.update({
            "compact_anchor_state_summary": compact_json,
            "correction_by_chunk_json": correction_json,
            "correction_by_chunk_png": correction_png,
        })
    raw = output_dir / "raw_anchor.pt"
    torch.save({
        "block_index": block.index,
        "latent_frame_index": int(block.latent_frame_indices[local_frame]),
        "retrieval_maps": {
            prediction: {field: torch.from_numpy(value.copy()) for field, value in maps.items()}
            for prediction, maps in raw_payload.items()
        },
        "raw_retrieval_maps": raw_m2.get("retrieval_maps", {}),
        "topk_match_tables": {
            prediction: {
                field: torch.from_numpy(value.copy())
                for field, value in tables.items()
            }
            for prediction, tables in raw_topk_payload.items()
        },
        "compact_anchor_state": compact_raw,
        "compact_anchor_state_summary": compact_summary,
    }, raw)
    outputs.update({"raw_anchor": raw, **detail_outputs})
    return {
        "available": True,
        "generated": True,
        "source": "selected raw responsibility block m2 retrieval_maps and compact_anchor_state",
        "predictions": sorted(retrieval),
        "displayed_predictions": displayed_predictions,
        "fields_by_prediction": {
            prediction: sorted(maps) for prediction, maps in raw_payload.items()
        },
        "topk_fields_by_prediction": {
            prediction: sorted(tables)
            for prediction, tables in raw_topk_payload.items()
        },
        "topk_visualization_policy": (
            "integer match indices remain tables in raw_anchor.pt; similarity "
            "is displayed only as explicitly labelled mean/max reductions over K"
        ),
        "missing_predictions_are_not_reconstructed": True,
        "compact_anchor_state": compact_summary,
        "high_dimensional_projection": compact_summary.get("projection"),
    }, outputs


def _video_frame(video: Any, frame_index: int, name: str) -> np.ndarray:
    if frame_index < 0 or frame_index >= len(video.frames):
        raise SchemaError(f"{name} has {len(video.frames)} frames; selected frame index is {frame_index}")
    return video.frames[frame_index]


def render_dashboard(
    source: Any,
    baseline: Any,
    edited: Any,
    block: Any,
    local_frame: int,
    selection: Any,
    velocity_scale: float,
    pixel_frame: int,
    pixel_group: tuple[int, int],
    output_dir: Path,
) -> dict[str, Path]:
    latent_frame = int(block.latent_frame_indices[local_frame])
    source_frame = _video_frame(source, pixel_frame, "source video")
    contact_frame = _video_frame(edited, pixel_frame, "edited video")
    late_pixel_frame = len(edited.frames) - 1
    late_frame = _video_frame(edited, late_pixel_frame, "edited video")
    common_height, common_width = block.spatial_shape

    source_image = Image.fromarray(source_frame)
    annotated = source_image.copy()
    draw = ImageDraw.Draw(annotated)
    center_x = (selection.column + 0.5) * source_image.width / common_width
    center_y = (selection.row + 0.5) * source_image.height / common_height
    crop_radius = max(48, int(round(min(source_image.size) * 0.16)))
    left = max(0, int(round(center_x)) - crop_radius)
    top = max(0, int(round(center_y)) - crop_radius)
    right = min(source_image.width, int(round(center_x)) + crop_radius)
    bottom = min(source_image.height, int(round(center_y)) + crop_radius)
    draw.rectangle((left, top, right - 1, bottom - 1), outline="#9c6ade", width=6)
    source_frame_png = output_dir / "source_frame.png"
    interface_crop_png = output_dir / "interface_crop.png"
    contact_result_png = output_dir / "contact_result.png"
    late_result_png = output_dir / "late_chunk_result.png"
    annotated.save(source_frame_png)
    source_image.crop((left, top, right, bottom)).save(interface_crop_png)
    Image.fromarray(contact_frame).save(contact_result_png)
    Image.fromarray(late_frame).save(late_result_png)

    cross_layers = sorted((output_dir / "cross_attention" / "step_007").glob("layer_*"))
    self_layers = sorted((output_dir / "self_attention" / "step_007").glob("layer_*"))
    if not cross_layers or not self_layers:
        raise SchemaError("Dashboard requires Tm cross/self attention layers")
    cross_layer = cross_layers[len(cross_layers) // 2]
    self_layer = self_layers[len(self_layers) // 2]
    panel_paths = [
        source_frame_png,
        cross_layer / "raw_scale" / "head_mean.png",
        output_dir / "responsibility" / "hand_occupancy.png",
        output_dir / "responsibility" / "temporal_posterior.png",
        output_dir / "responsibility" / "field_observation.png",
        output_dir / "responsibility" / "composite_soft_roles.png",
        cross_layer / "raw_scale" / "montage_all_heads.png",
        self_layer / "query_interface_query" / "raw_scale" / "montage_all_heads.png",
        self_layer / "query_object_query" / "raw_scale" / "montage_all_heads.png",
        self_layer / "query_hand_query" / "raw_scale" / "montage_all_heads.png",
        output_dir / "permissions" / "appearance_access.png",
        output_dir / "interface_token_card.png",
        output_dir / "update_Tm" / "conflict_score.png",
        output_dir / "update_Tm" / "removed_component.png",
        output_dir / "update_Tm" / "safe_residual_norm.png",
        output_dir / "update_Tm" / "role_allocated_correction.png",
        output_dir / "update_Tm" / "controlled_velocity_change.png",
        output_dir / "update_Tm" / "velocity_joint_pca.png",
        output_dir / "permissions" / "source_retention.png",
        output_dir / "permissions" / "appearance_access.png",
        output_dir / "permissions" / "reference_write_eligibility.png",
        output_dir / "permissions" / "anchor_read_access.png",
        contact_result_png,
        late_result_png,
    ]
    panel_titles = [
        f"Source + interface crop box (pixel {pixel_frame})",
        f"Cross-Attn mean (Tm/{cross_layer.name})",
        "Hand occupancy",
        "Temporal propagation",
        "Branch-response evidence",
        "Composite soft roles",
        f"Cross-Attn all heads (Tm/{cross_layer.name})",
        f"Self all heads: Interface ({self_layer.name})",
        "Self all heads: Object Core",
        "Self all heads: Hand",
        "Role-Conditioned Appearance Access",
        "Same Mixed Token, Different Permissions",
        "Conflict score",
        "Removed conflicting component",
        "Safe residual",
        "Role-allocated source correction",
        "Controlled velocity change",
        "Six-field joint PCA",
        "Source Retention ||rho_role||",
        "Appearance Access",
        "Reference Write Eligibility",
        "Anchor Read Access",
        f"Contact result (pixel {pixel_frame})",
        f"Late-chunk result (pixel {late_pixel_frame})",
    ]
    missing = [str(path) for path in panel_paths if not path.is_file()]
    if missing:
        raise SchemaError("Dashboard component(s) missing: " + ", ".join(missing))
    figure, axes = plt.subplots(4, 6, figsize=(30, 18), squeeze=False)
    row_labels = (
        "Evidence to Roles",
        "Attention and Access",
        "Role-Guided Update",
        "Permissions and Output",
    )
    for index, (axis, path, title) in enumerate(zip(axes.flat, panel_paths, panel_titles)):
        axis.imshow(np.asarray(Image.open(path).convert("RGB")))
        axis.set_title(title, fontsize=10)
        axis.set_xticks([]); axis.set_yticks([])
        if index % 6 == 0:
            axis.set_ylabel(row_labels[index // 6], fontsize=12, fontweight="bold")
    figure.suptitle(
        f"Single-case diagnostics: block {block.index}, latent {latent_frame}, "
        f"pixels [{pixel_group[0]},{pixel_group[1]}), shown {pixel_frame}\n"
        "Same Mixed Token, Different Operational Permissions",
        fontsize=18,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    png = output_dir / "ours_mechanism_dashboard.png"
    pdf = output_dir / "ours_mechanism_dashboard.pdf"
    figure.savefig(png, dpi=180, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return {
        "ours_mechanism_dashboard_png": png,
        "ours_mechanism_dashboard_pdf": pdf,
        "source_frame": source_frame_png,
        "interface_crop": interface_crop_png,
        "contact_result": contact_result_png,
        "late_chunk_result": late_result_png,
    }


def _ensure_owned_outputs_absent(output_dir: Path) -> None:
    owned = [
        "cross_attention", "self_attention", "responsibility", "permissions",
        "update_T0", "update_T1", "update_Tm", "anchor",
        "ours_mechanism_dashboard.png", "ours_mechanism_dashboard.pdf",
        "interface_token_card.png", "interface_token_card.json",
        "interface_token_values.json",
        "edited_video.mp4", "metrics.json", "metadata.json",
    ]
    collisions = [output_dir / name for name in owned if (output_dir / name).exists()]
    if collisions:
        relative = ", ".join(path.name for path in collisions)
        raise FileExistsError(
            f"Refusing to overwrite existing visualization outputs in {output_dir}: {relative}"
        )


def run(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = args.output_dir.resolve()
    _ensure_owned_outputs_absent(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_case = load_selection(args.selection)
    run_manifest = load_run_manifest(args.run_manifest)
    if Path(run_manifest["case"]["source_video"]).resolve() != args.source_video.resolve():
        raise SchemaError("run manifest source_video does not match --source-video")
    _, block, _, local_frame, selection = select_responsibility(
        args.responsibility_dir, selected_case
    )
    selected_latent_frame = int(selected_case["global_latent_frame"])
    artifacts = discover_attention_artifacts(args.attention_dir)
    raw_block, raw_block_path = _raw_responsibility_block(
        args.responsibility_dir, int(selected_case["block_index"])
    )
    groups = load_temporal_groups(args.hand_role_input)
    source, edited, baseline = (
        read_video(args.source_video),
        read_video(args.edited_video),
        read_video(args.baseline_video),
    )
    pixel_frame, pixel_group = selected_pixel_frame(
        groups,
        selected_latent_frame,
        {"source video": source, "edited video": edited, "baseline video": baseline},
    )

    outputs: dict[str, Path] = {}
    attention_metadata, attention_metrics = render_attention(
        artifacts, output_dir, selected_case
    )
    outputs.update(render_responsibility(block, local_frame, output_dir / "responsibility"))
    outputs.update(render_permissions(block, local_frame, output_dir / "permissions"))
    card, card_outputs = interface_card(block, local_frame, selection, output_dir)
    outputs.update(card_outputs)
    velocity_metrics, velocity_outputs = render_velocity(
        raw_block, block, local_frame, selection, raw_block_path, output_dir
    )
    outputs.update(velocity_outputs)
    anchor_metadata, anchor_outputs = render_anchor(
        raw_block, block, local_frame, raw_block_path, output_dir / "anchor"
    )
    outputs.update(anchor_outputs)
    outputs.update(render_dashboard(
        source, baseline, edited, block, local_frame, selection,
        velocity_metrics["magnitude_shared_p99"], pixel_frame, pixel_group, output_dir,
    ))
    edited_copy = output_dir / "edited_video.mp4"
    if args.edited_video.resolve() != edited_copy.resolve():
        shutil.copy2(args.edited_video, edited_copy)
    outputs["edited_video"] = edited_copy

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "selection": card,
        "role_probability_sum_max_error": float(
            np.abs(
                sum(
                    block.maps[name][local_frame]
                    for name in responsibility_viz.ROLE_FIELDS
                )
                - 1.0
            ).max(initial=0.0)
        ),
        "attention": attention_metrics,
        "velocity": velocity_metrics,
        "selected_frame_role_statistics": {
            name: _distribution(block.maps[name][local_frame]) for name in responsibility_viz.ROLE_FIELDS
        },
        "selected_frame_permission_statistics": {
            name: _distribution(block.maps[name][local_frame]) for name in PERMISSION_FIELDS
        },
    }
    metrics_path = output_dir / "metrics.json"
    _write_json(metrics_path, metrics)
    outputs["metrics"] = metrics_path

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "responsibility_dir": str(args.responsibility_dir.resolve()),
            "attention_dir": str(args.attention_dir.resolve()),
            "source_video": str(args.source_video.resolve()),
            "edited_video": str(args.edited_video.resolve()),
            "baseline_video": str(args.baseline_video.resolve()),
            "selection": str(args.selection.resolve()),
            "hand_role_input": str(args.hand_role_input.resolve()),
            "run_manifest": str(args.run_manifest.resolve()),
        },
        "selection": {
            "block_index": block.index,
            "latent_frame_index": int(block.latent_frame_indices[local_frame]),
            "local_frame_index": local_frame,
            "interface_token": card,
            "pixel_frame_range": list(pixel_group),
            "representative_pixel_frame": pixel_frame,
            "latent_to_pixel_policy": (
                "exact hand_role_input.npz causal_temporal_groups interval; "
                "display uses its left-middle representative frame"
            ),
        },
        "case_provenance": {
            **run_manifest["case"],
            "git_commit": run_manifest["git_commit"],
            "baseline": run_manifest["baseline"],
            "capture": run_manifest["capture"],
            "diagnostic_updates": {
                label: {
                    "prediction_index": run_manifest["diagnostic_updates"][label],
                    "noise_level": velocity_metrics["updates"][label]["noise_level"],
                }
                for _, label in VELOCITY_UPDATES
            },
            "noise_level_source": "raw responsibility artifact velocity_predictions.*.noise_level",
            "manifest_source": str(args.run_manifest.resolve()),
        },
        "responsibility": {
            "source_artifact": block.source_path,
            "schema_version": responsibility_viz.SCHEMA_VERSION,
            "common_spatial_shape": list(block.spatial_shape),
            "native_role_spatial_shape": list(block.raw_role_spatial_shape),
            "role_alignment": block.role_alignment,
        },
        "attention": attention_metadata,
        "attention_visualization": {
            "normalized_color_scale": [0.0, 1.0],
            "normalization": "independent exact min-max per displayed head/mean/std map",
            "raw_scale": "one exact min/max shared by the full layer; for self attention it is shared across all three queries",
            "directory_partition": "kind/step_NNN/layer_NN prevents 30-layer x multi-step overwrites",
            "query_selection_source": "raw artifact metadata.queries",
            "self_history_policy": "mass statistics only; no spatial reconstruction",
            "colormap": "turbo RGB for scalar attention maps",
            "raw_attention_directories": {
                "cross": str((output_dir / "cross_attention").resolve()),
                "self": str((output_dir / "self_attention").resolve()),
            },
        },
        "dashboard": {
            "layout": "4 rows x 6 columns",
            "rows": [
                "Evidence to Roles",
                "Attention and Access",
                "Role-Guided Update",
                "Permissions and Output",
            ],
            "attention_selection": "Tm prediction and fixed middle transformer layer",
            "role_colors": ROLE_COLORS,
        },
        "velocity": {
            "source_artifact": str(raw_block_path),
            "vector_fields": list(VELOCITY_VECTOR_FIELDS),
            "updates": {
                label: {
                    "prediction": prediction,
                    "directory": str((output_dir / f"update_{label}").resolve()),
                    "pca_metadata": str(outputs[f"update_{label}_pca_metadata"].resolve()),
                }
                for prediction, label in VELOCITY_UPDATES
            },
            "pca_input_policy": "six raw velocity_predictions tensors only",
        },
        "anchor": anchor_metadata,
        "videos": {
            "frame_index_policy": "strict causal_temporal_groups latent-to-pixel mapping",
            "causal_temporal_groups_source": (
                f"{args.hand_role_input.resolve()}:causal_temporal_groups"
            ),
            "selected_pixel_group": list(pixel_group),
            "representative_pixel_frame": pixel_frame,
            "source": {"frames": len(source.frames), "fps": source.fps},
            "baseline": {"frames": len(baseline.frames), "fps": baseline.fps},
            "edited": {"frames": len(edited.frames), "fps": edited.fps},
        },
        "no_approximate_semantic_reconstruction": True,
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
    }
    comparison_path = args.run_manifest.resolve().parent / "decoded_video_comparison.json"
    if comparison_path.is_file():
        metadata["diagnostic_output_comparison"] = {
            "source": str(comparison_path),
            **json.loads(comparison_path.read_text(encoding="utf-8")),
        }
    metadata_path = output_dir / "metadata.json"
    outputs["metadata"] = metadata_path
    metadata["outputs"]["metadata"] = str(metadata_path.resolve())
    _write_json(metadata_path, metadata)
    return outputs


def main(argv: list[str] | None = None) -> None:
    paths = run(parse_args(argv))
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
