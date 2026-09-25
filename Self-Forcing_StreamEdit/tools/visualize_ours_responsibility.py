#!/usr/bin/env python3
"""Strict offline visualizer for the Ours responsibility hook.

Hook contract (``ours-responsibility-v1``)
===============================================================
The artifact input is either a directory containing ``raw/*.pt``,
``responsibility/block_NNN.pt`` or direct ``block_NNN.pt`` files, or one such
file.  The native inference hook stores a nested per-block dictionary with::

    global_latent_frame_indices
    metadata
    initial.{source_attention, hand_occupancy, hand_proximity,
             temporal_confidence, temporal_posterior, connected_support,
             object_posterior}
    first_response.{object_posterior_prior, field_score, field_observation,
                    refined_object_posterior, q_*, role_entropy}
    velocity_step0.{rho, d, rho_safe, removed, rho_role,
                    safe_residual_coefficient, full_residual_coefficient,
                    effective_background_gated_coefficient}
    appearance_access.prediction_*.{source,map.{layers,by_layer,mean}}
    m2.{final_attention_request, m2_canonical_read_gate,
        m2_canonical_write_gate, write_eligibility, selected_write_tokens,
        bank_initialized, bank_initialized_before_block,
        bank_initialization_chunk, retrieval_maps}

The common token grid is the ``initial.object_posterior`` grid (the real cook
run uses 30x52).  Support, hand evidence, object posterior, and M2 gates stay
on that grid.  Native q and velocity fields use 60x104.  Velocity magnitudes
are the exact channel RMS of the five stored vectors and are area-resampled
to the common grid only for aligned regional metrics; their native values are
retained for plots, diagnostics, p99 scaling, and raw statistics.

A normalized flat v1 block with explicit scalar/index fields is also accepted
for focused testing and external hooks::

    schema_version, block_index, latent_frame_indices, spatial_shape
    valid_token, connected_object_support, hand_probability|hand_occupancy
    initial_object_posterior, object_posterior_prior, object_posterior
    field_score, field_observation, role_entropy
    safe_residual_coefficient, full_residual_coefficient
    effective_background_gated_coefficient
    appearance_access, reference_read_gate, reference_write_gate
    rho_magnitude, d_magnitude, rho_safe_magnitude
    removed_magnitude, rho_role_magnitude

The native role maps below may be on a different grid (the real cook run uses
60x104)::

    q_object, q_interface, q_hand, q_background

The preferred hook contract additionally stores the posterior common-grid maps
as ``q_object_token_grid``, ``q_interface_token_grid``,
``q_hand_token_grid``, and ``q_background_token_grid``.  A complete saved tuple is used verbatim.  If
it is absent, and only then, the visualizer exactly applies
``PosteriorResidualFlowRouter._resize_roles`` semantics: four-channel
bilinear interpolation with ``align_corners=False``, clamp-min-zero, then
normalization across roles with denominator clamp-min 1e-6.  Partial tuples
are errors.  This is the only permitted cross-grid derivation.

All q/permission/hand maps are probabilities in [0,1].  The role tuple must
sum to one on valid tokens.  Velocity maps are finite non-negative magnitudes.
Missing or ambiguous data is an error: this tool never reconstructs a semantic
field from another diagnostic.  Container files may hold a direct block dict,
a ``{"blocks": ...}`` wrapper, a list of block dicts, or ``block_NNN`` keys.

The hand-role NPZ is authoritative for ``causal_temporal_groups``.  It is used
to align latent maps to source and edited pixel frames without guessing a
temporal stride.  Interface-token selection is always automatic: argmax
q_interface over valid_token & connected_object_support & (hand evidence > 0).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import shutil
from pathlib import Path
from typing import Any, Iterable, NamedTuple

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


SCHEMA_VERSION = "ours-responsibility-v1"
ROLE_FIELDS = (
    "q_object",
    "q_interface",
    "q_hand",
    "q_background",
)
COEFFICIENT_FIELDS = (
    "safe_residual_coefficient",
    "full_residual_coefficient",
    "effective_background_gated_coefficient",
)
PERMISSION_FIELDS = (
    "appearance_access",
    "reference_read_gate",
    "reference_write_gate",
)
AUX_PROBABILITY_FIELDS = (
    "initial_object_posterior",
    "object_posterior_prior",
    "object_posterior",
    "field_score",
    "field_observation",
    "role_entropy",
)
VELOCITY_FIELDS = (
    "rho_magnitude",
    "d_magnitude",
    "rho_safe_magnitude",
    "removed_magnitude",
    "rho_role_magnitude",
)
VELOCITY_VECTOR_FIELDS = ("rho", "d", "rho_safe", "removed", "rho_role")
MASK_FIELDS = ("valid_token", "connected_object_support")
EVIDENCE_FIELDS = (
    "source_attention",
    "hand_proximity",
    "temporal_confidence",
    "temporal_posterior",
)
ROLE_LABELS = ("Object Core", "Interface", "Hand", "Background")
ROLE_COLORS = np.asarray(
    (
        (55, 126, 255),   # blue
        (157, 89, 255),   # purple
        (255, 111, 97),   # coral
        (132, 136, 143),  # gray
    ),
    dtype=np.float32,
) / 255.0
MIXED_INTERFACE_THRESHOLD = 0.20
QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


class SchemaError(ValueError):
    """Raised when a hook artifact cannot satisfy the explicit v1 schema."""


class Block(NamedTuple):
    index: int
    latent_frame_indices: np.ndarray
    spatial_shape: tuple[int, int]
    maps: dict[str, np.ndarray]
    raw_role_maps: dict[str, np.ndarray]
    raw_role_spatial_shape: tuple[int, int]
    role_alignment: str
    raw_velocity_maps: dict[str, np.ndarray]
    velocity_spatial_shape: tuple[int, int]
    velocity_alignment: str
    hook_extras: dict[str, Any]
    hand_field: str
    source_path: str


class TokenSelection(NamedTuple):
    block_position: int
    local_frame: int
    row: int
    column: int
    score: float


class Video(NamedTuple):
    frames: np.ndarray
    fps: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render strict offline Ours responsibility diagnostics."
    )
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--edited-video", type=Path, required=True)
    parser.add_argument("--hand-role-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-keyframes", type=int, default=5)
    parser.add_argument("--crop-radius-tokens", type=int, default=3)
    return parser.parse_args(argv)


def _scalar(value: Any, name: str) -> Any:
    array = _to_numpy(value)
    if array.size != 1:
        raise SchemaError(f"{name} must be scalar, got shape {array.shape}")
    return array.reshape(-1)[0].item()


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _artifact_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"block[_-]?(\d+)", path.stem)
    return (int(match.group(1)) if match else 10**12, str(path))


def discover_artifacts(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Artifact input does not exist: {path}")
    candidates = set(path.glob("block_*.pt"))
    candidates.update(path.glob("raw/*.pt"))
    candidates.update(path.glob("responsibility/block_*.pt"))
    if path.name == "raw":
        candidates.update(path.glob("*.pt"))
    files = sorted((item for item in candidates if item.is_file()), key=_artifact_sort_key)
    if not files:
        raise FileNotFoundError(
            f"No raw/*.pt or block_NNN.pt responsibility artifacts in {path}"
        )
    return files


def _looks_like_block(value: Any) -> bool:
    return isinstance(value, dict) and (
        "block_index" in value
        or "q_interface" in value
        or "global_latent_frame_indices" in value
    )


def _extract_blocks(payload: Any, path: Path) -> list[dict[str, Any]]:
    if _looks_like_block(payload):
        return [payload]
    if isinstance(payload, (list, tuple)):
        if not payload or not all(_looks_like_block(item) for item in payload):
            raise SchemaError(f"{path}: list payload must contain only block dicts")
        return list(payload)
    if isinstance(payload, dict) and "blocks" in payload:
        blocks = payload["blocks"]
        if isinstance(blocks, dict):
            blocks = list(blocks.values())
        if not isinstance(blocks, (list, tuple)) or not all(
            _looks_like_block(item) for item in blocks
        ):
            raise SchemaError(f"{path}: 'blocks' must contain block dicts")
        return list(blocks)
    if isinstance(payload, dict) and payload and all(
        re.fullmatch(r"block[_-]?\d+", str(key)) and _looks_like_block(value)
        for key, value in payload.items()
    ):
        return list(payload.values())
    raise SchemaError(
        f"{path}: expected a block dict, list, blocks wrapper, or block_NNN mapping"
    )


def _canonical_map(
    value: Any,
    *,
    name: str,
    temporal_size: int,
    spatial_shape: tuple[int, int],
) -> np.ndarray:
    array = _to_numpy(value)
    height, width = spatial_shape
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.shape == (temporal_size, height * width):
        array = array.reshape(temporal_size, height, width)
    if temporal_size == 1 and array.shape == (height, width):
        array = array[None]
    expected = (temporal_size, height, width)
    if array.shape != expected:
        raise SchemaError(f"{name} has shape {array.shape}; expected {expected}")
    return np.asarray(array)


def _canonical_native_role(
    value: Any, *, name: str, temporal_size: int
) -> np.ndarray:
    """Losslessly remove only an explicit singleton batch/channel axis."""
    array = _to_numpy(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    elif array.ndim == 5 and array.shape[0] == 1 and array.shape[2] == 1:
        array = array[0, :, 0]
    if array.ndim != 3 or array.shape[0] != temporal_size:
        raise SchemaError(
            f"{name} has shape {array.shape}; expected [1,T,H,W] or [T,H,W] "
            f"with T={temporal_size}"
        )
    return np.asarray(array)


def _resize_role_tuple_exact(
    role_maps: dict[str, np.ndarray],
    fields: tuple[str, ...],
    spatial_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Match PosteriorResidualFlowRouter._resize_roles exactly on CPU."""
    role_tensor = torch.from_numpy(
        np.stack([role_maps[name] for name in fields], axis=1)
    ).float()
    resized = torch.nn.functional.interpolate(
        role_tensor,
        size=spatial_shape,
        mode="bilinear",
        align_corners=False,
    )
    resized = resized.clamp_min(0.0)
    resized = resized / resized.sum(dim=1, keepdim=True).clamp_min(1e-6)
    values = resized.cpu().numpy()
    return {name: values[:, index] for index, name in enumerate(fields)}


def _load_optional_token_role_tuple(
    raw: dict[str, Any],
    *,
    native_fields: tuple[str, ...],
    suffix: str,
    temporal_size: int,
    spatial_shape: tuple[int, int],
) -> dict[str, np.ndarray] | None:
    names = tuple(f"{name}{suffix}" for name in native_fields)
    present = tuple(name for name in names if name in raw)
    if present and len(present) != len(names):
        missing = sorted(set(names) - set(present))
        raise SchemaError(
            "Partial token-grid role tuple is forbidden; missing " + ", ".join(missing)
        )
    if not present:
        return None
    return {
        native: _validate_probability(
            _canonical_map(
                raw[saved],
                name=saved,
                temporal_size=temporal_size,
                spatial_shape=spatial_shape,
            ),
            saved,
        )
        for native, saved in zip(native_fields, names)
    }


def _validate_mask(array: np.ndarray, name: str) -> np.ndarray:
    if not np.isfinite(array).all():
        raise SchemaError(f"{name} contains non-finite values")
    if not np.isin(array, (0, 1, False, True)).all():
        raise SchemaError(f"{name} must be binary")
    return array.astype(bool, copy=False)


def _validate_probability(array: np.ndarray, name: str) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise SchemaError(f"{name} contains non-finite values")
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise SchemaError(
            f"{name} must be in [0,1], observed [{minimum:.6g},{maximum:.6g}]"
        )
    return np.clip(array, 0.0, 1.0)


def _required(mapping: Any, key: str, scope: str) -> Any:
    if not isinstance(mapping, dict):
        raise SchemaError(f"{scope} must be a dictionary")
    if key not in mapping:
        raise SchemaError(f"{scope}: missing required field {key}")
    return mapping[key]


def _integer_indices(value: Any, name: str) -> np.ndarray:
    array = _to_numpy(value).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise SchemaError(f"{name} must be a non-empty finite vector")
    rounded = np.rint(array)
    if not np.array_equal(array, rounded):
        raise SchemaError(f"{name} must contain integer-valued indices")
    return rounded.astype(np.int64)


def _map_without_known_grid(value: Any, name: str, temporal_size: int) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim == 5 and array.shape[0] == 1 and array.shape[2] == 1:
        array = array[0, :, 0]
    elif array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[0] != temporal_size:
        raise SchemaError(
            f"{name} must have shape [1,T,H,W], [1,T,1,H,W], or [T,H,W]; "
            f"got {array.shape} for T={temporal_size}"
        )
    return np.asarray(array)


def _flattened_or_spatial_map(
    value: Any,
    *,
    name: str,
    temporal_size: int,
    candidate_shapes: tuple[tuple[int, int], ...],
) -> np.ndarray:
    array = _to_numpy(value)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2 and array.shape[0] == temporal_size:
        for height, width in candidate_shapes:
            if array.shape[1] == height * width:
                return array.reshape(temporal_size, height, width)
    if array.ndim == 1:
        for height, width in candidate_shapes:
            if array.size == temporal_size * height * width:
                return array.reshape(temporal_size, height, width)
    if array.ndim == 3 and array.shape[0] == temporal_size:
        if tuple(array.shape[-2:]) in candidate_shapes:
            return array
    # Layer aggregation can retain repeated batch rows.  They are safe to
    # collapse only when they are exactly the same captured map.
    if array.ndim == 2:
        for height, width in candidate_shapes:
            expected = temporal_size * height * width
            if array.shape[1] == expected and np.all(array == array[:1]):
                return array[0].reshape(temporal_size, height, width)
    raise SchemaError(
        f"{name} shape {array.shape} cannot be losslessly mapped to T={temporal_size}, "
        f"candidate grids={candidate_shapes}"
    )


def _resize_scalar_exact(
    value: np.ndarray,
    spatial_shape: tuple[int, int],
    *,
    mode: str,
) -> np.ndarray:
    if tuple(value.shape[-2:]) == spatial_shape:
        return value.astype(np.float32, copy=False)
    tensor = torch.from_numpy(np.asarray(value)).float().unsqueeze(1)
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    resized = torch.nn.functional.interpolate(
        tensor, size=spatial_shape, mode=mode, **kwargs
    )
    return resized[:, 0].cpu().numpy()


def _velocity_rms(value: Any, name: str, temporal_size: int) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 4 or array.shape[0] != temporal_size:
        raise SchemaError(
            f"velocity_step0.{name} must be [1,T,C,H,W] or [T,C,H,W], "
            f"got {array.shape}"
        )
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise SchemaError(f"velocity_step0.{name} contains non-finite values")
    return np.sqrt(np.mean(np.square(array, dtype=np.float32), axis=1))


def _prediction_index(name: str) -> int:
    match = re.fullmatch(r"prediction_(\d+)", name)
    if match is None:
        raise SchemaError(f"Invalid prediction key: {name}")
    return int(match.group(1))


def _validate_aggregated_map(entry: Any, scope: str) -> Any:
    _required(entry, "layers", scope)
    _required(entry, "by_layer", scope)
    return _required(entry, "mean", scope)


def _adapt_nested_hook(raw: dict[str, Any], source_path: Path) -> dict[str, Any]:
    """Convert the actual nested inference artifact without semantic guesses."""
    for section in ("metadata", "initial", "first_response", "velocity_step0", "appearance_access", "m2"):
        _required(raw, section, str(source_path))
    latent_index_field = (
        "global_latent_frame_indices"
        if "global_latent_frame_indices" in raw
        else "latent_frame_indices"
    )
    latent_indices = _integer_indices(
        _required(raw, latent_index_field, str(source_path)), latent_index_field
    )
    temporal_size = len(latent_indices)
    if (latent_indices < 0).any() or (np.diff(latent_indices) <= 0).any():
        raise SchemaError(f"{latent_index_field} must be non-negative and increasing")
    filename_match = re.search(r"block[_-]?(\d+)", source_path.stem)
    if filename_match is None:
        raise SchemaError("Nested hook blocks require a block_NNN filename")
    block_index = int(filename_match.group(1))

    initial = raw["initial"]
    first = raw["first_response"]
    velocity = raw["velocity_step0"]
    appearance = raw["appearance_access"]
    m2 = raw["m2"]
    for name in (
        "source_attention", "hand_occupancy", "hand_proximity",
        "temporal_confidence", "temporal_posterior", "connected_support",
        "object_posterior",
    ):
        _required(initial, name, "initial")
    for name in (
        "object_posterior_prior", "field_score", "field_observation",
        "refined_object_posterior", *ROLE_FIELDS, "role_entropy",
    ):
        _required(first, name, "first_response")
    for name in VELOCITY_VECTOR_FIELDS + COEFFICIENT_FIELDS:
        _required(velocity, name, "velocity_step0")
    for name in (
        "final_attention_request", "m2_canonical_read_gate",
        "m2_canonical_write_gate", "write_eligibility",
        "selected_write_tokens", "bank_initialized",
        "bank_initialized_before_block", "bank_initialization_chunk",
        "retrieval_maps",
    ):
        _required(m2, name, "m2")

    initial_object = _map_without_known_grid(
        initial["object_posterior"], "initial.object_posterior", temporal_size
    )
    spatial_shape = tuple(int(value) for value in initial_object.shape[-2:])
    raw_roles = {
        name: _map_without_known_grid(first[name], f"first_response.{name}", temporal_size)
        for name in ROLE_FIELDS
    }
    role_shapes = {tuple(value.shape[-2:]) for value in raw_roles.values()}
    if len(role_shapes) != 1:
        raise SchemaError(f"first_response q maps do not share one grid: {role_shapes}")
    role_shape = next(iter(role_shapes))

    expected_predictions = {"prediction_000", "prediction_007", "prediction_014"}
    missing_predictions = sorted(expected_predictions - set(appearance))
    if missing_predictions:
        raise SchemaError(
            "appearance_access missing required predictions: " + ", ".join(missing_predictions)
        )
    for prediction_name, prediction in appearance.items():
        _prediction_index(prediction_name)
        _required(prediction, "source", f"appearance_access.{prediction_name}")
        map_entry = _required(prediction, "map", f"appearance_access.{prediction_name}")
        _validate_aggregated_map(map_entry, f"appearance_access.{prediction_name}.map")
    appearance_maps: dict[str, np.ndarray] = {}
    appearance_sources: dict[str, str] = {}
    for prediction_name, prediction in appearance.items():
        mean = _validate_aggregated_map(
            prediction["map"], f"appearance_access.{prediction_name}.map"
        )
        native = _flattened_or_spatial_map(
            mean,
            name=f"appearance_access.{prediction_name}.map.mean",
            temporal_size=temporal_size,
            candidate_shapes=(spatial_shape, role_shape),
        )
        appearance_maps[prediction_name] = _resize_scalar_exact(
            native, spatial_shape, mode="area"
        )
        appearance_sources[prediction_name] = str(prediction["source"])
    final_appearance_name = max(appearance, key=_prediction_index)

    retrieval = m2["retrieval_maps"]
    if not isinstance(retrieval, dict):
        raise SchemaError("m2.retrieval_maps must be a dictionary")
    retrieval_summary: dict[str, Any] = {}
    for prediction_name, prediction in retrieval.items():
        _prediction_index(prediction_name)
        mean = _validate_aggregated_map(
            prediction, f"m2.retrieval_maps.{prediction_name}"
        )
        if not isinstance(mean, dict):
            raise SchemaError(f"m2.retrieval_maps.{prediction_name}.mean must be a dict")
        required_retrieval = {
            "admitted", "gate", "correction_rms", "native_rms", "joint_similarity"
        }
        missing = sorted(required_retrieval - set(mean))
        if missing:
            raise SchemaError(
                f"m2.retrieval_maps.{prediction_name}.mean missing: {', '.join(missing)}"
            )
        retrieval_summary[prediction_name] = mean

    adapted: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "block_index": block_index,
        "latent_frame_indices": latent_indices,
        "spatial_shape": np.asarray(spatial_shape, dtype=np.int64),
        # Every spatial token for every explicitly listed real latent frame is
        # structurally valid; the hook has no padded-token field.
        "valid_token": np.ones((temporal_size, *spatial_shape), dtype=bool),
        "connected_object_support": initial["connected_support"],
        "hand_occupancy": initial["hand_occupancy"],
        "source_attention": initial["source_attention"],
        "hand_proximity": initial["hand_proximity"],
        "temporal_confidence": initial["temporal_confidence"],
        "temporal_posterior": initial["temporal_posterior"],
        "initial_object_posterior": initial_object,
        "appearance_access": appearance_maps[final_appearance_name],
        "reference_read_gate": _flattened_or_spatial_map(
            m2["final_attention_request"],
            name="m2.final_attention_request",
            temporal_size=temporal_size,
            candidate_shapes=(spatial_shape,),
        ),
        "reference_write_gate": _flattened_or_spatial_map(
            m2["write_eligibility"],
            name="m2.write_eligibility",
            temporal_size=temporal_size,
            candidate_shapes=(spatial_shape,),
        ),
        **raw_roles,
    }
    for output_name, source_name in (
        ("object_posterior_prior", "object_posterior_prior"),
        ("object_posterior", "refined_object_posterior"),
        ("field_score", "field_score"),
        ("field_observation", "field_observation"),
        ("role_entropy", "role_entropy"),
    ):
        native = _map_without_known_grid(
            first[source_name], f"first_response.{source_name}", temporal_size
        )
        adapted[output_name] = _resize_scalar_exact(
            native, spatial_shape, mode="area"
        )
    for name in COEFFICIENT_FIELDS:
        native = _map_without_known_grid(
            velocity[name], f"velocity_step0.{name}", temporal_size
        )
        adapted[name] = _resize_scalar_exact(native, spatial_shape, mode="area")
    raw_velocity_maps: dict[str, np.ndarray] = {}
    for vector_name, magnitude_name in zip(VELOCITY_VECTOR_FIELDS, VELOCITY_FIELDS):
        magnitude = _velocity_rms(velocity[vector_name], vector_name, temporal_size)
        raw_velocity_maps[magnitude_name] = magnitude
        adapted[magnitude_name] = _resize_scalar_exact(
            magnitude, spatial_shape, mode="area"
        )
    for name in ROLE_FIELDS:
        token_name = f"{name}_token_grid"
        if token_name in first:
            adapted[token_name] = first[token_name]
    adapted["__raw_velocity_maps"] = raw_velocity_maps
    adapted["__velocity_alignment"] = "channel_RMS_then_area_to_common_token_grid"
    adapted["__hook_extras"] = {
        "metadata": raw["metadata"],
        "appearance_final_prediction": final_appearance_name,
        "appearance_final_source": appearance[final_appearance_name]["source"],
        "appearance_prediction_maps": appearance_maps,
        "appearance_prediction_sources": appearance_sources,
        "m2_canonical_read_gate": m2["m2_canonical_read_gate"],
        "m2_canonical_write_gate": m2["m2_canonical_write_gate"],
        "selected_write_tokens": m2["selected_write_tokens"],
        "bank_initialized": m2["bank_initialized"],
        "bank_initialized_before_block": m2["bank_initialized_before_block"],
        "bank_initialization_chunk": m2["bank_initialization_chunk"],
        "retrieval_mean_maps": retrieval_summary,
    }
    return adapted


def validate_block(raw: dict[str, Any], source_path: Path) -> Block:
    if all(
        section in raw
        for section in (
            "initial",
            "first_response",
            "velocity_step0",
            "appearance_access",
            "m2",
        )
    ):
        raw = _adapt_nested_hook(raw, source_path)
    scalar_fields = (
        "schema_version", "block_index", "spatial_shape", "latent_frame_indices"
    )
    missing = [name for name in scalar_fields if name not in raw]
    hand_fields = [name for name in ("hand_probability", "hand_occupancy") if name in raw]
    common_maps = (
        MASK_FIELDS
        + AUX_PROBABILITY_FIELDS
        + COEFFICIENT_FIELDS
        + PERMISSION_FIELDS
        + VELOCITY_FIELDS
    )
    required_maps = common_maps + ROLE_FIELDS
    missing.extend(name for name in required_maps if name not in raw)
    if not hand_fields:
        missing.append("hand_probability|hand_occupancy")
    if missing:
        raise SchemaError(f"{source_path}: missing required fields: {', '.join(missing)}")

    version = str(_scalar(raw["schema_version"], "schema_version"))
    if version != SCHEMA_VERSION:
        raise SchemaError(
            f"{source_path}: schema_version={version!r}; expected {SCHEMA_VERSION!r}"
        )
    block_index = int(_scalar(raw["block_index"], "block_index"))
    if block_index < 0:
        raise SchemaError("block_index must be non-negative")
    latent_indices = _integer_indices(
        raw["latent_frame_indices"], "latent_frame_indices"
    )
    if (latent_indices < 0).any() or (np.diff(latent_indices) <= 0).any():
        raise SchemaError("latent_frame_indices must be non-negative and strictly increasing")
    spatial = _to_numpy(raw["spatial_shape"]).reshape(-1)
    if spatial.size != 2 or not np.issubdtype(spatial.dtype, np.integer):
        raise SchemaError("spatial_shape must be integer [height,width]")
    spatial_shape = (int(spatial[0]), int(spatial[1]))
    if min(spatial_shape) <= 0:
        raise SchemaError("spatial_shape entries must be positive")

    maps: dict[str, np.ndarray] = {}
    for name in common_maps + tuple(hand_fields):
        maps[name] = _canonical_map(
            raw[name],
            name=name,
            temporal_size=len(latent_indices),
            spatial_shape=spatial_shape,
        )
    for name in EVIDENCE_FIELDS:
        if name in raw:
            maps[name] = _validate_probability(
                _canonical_map(
                    raw[name], name=name, temporal_size=len(latent_indices),
                    spatial_shape=spatial_shape,
                ),
                name,
            )
    for name in MASK_FIELDS:
        maps[name] = _validate_mask(maps[name], name)
    for name in (
        AUX_PROBABILITY_FIELDS
        + COEFFICIENT_FIELDS
        + PERMISSION_FIELDS
        + tuple(hand_fields)
    ):
        maps[name] = _validate_probability(maps[name], name)

    raw_role_maps: dict[str, np.ndarray] = {}
    for name in ROLE_FIELDS:
        raw_role_maps[name] = _validate_probability(
            _canonical_native_role(
                raw[name], name=name, temporal_size=len(latent_indices)
            ),
            name,
        )
    raw_shapes = {value.shape[-2:] for value in raw_role_maps.values()}
    if len(raw_shapes) != 1:
        raise SchemaError(f"Native q role maps must share one grid, got {raw_shapes}")
    raw_role_shape = tuple(int(value) for value in next(iter(raw_shapes)))
    total = np.stack(
        [raw_role_maps[name] for name in ROLE_FIELDS], axis=-1
    ).sum(axis=-1)
    if not np.allclose(total, 1.0, atol=5e-3, rtol=0.0):
        raise SchemaError(
            f"{source_path}: native posterior roles must sum to one; observed "
            f"[{total.min():.6g},{total.max():.6g}]"
        )

    saved_posterior = _load_optional_token_role_tuple(
        raw,
        native_fields=ROLE_FIELDS,
        suffix="_token_grid",
        temporal_size=len(latent_indices),
        spatial_shape=spatial_shape,
    )
    resized_posterior = _resize_role_tuple_exact(raw_role_maps, ROLE_FIELDS, spatial_shape)
    maps.update(saved_posterior or resized_posterior)
    role_alignment = (
        "hook_saved_token_grid"
        if saved_posterior is not None
        else "PosteriorResidualFlowRouter_bilinear_align_corners_false_clamp_renormalize"
    )
    for name in VELOCITY_FIELDS:
        maps[name] = maps[name].astype(np.float32, copy=False)
        if not np.isfinite(maps[name]).all() or (maps[name] < 0).any():
            raise SchemaError(f"{name} must contain finite non-negative magnitudes")
    hidden_raw_velocity = raw.get("__raw_velocity_maps")
    if hidden_raw_velocity is None:
        raw_velocity_maps = {
            name: maps[name].copy() for name in VELOCITY_FIELDS
        }
        velocity_alignment = "already_on_common_token_grid"
    else:
        if not isinstance(hidden_raw_velocity, dict):
            raise SchemaError("__raw_velocity_maps must be a dictionary")
        raw_velocity_maps = {}
        for name in VELOCITY_FIELDS:
            if name not in hidden_raw_velocity:
                raise SchemaError(f"__raw_velocity_maps missing {name}")
            value = _map_without_known_grid(
                hidden_raw_velocity[name], name, len(latent_indices)
            ).astype(np.float32, copy=False)
            if not np.isfinite(value).all() or (value < 0).any():
                raise SchemaError(f"raw {name} must be finite and non-negative")
            raw_velocity_maps[name] = value
        velocity_alignment = str(
            raw.get("__velocity_alignment", "unspecified")
        )
    velocity_shapes = {value.shape[-2:] for value in raw_velocity_maps.values()}
    if len(velocity_shapes) != 1:
        raise SchemaError(f"Raw velocity magnitudes must share one grid: {velocity_shapes}")
    velocity_shape = tuple(int(value) for value in next(iter(velocity_shapes)))
    valid = maps["valid_token"]
    if not valid.any():
        raise SchemaError(f"{source_path}: valid_token is empty")
    total = np.stack([maps[name] for name in ROLE_FIELDS], axis=-1).sum(axis=-1)
    if not np.allclose(total[valid], 1.0, atol=5e-3, rtol=0.0):
        observed = total[valid]
        raise SchemaError(
            f"{source_path}: common-grid posterior roles must sum to one on "
            f"valid tokens; observed [{observed.min():.6g},{observed.max():.6g}]"
        )
    return Block(
        block_index,
        latent_indices,
        spatial_shape,
        maps,
        raw_role_maps,
        raw_role_shape,
        role_alignment,
        raw_velocity_maps,
        velocity_shape,
        velocity_alignment,
        raw.get("__hook_extras", {}),
        hand_fields[0],
        str(source_path.resolve()),
    )


def load_blocks(path: Path) -> list[Block]:
    blocks: list[Block] = []
    for artifact_path in discover_artifacts(path):
        try:
            payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        except TypeError:  # torch < 2.0
            payload = torch.load(artifact_path, map_location="cpu")
        for raw in _extract_blocks(payload, artifact_path):
            blocks.append(validate_block(raw, artifact_path))
    if not blocks:
        raise SchemaError("No block dictionaries were loaded")
    blocks.sort(key=lambda block: block.index)
    block_indices = [block.index for block in blocks]
    if len(set(block_indices)) != len(block_indices):
        raise SchemaError(f"Duplicate block_index values: {block_indices}")
    spatial_shapes = {block.spatial_shape for block in blocks}
    if len(spatial_shapes) != 1:
        raise SchemaError(f"All blocks must share one spatial_shape, got {spatial_shapes}")
    raw_role_shapes = {block.raw_role_spatial_shape for block in blocks}
    if len(raw_role_shapes) != 1:
        raise SchemaError(
            f"All blocks must share one native role grid, got {raw_role_shapes}"
        )
    latent = np.concatenate([block.latent_frame_indices for block in blocks])
    if len(np.unique(latent)) != len(latent):
        raise SchemaError("latent_frame_indices overlap across blocks; hook must emit unique frames")
    if (np.diff(latent) <= 0).any():
        raise SchemaError("Blocks must be ordered by strictly increasing latent_frame_indices")
    return blocks


def load_temporal_groups(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing hand-role input: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "causal_temporal_groups" not in data:
            raise SchemaError(f"{path}: missing causal_temporal_groups")
        groups = np.asarray(data["causal_temporal_groups"])
    if groups.ndim != 2 or groups.shape[1] != 2 or not np.issubdtype(groups.dtype, np.integer):
        raise SchemaError("causal_temporal_groups must be integer [latent_frames,2]")
    groups = groups.astype(np.int64)
    if (groups[:, 0] < 0).any() or (groups[:, 1] <= groups[:, 0]).any():
        raise SchemaError("causal_temporal_groups must contain non-empty [start,end) ranges")
    if (np.diff(groups[:, 0]) < 0).any() or (np.diff(groups[:, 1]) < 0).any():
        raise SchemaError("causal_temporal_groups must be monotonic")
    return groups


def select_interface_token(blocks: list[Block]) -> TokenSelection:
    best: TokenSelection | None = None
    for block_position, block in enumerate(blocks):
        maps = block.maps
        candidate = (
            maps["valid_token"]
            & maps["connected_object_support"]
            & (maps[block.hand_field] > 0.0)
        )
        if not candidate.any():
            continue
        scores = np.where(candidate, maps["q_interface"], -np.inf)
        flat = int(np.argmax(scores))
        local_frame, row, column = np.unravel_index(flat, scores.shape)
        current = TokenSelection(
            int(block_position),
            int(local_frame),
            int(row),
            int(column),
            float(scores.flat[flat]),
        )
        if best is None or current.score > best.score:
            best = current
    if best is None:
        raise SchemaError(
            "No token satisfies valid_token & connected_object_support & "
            "(hand_probability/hand_occupancy > 0)"
        )
    return best


def _valid_values(blocks: list[Block], field: str) -> np.ndarray:
    values = [block.maps[field][block.maps["valid_token"]] for block in blocks]
    return np.concatenate(values).astype(np.float64)


def _distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "quantiles": {}}
    quantile_values = np.quantile(values, QUANTILES)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "quantiles": {
            f"p{int(round(q * 100)):02d}": float(value)
            for q, value in zip(QUANTILES, quantile_values)
        },
    }


def velocity_statistics(blocks: list[Block]) -> tuple[float, dict[str, Any]]:
    raw_values = {
        name: np.concatenate(
            [block.raw_velocity_maps[name].reshape(-1) for block in blocks]
        ).astype(np.float64)
        for name in VELOCITY_FIELDS
    }
    by_field = {name: _distribution(value) for name, value in raw_values.items()}
    pooled = np.concatenate(list(raw_values.values()))
    pooled_stats = _distribution(pooled)
    p99 = float(np.quantile(pooled, 0.99))
    if p99 <= 0.0:
        p99 = 1.0
    return p99, {"pooled": pooled_stats, "by_field": by_field}


def _weighted_statistics(values: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    denominator = float(weights.sum())
    if denominator <= 0.0:
        return {"weight_sum": 0.0, "mean": None, "rms": None}
    return {
        "weight_sum": denominator,
        "mean": float(np.sum(values * weights) / denominator),
        "rms": float(math.sqrt(np.sum(values * values * weights) / denominator)),
    }


def compute_metrics(blocks: list[Block], selected: TokenSelection) -> dict[str, Any]:
    analysis_fields = (
        ROLE_FIELDS
        + AUX_PROBABILITY_FIELDS
        + COEFFICIENT_FIELDS
        + PERMISSION_FIELDS
        + VELOCITY_FIELDS
    )
    role_chunks = []
    field_chunks: dict[str, list[np.ndarray]] = {name: [] for name in analysis_fields}
    for block in blocks:
        valid = block.maps["valid_token"]
        role_chunks.append(np.stack([block.maps[name][valid] for name in ROLE_FIELDS], axis=-1))
        for name in analysis_fields:
            field_chunks[name].append(block.maps[name][valid].astype(np.float64))
    roles = np.concatenate(role_chunks, axis=0)
    values = {name: np.concatenate(chunks) for name, chunks in field_chunks.items()}
    dominant = roles.argmax(axis=-1)
    regions: dict[str, Any] = {}
    for role_index, label in enumerate(ROLE_LABELS):
        hard_mask = dominant == role_index
        hard = {
            "token_count": int(hard_mask.sum()),
            "fraction": float(hard_mask.mean()),
            "fields": {
                name: _distribution(field_values[hard_mask])
                for name, field_values in values.items()
            },
        }
        weights = roles[:, role_index].astype(np.float64)
        soft = {
            "weight_sum": float(weights.sum()),
            "weight_fraction": float(weights.sum() / max(float(weights.size), 1.0)),
            "fields": {
                name: _weighted_statistics(field_values, weights)
                for name, field_values in values.items()
            },
        }
        regions[label] = {"hard_argmax": hard, "soft_weighted": soft}

    role_sum_error = np.abs(roles.sum(axis=-1) - 1.0)
    rho = values["rho_magnitude"]
    removed = values["removed_magnitude"]
    initial_support = values["initial_object_posterior"]
    refined_support = values["object_posterior"]

    anchor_blocks: list[dict[str, Any]] = []
    selected_write_total = 0
    selected_write_role_valid = 0
    retrieval_requested = 0
    retrieval_matched = 0.0
    retrieval_correction_energy = 0.0
    retrieval_native_energy = 0.0
    retrieval_observations = 0
    for block in blocks:
        extras = block.hook_extras
        if not extras:
            continue
        initialized = bool(float(_scalar(extras["bank_initialized"], "bank_initialized")))
        initialized_before = bool(float(_scalar(
            extras["bank_initialized_before_block"], "bank_initialized_before_block"
        )))
        initialization_chunk = int(float(_scalar(
            extras["bank_initialization_chunk"], "bank_initialization_chunk"
        )))
        selected_write = _flattened_or_spatial_map(
            extras["selected_write_tokens"], name="selected_write_tokens",
            temporal_size=len(block.latent_frame_indices),
            candidate_shapes=(block.spatial_shape,),
        ) > 0.5
        dominant_block = np.stack(
            [block.maps[name] for name in ROLE_FIELDS], axis=-1
        ).argmax(axis=-1)
        selected_write_total += int(selected_write.sum())
        selected_write_role_valid += int(
            (selected_write & ((dominant_block == 0) | (dominant_block == 1))).sum()
        )
        retrieval = extras.get("retrieval_mean_maps", {})
        retrieval_active = bool(retrieval)
        for prediction in retrieval.values():
            admitted = _flattened_or_spatial_map(
                prediction["admitted"], name="retrieval.admitted",
                temporal_size=len(block.latent_frame_indices),
                candidate_shapes=(block.spatial_shape,),
            )
            correction = _flattened_or_spatial_map(
                prediction["correction_rms"], name="retrieval.correction_rms",
                temporal_size=len(block.latent_frame_indices),
                candidate_shapes=(block.spatial_shape,),
            )
            native = _flattened_or_spatial_map(
                prediction["native_rms"], name="retrieval.native_rms",
                temporal_size=len(block.latent_frame_indices),
                candidate_shapes=(block.spatial_shape,),
            )
            requested = block.maps["reference_read_gate"] > 0
            retrieval_requested += int(requested.sum())
            retrieval_matched += float(admitted[requested].sum())
            matched = admitted > 0
            retrieval_correction_energy += float(np.square(correction[matched]).sum())
            retrieval_native_energy += float(np.square(native[matched]).sum())
            retrieval_observations += int(matched.sum())
        anchor_blocks.append({
            "block_index": block.index,
            "anchor_initialized_before_block": initialized_before,
            "anchor_initialized_after_block": initialized,
            "anchor_initialization_chunk": initialization_chunk,
            "retrieval_executed": retrieval_active,
            "retrieval_prediction_count": len(retrieval),
            "selected_write_token_count": int(selected_write.sum()),
        })

    requested_region_summary: dict[str, Any] = {}
    for label in ROLE_LABELS:
        hard = regions[label]["hard_argmax"]
        requested_region_summary[label] = {
            "token_count": hard["token_count"],
            "mean_q_interface": hard["fields"]["q_interface"]["mean"],
            "mean_source_correction_magnitude": hard["fields"]["rho_role_magnitude"]["mean"],
            "mean_appearance_access": hard["fields"]["appearance_access"]["mean"],
            "write_eligibility_rate_at_0_5": (
                float(np.mean(values["reference_write_gate"][dominant == ROLE_LABELS.index(label)] >= 0.5))
                if hard["token_count"] else None
            ),
        }
    requested_statistics = {
        "role_probability_sum_max_error": float(role_sum_error.max(initial=0.0)),
        "interface_region_mean_q_interface": requested_region_summary["Interface"]["mean_q_interface"],
        "regions": requested_region_summary,
        "conflicting_energy_removed_fraction": (
            float(np.square(removed).sum() / max(float(np.square(rho).sum()), 1e-12))
        ),
        "support_area": {
            "initial_soft_token_sum": float(initial_support.sum()),
            "refined_soft_token_sum": float(refined_support.sum()),
            "initial_mean": float(initial_support.mean()),
            "refined_mean": float(refined_support.mean()),
            "soft_area_change": float(refined_support.sum() - initial_support.sum()),
        },
        "anchor": {
            "blocks": anchor_blocks,
            "initialized": bool(anchor_blocks and anchor_blocks[-1]["anchor_initialized_after_block"]),
            "initialization_chunk": (
                anchor_blocks[-1]["anchor_initialization_chunk"] if anchor_blocks else None
            ),
            "selected_write_token_count": selected_write_total,
            "write_token_object_or_interface_purity": (
                float(selected_write_role_valid / selected_write_total)
                if selected_write_total else None
            ),
            "retrieval_match_rate_over_requested_tokens_and_predictions": (
                float(retrieval_matched / retrieval_requested)
                if retrieval_requested else None
            ),
            "retrieval_matched_token_observations": retrieval_observations,
            "correction_to_native_output_rms": (
                float(math.sqrt(retrieval_correction_energy / retrieval_native_energy))
                if retrieval_native_energy > 0 else None
            ),
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "valid_token_count": int(len(dominant)),
        "region_assignment": "argmax(q_object,q_interface,q_hand,q_background)",
        "regions": regions,
        "requested_statistics": requested_statistics,
        "selected_interface_token": token_card_values(blocks, selected),
    }


def token_card_values(blocks: list[Block], selected: TokenSelection) -> dict[str, Any]:
    block = blocks[selected.block_position]
    index = (selected.local_frame, selected.row, selected.column)
    fields = (
        (block.hand_field,)
        + ROLE_FIELDS
        + AUX_PROBABILITY_FIELDS
        + COEFFICIENT_FIELDS
        + PERMISSION_FIELDS
        + VELOCITY_FIELDS
    )
    values = {name: float(block.maps[name][index]) for name in fields}
    candidate_scores: list[np.ndarray] = []
    for candidate_block in blocks:
        candidate = (
            candidate_block.maps["valid_token"]
            & candidate_block.maps["connected_object_support"]
            & (candidate_block.maps[candidate_block.hand_field] > 0)
        )
        candidate_scores.append(candidate_block.maps["q_interface"][candidate])
    all_candidate_scores = np.concatenate(candidate_scores)
    max_tie_count = int(np.isclose(
        all_candidate_scores, selected.score, atol=1e-7, rtol=0.0
    ).sum())
    return {
        "selection_rule": (
            "argmax q_interface within valid_token & connected_object_support "
            "& hand_evidence>0"
        ),
        "block_index": block.index,
        "latent_frame_index": int(block.latent_frame_indices[selected.local_frame]),
        "local_frame_index": selected.local_frame,
        "token_row": selected.row,
        "token_column": selected.column,
        "spatial_shape": list(block.spatial_shape),
        "connected_object_support": bool(block.maps["connected_object_support"][index]),
        "valid_token": bool(block.maps["valid_token"][index]),
        "candidate_count": int(all_candidate_scores.size),
        "maximum_q_interface_tie_count": max_tie_count,
        "tie_break": "block order, then local frame, row, column",
        "values": values,
    }


def role_composite(block: Block, local_frame: int) -> tuple[np.ndarray, np.ndarray]:
    roles = np.stack([block.maps[name][local_frame] for name in ROLE_FIELDS], axis=-1)
    dominant = roles.argmax(axis=-1)
    confidence = roles.max(axis=-1)
    brightness = 0.25 + 0.75 * confidence
    rgb = ROLE_COLORS[dominant] * brightness[..., None]
    rgb[~block.maps["valid_token"][local_frame]] = 0.0
    mixed = (
        (block.maps["q_interface"][local_frame] >= MIXED_INTERFACE_THRESHOLD)
        & (block.maps["q_interface"][local_frame] < 1.0 - 1e-6)
        & ((roles.sum(axis=-1) - block.maps["q_interface"][local_frame]) > 1e-6)
        & block.maps["valid_token"][local_frame]
    )
    return np.clip(rgb, 0.0, 1.0), mixed


def read_video(path: Path) -> Video:
    if not path.is_file():
        raise FileNotFoundError(f"Missing video: {path}")
    frames: list[np.ndarray] = []
    fps = 0.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.average_rate is not None:
            fps = float(stream.average_rate)
        for frame in container.decode(video=0):
            frames.append(frame.to_rgb().to_ndarray())
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return Video(np.stack(frames), fps)


def validate_alignment(
    blocks: list[Block], groups: np.ndarray, source: Video, edited: Video
) -> None:
    latent = np.concatenate([block.latent_frame_indices for block in blocks])
    if int(latent.max()) >= len(groups):
        raise SchemaError(
            f"Artifact latent index {latent.max()} exceeds {len(groups)} causal groups"
        )
    end = int(groups[latent, 1].max())
    if end > len(source.frames) or end > len(edited.frames):
        raise SchemaError(
            f"causal_temporal_groups require {end} pixel frames, but source/edited "
            f"have {len(source.frames)}/{len(edited.frames)}"
        )


def _record_sequence(blocks: list[Block]) -> list[tuple[int, int]]:
    return [
        (block_position, local_frame)
        for block_position, block in enumerate(blocks)
        for local_frame in range(len(block.latent_frame_indices))
    ]


def choose_key_records(
    blocks: list[Block], selected: TokenSelection, maximum: int
) -> list[tuple[int, int]]:
    if maximum < 1:
        raise ValueError("max-keyframes must be positive")
    records = _record_sequence(blocks)
    selected_record = (selected.block_position, selected.local_frame)
    count = min(maximum, len(records))
    if count == 1:
        chosen = {selected_record}
    else:
        required = [selected_record, records[-1]]
        if count >= 3:
            required.append(records[0])
        chosen = set(required)
        positions = np.linspace(0, len(records) - 1, count).round().astype(int).tolist()
        for position in positions:
            if len(chosen) >= count:
                break
            chosen.add(records[position])
        if len(chosen) < count:
            for record in records:
                if len(chosen) >= count:
                    break
                chosen.add(record)
    return [record for record in records if record in chosen]


def representative_pixel_frame(groups: np.ndarray, latent_index: int) -> int:
    start, end = (int(value) for value in groups[latent_index])
    return start + (end - start - 1) // 2


def _add_mixed_border(axis: plt.Axes, mixed: np.ndarray) -> None:
    if mixed.any() and not mixed.all():
        axis.contour(
            mixed.astype(np.float32),
            levels=(0.5,),
            colors=(ROLE_COLORS[1],),
            linewidths=1.5,
        )


def _map_axis(
    axis: plt.Axes,
    value: np.ndarray,
    title: str,
    *,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    mixed: np.ndarray | None = None,
) -> None:
    axis.imshow(value, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    if mixed is not None:
        _add_mixed_border(axis, mixed)
    axis.set_title(title, fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])


def save_role_maps(blocks: list[Block], records: list[tuple[int, int]], path: Path) -> None:
    figure, axes = plt.subplots(len(records), 5, figsize=(13, 2.6 * len(records)), squeeze=False)
    for row, (block_position, local_frame) in enumerate(records):
        block = blocks[block_position]
        rgb, mixed = role_composite(block, local_frame)
        axis = axes[row, 0]
        axis.imshow(rgb, interpolation="nearest")
        _add_mixed_border(axis, mixed)
        axis.set_title(
            f"Composite B{block.index} L{block.latent_frame_indices[local_frame]}", fontsize=8
        )
        axis.set_xticks([]); axis.set_yticks([])
        for column, (field, label) in enumerate(zip(ROLE_FIELDS, ROLE_LABELS), start=1):
            _map_axis(axes[row, column], block.maps[field][local_frame], f"q {label}")
    figure.suptitle(
        "Soft roles: dominant hue, confidence brightness; purple border = mixed interface",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_permission_maps(
    blocks: list[Block],
    records: list[tuple[int, int]],
    velocity_p99: float,
    path: Path,
) -> None:
    figure, axes = plt.subplots(len(records), 6, figsize=(16, 2.5 * len(records)), squeeze=False)
    for row, (block_position, local_frame) in enumerate(records):
        block = blocks[block_position]
        _, mixed = role_composite(block, local_frame)
        appearance_maps = block.hook_extras.get("appearance_prediction_maps", {})
        panels = (
            ("Source Retention ||rho_role||", block.maps["rho_role_magnitude"][local_frame], velocity_p99),
            ("Appearance first: base", appearance_maps.get("prediction_000", block.maps["appearance_access"])[local_frame], 1.0),
            ("Appearance middle: role", appearance_maps.get("prediction_007", block.maps["appearance_access"])[local_frame], 1.0),
            ("Appearance final: role", appearance_maps.get("prediction_014", block.maps["appearance_access"])[local_frame], 1.0),
            (
                "Anchor Read Access" if bool(float(_scalar(
                    block.hook_extras.get("bank_initialized_before_block", 1.0),
                    "bank_initialized_before_block",
                ))) else "Anchor not initialized\n(read request only)",
                block.maps["reference_read_gate"][local_frame], 1.0,
            ),
            ("Reference Write Eligibility", block.maps["reference_write_gate"][local_frame], 1.0),
        )
        for column, (label, value, vmax) in enumerate(panels):
            _map_axis(
                axes[row, column],
                value,
                f"{label}\nB{block.index} L{block.latent_frame_indices[local_frame]}",
                cmap="magma",
                vmax=vmax,
                mixed=mixed,
            )
    figure.suptitle(
        "Operation-specific permissions: source retention / appearance access / selective write / broader read",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_refinement(blocks: list[Block], records: list[tuple[int, int]], path: Path) -> None:
    figure, axes = plt.subplots(len(records), 6, figsize=(15, 2.5 * len(records)), squeeze=False)
    for row, (block_position, local_frame) in enumerate(records):
        block = blocks[block_position]
        initial = block.maps["initial_object_posterior"][local_frame]
        prior = block.maps["object_posterior_prior"][local_frame]
        refined = block.maps["object_posterior"][local_frame]
        composite, mixed = role_composite(block, local_frame)
        _map_axis(axes[row, 0], initial, "Initial Object Support")
        _map_axis(axes[row, 1], prior, "Source-Grounded Prior")
        _map_axis(axes[row, 2], block.maps["field_score"][local_frame], "Branch-Response Evidence")
        _map_axis(axes[row, 3], block.maps["field_observation"][local_frame], "Restricted Response")
        _map_axis(axes[row, 4], refined, "Refined Object Support")
        axes[row, 5].imshow(composite, interpolation="nearest")
        _add_mixed_border(axes[row, 5], mixed)
        axes[row, 5].set_title(
            f"Final Soft Roles\nB{block.index} L{block.latent_frame_indices[local_frame]}"
        )
        axes[row, 5].set_xticks([]); axes[row, 5].set_yticks([])
    figure.suptitle(
        "Initial Object Support → Branch-Response Evidence → Refined Object Support → Final Soft Roles",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_velocity_control(
    blocks: list[Block], records: list[tuple[int, int]], p99: float, path: Path
) -> None:
    figure, axes = plt.subplots(len(records), len(VELOCITY_FIELDS), figsize=(15, 2.6 * len(records)), squeeze=False)
    for row, (block_position, local_frame) in enumerate(records):
        block = blocks[block_position]
        for column, field in enumerate(VELOCITY_FIELDS):
            _map_axis(
                axes[row, column],
                block.raw_velocity_maps[field][local_frame],
                f"{field}\nB{block.index} L{block.latent_frame_indices[local_frame]}",
                cmap="inferno",
                vmax=p99,
            )
    figure.suptitle(f"Velocity magnitudes; one run-shared robust p99 scale = {p99:.6g}", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _label_image(frame: np.ndarray, label: str, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(frame).resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline="#666666")
    box = draw.textbbox((0, 0), label, font=_font(14))
    draw.rectangle((3, 3, box[2] + 9, box[3] + 8), fill="black")
    draw.text((6, 4), label, fill="white", font=_font(14))
    return image


def save_keyframes(
    blocks: list[Block],
    records: list[tuple[int, int]],
    groups: np.ndarray,
    source: Video,
    edited: Video,
    path: Path,
    keyframe_dir: Path,
) -> list[int]:
    size = (260, 150)
    canvas = Image.new("RGB", (len(records) * size[0], 2 * size[1]), "white")
    pixel_indices: list[int] = []
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    for column, (block_position, local_frame) in enumerate(records):
        latent = int(blocks[block_position].latent_frame_indices[local_frame])
        pixel = representative_pixel_frame(groups, latent)
        pixel_indices.append(pixel)
        for row, (label, video) in enumerate((("source", source), ("edited", edited))):
            tile = _label_image(video.frames[pixel], f"{label} F{pixel} / L{latent}", size)
            canvas.paste(tile, (column * size[0], row * size[1]))
            Image.fromarray(video.frames[pixel]).save(
                keyframe_dir / f"{label}_frame_{pixel:04d}_latent_{latent:03d}.png"
            )
    canvas.save(path)
    return pixel_indices


def token_crop_box(
    frame_shape: tuple[int, ...], spatial_shape: tuple[int, int], row: int, column: int, radius: int
) -> tuple[int, int, int, int]:
    if radius < 0:
        raise ValueError("crop-radius-tokens must be non-negative")
    frame_height, frame_width = frame_shape[:2]
    token_height, token_width = spatial_shape
    x0 = max(0, math.floor((column - radius) * frame_width / token_width))
    y0 = max(0, math.floor((row - radius) * frame_height / token_height))
    x1 = min(frame_width, math.ceil((column + radius + 1) * frame_width / token_width))
    y1 = min(frame_height, math.ceil((row + radius + 1) * frame_height / token_height))
    if x1 <= x0 or y1 <= y0:
        raise SchemaError("Selected token produced an empty pixel crop")
    return x0, y0, x1, y1


def save_interface_crop(
    blocks: list[Block],
    selected: TokenSelection,
    groups: np.ndarray,
    source: Video,
    edited: Video,
    radius: int,
    path: Path,
    keyframe_dir: Path,
) -> dict[str, Any]:
    block = blocks[selected.block_position]
    latent = int(block.latent_frame_indices[selected.local_frame])
    pixel = representative_pixel_frame(groups, latent)
    source_box = token_crop_box(
        source.frames[pixel].shape, block.spatial_shape, selected.row, selected.column, radius
    )
    edited_box = token_crop_box(
        edited.frames[pixel].shape, block.spatial_shape, selected.row, selected.column, radius
    )
    tiles = []
    for label, video, box in (("source", source, source_box), ("edited", edited, edited_box)):
        x0, y0, x1, y1 = box
        crop = Image.fromarray(video.frames[pixel][y0:y1, x0:x1])
        crop.save(keyframe_dir / f"interface_crop_{label}_frame_{pixel:04d}.png")
        tiles.append(_label_image(np.asarray(crop), f"{label} interface crop F{pixel}", (360, 260)))
    canvas = Image.new("RGB", (720, 260), "white")
    canvas.paste(tiles[0], (0, 0)); canvas.paste(tiles[1], (360, 0))
    canvas.save(path)
    return {
        "pixel_frame_index": pixel,
        "source_xyxy": list(source_box),
        "edited_xyxy": list(edited_box),
        "radius_tokens": radius,
    }


def save_token_card(card: dict[str, Any], path: Path) -> None:
    values = card["values"]
    lines = [
        "Automatically selected interface token (actual hook values)",
        f"block={card['block_index']} latent={card['latent_frame_index']} "
        f"token=({card['token_row']},{card['token_column']})",
        f"grid={tuple(card['spatial_shape'])} valid={card['valid_token']} "
        f"connected={card['connected_object_support']}",
    ]
    lines.extend(f"{name}: {value:.7g}" for name, value in values.items())
    font = _font(18)
    image = Image.new("RGB", (760, 50 + 27 * len(lines)), "white")
    draw = ImageDraw.Draw(image)
    for line_index, line in enumerate(lines):
        draw.text((18, 15 + 27 * line_index), line, fill="black", font=font)
    image.save(path)


def save_pipeline(
    blocks: list[Block],
    selected: TokenSelection,
    groups: np.ndarray,
    source: Video,
    edited: Video,
    p99: float,
    png_path: Path,
    pdf_path: Path,
) -> None:
    block = blocks[selected.block_position]
    local = selected.local_frame
    latent = int(block.latent_frame_indices[local])
    pixel = representative_pixel_frame(groups, latent)
    late_block = blocks[-1]
    late_local = len(late_block.latent_frame_indices) - 1
    late_latent = int(late_block.latent_frame_indices[late_local])
    late_pixel = representative_pixel_frame(groups, late_latent)
    composite, mixed = role_composite(block, local)
    card = token_card_values(blocks, selected)
    values = card["values"]
    selected_is_mixed = bool(
        values["q_interface"] < 1.0 - 1e-6
        and sum(values[name] for name in ROLE_FIELDS if name != "q_interface") > 1e-6
    )

    figure = plt.figure(figsize=(19, 10))
    grid = figure.add_gridspec(4, 5, width_ratios=(1.35, 1, 1, 1, 1.35))
    source_axis = figure.add_subplot(grid[0:2, 0])
    source_axis.imshow(source.frames[pixel]); source_axis.set_title(f"Source Frame\nF{pixel} / L{latent} / B{block.index}")
    crop_axis = figure.add_subplot(grid[2:4, 0])
    source_box = token_crop_box(
        source.frames[pixel].shape, block.spatial_shape,
        selected.row, selected.column, 3,
    )
    x0, y0, x1, y1 = source_box
    crop_axis.imshow(source.frames[pixel][y0:y1, x0:x1])
    crop_axis.set_title("Hand-object interface crop")

    evidence_panels = (
        ("Clean-source semantic attention", block.maps.get("source_attention", block.maps["initial_object_posterior"])[local]),
        ("Latent-aligned hand occupancy", block.maps[block.hand_field][local]),
        ("First paired branch response", block.maps["field_score"][local]),
        ("Connected object support", block.maps["connected_object_support"][local]),
    )
    for row, (title, value) in enumerate(evidence_panels):
        _map_axis(figure.add_subplot(grid[row, 1]), value, title)

    role_axis = figure.add_subplot(grid[0:2, 2])
    role_axis.imshow(composite, interpolation="nearest")
    _add_mixed_border(role_axis, mixed)
    role_axis.set_title("Composite Soft Responsibilities")
    role_axis.set_xticks([]); role_axis.set_yticks([])
    _map_axis(
        figure.add_subplot(grid[2, 2]), block.maps["q_interface"][local],
        "q_interface [0,1]",
    )
    card_axis = figure.add_subplot(grid[3, 2]); card_axis.axis("off")
    card_axis.set_title(
        "Mixed Interface Token" if selected_is_mixed else "Interface Token (pure in this run)",
        fontsize=9,
    )
    card_axis.text(
        0.0, 1.0,
        "\n".join((
            f"q_obj={values['q_object']:.3f}  q_int={values['q_interface']:.3f}",
            f"q_hand={values['q_hand']:.3f}  q_bg={values['q_background']:.3f}",
            f"source retention={values['rho_role_magnitude']:.3f}",
            f"appearance access={values['appearance_access']:.3f}",
            f"write={values['reference_write_gate']:.3f}  read={values['reference_read_gate']:.3f}",
            f"entropy={values['role_entropy']:.3f}",
        )),
        va="top", ha="left", fontsize=8,
    )

    permission_panels = (
        ("Source Retention\nRole-Allocated ||rho_role||", block.maps["rho_role_magnitude"][local], p99),
        ("Role-Conditioned\nAppearance Access", block.maps["appearance_access"][local], 1.0),
        ("Reference Write Eligibility", block.maps["reference_write_gate"][local], 1.0),
        ("Anchor Read Access", block.maps["reference_read_gate"][local], 1.0),
    )
    for row, (title, value, vmax) in enumerate(permission_panels):
        _map_axis(
            figure.add_subplot(grid[row, 3]), value, title,
            cmap="magma", vmax=vmax, mixed=mixed,
        )

    edited_axis = figure.add_subplot(grid[0:2, 4])
    edited_axis.imshow(edited.frames[pixel])
    edited_axis.set_title(f"Edited contact frame\nF{pixel} / L{latent} / B{block.index}")
    late_axis = figure.add_subplot(grid[2:4, 4])
    late_axis.imshow(edited.frames[late_pixel])
    late_axis.set_title(f"Edited late-chunk frame\nF{late_pixel} / L{late_latent} / B{late_block.index}")
    for axis in (source_axis, crop_axis, edited_axis, late_axis):
        axis.set_xticks([]); axis.set_yticks([])
    conclusion = (
        "Same mixed token, different operational permissions"
        if selected_is_mixed
        else "Strict argmax token is pure interface in this run; mixed-token claim is not supported"
    )
    figure.suptitle(
        "Source Frame → Interaction Evidence → Soft Responsibilities → "
        f"Operation-Specific Permissions → Edited Result\n{conclusion}",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)


def save_diagnostics(
    blocks: list[Block], selected: TokenSelection, groups: np.ndarray, p99: float, path: Path
) -> None:
    field_names = (
        MASK_FIELDS
        + ROLE_FIELDS
        + AUX_PROBABILITY_FIELDS
        + COEFFICIENT_FIELDS
        + PERMISSION_FIELDS
        + VELOCITY_FIELDS
    )
    hand = []
    payload: dict[str, np.ndarray] = {
        name: np.concatenate([block.maps[name] for block in blocks], axis=0)
        for name in field_names
    }
    for block in blocks:
        hand.append(block.maps[block.hand_field])
    payload["hand_evidence"] = np.concatenate(hand, axis=0)
    for name in EVIDENCE_FIELDS:
        if all(name in block.maps for block in blocks):
            payload[name] = np.concatenate([block.maps[name] for block in blocks], axis=0)
    for name in ROLE_FIELDS:
        payload[f"native_{name}"] = np.concatenate(
            [block.raw_role_maps[name] for block in blocks], axis=0
        )
    for name in VELOCITY_FIELDS:
        payload[f"native_{name}"] = np.concatenate(
            [block.raw_velocity_maps[name] for block in blocks], axis=0
        )
    payload["latent_frame_indices"] = np.concatenate(
        [block.latent_frame_indices for block in blocks]
    )
    payload["block_index_per_latent"] = np.concatenate(
        [np.full(len(block.latent_frame_indices), block.index, dtype=np.int64) for block in blocks]
    )
    payload["causal_temporal_groups"] = groups
    payload["velocity_shared_p99"] = np.asarray(p99, dtype=np.float64)
    payload["selected_block_index"] = np.asarray(blocks[selected.block_position].index)
    payload["selected_local_frame"] = np.asarray(selected.local_frame)
    payload["selected_token_row"] = np.asarray(selected.row)
    payload["selected_token_column"] = np.asarray(selected.column)
    if all(block.hook_extras for block in blocks):
        for prediction_name in ("prediction_000", "prediction_007", "prediction_014"):
            payload[f"appearance_access_{prediction_name}"] = np.concatenate(
                [block.hook_extras["appearance_prediction_maps"][prediction_name]
                 for block in blocks], axis=0
            )
        payload["anchor_read_access"] = np.concatenate(
            [block.maps["reference_read_gate"] for block in blocks], axis=0
        )
        payload["reference_write_eligibility"] = np.concatenate(
            [block.maps["reference_write_gate"] for block in blocks], axis=0
        )
        for output_name, extra_name in (
            ("m2_canonical_read_gate", "m2_canonical_read_gate"),
            ("m2_canonical_write_gate", "m2_canonical_write_gate"),
            ("selected_write_tokens", "selected_write_tokens"),
        ):
            payload[output_name] = np.concatenate([
                _flattened_or_spatial_map(
                    block.hook_extras[extra_name], name=extra_name,
                    temporal_size=len(block.latent_frame_indices),
                    candidate_shapes=(block.spatial_shape,),
                )
                for block in blocks
            ], axis=0)
        for prediction_name in ("prediction_000", "prediction_007", "prediction_014"):
            for field in ("admitted", "gate", "correction_rms", "native_rms", "joint_similarity"):
                chunks = []
                validity = []
                for block in blocks:
                    retrieval = block.hook_extras["retrieval_mean_maps"]
                    if prediction_name not in retrieval:
                        chunks.append(np.full(
                            (len(block.latent_frame_indices), *block.spatial_shape),
                            np.nan, dtype=np.float32,
                        ))
                        validity.extend([False] * len(block.latent_frame_indices))
                    else:
                        chunks.append(_flattened_or_spatial_map(
                            retrieval[prediction_name][field],
                            name=f"retrieval.{prediction_name}.{field}",
                            temporal_size=len(block.latent_frame_indices),
                            candidate_shapes=(block.spatial_shape,),
                        ))
                        validity.extend([True] * len(block.latent_frame_indices))
                payload[f"retrieval_{prediction_name}_{field}"] = np.concatenate(chunks, axis=0)
                payload[f"retrieval_{prediction_name}_{field}_valid"] = np.asarray(validity)
    np.savez_compressed(path, **payload)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _read_run_config(edited_video: Path) -> dict[str, Any]:
    config_path = edited_video.resolve().parent / "ours_responsibility_inference_config.txt"
    if not config_path.is_file():
        return {"config_path": None, "available": False}
    entries: dict[str, str] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            entries[key.strip()] = value.strip()
    command = entries.get("command", "")
    tokens = shlex.split(command)
    def option(name: str) -> str | None:
        return tokens[tokens.index(name) + 1] if name in tokens and tokens.index(name) + 1 < len(tokens) else None
    return {
        "available": True,
        "config_path": str(config_path),
        "runner": tokens[1] if len(tokens) > 1 else None,
        "full_command": command,
        "source_video": option("--data_path"),
        "hand_mask": option("--hand_mask_video"),
        "source_prompt": option("--src_prompt"),
        "target_prompt": option("--trg_prompt"),
        "seed": int(option("--seed")) if option("--seed") is not None else None,
        "inference_steps": int(option("--step")) if option("--step") is not None else None,
        "rollout_chunk_size": int(option("--rollout_chunk_size")) if option("--rollout_chunk_size") is not None else None,
        "checkpoint": option("--checkpoint_path"),
        "wan_models_root": entries.get("wan_models_root"),
        "configuration": entries,
    }


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.max_keyframes < 1:
        raise ValueError("--max-keyframes must be positive")
    if args.crop_radius_tokens < 0:
        raise ValueError("--crop-radius-tokens must be non-negative")
    blocks = load_blocks(args.artifacts)
    groups = load_temporal_groups(args.hand_role_input)
    source = read_video(args.source_video)
    edited = read_video(args.edited_video)
    validate_alignment(blocks, groups, source, edited)
    selected = select_interface_token(blocks)
    p99, velocity_stats = velocity_statistics(blocks)
    records = choose_key_records(blocks, selected, args.max_keyframes)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    keyframe_dir = output_dir / "ours_selected_keyframes"
    paths = {
        "pipeline_png": output_dir / "ours_responsibility_pipeline.png",
        "pipeline_pdf": output_dir / "ours_responsibility_pipeline.pdf",
        "role_maps": output_dir / "ours_role_maps.png",
        "permission_maps": output_dir / "ours_permission_maps.png",
        "refinement": output_dir / "ours_refinement.png",
        "velocity_control": output_dir / "ours_velocity_control.png",
        "edited_video": output_dir / "ours_edited_video.mp4",
        "diagnostics": output_dir / "ours_diagnostics.npz",
        "metrics": output_dir / "metrics.json",
        "metadata": output_dir / "metadata.json",
        "keyframes": output_dir / "ours_keyframes.png",
        "interface_crop": output_dir / "ours_interface_crop.png",
        "token_card": output_dir / "ours_token_card.png",
    }
    save_role_maps(blocks, records, paths["role_maps"])
    save_permission_maps(blocks, records, p99, paths["permission_maps"])
    save_refinement(blocks, records, paths["refinement"])
    save_velocity_control(blocks, records, p99, paths["velocity_control"])
    save_pipeline(
        blocks, selected, groups, source, edited, p99,
        paths["pipeline_png"], paths["pipeline_pdf"],
    )
    pixel_keyframes = save_keyframes(
        blocks, records, groups, source, edited, paths["keyframes"], keyframe_dir
    )
    crop = save_interface_crop(
        blocks, selected, groups, source, edited, args.crop_radius_tokens,
        paths["interface_crop"], keyframe_dir,
    )
    card = token_card_values(blocks, selected)
    save_token_card(card, paths["token_card"])
    save_diagnostics(blocks, selected, groups, p99, paths["diagnostics"])
    if args.edited_video.resolve() != paths["edited_video"].resolve():
        shutil.copy2(args.edited_video, paths["edited_video"])

    metrics = compute_metrics(blocks, selected)
    metrics["velocity_shared_robust_p99"] = p99
    _write_json(paths["metrics"], metrics)
    run_config = _read_run_config(args.edited_video)
    hook_metadata = blocks[0].hook_extras.get("metadata", {})
    selected_values = card["values"]
    selected_is_mixed = bool(
        selected_values["q_interface"] < 1.0 - 1e-6
        and sum(selected_values[name] for name in ROLE_FIELDS if name != "q_interface") > 1e-6
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_files": [block.source_path for block in blocks],
        "source_video": str(args.source_video.resolve()),
        "edited_video_input": str(args.edited_video.resolve()),
        "hand_role_input": str(args.hand_role_input.resolve()),
        "source_video_metadata": {
            "frame_count": int(len(source.frames)),
            "height": int(source.frames.shape[1]),
            "width": int(source.frames.shape[2]),
            "fps": source.fps,
        },
        "edited_video_metadata": {
            "frame_count": int(len(edited.frames)),
            "height": int(edited.frames.shape[1]),
            "width": int(edited.frames.shape[2]),
            "fps": edited.fps,
        },
        "blocks": [
            {
                "block_index": block.index,
                "latent_frame_indices": block.latent_frame_indices.tolist(),
                "spatial_shape": list(block.spatial_shape),
                "native_role_spatial_shape": list(block.raw_role_spatial_shape),
                "role_alignment": block.role_alignment,
                "native_velocity_spatial_shape": list(block.velocity_spatial_shape),
                "velocity_alignment": block.velocity_alignment,
                "hand_evidence_field": block.hand_field,
            }
            for block in blocks
        ],
        "causal_temporal_groups_source": "hand_role_input.npz:causal_temporal_groups",
        "selected_interface_token": card,
        "selected_interface_token_is_mixed": selected_is_mixed,
        "claimed_mixed_token_conclusion_supported": selected_is_mixed,
        "conclusion": (
            "The strict argmax token is a mixed interface token and receives operation-specific permissions."
            if selected_is_mixed else
            "The strict argmax token is pure interface (q_interface=1); this run does not support the same-mixed-token claim."
        ),
        "run": run_config,
        "real_variable_sources": {
            "source_variable_paths": hook_metadata.get("source_variable_paths", {}),
            "variable_sources": hook_metadata.get("variable_sources", {}),
            "prediction_index_labels": hook_metadata.get("prediction_index_labels", {}),
        },
        "anchor": metrics["requested_statistics"]["anchor"],
        "selected_pixel_crop": crop,
        "selected_keyframe_pixel_indices": pixel_keyframes,
        "role_visualization": {
            "dominant_rule": "argmax posterior soft role",
            "selection_is_independent_of_mixed_display": True,
            "confidence_brightness": "0.25 + 0.75 * max_role_probability",
            "mixed_interface_rule": (
                f"q_interface >= {MIXED_INTERFACE_THRESHOLD}, q_interface < 1, "
                "and another role has nonzero probability"
            ),
            "colors_rgb_0_255": {
                label: [int(round(channel * 255)) for channel in color]
                for label, color in zip(ROLE_LABELS, ROLE_COLORS)
            },
        },
        "role_grid_alignment": {
            "preferred": "complete hook-saved q_*_token_grid tuple",
            "fallback": (
                "exact PosteriorResidualFlowRouter._resize_roles: bilinear, "
                "align_corners=False, clamp_min(0), normalize roles with eps=1e-6"
            ),
            "native_maps_preserved_in_diagnostics": True,
        },
        "velocity_scaling": {
            "rule": "one pooled p99 over native magnitudes, all blocks, and all five velocity fields",
            "shared_robust_p99": p99,
            "raw_statistics": velocity_stats,
        },
        "no_approximate_reconstruction": True,
        "edited_video_copy": "byte-preserving copy of edited_video_input",
        "outputs": {name: str(path.resolve()) for name, path in paths.items()},
    }
    _write_json(paths["metadata"], metadata)
    return paths


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = run(args)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
