"""Read-only raw attention diagnostics for mechanism analysis.

The helpers in this module recompute attention probabilities from the exact
Q/K tensors used by the model.  They never feed their results back into the
forward pass.
"""

import math
import os
import threading
from pathlib import Path

import torch


_WRITE_LOCK = threading.Lock()
_WRITTEN_PATHS = set()


def _diagnostics_config(cache):
    if not isinstance(cache, dict):
        return None
    config = cache.get("mechanism_diagnostics_config")
    if config is None and isinstance(cache.get("shared_dict"), dict):
        config = cache["shared_dict"].get(
            "mechanism_diagnostics_config"
        )
    return config if isinstance(config, dict) else None


def _capture_coordinates(config, layer_index):
    if config is None or not config.get("active", False):
        return None
    if "step_index" not in config:
        raise ValueError(
            "Active mechanism diagnostics require step_index"
        )
    step_index = int(config["step_index"])
    capture_steps = config.get("capture_steps")
    if capture_steps is not None:
        if isinstance(capture_steps, int):
            capture_steps = (capture_steps,)
        if step_index not in {int(step) for step in capture_steps}:
            return None
    if layer_index is None:
        layer_index = config.get("block_index")
    if layer_index is None:
        raise ValueError(
            "Active mechanism diagnostics require block_index"
        )
    return step_index, int(layer_index)


def _validate_qk(query, key):
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("Attention diagnostics expect Q/K=[tokens,heads,dim]")
    if query.shape[1:] != key.shape[1:]:
        raise ValueError("Diagnostic Q/K head dimensions do not match")


@torch.no_grad()
def attention_probabilities_fp32(
    query,
    key,
    *,
    query_indices=None,
    key_indices=None,
    query_chunk_size=128,
):
    """Compute exact scaled-dot-product probabilities in fp32.

    The softmax denominator always spans every key.  Optional key selection is
    applied only after the complete softmax, which is important for prompt
    phrase probabilities.
    """
    _validate_qk(query, key)
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")

    device = query.device
    if key.device != device:
        raise ValueError("Diagnostic Q/K must be on the same device")
    if query_indices is not None:
        query_indices = torch.as_tensor(
            query_indices, dtype=torch.long, device=device
        ).flatten()
        if query_indices.numel() == 0:
            raise ValueError("query_indices must not be empty")
        if (
            query_indices.min().item() < 0
            or query_indices.max().item() >= query.shape[0]
        ):
            raise IndexError("Diagnostic query index is out of range")
        query = query.index_select(0, query_indices)
    if key_indices is not None:
        key_indices = torch.as_tensor(
            key_indices, dtype=torch.long, device=device
        ).flatten()
        if key_indices.numel() == 0:
            raise ValueError("key_indices must not be empty")
        if (
            key_indices.min().item() < 0
            or key_indices.max().item() >= key.shape[0]
        ):
            raise IndexError("Diagnostic key index is out of range")

    query_f32 = query.detach().to(dtype=torch.float32)
    key_f32 = key.detach().to(dtype=torch.float32)
    scale = 1.0 / math.sqrt(query.shape[-1])
    chunks = []
    with torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, query_f32.shape[0], query_chunk_size):
            query_chunk = query_f32[start:start + query_chunk_size]
            scores = torch.einsum(
                "qhd,khd->hqk", query_chunk, key_f32
            ) * scale
            probabilities = torch.softmax(
                scores, dim=-1, dtype=torch.float32
            )
            if key_indices is not None:
                probabilities = probabilities.index_select(-1, key_indices)
            chunks.append(probabilities.detach().cpu())
    return torch.cat(chunks, dim=1)


def _select_batch_indices(indices, batch_index, batch_size, name):
    if isinstance(indices, torch.Tensor):
        if indices.ndim <= 1:
            return indices
        if indices.ndim == 2 and indices.shape[0] == batch_size:
            return indices[batch_index]
        raise ValueError(f"{name} tensor has an invalid batch dimension")
    if not isinstance(indices, (list, tuple)):
        raise ValueError(f"{name} must be a sequence of token indices")
    if indices and isinstance(indices[0], (list, tuple, torch.Tensor)):
        if len(indices) != batch_size:
            raise ValueError(f"Batched {name} must match attention batch")
        return indices[batch_index]
    return indices


def _query_spec(config):
    indices = config.get("query_indices")
    if isinstance(indices, dict):
        names = [str(name) for name in indices]
        indices = list(indices.values())
    else:
        if indices is None:
            raise ValueError(
                "Active self-attention diagnostics require query_indices"
            )
        indices = list(indices)
        names = config.get("query_names")
        if names is None:
            names = [f"query_{index}" for index in indices]
        elif isinstance(names, dict):
            names = [str(names.get(index, index)) for index in indices]
        else:
            names = [str(name) for name in names]
    if len(indices) != 3:
        raise ValueError(
            "Self-attention diagnostics require exactly three query indices"
        )
    if len(names) != len(indices):
        raise ValueError("query_names must align with query_indices")
    return [int(index) for index in indices], names


def _normalise_segments(
    key_segments, key_length, frame_seqlen, current_tokens=None
):
    segments = []
    for segment in key_segments:
        current = {
            "name": str(segment["name"]),
            "start": int(segment["start"]),
            "end": int(segment["end"]),
        }
        if not 0 <= current["start"] <= current["end"] <= key_length:
            raise ValueError("Self-attention key segment is out of range")
        segments.append(current)

    history_frames = []
    history_chunks = []
    if frame_seqlen is not None:
        frame_seqlen = int(frame_seqlen)
        if frame_seqlen <= 0:
            raise ValueError("frame_seqlen must be positive")
        for segment in segments:
            if segment["name"] != "target_history":
                continue
            start, end = segment["start"], segment["end"]
            remainder = (end - start) % frame_seqlen
            if remainder:
                history_frames.append({
                    "name": "history_prefix",
                    "start": start,
                    "end": start + remainder,
                })
                start += remainder
            frame_index = 0
            while start < end:
                history_frames.append({
                    "name": f"history_frame_{frame_index:03d}",
                    "start": start,
                    "end": start + frame_seqlen,
                })
                start += frame_seqlen
                frame_index += 1
    if current_tokens is not None:
        current_tokens = int(current_tokens)
        if current_tokens <= 0:
            raise ValueError("current_tokens must be positive")
        for segment in segments:
            if segment["name"] != "target_history":
                continue
            start, end = segment["start"], segment["end"]
            remainder = (end - start) % current_tokens
            if remainder:
                history_chunks.append({
                    "name": "history_chunk_prefix",
                    "start": start,
                    "end": start + remainder,
                })
                start += remainder
            chunk_index = 0
            while start < end:
                history_chunks.append({
                    "name": f"previous_chunk_{chunk_index:03d}",
                    "start": start,
                    "end": start + current_tokens,
                })
                start += current_tokens
                chunk_index += 1
    return segments, history_frames, history_chunks


def _output_path(config, kind, step_index, layer_index):
    output_dir = config.get("output_dir")
    if output_dir is None:
        raise ValueError("Active mechanism diagnostics require output_dir")
    return (
        Path(output_dir)
        / "raw_attention"
        / kind
        / f"step_{step_index:03d}"
        / f"layer_{layer_index:02d}.pt"
    )


def _save_once(path, payload):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path_key = str(path)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _WRITE_LOCK:
        if path_key in _WRITTEN_PATHS or path.exists():
            return False
        _WRITTEN_PATHS.add(path_key)
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        with _WRITE_LOCK:
            _WRITTEN_PATHS.discard(path_key)
        return False
    else:
        os.close(lock_fd)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        if path.exists():
            return False
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        with _WRITE_LOCK:
            _WRITTEN_PATHS.discard(path_key)
        if temporary.exists():
            temporary.unlink()
        raise
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
    return True


@torch.no_grad()
def maybe_capture_cross_attention(
    cache,
    query,
    key,
    *,
    layer_index=None,
    key_lens=None,
):
    """Save target-branch phrase probabilities as ``[heads,Lq,P]``."""
    config = _diagnostics_config(cache)
    coordinates = _capture_coordinates(config, layer_index)
    if coordinates is None:
        return None
    step_index, layer_index = coordinates
    path = _output_path(config, "cross", step_index, layer_index)
    if path.exists():
        return None
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("Cross-attention diagnostics expect Q/K=[B,L,H,D]")
    batch_index = int(config.get("batch_index", query.shape[0] - 1))
    if batch_index < 0:
        batch_index += query.shape[0]
    if not 0 <= batch_index < query.shape[0]:
        raise IndexError("Diagnostic cross-attention batch_index is invalid")
    object_indices = _select_batch_indices(
        config.get("object_token_indices"),
        batch_index,
        query.shape[0],
        "object_token_indices",
    )
    object_indices = [int(index) for index in object_indices]
    valid_key_length = int(key.shape[1])
    if key_lens is not None:
        lengths = torch.as_tensor(key_lens).detach().flatten()
        length_index = batch_index if lengths.numel() > 1 else 0
        if length_index >= lengths.numel():
            raise IndexError("Diagnostic cross-attention key length is missing")
        valid_key_length = int(lengths[length_index].item())
        if not 0 < valid_key_length <= key.shape[1]:
            raise ValueError("Diagnostic cross-attention key length is invalid")
    if any(index < 0 or index >= valid_key_length for index in object_indices):
        raise IndexError("Object phrase token lies outside the valid text context")
    probabilities = attention_probabilities_fp32(
        query[batch_index],
        key[batch_index, :valid_key_length],
        query_chunk_size=int(config.get("query_chunk_size", 128)),
    )
    probability_rule = "scaled_dot_product_softmax_over_valid_text"
    enhancement_applied = False
    if (
        cache is not None
        and cache.get("apply_enhance", False)
        and float(cache.get("fg_boost_factor", 1.0)) != 1.0
        and cache.get("fg_indices") is not None
        and query.shape[0] % 2 == 0
        and batch_index >= query.shape[0] // 2
    ):
        target_batch = batch_index - query.shape[0] // 2
        configured_indices = cache["fg_indices"]
        if not isinstance(configured_indices[0], list):
            foreground_indices = configured_indices
        else:
            target_indices = configured_indices[query.shape[0] // 2:]
            foreground_indices = target_indices[target_batch]
        weights = probabilities.new_ones(valid_key_length)
        valid_foreground_indices = [
            int(index)
            for index in foreground_indices
            if 0 <= int(index) < valid_key_length
        ]
        if valid_foreground_indices:
            weights[valid_foreground_indices] = float(
                cache["fg_boost_factor"]
            )
            weighted = probabilities * weights[None, None, :]
            weighted = weighted / weighted.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(weighted.dtype).tiny
            )
            query_foreground = cache.get("current_src_fg_mask")
            if query_foreground is None:
                probabilities = weighted
            else:
                query_foreground = (
                    query_foreground[target_batch].detach().to("cpu").bool()
                )
                if query_foreground.numel() != probabilities.shape[1]:
                    raise ValueError(
                        "Cross-attention foreground query mask does not align"
                    )
                probabilities[:, query_foreground] = weighted[:, query_foreground]
            enhancement_applied = True
            probability_rule = (
                "effective_post_softmax_probability_after_fg_value_"
                "importance_normalization"
            )
    selected_probabilities = probabilities[..., object_indices]
    phrase_probability = selected_probabilities.mean(dim=-1, keepdim=True)
    grid_shape = tuple(int(value) for value in config["grid_shape"])
    payload = {
        "kind": "cross",
        "step_index": step_index,
        "block_index": layer_index,
        "layer_index": layer_index,
        "batch_index": batch_index,
        "probabilities": selected_probabilities,
        "raw_attention": phrase_probability,
        "object_token_indices": torch.tensor(
            object_indices, dtype=torch.long
        ),
        "query_length": int(query.shape[1]),
        "key_length": int(key.shape[1]),
        "valid_key_length": valid_key_length,
        "num_heads": int(query.shape[2]),
        "head_dim": int(query.shape[3]),
        "softmax_scale": 1.0 / math.sqrt(query.shape[3]),
        "full_key_denominator": True,
        "metadata": {
            "kind": "cross",
            "step_index": step_index,
            "layer_index": layer_index,
            "batch_index": 0,
            "tensor_layout": "heads_query_key",
            "query_spatial_shape": grid_shape,
            "latent_frame_indices": tuple(
                int(value)
                for value in config["latent_frame_indices"]
            ),
            "selected_key_index": 0,
            "object_phrase": config.get("object_phrase"),
            "object_token_indices": object_indices,
            "target_token_ids": config.get("target_token_ids"),
            "target_token_strings": config.get("target_token_strings"),
            "aggregation": (
                "arithmetic mean of post-softmax probabilities over the "
                "matched phrase tokens"
            ),
            "full_key_denominator": True,
            "padded_key_length": int(key.shape[1]),
            "valid_key_length": valid_key_length,
            "context_length_mask_applied": key_lens is not None,
            "foreground_importance_enhancement_applied": enhancement_applied,
            "probability_rule": probability_rule,
            "noise_level": config.get("noise_level"),
            "block_index": config.get("block_index"),
        },
    }
    return path if _save_once(path, payload) else None


@torch.no_grad()
def maybe_capture_self_attention(
    cache,
    query,
    key,
    *,
    layer_index=None,
    batch_index=0,
    current_target_segment,
    key_segments,
    path_name,
):
    """Save three selected-query probabilities as ``[queries,heads,K]``."""
    config = _diagnostics_config(cache)
    coordinates = _capture_coordinates(config, layer_index)
    if coordinates is None:
        return None
    step_index, layer_index = coordinates
    path = _output_path(config, "self", step_index, layer_index)
    if path.exists():
        return None
    requested_batch = int(config.get("batch_index", 0))
    if batch_index != requested_batch:
        return None
    query_indices, query_names = _query_spec(config)
    frame_seqlen = config.get("frame_seqlen")
    segments, history_frames, history_chunks = _normalise_segments(
        key_segments,
        key.shape[0],
        frame_seqlen,
        current_tokens=config.get("current_tokens"),
    )
    current_target_segment = {
        "start": int(current_target_segment[0]),
        "end": int(current_target_segment[1]),
    }
    if not (
        0
        <= current_target_segment["start"]
        <= current_target_segment["end"]
        <= key.shape[0]
    ):
        raise ValueError("Current target key segment is out of range")
    probabilities = attention_probabilities_fp32(
        query,
        key,
        query_indices=query_indices,
        query_chunk_size=len(query_indices),
    ).contiguous()
    grid_shape = tuple(int(value) for value in config["grid_shape"])
    current_range = (
        current_target_segment["start"],
        current_target_segment["end"],
    )
    history_ranges = [
        {
            "name": segment["name"],
            "range": [segment["start"], segment["end"]],
        }
        for segment in segments
        if not (
            segment["start"] == current_range[0]
            and segment["end"] == current_range[1]
        )
    ]
    payload = {
        "kind": "self",
        "path_name": str(path_name),
        "step_index": step_index,
        "block_index": layer_index,
        "layer_index": layer_index,
        "batch_index": int(batch_index),
        "probabilities": probabilities,
        "raw_attention": probabilities,
        "query_indices": torch.tensor(query_indices, dtype=torch.long),
        "query_names": query_names,
        "query_length": int(query.shape[0]),
        "key_length": int(key.shape[0]),
        "num_heads": int(query.shape[1]),
        "head_dim": int(query.shape[2]),
        "softmax_scale": 1.0 / math.sqrt(query.shape[2]),
        "current_target_key_segment": current_target_segment,
        "key_segments": segments,
        "history_segments": history_frames,
        "history_chunk_segments": history_chunks,
        "frame_seqlen": (
            int(frame_seqlen) if frame_seqlen is not None else None
        ),
        "configured_current_tokens": (
            int(config["current_tokens"])
            if config.get("current_tokens") is not None
            else None
        ),
        "current_tokens": (
            current_target_segment["end"]
            - current_target_segment["start"]
        ),
        "full_key_denominator": True,
        "metadata": {
            "kind": "self",
            "step_index": step_index,
            "layer_index": layer_index,
            "batch_index": 0,
            "tensor_layout": "heads_query_key",
            "queries": [
                {
                    "name": name,
                    "index": position,
                    "original_query_index": query_index,
                }
                for position, (name, query_index) in enumerate(
                    zip(query_names, query_indices)
                )
            ],
            "current_key_range": list(current_range),
            "current_key_spatial_shape": grid_shape,
            "latent_frame_indices": tuple(
                int(value)
                for value in config["latent_frame_indices"]
            ),
            "history_key_ranges": history_ranges,
            "history_frame_ranges": [
                {
                    "name": segment["name"],
                    "range": [segment["start"], segment["end"]],
                }
                for segment in history_frames
            ],
            "history_chunk_ranges": [
                {
                    "name": segment["name"],
                    "range": [segment["start"], segment["end"]],
                }
                for segment in history_chunks
            ],
            "key_segments": segments,
            "path_name": str(path_name),
            "full_key_denominator": True,
            "noise_level": config.get("noise_level"),
            "block_index": config.get("block_index"),
        },
    }
    return path if _save_once(path, payload) else None
