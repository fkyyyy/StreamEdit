from __future__ import annotations

import math

import torch

from utils.mechanism_attention_diagnostics import (
    maybe_capture_cross_attention,
    maybe_capture_self_attention,
)


def _config(tmp_path):
    return {
        "mechanism_diagnostics_config": {
            "active": True,
            "output_dir": str(tmp_path),
            "step_index": 1,
            "capture_steps": (0, 1, 7),
            "block_index": 0,
            "batch_index": 0,
            "object_token_indices": (1, 3),
            "object_phrase": "wooden spatula",
            "target_token_ids": (10, 11, 12, 13),
            "target_token_strings": ("a", "wooden", "spat", "ula"),
            "query_indices": (0, 2, 3),
            "query_names": ("interface_query", "object_query", "hand_query"),
            "grid_shape": (1, 2, 2),
            "latent_frame_indices": (4,),
            "frame_seqlen": 4,
            "current_tokens": 4,
            "noise_level": 0.5,
        }
    }


def test_cross_capture_uses_full_softmax_denominator_and_raw_head_mean(tmp_path):
    cache = _config(tmp_path)
    query = torch.arange(1 * 4 * 2 * 3, dtype=torch.float32).reshape(1, 4, 2, 3) / 20
    key = torch.arange(1 * 5 * 2 * 3, dtype=torch.float32).reshape(1, 5, 2, 3) / 30
    query_before = query.clone()
    key_before = key.clone()

    path = maybe_capture_cross_attention(
        cache, query, key, layer_index=2, key_lens=torch.tensor([4])
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    scores = torch.einsum("qhd,khd->hqk", query[0], key[0]) / math.sqrt(3)
    expected = scores[..., :4].softmax(dim=-1)[..., [1, 3]]

    torch.testing.assert_close(payload["probabilities"], expected)
    torch.testing.assert_close(payload["raw_attention"], expected.mean(dim=-1, keepdim=True))
    assert payload["metadata"]["tensor_layout"] == "heads_query_key"
    assert payload["metadata"]["full_key_denominator"] is True
    assert payload["metadata"]["valid_key_length"] == 4
    assert payload["metadata"]["context_length_mask_applied"] is True
    assert payload["metadata"]["object_token_indices"] == [1, 3]
    torch.testing.assert_close(query, query_before, rtol=0, atol=0)
    torch.testing.assert_close(key, key_before, rtol=0, atol=0)


def test_cross_capture_matches_effective_foreground_importance_weights(tmp_path):
    cache = _config(tmp_path)
    cache["mechanism_diagnostics_config"]["batch_index"] = 1
    cache.update({
        "apply_enhance": True,
        "fg_boost_factor": 4.0,
        "fg_indices": [1, 3],
        "current_src_fg_mask": torch.tensor([[True, False, True, False]]),
    })
    query = torch.arange(2 * 4 * 2 * 3, dtype=torch.float32).reshape(2, 4, 2, 3) / 20
    key = torch.arange(2 * 5 * 2 * 3, dtype=torch.float32).reshape(2, 5, 2, 3) / 30

    path = maybe_capture_cross_attention(
        cache, query, key, layer_index=4, key_lens=torch.tensor([4, 4])
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    scores = torch.einsum("qhd,khd->hqk", query[1], key[1, :4]) / math.sqrt(3)
    base = scores.softmax(dim=-1)
    weights = torch.tensor([1.0, 4.0, 1.0, 4.0])
    enhanced = base * weights
    enhanced = enhanced / enhanced.sum(dim=-1, keepdim=True)
    expected = base.clone()
    expected[:, [0, 2]] = enhanced[:, [0, 2]]
    expected = expected[..., [1, 3]]

    torch.testing.assert_close(payload["probabilities"], expected)
    assert payload["metadata"]["foreground_importance_enhancement_applied"] is True
    assert payload["metadata"]["probability_rule"].startswith("effective_post_softmax")


def test_self_capture_keeps_only_queries_and_declares_history_chunks(tmp_path):
    cache = _config(tmp_path)
    query = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3) / 20
    key = torch.arange(12 * 2 * 3, dtype=torch.float32).reshape(12, 2, 3) / 30
    query_before = query.clone()
    key_before = key.clone()
    segments = [
        {"name": "target_history", "start": 0, "end": 8},
        {"name": "current_target", "start": 8, "end": 12},
    ]

    path = maybe_capture_self_attention(
        cache,
        query,
        key,
        layer_index=3,
        batch_index=0,
        current_target_segment=(8, 12),
        key_segments=segments,
        path_name="factorized_native_output",
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    scores = torch.einsum("qhd,khd->hqk", query[[0, 2, 3]], key) / math.sqrt(3)
    expected = scores.softmax(dim=-1)

    torch.testing.assert_close(payload["raw_attention"], expected)
    assert tuple(payload["raw_attention"].shape) == (2, 3, 12)
    assert payload["metadata"]["current_key_range"] == [8, 12]
    assert [entry["name"] for entry in payload["metadata"]["history_chunk_ranges"]] == [
        "previous_chunk_000",
        "previous_chunk_001",
    ]
    assert [entry["original_query_index"] for entry in payload["metadata"]["queries"]] == [0, 2, 3]
    torch.testing.assert_close(query, query_before, rtol=0, atol=0)
    torch.testing.assert_close(key, key_before, rtol=0, atol=0)
