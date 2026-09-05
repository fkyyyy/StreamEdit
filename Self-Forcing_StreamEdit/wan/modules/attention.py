# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

# FLASH_ATTN_3_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
    'fuse_aligned_memory',
    'fuse_factorized_aligned_memory',
    'blend_factorized_with_native_fallback',
    'build_factorized_history_read_mask',
    'blend_target_owned_tensor',
    'suppress_source_preserve_on_target_owned_history',
    'build_target_owned_source_background_mask',
    'scatter_target_owned_output',
    'apply_target_identity_value_correction',
    'resolve_target_identity_correction_strength',
    'materialize_immutable_target_value',
    'blend_source_addressed_residual',
    'arbitrate_projected_attention_output',
    'project_source_addressed_target_value',
    'source_addressed_anchor_attention_delta',
    'immutable_canonical_anchor_attention_delta',
    'scatter_source_addressed_anchor_delta',
    'source_flow_gated_multiframe_sink_attention',
    'closed_loop_counterfactual_memory_attention',
    'source_addressed_native_history_attention',
    'arbitrate_verified_factorized_attention',
    'diagnose_attention_segment',
    'route_source_background_kv',
    'build_counterfactual_segment_values',
    'counterfactual_replace_attention_segment',
]


def route_source_background_kv(
    source_key,
    source_value,
    target_value,
    background_mask,
    *,
    suppress_value=False,
    drop_pair=False,
):
    """Select the late source-background K/V segment as one unit.

    ``suppress_value`` preserves the historical 967g/967h ablation in which
    source keys are paired with target values.  ``drop_pair`` is the
    appearance-safe successor: neither member of the source-addressed pair is
    admitted, so a source key can never retrieve a non-corresponding target
    value.
    """
    if suppress_value and drop_pair:
        raise ValueError(
            "suppress_value and drop_pair are mutually exclusive"
        )
    if background_mask.ndim != 1:
        raise ValueError(
            "Source-background mask must be one-dimensional"
        )
    expected_tokens = source_key.shape[0]
    if (
        source_value.shape[0] != expected_tokens
        or target_value.shape[0] != expected_tokens
        or background_mask.shape[0] != expected_tokens
    ):
        raise ValueError(
            "Source-background K/V tensors and mask must align"
        )
    if drop_pair:
        return None
    selected_value = target_value if suppress_value else source_value
    return (
        source_key[background_mask],
        selected_value[background_mask],
    )


def counterfactual_replace_attention_segment(
    native_output,
    source_contribution,
    target_contribution,
    foreground_mask,
    *,
    eps=1e-8,
    max_scale=4.0,
):
    """Replace one source-V attention contribution with target content.

    ``source_contribution`` and ``target_contribution`` must be evaluated with
    exactly the same query, complete key sequence and softmax denominator; only
    the selected segment's values differ.  The target contribution is scaled
    by one positive scalar per query to match the source contribution RMS.
    Consequently the replacement preserves contribution energy without
    restoring the source value direction.  Queries outside the existing
    automatic foreground are returned bit-for-bit from ``native_output``.
    """
    tensors = {
        "native_output": native_output,
        "source_contribution": source_contribution,
        "target_contribution": target_contribution,
    }
    if any(value.ndim != 4 for value in tensors.values()):
        raise ValueError(
            "Attention outputs must have shape [B,L,H,D]"
        )
    if len({value.shape for value in tensors.values()}) != 1:
        raise ValueError(
            "Native, source and target attention contributions must align"
        )
    if foreground_mask.shape != native_output.shape[:2]:
        raise ValueError(
            "foreground_mask must align with attention queries"
        )
    if float(eps) <= 0:
        raise ValueError("eps must be positive")
    if float(max_scale) < 1.0:
        raise ValueError("max_scale must be at least 1")

    native = native_output.float()
    source = source_contribution.float()
    target = target_contribution.float()
    source_rms = source.square().mean(dim=(2, 3)).sqrt()
    target_rms = target.square().mean(dim=(2, 3)).sqrt()
    target_valid = target_rms > float(eps)
    raw_scale = source_rms / target_rms.clamp_min(float(eps))
    scale = torch.where(
        target_valid,
        raw_scale.clamp(max=float(max_scale)),
        torch.zeros_like(raw_scale),
    ).detach()
    matched_target = target * scale[:, :, None, None]
    foreground = foreground_mask.detach().bool()
    active = foreground & target_valid
    corrected = native - source + matched_target
    output = torch.where(
        active[:, :, None, None],
        corrected.to(native_output.dtype),
        native_output,
    )

    foreground_count = foreground.float().sum().clamp_min(1.0)

    def foreground_mean(value):
        return (value * foreground.float()).sum() / foreground_count

    diagnostics = {
        "foreground_fraction": foreground.float().mean().detach(),
        "active_fraction": active.float().mean().detach(),
        "source_contribution_rms": foreground_mean(
            source_rms
        ).detach(),
        "target_contribution_rms_raw": foreground_mean(
            target_rms
        ).detach(),
        "target_contribution_rms_matched": foreground_mean(
            matched_target.square().mean(dim=(2, 3)).sqrt()
        ).detach(),
        "replacement_scale": foreground_mean(scale).detach(),
        "replacement_scale_max": torch.where(
            foreground, scale, torch.zeros_like(scale)
        ).max().detach(),
        "scale_capped_fraction": foreground_mean(
            (target_valid & (raw_scale > float(max_scale))).float()
        ).detach(),
        "degenerate_target_fraction": foreground_mean(
            (~target_valid).float()
        ).detach(),
        "correction_rms": foreground_mean(
            (corrected - native).square().mean(dim=(2, 3)).sqrt()
        ).detach(),
    }
    return output, diagnostics


def build_counterfactual_segment_values(
    native_value,
    target_segment_value,
    *,
    segment_start,
    segment_end,
):
    """Build source and target V-only rows for an exact segment audit."""
    if native_value.ndim != 3:
        raise ValueError("native_value must have shape [K,H,D]")
    start, end = int(segment_start), int(segment_end)
    if not 0 <= start < end <= native_value.shape[0]:
        raise ValueError("Counterfactual segment lies outside native V")
    expected_shape = (end - start,) + tuple(native_value.shape[1:])
    if target_segment_value.shape != expected_shape:
        raise ValueError(
            "target_segment_value must align with the selected native V "
            f"segment {expected_shape}, got "
            f"{tuple(target_segment_value.shape)}"
        )
    counterfactual = torch.zeros(
        (2,) + tuple(native_value.shape),
        dtype=native_value.dtype,
        device=native_value.device,
    )
    counterfactual[0, start:end] = native_value[start:end]
    counterfactual[1, start:end] = target_segment_value
    return counterfactual


def _evenly_spaced_mask_indices(mask, max_samples):
    """Select deterministic, approximately uniform positions from a mask."""
    if mask.ndim != 1:
        raise ValueError("Query mask must be one-dimensional")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    positions = torch.nonzero(mask.bool(), as_tuple=False).flatten()
    if positions.numel() <= max_samples:
        return positions
    offsets = torch.linspace(
        0,
        positions.numel() - 1,
        steps=max_samples,
        device=positions.device,
    ).round().long()
    return positions[offsets]


@torch.no_grad()
def diagnose_attention_segment(
    query,
    key,
    *,
    segment_start,
    segment_end,
    foreground_mask,
    max_query_samples=16,
    key_chunk_size=4096,
    softmax_scale=None,
):
    """Estimate the softmax mass assigned to one contiguous KV segment.

    ``query`` and ``key`` use the native Wan layout ``[L, H, D]``.  Query
    positions are sampled independently from the foreground and background,
    while every key participates in the denominator.  Key chunking avoids
    materializing the full attention matrix for long native histories.

    The returned ``all`` mass is population-weighted by the actual foreground
    coverage rather than by the deliberately balanced diagnostic sample.
    """
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("Query and key must have shape [L,H,D]")
    if query.shape[1:] != key.shape[1:]:
        raise ValueError("Query and key head dimensions must match")
    if foreground_mask.shape != (query.shape[0],):
        raise ValueError(
            "Foreground mask must align with the query sequence"
        )
    if not 0 <= int(segment_start) <= int(segment_end) <= key.shape[0]:
        raise ValueError("Attention segment lies outside the key sequence")
    if key_chunk_size <= 0:
        raise ValueError("key_chunk_size must be positive")

    foreground_mask = foreground_mask.bool()
    foreground_indices = _evenly_spaced_mask_indices(
        foreground_mask, max_query_samples
    )
    background_indices = _evenly_spaced_mask_indices(
        ~foreground_mask, max_query_samples
    )
    query_indices = torch.cat(
        [foreground_indices, background_indices], dim=0
    )
    zero = query.new_zeros((), dtype=torch.float32)
    if query_indices.numel() == 0 or segment_start == segment_end:
        per_head_zero = query.new_zeros(
            query.shape[1], dtype=torch.float32
        )
        return {
            "all": zero,
            "foreground": zero,
            "background": zero,
            "all_per_head": per_head_zero,
            "foreground_per_head": per_head_zero,
            "background_per_head": per_head_zero,
            "foreground_fraction": foreground_mask.float().mean(),
            "foreground_samples": int(foreground_indices.numel()),
            "background_samples": int(background_indices.numel()),
        }

    sampled_query = query[query_indices].float()
    scale = (
        float(softmax_scale)
        if softmax_scale is not None
        else query.shape[-1] ** -0.5
    )
    denominator_lse = torch.full(
        (query.shape[1], query_indices.numel()),
        -torch.inf,
        dtype=torch.float32,
        device=query.device,
    )
    segment_lse = torch.full_like(denominator_lse, -torch.inf)
    for key_start in range(0, key.shape[0], key_chunk_size):
        key_end = min(key_start + key_chunk_size, key.shape[0])
        logits = torch.einsum(
            "qhd,khd->hqk",
            sampled_query,
            key[key_start:key_end].float(),
        ) * scale
        denominator_lse = torch.logaddexp(
            denominator_lse, logits.logsumexp(dim=-1)
        )
        overlap_start = max(int(segment_start), key_start)
        overlap_end = min(int(segment_end), key_end)
        if overlap_start < overlap_end:
            local_start = overlap_start - key_start
            local_end = overlap_end - key_start
            segment_lse = torch.logaddexp(
                segment_lse,
                logits[..., local_start:local_end].logsumexp(dim=-1),
            )

    mass_per_head_query = (segment_lse - denominator_lse).exp()
    mass_per_query = mass_per_head_query.mean(dim=0)
    foreground_count = foreground_indices.numel()
    foreground_mass = (
        mass_per_query[:foreground_count].mean()
        if foreground_count
        else zero
    )
    background_mass = (
        mass_per_query[foreground_count:].mean()
        if background_indices.numel()
        else zero
    )
    foreground_fraction = foreground_mask.float().mean()
    foreground_mass_per_head = (
        mass_per_head_query[:, :foreground_count].mean(dim=1)
        if foreground_count
        else query.new_zeros(query.shape[1], dtype=torch.float32)
    )
    background_mass_per_head = (
        mass_per_head_query[:, foreground_count:].mean(dim=1)
        if background_indices.numel()
        else query.new_zeros(query.shape[1], dtype=torch.float32)
    )
    all_mass = (
        foreground_fraction * foreground_mass
        + (1.0 - foreground_fraction) * background_mass
    )
    all_mass_per_head = (
        foreground_fraction * foreground_mass_per_head
        + (1.0 - foreground_fraction) * background_mass_per_head
    )
    return {
        "all": all_mass,
        "foreground": foreground_mass,
        "background": background_mass,
        "all_per_head": all_mass_per_head,
        "foreground_per_head": foreground_mass_per_head,
        "background_per_head": background_mass_per_head,
        "foreground_fraction": foreground_fraction,
        "foreground_samples": int(foreground_indices.numel()),
        "background_samples": int(background_indices.numel()),
    }


def resolve_target_identity_correction_strength(
    configured_strength,
    *,
    factorized_target_identity=False,
    immutable_factorized_identity=False,
    prototype_value_is_residual=False,
):
    """Resolve the configured gain for every explicit identity operator.

    Legacy non-factorized absolute anchors intentionally retain their hard
    replacement behavior.  An immutable factorized anchor is an explicit,
    user-configured operator regardless of whether it stores residual or
    absolute values, so both forms must honor ``configured_strength``.
    """
    if not 0.0 <= float(configured_strength) <= 1.0:
        raise ValueError(
            "Identity correction strength must lie in [0, 1]"
        )
    if (
        factorized_target_identity
        or immutable_factorized_identity
        or prototype_value_is_residual
    ):
        return float(configured_strength)
    return 1.0


def closed_loop_counterfactual_memory_attention(
    native_output,
    current_source_query,
    current_source_key,
    current_source_value,
    current_target_key,
    current_target_value,
    canonical_source_key,
    canonical_source_value,
    canonical_target_key,
    canonical_target_value,
    canonical_support,
    canonical_token_index,
    mapped_current_index,
    correspondence_support,
    correspondence_confidence,
    owner_gate,
    appearance_trust,
    transport_confidence,
    *,
    current_address_key=None,
    canonical_address_key=None,
    tokens_per_frame,
    spatial_shape,
    canonical_frame_count,
    current_frame_count,
    topk_per_frame=8,
    min_source_similarity=0.35,
    source_logit_bias=1.0,
    flow_radius=2.0,
    strength=1.0,
    max_error_ratio=1.0,
):
    """Apply a timestep-synchronous counterfactual feedback correction.

    The immutable B0 bank and the current block are evaluated with the same
    source query and the same flow-corresponded identity slots.  The desired
    B0 target-minus-source response is compared with the response already
    present in the current source/target pair; only their error is added to
    the native output.  Clean-source optical flow supplies the coordinate
    correspondence, while clean-source keys only refine candidates inside a
    small flow neighborhood.

    Unsupported queries are selected from ``native_output`` with
    ``torch.where`` so abstention is bit-for-bit exact.
    """
    current_tensors = {
        "native_output": native_output,
        "current_source_query": current_source_query,
        "current_source_key": current_source_key,
        "current_source_value": current_source_value,
        "current_target_key": current_target_key,
        "current_target_value": current_target_value,
    }
    canonical_tensors = {
        "canonical_source_key": canonical_source_key,
        "canonical_source_value": canonical_source_value,
        "canonical_target_key": canonical_target_key,
        "canonical_target_value": canonical_target_value,
    }
    if any(value.ndim != 4 for value in current_tensors.values()):
        raise ValueError(
            "Counterfactual current tensors must have shape [B,L,H,D]"
        )
    if any(value.ndim != 4 for value in canonical_tensors.values()):
        raise ValueError(
            "Counterfactual canonical tensors must have shape [B,K,H,D]"
        )
    if len({value.shape for value in current_tensors.values()}) != 1:
        raise ValueError(
            "Counterfactual current Q/K/V and output must align"
        )
    if len({value.shape for value in canonical_tensors.values()}) != 1:
        raise ValueError(
            "Counterfactual canonical source/target K/V must align"
        )

    batch, query_length, heads, head_dim = current_source_query.shape
    canonical_length = canonical_source_key.shape[1]
    if canonical_length <= 0:
        raise ValueError(
            "Counterfactual canonical memory must contain at least one slot"
        )
    if canonical_source_key.shape[0] != batch or (
        canonical_source_key.shape[2:] != (heads, head_dim)
    ):
        raise ValueError(
            "Counterfactual canonical K/V must align with query heads"
        )
    if query_length != int(tokens_per_frame) * int(current_frame_count):
        raise ValueError(
            "Counterfactual current tokens do not form complete frames"
        )
    expected_query = (batch, query_length)
    expected_canonical = (batch, canonical_length)
    expected_correspondence = (
        batch, int(current_frame_count), canonical_length
    )
    for name, value in (
        ("owner_gate", owner_gate),
        ("appearance_trust", appearance_trust),
        ("transport_confidence", transport_confidence),
    ):
        if value.shape != expected_query:
            raise ValueError(f"Counterfactual {name} must align with queries")
    if (
        canonical_support.shape != expected_canonical
        or canonical_token_index.shape != expected_canonical
    ):
        raise ValueError(
            "Counterfactual canonical support/index must align with K/V"
        )
    for name, value in (
        ("mapped_current_index", mapped_current_index),
        ("correspondence_support", correspondence_support),
        ("correspondence_confidence", correspondence_confidence),
    ):
        if value.shape != expected_correspondence:
            raise ValueError(
                f"Counterfactual {name} must have shape [B,F,K]"
            )
    if int(topk_per_frame) <= 0:
        raise ValueError(
            "Counterfactual top-k per canonical frame must be positive"
        )
    if not -1.0 < float(min_source_similarity) < 1.0:
        raise ValueError(
            "Counterfactual source similarity must lie in (-1, 1)"
        )
    if float(flow_radius) < 0.0:
        raise ValueError("Counterfactual flow radius must be non-negative")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("Counterfactual strength must lie in [0, 1]")
    if float(max_error_ratio) <= 0.0:
        raise ValueError("Counterfactual max error ratio must be positive")

    address_query = (
        current_source_key if current_address_key is None
        else current_address_key
    )
    address_memory = (
        canonical_source_key if canonical_address_key is None
        else canonical_address_key
    )
    if (
        address_query.shape != current_source_key.shape
        or address_memory.shape != canonical_source_key.shape
    ):
        raise ValueError(
            "Counterfactual clean-source address keys must align with K/V"
        )
    source_query = torch.nn.functional.normalize(
        address_query.detach().float().flatten(2), dim=-1, eps=1e-6
    )
    source_memory = torch.nn.functional.normalize(
        address_memory.detach().float().flatten(2), dim=-1, eps=1e-6
    )
    source_similarity = torch.einsum(
        "bqd,bkd->bqk", source_query, source_memory
    )

    query_index = torch.arange(query_length, device=native_output.device)
    query_frame = torch.div(
        query_index, int(tokens_per_frame), rounding_mode="floor"
    )
    query_spatial = query_index.remainder(int(tokens_per_frame))
    mapped_for_query = mapped_current_index.gather(
        1,
        query_frame[None, :, None].expand(batch, -1, canonical_length),
    ).long()
    correspondence_for_query = correspondence_support.gather(
        1,
        query_frame[None, :, None].expand(batch, -1, canonical_length),
    ).bool()
    correspondence_confidence_for_query = correspondence_confidence.gather(
        1,
        query_frame[None, :, None].expand(batch, -1, canonical_length),
    ).float().clamp(0.0, 1.0)
    mapped_spatial = mapped_for_query.remainder(int(tokens_per_frame))

    spatial_height, spatial_width = (int(v) for v in spatial_shape)
    if (
        spatial_height <= 0
        or spatial_width <= 0
        or spatial_height * spatial_width != int(tokens_per_frame)
    ):
        raise ValueError(
            "Counterfactual spatial shape must match tokens_per_frame"
        )
    query_y = torch.div(query_spatial, spatial_width, rounding_mode="floor")
    query_x = query_spatial.remainder(spatial_width)
    mapped_y = torch.div(
        mapped_spatial, spatial_width, rounding_mode="floor"
    )
    mapped_x = mapped_spatial.remainder(spatial_width)
    flow_distance = torch.sqrt(
        (mapped_x.float() - query_x[:, None].float()).square()
        + (mapped_y.float() - query_y[:, None].float()).square()
    )
    flow_neighborhood = (
        correspondence_for_query
        & (mapped_for_query >= 0)
        & (flow_distance <= float(flow_radius))
    )

    canonical_frame = torch.div(
        canonical_token_index.detach().long(),
        int(tokens_per_frame),
        rounding_mode="floor",
    )
    canonical_valid = (
        canonical_support.detach().bool()
        & (canonical_frame >= 0)
        & (canonical_frame < int(canonical_frame_count))
    )
    source_similarity = source_similarity.masked_fill(
        ~(canonical_valid[:, None] & flow_neighborhood), -torch.inf
    )
    candidate_mask = torch.zeros_like(flow_neighborhood)
    frame_candidate_counts = []
    for frame_index in range(int(canonical_frame_count)):
        frame_support = canonical_frame == frame_index
        frame_similarity = source_similarity.masked_fill(
            ~frame_support[:, None], -torch.inf
        )
        candidate_count = min(int(topk_per_frame), canonical_length)
        selected_similarity, selected_index = frame_similarity.topk(
            candidate_count, dim=-1
        )
        selected_valid = (
            torch.isfinite(selected_similarity)
            & (selected_similarity >= float(min_source_similarity))
        )
        selected_mask = torch.zeros_like(candidate_mask)
        selected_mask.scatter_(2, selected_index, selected_valid)
        candidate_mask |= selected_mask
        frame_candidate_counts.append(selected_valid.float().sum(dim=-1))

    frame_candidate_count = torch.stack(frame_candidate_counts, dim=-1)
    candidate_count = candidate_mask.float().sum(dim=-1)
    best_source_similarity = source_similarity.masked_fill(
        ~candidate_mask, -torch.inf
    ).amax(dim=-1)
    best_source_similarity = torch.where(
        torch.isfinite(best_source_similarity),
        best_source_similarity,
        torch.full_like(best_source_similarity, -1.0),
    )
    selected_correspondence_confidence = (
        correspondence_confidence_for_query * candidate_mask.float()
    ).sum(dim=-1) / candidate_count.clamp_min(1.0)

    owner = owner_gate.detach().float().clamp(0.0, 1.0)
    appearance = appearance_trust.detach().float().clamp(0.0, 1.0)
    transport = transport_confidence.detach().float().clamp(0.0, 1.0)
    admitted = (
        (owner > 0.0)
        & (appearance > 0.0)
        & (transport > 0.0)
        & (candidate_count > 0.0)
        & (selected_correspondence_confidence > 0.0)
    )

    desired_response = native_output.new_zeros(
        native_output.shape, dtype=torch.float32
    )
    current_response = torch.zeros_like(desired_response)
    attention_entropy = owner.new_zeros(expected_query)
    attention_peak = owner.new_zeros(expected_query)

    def paired_response(query, source_key, source_value, target_key,
                        target_value, logit_prior, support):
        scale = head_dim ** -0.5
        source_logits = torch.einsum(
            "qhd,qkhd->hqk", query, source_key
        ) * scale + logit_prior[None]
        target_logits = torch.einsum(
            "qhd,qkhd->hqk", query, target_key
        ) * scale + logit_prior[None]
        source_logits = source_logits.masked_fill(
            ~support[None], -torch.inf
        )
        target_logits = target_logits.masked_fill(
            ~support[None], -torch.inf
        )
        source_weight = torch.softmax(source_logits, dim=-1)
        target_weight = torch.softmax(target_logits, dim=-1)
        source_output = torch.einsum(
            "hqk,qkhd->qhd", source_weight, source_value
        )
        target_output = torch.einsum(
            "hqk,qkhd->qhd", target_weight, target_value
        )
        return target_output - source_output, target_weight

    # Query-specific gathers have shape [Q, candidates, heads, head_dim].
    # At Wan channel widths, materializing this for a complete video block
    # can consume several GiB even though only a small owner region is read.
    # Bound the temporary working set without changing candidate selection or
    # any per-query result.
    query_chunk_size = 64
    for batch_index in range(batch):
        active_query = torch.nonzero(
            admitted[batch_index], as_tuple=False
        ).flatten()
        if active_query.numel() == 0:
            continue
        selected_capacity = min(
            canonical_length,
            int(topk_per_frame) * int(canonical_frame_count),
        )
        for query_left in range(0, active_query.numel(), query_chunk_size):
            query_index_chunk = active_query[
                query_left:query_left + query_chunk_size
            ]
            selected_score, selected_slot = source_similarity[
                batch_index, query_index_chunk
            ].masked_fill(
                ~candidate_mask[batch_index, query_index_chunk], -torch.inf
            ).topk(selected_capacity, dim=-1)
            selected_support = torch.isfinite(selected_score)
            selected_current = mapped_for_query[
                batch_index, query_index_chunk
            ].gather(1, selected_slot)
            query = current_source_query[
                batch_index, query_index_chunk
            ].float()
            # Invalid padding is masked again inside paired_response. Keeping
            # its additive prior finite avoids -inf arithmetic in mixed
            # precision kernels while preserving exactly the same support.
            prior = float(source_logit_bias) * torch.where(
                selected_support, selected_score,
                torch.zeros_like(selected_score),
            )

            def gather_canonical(value):
                return value[batch_index][selected_slot].float()

            def gather_current(value):
                return value[batch_index][selected_current].float()

            desired, target_weight = paired_response(
                query, gather_canonical(canonical_source_key),
                gather_canonical(canonical_source_value),
                gather_canonical(canonical_target_key),
                gather_canonical(canonical_target_value), prior,
                selected_support,
            )
            current, _ = paired_response(
                query, gather_current(current_source_key),
                gather_current(current_source_value),
                gather_current(current_target_key),
                gather_current(current_target_value), prior, selected_support,
            )
            desired_response[batch_index, query_index_chunk] = desired
            current_response[batch_index, query_index_chunk] = current
            weight_safe = target_weight.clamp_min(1e-12)
            entropy = -(
                target_weight * weight_safe.log()
            ).sum(dim=-1).mean(dim=0)
            entropy_scale = (
                selected_support.float().sum(dim=-1).clamp_min(2.0).log()
            )
            attention_entropy[batch_index, query_index_chunk] = torch.where(
                selected_support.sum(dim=-1) > 1, entropy / entropy_scale,
                torch.zeros_like(entropy),
            )
            attention_peak[batch_index, query_index_chunk] = (
                target_weight.amax(dim=-1).mean(dim=0)
            )

    error = desired_response - current_response
    desired_norm = desired_response.flatten(2).norm(dim=-1)
    current_norm = current_response.flatten(2).norm(dim=-1)
    error_norm = error.flatten(2).norm(dim=-1)
    reference_norm = torch.maximum(desired_norm, current_norm)
    clip_scale = (
        float(max_error_ratio) * reference_norm / error_norm.clamp_min(1e-6)
    ).clamp(max=1.0)
    clipped_error = error * clip_scale[..., None, None]
    applied_gain = (
        owner * appearance * transport
        * selected_correspondence_confidence
        * float(strength)
    ) * admitted.float()
    active = admitted & (error_norm > 1e-7)
    candidate_output = (
        native_output.float()
        + applied_gain[..., None, None] * clipped_error
    ).to(native_output.dtype)
    output = torch.where(
        active[..., None, None], candidate_output, native_output
    )
    output_delta = (
        output.float() - native_output.float()
    ).abs().mean(dim=(-1, -2))
    residual_error = (
        error - applied_gain[..., None, None] * clipped_error
    ).flatten(2).norm(dim=-1)

    return output, {
        "admitted": admitted.detach(),
        "lineage_admitted": torch.zeros_like(owner).detach(),
        "lineage_confidence": torch.zeros_like(owner).detach(),
        "lineage_tokens": owner.new_zeros((batch,)),
        "read_strength": applied_gain.detach(),
        "applied_read_strength": applied_gain.detach(),
        "output_delta": output_delta.detach(),
        "best_similarity": best_source_similarity.detach(),
        "request_strength": owner.detach(),
        "request_support": (owner > 0.0).float().detach(),
        "read_scope": torch.ones_like(owner).detach(),
        "address_confidence": selected_correspondence_confidence.detach(),
        "flow_transport_confidence": (
            correspondence_confidence_for_query.amax(dim=-1) * transport
        ).detach(),
        "flow_appearance_trust": appearance.detach(),
        "flow_local_transport_confidence": transport.detach(),
        "multiframe_identity_sink": owner.new_zeros(()),
        "canonical_appearance_delta": output_delta.detach(),
        "canonical_payload_exclusive": admitted.float().detach(),
        "mutable_target_payload_enabled": torch.zeros_like(owner).detach(),
        "recent_entry_admitted": torch.zeros_like(admitted),
        "canonical_fallback_admitted": admitted.detach(),
        "recent_payload_consistency": torch.zeros_like(owner).detach(),
        "recent_payload_trust": torch.zeros_like(owner).detach(),
        "recent_payload_rejected": torch.zeros_like(admitted),
        "canonical_payload_weight": admitted.float().detach(),
        "canonical_payload_weight_on_read": admitted.float().mean().detach(),
        "canonical_candidates": canonical_support.float().sum(dim=-1).detach(),
        "recent_tokens": owner.new_zeros((batch,)),
        "recent_payload_tokens": owner.new_zeros((batch,)),
        "admitted_part_similarity": best_source_similarity[admitted].mean().detach()
        if admitted.any() else owner.new_tensor(-1.0),
        "admitted_part_candidate_fraction": (
            candidate_count[admitted].mean().detach()
            if admitted.any() else owner.new_zeros(())
        ),
        "admitted_baseline_output_delta": (
            output_delta.sum() / admitted.float().sum().clamp_min(1.0)
        ).detach(),
        "admitted_part_refinement_scale": owner.new_zeros(()),
        "flow_indexed_read": owner.new_ones(owner.shape).detach(),
        "flow_appearance_trust_on_read": (
            (appearance * admitted.float()).sum()
            / admitted.float().sum().clamp_min(1.0)
        ).detach(),
        "flow_local_transport_confidence_on_read": (
            (transport * admitted.float()).sum()
            / admitted.float().sum().clamp_min(1.0)
        ).detach(),
        "sink_admitted": torch.zeros_like(admitted),
        "tccm_enabled": owner.new_ones(()),
        "tccm_admitted": admitted.detach(),
        "tccm_source_similarity": best_source_similarity.detach(),
        "tccm_correspondence_confidence": (
            selected_correspondence_confidence.detach()
        ),
        "tccm_candidate_count": candidate_count.detach(),
        "tccm_frame_candidate_count": frame_candidate_count.detach(),
        "tccm_desired_norm": desired_norm.detach(),
        "tccm_current_norm": current_norm.detach(),
        "tccm_error_norm": error_norm.detach(),
        "tccm_residual_error_norm": residual_error.detach(),
        "tccm_gain": applied_gain.detach(),
        "tccm_clip_scale": clip_scale.detach(),
        "tccm_attention_entropy": attention_entropy.detach(),
        "tccm_attention_peak": attention_peak.detach(),
    }


def arbitrate_verified_factorized_attention(
    native_output,
    source_mixed_native_output,
    factorized_output,
    verified_read_strength,
    *,
    strength=1.0,
):
    """Transfer attention authority only at verified target-memory reads.

    ``source_mixed_native_output`` is the unmodified StreamGVE attention
    result. ``native_output`` may additionally contain the independent
    role-fixed target-KV correction.  The arbitration replaces only the
    source-mixed backbone component and preserves that target-memory
    correction exactly. Queries without a successful automatic owner/read
    transaction are bit-for-bit native fallback.
    """
    outputs = {
        "native_output": native_output,
        "source_mixed_native_output": source_mixed_native_output,
        "factorized_output": factorized_output,
    }
    if any(value.ndim != 4 for value in outputs.values()):
        raise ValueError(
            "Attention authority tensors must have shape [B,L,H,D]"
        )
    if not (
        native_output.shape
        == source_mixed_native_output.shape
        == factorized_output.shape
    ):
        raise ValueError(
            "Native and factorized attention outputs must share shape"
        )
    if verified_read_strength.shape != native_output.shape[:2]:
        raise ValueError(
            "Verified attention strength must have shape [B,L]"
        )
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError(
            "Verified attention authority strength must lie in [0, 1]"
        )

    authority = (
        verified_read_strength.detach().float().clamp(0.0, 1.0)
        * float(strength)
    )[..., None, None]
    target_memory_delta = (
        native_output.float() - source_mixed_native_output.float()
    )
    arbitrated_backbone = (
        source_mixed_native_output.float()
        + authority
        * (
            factorized_output.float()
            - source_mixed_native_output.float()
        )
    )
    arbitrated = (arbitrated_backbone + target_memory_delta).to(
        native_output.dtype
    )
    # Preserve the exact baseline outside the verified transaction. This is
    # stronger than algebraic equivalence because it avoids round-off from
    # subtracting and re-adding the target-memory residual.
    return torch.where(authority > 0.0, arbitrated, native_output)


def source_flow_gated_multiframe_sink_attention(
    native_output,
    target_query,
    current_source_key,
    canonical_source_key,
    canonical_target_key,
    canonical_source_value,
    canonical_target_value,
    canonical_support,
    canonical_token_index,
    owner_gate,
    appearance_trust,
    transport_confidence,
    *,
    flow_support=None,
    tokens_per_frame,
    frame_count,
    topk_per_frame=8,
    min_source_similarity=0.35,
    source_logit_bias=1.0,
    strength=1.0,
):
    """Read an immutable multi-frame identity residual sink.

    Clean-source keys are the only global address.  They choose a bounded
    candidate set independently in every frozen ignition frame.  Target
    queries may then select part/identity evidence *only* inside that set;
    target keys can never authorize an unrelated canonical token.  Values
    are immutable target-minus-source residuals, leaving current native
    attention responsible for pose, geometry, motion, and occlusion.

    The write gate is deliberately factorized: automatic owner confidence,
    persistent appearance trust, and block-local flow reliability are each
    applied once.  Unsupported queries use a bit-for-bit native fallback.
    """
    query_tensors = {
        "native_output": native_output,
        "target_query": target_query,
        "current_source_key": current_source_key,
    }
    canonical_tensors = {
        "canonical_source_key": canonical_source_key,
        "canonical_target_key": canonical_target_key,
        "canonical_source_value": canonical_source_value,
        "canonical_target_value": canonical_target_value,
    }
    if any(value.ndim != 4 for value in query_tensors.values()):
        raise ValueError(
            "Multi-frame sink query tensors must have shape [B,L,H,D]"
        )
    if any(value.ndim != 4 for value in canonical_tensors.values()):
        raise ValueError(
            "Multi-frame sink tensors must have shape [B,K,H,D]"
        )
    if len({value.shape for value in query_tensors.values()}) != 1:
        raise ValueError(
            "Multi-frame sink queries, source keys, and output must align"
        )
    if len({value.shape for value in canonical_tensors.values()}) != 1:
        raise ValueError(
            "Multi-frame sink canonical source/target K/V must align"
        )

    batch, query_length, heads, head_dim = target_query.shape
    canonical_shape = canonical_source_key.shape
    if canonical_shape[0] != batch or canonical_shape[2:] != (heads, head_dim):
        raise ValueError(
            "Multi-frame sink canonical K/V must align with query heads"
        )
    expected_query_map = (batch, query_length)
    expected_sink_map = canonical_shape[:2]
    for name, value in (
        ("owner_gate", owner_gate),
        ("appearance_trust", appearance_trust),
        ("transport_confidence", transport_confidence),
    ):
        if value.shape != expected_query_map:
            raise ValueError(
                f"Multi-frame sink {name} must align with current queries"
            )
    if flow_support is None:
        flow_support = torch.ones(
            expected_query_map, dtype=torch.bool, device=owner_gate.device
        )
    elif flow_support.shape != expected_query_map:
        raise ValueError(
            "Multi-frame sink flow support must align with current queries"
        )
    if (
        canonical_support.shape != expected_sink_map
        or canonical_token_index.shape != expected_sink_map
    ):
        raise ValueError(
            "Multi-frame sink support/index must align with canonical K/V"
        )
    if int(tokens_per_frame) <= 0 or int(frame_count) <= 0:
        raise ValueError(
            "Multi-frame sink frame and token counts must be positive"
        )
    if int(topk_per_frame) <= 0:
        raise ValueError(
            "Multi-frame sink source top-k per frame must be positive"
        )
    if not -1.0 < float(min_source_similarity) < 1.0:
        raise ValueError(
            "Multi-frame sink source similarity must lie in (-1, 1)"
        )
    if not torch.isfinite(torch.tensor(float(source_logit_bias))):
        raise ValueError(
            "Multi-frame sink source logit bias must be finite"
        )
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError(
            "Multi-frame sink strength must lie in [0, 1]"
        )

    source_query = torch.nn.functional.normalize(
        current_source_key.detach().float().flatten(2), dim=-1, eps=1e-6
    )
    source_memory = torch.nn.functional.normalize(
        canonical_source_key.detach().float().flatten(2), dim=-1, eps=1e-6
    )
    source_similarity = torch.einsum(
        "bqd,bkd->bqk", source_query, source_memory
    )
    canonical_valid = canonical_support.detach().bool()
    source_similarity = source_similarity.masked_fill(
        ~canonical_valid[:, None], -torch.inf
    )

    canonical_frame = torch.div(
        canonical_token_index.detach().long(),
        int(tokens_per_frame),
        rounding_mode="floor",
    )
    valid_frame = (canonical_frame >= 0) & (canonical_frame < int(frame_count))
    canonical_valid = canonical_valid & valid_frame
    candidate_mask = torch.zeros(
        (batch, query_length, canonical_shape[1]),
        dtype=torch.bool,
        device=source_similarity.device,
    )
    frame_candidate_counts = []
    # Equal per-frame source budgets preserve all available ignition views.
    # Target-query attention remains free to prefer the view matching the
    # current part, but a token-rich frame gets no extra addressing rights.
    for frame_index in range(int(frame_count)):
        frame_support = (
            canonical_valid
            & (canonical_frame == frame_index)
        )
        frame_similarity = source_similarity.masked_fill(
            ~frame_support[:, None], -torch.inf
        )
        candidate_count = min(int(topk_per_frame), canonical_shape[1])
        selected_similarity, selected_index = frame_similarity.topk(
            candidate_count, dim=-1
        )
        selected_valid = (
            torch.isfinite(selected_similarity)
            & (selected_similarity >= float(min_source_similarity))
        )
        frame_candidate_mask = torch.zeros_like(candidate_mask)
        frame_candidate_mask.scatter_(
            2, selected_index, selected_valid
        )
        candidate_mask |= frame_candidate_mask
        frame_candidate_counts.append(selected_valid.float().sum(dim=-1))

    frame_candidate_count = torch.stack(frame_candidate_counts, dim=-1)
    candidate_count = candidate_mask.float().sum(dim=-1)
    has_candidate = candidate_count > 0
    best_source_similarity = source_similarity.masked_fill(
        ~candidate_mask, -torch.inf
    ).amax(dim=-1)
    best_source_similarity = torch.where(
        torch.isfinite(best_source_similarity),
        best_source_similarity,
        torch.full_like(best_source_similarity, -1.0),
    )

    owner = owner_gate.detach().float().clamp(0.0, 1.0)
    appearance = appearance_trust.detach().float().clamp(0.0, 1.0)
    local_transport = (
        transport_confidence.detach().float().clamp(0.0, 1.0)
    )
    admitted = (
        (owner > 0.0)
        & flow_support.detach().bool()
        & (appearance > 0.0)
        & (local_transport > 0.0)
        & has_candidate
    )

    canonical_residual = (
        canonical_target_value.float() - canonical_source_value.float()
    )
    sink_delta = native_output.new_zeros(
        native_output.shape, dtype=torch.float32
    )
    attention_entropy = owner.new_zeros((batch, query_length))
    attention_peak = owner.new_zeros((batch, query_length))
    frame_attention = owner.new_zeros(
        (batch, query_length, int(frame_count))
    )
    # Build target-query logits only for queries that passed all preceding
    # source/owner/flow gates. This avoids materializing a dense
    # [B,H,video_tokens,sink_tokens] tensor for background and hand tokens.
    for batch_index in range(batch):
        query_index = torch.nonzero(
            admitted[batch_index], as_tuple=False
        ).flatten()
        if query_index.numel() == 0:
            continue
        selected_target_query = target_query[batch_index, query_index].float()
        target_logits = torch.einsum(
            "qhd,khd->hqk",
            selected_target_query,
            canonical_target_key[batch_index].float(),
        ) * (head_dim ** -0.5)
        target_logits = target_logits + (
            float(source_logit_bias)
            * source_similarity[batch_index, query_index][None]
        )

        # Remove the accidental prior caused by unequal candidate counts
        # across frames. This is a frame-diversity prior, not a manually
        # specified object trajectory or geometry prior.
        slot_frame_count = target_logits.new_ones(
            (query_index.numel(), canonical_shape[1])
        )
        for frame_index in range(int(frame_count)):
            slot_frame_count = torch.where(
                (canonical_frame[batch_index] == frame_index)[None],
                frame_candidate_count[
                    batch_index, query_index, frame_index, None
                ].clamp_min(1.0),
                slot_frame_count,
            )
        target_logits = target_logits - slot_frame_count.log()[None]
        selected_candidates = candidate_mask[batch_index, query_index]
        attention_weight = torch.softmax(
            target_logits.masked_fill(
                ~selected_candidates[None], -torch.inf
            ),
            dim=-1,
        )
        sink_delta[batch_index, query_index] = torch.einsum(
            "hqk,khd->qhd",
            attention_weight, canonical_residual[batch_index],
        )
        weight_safe = attention_weight.clamp_min(1e-12)
        selected_entropy = -(
            attention_weight * weight_safe.log()
        ).sum(dim=-1).mean(dim=0)
        selected_candidate_count = candidate_count[
            batch_index, query_index
        ]
        attention_entropy[batch_index, query_index] = torch.where(
            selected_candidate_count > 1.0,
            selected_entropy
            / selected_candidate_count.clamp_min(2.0).log(),
            torch.zeros_like(selected_entropy),
        )
        attention_peak[batch_index, query_index] = (
            attention_weight.amax(dim=-1).mean(dim=0)
        )
        for frame_index in range(int(frame_count)):
            frame_attention[batch_index, query_index, frame_index] = (
                attention_weight
                * (
                    canonical_frame[batch_index] == frame_index
                )[None, None].float()
            ).sum(dim=-1).mean(dim=0)

    applied_strength = (
        owner * appearance * local_transport * float(strength)
        * admitted.float()
    )
    candidate_output = (
        native_output.float()
        + applied_strength[..., None, None] * sink_delta
    ).to(native_output.dtype)
    output = torch.where(
        admitted[..., None, None], candidate_output, native_output
    )

    selected_frame = frame_attention.argmax(dim=-1).to(torch.float32)
    selected_frame = torch.where(
        admitted, selected_frame, torch.full_like(selected_frame, -1.0)
    )
    sink_coverage = (
        (frame_candidate_count > 0).float().sum(dim=-1)
        / float(frame_count)
    )
    output_delta = (
        output.float() - native_output.float()
    ).abs().mean(dim=(-1, -2))
    admitted_count = admitted.float().sum().clamp_min(1.0)

    def admitted_mean(value):
        return (value * admitted.float()).sum() / admitted_count

    return output, {
        "sink_admitted": admitted.detach(),
        "sink_selected_frame": selected_frame.detach(),
        "sink_source_similarity": best_source_similarity.detach(),
        "sink_attention_entropy": attention_entropy.detach(),
        "sink_attention_peak": attention_peak.detach(),
        "sink_coverage": sink_coverage.detach(),
        "sink_applied_strength": applied_strength.detach(),
        "sink_candidate_count": candidate_count.detach(),
        "sink_frame_candidate_count": frame_candidate_count.detach(),
        "sink_frame_attention": frame_attention.detach(),
        "sink_delta": sink_delta.detach(),
        "sink_selected_frame_on_read": admitted_mean(
            selected_frame.clamp_min(0.0)
        ).detach(),
        "sink_source_similarity_on_read": admitted_mean(
            best_source_similarity
        ).detach(),
        "sink_attention_entropy_on_read": admitted_mean(
            attention_entropy
        ).detach(),
        "sink_attention_peak_on_read": admitted_mean(
            attention_peak
        ).detach(),
        "sink_coverage_on_read": admitted_mean(
            sink_coverage
        ).detach(),
        "sink_applied_strength_on_read": admitted_mean(
            applied_strength
        ).detach(),
        "output_delta": output_delta.detach(),
    }


def source_addressed_native_history_attention(
    native_output,
    target_query,
    canonical_target_key,
    canonical_target_value,
    recent_target_key,
    recent_target_value,
    recent_support,
    current_target_key,
    current_target_value,
    current_source_key,
    canonical_source_key,
    canonical_support,
    query_request,
    *,
    current_source_value=None,
    canonical_source_value=None,
    recent_source_value=None,
    topk=8,
    min_similarity=0.35,
    min_request=0.5,
    canonical_logit_bias=0.0,
    source_part_consistency=False,
    min_part_similarity=0.45,
    part_similarity_margin=0.08,
    part_bias_strength=0.5,
    part_refinement_ratio=0.25,
    payload_invariant_lineage=False,
    recent_source_key=None,
    recent_lineage_index=None,
    recent_lineage_support=None,
    recent_lineage_confidence=None,
    payload_blend_strength=0.35,
    consistent_transaction=False,
    entry_bridge=False,
    motion_owner_dense_read=False,
    entry_query_count=0,
    entry_bridge_strength=1.0,
    dual_evidence_arbitration=False,
    min_payload_consistency=0.15,
    recent_payload_support=None,
    residual_rebased_payload=False,
    last_trusted_appearance=False,
    flow_indexed_value_residual=None,
    flow_indexed_support=None,
    flow_indexed_confidence=None,
    flow_indexed_appearance_trust=None,
    flow_indexed_transport_confidence=None,
    multiframe_identity_sink=False,
    canonical_token_index=None,
    canonical_tokens_per_frame=None,
    canonical_frame_count=None,
    multiframe_sink_topk_per_frame=8,
    multiframe_sink_source_logit_bias=1.0,
    multiframe_sink_strength=1.0,
):
    """Read only native target K/V through a source-addressed gate.

    ``target_query`` and all target keys are expected to carry the desired
    fixed-relative RoPE before entering this function.  Source keys remain
    pre-RoPE and are used only to select compatible tokens in the immutable
    clean-target frame.  The value payload is never projected, averaged into
    a prototype, or modified by a target-minus-source residual.

    The recent and current target K/V form the short-term continuation path.
    Source matching admits top-k long-term canonical keys per object query.
    Queries that fail either the role request or source correspondence are
    copied bit-for-bit from ``native_output``.

    With ``payload_invariant_lineage``, the mutable history contains only
    clean-source addresses and their immutable canonical-slot lineage.  It
    may select canonical target K/V, but no generated recent target K/V is
    consumed.  Thus motion correspondence can advance without recursively
    changing the stored appearance payload.

    Combining ``payload_invariant_lineage`` with ``consistent_transaction``
    makes the payload explicitly appearance-only.  Target and clean-source
    values are read with the same canonical target-key attention weights and
    only their difference is added to the native output.  The current native
    stream therefore remains the carrier of pose, scale, motion, and
    occlusion, while a generated target value can never become future memory.

    With ``source_part_consistency``, centered clean-source value signatures
    provide a bounded soft bias for the already source-addressed canonical
    candidates.  Recent and current keys are never removed: they are the
    local motion/geometry support needed by small objects under large pose
    changes.  The part-aware refinement is measured against the unmodified
    role-fixed read and clipped to a fraction of that read's native-output
    residual.  This makes the proven role-fixed path an explicit trust
    region instead of allowing sparse candidate renormalization to dominate.

    ``entry_bridge`` separates short- and long-timescale memory.  Only the
    first ``entry_query_count`` tokens (normally one latent frame) may consume
    memory.  A source-addressed match in the complete immediately preceding
    clean target block has strict priority; immutable canonical K/V is used
    only when that recent correspondence is unavailable.  All non-entry and
    non-owner queries are copied exactly from ``native_output``.

    ``motion_owner_dense_read`` removes only the first-frame restriction. A
    query must still lie in the causal motion owner and pass clean-source key
    matching. It never modifies the write transaction or admits background.

    ``dual_evidence_arbitration`` prevents a clean-source address match from
    authorizing an arbitrary mutable target payload.  The recent target-minus-
    source value residual is compared with the source-aligned immutable
    canonical residual.  Canonical and recent attention outputs are computed
    independently and combined through a convex payload-consistency trust
    weight.  A rejected recent payload therefore falls back to canonical
    appearance without sparsifying or renormalizing the recent key bank.
    """
    tensors_4d = {
        "native_output": native_output,
        "target_query": target_query,
        "canonical_target_key": canonical_target_key,
        "canonical_target_value": canonical_target_value,
        "recent_target_key": recent_target_key,
        "recent_target_value": recent_target_value,
        "current_target_key": current_target_key,
        "current_target_value": current_target_value,
        "current_source_key": current_source_key,
        "canonical_source_key": canonical_source_key,
    }
    if any(value.ndim != 4 for value in tensors_4d.values()):
        raise ValueError(
            "Native history attention tensors must have shape [B,L,H,D]"
        )
    if topk <= 0:
        raise ValueError("Native history topk must be positive")
    if not -1.0 < float(min_similarity) < 1.0:
        raise ValueError(
            "Native history minimum similarity must lie in (-1, 1)"
        )
    if not 0.0 <= float(min_request) <= 1.0:
        raise ValueError(
            "Native history minimum request must lie in [0, 1]"
        )
    if not torch.isfinite(torch.tensor(float(canonical_logit_bias))):
        raise ValueError("Canonical logit bias must be finite")
    if not -1.0 < float(min_part_similarity) < 1.0:
        raise ValueError(
            "Native history part similarity must lie in (-1, 1)"
        )
    if not 0.0 <= float(part_similarity_margin) <= 2.0:
        raise ValueError(
            "Native history part similarity margin must lie in [0, 2]"
        )
    if not 0.0 <= float(part_bias_strength) <= 4.0:
        raise ValueError(
            "Native history part bias strength must lie in [0, 4]"
        )
    if not 0.0 <= float(part_refinement_ratio) <= 1.0:
        raise ValueError(
            "Native history part refinement ratio must lie in [0, 1]"
        )
    if not 0.0 <= float(payload_blend_strength) <= 1.0:
        raise ValueError(
            "Native history payload blend strength must lie in [0, 1]"
        )
    if not 0.0 <= float(entry_bridge_strength) <= 1.0:
        raise ValueError(
            "Native history entry bridge strength must lie in [0, 1]"
        )
    if not 0.0 <= float(min_payload_consistency) <= 1.0:
        raise ValueError(
            "Native history payload consistency must lie in [0, 1]"
        )

    batch, query_length, heads, head_dim = target_query.shape
    if entry_bridge:
        if payload_invariant_lineage or not consistent_transaction:
            raise ValueError(
                "Native history entry bridge requires a non-invariant "
                "consistent transaction"
            )
        if not 0 < int(entry_query_count) <= query_length:
            raise ValueError(
                "Native history entry query count must lie in [1, L]"
            )
    if motion_owner_dense_read and not entry_bridge:
        raise ValueError(
            "Motion-owner dense read requires the dense recent-target "
            "bridge"
        )
    if dual_evidence_arbitration and not (
        consistent_transaction and entry_bridge
    ):
        raise ValueError(
            "Dual-evidence arbitration requires the consistent recent-"
            "entry bridge"
        )
    query_shape = (batch, query_length, heads, head_dim)
    if native_output.shape != query_shape:
        raise ValueError(
            "Native output and fixed-relative query must share shape"
        )
    if current_source_key.shape != query_shape:
        raise ValueError(
            "Current source keys must align with target queries"
        )
    canonical_shape = canonical_target_key.shape
    if canonical_target_value.shape != canonical_shape:
        raise ValueError(
            "Canonical target keys and values must share shape"
        )
    if canonical_source_key.shape != canonical_shape:
        raise ValueError(
            "Canonical source and target keys must share shape"
        )
    if canonical_shape[0] != batch or canonical_shape[2:] != (heads, head_dim):
        raise ValueError(
            "Canonical native history must align with query heads"
        )
    if recent_target_key.shape != recent_target_value.shape:
        raise ValueError(
            "Recent target keys and values must share shape"
        )
    if recent_support.shape != recent_target_key.shape[:2]:
        raise ValueError(
            "Recent support must align with recent target K/V"
        )
    token_atomic_payload = recent_payload_support is not None
    if recent_payload_support is None:
        recent_payload_support = recent_support
    if recent_payload_support.shape != recent_target_key.shape[:2]:
        raise ValueError(
            "Recent payload support must align with recent target K/V"
        )
    if residual_rebased_payload and not (
        token_atomic_payload and consistent_transaction
    ):
        raise ValueError(
            "Residual-rebased payloads require a token-atomic consistent "
            "transaction"
        )
    if residual_rebased_payload and recent_source_value is None:
        raise ValueError(
            "Residual-rebased payloads require recent clean-source values"
        )
    if last_trusted_appearance and not residual_rebased_payload:
        raise ValueError(
            "Last-trusted appearance requires residual-rebased payloads"
        )
    flow_indexed_read = flow_indexed_value_residual is not None
    if flow_indexed_read:
        if flow_indexed_support is None or flow_indexed_confidence is None:
            raise ValueError(
                "Flow-indexed residual read requires support and confidence"
            )
        if flow_indexed_value_residual.shape != native_output.shape:
            raise ValueError(
                "Flow-indexed V residual must align with current queries"
            )
        if (
            flow_indexed_support.shape != (batch, query_length)
            or flow_indexed_confidence.shape != (batch, query_length)
        ):
            raise ValueError(
                "Flow-indexed support/confidence must align with queries"
            )
        for name, value in (
            ("appearance trust", flow_indexed_appearance_trust),
            ("transport confidence", flow_indexed_transport_confidence),
        ):
            if value is not None and value.shape != (batch, query_length):
                raise ValueError(
                    f"Flow-indexed {name} must align with queries"
                )
        if not (
            consistent_transaction
            and residual_rebased_payload
            and last_trusted_appearance
        ):
            raise ValueError(
                "Flow-indexed read requires a last-trusted residual "
                "transaction"
            )
    if multiframe_identity_sink:
        if not flow_indexed_read:
            raise ValueError(
                "Multi-frame identity sink requires flow-indexed trust"
            )
        if canonical_source_value is None:
            raise ValueError(
                "Multi-frame identity sink requires canonical source values"
            )
        if canonical_source_value.shape != canonical_shape:
            raise ValueError(
                "Canonical source values must align with the identity sink"
            )
        if canonical_token_index is None or (
            canonical_token_index.shape != canonical_shape[:2]
        ):
            raise ValueError(
                "Multi-frame identity sink requires canonical token indices"
            )
        if (
            canonical_tokens_per_frame is None
            or canonical_frame_count is None
        ):
            raise ValueError(
                "Multi-frame identity sink requires canonical frame metadata"
            )
    if current_target_key.shape != current_target_value.shape:
        raise ValueError(
            "Current target keys and values must share shape"
        )
    for name, value in (
        ("recent_target_key", recent_target_key),
        ("current_target_key", current_target_key),
    ):
        if value.shape[0] != batch or value.shape[2:] != (heads, head_dim):
            raise ValueError(
                f"{name} must align with query batch and heads"
            )
    if current_target_key.shape[1] != query_length:
        raise ValueError(
            "Current target K/V must align with target queries"
        )
    if canonical_support.shape != canonical_shape[:2]:
        raise ValueError(
            "Canonical support must align with canonical tokens"
        )
    if query_request.shape != (batch, query_length):
        raise ValueError(
            "Native history requests must align with current queries"
        )
    if source_part_consistency:
        source_values = {
            "current_source_value": current_source_value,
            "canonical_source_value": canonical_source_value,
        }
        if any(value is None for value in source_values.values()):
            raise ValueError(
                "Part-consistent native history requires current, "
                "and canonical clean-source values"
            )
        expected_shapes = {
            "current_source_value": current_source_key.shape,
            "canonical_source_value": canonical_source_key.shape,
        }
        for name, value in source_values.items():
            if value.shape != expected_shapes[name]:
                raise ValueError(
                    f"{name} must align with its corresponding keys"
                )
    if dual_evidence_arbitration:
        payload_evidence_values = {
            "canonical_source_value": canonical_source_value,
            "recent_source_value": recent_source_value,
        }
        if any(value is None for value in payload_evidence_values.values()):
            raise ValueError(
                "Dual-evidence arbitration requires canonical and recent "
                "clean-source values"
            )
        expected_shapes = {
            "canonical_source_value": canonical_target_value.shape,
            "recent_source_value": recent_target_value.shape,
        }
        for name, value in payload_evidence_values.items():
            if value.shape != expected_shapes[name]:
                raise ValueError(
                    f"{name} must align with its target payload"
                )
    if payload_invariant_lineage:
        lineage_tensors = {
            "recent_source_key": recent_source_key,
            "recent_lineage_index": recent_lineage_index,
            "recent_lineage_support": recent_lineage_support,
            "recent_lineage_confidence": recent_lineage_confidence,
        }
        if any(value is None for value in lineage_tensors.values()):
            raise ValueError(
                "Payload-invariant native history requires complete "
                "source lineage metadata"
            )
        if recent_source_key.ndim != 4 or (
            recent_source_key.shape[0] != batch
            or recent_source_key.shape[2:] != (heads, head_dim)
        ):
            raise ValueError(
                "Recent source lineage keys must align with queries"
            )
        lineage_shape = recent_source_key.shape[:2]
        for name, value in (
            ("recent_lineage_index", recent_lineage_index),
            ("recent_lineage_support", recent_lineage_support),
            ("recent_lineage_confidence", recent_lineage_confidence),
        ):
            if value.shape != lineage_shape:
                raise ValueError(
                    f"{name} must align with recent source lineage keys"
                )
        if recent_lineage_support.any():
            valid_lineage_index = recent_lineage_index[
                recent_lineage_support.bool()
            ]
            if (
                valid_lineage_index.min() < 0
                or valid_lineage_index.max() >= canonical_shape[1]
            ):
                raise ValueError(
                    "Source lineage references an invalid canonical slot"
                )

    def source_part_signature(value):
        flat = value.detach().float().flatten(2)
        centered = flat - flat.mean(dim=-1, keepdim=True)
        return torch.nn.functional.normalize(
            centered, dim=-1, eps=1e-6
        )

    # Clean-source correspondence is deliberately computed before RoPE.
    # Flattening heads produces one stable content address per visual token
    # without injecting a learned or hand-designed appearance payload.
    source_query = torch.nn.functional.normalize(
        current_source_key.detach().float().flatten(2),
        dim=-1,
    )
    source_memory = torch.nn.functional.normalize(
        canonical_source_key.detach().float().flatten(2),
        dim=-1,
    )
    similarity = torch.einsum("bqd,bkd->bqk", source_query, source_memory)
    canonical_valid = canonical_support.detach().bool()
    similarity = similarity.masked_fill(
        ~canonical_valid[:, None, :], -torch.inf
    )
    candidate_count = min(int(topk), canonical_shape[1])
    best_similarity, best_index = similarity.topk(
        candidate_count, dim=-1
    )
    direct_best_match = best_similarity[..., 0]
    lineage_best_similarity = None
    lineage_best_index = None
    lineage_selected_canonical = None
    lineage_match_confidence = None
    if payload_invariant_lineage:
        source_lineage = torch.nn.functional.normalize(
            recent_source_key.detach().float().flatten(2), dim=-1
        )
        lineage_similarity = torch.einsum(
            "bqd,bkd->bqk", source_query, source_lineage
        )
        lineage_valid = (
            recent_lineage_support.detach().bool()
            & (recent_lineage_index.detach() >= 0)
        )
        lineage_similarity = lineage_similarity.masked_fill(
            ~lineage_valid[:, None, :], -torch.inf
        )
        lineage_candidate_count = min(
            int(topk), recent_source_key.shape[1]
        )
        if lineage_candidate_count > 0:
            lineage_best_similarity, lineage_best_index = (
                lineage_similarity.topk(lineage_candidate_count, dim=-1)
            )
            lineage_selected_canonical = (
                recent_lineage_index.detach().long()[:, None, :]
                .expand(-1, query_length, -1)
                .gather(2, lineage_best_index)
            )
            lineage_selected_confidence = (
                recent_lineage_confidence.detach().float()[:, None, :]
                .expand(-1, query_length, -1)
                .gather(2, lineage_best_index)
            )
            lineage_match = lineage_best_similarity[..., 0]
            lineage_match_confidence = lineage_selected_confidence[..., 0]
        else:
            lineage_match = direct_best_match.new_full(
                direct_best_match.shape, -torch.inf
            )
            lineage_match_confidence = direct_best_match.new_zeros(
                direct_best_match.shape
            )
        best_match = torch.maximum(direct_best_match, lineage_match)
    else:
        best_match = direct_best_match
    request = query_request.detach().float().clamp(0.0, 1.0)
    if entry_bridge:
        if motion_owner_dense_read:
            # Geometry ownership is a high-recall request, not an appearance
            # admission. Every owned frame may ask for memory; the independent
            # clean-source address check below still decides whether it reads.
            entry_query = torch.ones_like(request, dtype=torch.bool)
            request_support = (request > 0.0) & entry_query
        else:
            entry_query = (
                torch.arange(query_length, device=request.device)
                < int(entry_query_count)
            )[None].expand(batch, -1)
            request_support = (request >= float(min_request)) & entry_query
    else:
        entry_query = torch.ones_like(request, dtype=torch.bool)
        request_support = (
            request > 0.0
            if (payload_invariant_lineage or consistent_transaction)
            else request >= float(min_request)
        )
    admitted = (
        request_support
        & torch.isfinite(best_match)
        & (best_match >= float(min_similarity))
    )

    if flow_indexed_read:
        # A source-flow coordinate is the identity address.  Target K never
        # participates in this path, so a generated key cannot redirect a
        # trusted residual to a different part or cause a chunk-boundary ID
        # switch. Low-confidence/occluded flow fails closed to native output.
        legacy_flow_confidence = (
            flow_indexed_confidence.detach().float().clamp(0.0, 1.0)
        )
        appearance_trust = (
            legacy_flow_confidence
            if flow_indexed_appearance_trust is None
            else flow_indexed_appearance_trust.detach().float().clamp(
                0.0, 1.0
            )
        )
        local_transport_confidence = (
            torch.ones_like(legacy_flow_confidence)
            if flow_indexed_transport_confidence is None
            else flow_indexed_transport_confidence.detach().float().clamp(
                0.0, 1.0
            )
        )
        flow_confidence = (
            legacy_flow_confidence
            if (
                flow_indexed_appearance_trust is None
                or flow_indexed_transport_confidence is None
            )
            else (appearance_trust * local_transport_confidence).clamp(
                0.0, 1.0
            )
        )
        if multiframe_identity_sink:
            output, sink_diagnostics = (
                source_flow_gated_multiframe_sink_attention(
                    native_output=native_output,
                    target_query=target_query,
                    current_source_key=current_source_key,
                    canonical_source_key=canonical_source_key,
                    canonical_target_key=canonical_target_key,
                    canonical_source_value=canonical_source_value,
                    canonical_target_value=canonical_target_value,
                    canonical_support=canonical_support,
                    canonical_token_index=canonical_token_index,
                    owner_gate=request,
                    appearance_trust=appearance_trust,
                    transport_confidence=local_transport_confidence,
                    flow_support=flow_indexed_support,
                    tokens_per_frame=int(canonical_tokens_per_frame),
                    frame_count=int(canonical_frame_count),
                    topk_per_frame=int(multiframe_sink_topk_per_frame),
                    min_source_similarity=float(min_similarity),
                    source_logit_bias=float(
                        multiframe_sink_source_logit_bias
                    ),
                    strength=float(multiframe_sink_strength),
                )
            )
            admitted = sink_diagnostics["sink_admitted"]
            applied_strength = sink_diagnostics["sink_applied_strength"]
            zeros = applied_strength.new_zeros(applied_strength.shape)
            admitted_count = admitted.float().sum().clamp_min(1.0)
            diagnostics = {
                "admitted": admitted,
                "request_strength": request.detach(),
                "request_support": request_support.detach().float(),
                "read_scope": entry_query.detach().float(),
                "address_confidence": (
                    sink_diagnostics["sink_source_similarity"]
                    .add(1.0).mul(0.5).clamp(0.0, 1.0)
                ),
                "best_similarity": sink_diagnostics[
                    "sink_source_similarity"
                ],
                "output_delta": sink_diagnostics["output_delta"],
                "read_strength": applied_strength,
                "applied_read_strength": applied_strength,
                "canonical_appearance_delta": sink_diagnostics[
                    "output_delta"
                ],
                "canonical_payload_exclusive": admitted.float(),
                "mutable_target_payload_enabled": zeros,
                "entry_query": entry_query.detach(),
                "recent_entry_admitted": torch.zeros_like(admitted),
                "canonical_fallback_admitted": admitted,
                "recent_payload_consistency": zeros,
                "residual_rebased_payload": torch.ones_like(zeros),
                "last_trusted_appearance": torch.ones_like(zeros),
                "recent_payload_trust": zeros,
                "recent_payload_rejected": torch.zeros_like(admitted),
                "canonical_payload_weight": admitted.float(),
                "recent_payload_consistency_on_match": zeros.new_zeros(()),
                "recent_payload_trust_on_match": zeros.new_zeros(()),
                "recent_payload_rejection_rate": zeros.new_zeros(()),
                "canonical_payload_weight_on_read": (
                    admitted.float().sum() / admitted_count
                ).detach(),
                "canonical_candidates": canonical_support.float().sum(
                    dim=-1
                ).detach(),
                "recent_tokens": zeros.new_full(
                    (batch,), float(recent_target_key.shape[1])
                ),
                "recent_payload_tokens": zeros.new_zeros((batch,)),
                "lineage_admitted": zeros,
                "lineage_confidence": zeros,
                "lineage_tokens": zeros.new_zeros((batch,)),
                "admitted_part_similarity": (
                    sink_diagnostics["sink_source_similarity"][admitted].mean()
                    if admitted.any() else zeros.new_tensor(-1.0)
                ).detach(),
                "admitted_part_candidate_fraction": (
                    sink_diagnostics["sink_coverage"][admitted].mean()
                    if admitted.any() else zeros.new_zeros(())
                ).detach(),
                "admitted_baseline_output_delta": (
                    sink_diagnostics["output_delta"].sum()
                    / admitted_count
                ).detach(),
                "admitted_part_refinement_scale": zeros.new_zeros(()),
                "flow_indexed_read": torch.ones_like(zeros),
                "flow_transport_confidence": flow_confidence.detach(),
                "flow_appearance_trust": appearance_trust.detach(),
                "flow_local_transport_confidence": (
                    local_transport_confidence.detach()
                ),
                "flow_appearance_trust_on_read": (
                    (appearance_trust * admitted.float()).sum()
                    / admitted_count
                ).detach(),
                "flow_local_transport_confidence_on_read": (
                    (local_transport_confidence * admitted.float()).sum()
                    / admitted_count
                ).detach(),
                "multiframe_identity_sink": torch.ones_like(zeros),
                **sink_diagnostics,
            }
            diagnostics["read_strength"] = applied_strength.detach()
            diagnostics["applied_read_strength"] = (
                applied_strength.detach()
            )
            return output, diagnostics
        admitted = (
            request_support
            & flow_indexed_support.detach().bool()
            & (flow_confidence > 0.0)
        )
        request_scale = max(float(min_request), 1e-6)
        request_confidence = (request / request_scale).clamp(0.0, 1.0)
        read_strength = torch.sqrt(
            (request_confidence * flow_confidence).clamp(0.0, 1.0)
        ) * admitted.float()
        applied_strength = read_strength * float(
            entry_bridge_strength if entry_bridge else payload_blend_strength
        )
        residual = flow_indexed_value_residual.to(
            device=native_output.device, dtype=torch.float32
        )
        output = (
            native_output.float()
            + applied_strength[:, :, None, None] * residual
        ).to(native_output.dtype)
        output_delta = (
            output.float() - native_output.float()
        ).abs().mean(dim=(-1, -2))
        zeros = torch.zeros_like(read_strength)
        admitted_count = admitted.float().sum().clamp_min(1.0)
        diagnostics = {
            "admitted": admitted.detach(),
            "request_strength": request.detach(),
            "request_support": request_support.detach().float(),
            "read_scope": entry_query.detach().float(),
            "address_confidence": flow_confidence.detach(),
            "best_similarity": flow_confidence.mul(2.0).sub(1.0).detach(),
            "output_delta": output_delta.detach(),
            "read_strength": read_strength.detach(),
            "applied_read_strength": applied_strength.detach(),
            "canonical_appearance_delta": zeros.detach(),
            "canonical_payload_exclusive": zeros.detach(),
            "mutable_target_payload_enabled": admitted.float().detach(),
            "entry_query": entry_query.detach(),
            "recent_entry_admitted": admitted.detach(),
            "canonical_fallback_admitted": torch.zeros_like(admitted),
            "recent_payload_consistency": zeros.detach(),
            "residual_rebased_payload": torch.ones_like(zeros),
            "last_trusted_appearance": torch.ones_like(zeros),
            "recent_payload_trust": flow_confidence.detach(),
            "recent_payload_rejected": torch.zeros_like(admitted),
            "canonical_payload_weight": zeros.detach(),
            "recent_payload_consistency_on_match": zeros.new_zeros(()),
            "recent_payload_trust_on_match": (
                (flow_confidence * admitted.float()).sum()
                / admitted.float().sum().clamp_min(1.0)
            ).detach(),
            "recent_payload_rejection_rate": zeros.new_zeros(()),
            "canonical_payload_weight_on_read": zeros.new_zeros(()),
            "canonical_candidates": canonical_support.float().sum(
                dim=-1
            ).detach(),
            "recent_tokens": zeros.new_full(
                (batch,), float(recent_target_key.shape[1])
            ),
            "recent_payload_tokens": flow_indexed_support.float().sum(
                dim=-1
            ).detach(),
            "lineage_admitted": zeros.detach(),
            "lineage_confidence": zeros.detach(),
            "lineage_tokens": zeros.new_zeros((batch,)),
            "admitted_part_similarity": zeros.new_full((), -1.0),
            "admitted_part_candidate_fraction": zeros.new_zeros(()),
            "admitted_baseline_output_delta": output_delta.sum().div(
                admitted.float().sum().clamp_min(1.0)
            ).detach(),
            "admitted_part_refinement_scale": zeros.new_zeros(()),
            "flow_indexed_read": torch.ones_like(zeros),
            "flow_transport_confidence": flow_confidence.detach(),
            "flow_appearance_trust": appearance_trust.detach(),
            "flow_local_transport_confidence": (
                local_transport_confidence.detach()
            ),
            "flow_appearance_trust_on_read": (
                (appearance_trust * admitted.float()).sum()
                / admitted_count
            ).detach(),
            "flow_local_transport_confidence_on_read": (
                (local_transport_confidence * admitted.float()).sum()
                / admitted_count
            ).detach(),
        }
        return output, diagnostics

    output = native_output.clone()
    output_delta = native_output.new_zeros((batch, query_length), dtype=torch.float32)
    effective_read_strength = output_delta.new_zeros(
        (batch, query_length)
    )
    applied_read_strength_map = output_delta.new_zeros(
        (batch, query_length)
    )
    canonical_appearance_delta = output_delta.new_zeros(
        (batch, query_length)
    )
    canonical_key_count = canonical_shape[1]
    recent_best_similarity = None
    recent_best_index = None
    recent_best_payload_support = None
    recent_match = torch.zeros_like(admitted)
    recent_payload_token_consistency = None
    recent_payload_consistency_output = output_delta.new_zeros(
        (batch, query_length)
    )
    recent_payload_trust_output = output_delta.new_zeros(
        (batch, query_length)
    )
    recent_payload_rejected = torch.zeros_like(admitted)
    canonical_direct_admitted = (
        request_support
        & torch.isfinite(direct_best_match)
        & (direct_best_match >= float(min_similarity))
    )
    if consistent_transaction and not payload_invariant_lineage:
        if recent_source_key is None:
            raise ValueError(
                "Consistent native-KV transactions require recent "
                "clean-source keys"
            )
        if recent_source_key.shape != recent_target_key.shape:
            raise ValueError(
                "Recent clean-source and target keys must share shape"
            )
        recent_count = recent_source_key.shape[1]
        if recent_count > 0:
            recent_source_memory = torch.nn.functional.normalize(
                recent_source_key.detach().float().flatten(2), dim=-1
            )
            recent_similarity = torch.einsum(
                "bqd,bkd->bqk", source_query, recent_source_memory
            ).masked_fill(
                ~recent_support.detach().bool()[:, None, :],
                -torch.inf,
            )
            recent_candidate_count = min(int(topk), recent_count)
            recent_best_similarity, recent_best_index = (
                recent_similarity.topk(recent_candidate_count, dim=-1)
            )
            recent_best_payload_support = (
                recent_payload_support.detach().bool()[:, None, :]
                .expand(-1, query_length, -1)
                .gather(2, recent_best_index)
            )
            recent_valid_candidates = (
                torch.isfinite(recent_best_similarity)
                & (recent_best_similarity >= float(min_similarity))
                & recent_best_payload_support
            )
            # 958 authorizes only the nearest clean-source address. In the
            # last-trusted ledger, payload support is intentionally sparse
            # and persists independently of the dense address table; use the
            # nearest *authorized* top-k address instead. This widens reads,
            # never writes, and every selected payload still passed the
            # one-to-one source-lineage transaction.
            recent_match = (
                recent_valid_candidates.any(dim=-1)
                if last_trusted_appearance
                else recent_valid_candidates[..., 0]
            )
            trusted_recent_similarity = torch.where(
                recent_valid_candidates,
                recent_best_similarity,
                torch.full_like(recent_best_similarity, -torch.inf),
            ).max(dim=-1).values
            best_match = torch.maximum(
                best_match, trusted_recent_similarity
            )
            admitted = (
                request_support
                & torch.isfinite(best_match)
                & (best_match >= float(min_similarity))
            )
            if token_atomic_payload:
                # Source K remains a dense address table, but target K/V is
                # consumable only at tokenwise committed locations. Missing
                # authorization is exact-native abstention; it must not pull
                # raw first-block K/V into a different pose or scale.
                admitted = request_support & recent_match
            if dual_evidence_arbitration:
                # Address evidence answers where the object moved.  It must
                # not also certify the generated appearance stored at that
                # address.  Audit every dense recent payload against the
                # immutable ignition edit residual using an independently
                # source-aligned canonical token.
                recent_to_canonical = torch.einsum(
                    "bqd,bkd->bqk",
                    recent_source_memory,
                    source_memory,
                ).masked_fill(
                    ~canonical_valid[:, None, :], -torch.inf
                )
                recent_canonical_similarity, recent_canonical_index = (
                    recent_to_canonical.max(dim=-1)
                )
                gather_index = recent_canonical_index[:, :, None, None].expand(
                    -1, -1, heads, head_dim
                )
                aligned_canonical_target = canonical_target_value.gather(
                    1, gather_index
                )
                aligned_canonical_source = canonical_source_value.gather(
                    1, gather_index
                )
                recent_residual = (
                    recent_target_value.float()
                    - recent_source_value.float()
                ).flatten(2)
                canonical_residual = (
                    aligned_canonical_target.float()
                    - aligned_canonical_source.float()
                ).flatten(2)
                recent_norm = recent_residual.norm(dim=-1)
                canonical_norm = canonical_residual.norm(dim=-1)
                residual_direction = (
                    torch.nn.functional.normalize(
                        recent_residual, dim=-1, eps=1e-6
                    )
                    * torch.nn.functional.normalize(
                        canonical_residual, dim=-1, eps=1e-6
                    )
                ).sum(dim=-1).clamp(0.0, 1.0)
                residual_scale_ratio = (
                    recent_norm / canonical_norm.clamp_min(1e-6)
                )
                residual_scale_agreement = torch.minimum(
                    residual_scale_ratio,
                    residual_scale_ratio.clamp_min(1e-6).reciprocal(),
                ).clamp(0.0, 1.0)
                payload_valid = (
                    recent_support.detach().bool()
                    & torch.isfinite(recent_canonical_similarity)
                    & (
                        recent_canonical_similarity
                        >= float(min_similarity)
                    )
                    & (recent_norm > 1e-6)
                    & (canonical_norm > 1e-6)
                )
                recent_payload_token_consistency = torch.where(
                    payload_valid,
                    residual_direction * residual_scale_agreement,
                    torch.zeros_like(residual_direction),
                )

    if payload_invariant_lineage:
        # The native branch already carries the current block's geometry and
        # motion.  This auxiliary read contains immutable appearance only.
        target_key = canonical_target_key
        target_value = canonical_target_value
    elif consistent_transaction:
        # The complete current/previous blocks already participate in native
        # causal self-attention.  This auxiliary branch is an appearance
        # transaction, so it may consume only explicitly committed object
        # payloads.
        target_key = torch.cat(
            [canonical_target_key, recent_target_key], dim=1
        )
        target_value = torch.cat(
            [canonical_target_value, recent_target_value], dim=1
        )
    else:
        target_key = torch.cat(
            [canonical_target_key, recent_target_key, current_target_key],
            dim=1,
        )
        target_value = torch.cat(
            [
                canonical_target_value,
                recent_target_value,
                current_target_value,
            ],
            dim=1,
        )
    if target_key.dtype != target_value.dtype:
        target_key = target_key.to(target_value.dtype)
    target_query = target_query.to(target_value.dtype)
    part_similarity_output = output_delta.new_full(
        (batch, query_length), -1.0
    )
    part_candidate_fraction = output_delta.new_zeros(
        (batch, query_length)
    )
    baseline_output_delta = output_delta.new_zeros(
        (batch, query_length)
    )
    part_refinement_scale = output_delta.new_zeros(
        (batch, query_length)
    )
    if source_part_consistency:
        current_part = source_part_signature(current_source_value)
        canonical_part = source_part_signature(canonical_source_value)

    # Batch items can contain different numbers of admitted object tokens.
    # A short loop keeps the candidate mask compact and, importantly, never
    # evaluates or alters unsupported hand/background/boundary queries.
    for batch_index in range(batch):
        query_index = torch.nonzero(
            admitted[batch_index], as_tuple=False
        ).flatten()
        if query_index.numel() == 0:
            continue
        selected_query = target_query[
            batch_index:batch_index + 1, query_index
        ]
        selected_match = best_similarity[batch_index, query_index]
        selected_index = best_index[batch_index, query_index]
        selected_valid = (
            torch.isfinite(selected_match)
            & (selected_match >= float(min_similarity))
        )
        key_mask = torch.ones(
            (query_index.numel(), target_key.shape[1]),
            dtype=torch.bool,
            device=target_key.device,
        )
        if consistent_transaction:
            key_mask.zero_()
        else:
            key_mask[:, :canonical_key_count] = False
        recent_start = canonical_key_count
        recent_end = (
            recent_start
            if payload_invariant_lineage
            else recent_start + recent_target_key.shape[1]
        )
        if not payload_invariant_lineage and not consistent_transaction:
            key_mask[:, recent_start:recent_end] = (
                recent_support[batch_index].to(
                    device=target_key.device, dtype=torch.bool
                )[None]
            )
        row = torch.arange(
            query_index.numel(), device=target_key.device
        )[:, None].expand_as(selected_index)
        selected_recent_match = recent_match[batch_index, query_index]
        selected_payload_trust = output_delta.new_zeros(
            (query_index.numel(),)
        )
        selected_payload_route_available = torch.ones(
            (query_index.numel(),), dtype=torch.bool,
            device=query_index.device,
        )
        selected_canonical_valid = (
            selected_valid & ~selected_recent_match[:, None]
            if entry_bridge and not token_atomic_payload
            else selected_valid
        )
        if token_atomic_payload:
            selected_canonical_valid = torch.zeros_like(
                selected_canonical_valid
            )
        key_mask[
            row[selected_canonical_valid],
            selected_index[selected_canonical_valid],
        ] = True
        if (
            consistent_transaction
            and recent_best_index is not None
            and recent_best_similarity is not None
        ):
            selected_recent_similarity = recent_best_similarity[
                batch_index, query_index
            ]
            selected_recent_index = recent_best_index[
                batch_index, query_index
            ]
            selected_recent_valid = (
                torch.isfinite(selected_recent_similarity)
                & (
                    selected_recent_similarity
                    >= float(min_similarity)
                )
                & recent_best_payload_support[
                    batch_index, query_index
                ]
            )
            if dual_evidence_arbitration:
                selected_recent_payload_consistency = (
                    recent_payload_token_consistency[
                        batch_index
                    ][selected_recent_index]
                )
                valid_payload_count = selected_recent_valid.float().sum(
                    dim=-1
                ).clamp_min(1.0)
                selected_payload_consistency = (
                    torch.where(
                        selected_recent_valid,
                        selected_recent_payload_consistency,
                        torch.zeros_like(
                            selected_recent_payload_consistency
                        ),
                    ).sum(dim=-1)
                    / valid_payload_count
                ).clamp(0.0, 1.0)
                selected_payload_trust = torch.where(
                    selected_recent_match
                    & (
                        selected_payload_consistency
                        >= float(min_payload_consistency)
                    ),
                    selected_payload_consistency,
                    torch.zeros_like(selected_payload_consistency),
                )
                recent_payload_consistency_output[
                    batch_index, query_index
                ] = torch.where(
                    selected_recent_match,
                    selected_payload_consistency,
                    torch.zeros_like(selected_payload_consistency),
                )
                recent_payload_trust_output[
                    batch_index, query_index
                ] = selected_payload_trust
                recent_payload_rejected[
                    batch_index, query_index
                ] = (
                    selected_recent_match
                    & (selected_payload_trust <= 0.0)
                )
            if entry_bridge:
                # A valid recent source correspondence owns this query.  The
                # immutable tier is a long-term fallback, never a competing
                # value bank that can pull entry geometry back to block zero.
                selected_recent_valid &= selected_recent_match[:, None]
            recent_row = torch.arange(
                query_index.numel(), device=target_key.device
            )[:, None].expand_as(selected_recent_index)
            key_mask[
                recent_row[selected_recent_valid],
                recent_start + selected_recent_index[
                    selected_recent_valid
                ],
            ] = True
        if payload_invariant_lineage and lineage_best_index is not None:
            selected_lineage_similarity = lineage_best_similarity[
                batch_index, query_index
            ]
            selected_lineage_canonical = lineage_selected_canonical[
                batch_index, query_index
            ]
            selected_lineage_valid = (
                torch.isfinite(selected_lineage_similarity)
                & (
                    selected_lineage_similarity
                    >= float(min_similarity)
                )
                & (selected_lineage_canonical >= 0)
            )
            lineage_row = torch.arange(
                query_index.numel(), device=target_key.device
            )[:, None].expand_as(selected_lineage_canonical)
            key_mask[
                lineage_row[selected_lineage_valid],
                selected_lineage_canonical[selected_lineage_valid],
            ] = True
        logit_bias = torch.zeros(
            (query_index.numel(), target_key.shape[1]),
            dtype=target_query.dtype,
            device=target_key.device,
        )
        logit_bias[:, :canonical_key_count] = float(canonical_logit_bias)
        baseline_attention_mask = logit_bias.masked_fill(
            ~key_mask, -torch.inf
        )
        baseline_output = torch.nn.functional.scaled_dot_product_attention(
            selected_query.transpose(1, 2),
            target_key[batch_index:batch_index + 1].transpose(1, 2),
            target_value[batch_index:batch_index + 1].transpose(1, 2),
            attn_mask=baseline_attention_mask[None, None],
        ).transpose(1, 2).contiguous()
        selected_output = baseline_output
        if residual_rebased_payload:
            # The persistent tier stores target-minus-source appearance at a
            # current clean-source address.  Read target and source values with
            # exactly the same target-key attention weights, then inject only
            # their difference into the current native stream.  This preserves
            # current pose, scale, motion, and occlusion instead of copying the
            # previous block's full target output.
            residual_source_value = torch.cat(
                [canonical_source_value, recent_source_value], dim=1
            ).to(device=target_value.device, dtype=target_value.dtype)
            source_baseline_output = (
                torch.nn.functional.scaled_dot_product_attention(
                    selected_query.transpose(1, 2),
                    target_key[
                        batch_index:batch_index + 1
                    ].transpose(1, 2),
                    residual_source_value[
                        batch_index:batch_index + 1
                    ].transpose(1, 2),
                    attn_mask=baseline_attention_mask[None, None],
                ).transpose(1, 2).contiguous()
            )
            selected_output = (
                native_output[
                    batch_index:batch_index + 1, query_index
                ].float()
                + baseline_output.float()
                - source_baseline_output.float()
            ).to(baseline_output.dtype)
        if dual_evidence_arbitration:
            # Do not delete recent keys and let a tiny surviving set dominate
            # through softmax renormalization.  Instead compute the immutable
            # canonical branch independently, then place the recent result in
            # a convex trust region determined by payload evidence.
            canonical_key_mask = torch.zeros_like(key_mask)
            canonical_key_mask[
                row[selected_valid], selected_index[selected_valid]
            ] = True
            canonical_available = selected_valid.any(dim=-1)
            selected_payload_route_available = (
                canonical_available
                | (selected_recent_match & (selected_payload_trust > 0.0))
            )
            # SDPA cannot consume an all-masked row.  Install a numerically
            # safe placeholder key for source locations with no immutable
            # correspondence; its result is replaced by exact native output.
            canonical_key_mask[~canonical_available, 0] = True
            canonical_attention_mask = logit_bias.masked_fill(
                ~canonical_key_mask, -torch.inf
            )
            canonical_output = (
                torch.nn.functional.scaled_dot_product_attention(
                    selected_query.transpose(1, 2),
                    target_key[
                        batch_index:batch_index + 1
                    ].transpose(1, 2),
                    target_value[
                        batch_index:batch_index + 1
                    ].transpose(1, 2),
                    attn_mask=canonical_attention_mask[None, None],
                ).transpose(1, 2).contiguous()
            )
            recent_mix = selected_payload_trust[:, None, None]
            canonical_or_native = torch.where(
                canonical_available[None, :, None, None],
                canonical_output.float(),
                native_output[
                    batch_index:batch_index + 1, query_index
                ].float(),
            )
            selected_output = torch.where(
                selected_recent_match[None, :, None, None],
                canonical_or_native
                + recent_mix[None]
                * (baseline_output.float() - canonical_or_native),
                baseline_output.float(),
            ).to(baseline_output.dtype)
        if payload_invariant_lineage:
            # A bounded correction retains native motion while making the
            # only cross-block target payload the ignition appearance. Query
            # confidence is consumed once as a soft read strength; the
            # correspondence threshold above remains binary access control.
            direct_admitted = (
                torch.isfinite(direct_best_match[
                    batch_index, query_index
                ])
                & (
                    direct_best_match[batch_index, query_index]
                    >= float(min_similarity)
                )
            )
            lineage_confidence = lineage_match_confidence[
                batch_index, query_index
            ].clamp(0.0, 1.0)
            address_confidence = torch.where(
                direct_admitted,
                torch.ones_like(lineage_confidence),
                lineage_confidence,
            )
            raw_read_strength = (
                query_request[batch_index, query_index]
                .detach().float().clamp(0.0, 1.0)
                * address_confidence
            )[:, None, None]
            applied_read_strength = (
                raw_read_strength * float(payload_blend_strength)
            )
            if consistent_transaction:
                if canonical_source_value is None:
                    raise ValueError(
                        "Canonical residual transactions require clean-"
                        "source canonical values"
                    )
                # Use exactly the same target-key attention weights for the
                # canonical target and source values. Common geometry/content
                # cancels in V_target - V_source; only the immutable edit
                # residual is injected into the current native stream.
                canonical_source_output = (
                    torch.nn.functional.scaled_dot_product_attention(
                        selected_query.transpose(1, 2),
                        target_key[
                            batch_index:batch_index + 1
                        ].transpose(1, 2),
                        canonical_source_value[
                            batch_index:batch_index + 1
                        ].to(
                            device=target_value.device,
                            dtype=target_value.dtype,
                        ).transpose(1, 2),
                        attn_mask=baseline_attention_mask[None, None],
                    ).transpose(1, 2).contiguous()
                )
                appearance_delta = (
                    baseline_output.float()
                    - canonical_source_output.float()
                )
                selected_output = (
                    native_output[
                        batch_index:batch_index + 1, query_index
                    ].float()
                    + applied_read_strength[None] * appearance_delta
                ).to(baseline_output.dtype)
                canonical_appearance_delta[
                    batch_index, query_index
                ] = appearance_delta.abs().mean(dim=(-1, -2))[0]
                # Authority/source arbitration should consume retrieval
                # confidence, not the independently tunable payload gain.
                effective_read_strength[batch_index, query_index] = (
                    raw_read_strength[:, 0, 0]
                )
            else:
                selected_output = (
                    native_output[
                        batch_index:batch_index + 1, query_index
                    ].float()
                    + applied_read_strength[None]
                    * (
                        baseline_output.float()
                        - native_output[
                            batch_index:batch_index + 1, query_index
                        ].float()
                    )
                ).to(baseline_output.dtype)
                # Preserve the legacy payload-invariant diagnostic contract.
                effective_read_strength[batch_index, query_index] = (
                    applied_read_strength[:, 0, 0]
                )
            applied_read_strength_map[batch_index, query_index] = (
                applied_read_strength[:, 0, 0]
            )
        elif consistent_transaction:
            # Preserve native motion/geometry and inject only a confidence-
            # calibrated memory residual.  ``min_request`` is the confidence
            # at which an owner request saturates, not a destructive cutoff.
            request_scale = max(float(min_request), 1e-6)
            query_confidence = (
                request[batch_index, query_index] / request_scale
            ).clamp(0.0, 1.0).sqrt()
            address_confidence = (
                (best_match[batch_index, query_index] - float(min_similarity))
                / max(1.0 - float(min_similarity), 1e-6)
            ).clamp(0.0, 1.0).sqrt()
            if entry_bridge and not motion_owner_dense_read:
                # The automatic owner request already encodes hand proximity,
                # semantic evidence, source transport and flow agreement.  Do
                # not square-root it for the entry transaction: high-confidence
                # owner queries should provide a real boundary condition, while
                # the hard threshold above keeps uncertain/background queries
                # on the exact native path.
                read_strength = query_confidence[:, None, None]
            elif motion_owner_dense_read:
                # A transported geometry owner widens reads only. Appearance
                # remains conditioned on an independently verified source
                # address, preventing the wider request from reading target
                # payload into background tokens.
                joint_read_evidence = (
                    request[batch_index, query_index]
                    * address_confidence
                ).clamp(0.0, 1.0)
                # The 958 product treats owner confidence and address
                # confidence as two independent attenuation gains.  In the
                # last-trusted transaction they are instead two observations
                # of the same access event: use their geometric mean.  A
                # payload must still pass both gates, but a verified residual
                # no longer loses most of its authority merely because both
                # calibrated confidences are soft.  Source arbitration later
                # consumes this exact same strength.
                if last_trusted_appearance:
                    joint_read_evidence = joint_read_evidence.sqrt()
                read_strength = joint_read_evidence[:, None, None]
            else:
                read_strength = (
                    query_confidence * address_confidence
                )[:, None, None]
            if dual_evidence_arbitration:
                read_strength = read_strength * (
                    selected_payload_route_available[:, None, None].float()
                )
            applied_read_strength = read_strength * float(
                entry_bridge_strength
                if entry_bridge
                else payload_blend_strength
            )
            effective_read_strength[batch_index, query_index] = (
                read_strength[:, 0, 0]
            )
            applied_read_strength_map[batch_index, query_index] = (
                applied_read_strength[:, 0, 0]
            )
            selected_output = (
                native_output[
                    batch_index:batch_index + 1, query_index
                ].float()
                + applied_read_strength[None]
                * (
                    selected_output.float()
                    - native_output[
                        batch_index:batch_index + 1, query_index
                    ].float()
                )
            ).to(baseline_output.dtype)
        baseline_delta = (
            selected_output.float()
            - native_output[batch_index:batch_index + 1, query_index].float()
        )
        baseline_output_delta[batch_index, query_index] = (
            baseline_delta.abs().mean(dim=(-1, -2))[0]
        )
        if source_part_consistency and float(part_bias_strength) > 0.0:
            selected_part = current_part[batch_index, query_index]
            canonical_part_similarity = torch.einsum(
                "qd,kd->qk",
                selected_part,
                canonical_part[batch_index],
            )
            selected_part_similarity = canonical_part_similarity.gather(
                dim=1, index=selected_index
            )
            valid_part_similarity = torch.where(
                selected_valid,
                selected_part_similarity,
                torch.full_like(selected_part_similarity, -torch.inf),
            )
            best_part = valid_part_similarity.max(dim=-1).values
            best_part = torch.where(
                torch.isfinite(best_part),
                best_part,
                torch.full_like(best_part, -1.0),
            )
            part_similarity_output[batch_index, query_index] = best_part
            valid_count = selected_valid.float().sum(dim=-1).clamp_min(1.0)
            part_candidate_fraction[batch_index, query_index] = (
                (
                    selected_valid
                    & (
                        selected_part_similarity
                        >= float(min_part_similarity)
                    )
                ).float().sum(dim=-1)
                / valid_count
            )

            # Only canonical logits receive a bounded preference.  Crucially,
            # no canonical candidate and no recent/current token is deleted.
            # This avoids the 941b failure where a tiny surviving key set was
            # renormalized into an order-of-magnitude attention intervention.
            part_scale_denominator = max(
                float(part_similarity_margin), 1e-6
            )
            selected_part_bias = (
                (
                    selected_part_similarity
                    - float(min_part_similarity)
                )
                / part_scale_denominator
            ).clamp(-1.0, 1.0) * float(part_bias_strength)
            part_logit_bias = torch.zeros_like(logit_bias)
            valid_row = row[selected_valid]
            valid_column = selected_index[selected_valid]
            part_logit_bias[valid_row, valid_column] = (
                selected_part_bias[selected_valid]
            ).to(part_logit_bias.dtype)
            refined_attention_mask = (
                logit_bias + part_logit_bias
            ).masked_fill(~key_mask, -torch.inf)
            refined_output = (
                torch.nn.functional.scaled_dot_product_attention(
                    selected_query.transpose(1, 2),
                    target_key[
                        batch_index:batch_index + 1
                    ].transpose(1, 2),
                    target_value[
                        batch_index:batch_index + 1
                    ].transpose(1, 2),
                    attn_mask=refined_attention_mask[None, None],
                ).transpose(1, 2).contiguous()
            )
            refinement = refined_output.float() - baseline_output.float()
            baseline_norm = baseline_delta.square().mean(
                dim=(-1, -2), keepdim=True
            ).sqrt()
            refinement_norm = refinement.square().mean(
                dim=(-1, -2), keepdim=True
            ).sqrt()
            refinement_limit = (
                baseline_norm * float(part_refinement_ratio)
            )
            refinement_scale = torch.minimum(
                torch.ones_like(refinement_norm),
                refinement_limit / refinement_norm.clamp_min(1e-6),
            )
            if consistent_transaction:
                selected_output = (
                    selected_output.float()
                    + refinement
                    * refinement_scale
                    * applied_read_strength[None]
                ).to(baseline_output.dtype)
            else:
                selected_output = (
                    baseline_output.float()
                    + refinement * refinement_scale
                ).to(baseline_output.dtype)
            part_refinement_scale[batch_index, query_index] = (
                refinement_scale[..., 0, 0][0]
            )
        selected_output = selected_output.to(native_output.dtype)
        output[batch_index, query_index] = selected_output[0]
        output_delta[batch_index, query_index] = (
            selected_output[0].float()
            - native_output[batch_index, query_index].float()
        ).abs().mean(dim=(-1, -2))

    admitted_count = admitted.float().sum().clamp_min(1.0)
    recent_address_admitted = admitted & recent_match & entry_query
    recent_address_count = recent_address_admitted.float().sum().clamp_min(1.0)
    canonical_payload_weight = torch.where(
        canonical_direct_admitted,
        torch.where(
            recent_match,
            1.0 - recent_payload_trust_output
            if dual_evidence_arbitration
            else torch.zeros_like(recent_payload_trust_output),
            torch.ones_like(recent_payload_trust_output),
        ),
        torch.zeros_like(recent_payload_trust_output),
    )
    if token_atomic_payload:
        # Canonical addresses are still computed for diagnostics, but this
        # mode deliberately never consumes their target payload.
        canonical_payload_weight = torch.zeros_like(
            canonical_payload_weight
        )
    diagnostics = {
        "admitted": admitted.detach(),
        "request_strength": request.detach(),
        "request_support": request_support.detach().float(),
        "read_scope": entry_query.detach().float(),
        "address_confidence": (
            (best_match - float(min_similarity))
            / max(1.0 - float(min_similarity), 1e-6)
        ).clamp(0.0, 1.0).detach(),
        "best_similarity": torch.where(
            torch.isfinite(best_match),
            best_match,
            torch.full_like(best_match, -1.0),
        ).detach(),
        "output_delta": output_delta.detach(),
        "read_strength": (
            effective_read_strength.detach()
            if (payload_invariant_lineage or consistent_transaction)
            else admitted.detach().float()
        ),
        "applied_read_strength": (
            applied_read_strength_map.detach()
        ),
        "canonical_appearance_delta": (
            canonical_appearance_delta.detach()
        ),
        "canonical_payload_exclusive": (
            torch.where(
                admitted,
                torch.ones_like(output_delta),
                torch.zeros_like(output_delta),
            )
            if payload_invariant_lineage
            else torch.zeros_like(output_delta)
        ).detach(),
        # Mutable target payload is structurally absent in lineage mode.
        # Exposing this explicitly makes accidental future regressions visible
        # in run logs instead of requiring post-hoc video diagnosis.
        "mutable_target_payload_enabled": (
            torch.zeros_like(output_delta)
            if payload_invariant_lineage
            else recent_payload_trust_output
            if dual_evidence_arbitration
            else torch.where(
                admitted,
                torch.ones_like(output_delta),
                torch.zeros_like(output_delta),
            )
        ).detach(),
        "entry_query": entry_query.detach(),
        "recent_entry_admitted": (
            admitted
            & recent_match
            & (
                recent_payload_trust_output > 0.0
                if dual_evidence_arbitration
                else torch.ones_like(recent_match)
            )
            & entry_query
        ).detach(),
        "canonical_fallback_admitted": (
            canonical_direct_admitted
            & ~torch.full_like(
                canonical_direct_admitted, token_atomic_payload
            )
            & (
                (~recent_match)
                | (
                    recent_payload_rejected
                    if dual_evidence_arbitration
                    else torch.zeros_like(recent_match)
                )
            )
            & entry_query
        ).detach(),
        "recent_payload_consistency": (
            recent_payload_consistency_output.detach()
        ),
        "residual_rebased_payload": output_delta.new_full(
            (batch, query_length), float(bool(residual_rebased_payload))
        ),
        "last_trusted_appearance": output_delta.new_full(
            (batch, query_length), float(bool(last_trusted_appearance))
        ),
        "recent_payload_trust": recent_payload_trust_output.detach(),
        "recent_payload_rejected": recent_payload_rejected.detach(),
        "canonical_payload_weight": canonical_payload_weight.detach(),
        "recent_payload_consistency_on_match": (
            (
                recent_payload_consistency_output
                * recent_address_admitted.float()
            ).sum()
            / recent_address_count
        ).detach(),
        "recent_payload_trust_on_match": (
            (recent_payload_trust_output * recent_address_admitted.float()).sum()
            / recent_address_count
        ).detach(),
        "recent_payload_rejection_rate": (
            recent_payload_rejected.float().sum() / recent_address_count
        ).detach(),
        "canonical_payload_weight_on_read": (
            canonical_payload_weight.sum()
            / admitted.float().sum().clamp_min(1.0)
        ).detach(),
        "canonical_candidates": canonical_valid.float().sum(dim=-1).detach(),
        "recent_tokens": (
            recent_support.float().sum(dim=-1).to(output_delta)
            if consistent_transaction
            else output_delta.new_full(
                (batch,),
                0.0
                if payload_invariant_lineage
                else float(recent_target_key.shape[1]),
            )
        ),
        "recent_payload_tokens": (
            recent_payload_support.float().sum(dim=-1).to(output_delta)
            if consistent_transaction
            else output_delta.new_zeros((batch,))
        ),
        "lineage_tokens": output_delta.new_full(
            (batch,),
            float(recent_source_key.shape[1])
            if payload_invariant_lineage
            else 0.0,
        ),
        "lineage_admitted": (
            (
                torch.isfinite(lineage_best_similarity[..., 0])
                & (
                    lineage_best_similarity[..., 0]
                    >= float(min_similarity)
                )
                & admitted
            ).detach()
            if payload_invariant_lineage
            and lineage_best_similarity is not None
            else torch.zeros_like(admitted)
        ),
        "lineage_confidence": (
            torch.where(
                admitted,
                lineage_match_confidence.clamp(0.0, 1.0),
                torch.zeros_like(lineage_match_confidence),
            ).detach()
            if payload_invariant_lineage
            and lineage_match_confidence is not None
            else torch.zeros_like(output_delta)
        ),
        "part_similarity": part_similarity_output.detach(),
        "part_candidate_fraction": (
            part_candidate_fraction.detach()
        ),
        "baseline_output_delta": baseline_output_delta.detach(),
        "part_refinement_scale": part_refinement_scale.detach(),
        "admitted_part_similarity": (
            (
                part_similarity_output
                * admitted.float()
            ).sum().div(admitted_count).detach()
            if source_part_consistency
            else output_delta.new_zeros(())
        ),
        "admitted_part_candidate_fraction": (
            (
                part_candidate_fraction
                * admitted.float()
            ).sum().div(admitted_count).detach()
            if source_part_consistency
            else output_delta.new_zeros(())
        ),
        "admitted_baseline_output_delta": (
            (
                baseline_output_delta * admitted.float()
            ).sum().div(admitted_count).detach()
        ),
        "admitted_part_refinement_scale": (
            (
                part_refinement_scale * admitted.float()
            ).sum().div(admitted_count).detach()
            if source_part_consistency
            else output_delta.new_zeros(())
        ),
    }
    return output, diagnostics


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out


def fuse_aligned_memory(
    target_key,
    target_value,
    source_key,
    source_value,
    preserve_action,
):
    """Fuse aligned target/source KV with a per-memory-token action."""
    if target_key.shape != source_key.shape:
        raise ValueError("Aligned target/source keys must share shape")
    if target_value.shape != source_value.shape:
        raise ValueError("Aligned target/source values must share shape")
    if target_key.shape[:2] != preserve_action.shape:
        raise ValueError(
            "Preserve action must align with memory tokens"
        )
    weight = preserve_action.float().clamp(0.0, 1.0)
    weight = weight[:, :, None, None]
    fused_key = (
        target_key.float()
        + weight * (source_key.float() - target_key.float())
    ).to(target_key.dtype)
    fused_value = (
        target_value.float()
        + weight * (source_value.float() - target_value.float())
    ).to(target_value.dtype)
    return fused_key, fused_value


def fuse_factorized_aligned_memory(
    target_key,
    target_value,
    source_key,
    source_value,
    source_key_action,
    source_value_action,
    target_memory_action,
    unknown_action,
):
    """Apply independent geometry and appearance provenance actions.

    Source keys may provide correspondence for target-owned object tokens,
    while their values remain target-owned. Unknown tokens use the native
    target-memory value instead of being silently converted to source-owned.
    """
    if target_key.shape != source_key.shape:
        raise ValueError(
            "Factorized aligned target/source keys must share shape"
        )
    if target_value.shape != source_value.shape:
        raise ValueError(
            "Factorized aligned target/source values must share shape"
        )
    actions = {
        "source_key_action": source_key_action,
        "source_value_action": source_value_action,
        "target_memory_action": target_memory_action,
        "unknown_action": unknown_action,
    }
    expected_shape = target_key.shape[:2]
    for name, action in actions.items():
        if action.shape != expected_shape:
            raise ValueError(
                f"{name} must align with factorized memory tokens"
            )

    source_key_weight = source_key_action.float()[..., None, None]
    source_value_weight = source_value_action.float()[..., None, None]
    target_value_weight = (
        target_memory_action.float() + unknown_action.float()
    )[..., None, None]
    fused_key = (
        target_key.float()
        + source_key_weight
        * (source_key.float() - target_key.float())
    ).to(target_key.dtype)
    fused_value = (
        target_value.float() * target_value_weight
        + source_value.float() * source_value_weight
    ).to(target_value.dtype)
    return fused_key, fused_value


def blend_factorized_with_native_fallback(
    factorized_output,
    native_output,
    unknown_action,
):
    """Blend abstaining queries while keeping endpoint paths exact."""
    if factorized_output.shape != native_output.shape:
        raise ValueError(
            "Factorized and native attention outputs must share shape"
        )
    if unknown_action.shape != factorized_output.shape[:2]:
        raise ValueError(
            "Unknown action must align with attention queries"
        )
    weight = unknown_action.float().clamp(0.0, 1.0)[..., None, None]
    blended = (
        factorized_output.float()
        + weight
        * (native_output.float() - factorized_output.float())
    ).to(factorized_output.dtype)
    blended = torch.where(
        (unknown_action >= 1.0)[..., None, None],
        native_output,
        blended,
    )
    return torch.where(
        (unknown_action <= 0.0)[..., None, None],
        factorized_output,
        blended,
    )


def build_factorized_history_read_mask(
    source_value_action,
    target_memory_action,
    eps=1e-6,
):
    """Expose only history with an assigned source/target provenance."""
    if source_value_action.shape != target_memory_action.shape:
        raise ValueError(
            "Source and target history actions must share shape"
        )
    if eps <= 0:
        raise ValueError("eps must be positive")
    return (
        source_value_action.float() + target_memory_action.float()
    ) > eps


def _validate_target_owned_mask(
    tensor,
    target_owned_mask,
    *,
    name,
):
    expected_shape = tensor.shape[:-2]
    if target_owned_mask.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got "
            f"{tuple(target_owned_mask.shape)}"
        )


def blend_target_owned_tensor(
    target,
    source,
    target_weight,
    target_owned_mask=None,
):
    """Blend aligned tensors while keeping owned tokens target-only.

    A missing ownership mask is deliberately the exact legacy blend.  The
    mask spans every leading token dimension and leaves head/channel axes
    untouched.
    """
    if target.shape != source.shape:
        raise ValueError(
            "Aligned target/source tensors must share shape"
        )
    blended = (
        target * target_weight
        + source * (1 - target_weight)
    )
    if target_owned_mask is None:
        return blended
    _validate_target_owned_mask(
        target,
        target_owned_mask,
        name="Target-owned mask",
    )
    owned = target_owned_mask.detach().bool()
    return torch.where(
        owned[..., None, None],
        target,
        blended,
    )


def suppress_source_preserve_on_target_owned_history(
    preserve_action,
    target_owned_history_mask=None,
):
    """Set source-preserve action to zero only on owned history."""
    if target_owned_history_mask is None:
        return preserve_action
    if preserve_action.shape != target_owned_history_mask.shape:
        raise ValueError(
            "Target-owned history and preserve action must share shape"
        )
    return torch.where(
        target_owned_history_mask.detach().bool(),
        torch.zeros_like(preserve_action),
        preserve_action,
    )


def build_target_owned_source_background_mask(
    current_edit_mask,
    current_target_owned_mask,
):
    """Keep source context except at edit and owned object tokens."""
    if current_edit_mask.shape != current_target_owned_mask.shape:
        raise ValueError(
            "Current edit and target-owned masks must share shape"
        )
    return (
        ~current_edit_mask.detach().bool()
        & ~current_target_owned_mask.detach().bool()
    )


def scatter_target_owned_output(
    legacy_output,
    target_owned_output,
    current_target_owned_mask,
):
    """Scatter compact owned-query output into exact legacy output."""
    if legacy_output.ndim != 4 or target_owned_output.ndim != 4:
        raise ValueError(
            "Attention outputs must have shape [B,L,H,D]"
        )
    _validate_target_owned_mask(
        legacy_output,
        current_target_owned_mask,
        name="Current target-owned mask",
    )
    owned = current_target_owned_mask.detach().bool()
    if legacy_output.shape[0] != 1 or owned.shape[0] != 1:
        raise ValueError(
            "Compact owned-query scatter expects one sample at a time"
        )
    if target_owned_output.shape != (
        1,
        int(owned.sum().item()),
        *legacy_output.shape[2:],
    ):
        raise ValueError(
            "Compact target-owned output does not match owned query count"
        )
    output = legacy_output.clone()
    output[:, owned[0]] = target_owned_output
    return output


def apply_target_identity_value_correction(
    correspondence_key,
    target_value,
    prototype_key,
    prototype_value,
    prototype_evidence,
    source_value=None,
    prototype_value_is_residual=False,
    residual_subspace=False,
    target_appearance_key=None,
    prototype_appearance_key=None,
    tokens_per_frame=None,
    support_mask=None,
    support_floor=0.0,
    min_similarity=0.55,
    correction_strength=1.0,
    return_diagnostics=False,
    eps=1e-6,
):
    """Transport target appearance using factorized correspondence."""
    if correspondence_key.shape != target_value.shape:
        raise ValueError(
            "Correspondence keys and target values must share shape"
        )
    if prototype_key.shape != prototype_value.shape:
        raise ValueError(
            "Identity prototype keys and values must share shape"
        )
    if residual_subspace and not prototype_value_is_residual:
        raise ValueError(
            "Residual-subspace correction requires residual prototypes"
        )
    if prototype_value_is_residual:
        if source_value is None:
            raise ValueError(
                "Residual identity correction requires source_value"
            )
        if source_value.shape != target_value.shape:
            raise ValueError(
                "Source and target values must share shape"
            )
    if correspondence_key.ndim != 4 or prototype_key.ndim != 4:
        raise ValueError(
            "Identity correction expects [B,L,H,D] tensors"
        )
    if correspondence_key.shape[0] != prototype_key.shape[0]:
        raise ValueError(
            "Correspondence query and identity memory must share batch size"
        )
    if correspondence_key.shape[2:] != prototype_key.shape[2:]:
        raise ValueError(
            "Correspondence query and identity memory must share heads"
        )
    if prototype_evidence.shape != prototype_key.shape[:2]:
        raise ValueError(
            "Identity evidence must have shape [B,P]"
        )
    if (target_appearance_key is None) != (
        prototype_appearance_key is None
    ):
        raise ValueError(
            "Current and prototype appearance keys must be provided "
            "together"
        )
    if target_appearance_key is not None:
        if target_appearance_key.shape != correspondence_key.shape:
            raise ValueError(
                "Current appearance and correspondence keys must align"
            )
        if prototype_appearance_key.shape != prototype_key.shape:
            raise ValueError(
                "Prototype appearance and correspondence keys must align"
            )
    if tokens_per_frame is not None and (
        tokens_per_frame <= 0
        or correspondence_key.shape[1] % tokens_per_frame != 0
    ):
        raise ValueError(
            "tokens_per_frame must evenly divide the target sequence"
        )
    if support_mask is not None and support_mask.shape != (
        correspondence_key.shape[:2]
    ):
        raise ValueError(
            "Identity support mask must have shape [B,L]"
        )
    if not -1.0 < min_similarity < 1.0:
        raise ValueError(
            "min_similarity must lie in (-1, 1)"
        )
    if not 0.0 <= support_floor <= 1.0:
        raise ValueError("support_floor must lie in [0, 1]")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if not 0.0 <= correction_strength <= 1.0:
        raise ValueError(
            "correction_strength must lie in [0, 1]"
        )

    key = torch.nn.functional.normalize(
        correspondence_key.float(),
        dim=-1,
    )
    memory_key = torch.nn.functional.normalize(
        prototype_key.float(),
        dim=-1,
    )
    correspondence_similarity = torch.einsum(
        "blhd,bphd->blph",
        key,
        memory_key,
    ).mean(dim=-1).clamp(-1.0, 1.0)
    assignment_similarity = correspondence_similarity
    joint_similarity = correspondence_similarity
    if target_appearance_key is not None:
        current_appearance = torch.nn.functional.normalize(
            target_appearance_key.float(),
            dim=-1,
        )
        memory_appearance = torch.nn.functional.normalize(
            prototype_appearance_key.float(),
            dim=-1,
        )
        appearance_similarity = torch.einsum(
            "blhd,bphd->blph",
            current_appearance,
            memory_appearance,
        ).mean(dim=-1).clamp(-1.0, 1.0)
        joint_similarity = torch.minimum(
            correspondence_similarity,
            appearance_similarity,
        )
        # Keep correspondence and appearance tied to the same prototype.
        # Averaging the two scores can assign a value from a prototype that
        # matches only one factor while another prototype supplied the joint
        # support signal.
        assignment_similarity = joint_similarity
    valid = prototype_evidence > eps
    confidence = prototype_evidence.float().clamp(0.0, 1.0)

    flat_similarity = assignment_similarity.flatten(1)
    median_similarity = torch.quantile(
        flat_similarity,
        0.50,
        dim=-1,
        keepdim=True,
    )
    absolute_deviation = (
        flat_similarity - median_similarity
    ).abs()
    robust_scale = (
        torch.quantile(
            absolute_deviation,
            0.50,
            dim=-1,
            keepdim=True,
        )
        / 0.6745
    ).clamp_min(eps)
    logits = assignment_similarity / robust_scale.unsqueeze(-1)
    logits = logits + confidence.clamp_min(eps).log()[:, None, :]
    logits = logits.masked_fill(
        ~valid[:, None, :],
        torch.finfo(logits.dtype).min,
    )
    assignment = torch.softmax(logits, dim=-1)
    has_memory = valid.any(dim=-1, keepdim=True)
    assignment = torch.where(
        has_memory.unsqueeze(-1),
        assignment,
        torch.zeros_like(assignment),
    )
    assignment_peak, selected_prototype = assignment.max(dim=-1)
    valid_count = valid.sum(dim=-1, keepdim=True)
    entropy_normalizer = valid_count.float().log().clamp_min(eps)
    assignment_entropy = -(
        assignment
        * assignment.clamp_min(eps).log()
    ).sum(dim=-1) / entropy_normalizer
    assignment_entropy = torch.where(
        has_memory & (valid_count > 1),
        assignment_entropy,
        torch.zeros_like(assignment_entropy),
    ).clamp(0.0, 1.0)
    if assignment.shape[-1] > 1:
        top_two = assignment.topk(k=2, dim=-1).values
        assignment_margin = top_two[..., 0] - top_two[..., 1]
    else:
        assignment_margin = assignment_peak
    assignment_peak = torch.where(
        has_memory,
        assignment_peak,
        torch.zeros_like(assignment_peak),
    )
    assignment_margin = torch.where(
        has_memory,
        assignment_margin,
        torch.zeros_like(assignment_margin),
    )
    selected_prototype = torch.where(
        has_memory,
        selected_prototype,
        torch.full_like(selected_prototype, -1),
    )

    retrieved_value = torch.einsum(
        "blp,bphd->blhd",
        assignment,
        prototype_value.float(),
    )
    best_similarity = joint_similarity.masked_fill(
        ~valid[:, None, :],
        -1.0,
    ).max(dim=-1).values
    support_similarity = (
        best_similarity
        if tokens_per_frame is None
        else best_similarity.reshape(
            best_similarity.shape[0],
            -1,
            tokens_per_frame,
        )
    )
    support_threshold = torch.quantile(
        support_similarity,
        0.90,
        dim=-1,
        keepdim=True,
    )
    high_match = torch.quantile(
        support_similarity,
        0.99,
        dim=-1,
        keepdim=True,
    )
    match_spread = high_match - support_threshold
    absolute_support = (
        (support_similarity - min_similarity)
        / (1.0 - min_similarity)
    ).clamp(0.0, 1.0)
    relative_match = (
        (support_similarity - support_threshold)
        / match_spread.clamp_min(eps)
    ).clamp(0.0, 1.0)
    relative_match = torch.where(
        match_spread > eps,
        relative_match,
        absolute_support,
    )
    relative_match = relative_match.reshape_as(best_similarity)
    absolute_match = absolute_support.reshape_as(best_similarity)
    assigned_confidence = (
        assignment * confidence[:, None, :]
    ).sum(dim=-1)
    identity_support = (
        torch.sqrt(relative_match * absolute_match)
        * assigned_confidence
    ).clamp(0.0, 1.0)
    identity_support = torch.where(
        has_memory,
        identity_support,
        torch.zeros_like(identity_support),
    )
    unmasked_identity_support = identity_support
    if support_mask is not None:
        owner_weight = support_mask.detach().float().clamp(0.0, 1.0)
        if support_floor > 0.0:
            # The clean-source owner decides *where* identity is allowed.
            # Attention similarity only modulates confidence inside it; it
            # must not erase most of an already verified object region.
            identity_support = owner_weight * torch.where(
                has_memory,
                support_floor
                + (1.0 - support_floor) * identity_support,
                torch.zeros_like(identity_support),
            )
        else:
            identity_support = identity_support * owner_weight
    appearance_subspace_coherence = None
    if residual_subspace:
        # A target-minus-source prototype is not pure appearance: it also
        # contains view, pose, boundary, and occlusion information.  The
        # component shared by independently clustered first-chunk
        # prototypes is the only direction we treat as view-invariant.
        # Everything orthogonal to it remains owned by the current chunk.
        prototype_weight = (
            confidence * valid.float()
        )
        normalized_prototype_weight = (
            prototype_weight
            / prototype_weight.sum(dim=-1, keepdim=True).clamp_min(eps)
        )
        mean_residual = torch.einsum(
            "bp,bphd->bhd",
            normalized_prototype_weight,
            prototype_value.float(),
        )
        mean_norm = mean_residual.square().sum(
            dim=-1, keepdim=True
        ).sqrt()
        mean_prototype_norm = torch.einsum(
            "bp,bph->bh",
            normalized_prototype_weight,
            prototype_value.float().square().sum(dim=-1).sqrt(),
        ).unsqueeze(-1)
        coherent_direction = (
            mean_residual / mean_norm.clamp_min(eps)
        )
        coherence = (
            mean_norm / mean_prototype_norm.clamp_min(eps)
        ).clamp(0.0, 1.0)
        coherence = torch.where(
            mean_prototype_norm > eps,
            coherence,
            torch.zeros_like(coherence),
        )

        current_residual = (
            target_value.float() - source_value.float()
        )
        current_coefficient = (
            current_residual
            * coherent_direction[:, None]
        ).sum(dim=-1, keepdim=True)
        target_coefficient = (
            retrieved_value
            * coherent_direction[:, None]
        ).sum(dim=-1, keepdim=True)

        # A one-anchor-magnitude trust region prevents a mismatched
        # correspondence from turning a local color direction into a large
        # low-frequency feature blob.  This is scale-adaptive and introduces
        # no case-specific threshold.
        prototype_coefficient = torch.einsum(
            "bphd,bhd->bph",
            prototype_value.float(),
            coherent_direction,
        )
        coefficient_scale = torch.einsum(
            "bp,bph->bh",
            normalized_prototype_weight,
            prototype_coefficient.square(),
        ).sqrt().unsqueeze(1).unsqueeze(-1)
        coefficient_delta = (
            target_coefficient - current_coefficient
        ).clamp(
            min=-coefficient_scale,
            max=coefficient_scale,
        )
        subspace_coherence = coherence.square()
        subspace_gain = (
            correction_strength
            * identity_support[:, :, None, None]
            * subspace_coherence[:, None]
        )
        corrected_value = (
            target_value.float()
            + subspace_gain
            * coefficient_delta
            * coherent_direction[:, None]
        ).to(target_value.dtype)
        appearance_subspace_coherence = (
            subspace_coherence.mean(dim=(-1, -2))[:, None]
            .expand_as(identity_support)
            .float()
        )
        # Downstream belief updates and diagnostics must see the effective
        # action, not merely the spatial permission supplied by the owner.
        identity_support = (
            identity_support * appearance_subspace_coherence
        )
    elif prototype_value_is_residual:
        current_residual = (
            target_value.float() - source_value.float()
        )
        corrected_residual = (
            current_residual
            + correction_strength
            * identity_support[:, :, None, None]
            * (retrieved_value - current_residual)
        )
        corrected_value = (
            source_value.float() + corrected_residual
        ).to(target_value.dtype)
    else:
        corrected_value = (
            target_value.float()
            + correction_strength
            * identity_support[:, :, None, None]
            * (retrieved_value - target_value.float())
        ).to(target_value.dtype)
    if not return_diagnostics:
        return corrected_value, identity_support.float()
    correction = corrected_value.float() - target_value.float()
    diagnostics = {
        "best_similarity": best_similarity.float(),
        "prototype_assignment_entropy": assignment_entropy.float(),
        "prototype_assignment_peak": assignment_peak.float(),
        "prototype_assignment_margin": assignment_margin.float(),
        "selected_prototype": selected_prototype.float(),
        "absolute_match": absolute_match.float(),
        "relative_match": relative_match.float(),
        "support_before_mask": (
            unmasked_identity_support.float()
        ),
        "support_after_mask": identity_support.float(),
        "support_mask": (
            torch.ones_like(identity_support)
            if support_mask is None
            else support_mask.detach().float().clamp(0.0, 1.0)
        ),
        "correction_ratio": (
            correction.square().mean(dim=(-1, -2)).sqrt()
            / target_value.float().square().mean(
                dim=(-1, -2)
            ).sqrt().clamp_min(eps)
        ),
    }
    if appearance_subspace_coherence is not None:
        diagnostics["appearance_subspace_coherence"] = (
            appearance_subspace_coherence
        )
        diagnostics["appearance_subspace_action"] = (
            identity_support.float()
        )
    return corrected_value, identity_support.float(), diagnostics


def materialize_immutable_target_value(
    *,
    correspondence_key,
    target_value,
    source_value,
    prototype_key,
    prototype_value,
    prototype_evidence,
    owner_weight,
    tokens_per_frame,
    prototype_value_is_residual=True,
    residual_subspace=False,
    support_floor=0.0,
    correction_strength=1.0,
):
    """Materialize a read-only target appearance into a fresh KV write.

    Owner weights are the sole spatial gate.  Tokens outside the owner are
    returned bitwise unchanged; empty owners therefore cannot create ghosts.
    Residual values reproduce the 920 behavior.  ``residual_subspace`` keeps
    the current value outside the cross-prototype coherent edit direction,
    instead of copying a complete first-view value into a new view.
    """
    corrected, support, diagnostics = (
        apply_target_identity_value_correction(
            correspondence_key=correspondence_key,
            target_value=target_value,
            source_value=source_value,
            prototype_key=prototype_key,
            prototype_value=prototype_value,
            prototype_evidence=prototype_evidence,
            prototype_value_is_residual=prototype_value_is_residual,
            residual_subspace=residual_subspace,
            tokens_per_frame=tokens_per_frame,
            support_mask=owner_weight,
            support_floor=support_floor,
            correction_strength=correction_strength,
            return_diagnostics=True,
        )
    )
    owner = owner_weight.detach().float() > 0.0
    materialized = torch.where(
        owner[:, :, None, None],
        corrected,
        target_value,
    )
    return materialized, support, diagnostics


def blend_source_addressed_residual(
    native_output,
    residual,
    support,
    *,
    strength=1.0,
):
    """Add a retrieved edit residual with an exact per-token fallback.

    This small operator is deliberately independent of the retrieval logic:
    zero support must reproduce ``native_output`` bit-for-bit, while positive
    support only changes the corresponding query tokens.
    """
    if native_output.shape != residual.shape:
        raise ValueError(
            "Native output and source-addressed residual must align"
        )
    if support.shape != native_output.shape[:2]:
        raise ValueError(
            "Source-addressed support must have shape [B,L]"
        )
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError(
            "Source-addressed residual strength must lie in [0, 1]"
        )
    gate = support.to(native_output.device).float().clamp(0.0, 1.0)
    correction = (
        float(strength)
        * gate[:, :, None, None]
        * residual.to(native_output.device).float()
    )
    return (native_output.float() + correction).to(native_output.dtype)


def arbitrate_projected_attention_output(
    native_output,
    projected_output,
    query_support,
    *,
    binary_access=False,
):
    """Confine a value projection to the queries that requested it.

    A key/value write mask alone is not a spatial influence mask: every
    self-attention query can read a projected value.  This operator compares
    the attention result with and without the projected values and admits the
    difference only on source-addressed object queries.  ``query_support ==
    0`` deliberately selects the original tensor with ``where`` so hand,
    boundary, background, and abstaining queries retain an exact native
    fallback instead of a numerically reconstructed one.
    """
    if native_output.shape != projected_output.shape:
        raise ValueError(
            "Native and projected attention outputs must align"
        )
    if query_support.shape != native_output.shape[:2]:
        raise ValueError(
            "Projected-attention query support must have shape [B,L]"
        )
    support = (
        query_support.to(native_output.device).float().clamp(0.0, 1.0)
    )
    # The projected V path may already have used the continuous confidence
    # to interpolate toward its canonical source-addressed value.  In that
    # case the query-side operator is access control, not a second confidence
    # estimator: supported queries may observe the counterfactual attention
    # result, while every unsupported query remains bitwise native.  Keeping
    # the legacy soft gate available makes this a measurable ablation.
    gate = (support > 0.0).to(support.dtype) if binary_access else support
    corrected = native_output.float() + gate[:, :, None, None] * (
        projected_output.float() - native_output.float()
    )
    return torch.where(
        gate[:, :, None, None] > 0.0,
        corrected.to(native_output.dtype),
        native_output,
    )


def source_addressed_anchor_attention_delta(
    source_query,
    source_key,
    canonical_residual,
    key_support,
    lineage_id,
):
    """Attend to canonical residuals without cross-part mixing.

    Q/K come from the current source branch and therefore follow the current
    pose, while the values were retrieved through clean-source addresses and
    contain only the immutable edit residual. A lineage mask prevents a
    cap/body (or other object-part) residual from crossing between
    independently transported source addresses.
    """
    if not (
        source_query.shape == source_key.shape == canonical_residual.shape
    ):
        raise ValueError(
            "Anchor query, key, and residual must share shape"
        )
    if source_query.ndim != 4 or source_query.shape[0] != 1:
        raise ValueError(
            "Compact anchor attention expects shape [1,L,H,D]"
        )
    if key_support.shape != source_query.shape[:2]:
        raise ValueError(
            "Anchor key support must have shape [1,L]"
        )
    if lineage_id.shape != source_query.shape[:2]:
        raise ValueError(
            "Anchor lineage ids must have shape [1,L]"
        )
    if source_query.shape[1] == 0:
        return canonical_residual.to(
            device=source_query.device,
            dtype=source_query.dtype,
        )
    lineage = lineage_id.detach().to(source_query.device)
    valid = lineage >= 0
    if not bool(valid.all()):
        raise ValueError(
            "Every compact anchor token must have a valid lineage"
        )
    lineage_mask = (
        lineage[:, :, None] == lineage[:, None, :]
    )[:, None]
    # The immutable memory intentionally keeps its payload in fp32, while
    # Wan commonly runs Q/K in bf16. SDPA requires Q/K/V to have one dtype,
    # so apply confidence in fp32 and cast the resulting anchor value to the
    # current attention compute dtype only at the kernel boundary.
    attention_key = source_key.to(
        device=source_query.device,
        dtype=source_query.dtype,
    )
    value = (
        canonical_residual.float()
        * key_support.to(canonical_residual.device).float()[
            :, :, None, None
        ]
    ).to(device=source_query.device, dtype=source_query.dtype)
    output = torch.nn.functional.scaled_dot_product_attention(
        source_query.transpose(1, 2),
        attention_key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=lineage_mask,
        dropout_p=0.0,
        is_causal=False,
    )
    return output.transpose(1, 2).contiguous()


def immutable_canonical_anchor_attention_delta(
    source_query,
    canonical_source_key,
    canonical_residual,
    canonical_evidence,
    query_lineage_id,
    canonical_lineage_id,
    *,
    query_key_mask=None,
    eps=1e-6,
):
    """Cross-attend current source queries to frozen ignition K/delta-V.

    Query and memory lengths are intentionally independent.  The current
    clean-source Q follows pose, while K and delta-V come only from the
    replay-verified first block. A source-grounded query/key mask is a hard
    routing constraint and canonical evidence is a logit prior, never a
    multiplier on edit magnitude. Callers scatter the result only to admitted
    owner queries.

    Q/K are pre-RoPE features.  This removes the absolute temporal phase
    mismatch between a later query and a first-block key without modifying
    the checkpoint's native RoPE or recent-history attention.
    """
    if source_query.ndim != 4 or canonical_source_key.ndim != 4:
        raise ValueError(
            "Canonical anchor Q/K must have shape [B,L,H,D]"
        )
    if canonical_residual.shape != canonical_source_key.shape:
        raise ValueError(
            "Canonical residual must align with canonical source keys"
        )
    if source_query.shape[0] != canonical_source_key.shape[0] or (
        source_query.shape[2:] != canonical_source_key.shape[2:]
    ):
        raise ValueError(
            "Canonical anchor Q/K must share batch and head dimensions"
        )
    if canonical_evidence.shape != canonical_source_key.shape[:2]:
        raise ValueError(
            "Canonical evidence must have shape [B,M]"
        )
    if query_lineage_id.shape != source_query.shape[:2]:
        raise ValueError(
            "Query lineage ids must have shape [B,Q]"
        )
    if canonical_lineage_id.shape != canonical_source_key.shape[:2]:
        raise ValueError(
            "Canonical lineage ids must have shape [B,M]"
        )
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if source_query.shape[1] == 0:
        return source_query.new_empty(source_query.shape)

    device = source_query.device
    query_lineage = query_lineage_id.detach().to(device)
    key_lineage = canonical_lineage_id.detach().to(device)
    evidence = canonical_evidence.detach().to(device).float().clamp(0.0, 1.0)
    exact_lineage = (
        (query_lineage[:, :, None] >= 0)
        & (query_lineage[:, :, None] == key_lineage[:, None, :])
        & (key_lineage[:, None, :] >= 0)
    )
    if query_key_mask is None:
        eligible = exact_lineage
    else:
        if query_key_mask.shape != (
            source_query.shape[0],
            source_query.shape[1],
            canonical_source_key.shape[1],
        ):
            raise ValueError(
                "Canonical query-key mask must have shape [B,Q,M]"
            )
        eligible = query_key_mask.detach().to(device).bool()
    eligible &= evidence[:, None, :] > 0.0
    if not bool(eligible.any(dim=-1).all()):
        raise ValueError(
            "Every compact canonical query needs a verified lineage key"
        )

    # Add log p(memory slot) to the QK logits.  Unlike 935, confidence does
    # not shrink delta-V and cannot make a valid canonical edit fade merely
    # because the object changed pose.
    log_prior = evidence.clamp_min(float(eps)).log()[:, None, None, :]
    attention_bias = torch.where(
        eligible[:, None],
        log_prior.expand(-1, 1, source_query.shape[1], -1),
        torch.full_like(
            log_prior.expand(-1, 1, source_query.shape[1], -1),
            float("-inf"),
        ),
    ).to(dtype=source_query.dtype)
    key = canonical_source_key.to(device=device, dtype=source_query.dtype)
    value = canonical_residual.to(device=device, dtype=source_query.dtype)
    output = torch.nn.functional.scaled_dot_product_attention(
        source_query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=attention_bias,
        dropout_p=0.0,
        is_causal=False,
    )
    return output.transpose(1, 2).contiguous()


def scatter_source_addressed_anchor_delta(
    native_output,
    anchor_delta,
    query_support,
    *,
    strength=1.0,
):
    """Add a compact canonical-attention delta on supported queries.

    The native branch remains responsible for motion and unrestricted recent
    target history. ``anchor_delta`` is computed from current clean-source
    Q/K and canonical residual values, so it does not compete against a
    growing history softmax. Continuous retrieval confidence is consumed
    while the anchor values are built; ``query_support`` is therefore binary
    access control here.

    Unsupported queries are copied from ``native_output`` without arithmetic,
    which makes hand/background/unknown fallback exact.
    """
    if native_output.ndim != 4:
        raise ValueError(
            "Native attention output must have shape [B,L,H,D]"
        )
    if query_support.shape != native_output.shape[:2]:
        raise ValueError(
            "Anchor query support must have shape [B,L]"
        )
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError(
            "Source-addressed anchor strength must lie in [0, 1]"
        )
    selected = query_support.detach().to(native_output.device) > 0.0
    selected_count = int(selected.sum().item())
    if anchor_delta.shape != (
        1, selected_count, *native_output.shape[2:]
    ):
        raise ValueError(
            "Compact anchor outputs must align with supported queries"
        )
    if native_output.shape[0] != 1:
        raise ValueError(
            "Compact anchor scatter expects one sample at a time"
        )
    if selected_count == 0:
        return native_output
    correction = float(strength) * anchor_delta.float()
    output = native_output.clone()
    output[:, selected[0]] = (
        native_output[:, selected[0]].float() + correction
    ).to(native_output.dtype)
    return output


def project_source_addressed_target_value(
    target_value,
    source_value,
    residual,
    support,
    *,
    strength=1.0,
):
    """Project target values toward a source-addressed edit anchor.

    The paired memory stores ``target_value - source_value`` at clean
    context.  Re-materializing it on the current source value gives a
    geometry-following target value instead of an additive perturbation to
    the already mixed attention output.  ``support == 0`` is an exact native
    fallback, which keeps hand, background, and uncertain tokens unchanged.
    """
    if not (
        target_value.shape == source_value.shape == residual.shape
    ):
        raise ValueError(
            "Target, source, and source-addressed residual values must "
            "align"
        )
    if support.shape != target_value.shape[:2]:
        raise ValueError(
            "Source-addressed support must have shape [B,L]"
        )
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError(
            "Source-addressed projection strength must lie in [0, 1]"
        )
    gate = (
        float(strength)
        * support.to(target_value.device).float().clamp(0.0, 1.0)
    )[:, :, None, None]
    desired_value = (
        source_value.float()
        + residual.to(target_value.device).float()
    )
    projected = target_value.float() + gate * (
        desired_value - target_value.float()
    )
    return torch.where(
        gate > 0.0,
        projected.to(target_value.dtype),
        target_value,
    )
