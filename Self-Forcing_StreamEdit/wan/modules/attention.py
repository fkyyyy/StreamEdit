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
    'blend_current_target_state',
    'apply_target_identity_value_correction',
]


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


def blend_current_target_state(
    target,
    source,
    blend_rate,
    edit_support=None,
):
    """Blend current states while optionally keeping edit tokens target-pure."""
    if target.shape != source.shape:
        raise ValueError(
            "Current target and source states must share shape"
        )
    if not 0.0 <= blend_rate <= 1.0:
        raise ValueError("blend_rate must lie in [0, 1]")
    if edit_support is None:
        return (
            target * blend_rate
            + source * (1.0 - blend_rate)
        )

    rate = torch.as_tensor(
        blend_rate,
        device=target.device,
        dtype=target.dtype,
    )
    if (
        edit_support.ndim != 1
        or edit_support.shape[0] != target.shape[0]
    ):
        raise ValueError(
            "Edit support must align with the token dimension"
        )
    token_rate = torch.where(
        edit_support.to(device=target.device, dtype=torch.bool),
        torch.ones((), device=target.device, dtype=target.dtype),
        rate,
    )
    rate = token_rate.reshape(
        token_rate.shape[0],
        *([1] * (target.ndim - 1)),
    )
    return target * rate + source * (1.0 - rate)


def apply_target_identity_value_correction(
    target_key,
    target_value,
    prototype_key,
    prototype_value,
    prototype_evidence,
    eps=1e-6,
):
    """Retrieve slow appearance prototypes while retaining current keys."""
    if target_key.shape != target_value.shape:
        raise ValueError(
            "Current target keys and values must share shape"
        )
    if prototype_key.shape != prototype_value.shape:
        raise ValueError(
            "Identity prototype keys and values must share shape"
        )
    if target_key.ndim != 4 or prototype_key.ndim != 4:
        raise ValueError(
            "Identity correction expects [B,L,H,D] tensors"
        )
    if target_key.shape[0] != prototype_key.shape[0]:
        raise ValueError(
            "Current target and identity memory must share batch size"
        )
    if target_key.shape[2:] != prototype_key.shape[2:]:
        raise ValueError(
            "Current target and identity memory must share heads"
        )
    if prototype_evidence.shape != prototype_key.shape[:2]:
        raise ValueError(
            "Identity evidence must have shape [B,P]"
        )
    if eps <= 0:
        raise ValueError("eps must be positive")

    key = torch.nn.functional.normalize(
        target_key.float(),
        dim=-1,
    )
    memory_key = torch.nn.functional.normalize(
        prototype_key.float(),
        dim=-1,
    )
    similarity = torch.einsum(
        "blhd,bphd->blph",
        key,
        memory_key,
    ).mean(dim=-1).clamp(-1.0, 1.0)
    valid = prototype_evidence > eps
    confidence = prototype_evidence.float().clamp(0.0, 1.0)

    flat_similarity = similarity.flatten(1)
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
    logits = similarity / robust_scale.unsqueeze(-1)
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

    retrieved_value = torch.einsum(
        "blp,bphd->blhd",
        assignment,
        prototype_value.float(),
    )
    best_similarity = similarity.masked_fill(
        ~valid[:, None, :],
        -1.0,
    ).max(dim=-1).values
    support_threshold = torch.quantile(
        best_similarity,
        0.90,
        dim=-1,
        keepdim=True,
    )
    high_match = torch.quantile(
        best_similarity,
        0.99,
        dim=-1,
        keepdim=True,
    )
    match_spread = high_match - support_threshold
    relative_match = (
        (best_similarity - support_threshold)
        / match_spread.clamp_min(eps)
    ).clamp(0.0, 1.0)
    relative_match = torch.where(
        match_spread > eps,
        relative_match,
        torch.zeros_like(relative_match),
    )
    absolute_match = (
        0.5 * (best_similarity + 1.0)
    ).clamp(0.0, 1.0)
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
    corrected_value = (
        target_value.float()
        + identity_support[:, :, None, None]
        * (retrieved_value - target_value.float())
    ).to(target_value.dtype)
    return corrected_value, identity_support.float()
