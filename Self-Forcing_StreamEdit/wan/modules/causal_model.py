from wan.modules.attention import (
    apply_target_identity_value_correction,
    arbitrate_verified_factorized_attention,
    arbitrate_projected_attention_output,
    attention,
    blend_target_owned_tensor,
    blend_factorized_with_native_fallback,
    blend_source_addressed_residual,
    closed_loop_counterfactual_memory_attention,
    build_factorized_history_read_mask,
    build_target_owned_source_background_mask,
    fuse_aligned_memory,
    fuse_factorized_aligned_memory,
    project_source_addressed_target_value,
    resolve_target_identity_correction_strength,
    immutable_canonical_anchor_attention_delta,
    source_addressed_anchor_attention_delta,
    source_addressed_native_history_attention,
    scatter_source_addressed_anchor_delta,
    scatter_target_owned_output,
    suppress_source_preserve_on_target_owned_history,
)
from wan.modules.contact_graph_attention import (
    apply_contact_graph_residual,
)
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch.distributed as dist

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        assert seq_len == x.size(1), 'seq_len=%d, x.size(1)=%d' % (seq_len, x.size(1))

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def causal_rope_apply_indexed(
    x, token_indices, grid_sizes, freqs, start_frame=0
):
    """Apply the native 3-D RoPE to a compact set of visual tokens.

    ``token_indices`` refer to positions in the original dense ``F*H*W``
    block.  Keeping them lets 937 retain only role-approved native K/V while
    preserving exactly the spatial and within-block temporal coordinates at
    which those keys were produced.
    """
    if x.ndim != 4 or token_indices.shape != x.shape[:2]:
        raise ValueError(
            "Indexed RoPE expects x=[B,L,H,D] and indices=[B,L]"
        )
    if x.shape[1] == 0:
        return x
    num_heads, complex_dim = x.size(2), x.size(3) // 2
    split_freqs = freqs.split(
        [
            complex_dim - 2 * (complex_dim // 3),
            complex_dim // 3,
            complex_dim // 3,
        ],
        dim=1,
    )
    output = []
    for batch_index, (_, height, width) in enumerate(
        grid_sizes.tolist()
    ):
        index = token_indices[batch_index].long()
        valid = index >= 0
        safe_index = index.clamp_min(0)
        spatial_size = height * width
        temporal_index = safe_index // spatial_size + int(start_frame)
        spatial_index = safe_index % spatial_size
        row_index = spatial_index // width
        col_index = spatial_index % width
        if temporal_index.max().item() >= split_freqs[0].shape[0]:
            raise ValueError("Indexed temporal RoPE exceeds frequency table")
        multiplier = torch.cat(
            [
                split_freqs[0][temporal_index],
                split_freqs[1][row_index],
                split_freqs[2][col_index],
            ],
            dim=-1,
        ).unsqueeze(1)
        value = torch.view_as_complex(
            x[batch_index].to(torch.float64).reshape(
                x.shape[1], num_heads, -1, 2
            )
        )
        value = torch.view_as_real(value * multiplier).flatten(2)
        value = torch.where(
            valid[:, None, None], value, torch.zeros_like(value)
        )
        output.append(value)
    return torch.stack(output).type_as(x)


def causal_rope_apply_multi_chunk(x, grid_sizes, freqs, query, start_frame=0):
    """
    Args:
        x: Tensor of shape [b, m*f, n, c]
        grid_sizes: Tensor of shape [b, 3], each row is (f, h, w)
        freqs: Rotary embeddings, split into 3 parts: [frame_freqs, h_freqs, w_freqs]
        start_frame: Starting index for frame-wise rotary frequencies

    Returns:
        Tensor of shape [b, m*f, n, c] with rotary embedding applied
    """
    n, c = x.size(2), x.size(3) // 2

    # Split freqs into 3 parts
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    last_chunk_start_frame = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        total_frames = x.size(1) // (h * w)  # m*f
        seq_len = total_frames * h * w       # total positions to apply rope
        q_frame = query.size(1) // (h * w)

        assert seq_len == x.size(1), 'seq_len=%d, x.size(1)=%d' % (seq_len, x.size(1))
        last_chunk_start_frame.append(total_frames - q_frame)

        # View input as complex
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))

        # Prepare frequency components
        frame_freq = freqs[0][start_frame : start_frame + total_frames]              # [m*f, c1]
        frame_freq = frame_freq.view(total_frames, 1, 1, -1).expand(total_frames, h, w, -1)

        h_freq = freqs[1][:h].view(1, h, 1, -1).expand(total_frames, h, w, -1)
        w_freq = freqs[2][:w].view(1, 1, w, -1).expand(total_frames, h, w, -1)

        freqs_i = torch.cat([frame_freq, h_freq, w_freq], dim=-1).reshape(seq_len, 1, -1)

        # Apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)

        # Append untouched part (e.g., padding)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)

    return torch.stack(output).type_as(x), last_chunk_start_frame


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        if (
            isinstance(kv_cache, dict)
            and kv_cache.pop("capture_current_query", False)
        ):
            kv_cache["current_query_feature"] = F.normalize(
                q.detach().float().mean(dim=2),
                dim=-1,
            ).to(dtype=v.dtype)
        if (
            isinstance(kv_cache, dict)
            and kv_cache.pop("capture_current_identity_key", False)
        ):
            kv_cache["current_identity_key"] = k.detach()

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

        elif kv_cache == 'local_self_attn':
            # for inference, only focus on the current chunk
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=0).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=0).type_as(v)

            x = attention(roped_query, roped_key, v)

        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_end = current_start + q.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = q.shape[1]
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
            
            local_start_index = local_end_index - num_new_tokens
            attn_seq_slice = slice(max(0, local_end_index - self.max_attention_size), local_end_index)

            if sink_tokens == 0:
                kv_cache["k"][:, local_start_index:local_end_index] = k
                kv_cache["v"][:, local_start_index:local_end_index] = v

                # apply rope for all cached k (except sink tokens)
                select_key = kv_cache["k"][:, attn_seq_slice]
                # Sink-free native caches keep pre-RoPE keys.  Preserve a
                # view for role-conditioned fixed-relative native history.
                unrotated_attn_key = select_key
                roped_key, last_chunk_start_frame = causal_rope_apply_multi_chunk(
                    select_key[:, sink_tokens: ], grid_sizes, freqs, q, start_frame=0
                )
                attn_key = torch.cat([select_key[:, : sink_tokens], roped_key.type_as(v)], dim=1)
                # relatively apply rope for query (batched)
                roped_query = torch.cat([
                    causal_rope_apply(
                        q[b_idx].unsqueeze(0), grid_sizes[b_idx].unsqueeze(0), freqs, 
                        start_frame=last_chunk_start_frame[b_idx]
                    ).type_as(v) for b_idx in range(len(last_chunk_start_frame))
                ], dim=0)
                # select value for attn
                attn_value = kv_cache["v"][:, attn_seq_slice]
            else:
                unrotated_attn_key = None
                current_start_frame = current_start // frame_seqlen
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                roped_key = causal_rope_apply(
                    k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
                attn_key = kv_cache["k"][:, attn_seq_slice]
                attn_value = kv_cache["v"][:, attn_seq_slice]

            # ====================================================================================================
            # generation or editing
            # ====================================================================================================
            
            if kv_cache.get("trg_fg_mask", None) is None or kv_cache.get("current_src_fg_mask", None) is None:
                x = attention(roped_query, attn_key, attn_value)
            else:
                # init
                src_query, trg_query = roped_query.chunk(2, dim=0)
                raw_src_query, raw_trg_query = q.chunk(2, dim=0)
                src_key, trg_key = attn_key.chunk(2, dim=0)
                src_value, trg_value = attn_value.chunk(2, dim=0)
                blender_rate = 1 - kv_cache['shared_dict']['current_timestep_next'] ** kv_cache['shared_dict']['blend_power']
                
                src_prev_key = src_key[:, : -num_new_tokens]
                src_prev_value = src_value[:, : -num_new_tokens]
                src_current_key = src_key[:, -num_new_tokens: ]
                src_current_value = src_value[:, -num_new_tokens: ]
                trg_prev_key = trg_key[:, : -num_new_tokens]
                trg_prev_value = trg_value[:, : -num_new_tokens]
                trg_current_key = trg_key[:, -num_new_tokens: ]
                trg_current_value = trg_value[:, -num_new_tokens: ]

                shared_dict = kv_cache["shared_dict"]
                target_owned_handoff = bool(
                    shared_dict.get(
                        "target_owned_object_handoff",
                        False,
                    )
                )
                current_target_owned_mask = None
                target_owned_history_mask = None
                if target_owned_handoff:
                    current_target_owned_mask = kv_cache.get(
                        "current_target_owned_mask"
                    )
                    target_owned_cache = kv_cache.get(
                        "target_owned_history_mask"
                    )
                    if (
                        current_target_owned_mask is None
                        or target_owned_cache is None
                    ):
                        raise RuntimeError(
                            "Target-owned handoff requires aligned current "
                            "and historical ownership masks"
                        )
                    if current_target_owned_mask.shape != (
                        b // 2,
                        num_new_tokens,
                    ):
                        raise ValueError(
                            "Current target-owned mask must align with "
                            "the current attention tokens"
                        )
                    target_owned_history_mask = (
                        target_owned_cache[:, attn_seq_slice][
                            :, :-num_new_tokens
                        ]
                    )
                    if target_owned_history_mask.shape != (
                        b // 2,
                        src_prev_key.shape[1],
                    ):
                        raise ValueError(
                            "Target-owned history must align with cached "
                            "attention tokens"
                        )
                identity_states = shared_dict.get(
                    "target_identity_memory",
                    {},
                )
                factorized_target_identity = bool(
                    shared_dict.get(
                        "factorized_target_identity",
                        False,
                    )
                )
                core_conditioned_identity = bool(
                    shared_dict.get(
                        "appearance_leakage_decomposition",
                        False,
                    )
                )
                immutable_factorized_identity = bool(
                    shared_dict.get(
                        "factorized_immutable_target_memory",
                        False,
                    )
                )
                immutable_residual_subspace = bool(
                    shared_dict.get(
                        "immutable_target_residual_subspace",
                        False,
                    )
                )
                layer_index = kv_cache.get("layer_index", -1)
                timestep_counterfactual_memory = bool(
                    shared_dict.get(
                        "native_history_timestep_counterfactual_memory",
                        False,
                    )
                )
                counterfactual_history = shared_dict.get(
                    "role_native_history_object"
                )
                timestep_index = int(
                    shared_dict.get("current_timestep_index", -1)
                )
                if (
                    timestep_counterfactual_memory
                    and counterfactual_history is not None
                    and layer_index in counterfactual_history.layers
                ):
                    timestep_selection_weight = kv_cache.get(
                        "current_causal_owner_mask"
                    )
                    if timestep_selection_weight is None:
                        timestep_selection_weight = kv_cache.get(
                            "current_target_memory_action"
                        )
                    if timestep_selection_weight is None:
                        raise RuntimeError(
                            "Timestep counterfactual capture requires an "
                            "automatic owner weight"
                        )
                    counterfactual_history.stage_timestep_counterfactual(
                        layer=layer_index,
                        timestep_index=timestep_index,
                        source_key=k.chunk(2, dim=0)[0],
                        source_value=v.chunk(2, dim=0)[0],
                        target_key=k.chunk(2, dim=0)[1],
                        target_value=v.chunk(2, dim=0)[1],
                        selection_weight=timestep_selection_weight,
                    )
                identity_state = identity_states.get(layer_index)
                if identity_state is not None:
                    raw_source_key, raw_target_key = k.chunk(2, dim=0)
                    source_identity_keys = shared_dict.get(
                        "source_identity_keys",
                        {},
                    )
                    correspondence_key = source_identity_keys.get(
                        layer_index,
                        raw_source_key,
                    ).to(
                        device=raw_source_key.device,
                        dtype=raw_source_key.dtype,
                    )
                    identity_read_mask = kv_cache.get(
                        "current_identity_read_mask"
                    )
                    support_mask = (
                        identity_read_mask
                        if (
                            factorized_target_identity
                            or core_conditioned_identity
                            or immutable_factorized_identity
                        )
                        else kv_cache["current_src_fg_mask"]
                    )
                    if (
                        (
                            factorized_target_identity
                            or core_conditioned_identity
                            or immutable_factorized_identity
                        )
                        and identity_read_mask is None
                    ):
                        raise RuntimeError(
                            "Core-conditioned identity requires an explicit "
                            "current read mask"
                        )
                    prototype_appearance_key = getattr(
                        identity_state,
                        "appearance_key",
                        None,
                    )
                    if (
                        factorized_target_identity
                        and not core_conditioned_identity
                        and prototype_appearance_key is None
                    ):
                        raise RuntimeError(
                            "Factorized identity memory is missing its "
                            "target appearance key"
                        )
                    (
                        trg_current_value,
                        identity_support,
                        identity_diagnostics,
                    ) = apply_target_identity_value_correction(
                        correspondence_key=correspondence_key,
                        target_value=trg_current_value,
                        source_value=src_current_value,
                        prototype_key=identity_state.key.to(
                            device=correspondence_key.device,
                            dtype=correspondence_key.dtype,
                        ),
                        prototype_value=identity_state.value.to(
                            device=trg_current_value.device,
                            dtype=trg_current_value.dtype,
                        ),
                        prototype_evidence=identity_state.evidence.to(
                            device=correspondence_key.device,
                        ),
                        prototype_value_is_residual=bool(
                            getattr(
                                identity_state,
                                "value_is_residual",
                                False,
                            )
                        ),
                        residual_subspace=immutable_residual_subspace,
                        target_appearance_key=(
                            raw_target_key
                            if (
                                factorized_target_identity
                                and not core_conditioned_identity
                            )
                            else None
                        ),
                        prototype_appearance_key=(
                            None
                            if (
                                not factorized_target_identity
                                or core_conditioned_identity
                                or prototype_appearance_key is None
                            )
                            else prototype_appearance_key.to(
                                device=trg_current_key.device,
                                dtype=trg_current_key.dtype,
                            )
                        ),
                        tokens_per_frame=frame_seqlen,
                        support_mask=support_mask,
                        support_floor=float(
                            shared_dict.get(
                                "identity_support_floor", 0.0
                            )
                        ),
                        correction_strength=(
                            resolve_target_identity_correction_strength(
                                shared_dict.get(
                                    "identity_correction_strength",
                                    1.0,
                                ),
                                factorized_target_identity=(
                                    factorized_target_identity
                                ),
                                immutable_factorized_identity=(
                                    immutable_factorized_identity
                                ),
                                prototype_value_is_residual=bool(
                                    getattr(
                                        identity_state,
                                        "value_is_residual",
                                        False,
                                    )
                                ),
                            )
                        ),
                        return_diagnostics=True,
                    )
                    shared_dict.setdefault(
                        "target_identity_support",
                        {},
                    )[layer_index] = identity_support.detach()
                    shared_dict.setdefault(
                        "target_identity_diagnostics",
                        {},
                    )[layer_index] = {
                        name: value.detach()
                        for name, value in identity_diagnostics.items()
                    }

                x_list = [
                    attention(src_query, src_key, src_value)   # source
                ]
                for b_idx in range(b // 2):
                    factorized_bayes_kv = all(
                        kv_cache.get(name) is not None
                        for name in (
                            "cached_source_key_action",
                            "cached_source_value_action",
                            "cached_target_memory_action",
                            "cached_unknown_action",
                            "current_source_key_action",
                            "current_source_value_action",
                            "current_target_memory_action",
                            "current_unknown_action",
                        )
                    )
                    if factorized_bayes_kv:
                        history_actions = {
                            name: kv_cache[f"cached_{name}"][
                                b_idx, attn_seq_slice
                            ][:-num_new_tokens]
                            for name in (
                                "source_key_action",
                                "source_value_action",
                                "target_memory_action",
                                "unknown_action",
                            )
                        }
                        current_actions = {
                            name: kv_cache[f"current_{name}"][b_idx]
                            for name in (
                                "source_key_action",
                                "source_value_action",
                                "target_memory_action",
                                "unknown_action",
                            )
                        }
                        if any(
                            action.shape != (src_prev_key.shape[1],)
                            for action in history_actions.values()
                        ):
                            raise ValueError(
                                "Cached factorized actions must align with "
                                "historical memory"
                            )
                        if any(
                            action.shape != (num_new_tokens,)
                            for action in current_actions.values()
                        ):
                            raise ValueError(
                                "Current factorized actions must align with "
                                "the current attention block"
                            )
                        paired_reads = kv_cache["shared_dict"].get(
                            "causal_paired_edit_memory", {}
                        )
                        paired_read = paired_reads.get(layer_index)
                        if paired_read is not None and (
                            paired_read.residual.shape
                            != (
                                b // 2,
                                num_new_tokens,
                                self.num_heads,
                                self.head_dim,
                            )
                            or paired_read.support.shape
                            != (b // 2, num_new_tokens)
                        ):
                            raise ValueError(
                                "Source-addressed residual read must "
                                "align with current target tokens"
                            )
                        paired_value_projection = bool(
                            kv_cache["shared_dict"].get(
                                "paired_memory_value_projection", False
                            )
                        )
                        query_gated_projection = bool(
                            kv_cache["shared_dict"].get(
                                "paired_memory_query_gated_projection",
                                False,
                            )
                        )
                        single_confidence = bool(
                            kv_cache["shared_dict"].get(
                                "paired_memory_single_confidence",
                                False,
                            )
                        )
                        dual_timescale_anchor = bool(
                            kv_cache["shared_dict"].get(
                                "paired_memory_dual_timescale_anchor",
                                False,
                            )
                        )
                        canonical_key_anchor = bool(
                            kv_cache["shared_dict"].get(
                                "paired_memory_canonical_key_anchor",
                                False,
                            )
                        )
                        role_fixed_native_history = bool(
                            kv_cache["shared_dict"].get(
                                "role_fixed_native_history", False
                            )
                        )
                        native_target_value = trg_current_value[
                            b_idx
                        ].unsqueeze(0)
                        attention_target_value = native_target_value
                        value_projection_delta = (
                            attention_target_value.new_zeros(())
                        )
                        if (
                            paired_value_projection
                            and paired_read is not None
                            and not dual_timescale_anchor
                        ):
                            if (
                                paired_read.source_value is None
                                or paired_read.source_value.shape
                                != paired_read.residual.shape
                            ):
                                raise ValueError(
                                    "Value projection requires aligned "
                                    "clean-source values"
                                )
                            projected_target_value = (
                                project_source_addressed_target_value(
                                    attention_target_value,
                                    paired_read.source_value[b_idx]
                                    .to(attention_target_value.device)
                                    .unsqueeze(0),
                                    paired_read.residual[b_idx]
                                    .to(attention_target_value.device)
                                    .unsqueeze(0),
                                    paired_read.support[b_idx]
                                    .to(attention_target_value.device)
                                    .unsqueeze(0),
                                    strength=float(
                                        kv_cache["shared_dict"].get(
                                            "paired_memory_read_strength",
                                            0.0,
                                        )
                                    ),
                                )
                            )
                            value_projection_delta = (
                                projected_target_value.float()
                                - attention_target_value.float()
                            ).abs().mean()
                            attention_target_value = (
                                projected_target_value
                            )

                        # Source Q/K supplies correspondence according to the
                        # native timestep schedule. Unknown history exactly
                        # recovers StreamGVE's foreground-key fallback.
                        history_native_key = (
                            kv_cache["trg_fg_mask"][
                                b_idx, attn_seq_slice
                            ][:-num_new_tokens].float()
                            * (1.0 - blender_rate)
                        )
                        history_source_key = (
                            history_actions["source_key_action"]
                            * (1.0 - blender_rate)
                            + history_actions["unknown_action"]
                            * history_native_key
                        ).clamp(0.0, 1.0)
                        history_known = (
                            history_actions["source_value_action"]
                            + history_actions["target_memory_action"]
                        ).clamp(0.0, 1.0)
                        history_read_mask = (
                            build_factorized_history_read_mask(
                                history_actions[
                                    "source_value_action"
                                ],
                                history_actions[
                                    "target_memory_action"
                                ],
                            )
                        )
                        normalized_history_source_value = torch.where(
                            history_read_mask,
                            history_actions["source_value_action"]
                            / history_known.clamp_min(1e-6),
                            torch.zeros_like(history_known),
                        )
                        normalized_history_target_value = torch.where(
                            history_read_mask,
                            history_actions["target_memory_action"]
                            / history_known.clamp_min(1e-6),
                            torch.zeros_like(history_known),
                        )
                        previous_key, previous_value = (
                            fuse_factorized_aligned_memory(
                                trg_prev_key[b_idx].unsqueeze(0),
                                trg_prev_value[b_idx].unsqueeze(0),
                                src_prev_key[b_idx].unsqueeze(0),
                                src_prev_value[b_idx].unsqueeze(0),
                                history_source_key.unsqueeze(0),
                                normalized_history_source_value.unsqueeze(0),
                                normalized_history_target_value.unsqueeze(0),
                                torch.zeros_like(
                                    history_known
                                ).unsqueeze(0),
                            )
                        )
                        previous_key = previous_key[:, history_read_mask]
                        previous_value = previous_value[:, history_read_mask]

                        current_source_key = (
                            (
                                current_actions["source_key_action"]
                                + current_actions["unknown_action"]
                            )
                            * (1.0 - blender_rate)
                        ).clamp(0.0, 1.0)
                        current_key, current_value = (
                            fuse_factorized_aligned_memory(
                                trg_current_key[b_idx].unsqueeze(0),
                                attention_target_value,
                                src_current_key[b_idx].unsqueeze(0),
                                src_current_value[b_idx].unsqueeze(0),
                                current_source_key.unsqueeze(0),
                                current_actions[
                                    "source_value_action"
                                ].unsqueeze(0),
                                current_actions[
                                    "target_memory_action"
                                ].unsqueeze(0),
                                current_actions[
                                    "unknown_action"
                                ].unsqueeze(0),
                            )
                        )
                        query_source_weight = current_source_key[
                            :, None, None
                        ]
                        factorized_query = (
                            trg_query[b_idx].float()
                            + query_source_weight
                            * (
                                src_query[b_idx].float()
                                - trg_query[b_idx].float()
                            )
                        ).to(trg_query.dtype)
                        factorized_output = attention(
                            factorized_query.unsqueeze(0),
                            torch.cat(
                                [
                                    previous_key.squeeze(0),
                                    current_key.squeeze(0),
                                ],
                                dim=0,
                            ).unsqueeze(0),
                            torch.cat(
                                [
                                    previous_value.squeeze(0),
                                    current_value.squeeze(0),
                                ],
                                dim=0,
                            ).unsqueeze(0),
                        )

                        # Unknown is a true abstention: reproduce native
                        # StreamGVE, including its late source-background
                        # context, and select that expert per unknown query.
                        native_prev_key = trg_prev_key[b_idx].clone()
                        native_prev_mask = kv_cache[
                            "trg_fg_mask"
                        ][b_idx, attn_seq_slice][:-num_new_tokens]
                        native_prev_key[native_prev_mask] = (
                            native_prev_key[native_prev_mask]
                            * blender_rate
                            + src_prev_key[b_idx, native_prev_mask]
                            * (1.0 - blender_rate)
                        )
                        native_key_list = [native_prev_key]
                        native_value_list = [trg_prev_value[b_idx]]
                        if kv_cache["shared_dict"][
                            "current_timestep_index"
                        ] > kv_cache["shared_dict"][
                            "total_timestep"
                        ] // 2:
                            native_background = ~kv_cache[
                                "current_src_fg_mask"
                            ][b_idx]
                            native_key_list.append(
                                src_current_key[b_idx][native_background]
                            )
                            native_value_list.append(
                                src_current_value[b_idx][native_background]
                            )
                        native_key_list.append(
                            trg_current_key[b_idx] * blender_rate
                            + src_current_key[b_idx]
                            * (1.0 - blender_rate)
                        )
                        # In query-gated mode this first pass must remain
                        # genuinely native.  A masked V write is still
                        # globally readable by self-attention, so using the
                        # projected value here would already contaminate the
                        # fallback before output arbitration.
                        native_value_list.append(
                            (
                                native_target_value
                                if query_gated_projection
                                else attention_target_value
                            ).squeeze(0)
                        )
                        native_query = (
                            trg_query[b_idx] * blender_rate
                            + src_query[b_idx] * (1.0 - blender_rate)
                        )
                        native_output = attention(
                            native_query.unsqueeze(0),
                            torch.cat(native_key_list, dim=0).unsqueeze(0),
                            torch.cat(native_value_list, dim=0).unsqueeze(0),
                        )
                        source_mixed_native_output = native_output
                        projected_native_output = native_output
                        ungated_projection_leakage = (
                            native_output.new_zeros(())
                        )
                        if (
                            query_gated_projection
                            and paired_value_projection
                            and paired_read is not None
                        ):
                            projected_value_list = [
                                *native_value_list[:-1],
                                attention_target_value.squeeze(0),
                            ]
                            projected_native_output = attention(
                                native_query.unsqueeze(0),
                                torch.cat(
                                    native_key_list, dim=0
                                ).unsqueeze(0),
                                torch.cat(
                                    projected_value_list, dim=0
                                ).unsqueeze(0),
                            )
                            unsupported_query = (
                                paired_read.support[b_idx]
                                .to(native_output.device)
                                .float()
                                <= 0.0
                            )
                            ungated_delta = (
                                projected_native_output.float()
                                - native_output.float()
                            ).abs().mean(dim=(-1, -2)).squeeze(0)
                            unsupported_count = (
                                unsupported_query.float().sum()
                                .clamp_min(1.0)
                            )
                            ungated_projection_leakage = (
                                ungated_delta
                                * unsupported_query.float()
                            ).sum() / unsupported_count
                        if bool(
                            kv_cache["shared_dict"].get(
                                "factorized_native_target_history",
                                False,
                            )
                        ):
                            # 923: the backbone continuation remains the
                            # authority.  Factorized Bayes still controls the
                            # velocity/source-appearance policy, but it must
                            # not prune or reconstruct the clean edited-target
                            # history consumed by self-attention.  Keep the
                            # old factorized result only as a counterfactual
                            # diagnostic so this hypothesis is measurable.
                            owner_weight = kv_cache.get(
                                "current_causal_owner_mask"
                            )
                            object_weight = (
                                owner_weight[b_idx].float()
                                if owner_weight is not None
                                else current_actions[
                                    "target_memory_action"
                                ].float()
                            )
                            token_gap = (
                                native_output.float()
                                - factorized_output.float()
                            ).abs().mean(dim=(-1, -2)).squeeze(0)
                            object_output_gap = (
                                token_gap * object_weight
                            ).sum() / object_weight.sum().clamp_min(1e-6)
                            history_read_ratio = (
                                history_read_mask.float().mean()
                                if history_read_mask.numel() > 0
                                else object_output_gap.new_tensor(1.0)
                            )
                            kv_cache["shared_dict"].setdefault(
                                "native_target_history_diagnostics",
                                {},
                            )[layer_index] = {
                                "factorized_history_read_ratio": (
                                    history_read_ratio.detach()
                                ),
                                "owner_output_gap": (
                                    object_output_gap.detach()
                                ),
                                "owner_coverage": (
                                    object_weight.mean().detach()
                                ),
                            }
                            native_history_reads = kv_cache[
                                "shared_dict"
                            ].get("role_native_history_reads", {})
                            native_history_read = native_history_reads.get(
                                layer_index
                            )
                            native_history_bypass = bool(
                                kv_cache["shared_dict"].get(
                                    "role_native_history_bypass", False
                                )
                            )
                            if (
                                role_fixed_native_history
                                and native_history_read is not None
                                and not native_history_bypass
                            ):
                                if sink_tokens != 0 or unrotated_attn_key is None:
                                    raise RuntimeError(
                                        "Role-fixed native history requires "
                                        "sink-free pre-RoPE caches"
                                    )
                                canonical = native_history_read.canonical
                                recent = native_history_read.recent
                                source_lineage = (
                                    native_history_read.source_lineage
                                )
                                flow_residual = (
                                    native_history_read.flow_residual
                                )
                                canonical_correspondence = (
                                    native_history_read
                                    .canonical_correspondence
                                )
                                payload_invariant_lineage = bool(
                                    kv_cache["shared_dict"].get(
                                        "native_history_payload_invariant_lineage",
                                        False,
                                    )
                                )
                                if (
                                    payload_invariant_lineage
                                    and source_lineage is None
                                ):
                                    raise RuntimeError(
                                        "Payload-invariant native history "
                                        "requires a source lineage tier"
                                    )
                                (
                                    native_recent_start_frame,
                                    native_current_start_frame,
                                ) = native_history_read.temporal_origins(
                                    coalesce_bootstrap_alias=bool(
                                        kv_cache["shared_dict"].get(
                                            "native_history_coalesce_bootstrap_time",
                                            False,
                                        )
                                    )
                                )
                                canonical_target_key = (
                                    causal_rope_apply_indexed(
                                        canonical.target_key.to(
                                            device=q.device, dtype=q.dtype
                                        ),
                                        canonical.token_index.to(q.device),
                                        grid_sizes[: b // 2],
                                        freqs,
                                        start_frame=0,
                                    )
                                )
                                recent_target_key = (
                                    q.new_empty(
                                        (b // 2, 0, self.num_heads, self.head_dim)
                                    )
                                    if recent is None
                                    else causal_rope_apply_indexed(
                                        recent.target_key.to(
                                            device=q.device, dtype=q.dtype
                                        ),
                                        recent.token_index.to(q.device),
                                        grid_sizes[: b // 2],
                                        freqs,
                                        start_frame=(
                                            native_recent_start_frame
                                        ),
                                    )
                                )
                                recent_target_value = (
                                    v.new_empty(
                                        (b // 2, 0, self.num_heads, self.head_dim)
                                    )
                                    if recent is None
                                    else recent.target_value.to(
                                        device=v.device, dtype=v.dtype
                                    )
                                )
                                current_fixed_query = causal_rope_apply(
                                    (
                                        raw_trg_query[b_idx] * blender_rate
                                        + raw_src_query[b_idx]
                                        * (1.0 - blender_rate)
                                    ).unsqueeze(0),
                                    grid_sizes[b_idx].unsqueeze(0),
                                    freqs,
                                    start_frame=native_current_start_frame,
                                ).type_as(v)
                                current_unrotated_target_key = (
                                    (
                                        unrotated_attn_key.chunk(2, dim=0)[1][
                                            b_idx, -num_new_tokens:
                                        ]
                                        * blender_rate
                                        + unrotated_attn_key.chunk(2, dim=0)[0][
                                            b_idx, -num_new_tokens:
                                        ]
                                        * (1.0 - blender_rate)
                                    ).unsqueeze(0)
                                )
                                current_fixed_key = causal_rope_apply(
                                    current_unrotated_target_key,
                                    grid_sizes[b_idx].unsqueeze(0),
                                    freqs,
                                    start_frame=native_current_start_frame,
                                ).type_as(v)
                                clean_source_cache = kv_cache.get(
                                    "k_src_clean"
                                )
                                clean_source_value_cache = kv_cache.get(
                                    "v_src_clean"
                                )
                                if (
                                    clean_source_cache is None
                                    or clean_source_value_cache is None
                                ):
                                    raise RuntimeError(
                                        "Role-fixed native history requires "
                                        "the clean-source K/V cache"
                                    )
                                current_unrotated_source_key = (
                                    clean_source_cache[
                                        b_idx:b_idx + 1,
                                        local_start_index:local_end_index,
                                    ].to(device=q.device, dtype=q.dtype)
                                )
                                current_clean_source_value = (
                                    clean_source_value_cache[
                                        b_idx:b_idx + 1,
                                        local_start_index:local_end_index,
                                    ].to(device=v.device, dtype=v.dtype)
                                )
                                timestep_frame = (
                                    None
                                    if not timestep_counterfactual_memory
                                    else counterfactual_history
                                    .read_timestep_counterfactual(
                                        layer_index, timestep_index
                                    )
                                )
                                if timestep_counterfactual_memory and (
                                    timestep_frame is None
                                    or canonical_correspondence is None
                                    or flow_residual is None
                                ):
                                    raise RuntimeError(
                                        "TCCM read is missing its synchronized "
                                        "B0 bank, canonical flow correspondence, "
                                        "or local trust state"
                                    )
                                if timestep_counterfactual_memory:
                                    appearance_trust = (
                                        flow_residual.appearance_trust
                                        if flow_residual.appearance_trust is not None
                                        else flow_residual.confidence
                                    )[b_idx:b_idx + 1].to(q.device)
                                    local_transport_confidence = (
                                        flow_residual.transport_confidence
                                        if flow_residual.transport_confidence is not None
                                        else flow_residual.confidence
                                    )[b_idx:b_idx + 1].to(q.device)
                                    native_history_output, native_history_diag = (
                                        closed_loop_counterfactual_memory_attention(
                                            native_output=native_output,
                                            current_source_query=raw_src_query[
                                                b_idx:b_idx + 1
                                            ],
                                            current_source_key=k.chunk(2, dim=0)[0][
                                                b_idx:b_idx + 1
                                            ],
                                            current_source_value=v.chunk(2, dim=0)[0][
                                                b_idx:b_idx + 1
                                            ],
                                            current_target_key=k.chunk(2, dim=0)[1][
                                                b_idx:b_idx + 1
                                            ],
                                            current_target_value=v.chunk(2, dim=0)[1][
                                                b_idx:b_idx + 1
                                            ],
                                            canonical_source_key=timestep_frame.source_key[
                                                b_idx:b_idx + 1
                                            ].to(device=q.device, dtype=q.dtype),
                                            canonical_source_value=timestep_frame.source_value[
                                                b_idx:b_idx + 1
                                            ].to(device=v.device, dtype=v.dtype),
                                            canonical_target_key=timestep_frame.target_key[
                                                b_idx:b_idx + 1
                                            ].to(device=q.device, dtype=q.dtype),
                                            canonical_target_value=timestep_frame.target_value[
                                                b_idx:b_idx + 1
                                            ].to(device=v.device, dtype=v.dtype),
                                            canonical_support=timestep_frame.support[
                                                b_idx:b_idx + 1
                                            ].to(q.device),
                                            canonical_token_index=timestep_frame.token_index[
                                                b_idx:b_idx + 1
                                            ].to(q.device),
                                            mapped_current_index=canonical_correspondence.current_index[
                                                b_idx:b_idx + 1
                                            ].to(q.device),
                                            correspondence_support=canonical_correspondence.support[
                                                b_idx:b_idx + 1
                                            ].to(q.device),
                                            correspondence_confidence=canonical_correspondence.confidence[
                                                b_idx:b_idx + 1
                                            ].to(q.device),
                                            owner_gate=(
                                                object_weight
                                                if bool(shared_dict.get(
                                                    "native_history_transactional_owner",
                                                    False,
                                                ))
                                                else object_weight * current_actions[
                                                    "target_memory_action"
                                                ].float()
                                            )[None].to(q.device),
                                            appearance_trust=appearance_trust,
                                            transport_confidence=(
                                                local_transport_confidence
                                            ),
                                            current_address_key=(
                                                current_unrotated_source_key
                                            ),
                                            canonical_address_key=canonical.source_key[
                                                b_idx:b_idx + 1
                                            ].to(device=q.device, dtype=q.dtype),
                                            tokens_per_frame=frame_seqlen,
                                            spatial_shape=tuple(
                                                int(value.item())
                                                for value in grid_sizes[b_idx][1:]
                                            ),
                                            canonical_frame_count=(
                                                timestep_frame.frame_count
                                            ),
                                            current_frame_count=(
                                                num_new_tokens // frame_seqlen
                                            ),
                                            topk_per_frame=int(shared_dict.get(
                                                "native_history_multiframe_sink_topk_per_frame",
                                                8,
                                            )),
                                            min_source_similarity=float(
                                                shared_dict.get(
                                                    "native_history_min_similarity",
                                                    0.35,
                                                )
                                            ),
                                            source_logit_bias=float(shared_dict.get(
                                                "native_history_multiframe_sink_source_logit_bias",
                                                1.0,
                                            )),
                                            flow_radius=float(shared_dict.get(
                                                "native_history_tccm_flow_radius",
                                                2.0,
                                            )),
                                            strength=float(shared_dict.get(
                                                "native_history_tccm_strength",
                                                1.0,
                                            )),
                                            max_error_ratio=float(shared_dict.get(
                                                "native_history_tccm_max_error_ratio",
                                                1.0,
                                            )),
                                        )
                                    )
                                else:
                                    native_history_output, native_history_diag = (
                                        source_addressed_native_history_attention(
                                        native_output,
                                        current_fixed_query,
                                        canonical_target_key[
                                            b_idx:b_idx + 1
                                        ],
                                        canonical.target_value[
                                            b_idx:b_idx + 1
                                        ].to(device=v.device, dtype=v.dtype),
                                        recent_target_key[
                                            b_idx:b_idx + 1
                                        ],
                                        recent_target_value[
                                            b_idx:b_idx + 1
                                        ],
                                        (
                                            torch.empty(
                                                (1, 0),
                                                dtype=torch.bool,
                                                device=q.device,
                                            )
                                            if recent is None
                                            else recent.support[
                                                b_idx:b_idx + 1
                                            ].to(q.device)
                                        ),
                                        current_fixed_key,
                                        native_target_value,
                                        current_unrotated_source_key,
                                        canonical.source_key[
                                            b_idx:b_idx + 1
                                        ].to(device=q.device, dtype=q.dtype),
                                        canonical.support[
                                            b_idx:b_idx + 1
                                        ].to(q.device),
                                        (
                                            object_weight
                                            if bool(
                                                kv_cache["shared_dict"].get(
                                                    "native_history_transactional_owner",
                                                    False,
                                                )
                                            )
                                            else (
                                                object_weight
                                                * current_actions[
                                                    "target_memory_action"
                                                ].float()
                                            )
                                        )[None].to(q.device),
                                        current_source_value=(
                                            current_clean_source_value
                                        ),
                                        canonical_source_value=(
                                            canonical.source_value[
                                                b_idx:b_idx + 1
                                            ].to(
                                                device=v.device,
                                                dtype=v.dtype,
                                            )
                                        ),
                                        recent_source_value=(
                                            v.new_empty(
                                                (
                                                    b // 2, 0,
                                                    self.num_heads,
                                                    self.head_dim,
                                                )
                                            )[b_idx:b_idx + 1]
                                            if recent is None
                                            else recent.source_value[
                                                b_idx:b_idx + 1
                                            ].to(
                                                device=v.device,
                                                dtype=v.dtype,
                                            )
                                        ),
                                        topk=int(
                                            kv_cache["shared_dict"].get(
                                                "native_history_topk", 8
                                            )
                                        ),
                                        min_similarity=float(
                                            kv_cache["shared_dict"].get(
                                                "native_history_min_similarity",
                                                0.35,
                                            )
                                        ),
                                        min_request=float(
                                            kv_cache["shared_dict"].get(
                                                "native_history_min_query_confidence",
                                                0.5,
                                            )
                                        ),
                                        canonical_logit_bias=float(
                                            kv_cache["shared_dict"].get(
                                                "native_history_canonical_logit_bias",
                                                1.0,
                                            )
                                        ),
                                        source_part_consistency=bool(
                                            kv_cache["shared_dict"].get(
                                                "native_history_source_part_consistency",
                                                False,
                                            )
                                        ),
                                        min_part_similarity=float(
                                            kv_cache["shared_dict"].get(
                                                "native_history_min_part_similarity",
                                                0.45,
                                            )
                                        ),
                                        part_similarity_margin=float(
                                            kv_cache["shared_dict"].get(
                                                "native_history_part_similarity_margin",
                                                0.08,
                                            )
                                        ),
                                        part_bias_strength=float(
                                            kv_cache["shared_dict"].get(
                                                "native_history_part_bias_strength",
                                                0.5,
                                            )
                                        ),
                                        part_refinement_ratio=float(
                                            kv_cache["shared_dict"].get(
                                                "native_history_part_refinement_ratio",
                                                0.25,
                                            )
                                        ),
                                        payload_invariant_lineage=(
                                            payload_invariant_lineage
                                        ),
                                        recent_source_key=(
                                            (
                                                recent.source_key[
                                                    b_idx:b_idx + 1
                                                ].to(
                                                    device=q.device,
                                                    dtype=q.dtype,
                                                )
                                                if (
                                                    recent is not None
                                                    and bool(
                                                        kv_cache["shared_dict"].get(
                                                            "native_history_consistent_transaction",
                                                            False,
                                                        )
                                                    )
                                                )
                                                else None
                                                if source_lineage is None
                                                else source_lineage.source_key[
                                                    b_idx:b_idx + 1
                                                ].to(
                                                    device=q.device,
                                                    dtype=q.dtype,
                                                )
                                            )
                                        ),
                                        recent_lineage_index=(
                                            None
                                            if source_lineage is None
                                            else source_lineage.canonical_index[
                                                b_idx:b_idx + 1
                                            ].to(q.device)
                                        ),
                                        recent_lineage_support=(
                                            None
                                            if source_lineage is None
                                            else source_lineage.support[
                                                b_idx:b_idx + 1
                                            ].to(q.device)
                                        ),
                                        recent_lineage_confidence=(
                                            None
                                            if source_lineage is None
                                            else source_lineage.confidence[
                                                b_idx:b_idx + 1
                                            ].to(q.device)
                                        ),
                                        payload_blend_strength=float(
                                            kv_cache["shared_dict"].get(
                                                "native_history_payload_blend_strength",
                                                0.35,
                                            )
                                        ),
                                        consistent_transaction=bool(
                                            kv_cache["shared_dict"].get(
                                                "native_history_consistent_transaction",
                                                False,
                                            )
                                        ),
                                        entry_bridge=bool(
                                            kv_cache["shared_dict"].get(
                                                "native_history_recent_entry_bridge",
                                                False,
                                            )
                                        ),
                                        motion_owner_dense_read=bool(
                                            kv_cache["shared_dict"].get(
                                                "native_history_motion_owner_dense_read",
                                                False,
                                            )
                                        ),
                                        entry_query_count=frame_seqlen,
                                        entry_bridge_strength=float(
                                            kv_cache["shared_dict"].get(
                                                "native_history_entry_bridge_strength",
                                                1.0,
                                            )
                                        ),
                                        dual_evidence_arbitration=bool(
                                            kv_cache["shared_dict"].get(
                                                "native_history_dual_evidence_arbitration",
                                                False,
                                            )
                                        ),
                                        min_payload_consistency=float(
                                            kv_cache["shared_dict"].get(
                                                "native_history_min_payload_consistency",
                                                0.15,
                                            )
                                        ),
                                        recent_payload_support=(
                                            None
                                            if recent is None
                                            or recent.payload_support is None
                                            else recent.payload_support[
                                                b_idx:b_idx + 1
                                            ].to(q.device)
                                        ),
                                        residual_rebased_payload=(
                                            False
                                            if recent is None
                                            else bool(
                                                recent.residual_rebased_payload
                                            )
                                        ),
                                        last_trusted_appearance=bool(
                                            kv_cache["shared_dict"].get(
                                                "native_history_last_trusted_appearance",
                                                False,
                                            )
                                        ),
                                        flow_indexed_value_residual=(
                                            None
                                            if flow_residual is None
                                            else flow_residual.value_residual[
                                                b_idx:b_idx + 1
                                            ].to(
                                                device=v.device,
                                                dtype=v.dtype,
                                            )
                                        ),
                                        flow_indexed_support=(
                                            None
                                            if flow_residual is None
                                            else flow_residual.support[
                                                b_idx:b_idx + 1
                                            ].to(q.device)
                                        ),
                                        flow_indexed_confidence=(
                                            None
                                            if flow_residual is None
                                            else flow_residual.confidence[
                                                b_idx:b_idx + 1
                                            ].to(q.device)
                                        ),
                                        flow_indexed_appearance_trust=(
                                            None
                                            if flow_residual is None
                                            or flow_residual.appearance_trust is None
                                            else flow_residual.appearance_trust[
                                                b_idx:b_idx + 1
                                            ].to(q.device)
                                        ),
                                        flow_indexed_transport_confidence=(
                                            None
                                            if flow_residual is None
                                            or flow_residual.transport_confidence is None
                                            else flow_residual.transport_confidence[
                                                b_idx:b_idx + 1
                                            ].to(q.device)
                                        ),
                                        multiframe_identity_sink=bool(
                                            kv_cache["shared_dict"].get(
                                                "native_history_multiframe_identity_sink",
                                                False,
                                            )
                                        ),
                                        canonical_token_index=(
                                            canonical.token_index[
                                                b_idx:b_idx + 1
                                            ].to(q.device)
                                        ),
                                        canonical_tokens_per_frame=(
                                            frame_seqlen
                                        ),
                                        canonical_frame_count=(
                                            canonical.frame_count
                                        ),
                                        multiframe_sink_topk_per_frame=int(
                                            kv_cache["shared_dict"].get(
                                                "native_history_multiframe_sink_topk_per_frame",
                                                8,
                                            )
                                        ),
                                        multiframe_sink_source_logit_bias=float(
                                            kv_cache["shared_dict"].get(
                                                "native_history_multiframe_sink_source_logit_bias",
                                                1.0,
                                            )
                                        ),
                                        multiframe_sink_strength=float(
                                            kv_cache["shared_dict"].get(
                                                "native_history_multiframe_sink_strength",
                                                1.0,
                                            )
                                        ),
                                        )
                                    )
                                native_output = native_history_output
                                native_history_diag.update({
                                    "bootstrap_alias": (
                                        native_output.new_tensor(
                                            float(
                                                native_history_read
                                                .recent_shares_canonical_time
                                            )
                                        )
                                    ),
                                    "recent_start_frame": (
                                        native_output.new_tensor(
                                            float(native_recent_start_frame)
                                        )
                                    ),
                                    "current_start_frame": (
                                        native_output.new_tensor(
                                            float(native_current_start_frame)
                                        )
                                    ),
                                    "bypass": native_output.new_zeros(()),
                                })
                                kv_cache["shared_dict"].setdefault(
                                    "role_native_history_diagnostics", {}
                                )[layer_index] = {
                                    name: value.detach()
                                    for name, value in native_history_diag.items()
                                }
                                kv_cache["shared_dict"].setdefault(
                                    "role_native_history_admissions", {}
                                ).setdefault(layer_index, []).append(
                                    (
                                        native_history_diag["read_strength"]
                                        if (
                                            payload_invariant_lineage
                                            or bool(
                                                kv_cache["shared_dict"].get(
                                                    "native_history_consistent_transaction",
                                                    False,
                                                )
                                            )
                                        )
                                        else native_history_diag["admitted"]
                                    ).detach()
                                )
                                if bool(
                                    kv_cache["shared_dict"].get(
                                        "native_history_verified_attention_authority",
                                        False,
                                    )
                                ):
                                    read_strength = native_history_diag[
                                        "read_strength"
                                    ][0].detach().float()
                                    authority_states = kv_cache[
                                        "shared_dict"
                                    ].setdefault(
                                        "verified_attention_authority_state",
                                        {},
                                    )
                                    authority_state = authority_states.get(
                                        b_idx
                                    )
                                    if authority_state is None:
                                        authority_state = {
                                            "strength_sum": (
                                                torch.zeros_like(read_strength)
                                            ),
                                            "support_sum": (
                                                torch.zeros_like(read_strength)
                                            ),
                                            "layer_count": 0,
                                        }
                                    authority_state["strength_sum"] = (
                                        authority_state["strength_sum"]
                                        + read_strength
                                    )
                                    authority_state["support_sum"] = (
                                        authority_state["support_sum"]
                                        + (read_strength > 0.0).float()
                                    )
                                    authority_state["layer_count"] += 1
                                    authority_states[b_idx] = authority_state
                            elif (
                                role_fixed_native_history
                                and native_history_read is not None
                                and native_history_bypass
                            ):
                                canonical = native_history_read.canonical
                                recent = native_history_read.recent
                                (
                                    native_recent_start_frame,
                                    native_current_start_frame,
                                ) = native_history_read.temporal_origins(
                                    coalesce_bootstrap_alias=bool(
                                        kv_cache["shared_dict"].get(
                                            "native_history_coalesce_bootstrap_time",
                                            False,
                                        )
                                    )
                                )
                                zero_query = object_weight.new_zeros(
                                    (1, object_weight.numel())
                                )
                                kv_cache["shared_dict"].setdefault(
                                    "role_native_history_diagnostics", {}
                                )[layer_index] = {
                                    "admitted": zero_query.bool(),
                                    "best_similarity": zero_query - 1.0,
                                    "output_delta": zero_query,
                                    "canonical_candidates": (
                                        canonical.support.float().sum(dim=-1)
                                    ),
                                    "recent_tokens": zero_query.new_full(
                                        (1,),
                                        float(
                                            0
                                            if recent is None
                                            else recent.target_key.shape[1]
                                        ),
                                    ),
                                    "bootstrap_alias": zero_query.new_tensor(
                                        float(
                                            native_history_read
                                            .recent_shares_canonical_time
                                        )
                                    ),
                                    "recent_start_frame": (
                                        zero_query.new_tensor(
                                            float(native_recent_start_frame)
                                        )
                                    ),
                                    "current_start_frame": (
                                        zero_query.new_tensor(
                                            float(native_current_start_frame)
                                        )
                                    ),
                                    "bypass": zero_query.new_ones(()),
                                }
                                kv_cache["shared_dict"].setdefault(
                                    "role_native_history_admissions", {}
                                ).setdefault(layer_index, []).append(
                                    zero_query.bool()
                                )
                            if (
                                paired_read is None
                                and not role_fixed_native_history
                            ):
                                # Exact 926/923 fallback outside configured
                                # memory layers and before bootstrap.
                                x_list.append(native_output)
                                continue
                            if bool(
                                kv_cache["shared_dict"].get(
                                    "native_history_verified_attention_authority",
                                    False,
                                )
                            ) and layer_index >= int(
                                kv_cache["shared_dict"].get(
                                    "native_history_attention_authority_start_layer",
                                    layer_index,
                                )
                            ):
                                authority_state = kv_cache[
                                    "shared_dict"
                                ].get(
                                    "verified_attention_authority_state", {}
                                ).get(b_idx)
                                if authority_state is not None:
                                    layer_count = max(
                                        int(authority_state["layer_count"]), 1
                                    )
                                    mean_strength = (
                                        authority_state["strength_sum"]
                                        / float(layer_count)
                                    )
                                    layer_agreement = (
                                        authority_state["support_sum"]
                                        / float(layer_count)
                                    )
                                    authority_gate = (
                                        mean_strength * layer_agreement
                                    ).clamp(0.0, 1.0)
                                    authority_input = native_output
                                    native_output = (
                                        arbitrate_verified_factorized_attention(
                                            native_output,
                                            source_mixed_native_output,
                                            factorized_output,
                                            authority_gate.unsqueeze(0),
                                            strength=float(
                                                kv_cache["shared_dict"].get(
                                                    "native_history_attention_authority_strength",
                                                    1.0,
                                                )
                                            ),
                                        )
                                    )
                                    authority_delta = (
                                        native_output.float()
                                        - authority_input.float()
                                    ).abs().mean(dim=(-1, -2)).squeeze(0)
                                    native_gap = (
                                        factorized_output.float()
                                        - source_mixed_native_output.float()
                                    ).abs().mean(dim=(-1, -2)).squeeze(0)
                                    active = authority_gate > 0.0
                                    active_count = (
                                        active.float().sum().clamp_min(1.0)
                                    )
                                    kv_cache["shared_dict"].setdefault(
                                        "verified_attention_authority_diagnostics",
                                        {},
                                    )[layer_index] = {
                                        "gate": authority_gate.detach(),
                                        "active": active.detach(),
                                        "active_gate": (
                                            authority_gate[active].sum()
                                            / active_count
                                        ).detach(),
                                        "active_output_delta": (
                                            authority_delta[active].sum()
                                            / active_count
                                        ).detach(),
                                        "active_native_factorized_gap": (
                                            native_gap[active].sum()
                                            / active_count
                                        ).detach(),
                                        "verified_layers": (
                                            authority_gate.new_tensor(
                                                float(layer_count)
                                            )
                                        ),
                                    }
                            read_support = (
                                paired_read.support[b_idx]
                                .to(native_output.device).float()
                                if paired_read is not None
                                else torch.zeros_like(object_weight)
                            )
                            read_residual = (
                                paired_read.residual[b_idx]
                                .to(native_output.device).float()
                                if paired_read is not None
                                else native_output.new_zeros(
                                    native_output.shape[1:]
                                )
                            )
                            read_strength = float(
                                kv_cache["shared_dict"].get(
                                    "paired_memory_read_strength", 0.0
                                )
                            )
                            anchor_delta = native_output.new_zeros(())
                            anchor_query_coverage = (
                                native_output.new_zeros(())
                            )
                            anchor_key_count = native_output.new_zeros(())
                            if role_fixed_native_history:
                                paired_output = native_output
                            elif dual_timescale_anchor:
                                # 935: motion remains in the untouched dense
                                # native target-history branch above.  A
                                # separate current-source attention computes
                                # only the canonical appearance delta, so the
                                # immutable signal does not share a softmax
                                # with an ever-growing, potentially drifting
                                # target history.  Current source Q/K provide
                                # pose-following addresses without changing
                                # the backbone RoPE layout.
                                canonical_anchors = (
                                    kv_cache["shared_dict"].get(
                                        "paired_memory_canonical_anchors",
                                        {},
                                    )
                                )
                                canonical_anchor = canonical_anchors.get(
                                    layer_index
                                )
                                if (
                                    canonical_key_anchor
                                    and canonical_anchor is None
                                ):
                                    # Before proposal bootstrap there is no
                                    # canonical bank.  The native path is the
                                    # exact and only valid fallback.
                                    supported = torch.zeros_like(
                                        read_support, dtype=torch.bool
                                    )
                                elif canonical_key_anchor:
                                    supported = canonical_anchor.query_support[
                                        b_idx
                                    ].to(native_output.device) > 0.0
                                else:
                                    supported = read_support > 0.0
                                anchor_query_coverage = (
                                    supported.float().mean()
                                )
                                anchor_key_count = (
                                    supported.float().sum()
                                )
                                if bool(supported.any()):
                                    if canonical_key_anchor:
                                        canonical_evidence = (
                                            canonical_anchor.evidence[b_idx]
                                            .to(native_output.device)
                                        )
                                        valid_keys = canonical_evidence > 0.0
                                        anchor_query = raw_src_query[
                                            b_idx, supported
                                        ].unsqueeze(0)
                                        canonical_anchor_delta = (
                                            immutable_canonical_anchor_attention_delta(
                                                anchor_query,
                                                canonical_anchor.source_key[
                                                    b_idx, valid_keys
                                                ].to(native_output.device).unsqueeze(0),
                                                canonical_anchor.target_value_residual[
                                                    b_idx, valid_keys
                                                ].to(native_output.device).unsqueeze(0),
                                                canonical_evidence[
                                                    valid_keys
                                                ].unsqueeze(0),
                                                canonical_anchor.query_lineage_id[
                                                    b_idx, supported
                                                ].to(native_output.device).unsqueeze(0),
                                                canonical_anchor.lineage_id[
                                                    b_idx, valid_keys
                                                ].to(native_output.device).unsqueeze(0),
                                                query_key_mask=(
                                                    canonical_anchor.query_key_mask[
                                                        b_idx, supported
                                                    ][:, valid_keys]
                                                    .to(native_output.device)
                                                    .unsqueeze(0)
                                                ),
                                            )
                                        )
                                        anchor_key_count = (
                                            valid_keys.float().sum()
                                        )
                                    else:
                                        anchor_query = src_query[
                                            b_idx, supported
                                        ].unsqueeze(0)
                                        anchor_key = src_current_key[
                                            b_idx, supported
                                        ].unsqueeze(0)
                                        anchor_residual = (
                                            read_residual[supported]
                                        ).unsqueeze(0)
                                        if paired_read.lineage_id is None:
                                            raise RuntimeError(
                                                "Dual-timescale anchor requires "
                                                "source-transport lineage ids"
                                            )
                                        anchor_lineage = paired_read.lineage_id[
                                            b_idx, supported
                                        ].to(native_output.device).unsqueeze(0)
                                        canonical_anchor_delta = (
                                            source_addressed_anchor_attention_delta(
                                                anchor_query,
                                                anchor_key,
                                                anchor_residual,
                                                read_support[
                                                    supported
                                                ].unsqueeze(0),
                                                anchor_lineage,
                                            )
                                        )
                                    paired_output = (
                                        scatter_source_addressed_anchor_delta(
                                            native_output,
                                            canonical_anchor_delta,
                                            supported.unsqueeze(0),
                                            strength=read_strength,
                                        )
                                    )
                                    anchor_delta = (
                                        canonical_anchor_delta.float()
                                    ).abs().mean()
                                else:
                                    paired_output = native_output
                            elif paired_value_projection:
                                # The residual has already been materialized
                                # as V_source + delta_V before attention. In
                                # query-gated mode, compare projected and raw
                                # attention and expose that difference only
                                # to queries with a successful paired read.
                                # This is an output-side access policy, not a
                                # second residual addition.
                                paired_output = (
                                    arbitrate_projected_attention_output(
                                        native_output,
                                        projected_native_output,
                                        read_support.unsqueeze(0),
                                        binary_access=single_confidence,
                                    )
                                    if query_gated_projection
                                    else native_output
                                )
                            else:
                                paired_output = (
                                    blend_source_addressed_residual(
                                        native_output,
                                        read_residual.unsqueeze(0),
                                        read_support.unsqueeze(0),
                                        strength=read_strength,
                                    )
                                )
                            correction = (
                                paired_output.float()
                                - native_output.float()
                            ).abs().mean(dim=(-1, -2)).squeeze(0)
                            matched_read = read_support > 0.0
                            matched_count = (
                                matched_read.float().sum().clamp_min(1.0)
                            )
                            if paired_read is not None:
                                kv_cache["shared_dict"].setdefault(
                                    "paired_edit_memory_diagnostics", {}
                                )[layer_index] = {
                                    "read_support": (
                                        read_support.mean().detach()
                                    ),
                                    "read_similarity": (
                                        (
                                            paired_read.best_similarity[b_idx]
                                            .float()
                                            * matched_read.float()
                                        ).sum().div(matched_count).detach()
                                    ),
                                    "correction": (
                                        (correction * matched_read.float())
                                        .sum().div(matched_count).detach()
                                    ),
                                    "value_projection": (
                                        value_projection_delta.detach()
                                    ),
                                    "ungated_projection_leakage": (
                                        ungated_projection_leakage.detach()
                                    ),
                                    "query_gated_projection": (
                                        read_support.new_tensor(
                                            float(query_gated_projection)
                                        ).detach()
                                    ),
                                    "dual_timescale_anchor": (
                                        read_support.new_tensor(
                                            float(dual_timescale_anchor)
                                        ).detach()
                                    ),
                                    "canonical_key_anchor": (
                                        read_support.new_tensor(
                                            float(canonical_key_anchor)
                                        ).detach()
                                    ),
                                    "anchor_delta": anchor_delta.detach(),
                                    "anchor_query_coverage": (
                                        anchor_query_coverage.detach()
                                    ),
                                    "anchor_key_count": (
                                        anchor_key_count.detach()
                                    ),
                                }
                            x_list.append(paired_output)
                            continue
                        x_list.append(
                            blend_factorized_with_native_fallback(
                                factorized_output,
                                native_output,
                                current_actions[
                                    "unknown_action"
                                ].unsqueeze(0),
                            )
                        )
                        continue

                    dual_belief_kv = all(
                        kv_cache.get(name) is not None
                        for name in (
                            "cached_preserve_kv_action",
                        )
                    )
                    if target_owned_handoff and not dual_belief_kv:
                        raise RuntimeError(
                            "Target-owned handoff requires aligned belief "
                            "KV history"
                        )
                    if dual_belief_kv:
                        cached_preserve_action = kv_cache[
                            "cached_preserve_kv_action"
                        ][b_idx, attn_seq_slice][:-num_new_tokens]
                        if cached_preserve_action.shape[0] != (
                            src_prev_key.shape[1]
                        ):
                            raise ValueError(
                                "Cached belief KV actions must align with "
                                "historical memory tokens"
                            )
                        legacy_prev_key, legacy_prev_value = (
                            fuse_aligned_memory(
                                trg_prev_key[b_idx].unsqueeze(0),
                                trg_prev_value[b_idx].unsqueeze(0),
                                src_prev_key[b_idx].unsqueeze(0),
                                src_prev_value[b_idx].unsqueeze(0),
                                cached_preserve_action.unsqueeze(0),
                            )
                        )
                        legacy_current_key = (
                            trg_current_key[b_idx] * blender_rate
                            + src_current_key[b_idx]
                            * (1 - blender_rate)
                        )
                        legacy_key_list = [
                            legacy_prev_key.squeeze(0)
                        ]
                        legacy_value_list = [
                            legacy_prev_value.squeeze(0)
                        ]
                        if kv_cache["shared_dict"][
                            "current_timestep_index"
                        ] > kv_cache["shared_dict"][
                            "total_timestep"
                        ] // 2:
                            current_edit_mask = kv_cache[
                                "current_src_fg_mask"
                            ][b_idx]
                            current_background_mask = ~current_edit_mask
                            legacy_key_list.append(
                                src_current_key[b_idx][
                                    current_background_mask
                                ]
                            )
                            legacy_value_list.append(
                                src_current_value[b_idx][
                                    current_background_mask
                                ]
                            )
                        legacy_key_list.append(legacy_current_key)
                        legacy_value_list.append(
                            trg_current_value[b_idx]
                        )
                        legacy_memory_key = torch.cat(
                            legacy_key_list,
                            dim=0,
                        )
                        legacy_memory_value = torch.cat(
                            legacy_value_list,
                            dim=0,
                        )
                        legacy_query = (
                            trg_query[b_idx] * blender_rate
                            + src_query[b_idx] * (1 - blender_rate)
                        )
                        legacy_target_output = attention(
                            legacy_query.unsqueeze(0),
                            legacy_memory_key.unsqueeze(0),
                            legacy_memory_value.unsqueeze(0),
                        )
                        owned = (
                            None
                            if current_target_owned_mask is None
                            else current_target_owned_mask[b_idx]
                        )
                        if owned is None or not owned.any():
                            x_list.append(legacy_target_output)
                            continue

                        owned_preserve_action = (
                            suppress_source_preserve_on_target_owned_history(
                                cached_preserve_action,
                                target_owned_history_mask[b_idx],
                            )
                        )
                        owned_prev_key, owned_prev_value = (
                            fuse_aligned_memory(
                                trg_prev_key[b_idx].unsqueeze(0),
                                trg_prev_value[b_idx].unsqueeze(0),
                                src_prev_key[b_idx].unsqueeze(0),
                                src_prev_value[b_idx].unsqueeze(0),
                                owned_preserve_action.unsqueeze(0),
                            )
                        )
                        owned_current_key = blend_target_owned_tensor(
                            trg_current_key[b_idx],
                            src_current_key[b_idx],
                            blender_rate,
                            owned,
                        )
                        owned_key_list = [owned_prev_key.squeeze(0)]
                        owned_value_list = [owned_prev_value.squeeze(0)]
                        if kv_cache["shared_dict"][
                            "current_timestep_index"
                        ] > kv_cache["shared_dict"][
                            "total_timestep"
                        ] // 2:
                            owned_background_mask = (
                                build_target_owned_source_background_mask(
                                    current_edit_mask,
                                    owned,
                                )
                            )
                            owned_key_list.append(
                                src_current_key[b_idx][
                                    owned_background_mask
                                ]
                            )
                            owned_value_list.append(
                                src_current_value[b_idx][
                                    owned_background_mask
                                ]
                            )
                        owned_key_list.append(owned_current_key)
                        owned_value_list.append(trg_current_value[b_idx])
                        owned_query = blend_target_owned_tensor(
                            trg_query[b_idx],
                            src_query[b_idx],
                            blender_rate,
                            owned,
                        )
                        owned_target_output = attention(
                            owned_query[owned].unsqueeze(0),
                            torch.cat(
                                owned_key_list, dim=0
                            ).unsqueeze(0),
                            torch.cat(
                                owned_value_list, dim=0
                            ).unsqueeze(0),
                        )
                        b_target_output = scatter_target_owned_output(
                            legacy_target_output,
                            owned_target_output,
                            owned.unsqueeze(0),
                        )
                        x_list.append(b_target_output)
                        continue

                    b_key_list = []
                    b_value_list = []
                    
                    #✨ masked-blended previous kv
                    b_trg_fg_mask = kv_cache["trg_fg_mask"][b_idx]                              # [L_cache_size, ]
                    b_trg_attn_fg_mask = b_trg_fg_mask[attn_seq_slice]                          # [L_attn_size, ]
                    b_trg_prev_fg_mask = b_trg_attn_fg_mask[: -num_new_tokens]
                    b_trg_prev_fg_key = trg_prev_key[b_idx]
                    b_trg_prev_fg_key[b_trg_prev_fg_mask] = b_trg_prev_fg_key[b_trg_prev_fg_mask] * blender_rate \
                        + src_prev_key[b_idx, b_trg_prev_fg_mask] * (1 - blender_rate)
                    b_trg_prev_fg_value = trg_prev_value[b_idx]
                    b_key_list.append(b_trg_prev_fg_key)
                    b_value_list.append(b_trg_prev_fg_value)

                    #✨ current source condition
                    b_src_current_fg_mask = kv_cache["current_src_fg_mask"][b_idx]              # [Lq, ]
                    b_src_current_bg_mask = ~b_src_current_fg_mask                              # [Lq, ]
                    # t^inj=0.5
                    if kv_cache['shared_dict']['current_timestep_index'] > kv_cache['shared_dict']['total_timestep'] // 2:
                        b_src_current_bg_key = src_current_key[b_idx][b_src_current_bg_mask]        # [L_bg, Nh, Dk]
                        b_src_current_bg_value = src_current_value[b_idx][b_src_current_bg_mask]    # [L_bg, Nh, Dv]
                        b_key_list.append(b_src_current_bg_key)
                        b_value_list.append(b_src_current_bg_value)

                    #✨ masked-blended current target condition
                    b_trg_current_key = trg_current_key[b_idx]                                  # [Lq, Nh, Dk]
                    b_trg_current_value = trg_current_value[b_idx]                              # [Lq, Nh, Dk]
                    b_trg_current_key = b_trg_current_key * blender_rate + src_current_key[b_idx] * (1 - blender_rate)
                    b_key_list.append(b_trg_current_key)
                    b_value_list.append(b_trg_current_value)

                    # Identity anchor: concat frozen first-block KV
                    identity_anchor = shared_dict.get(
                        "identity_anchor_kv"
                    )
                    if identity_anchor is not None:
                        anchor_layer = identity_anchor[layer_index]
                        b_key_list.append(
                            anchor_layer["k"][b_idx].to(
                                b_trg_current_key.dtype
                            )
                        )
                        b_value_list.append(
                            anchor_layer["v"][b_idx].to(
                                b_trg_current_value.dtype
                            )
                        )

                    # store and concatenate key and value
                    b_trg_key = torch.cat(b_key_list, dim=0)
                    b_trg_value = torch.cat(b_value_list, dim=0)

                    #✨ query blending
                    b_query = trg_query[b_idx] * blender_rate + src_query[b_idx] * (1 - blender_rate)
                    b_target_output = attention(
                        b_query.unsqueeze(0),
                        b_trg_key.unsqueeze(0),
                        b_trg_value.unsqueeze(0),
                    )
                    contact_graph_mode = shared_dict.get(
                        "contact_graph_mode",
                        "no_graph",
                    )
                    contact_graphs = shared_dict.get("contact_graphs")
                    layer_start = shared_dict.get(
                        "contact_graph_layer_start",
                        0,
                    )
                    layer_end = shared_dict.get(
                        "contact_graph_layer_end",
                        0,
                    )
                    if (
                        contact_graph_mode != "no_graph"
                        and contact_graphs is not None
                        and layer_start <= layer_index < layer_end
                    ):
                        b_target_output = apply_contact_graph_residual(
                            target_output=b_target_output,
                            source_query=src_query[b_idx],
                            target_query=trg_query[b_idx],
                            source_key=src_current_key[b_idx],
                            target_key=trg_current_key[b_idx],
                            source_value=src_current_value[b_idx],
                            target_value=trg_current_value[b_idx],
                            graph=contact_graphs[b_idx],
                            mode=contact_graph_mode,
                            strength=shared_dict[
                                "contact_graph_strength"
                            ],
                        )
                    x_list.append(b_target_output)
                x = torch.cat(x_list, dim=0)

            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)
            kv_cache["sink_tokens"] = sink_tokens
            kv_cache["num_new_tokens"] = num_new_tokens

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start)

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
