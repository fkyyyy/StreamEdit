"""Frozen first-block target-minus-source value memory for M1.

The bank owns no retrieval logic.  It only snapshots aligned pre-RoPE clean
source keys and target-minus-source value residuals after the first clean
target commit.  Keeping the state container separate makes the no-upsert
contract explicit and easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import torch


@dataclass(frozen=True)
class ImmutableDeltaVLayerState:
    source_key: torch.Tensor
    delta_value: torch.Tensor
    support: torch.Tensor

    def validate(self) -> None:
        if self.source_key.ndim != 4 or self.delta_value.ndim != 4:
            raise ValueError(
                "M1 source keys and delta values must have shape [B,L,H,D]"
            )
        if self.source_key.shape != self.delta_value.shape:
            raise ValueError(
                "M1 source keys and delta values must align"
            )
        if self.support.shape != self.source_key.shape[:2]:
            raise ValueError(
                "M1 support must align with the bank token sequence"
            )


class ImmutableDeltaVBank:
    """A write-once collection of layer-local delta-V tensors."""

    def __init__(self, layers: Iterable[int]):
        self.layers = tuple(int(layer) for layer in layers)
        if not self.layers:
            raise ValueError("M1 layers must not be empty")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("M1 layers must be unique")
        if any(layer < 0 for layer in self.layers):
            raise ValueError("M1 layers must be non-negative")
        self._states: Dict[int, ImmutableDeltaVLayerState] = {}

    @property
    def is_frozen(self) -> bool:
        return bool(self._states)

    @torch.no_grad()
    def freeze(
        self,
        *,
        source_kv_cache,
        target_kv_cache,
        source_keys: Mapping[int, torch.Tensor],
        support: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Freeze the first clean block; a second write is forbidden."""
        if self.is_frozen:
            raise RuntimeError("M1 immutable delta-V bank is already frozen")
        if support.ndim != 2:
            raise ValueError("M1 support must have shape [B,L]")

        support = support.detach().bool()
        support_counts = support.sum(dim=1)
        max_slots = int(support_counts.max().item())
        if max_slots <= 0:
            raise RuntimeError(
                "M1 first-block automatic SOG support is empty"
            )
        states: Dict[int, ImmutableDeltaVLayerState] = {}
        for layer in self.layers:
            if layer not in source_keys:
                raise RuntimeError(
                    f"Missing M1 clean-source key capture at layer {layer}"
                )
            source_cache = source_kv_cache[layer]
            target_cache = target_kv_cache[layer]
            source_count = int(source_cache["num_new_tokens"])
            target_count = int(target_cache["num_new_tokens"])
            if source_count != target_count or source_count != support.shape[1]:
                raise ValueError(
                    "M1 clean source, clean target, and support token counts "
                    "must match"
                )
            source_end = int(source_cache["local_end_index"].item())
            target_end = int(target_cache["local_end_index"].item())
            source_value = source_cache["v"][
                :, source_end - source_count:source_end
            ]
            target_value = target_cache["v"][
                :, target_end - target_count:target_end
            ]
            source_key = source_keys[layer]
            if source_key.shape != source_value.shape:
                raise ValueError(
                    f"M1 key/value shape mismatch at layer {layer}: "
                    f"{tuple(source_key.shape)} != {tuple(source_value.shape)}"
                )
            delta_value = (
                target_value.detach().float()
                - source_value.detach().float()
            ).to(dtype=target_value.dtype)
            compact_key = source_key.new_zeros(
                (source_key.shape[0], max_slots)
                + tuple(source_key.shape[2:])
            )
            compact_delta = delta_value.new_zeros(
                (delta_value.shape[0], max_slots)
                + tuple(delta_value.shape[2:])
            )
            compact_support = torch.zeros(
                (support.shape[0], max_slots),
                dtype=torch.bool,
                device=support.device,
            )
            for batch_index in range(support.shape[0]):
                selected = torch.nonzero(
                    support[batch_index], as_tuple=False
                ).flatten()
                count = int(selected.numel())
                if count == 0:
                    continue
                compact_key[batch_index, :count] = (
                    source_key[batch_index, selected].detach()
                )
                compact_delta[batch_index, :count] = (
                    delta_value[batch_index, selected].detach()
                )
                compact_support[batch_index, :count] = True
            state = ImmutableDeltaVLayerState(
                source_key=compact_key.clone(),
                delta_value=compact_delta.clone(),
                support=compact_support.clone(),
            )
            state.validate()
            states[layer] = state

        if not states:
            raise RuntimeError("M1 freeze produced no layer state")
        self._states = states
        support_fraction = support.float().mean()
        delta_rms_rows = []
        for state in states.values():
            valid_delta = state.delta_value[state.support]
            delta_rms_rows.append(
                valid_delta.float().square().mean().sqrt()
            )
        delta_rms = torch.stack(delta_rms_rows).mean()
        return {
            "layers": torch.tensor(float(len(states))),
            "support_fraction": support_fraction,
            "support_tokens": support_counts.float().mean(),
            "bank_slots": torch.tensor(float(max_slots)),
            "delta_value_rms": delta_rms,
        }

    def export(self) -> Dict[int, Dict[str, torch.Tensor]]:
        return {
            layer: {
                "source_key": state.source_key,
                "delta_value": state.delta_value,
                "support": state.support,
            }
            for layer, state in self._states.items()
        }
