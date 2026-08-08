from dataclasses import dataclass
import math
from typing import Dict, Iterable, Mapping

import torch
import torch.nn.functional as F

from .control_belief import CausalControlBelief


@dataclass(frozen=True)
class TargetIdentityLayerState:
    """Slow, position-free target appearance prototypes for one layer."""

    key: torch.Tensor
    value: torch.Tensor
    evidence: torch.Tensor

    def validate(self) -> None:
        if self.key.ndim != 4:
            raise ValueError(
                "Identity keys must have shape [B,P,H,D]"
            )
        if self.value.shape != self.key.shape:
            raise ValueError(
                "Identity keys and values must share shape"
            )
        if self.evidence.shape != self.key.shape[:2]:
            raise ValueError(
                "Identity evidence must have shape [B,P]"
            )
        for name, value in (
            ("key", self.key),
            ("value", self.value),
            ("evidence", self.evidence),
        ):
            if not torch.isfinite(value.float()).all():
                raise ValueError(
                    f"Target identity state '{name}' is not finite"
                )
        if self.evidence.min() < 0:
            raise ValueError(
                "Target identity evidence must be non-negative"
            )


@dataclass(frozen=True)
class TargetIdentityUpdate:
    """Diagnostics for one slow identity-memory update."""

    write_weight: torch.Tensor
    observation_evidence: torch.Tensor
    update_gain: torch.Tensor
    accumulated_evidence: torch.Tensor

    def validate(self) -> None:
        if self.write_weight.ndim != 2:
            raise ValueError(
                "Identity write weights must have shape [B,L]"
            )
        layer_shapes = {
            tuple(value.shape)
            for value in (
                self.observation_evidence,
                self.update_gain,
                self.accumulated_evidence,
            )
        }
        if len(layer_shapes) != 1:
            raise ValueError(
                "Identity update diagnostics must share [layers,B,P]"
            )
        for name, value in (
            ("write_weight", self.write_weight),
            ("observation_evidence", self.observation_evidence),
            ("update_gain", self.update_gain),
            ("accumulated_evidence", self.accumulated_evidence),
        ):
            if not torch.isfinite(value.float()).all():
                raise ValueError(
                    f"Identity update '{name}' is not finite"
                )
            if value.min() < 0:
                raise ValueError(
                    f"Identity update '{name}' must be non-negative"
                )


class SlowTargetIdentityMemory:
    """Maintain reliable target appearance without spatially locking pose."""

    def __init__(
        self,
        layers: Iterable[int] = (8, 12, 16, 20),
        num_prototypes: int = 4,
        eps: float = 1e-6,
    ):
        self.layers = tuple(layers)
        if not self.layers:
            raise ValueError("Identity layers must not be empty")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("Identity layers must be unique")
        if any(layer < 0 for layer in self.layers):
            raise ValueError(
                "Identity layer indices must be non-negative"
            )
        if num_prototypes <= 0:
            raise ValueError(
                "num_prototypes must be positive"
            )
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.num_prototypes = num_prototypes
        self.eps = eps
        self.states: Dict[int, TargetIdentityLayerState] = {}

    @staticmethod
    def _descriptor(key: torch.Tensor) -> torch.Tensor:
        return F.normalize(
            key.float().flatten(2),
            dim=-1,
        )

    def _bootstrap_centers(
        self,
        descriptor: torch.Tensor,
        weight: torch.Tensor,
    ):
        batch, _, dim = descriptor.shape
        centers = descriptor.new_zeros(
            batch,
            self.num_prototypes,
            dim,
        )
        valid = torch.zeros(
            batch,
            self.num_prototypes,
            dtype=torch.bool,
            device=descriptor.device,
        )
        for batch_index in range(batch):
            candidates = torch.nonzero(
                weight[batch_index] > self.eps,
                as_tuple=False,
            ).flatten()
            if candidates.numel() == 0:
                continue
            selected = []
            first = candidates[
                weight[batch_index, candidates].argmax()
            ]
            selected.append(first)
            for slot in range(self.num_prototypes):
                if slot > 0:
                    if len(selected) >= candidates.numel():
                        break
                    selected_features = descriptor[
                        batch_index,
                        torch.stack(selected),
                    ]
                    similarity = torch.einsum(
                        "nd,pd->np",
                        descriptor[batch_index, candidates],
                        selected_features,
                    )
                    diversity = (
                        1.0 - similarity.max(dim=-1).values
                    ).clamp_min(0.0)
                    score = (
                        weight[batch_index, candidates] * diversity
                    )
                    used = torch.zeros_like(
                        candidates,
                        dtype=torch.bool,
                    )
                    for selected_index in selected:
                        used |= candidates == selected_index
                    score = score.masked_fill(used, -1.0)
                    selected.append(candidates[score.argmax()])
                centers[batch_index, slot] = descriptor[
                    batch_index,
                    selected[-1],
                ]
                valid[batch_index, slot] = True
        return centers, valid

    def _assign(
        self,
        descriptor: torch.Tensor,
        centers: torch.Tensor,
        center_valid: torch.Tensor,
    ) -> torch.Tensor:
        similarity = torch.einsum(
            "bld,bpd->blp",
            descriptor,
            F.normalize(centers.float(), dim=-1),
        ).clamp(-1.0, 1.0)
        flat_similarity = similarity.flatten(1)
        median = torch.quantile(
            flat_similarity,
            0.50,
            dim=-1,
            keepdim=True,
        )
        absolute_deviation = (
            flat_similarity - median
        ).abs()
        robust_scale = (
            torch.quantile(
                absolute_deviation,
                0.50,
                dim=-1,
                keepdim=True,
            )
            / 0.6745
        ).clamp_min(self.eps)
        logits = similarity / robust_scale.unsqueeze(-1)
        logits = logits.masked_fill(
            ~center_valid[:, None, :],
            torch.finfo(logits.dtype).min,
        )
        assignment = torch.softmax(logits, dim=-1)
        has_center = center_valid.any(dim=-1, keepdim=True)
        return torch.where(
            has_center.unsqueeze(-1),
            assignment,
            torch.zeros_like(assignment),
        )

    def _observe(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        weight: torch.Tensor,
        state: TargetIdentityLayerState | None,
    ):
        descriptor = self._descriptor(key)
        if state is None:
            centers, center_valid = self._bootstrap_centers(
                descriptor,
                weight,
            )
        else:
            centers = self._descriptor(state.key)
            center_valid = state.evidence > self.eps
            missing = ~center_valid.any(dim=-1)
            if missing.any():
                bootstrap_centers, bootstrap_valid = (
                    self._bootstrap_centers(
                        descriptor,
                        weight,
                    )
                )
                centers = torch.where(
                    missing[:, None, None],
                    bootstrap_centers,
                    centers,
                )
                center_valid = torch.where(
                    missing[:, None],
                    bootstrap_valid,
                    center_valid,
                )
        assignment = self._assign(
            descriptor,
            centers,
            center_valid,
        )
        responsibility = assignment * weight.unsqueeze(-1)
        mass = responsibility.sum(dim=1)
        normalizer = mass.clamp_min(self.eps)
        observation_key = torch.einsum(
            "blp,blhd->bphd",
            responsibility,
            key.float(),
        ) / normalizer[:, :, None, None]
        observation_value = torch.einsum(
            "blp,blhd->bphd",
            responsibility,
            value.float(),
        ) / normalizer[:, :, None, None]

        support = (weight > self.eps).float()
        assignment_support = (
            assignment * support.unsqueeze(-1)
        ).sum(dim=1)
        observation_evidence = (
            mass / assignment_support.clamp_min(self.eps)
        )
        observation_evidence = torch.where(
            mass > self.eps,
            observation_evidence,
            torch.zeros_like(observation_evidence),
        ).clamp(0.0, 1.0)
        observation_key = F.normalize(
            observation_key,
            dim=-1,
        ) * math.sqrt(key.shape[-1])
        return (
            observation_key,
            observation_value,
            observation_evidence,
        )

    @torch.no_grad()
    def update(
        self,
        kv_cache,
        write_weight: torch.Tensor,
    ) -> TargetIdentityUpdate:
        if write_weight.ndim != 2:
            raise ValueError(
                "Identity write_weight must have shape [B,L]"
            )
        write_weight = write_weight.detach().float().clamp(
            0.0,
            1.0,
        )
        observation_evidence = []
        update_gains = []
        accumulated_evidence = []
        for layer in self.layers:
            cache = kv_cache[layer]
            num_new_tokens = cache.get("num_new_tokens")
            local_end = cache["local_end_index"].item()
            if num_new_tokens != write_weight.shape[1]:
                raise ValueError(
                    "Identity write weights and target KV must align: "
                    f"{write_weight.shape[1]} != {num_new_tokens}"
                )
            key = cache["k"][
                :,
                local_end - num_new_tokens:local_end,
            ]
            value = cache["v"][
                :,
                local_end - num_new_tokens:local_end,
            ]
            previous = self.states.get(layer)
            (
                observed_key,
                observed_value,
                observed_evidence,
            ) = self._observe(
                key,
                value,
                write_weight,
                previous,
            )
            if previous is None:
                old_evidence = torch.zeros_like(observed_evidence)
                old_key = torch.zeros_like(observed_key)
                old_value = torch.zeros_like(observed_value)
            else:
                old_evidence = previous.evidence.float()
                old_key = previous.key.float()
                old_value = previous.value.float()
            total_evidence = old_evidence + observed_evidence
            gain = torch.where(
                total_evidence > self.eps,
                observed_evidence
                / total_evidence.clamp_min(self.eps),
                torch.zeros_like(total_evidence),
            )
            new_key = (
                old_key
                + gain[:, :, None, None]
                * (observed_key - old_key)
            )
            new_key = F.normalize(
                new_key,
                dim=-1,
            ) * math.sqrt(key.shape[-1])
            new_value = (
                old_value
                + gain[:, :, None, None]
                * (observed_value - old_value)
            )
            valid = total_evidence > self.eps
            new_key = torch.where(
                valid[:, :, None, None],
                new_key,
                torch.zeros_like(new_key),
            )
            new_value = torch.where(
                valid[:, :, None, None],
                new_value,
                torch.zeros_like(new_value),
            )
            state = TargetIdentityLayerState(
                key=new_key.to(key.dtype).detach(),
                value=new_value.to(value.dtype).detach(),
                evidence=total_evidence.detach(),
            )
            state.validate()
            self.states[layer] = state
            observation_evidence.append(observed_evidence)
            update_gains.append(gain)
            accumulated_evidence.append(total_evidence)

        update = TargetIdentityUpdate(
            write_weight=write_weight,
            observation_evidence=torch.stack(
                observation_evidence,
                dim=0,
            ),
            update_gain=torch.stack(update_gains, dim=0),
            accumulated_evidence=torch.stack(
                accumulated_evidence,
                dim=0,
            ),
        )
        update.validate()
        return update

    def export(
        self,
    ) -> Mapping[int, TargetIdentityLayerState]:
        return self.states


def strengthen_belief_with_target_identity(
    belief: CausalControlBelief,
    identity_support: torch.Tensor,
    hand_mask: torch.Tensor,
    eps: float = 1e-6,
) -> CausalControlBelief:
    """Use reliable target identity as edit evidence, not as a new seed."""
    belief.validate()
    if identity_support.ndim != 4:
        raise ValueError(
            "Identity support must have shape [B,T,H,W]"
        )
    if hand_mask.shape != belief.edit_belief.shape:
        raise ValueError(
            "Hand mask and control belief must share shape"
        )
    if identity_support.shape[:2] != belief.edit_belief.shape[:2]:
        raise ValueError(
            "Identity support and belief must share [B,T]"
        )
    batch, frames = identity_support.shape[:2]
    target_size = belief.edit_belief.shape[-2:]
    support = F.interpolate(
        identity_support.float().reshape(
            batch * frames,
            1,
            *identity_support.shape[-2:],
        ),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).reshape(batch, frames, *target_size).clamp(0.0, 1.0)

    edit_belief = (
        1.0
        - (1.0 - belief.edit_belief)
        * (1.0 - support)
    ).clamp(0.0, 1.0)
    edit_precision = torch.maximum(
        belief.edit_precision,
        support,
    )
    preserve_release = support * (
        1.0 - hand_mask.float().clamp(0.0, 1.0)
    )
    preserve_belief = (
        belief.preserve_belief * (1.0 - preserve_release)
    ).clamp(0.0, 1.0)
    visibility = (
        1.0
        - (1.0 - belief.visibility) * (1.0 - support)
    ).clamp(0.0, 1.0)
    conflict = (edit_belief * preserve_belief).clamp(0.0, 1.0)
    responsibility = (
        edit_belief + preserve_belief
    ).clamp_min(eps)
    uncertainty = (
        edit_belief * (1.0 - edit_precision)
        + preserve_belief * (1.0 - belief.preserve_precision)
    ) / responsibility
    updated = CausalControlBelief(
        edit_belief=edit_belief.float(),
        preserve_belief=preserve_belief.float(),
        edit_precision=edit_precision.float(),
        preserve_precision=belief.preserve_precision.float(),
        visibility=visibility.float(),
        uncertainty=uncertainty.clamp(0.0, 1.0).float(),
        conflict=conflict.float(),
    )
    updated.validate()
    return updated
