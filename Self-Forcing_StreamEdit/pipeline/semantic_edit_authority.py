from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch


def apply_semantic_transaction_gate(
    transaction_weight: torch.Tensor,
    semantic_support: torch.Tensor,
) -> torch.Tensor:
    """Keep transaction confidence only on semantically authorized tokens.

    Semantic competition is a permission decision, not an additional
    confidence estimator.  Once a token wins the edit-vs-preserve contest,
    the automatic hand/flow transaction remains responsible for read/write
    strength.  This avoids multiplying two unrelated calibrated scores and
    accidentally pushing a valid request below the KV admission threshold.
    """
    if transaction_weight.ndim != 2 or semantic_support.ndim != 2:
        raise ValueError(
            "Semantic transaction inputs must have shape [B,L]"
        )
    if transaction_weight.shape != semantic_support.shape:
        raise ValueError(
            "Transaction weight and semantic support must share [B,L]"
        )
    transaction = transaction_weight.detach().float()
    if not torch.isfinite(transaction).all():
        raise ValueError("Transaction weight must be finite")
    if transaction.min() < 0 or transaction.max() > 1:
        raise ValueError("Transaction weight must lie in [0, 1]")
    permission = semantic_support.detach().bool()
    return transaction * permission.float()


def _framewise_quantile_normalize(
    value: torch.Tensor,
    *,
    low_quantile: float,
    high_quantile: float,
    eps: float,
) -> torch.Tensor:
    if value.ndim != 4:
        raise ValueError(
            "Semantic attention must have shape [B,T,H,W]"
        )
    flat = value.detach().float().flatten(2)
    low = torch.quantile(
        flat, low_quantile, dim=-1, keepdim=True
    )
    high = torch.quantile(
        flat, high_quantile, dim=-1, keepdim=True
    )
    normalized = (flat - low) / (high - low).clamp_min(eps)
    return normalized.clamp(0.0, 1.0).reshape_as(value)


def _normalize_group(
    groups: Mapping[str, torch.Tensor],
    *,
    low_quantile: float,
    high_quantile: float,
    eps: float,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if not groups:
        raise ValueError("Semantic attention group must not be empty")
    shapes = {tuple(value.shape) for value in groups.values()}
    if len(shapes) != 1:
        raise ValueError(
            "Semantic phrase attention maps must share one shape"
    )
    normalized = {
        name: _framewise_quantile_normalize(
            value,
            low_quantile=low_quantile,
            high_quantile=high_quantile,
            eps=eps,
        )
        for name, value in groups.items()
    }
    return torch.stack(list(normalized.values()), dim=0).amax(
        dim=0
    ), normalized


@dataclass(frozen=True)
class SemanticEditAuthority:
    """Prompt-derived local edit permission inside a tracked owner.

    Object ownership answers *which moving instance is being tracked*.  This
    class answers the orthogonal question *which part of that instance may be
    changed*.  The latter is a competition between target edit phrases and
    explicit preserve phrases, evaluated on the clean source latent.
    """

    edit_likelihood: torch.Tensor
    preserve_likelihood: torch.Tensor
    semantic_advantage: torch.Tensor
    owner_weight: torch.Tensor
    authority: torch.Tensor
    support: torch.Tensor
    edit_phrase_likelihoods: Mapping[str, torch.Tensor]
    preserve_phrase_likelihoods: Mapping[str, torch.Tensor]

    def as_debug_maps(self) -> Dict[str, torch.Tensor]:
        maps = {
            "target_edit_likelihood": self.edit_likelihood,
            "target_preserve_likelihood": self.preserve_likelihood,
            "target_semantic_advantage": self.semantic_advantage,
            "target_semantic_owner_weight": self.owner_weight,
            "target_edit_authority": self.authority,
            "target_edit_authority_support": self.support.float(),
        }
        maps.update({
            f"target_semantic_{name}": value
            for name, value in self.edit_phrase_likelihoods.items()
        })
        maps.update({
            f"target_semantic_{name}": value
            for name, value in self.preserve_phrase_likelihoods.items()
        })
        return maps


def build_semantic_edit_authority(
    *,
    edit_attention: Mapping[str, torch.Tensor],
    preserve_attention: Mapping[str, torch.Tensor],
    owner_weight: torch.Tensor,
    margin: float = 0.10,
    min_confidence: float = 0.20,
    low_quantile: float = 0.50,
    high_quantile: float = 0.95,
    eps: float = 1e-6,
) -> SemanticEditAuthority:
    """Build a calibrated edit-vs-preserve authority map.

    Each phrase is normalized independently per frame before a max reduction.
    This prevents a long list of background nouns from diluting a focused
    response such as ``screw cap``.  No externally supplied object mask is
    consumed: ``owner_weight`` is expected to be inferred causally from hand,
    source semantics, source features, and flow.
    """
    if not 0.0 <= margin < 1.0:
        raise ValueError("Semantic competition margin must lie in [0, 1)")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError(
            "Semantic authority confidence must lie in [0, 1]"
        )
    if not 0.0 <= low_quantile < high_quantile <= 1.0:
        raise ValueError(
            "Semantic quantiles must satisfy 0 <= low < high <= 1"
        )
    if eps <= 0:
        raise ValueError("eps must be positive")
    if owner_weight.ndim != 4:
        raise ValueError("Owner weight must have shape [B,T,H,W]")

    edit, edit_phrase_likelihoods = _normalize_group(
        edit_attention,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
        eps=eps,
    )
    preserve, preserve_phrase_likelihoods = _normalize_group(
        preserve_attention,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
        eps=eps,
    )
    if edit.shape != preserve.shape or edit.shape != owner_weight.shape:
        raise ValueError(
            "Edit, preserve, and owner maps must share [B,T,H,W]"
        )
    owner = owner_weight.detach().float().clamp(0.0, 1.0)
    advantage = (edit - preserve - float(margin)).clamp_min(0.0)
    advantage = (advantage / max(1.0 - float(margin), eps)).clamp(
        0.0, 1.0
    )
    authority = (advantage * owner).clamp(0.0, 1.0)
    support = authority >= float(min_confidence)
    # A below-threshold semantic tail is abstention, not weak permission.
    authority = authority * support.float()
    return SemanticEditAuthority(
        edit_likelihood=edit,
        preserve_likelihood=preserve,
        semantic_advantage=advantage,
        owner_weight=owner,
        authority=authority,
        support=support,
        edit_phrase_likelihoods=edit_phrase_likelihoods,
        preserve_phrase_likelihoods=preserve_phrase_likelihoods,
    )
