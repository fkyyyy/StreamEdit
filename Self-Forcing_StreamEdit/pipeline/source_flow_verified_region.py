"""Conservative fusion of semantic proposals and causal flow ownership.

The semantic/token branch proposes object extent.  Clean-source motion
ownership verifies where that proposal is allowed to control editing, while
reliable background evidence can veto fringe responses.  No object mask or
generated-frame measurement is consumed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius < 0:
        raise ValueError("owner_radius must be non-negative")
    if radius == 0:
        return mask.bool()
    batch, frames, height, width = mask.shape
    flat = mask.reshape(batch * frames, 1, height, width).float()
    return (
        F.max_pool2d(
            flat,
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        )
        .reshape_as(mask)
        .bool()
    )


@dataclass(frozen=True)
class SourceFlowVerifiedRegion:
    """A flow-verified edit support and its auditable components."""

    posterior: torch.Tensor
    support: torch.Tensor
    semantic_proposal: torch.Tensor
    verified_semantic: torch.Tensor
    recovered_owner: torch.Tensor
    owner_neighborhood: torch.Tensor
    reliable_background_veto: torch.Tensor
    hand_exclusion: torch.Tensor

    def as_debug_maps(self) -> Dict[str, torch.Tensor]:
        return {
            "source_flow_region_proposal": self.semantic_proposal.float(),
            "source_flow_region_verified_semantic": (
                self.verified_semantic.float()
            ),
            "source_flow_region_recovered_owner": (
                self.recovered_owner.float()
            ),
            "source_flow_region_owner_neighborhood": (
                self.owner_neighborhood.float()
            ),
            "source_flow_region_background_veto": (
                self.reliable_background_veto.float()
            ),
            "source_flow_region_hand_exclusion": (
                self.hand_exclusion.float()
            ),
            "source_flow_verified_support": self.support.float(),
            "source_flow_verified_posterior": self.posterior,
        }


@torch.no_grad()
def build_source_flow_verified_region(
    *,
    object_posterior: torch.Tensor,
    posterior_threshold: torch.Tensor,
    owner_support: torch.Tensor,
    hand_exclusion: torch.Tensor,
    background_likelihood: torch.Tensor | None = None,
    flow_confidence: torch.Tensor | None = None,
    owner_radius: int = 1,
    background_veto_threshold: float = 0.55,
    background_veto_min_confidence: float = 0.50,
) -> SourceFlowVerifiedRegion:
    """Verify a token proposal using independent clean-source motion.

    A token is editable when it is either a causal owner token or a semantic
    proposal within ``owner_radius`` cells of that owner.  Reliable flow
    background vetoes only the semantic extension; the transported owner is
    retained as the recovery path.  Hard exclusion is applied last so no
    later union can silently re-authorize an excluded token.
    """
    if object_posterior.ndim != 4:
        raise ValueError(
            "object_posterior must have shape [B,T,H,W]"
        )
    shape = object_posterior.shape
    if posterior_threshold.shape not in {
        shape,
        (shape[0], shape[1], 1, 1),
    }:
        raise ValueError(
            "posterior_threshold must broadcast over [B,T,H,W]"
        )

    def align(name: str, value: torch.Tensor) -> torch.Tensor:
        if value.shape == shape:
            return value
        if value.shape == (shape[0], shape[1] * shape[2] * shape[3]):
            return value.reshape(shape)
        raise ValueError(f"{name} must align with {tuple(shape)}")

    if owner_radius < 0:
        raise ValueError("owner_radius must be non-negative")
    for name, value in (
        ("background_veto_threshold", background_veto_threshold),
        (
            "background_veto_min_confidence",
            background_veto_min_confidence,
        ),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if (background_likelihood is None) != (flow_confidence is None):
        raise ValueError(
            "background_likelihood and flow_confidence must be supplied "
            "together"
        )

    posterior = object_posterior.detach().float().clamp(0.0, 1.0)
    threshold = posterior_threshold.detach().float().clamp(0.0, 1.0)
    owner = align("owner_support", owner_support).detach().bool()
    hand = align("hand_exclusion", hand_exclusion).detach().float() > 0
    proposal = posterior >= threshold
    owner_neighborhood = _dilate(owner, owner_radius)

    if background_likelihood is None:
        background_veto = torch.zeros_like(proposal)
    else:
        background = align(
            "background_likelihood", background_likelihood
        ).detach().float().clamp(0.0, 1.0)
        confidence = align(
            "flow_confidence", flow_confidence
        ).detach().float().clamp(0.0, 1.0)
        background_veto = (
            (background >= float(background_veto_threshold))
            & (confidence >= float(background_veto_min_confidence))
        )

    verified_semantic = (
        proposal & owner_neighborhood & ~background_veto
    )
    recovered_owner = owner & ~verified_semantic
    support = (verified_semantic | recovered_owner) & ~hand

    # Preserve semantic confidence inside the verified extent.  Owner tokens
    # below the semantic threshold get the minimum editable confidence rather
    # than an arbitrary unit score.
    owner_floor = torch.maximum(posterior, threshold.expand_as(posterior))
    verified_posterior = torch.where(
        recovered_owner, owner_floor, posterior
    ) * support.float()

    return SourceFlowVerifiedRegion(
        posterior=verified_posterior,
        support=support,
        semantic_proposal=proposal,
        verified_semantic=verified_semantic,
        recovered_owner=recovered_owner,
        owner_neighborhood=owner_neighborhood,
        reliable_background_veto=background_veto,
        hand_exclusion=hand,
    )
