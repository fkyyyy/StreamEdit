from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict

import torch
import torch.nn.functional as F

from .target_identity_memory import (
    CausalIdentityOwnerTracker,
    build_oracle_source_owner_weight,
)


class CausalOwnershipState(IntEnum):
    """Lifecycle of a source-coordinate object owner."""

    UNINITIALIZED = 0
    VISIBLE = 1
    OCCLUDED = 2
    ABSENT = 3


@dataclass(frozen=True)
class CausalObjectOwnership:
    """Training-free object ownership predicted before denoising."""

    owner_weight: torch.Tensor
    owner_support: torch.Tensor
    transported_weight: torch.Tensor
    observation_weight: torch.Tensor
    match_similarity: torch.Tensor
    match_confidence: torch.Tensor
    semantic_support: torch.Tensor
    state_code: torch.Tensor
    missing_frames: torch.Tensor
    diagnostics: Dict[str, torch.Tensor] | None = None

    def validate(self) -> None:
        if self.owner_weight.ndim != 2:
            raise ValueError(
                "Causal owner weights must have shape [B,L]"
            )
        token_shape = self.owner_weight.shape
        for name, value in (
            ("owner_support", self.owner_support),
            ("transported_weight", self.transported_weight),
            ("observation_weight", self.observation_weight),
            ("match_similarity", self.match_similarity),
            ("match_confidence", self.match_confidence),
            ("semantic_support", self.semantic_support),
        ):
            if value.shape != token_shape:
                raise ValueError(
                    f"Causal ownership '{name}' must match [B,L]"
                )
        if self.state_code.ndim != 2:
            raise ValueError(
                "Causal ownership state must have shape [B,T]"
            )
        if self.missing_frames.shape != self.state_code.shape:
            raise ValueError(
                "Causal ownership missing-frame count must match state"
            )
        for name, value in (
            ("owner_weight", self.owner_weight),
            ("transported_weight", self.transported_weight),
            ("observation_weight", self.observation_weight),
            ("match_confidence", self.match_confidence),
            ("semantic_support", self.semantic_support),
        ):
            if not torch.isfinite(value.float()).all():
                raise ValueError(
                    f"Causal ownership '{name}' is not finite"
                )
            if value.min() < 0 or value.max() > 1:
                raise ValueError(
                    f"Causal ownership '{name}' must lie in [0, 1]"
                )
        if (
            self.match_similarity.min() < -1
            or self.match_similarity.max() > 1
        ):
            raise ValueError(
                "Causal ownership similarity must lie in [-1, 1]"
            )

    def as_debug_maps(
        self,
        shape: tuple[int, int, int, int],
    ) -> Dict[str, torch.Tensor]:
        batch, frames, height, width = shape
        if self.owner_weight.shape != (
            batch,
            frames * height * width,
        ):
            raise ValueError(
                "Debug shape does not match causal owner tokens"
            )

        def reshape(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(shape).float()

        state = self.state_code[:, :, None, None].expand(shape)
        missing = self.missing_frames[:, :, None, None].expand(shape)
        maps = {
            "causal_owner_weight": reshape(self.owner_weight),
            "causal_owner_support": reshape(
                self.owner_support.float()
            ),
            "causal_owner_transport": reshape(
                self.transported_weight
            ),
            "causal_owner_observation": reshape(
                self.observation_weight
            ),
            "causal_owner_similarity": reshape(
                self.match_similarity
            ),
            "causal_owner_confidence": reshape(
                self.match_confidence
            ),
            "causal_owner_semantic": reshape(
                self.semantic_support
            ),
            "causal_owner_state": state.float(),
            "causal_owner_missing_frames": missing.float(),
        }
        if self.diagnostics is not None:
            for name, value in self.diagnostics.items():
                if value.shape != self.owner_weight.shape:
                    raise ValueError(
                        f"Causal ownership diagnostic '{name}' must match [B,L]"
                    )
                maps[name] = reshape(value)
        return maps


@dataclass(frozen=True)
class TransactionalOwnerSupport:
    """Separate conservative writes from contact/occlusion reads.

    ``write_weight`` is the visible non-hand core used by the existing
    memory commit. ``read_weight`` may additionally include source-labelled
    contact pixels and a short source-feature-transported lifecycle.  The
    latter two supports are read-only so uncertain hand/object boundaries
    can recover an existing target identity without contaminating it.
    """

    read_weight: torch.Tensor
    write_weight: torch.Tensor
    contact_weight: torch.Tensor
    lifecycle_weight: torch.Tensor
    missing_observation_frames: torch.Tensor
    diagnostics: Dict[str, torch.Tensor] | None = None

    def validate(self) -> None:
        expected = self.read_weight.shape
        if len(expected) != 2:
            raise ValueError(
                "Transactional owner weights must have shape [B,L]"
            )
        for name, value in (
            ("write_weight", self.write_weight),
            ("contact_weight", self.contact_weight),
            ("lifecycle_weight", self.lifecycle_weight),
        ):
            if value.shape != expected:
                raise ValueError(
                    f"Transactional owner '{name}' must match [B,L]"
                )
            if not torch.isfinite(value.float()).all():
                raise ValueError(
                    f"Transactional owner '{name}' is not finite"
                )
            if value.min() < 0 or value.max() > 1:
                raise ValueError(
                    f"Transactional owner '{name}' must lie in [0, 1]"
                )
        if not torch.isfinite(self.read_weight.float()).all():
            raise ValueError(
                "Transactional owner 'read_weight' is not finite"
            )
        if self.read_weight.min() < 0 or self.read_weight.max() > 1:
            raise ValueError(
                "Transactional owner 'read_weight' must lie in [0, 1]"
            )
        if self.missing_observation_frames.ndim != 2:
            raise ValueError(
                "Missing observation counts must have shape [B,T]"
            )
        if self.diagnostics is not None:
            for name, value in self.diagnostics.items():
                if value.shape != expected:
                    raise ValueError(
                        f"Transactional owner diagnostic '{name}' must "
                        "match [B,L]"
                    )
                if not torch.isfinite(value.float()).all():
                    raise ValueError(
                        f"Transactional owner diagnostic '{name}' is "
                        "not finite"
                    )

    def as_debug_maps(
        self, shape: tuple[int, int, int, int]
    ) -> Dict[str, torch.Tensor]:
        batch, frames, height, width = shape
        if self.read_weight.shape != (
            batch, frames * height * width
        ):
            raise ValueError(
                "Debug shape does not match transactional owner tokens"
            )

        def reshape(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(shape).float()

        maps = {
            "native_owner_read": reshape(self.read_weight),
            "native_owner_write": reshape(self.write_weight),
            "native_owner_contact_read": reshape(
                self.contact_weight
            ),
            "native_owner_lifecycle_read": reshape(
                self.lifecycle_weight
            ),
            "native_owner_missing_observation_frames": (
                self.missing_observation_frames[:, :, None, None]
                .expand(shape).float()
            ),
        }
        if self.diagnostics is not None:
            maps.update({
                f"native_owner_{name}": reshape(value)
                for name, value in self.diagnostics.items()
            })
        return maps


def build_motion_owner_read_weight(
    ownership: CausalObjectOwnership,
    transaction: TransactionalOwnerSupport,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover a high-recall read request without widening KV writes.

    The transactional core intentionally demands agreement from the role and
    velocity experts.  That is appropriate for writes, but using the same
    sparse support for reads makes a correctly transported geometry owner lose
    its target appearance.  Motion ownership supplies only a *request* here;
    the native-history attention still has to verify a clean-source address
    before any target K/V is consumed.

    Returns the recovered read weight and the read-only increment, both in the
    canonical flattened ``[B,L]`` owner layout.
    """
    ownership.validate()
    transaction.validate()
    if transaction.read_weight.shape != ownership.owner_weight.shape:
        raise ValueError(
            "Motion owner and transactional read weights must align"
        )
    geometry_request = (
        ownership.owner_weight.detach().float()
        * ownership.owner_support.detach().float()
    ).clamp(0.0, 1.0)
    base_read = transaction.read_weight.detach().float().clamp(0.0, 1.0)
    recovered = torch.maximum(base_read, geometry_request)
    increment = (recovered - base_read).clamp(0.0, 1.0)
    return recovered, increment


def _enclosed_owner_holes(owner_support: torch.Tensor) -> torch.Tensor:
    """Return background cells not connected to a frame boundary.

    This is a pure-torch, eight-connected flood fill. Eight-connectivity is
    deliberately conservative: a diagonal path to the image exterior keeps a
    cell classified as exterior, so this operation can never behave like an
    ordinary dilation around an open owner contour.
    """
    if owner_support.ndim != 4:
        raise ValueError(
            "Owner topology support must have shape [B,T,H,W]"
        )
    background = ~owner_support.detach().bool()
    exterior = torch.zeros_like(background)
    exterior[..., 0, :] = background[..., 0, :]
    exterior[..., -1, :] = background[..., -1, :]
    exterior[..., :, 0] |= background[..., :, 0]
    exterior[..., :, -1] |= background[..., :, -1]
    flat_background = background.flatten(0, 1)[:, None].float()
    flat_exterior = exterior.flatten(0, 1)[:, None]
    # The small latent grids used here make a converged flood fill cheaper
    # than adding a runtime dependency on a CPU connected-component package.
    max_iterations = owner_support.shape[-2] * owner_support.shape[-1]
    for _ in range(max_iterations):
        expanded = (
            torch.nn.functional.max_pool2d(
                flat_exterior.float(), kernel_size=3, stride=1, padding=1
            )
            > 0
        ) & flat_background.bool()
        if torch.equal(expanded, flat_exterior):
            break
        flat_exterior = expanded
    exterior = flat_exterior[:, 0].reshape_as(background)
    return background & ~exterior


def build_topology_complete_motion_owner_read_weight(
    ownership: CausalObjectOwnership,
    transaction: TransactionalOwnerSupport,
    *,
    shape: tuple[int, int, int, int],
    hand_exclusion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fill only enclosed read holes using independent automatic evidence.

    Geometry provides the closed contour, while source semantics and clean-
    source flow affinity provide the read strength. The external hand mask is
    used only as an exclusion. The conservative transactional write tensor is
    never modified.
    """
    ownership.validate()
    transaction.validate()
    batch, frames, height, width = shape
    if ownership.owner_weight.shape != (batch, frames * height * width):
        raise ValueError(
            "Topology shape must align with flattened causal ownership"
        )
    if hand_exclusion.shape != shape:
        raise ValueError(
            "Hand exclusion must align with owner topology shape"
        )
    if (
        ownership.diagnostics is None
        or "motion_hand_affinity" not in ownership.diagnostics
    ):
        raise ValueError(
            "Topology-complete reads require clean-source motion affinity"
        )

    base_read, motion_increment = build_motion_owner_read_weight(
        ownership, transaction
    )
    owner_support = ownership.owner_support.reshape(shape)
    enclosed_holes = _enclosed_owner_holes(owner_support)
    source_semantic = ownership.semantic_support.reshape(shape).float().clamp(
        0.0, 1.0
    )
    motion_affinity = ownership.diagnostics[
        "motion_hand_affinity"
    ].reshape(shape).float().clamp(0.0, 1.0)
    evidence = torch.sqrt(source_semantic * motion_affinity)
    if transaction.diagnostics is not None:
        field_agreement = transaction.diagnostics.get(
            "automatic_field_agreement"
        )
        if field_agreement is not None:
            evidence = evidence * field_agreement.reshape(shape).float().clamp(
                0.0, 1.0
            )
    evidence = evidence * (1.0 - hand_exclusion.detach().float().clamp(0, 1))
    topology_request = (enclosed_holes.float() * evidence).reshape(
        batch, -1
    )
    recovered = torch.maximum(base_read, topology_request)
    topology_increment = (recovered - base_read).clamp(0.0, 1.0)
    return (
        recovered,
        motion_increment,
        topology_increment,
        enclosed_holes.reshape(batch, -1),
    )


class CausalReadOnlyOwnerTracker:
    """Complete an oracle owner only for source-addressed KV reads.

    The exact full source mask supplies contact pixels that were removed
    from the write-safe core by hand exclusion.  When that observation is
    briefly missing, clean-source feature correspondence may carry it for a
    bounded number of latent frames.  Neither extension is ever committed
    as canonical target memory.
    """

    def __init__(
        self,
        *,
        max_candidates: int = 512,
        max_area_fraction: float = 0.18,
        min_similarity: float = 0.55,
        max_missing_frames: int = 2,
        eps: float = 1e-6,
    ):
        if max_missing_frames < 0:
            raise ValueError(
                "max_missing_frames must be non-negative"
            )
        self.max_missing_frames = int(max_missing_frames)
        self.eps = float(eps)
        self._missing_count: torch.Tensor | None = None
        self.transport = CausalIdentityOwnerTracker(
            max_candidates=max_candidates,
            max_area_fraction=max_area_fraction,
            min_similarity=min_similarity,
            recover_visibility_from_source_match=True,
            eps=eps,
        )

    @torch.no_grad()
    def __call__(
        self,
        *,
        source_features: torch.Tensor,
        full_owner_weight: torch.Tensor,
        core_owner_weight: torch.Tensor,
        source_semantic: torch.Tensor,
        hand_proximity: torch.Tensor,
        tokens_per_frame: int,
        spatial_shape: tuple[int, int],
    ) -> TransactionalOwnerSupport:
        expected = source_features.shape[:2]
        for name, value in (
            ("full_owner_weight", full_owner_weight),
            ("core_owner_weight", core_owner_weight),
            ("source_semantic", source_semantic),
            ("hand_proximity", hand_proximity),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {tuple(expected)}")
        if tokens_per_frame <= 0 or expected[1] % tokens_per_frame:
            raise ValueError(
                "tokens_per_frame must evenly divide source tokens"
            )
        batch = expected[0]
        frames = expected[1] // tokens_per_frame
        full = full_owner_weight.detach().float().clamp(0.0, 1.0)
        core = core_owner_weight.detach().float().clamp(0.0, 1.0)
        observed = (
            full.reshape(batch, frames, tokens_per_frame)
            > self.eps
        ).any(dim=-1)

        # The transport call predicts a read support and normally advances
        # its own state.  Transactional ownership has a stricter contract:
        # only the complete, externally observed source owner may advance
        # the address track.  Snapshot and restore before committing the last
        # verified observation so lifecycle-only predictions cannot drift.
        previous_features = self.transport.previous_features
        previous_weight = self.transport.previous_weight
        tracked = self.transport(
            source_features=source_features,
            observation_weight=full,
            source_semantic=source_semantic,
            hand_mask=torch.zeros_like(full, dtype=torch.bool),
            presence_support=hand_proximity,
            tokens_per_frame=tokens_per_frame,
            frame_visible=observed,
            spatial_shape=spatial_shape,
        )
        self.transport.previous_features = previous_features
        self.transport.previous_weight = previous_weight
        if observed.any():
            self.transport.commit_verified(
                source_features=source_features,
                verified_weight=full,
                tokens_per_frame=tokens_per_frame,
            )

        missing = (
            torch.zeros(batch, dtype=torch.long, device=full.device)
            if self._missing_count is None
            else self._missing_count.to(full.device).clone()
        )
        missing_frames = []
        lifecycle_frames = []
        transported = tracked.read_weight.reshape(
            batch, frames, tokens_per_frame
        )
        full_frames = full.reshape(batch, frames, tokens_per_frame)
        for frame_index in range(frames):
            missing = torch.where(
                observed[:, frame_index],
                torch.zeros_like(missing),
                missing + 1,
            )
            allow = (
                ~observed[:, frame_index]
                & (missing <= self.max_missing_frames)
            )
            lifecycle_frames.append(
                transported[:, frame_index] * allow[:, None].float()
            )
            missing_frames.append(missing.clone())
        self._missing_count = missing.detach().cpu()

        lifecycle = torch.stack(lifecycle_frames, dim=1).reshape(expected)
        # A write is legal only where both the complete source observation
        # and the conservative non-hand core agree.  This also protects the
        # contract when the upstream causal tracker has transported a core
        # beyond the current visible source matte.
        write = torch.minimum(core, full)
        contact = (full - write).clamp(0.0, 1.0)
        read = torch.maximum(full, lifecycle)
        result = TransactionalOwnerSupport(
            read_weight=read,
            write_weight=write,
            contact_weight=contact,
            lifecycle_weight=lifecycle,
            missing_observation_frames=torch.stack(
                missing_frames, dim=1
            ),
        )
        result.validate()
        return result


class AutomaticTransactionalOwnerTracker:
    """Build read/write ownership from hand-conditioned evidence only.

    This tracker deliberately consumes no object matte.  The persistent
    source-coordinate owner is supplied by :class:`CausalObjectOwnershipTracker`;
    hand-conditioned role posteriors decide object core versus contact, and
    target/source velocity disagreement is used as an online consistency test.

    The transaction is asymmetric:

    * visible, non-hand, high-confidence object cores may update target KV;
    * inferred contact/boundary and bounded source-transported lifecycle tokens
      may read existing target KV but can never update it;
    * unreliable or conflicting evidence abstains instead of expanding support.
    """

    def __init__(
        self,
        *,
        max_missing_frames: int = 2,
        blockwise_lifecycle: bool = False,
        eps: float = 1e-6,
    ):
        if max_missing_frames < 0:
            raise ValueError(
                "max_missing_frames must be non-negative"
            )
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.max_missing_frames = int(max_missing_frames)
        self.blockwise_lifecycle = bool(blockwise_lifecycle)
        self.eps = float(eps)
        self._missing_count: torch.Tensor | None = None

    def _frame_normalize(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:
        scale = value.flatten(2).amax(dim=-1, keepdim=True)
        return (
            value.flatten(2) / scale.clamp_min(self.eps)
        ).reshape_as(value).clamp(0.0, 1.0)

    @torch.no_grad()
    def __call__(
        self,
        *,
        ownership: CausalObjectOwnership,
        object_posterior: torch.Tensor,
        posterior_threshold: torch.Tensor,
        source_attention: torch.Tensor,
        hand_probability: torch.Tensor,
        hand_proximity: torch.Tensor,
        object_role: torch.Tensor,
        boundary_role: torch.Tensor,
        field_likelihood: torch.Tensor | None = None,
        field_reliability: torch.Tensor | None = None,
        update_state: bool = True,
    ) -> TransactionalOwnerSupport:
        ownership.validate()
        expected_map = object_posterior.shape
        if len(expected_map) != 4:
            raise ValueError(
                "Automatic owner evidence must have shape [B,T,H,W]"
            )
        for name, value in (
            ("source_attention", source_attention),
            ("hand_probability", hand_probability),
            ("hand_proximity", hand_proximity),
            ("object_role", object_role),
            ("boundary_role", boundary_role),
        ):
            if value.shape != expected_map:
                raise ValueError(
                    f"{name} must have shape {tuple(expected_map)}"
                )
        if posterior_threshold.shape not in {
            expected_map,
            (expected_map[0], expected_map[1], 1, 1),
        }:
            raise ValueError(
                "posterior_threshold must broadcast over [B,T,H,W]"
            )
        if ownership.owner_weight.numel() != object_posterior.numel():
            raise ValueError(
                "Automatic role maps and causal ownership must align"
            )
        for name, value in (
            ("field_likelihood", field_likelihood),
            ("field_reliability", field_reliability),
        ):
            if value is not None and value.shape not in {
                expected_map,
                (expected_map[0], expected_map[1], 1, 1),
            }:
                raise ValueError(
                    f"{name} must broadcast over [B,T,H,W]"
                )

        posterior = object_posterior.detach().float().clamp(0.0, 1.0)
        threshold = posterior_threshold.detach().float().clamp(
            self.eps, 1.0
        )
        selected = posterior >= threshold
        posterior_confidence = self._frame_normalize(posterior)
        posterior_confidence = posterior_confidence * selected.float()
        semantic_confidence = self._frame_normalize(
            source_attention.detach().float().clamp(0.0, 1.0)
        )
        object_confidence = self._frame_normalize(
            object_role.detach().float().clamp(0.0, 1.0)
        )
        boundary_confidence = self._frame_normalize(
            boundary_role.detach().float().clamp(0.0, 1.0)
        )
        hand = hand_probability.detach().float().clamp(0.0, 1.0)
        proximity = hand_proximity.detach().float().clamp(0.0, 1.0)

        if field_likelihood is None:
            likelihood = torch.ones_like(posterior)
            reliability = torch.zeros_like(posterior)
            flow_support = torch.zeros_like(posterior)
        else:
            likelihood = field_likelihood.detach().float().clamp(0.0, 1.0)
            if field_reliability is None:
                reliability = torch.ones_like(likelihood)
            else:
                reliability = field_reliability.detach().float().clamp(
                    0.0, 1.0
                )
            likelihood, reliability = torch.broadcast_tensors(
                likelihood, reliability
            )
            likelihood = likelihood.expand(expected_map)
            reliability = reliability.expand(expected_map)
            flow_support = (likelihood * reliability).clamp(0.0, 1.0)
        # A reliable contradiction vetoes a transaction.  An unreliable
        # field cannot create support and therefore leaves source evidence
        # unchanged instead of hallucinating a new owner.
        field_agreement = (
            1.0 - reliability * (1.0 - likelihood)
        ).clamp(0.0, 1.0)

        def owner_map(value: torch.Tensor) -> torch.Tensor:
            return value.detach().float().reshape(expected_map).clamp(
                0.0, 1.0
            )

        owner = owner_map(ownership.owner_weight)
        observation = owner_map(ownership.observation_weight)
        transported = owner_map(ownership.transported_weight)
        match_confidence = owner_map(ownership.match_confidence)
        source_semantic = owner_map(ownership.semantic_support)
        semantic_agreement = torch.maximum(
            semantic_confidence, source_semantic
        )
        non_hand = (1.0 - hand).clamp(0.0, 1.0)
        inferred_owner = torch.maximum(
            owner,
            posterior_confidence
            * semantic_agreement
            * flow_support
            * non_hand,
        ).clamp(0.0, 1.0)
        owner_confidence = self._frame_normalize(inferred_owner)

        role_core = torch.maximum(
            posterior_confidence, object_confidence
        )
        # The geometric agreement avoids collapsing several correlated role
        # probabilities by multiplying all of them twice.  It still requires
        # both source-coordinate ownership and an inferred object core.
        core_confidence = (
            torch.sqrt(owner_confidence * role_core)
            * (0.5 + 0.5 * semantic_agreement)
            * field_agreement
            * non_hand
            * (1.0 - boundary_confidence)
        ).clamp(0.0, 1.0)
        flow_verified_observation = (
            posterior_confidence
            * semantic_agreement
            * flow_support
            * non_hand
        ).clamp(0.0, 1.0)
        verified_observation = torch.maximum(
            observation, flow_verified_observation
        )
        observation_confidence = self._frame_normalize(
            verified_observation
        )
        observed = verified_observation.reshape(
            expected_map[0], expected_map[1], -1
        ).gt(self.eps).any(dim=-1)
        visible = ownership.state_code == int(
            CausalOwnershipState.VISIBLE
        )
        flow_visible = flow_verified_observation.reshape(
            expected_map[0], expected_map[1], -1
        ).gt(self.eps).any(dim=-1)
        write_frame = observed & (visible | flow_visible)
        write = (
            core_confidence
            * torch.sqrt(observation_confidence)
            * write_frame[:, :, None, None].float()
        ).clamp(0.0, 1.0)

        # Hand/object boundaries are intentionally read-only.  Requiring both
        # persistent source ownership and an inferred boundary prevents a
        # generic hand neighbourhood from becoming an object mask.
        contact = (
            owner_confidence
            * boundary_confidence
            * posterior_confidence
            * torch.sqrt(proximity.clamp_min(0.0))
            * (0.5 + 0.5 * semantic_agreement)
            * field_agreement
        ).clamp(0.0, 1.0)

        missing = ownership.missing_frames.to(owner.device)
        missing_count = (
            torch.zeros(
                expected_map[0], dtype=torch.long, device=owner.device
            )
            if self._missing_count is None
            else self._missing_count.to(owner.device).clone()
        )
        missing_frames = []
        lifecycle_frames = []
        if self.blockwise_lifecycle:
            # One tracker call corresponds to one generated causal block. A
            # three-frame latent block consumes one missing-observation budget.
            observed_in_block = observed.any(dim=1)
            missing_count = torch.where(
                observed_in_block,
                torch.zeros_like(missing_count),
                missing_count + 1,
            )
            allow_block_lifecycle = (
                ~observed_in_block
                & (missing_count <= self.max_missing_frames)
            )
            for frame_index in range(expected_map[1]):
                allow_lifecycle = (
                    ~observed[:, frame_index]
                    & (observed_in_block | allow_block_lifecycle)
                )
                lifecycle_frames.append(
                    transported[:, frame_index]
                    * match_confidence[:, frame_index]
                    * torch.maximum(
                        proximity[:, frame_index],
                        semantic_agreement[:, frame_index],
                    )
                    * field_agreement[:, frame_index]
                    * allow_lifecycle[:, None, None].float()
                )
                missing_frames.append(missing_count.clone())
        else:
            for frame_index in range(expected_map[1]):
                missing_count = torch.where(
                    observed[:, frame_index],
                    torch.zeros_like(missing_count),
                    missing_count + 1,
                )
                allow_lifecycle = (
                    ~observed[:, frame_index]
                    & (missing_count <= self.max_missing_frames)
                )
                lifecycle_frames.append(
                    transported[:, frame_index]
                    * match_confidence[:, frame_index]
                    * torch.maximum(
                        proximity[:, frame_index],
                        semantic_agreement[:, frame_index],
                    )
                    * field_agreement[:, frame_index]
                    * allow_lifecycle[:, None, None].float()
                )
                missing_frames.append(missing_count.clone())
        if update_state:
            self._missing_count = missing_count.detach().cpu()
        missing_frames = torch.stack(missing_frames, dim=1)
        lifecycle = (
            torch.stack(lifecycle_frames, dim=1)
        ).clamp(0.0, 1.0)

        # Visible transported cores may read even when the current detector
        # is weak, but only source correspondence can provide that support.
        core_read = (
            core_confidence
            * visible[:, :, None, None].float()
        ).clamp(0.0, 1.0)
        read = torch.maximum(
            torch.maximum(write, core_read),
            torch.maximum(contact, lifecycle),
        )
        uncertainty = (
            1.0
            - torch.maximum(
                torch.maximum(core_confidence, contact), lifecycle
            )
        ).clamp(0.0, 1.0)

        result = TransactionalOwnerSupport(
            read_weight=read.reshape_as(ownership.owner_weight),
            write_weight=write.reshape_as(ownership.owner_weight),
            contact_weight=contact.reshape_as(ownership.owner_weight),
            lifecycle_weight=lifecycle.reshape_as(
                ownership.owner_weight
            ),
            missing_observation_frames=missing_frames,
            diagnostics={
                "automatic_core_confidence": core_confidence.reshape_as(
                    ownership.owner_weight
                ),
                "automatic_field_agreement": field_agreement.reshape_as(
                    ownership.owner_weight
                ),
                "automatic_semantic_agreement": (
                    semantic_agreement.reshape_as(ownership.owner_weight)
                ),
                "automatic_uncertainty": uncertainty.reshape_as(
                    ownership.owner_weight
                ),
            },
        )
        result.validate()
        return result


class CausalObjectOwnershipTracker:
    """Maintain object ownership in clean-source coordinates.

    Text attention is used only to ignite the track.  Once an owner exists,
    clean-source query correspondence transports it independently of the
    current detector.  A missing match emits no owner (so it cannot ghost),
    while the last verified source state is retained for later re-association.
    """

    def __init__(
        self,
        *,
        max_candidates: int = 512,
        max_area_fraction: float = 0.18,
        local_radius: int = 10,
        max_area_growth: float = 1.15,
        min_similarity: float = 0.55,
        min_owner_weight: float = 0.05,
        max_occluded_frames: int = 3,
        eps: float = 1e-6,
    ):
        if not 0.0 <= min_owner_weight <= 1.0:
            raise ValueError(
                "min_owner_weight must lie in [0, 1]"
            )
        if max_occluded_frames < 0:
            raise ValueError(
                "max_occluded_frames must be non-negative"
            )
        self.min_owner_weight = float(min_owner_weight)
        self.max_occluded_frames = int(max_occluded_frames)
        self.eps = float(eps)
        self.transport = CausalIdentityOwnerTracker(
            max_candidates=max_candidates,
            max_area_fraction=max_area_fraction,
            local_radius=local_radius,
            max_area_growth=max_area_growth,
            min_similarity=min_similarity,
            recover_visibility_from_source_match=True,
            eps=eps,
        )
        self._missing_count: torch.Tensor | None = None
        # A causal signature of the edit response, not a 2-D displacement.
        # Clean-source queries transport ownership; this vector only verifies
        # that the transported token still reacts like the edited object.
        self._velocity_signature: torch.Tensor | None = None
        self._velocity_signature_live: torch.Tensor | None = None

    @staticmethod
    def _velocity_tokens(
        edit_response: torch.Tensor,
        spatial_shape: tuple[int, int],
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edit_response.ndim != 5:
            raise ValueError(
                "edit_response must have shape [B,T,C,H,W]"
            )
        batch, frames, channels = edit_response.shape[:3]
        height, width = spatial_shape
        vector = F.adaptive_avg_pool2d(
            edit_response.detach().float().reshape(
                batch * frames, channels, *edit_response.shape[-2:]
            ),
            output_size=spatial_shape,
        ).reshape(batch, frames, channels, height, width)
        vector = vector.permute(0, 1, 3, 4, 2).reshape(
            batch, frames * height * width, channels
        )
        magnitude = vector.square().mean(dim=-1).sqrt()
        frame_magnitude = magnitude.reshape(
            batch, frames, height * width
        )
        median = torch.quantile(
            frame_magnitude, 0.50, dim=-1, keepdim=True
        )
        high = torch.quantile(
            frame_magnitude, 0.95, dim=-1, keepdim=True
        )
        likelihood = (
            (frame_magnitude - median) / (high - median).clamp_min(eps)
        ).clamp(0.0, 1.0).reshape_as(magnitude)
        unit = F.normalize(vector, dim=-1, eps=eps)
        return unit, magnitude, likelihood

    @torch.no_grad()
    def refine_with_velocity(
        self,
        *,
        ownership: CausalObjectOwnership,
        source_features: torch.Tensor,
        edit_response: torch.Tensor,
        hand_mask: torch.Tensor,
        tokens_per_frame: int,
        spatial_shape: tuple[int, int],
        min_response: float = 0.10,
        min_signature_similarity: float = 0.0,
        transport_floor: float = 0.25,
        signature_momentum: float = 0.80,
        update_state: bool = True,
    ) -> CausalObjectOwnership:
        """Verify source-query transport with counterfactual velocity.

        ``edit_response`` is ``v_target - v_source`` at the first denoising
        step.  It is deliberately treated as a per-token causal signature,
        never as image-space flow.  The source-query matcher decides where a
        previous owner moved; response magnitude and signature agreement decide
        whether that correspondence may retain edit ownership.
        """
        ownership.validate()
        if not 0.0 <= min_response <= 1.0:
            raise ValueError("min_response must lie in [0, 1]")
        if not -1.0 < min_signature_similarity < 1.0:
            raise ValueError(
                "min_signature_similarity must lie in (-1, 1)"
            )
        if not 0.0 <= transport_floor <= 1.0:
            raise ValueError("transport_floor must lie in [0, 1]")
        if not 0.0 <= signature_momentum < 1.0:
            raise ValueError("signature_momentum must lie in [0, 1)")
        expected = source_features.shape[:2]
        if ownership.owner_weight.shape != expected:
            raise ValueError(
                "ownership and source_features must share [B,L]"
            )
        if hand_mask.shape != expected:
            raise ValueError("hand_mask must align with source tokens")
        if expected[1] % tokens_per_frame:
            raise ValueError(
                "tokens_per_frame must evenly divide source tokens"
            )
        batch = expected[0]
        frames = expected[1] // tokens_per_frame
        if edit_response.shape[:2] != (batch, frames):
            raise ValueError(
                "edit_response and ownership must share [B,T]"
            )
        if spatial_shape[0] * spatial_shape[1] != tokens_per_frame:
            raise ValueError(
                "spatial_shape must match tokens_per_frame"
            )

        unit, magnitude, response_likelihood = self._velocity_tokens(
            edit_response, spatial_shape, self.eps
        )
        hand = hand_mask.detach().bool()
        old_signature = self._velocity_signature
        old_signature_live = self._velocity_signature_live
        old_missing = self._missing_count
        previous_weight = self.transport.previous_weight
        previous_features = self.transport.previous_features
        previous_area_weight = (
            unit.new_zeros(batch, tokens_per_frame)
            if previous_weight is None
            else previous_weight.to(unit.device).float().clone()
        )
        normalized_source = F.normalize(
            source_features.detach().float(), dim=-1, eps=self.eps
        ).reshape(batch, frames, tokens_per_frame, -1)
        reference_features = (
            normalized_source[:, 0].clone()
            if previous_features is None
            else previous_features.to(unit.device).float().clone()
        )
        if reference_features.shape != (
            batch, tokens_per_frame, source_features.shape[-1]
        ):
            raise ValueError(
                "Source-query reference shape changed across blocks"
            )
        owner_live = (previous_area_weight > self.eps).any(dim=-1)
        signature = (
            unit.new_zeros(batch, unit.shape[-1])
            if old_signature is None
            else F.normalize(
                old_signature.to(unit.device).float(),
                dim=-1,
                eps=self.eps,
            )
        )
        signature_live = (
            torch.zeros(batch, dtype=torch.bool, device=unit.device)
            if old_signature_live is None
            else old_signature_live.to(unit.device).bool()
        )
        if signature.shape != (batch, unit.shape[-1]):
            raise ValueError(
                "Velocity signature shape changed across causal blocks"
            )
        if signature_live.shape != (batch,):
            raise ValueError(
                "Velocity signature batch size changed across calls"
            )

        unit_frames = unit.reshape(
            batch, frames, tokens_per_frame, -1
        )
        magnitude_frames = magnitude.reshape(
            batch, frames, tokens_per_frame
        )
        response_frames = response_likelihood.reshape(
            batch, frames, tokens_per_frame
        )
        hand_frames = hand.reshape(batch, frames, tokens_per_frame)
        observation_frames = ownership.observation_weight.float().reshape(
            batch, frames, tokens_per_frame
        )
        verified_frames = []
        support_frames = []
        ignition_frames = []
        verified_transport_frames = []
        query_similarity_frames = []
        query_confidence_frames = []
        query_cycle_frames = []
        response_active_frames = []
        signature_similarity_frames = []
        signature_confidence_frames = []
        signature_active_frames = []
        verification_frames = []
        missing = (
            torch.zeros(batch, dtype=torch.long, device=unit.device)
            if old_missing is None
            else old_missing.to(unit.device).clone()
        )
        state_frames = []
        missing_frames = []
        for frame_index in range(frames):
            (
                transported,
                query_similarity,
                query_confidence,
                query_cycle,
            ) = self.transport._transport(
                normalized_source[:, frame_index],
                reference_features,
                previous_area_weight,
                spatial_shape=spatial_shape,
            )
            current_unit = unit_frames[:, frame_index]
            current_response = response_frames[:, frame_index]
            current_hand = hand_frames[:, frame_index]
            current_similarity = torch.einsum(
                "bnc,bc->bn", current_unit, signature
            ).clamp(-1.0, 1.0)
            current_confidence = (
                (current_similarity - min_signature_similarity)
                / (1.0 - min_signature_similarity)
            ).clamp(0.0, 1.0)
            current_confidence = torch.where(
                signature_live[:, None],
                current_confidence,
                torch.ones_like(current_confidence),
            )
            response_active = current_response >= min_response
            signature_active = (
                ~signature_live[:, None]
                | (current_similarity >= min_signature_similarity)
            )
            verification = torch.sqrt(
                current_response * current_confidence
            )
            ignition = (
                observation_frames[:, frame_index]
                * current_response
                * response_active.float()
                * (~current_hand).float()
            ).clamp(0.0, 1.0)
            transported_gate = (
                transport_floor
                + (1.0 - transport_floor) * verification
            ) * response_active.float() * signature_active.float()
            verified_transport = (
                transported
                * transported_gate
                * (~current_hand).float()
            ).clamp(0.0, 1.0)
            # Once any frame has ignited an owner, all later frames in the
            # same chunk must use source-query transport.  This prevents the
            # per-frame semantic proposal from relocating ownership.
            candidate = torch.where(
                owner_live[:, None], verified_transport, ignition
            )
            verified = self.transport._bound_area(
                candidate, ignition, previous_area_weight
            )
            support = verified >= self.min_owner_weight
            verified = verified * support.float()
            present = support.any(dim=-1)

            # Update the causal response signature online so the first
            # verified frame already constrains later frames in this chunk.
            signature_weight = (
                verified * magnitude_frames[:, frame_index]
            )
            proposed_signature = (
                current_unit * signature_weight[:, :, None]
            ).sum(dim=1) / signature_weight.sum(
                dim=-1, keepdim=True
            ).clamp_min(self.eps)
            proposed_norm = proposed_signature.norm(dim=-1)
            proposed_signature = F.normalize(
                proposed_signature, dim=-1, eps=self.eps
            )
            valid_signature = present & (proposed_norm > self.eps)
            blended_signature = F.normalize(
                signature_momentum * signature
                + (1.0 - signature_momentum) * proposed_signature,
                dim=-1,
                eps=self.eps,
            )
            next_signature = torch.where(
                signature_live[:, None],
                blended_signature,
                proposed_signature,
            )
            signature = torch.where(
                valid_signature[:, None], next_signature, signature
            )
            signature_live = signature_live | valid_signature
            previous_area_weight = torch.where(
                present[:, None], verified, previous_area_weight
            )
            reference_features = torch.where(
                present[:, None, None],
                normalized_source[:, frame_index],
                reference_features,
            )
            owner_live = owner_live | present
            missing = torch.where(
                present, torch.zeros_like(missing), missing + 1
            )
            state = torch.full_like(
                missing, int(CausalOwnershipState.UNINITIALIZED)
            )
            state = torch.where(
                owner_live & ~present,
                torch.full_like(state, int(CausalOwnershipState.ABSENT)),
                state,
            )
            state = torch.where(
                owner_live
                & ~present
                & (missing <= self.max_occluded_frames),
                torch.full_like(state, int(CausalOwnershipState.OCCLUDED)),
                state,
            )
            state = torch.where(
                present,
                torch.full_like(state, int(CausalOwnershipState.VISIBLE)),
                state,
            )
            verified_frames.append(verified)
            support_frames.append(support)
            ignition_frames.append(ignition)
            verified_transport_frames.append(verified_transport)
            query_similarity_frames.append(query_similarity)
            query_confidence_frames.append(query_confidence)
            query_cycle_frames.append(query_cycle)
            response_active_frames.append(response_active.float())
            signature_similarity_frames.append(current_similarity)
            signature_confidence_frames.append(current_confidence)
            signature_active_frames.append(signature_active.float())
            verification_frames.append(verification)
            state_frames.append(state)
            missing_frames.append(missing.clone())

        def flatten(values: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack(values, dim=1).reshape(batch, -1)

        verified = flatten(verified_frames)
        support = flatten(support_frames).bool()
        ignition = flatten(ignition_frames)
        verified_transport = flatten(verified_transport_frames)
        query_similarity = flatten(query_similarity_frames)
        query_confidence = flatten(query_confidence_frames)
        query_cycle = flatten(query_cycle_frames)
        response_active = flatten(response_active_frames)
        signature_similarity = flatten(signature_similarity_frames)
        signature_confidence = flatten(signature_confidence_frames)
        signature_active = flatten(signature_active_frames)
        verification = flatten(verification_frames)

        if update_state:
            self.transport.commit_verified(
                source_features=source_features,
                verified_weight=verified,
                tokens_per_frame=tokens_per_frame,
            )
            self._velocity_signature = signature.detach().cpu()
            self._velocity_signature_live = (
                signature_live.detach().cpu()
            )
            self._missing_count = missing.detach().cpu()

        result = CausalObjectOwnership(
            owner_weight=verified,
            owner_support=support,
            transported_weight=verified_transport,
            observation_weight=ignition,
            match_similarity=query_similarity,
            match_confidence=query_confidence,
            semantic_support=ownership.semantic_support,
            state_code=torch.stack(state_frames, dim=1),
            missing_frames=torch.stack(missing_frames, dim=1),
            diagnostics={
                "velocity_owner_response_magnitude": magnitude,
                "velocity_owner_response_likelihood": response_likelihood,
                "velocity_owner_response_active": response_active.float(),
                "velocity_owner_signature_similarity": signature_similarity,
                "velocity_owner_signature_confidence": signature_confidence,
                "velocity_owner_signature_active": signature_active.float(),
                "velocity_owner_verification": verification,
                "velocity_owner_ignition": ignition,
                "velocity_owner_verified_transport": verified_transport,
                "velocity_owner_query_cycle_confidence": (
                    query_cycle
                ),
            },
        )
        result.validate()
        return result

    @torch.no_grad()
    def commit_verified(
        self,
        *,
        source_features: torch.Tensor,
        verified_weight: torch.Tensor,
        tokens_per_frame: int,
    ) -> None:
        """Replace the next-block source anchor with verified evidence.

        This is used after the first velocity observation.  Callers must pass
        only the conservative write core; contact and lifecycle support are
        intentionally excluded from persistent ownership.
        """
        if verified_weight.shape != source_features.shape[:2]:
            raise ValueError(
                "verified_weight must align with source features"
            )
        if source_features.shape[1] % tokens_per_frame:
            raise ValueError(
                "tokens_per_frame must evenly divide source tokens"
            )
        self.transport.commit_verified(
            source_features=source_features,
            verified_weight=verified_weight,
            tokens_per_frame=tokens_per_frame,
        )
        batch = source_features.shape[0]
        frames = source_features.shape[1] // tokens_per_frame
        observed = verified_weight.detach().reshape(
            batch, frames, tokens_per_frame
        ).gt(self.eps).any(dim=-1)
        missing = (
            torch.zeros(
                batch, dtype=torch.long, device=source_features.device
            )
            if self._missing_count is None
            else self._missing_count.to(source_features.device).clone()
        )
        for frame_index in range(frames):
            missing = torch.where(
                observed[:, frame_index],
                torch.zeros_like(missing),
                missing + 1,
            )
        self._missing_count = missing.detach().cpu()

    @torch.no_grad()
    def __call__(
        self,
        *,
        source_features: torch.Tensor,
        observation_weight: torch.Tensor,
        source_semantic: torch.Tensor,
        hand_mask: torch.Tensor,
        hand_proximity: torch.Tensor,
        tokens_per_frame: int,
        detector_visible: torch.Tensor,
        spatial_shape: tuple[int, int],
        update_state: bool = True,
    ) -> CausalObjectOwnership:
        expected = source_features.shape[:2]
        for name, value in (
            ("observation_weight", observation_weight),
            ("source_semantic", source_semantic),
            ("hand_mask", hand_mask),
            ("hand_proximity", hand_proximity),
        ):
            if value.shape != expected:
                raise ValueError(
                    f"{name} must have shape {tuple(expected)}"
                )
        if tokens_per_frame <= 0 or expected[1] % tokens_per_frame:
            raise ValueError(
                "tokens_per_frame must evenly divide source tokens"
            )
        frames = expected[1] // tokens_per_frame
        if detector_visible.numel() != expected[0] * frames:
            raise ValueError(
                "detector_visible must contain one value per frame"
            )

        previous_features = self.transport.previous_features
        previous_weight = self.transport.previous_weight
        previous_missing_count = self._missing_count
        had_anchor = (
            torch.zeros(
                expected[0],
                dtype=torch.bool,
                device=source_features.device,
            )
            if previous_weight is None
            else (
                previous_weight.to(source_features.device) > self.eps
            ).any(dim=-1)
        )
        tracked = self.transport(
            source_features=source_features,
            observation_weight=observation_weight,
            source_semantic=source_semantic,
            hand_mask=hand_mask,
            presence_support=hand_proximity,
            tokens_per_frame=tokens_per_frame,
            frame_visible=detector_visible,
            spatial_shape=spatial_shape,
        )
        support = tracked.read_weight >= self.min_owner_weight
        owner = tracked.read_weight * support.float()
        support_by_frame = support.reshape(
            expected[0], frames, tokens_per_frame
        ).any(dim=-1)

        if self._missing_count is None:
            missing = torch.zeros(
                expected[0],
                dtype=torch.long,
                device=source_features.device,
            )
        else:
            if self._missing_count.shape != (expected[0],):
                raise ValueError(
                    "Causal owner batch size changed across calls"
                )
            missing = self._missing_count.to(
                source_features.device
            ).clone()

        state_frames = []
        missing_frames = []
        anchor_live = had_anchor.clone()
        for frame_index in range(frames):
            present = support_by_frame[:, frame_index]
            anchor_live = anchor_live | present
            missing = torch.where(
                present,
                torch.zeros_like(missing),
                missing + 1,
            )
            state = torch.full_like(
                missing, int(CausalOwnershipState.UNINITIALIZED)
            )
            state = torch.where(
                anchor_live & ~present,
                torch.full_like(
                    state, int(CausalOwnershipState.ABSENT)
                ),
                state,
            )
            state = torch.where(
                anchor_live
                & ~present
                & (missing <= self.max_occluded_frames),
                torch.full_like(
                    state, int(CausalOwnershipState.OCCLUDED)
                ),
                state,
            )
            state = torch.where(
                present,
                torch.full_like(
                    state, int(CausalOwnershipState.VISIBLE)
                ),
                state,
            )
            state_frames.append(state)
            missing_frames.append(missing.clone())
        self._missing_count = missing.detach().cpu()

        result = CausalObjectOwnership(
            owner_weight=owner,
            owner_support=support,
            transported_weight=tracked.transported_weight,
            observation_weight=tracked.observation_weight,
            match_similarity=tracked.match_similarity,
            match_confidence=tracked.match_confidence,
            semantic_support=tracked.semantic_support,
            state_code=torch.stack(state_frames, dim=1),
            missing_frames=torch.stack(missing_frames, dim=1),
            diagnostics={
                "causal_owner_query_cycle_confidence": (
                    tracked.cycle_confidence
                ),
            },
        )
        result.validate()
        if not update_state:
            self.transport.previous_features = previous_features
            self.transport.previous_weight = previous_weight
            self._missing_count = previous_missing_count
        return result


@torch.no_grad()
def build_oracle_causal_ownership(
    *,
    source_owner_mask: torch.Tensor,
    hand_mask: torch.Tensor,
    spatial_shape: tuple[int, int],
    hand_already_excluded: bool = False,
) -> CausalObjectOwnership:
    """Expose an exact source mask through the causal-owner interface.

    This is an oracle ablation, not a deployable tracker.  Empty frames stay
    empty so that the experiment measures localization independently from
    target-memory quality without introducing temporal owner hallucination.
    """
    owner_map = build_oracle_source_owner_weight(
        source_owner_mask=source_owner_mask,
        hand_mask=hand_mask,
        spatial_shape=spatial_shape,
        hand_already_excluded=hand_already_excluded,
    )
    batch, frames, height, width = owner_map.shape
    owner_weight = owner_map.reshape(batch, -1)
    owner_support = owner_weight > 0.0
    visible = owner_map.flatten(2).any(dim=-1)
    state = torch.where(
        visible,
        torch.full_like(
            visible, int(CausalOwnershipState.VISIBLE), dtype=torch.long
        ),
        torch.full_like(
            visible, int(CausalOwnershipState.ABSENT), dtype=torch.long
        ),
    )
    missing = torch.zeros(
        batch, dtype=torch.long, device=owner_map.device
    )
    missing_frames = []
    for frame_index in range(frames):
        missing = torch.where(
            visible[:, frame_index],
            torch.zeros_like(missing),
            missing + 1,
        )
        missing_frames.append(missing.clone())
    result = CausalObjectOwnership(
        owner_weight=owner_weight,
        owner_support=owner_support,
        transported_weight=owner_weight,
        observation_weight=owner_weight,
        match_similarity=owner_weight.mul(2.0).sub(1.0),
        match_confidence=owner_weight,
        semantic_support=owner_weight,
        state_code=state,
        missing_frames=torch.stack(missing_frames, dim=1),
    )
    result.validate()
    return result
