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


@dataclass(frozen=True)
class TargetIdentityTokenPropagation:
    """Causal token-match gating applied before identity-memory writes."""

    write_weight: torch.Tensor
    base_write_weight: torch.Tensor
    support_weight: torch.Tensor
    match_confidence: torch.Tensor
    best_similarity: torch.Tensor
    matched_previous_weight: torch.Tensor
    has_previous: torch.Tensor

    def validate(self) -> None:
        if self.write_weight.ndim != 2:
            raise ValueError(
                "Token propagation weights must have shape [B,L]"
            )
        expected_shape = self.write_weight.shape
        for name, value in (
            ("base_write_weight", self.base_write_weight),
            ("support_weight", self.support_weight),
            ("match_confidence", self.match_confidence),
            ("best_similarity", self.best_similarity),
            (
                "matched_previous_weight",
                self.matched_previous_weight,
            ),
        ):
            if value.shape != expected_shape:
                raise ValueError(
                    f"Token propagation '{name}' must match [B,L]"
                )
        if self.has_previous.shape != tuple(expected_shape[:1]) + (1,):
            raise ValueError(
                "Token propagation has_previous must have shape [B,1]"
            )
        for name, value in (
            ("write_weight", self.write_weight),
            ("base_write_weight", self.base_write_weight),
            ("support_weight", self.support_weight),
            ("match_confidence", self.match_confidence),
            ("best_similarity", self.best_similarity),
            (
                "matched_previous_weight",
                self.matched_previous_weight,
            ),
            ("has_previous", self.has_previous.float()),
        ):
            if not torch.isfinite(value.float()).all():
                raise ValueError(
                    f"Token propagation '{name}' is not finite"
                )
        for name, value in (
            ("write_weight", self.write_weight),
            ("base_write_weight", self.base_write_weight),
            ("support_weight", self.support_weight),
            ("match_confidence", self.match_confidence),
            (
                "matched_previous_weight",
                self.matched_previous_weight,
            ),
        ):
            if value.min() < 0 or value.max() > 1:
                raise ValueError(
                    f"Token propagation '{name}' must lie in [0, 1]"
                )
        if self.best_similarity.min() < -1 or self.best_similarity.max() > 1:
            raise ValueError(
                "Token propagation best_similarity must lie in [-1, 1]"
            )


class CausalObjectTokenPropagator:
    """Gate identity writes by matching current source tokens to prior ones."""

    def __init__(
        self,
        min_similarity: float = 0.55,
        gate_strength: float = 0.85,
        max_candidates: int = 512,
        eps: float = 1e-6,
    ):
        if not -1.0 < min_similarity < 1.0:
            raise ValueError("min_similarity must lie in (-1, 1)")
        if not 0.0 <= gate_strength <= 1.0:
            raise ValueError("gate_strength must lie in [0, 1]")
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.min_similarity = float(min_similarity)
        self.gate_strength = float(gate_strength)
        self.max_candidates = int(max_candidates)
        self.eps = float(eps)
        self.previous_features: torch.Tensor | None = None
        self.previous_weight: torch.Tensor | None = None

    def _candidate_indices(
        self,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        candidates = torch.nonzero(
            weight > self.eps,
            as_tuple=False,
        ).flatten()
        if candidates.numel() > self.max_candidates:
            selected = torch.topk(
                weight[candidates],
                k=self.max_candidates,
            ).indices
            candidates = candidates[selected]
        return candidates

    @torch.no_grad()
    def __call__(
        self,
        source_features: torch.Tensor,
        base_write_weight: torch.Tensor,
        support_weight: torch.Tensor | None = None,
    ) -> TargetIdentityTokenPropagation:
        if source_features.ndim != 3:
            raise ValueError(
                "Token propagation source_features must have shape [B,L,D]"
            )
        if base_write_weight.ndim != 2:
            raise ValueError(
                "Token propagation write weights must have shape [B,L]"
            )
        if source_features.shape[:2] != base_write_weight.shape:
            raise ValueError(
                "Token propagation features and weights must align"
            )

        current_features = F.normalize(
            source_features.detach().float(),
            dim=-1,
        )
        base_weight = base_write_weight.detach().float().clamp(0.0, 1.0)
        support = (
            base_weight
            if support_weight is None
            else support_weight.detach().float().clamp(0.0, 1.0)
        )
        if support.shape != base_weight.shape:
            raise ValueError(
                "Token propagation support_weight must match [B,L]"
            )
        write_weight = base_weight.clone()
        match_confidence = torch.ones_like(base_weight)
        best_similarity = torch.zeros_like(base_weight)
        matched_previous_weight = torch.zeros_like(base_weight)
        has_previous = torch.zeros(
            base_weight.shape[0],
            1,
            dtype=torch.bool,
            device=base_weight.device,
        )

        if (
            self.previous_features is not None
            and self.previous_weight is not None
        ):
            previous_features = self.previous_features.to(
                device=current_features.device,
            )
            previous_weight = self.previous_weight.to(
                device=base_weight.device,
            )
            if (
                previous_features.shape[0] != current_features.shape[0]
                or previous_features.shape[-1] != current_features.shape[-1]
                or previous_weight.shape[0] != base_weight.shape[0]
                or previous_features.shape[:2] != previous_weight.shape
            ):
                raise ValueError(
                    "Previous token propagation state is incompatible with "
                    "the current features"
                )
            for batch_index in range(base_weight.shape[0]):
                candidates = self._candidate_indices(
                    previous_weight[batch_index],
                )
                if candidates.numel() == 0:
                    continue
                has_previous[batch_index, 0] = True
                candidate_features = previous_features[
                    batch_index,
                    candidates,
                ]
                similarity = torch.matmul(
                    current_features[batch_index],
                    candidate_features.T,
                ).clamp(-1.0, 1.0)
                best, matched = similarity.max(dim=-1)
                matched_weight = previous_weight[
                    batch_index,
                    candidates[matched],
                ].clamp(0.0, 1.0)
                absolute_match = (
                    (best - self.min_similarity)
                    / (1.0 - self.min_similarity)
                ).clamp(0.0, 1.0)
                low = torch.quantile(best, 0.50)
                high = torch.quantile(best, 0.95)
                spread = (high - low).clamp_min(self.eps)
                relative_match = ((best - low) / spread).clamp(0.0, 1.0)
                relative_match = torch.where(
                    (high - low) > self.eps,
                    relative_match,
                    absolute_match,
                )
                confidence = torch.sqrt(
                    absolute_match * relative_match
                ).clamp(0.0, 1.0)
                propagated_weight = (
                    confidence * matched_weight
                ).clamp(0.0, 1.0)
                multiplier = (
                    1.0
                    - self.gate_strength
                    + self.gate_strength * propagated_weight
                )
                write_weight[batch_index] = (
                    base_weight[batch_index] * multiplier
                ).clamp(0.0, 1.0)
                match_confidence[batch_index] = confidence
                best_similarity[batch_index] = best
                matched_previous_weight[batch_index] = matched_weight

        self.previous_features = current_features.detach().cpu()
        self.previous_weight = support.detach().cpu()

        result = TargetIdentityTokenPropagation(
            write_weight=write_weight,
            base_write_weight=base_weight,
            support_weight=support,
            match_confidence=match_confidence,
            best_similarity=best_similarity,
            matched_previous_weight=matched_previous_weight,
            has_previous=has_previous,
        )
        result.validate()
        return result


@dataclass(frozen=True)
class ReferenceIdentityBootstrap:
    """Object evidence extracted from an aligned target reference."""

    write_weight: torch.Tensor
    change_score: torch.Tensor
    semantic_score: torch.Tensor
    joint_score: torch.Tensor
    hand_contact_score: torch.Tensor

    def validate(self) -> None:
        if self.write_weight.ndim != 2:
            raise ValueError(
                "Reference identity weights must have shape [B,L]"
            )
        if self.change_score.ndim != 4:
            raise ValueError(
                "Reference identity maps must have shape [B,T,H,W]"
            )
        shapes = {
            tuple(value.shape)
            for value in (
                self.change_score,
                self.semantic_score,
                self.joint_score,
                self.hand_contact_score,
            )
        }
        if len(shapes) != 1:
            raise ValueError(
                "Reference identity debug maps must share shape"
            )
        if self.write_weight.shape != (
            self.change_score.shape[0],
            math.prod(self.change_score.shape[1:]),
        ):
            raise ValueError(
                "Reference identity weights and maps must align"
            )
        for name, value in (
            ("write_weight", self.write_weight),
            ("change_score", self.change_score),
            ("semantic_score", self.semantic_score),
            ("joint_score", self.joint_score),
            ("hand_contact_score", self.hand_contact_score),
        ):
            if not torch.isfinite(value.float()).all():
                raise ValueError(
                    f"Reference identity '{name}' is not finite"
                )
            if value.min() < 0 or value.max() > 1:
                raise ValueError(
                    f"Reference identity '{name}' must lie in [0, 1]"
                )

    def as_debug_maps(self) -> Dict[str, torch.Tensor]:
        return {
            "reference_identity_change": self.change_score.float(),
            "reference_identity_semantic": self.semantic_score.float(),
            "reference_identity_joint": self.joint_score.float(),
            "reference_identity_hand_contact": (
                self.hand_contact_score.float()
            ),
            "reference_identity_write": self.write_weight.reshape_as(
                self.joint_score
            ).float(),
        }


def _largest_weighted_component_mask(
    weight: torch.Tensor,
    eps: float,
    hand_contact_score: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select one 8-connected edited instance per reference frame."""
    if weight.ndim != 4:
        raise ValueError(
            "Reference component weights must have shape [B,T,H,W]"
        )
    if (
        hand_contact_score is not None
        and hand_contact_score.shape != weight.shape
    ):
        raise ValueError(
            "Reference hand contact and component weights must align"
        )
    candidate = (weight.detach().float() > eps).cpu()
    cpu_weight = weight.detach().float().cpu()
    cpu_contact = (
        hand_contact_score.detach().float().cpu()
        if hand_contact_score is not None
        else torch.zeros_like(cpu_weight)
    )
    selected = torch.zeros_like(candidate)
    batch, frames, height, width = candidate.shape
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    for batch_index in range(batch):
        for frame_index in range(frames):
            visited = torch.zeros(
                (height, width),
                dtype=torch.bool,
            )
            best_component = []
            best_score = (-1.0, -1.0, -1)
            for row in range(height):
                for col in range(width):
                    if (
                        visited[row, col]
                        or not candidate[
                            batch_index,
                            frame_index,
                            row,
                            col,
                        ]
                    ):
                        continue
                    stack = [(row, col)]
                    visited[row, col] = True
                    component = []
                    mass = 0.0
                    contact_mass = 0.0
                    while stack:
                        current_row, current_col = stack.pop()
                        component.append(
                            (current_row, current_col)
                        )
                        mass += float(
                            cpu_weight[
                                batch_index,
                                frame_index,
                                current_row,
                                current_col,
                            ]
                        )
                        contact_mass += float(
                            cpu_weight[
                                batch_index,
                                frame_index,
                                current_row,
                                current_col,
                            ]
                            * cpu_contact[
                                batch_index,
                                frame_index,
                                current_row,
                                current_col,
                            ]
                        )
                        for row_offset, col_offset in neighbors:
                            next_row = current_row + row_offset
                            next_col = current_col + col_offset
                            if (
                                0 <= next_row < height
                                and 0 <= next_col < width
                                and not visited[next_row, next_col]
                                and candidate[
                                    batch_index,
                                    frame_index,
                                    next_row,
                                    next_col,
                                ]
                            ):
                                visited[next_row, next_col] = True
                                stack.append((next_row, next_col))
                    score = (
                        contact_mass,
                        mass,
                        len(component),
                    )
                    if score > best_score:
                        best_score = score
                        best_component = component
            for row, col in best_component:
                selected[
                    batch_index,
                    frame_index,
                    row,
                    col,
                ] = True
    return selected.to(device=weight.device)


def build_reference_identity_bootstrap(
    source_latent: torch.Tensor,
    target_latent: torch.Tensor,
    target_attention: torch.Tensor,
    hand_mask: torch.Tensor | None = None,
    patch_size: int = 2,
    eps: float = 1e-6,
) -> ReferenceIdentityBootstrap:
    """Locate customized content from an aligned source/reference pair."""
    if source_latent.shape != target_latent.shape:
        raise ValueError(
            "Source and target reference latents must share shape"
        )
    if source_latent.ndim != 5:
        raise ValueError(
            "Reference latents must have shape [B,T,C,H,W]"
        )
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")
    batch, frames, _, height, width = source_latent.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            "Reference latent size must be divisible by patch_size"
        )
    token_height = height // patch_size
    token_width = width // patch_size
    token_count = frames * token_height * token_width
    if target_attention.shape != (batch, token_count):
        raise ValueError(
            "Target attention and reference token grid must align"
        )
    if hand_mask is not None and hand_mask.shape != (
        batch,
        frames,
        height,
        width,
    ):
        raise ValueError(
            "Reference hand mask must match latent spatial shape"
        )

    latent_change = (
        target_latent.float() - source_latent.float()
    ).abs().mean(dim=2)
    change_score = F.avg_pool2d(
        latent_change.reshape(
            batch * frames,
            1,
            height,
            width,
        ),
        kernel_size=patch_size,
        stride=patch_size,
    ).reshape(batch, frames, token_height, token_width)

    def robust_unit(value: torch.Tensor) -> torch.Tensor:
        flat = value.flatten(2)
        median = torch.quantile(
            flat,
            0.50,
            dim=-1,
            keepdim=True,
        )
        residual = (flat - median).clamp_min(0.0)
        high = torch.quantile(
            residual,
            0.95,
            dim=-1,
            keepdim=True,
        )
        normalized = residual / high.clamp_min(eps)
        normalized = torch.where(
            high > eps,
            normalized,
            torch.zeros_like(normalized),
        )
        return normalized.clamp(0.0, 1.0).reshape_as(value)

    change_score = robust_unit(change_score)
    raw_semantic = target_attention.float().clamp_min(0.0).reshape(
        batch,
        frames,
        token_height,
        token_width,
    )
    semantic_score = robust_unit(raw_semantic)
    semantic_available = (
        semantic_score.flatten(2).amax(dim=-1, keepdim=True) > eps
    ).reshape(batch, frames, 1, 1)
    joint_score = torch.where(
        semantic_available,
        torch.sqrt(change_score * semantic_score),
        change_score,
    )

    hand_contact_score = torch.zeros_like(joint_score)
    if hand_mask is not None:
        hand_core = F.avg_pool2d(
            hand_mask.float().reshape(
                batch * frames,
                1,
                height,
                width,
            ),
            kernel_size=patch_size,
            stride=patch_size,
        ).reshape_as(joint_score).clamp(0.0, 1.0)
        contact_radius = max(
            1,
            round(min(token_height, token_width) * 0.067),
        )
        hand_contact_score = F.max_pool2d(
            hand_core.reshape(
                batch * frames,
                1,
                token_height,
                token_width,
            ),
            kernel_size=2 * contact_radius + 1,
            stride=1,
            padding=contact_radius,
        ).reshape_as(joint_score)
        joint_score = joint_score * (1.0 - hand_core)

    flat_joint = joint_score.flatten(2)
    center = torch.quantile(
        flat_joint,
        0.50,
        dim=-1,
        keepdim=True,
    )
    deviation = (flat_joint - center).abs()
    robust_scale = (
        torch.quantile(
            deviation,
            0.50,
            dim=-1,
            keepdim=True,
        )
        / 0.6745
    )
    threshold = center + robust_scale
    high = torch.quantile(
        flat_joint,
        0.95,
        dim=-1,
        keepdim=True,
    )
    write_weight = (
        (flat_joint - threshold)
        / (high - threshold).clamp_min(eps)
    ).clamp(0.0, 1.0)
    write_weight = torch.where(
        high > threshold + eps,
        write_weight,
        torch.zeros_like(write_weight),
    ).reshape(batch, frames, token_height, token_width)
    component_mask = _largest_weighted_component_mask(
        write_weight,
        eps=eps,
        hand_contact_score=hand_contact_score,
    )
    write_weight = (
        write_weight * component_mask.float()
    ).reshape(batch, -1)

    result = ReferenceIdentityBootstrap(
        write_weight=write_weight.float(),
        change_score=change_score.float(),
        semantic_score=semantic_score.float(),
        joint_score=joint_score.float(),
        hand_contact_score=hand_contact_score.float(),
    )
    result.validate()
    return result


class SlowTargetIdentityMemory:
    """Maintain reliable target appearance without spatially locking pose."""

    def __init__(
        self,
        layers: Iterable[int] = (8, 12, 16, 20),
        num_prototypes: int = 4,
        reference_prior_evidence: float = 8.0,
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
        if reference_prior_evidence <= 0:
            raise ValueError(
                "reference_prior_evidence must be positive"
            )
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.num_prototypes = num_prototypes
        self.reference_prior_evidence = (
            float(reference_prior_evidence)
        )
        self.eps = eps
        self.states: Dict[int, TargetIdentityLayerState] = {}
        self.reference_bootstrapped = False

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

    @torch.no_grad()
    def bootstrap_reference(
        self,
        kv_cache,
        write_weight: torch.Tensor,
    ) -> TargetIdentityUpdate:
        if self.reference_bootstrapped:
            raise RuntimeError(
                "Target identity reference was already bootstrapped"
            )
        if self.states:
            raise RuntimeError(
                "Target identity reference must be bootstrapped before "
                "online identity updates"
            )
        update = self.update(kv_cache, write_weight)
        for layer, state in tuple(self.states.items()):
            authoritative_evidence = torch.where(
                state.evidence > self.eps,
                torch.full_like(
                    state.evidence,
                    self.reference_prior_evidence,
                ),
                torch.zeros_like(state.evidence),
            )
            anchored_state = TargetIdentityLayerState(
                key=state.key,
                value=state.value,
                evidence=authoritative_evidence,
            )
            anchored_state.validate()
            self.states[layer] = anchored_state
        self.reference_bootstrapped = True
        anchored_update = TargetIdentityUpdate(
            write_weight=update.write_weight,
            observation_evidence=update.observation_evidence,
            update_gain=update.update_gain,
            accumulated_evidence=torch.stack(
                [
                    self.states[layer].evidence
                    for layer in self.layers
                ],
                dim=0,
            ),
        )
        anchored_update.validate()
        return anchored_update

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


def _expand_token_map_to_belief(
    token_map: torch.Tensor,
    belief: CausalControlBelief,
) -> torch.Tensor:
    if token_map.ndim != 4:
        raise ValueError("token_map must have shape [B,T,H,W]")
    batch, frames, height, width = belief.edit_belief.shape
    if token_map.shape[:2] != (batch, frames):
        raise ValueError("token_map and belief must share [B,T]")
    return F.interpolate(
        token_map.float().reshape(
            batch * frames,
            1,
            *token_map.shape[-2:],
        ),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).reshape(batch, frames, height, width).clamp(0.0, 1.0)


def inject_committed_memory_into_belief(
    belief: CausalControlBelief,
    committed_token_edit: torch.Tensor,
    committed_token_precision: torch.Tensor,
    hand_mask: torch.Tensor,
    feedback_strength: float,
    eps: float = 1e-6,
) -> tuple[CausalControlBelief, Dict[str, torch.Tensor]]:
    """Materialize transported edit memory into the current action belief."""
    if feedback_strength <= 0:
        return belief, {}
    if committed_token_edit.shape != committed_token_precision.shape:
        raise ValueError(
            "Committed edit and precision token maps must share shape"
        )
    if hand_mask.shape != belief.edit_belief.shape:
        raise ValueError("hand_mask and belief must share shape")

    committed_edit_full = _expand_token_map_to_belief(
        committed_token_edit,
        belief,
    )
    committed_precision_full = _expand_token_map_to_belief(
        committed_token_precision,
        belief,
    )
    object_space = (1.0 - hand_mask.float().clamp(0.0, 1.0))
    committed_evidence = (
        committed_edit_full
        * committed_precision_full.clamp(0.0, 1.0)
        * object_space
    ).clamp(0.0, 1.0)
    committed_evidence = (
        committed_evidence * float(feedback_strength)
    ).clamp(0.0, 1.0)

    old_edit_belief = belief.edit_belief.float()
    old_edit_precision = belief.edit_precision.float()
    edit_belief = (
        1.0
        - (1.0 - old_edit_belief)
        * (1.0 - committed_evidence)
    ).clamp(0.0, 1.0)
    edit_precision = torch.where(
        edit_belief > eps,
        (
            old_edit_belief * old_edit_precision
            + committed_evidence * committed_precision_full
        )
        / (old_edit_belief + committed_evidence).clamp_min(eps),
        old_edit_precision,
    ).clamp(0.0, 1.0)

    preserve_release = committed_evidence.clamp(0.0, 0.90)
    preserve_belief = (
        belief.preserve_belief.float()
        * (1.0 - preserve_release)
    ).clamp(0.0, 1.0)
    visibility = torch.maximum(
        belief.visibility.float(),
        committed_evidence,
    )
    conflict = (edit_belief * preserve_belief).clamp(0.0, 1.0)
    responsibility = (edit_belief + preserve_belief).clamp_min(eps)
    updated_uncertainty = (
        edit_belief * (1.0 - edit_precision)
        + preserve_belief
        * (1.0 - belief.preserve_precision.float())
    ) / responsibility
    uncertainty = torch.where(
        committed_evidence > eps,
        updated_uncertainty,
        belief.uncertainty.float(),
    )
    injected = CausalControlBelief(
        edit_belief=edit_belief.float(),
        preserve_belief=preserve_belief.float(),
        edit_precision=edit_precision.float(),
        preserve_precision=belief.preserve_precision.float(),
        visibility=visibility.float(),
        uncertainty=uncertainty.clamp(0.0, 1.0).float(),
        conflict=conflict.float(),
    )
    injected.validate()
    return injected, {
        "committed_memory_edit": committed_edit_full.float(),
        "committed_memory_precision": committed_precision_full.float(),
        "committed_memory_evidence": committed_evidence.float(),
        "committed_memory_preserve_release": preserve_release.float(),
    }
