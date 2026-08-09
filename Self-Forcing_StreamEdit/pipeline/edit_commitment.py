from dataclasses import dataclass
from typing import Dict, Mapping

import torch
import torch.nn.functional as F

from .control_belief import CausalControlBelief


@dataclass(frozen=True)
class EditCommitmentResult:
    """Persistent target-edit state transported on the source stream."""

    belief: CausalControlBelief
    trigger: torch.Tensor
    transported: torch.Tensor
    transport_precision: torch.Tensor
    anchor_transport: torch.Tensor
    anchor_precision: torch.Tensor
    semantic_presence: torch.Tensor
    semantic_absence: torch.Tensor
    commitment: torch.Tensor
    commitment_precision: torch.Tensor
    state_precision: torch.Tensor
    effective_commitment: torch.Tensor
    edit_support: torch.Tensor

    def validate(self) -> None:
        token_values = {
            "trigger": self.trigger,
            "transported": self.transported,
            "transport_precision": self.transport_precision,
            "anchor_transport": self.anchor_transport,
            "anchor_precision": self.anchor_precision,
            "semantic_presence": self.semantic_presence,
            "semantic_absence": self.semantic_absence,
            "commitment": self.commitment,
            "commitment_precision": self.commitment_precision,
            "state_precision": self.state_precision,
            "effective_commitment": self.effective_commitment,
            "edit_support": self.edit_support,
        }
        shapes = {tuple(value.shape) for value in token_values.values()}
        if len(shapes) != 1 or self.trigger.ndim != 4:
            raise ValueError(
                "Edit commitment maps must share shape [B,T,H,W]"
            )
        for name, value in token_values.items():
            if not torch.isfinite(value.float()).all():
                raise ValueError(
                    f"Edit commitment '{name}' is not finite"
                )
            if value.dtype != torch.bool and (
                value.min() < 0 or value.max() > 1
            ):
                raise ValueError(
                    f"Edit commitment '{name}' must lie in [0, 1]"
                )
        self.belief.validate()

    def as_debug_maps(self) -> Dict[str, torch.Tensor]:
        return {
            "commitment_trigger": self.trigger.float(),
            "commitment_transport": self.transported.float(),
            "commitment_transport_precision": (
                self.transport_precision.float()
            ),
            "commitment_anchor_transport": (
                self.anchor_transport.float()
            ),
            "commitment_anchor_precision": (
                self.anchor_precision.float()
            ),
            "commitment_semantic_presence": (
                self.semantic_presence.float()
            ),
            "commitment_semantic_absence": (
                self.semantic_absence.float()
            ),
            "commitment_posterior": self.commitment.float(),
            "commitment_precision": self.commitment_precision.float(),
            "commitment_state_precision": self.state_precision.float(),
            "commitment_effective": self.effective_commitment.float(),
            "commitment_edit_support": self.edit_support.float(),
        }


class EditCommitmentController:
    """Turn hand-triggered edits into persistent, presence-gated state."""

    def __init__(
        self,
        topk: int = 4,
        reference_radius_ratio: float = 0.15,
        eps: float = 1e-6,
    ):
        if topk <= 0:
            raise ValueError("topk must be positive")
        if not 0.0 < reference_radius_ratio <= 0.5:
            raise ValueError(
                "reference_radius_ratio must lie in (0, 0.5]"
            )
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.topk = topk
        self.reference_radius_ratio = reference_radius_ratio
        self.eps = eps
        self.previous_features = None
        self.previous_commitment = None
        self.previous_precision = None
        self.anchor_features = None
        self.anchor_commitment = None
        self.anchor_precision = None
        self.anchor_score = None
        self.reference_bootstrapped = False
        self.last_spatial_radius = None
        self._spatial_mask_cache = {}

    @torch.no_grad()
    def bootstrap_reference(
        self,
        source_features: torch.Tensor,
        edit_precision: torch.Tensor,
    ):
        """Seed persistent edit responsibility from an aligned reference."""
        if self.reference_bootstrapped:
            raise RuntimeError(
                "Edit commitment reference was already bootstrapped"
            )
        if any(
            state is not None
            for state in (
                self.previous_features,
                self.previous_commitment,
                self.previous_precision,
                self.anchor_features,
                self.anchor_commitment,
                self.anchor_precision,
            )
        ):
            raise RuntimeError(
                "Edit commitment reference must be bootstrapped before "
                "online updates"
            )
        if source_features.ndim != 3:
            raise ValueError(
                "Reference source_features must have shape [B,L,D]"
            )
        if edit_precision.ndim != 2:
            raise ValueError(
                "Reference edit_precision must have shape [B,L]"
            )
        if source_features.shape[:2] != edit_precision.shape:
            raise ValueError(
                "Reference source features and edit precision must align"
            )
        if not torch.isfinite(source_features.float()).all():
            raise ValueError(
                "Reference source features must be finite"
            )
        if not torch.isfinite(edit_precision.float()).all():
            raise ValueError(
                "Reference edit precision must be finite"
            )

        features = F.normalize(
            source_features.detach().float(),
            dim=-1,
        )
        precision = edit_precision.detach().float().clamp(0.0, 1.0)
        commitment = (precision > self.eps).float()

        self.previous_features = features
        self.previous_commitment = commitment
        self.previous_precision = precision
        self.anchor_features = features.clone()
        self.anchor_commitment = commitment.clone()
        self.anchor_precision = precision.clone()
        self.anchor_score = precision.mean(dim=-1, keepdim=True)
        self.reference_bootstrapped = True
        return commitment, precision

    @staticmethod
    def _required(
        debug: Mapping[str, torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        value = debug.get(name)
        if value is None:
            raise ValueError(f"Missing commitment evidence: {name}")
        return value.float()

    def _transport(
        self,
        current_features: torch.Tensor,
        reference_features: torch.Tensor,
        reference_commitment: torch.Tensor,
        reference_precision: torch.Tensor,
        spatial_shape=None,
        spatial_radius=None,
    ):
        similarity = torch.einsum(
            "bmd,bnd->bmn",
            reference_features,
            current_features,
        ).clamp(-1.0, 1.0)
        if (spatial_shape is None) != (spatial_radius is None):
            raise ValueError(
                "spatial_shape and spatial_radius must be set together"
            )
        if spatial_shape is not None:
            token_height, token_width = spatial_shape
            token_count = token_height * token_width
            if (
                token_count != reference_features.shape[1]
                or token_count != current_features.shape[1]
            ):
                raise ValueError(
                    "Spatial transport grid and feature tokens must align"
                )
            if spatial_radius <= 0:
                raise ValueError(
                    "spatial_radius must be positive"
                )
            device = current_features.device
            cache_key = (
                token_height,
                token_width,
                int(spatial_radius),
                device.type,
                device.index,
            )
            local_mask = self._spatial_mask_cache.get(cache_key)
            if local_mask is None:
                rows = torch.arange(
                    token_height,
                    device=device,
                ).repeat_interleave(token_width)
                cols = torch.arange(
                    token_width,
                    device=device,
                ).repeat(token_height)
                local_mask = (
                    (rows[:, None] - rows[None, :]).abs()
                    <= spatial_radius
                ) & (
                    (cols[:, None] - cols[None, :]).abs()
                    <= spatial_radius
                )
                self._spatial_mask_cache[cache_key] = local_mask
            similarity = similarity.masked_fill(
                ~local_mask.unsqueeze(0),
                torch.finfo(similarity.dtype).min,
            )
        topk = min(self.topk, similarity.shape[-1])
        top_similarity, top_index = similarity.topk(topk, dim=-1)

        best_similarity = top_similarity[..., 0]
        similarity_low = torch.quantile(
            best_similarity,
            0.25,
            dim=-1,
            keepdim=True,
        )
        similarity_high = torch.quantile(
            best_similarity,
            0.75,
            dim=-1,
            keepdim=True,
        )
        temperature = (
            similarity_high - similarity_low
        ).clamp(0.03, 0.20).unsqueeze(-1)
        splat_weight = torch.softmax(
            top_similarity / temperature,
            dim=-1,
        )

        median_similarity = torch.quantile(
            best_similarity,
            0.50,
            dim=-1,
            keepdim=True,
        )
        high_similarity = torch.quantile(
            best_similarity,
            0.90,
            dim=-1,
            keepdim=True,
        )
        similarity_spread = high_similarity - median_similarity
        relative_confidence = (
            (best_similarity - median_similarity)
            / (similarity_spread + self.eps)
        ).clamp(0.0, 1.0)
        relative_confidence = torch.where(
            similarity_spread <= self.eps,
            torch.ones_like(relative_confidence),
            relative_confidence,
        )
        absolute_confidence = (
            0.5 * (best_similarity + 1.0)
        ).clamp(0.0, 1.0)
        match_confidence = torch.sqrt(
            relative_confidence * absolute_confidence
        )

        precision_contribution = (
            splat_weight
            * match_confidence.unsqueeze(-1)
            * reference_precision.unsqueeze(-1)
        )
        action_contribution = (
            precision_contribution
            * reference_commitment.unsqueeze(-1)
        )
        active_reference = (
            reference_precision > self.eps
        ).float().unsqueeze(-1)
        match_contribution = (
            splat_weight
            * match_confidence.unsqueeze(-1)
            * active_reference
        )

        batch, current_tokens, _ = current_features.shape

        def splat(value: torch.Tensor) -> torch.Tensor:
            output = value.new_zeros(batch, current_tokens)
            output.scatter_add_(
                1,
                top_index.reshape(batch, -1),
                value.reshape(batch, -1),
            )
            return output

        precision_mass = splat(precision_contribution)
        action_mass = splat(action_contribution)
        match_mass = splat(match_contribution)
        transported = torch.where(
            precision_mass > self.eps,
            action_mass / precision_mass.clamp_min(self.eps),
            torch.zeros_like(action_mass),
        ).clamp(0.0, 1.0)
        transported_precision = precision_mass.clamp(0.0, 1.0)
        current_match_confidence = match_mass.clamp(0.0, 1.0)
        return (
            transported,
            transported_precision,
            current_match_confidence,
        )

    def __call__(
        self,
        belief: CausalControlBelief,
        debug: Mapping[str, torch.Tensor],
        hand_mask: torch.Tensor,
        source_features: torch.Tensor,
    ) -> EditCommitmentResult:
        belief.validate()
        if hand_mask.ndim != 4:
            raise ValueError("hand_mask must have shape [B,T,H,W]")
        if source_features.ndim != 3:
            raise ValueError(
                "source_features must have shape [B,L,D]"
            )

        source_attention = self._required(
            debug,
            "source_attention",
        )
        hand_proximity = self._required(
            debug,
            "hand_proximity",
        )
        if source_attention.shape != hand_proximity.shape:
            raise ValueError(
                "Source attention and hand proximity must share shape"
            )
        batch, frames, token_height, token_width = (
            source_attention.shape
        )
        tokens_per_frame = token_height * token_width
        spatial_shape = None
        spatial_radius = None
        if self.reference_bootstrapped:
            spatial_shape = (token_height, token_width)
            spatial_radius = max(
                1,
                round(
                    min(token_height, token_width)
                    * self.reference_radius_ratio
                ),
            )
        self.last_spatial_radius = spatial_radius
        expected_tokens = frames * tokens_per_frame
        if source_features.shape[:2] != (batch, expected_tokens):
            raise ValueError(
                "Source features and commitment grid must align"
            )
        if hand_mask.shape[:2] != (batch, frames):
            raise ValueError(
                "Hand mask and commitment grid must share [B,T]"
            )

        height, width = belief.edit_belief.shape[-2:]
        if hand_mask.shape[-2:] != (height, width):
            raise ValueError(
                "Hand mask and control belief must share spatial size"
            )
        if height != token_height * 2 or width != token_width * 2:
            raise ValueError(
                "Commitment token grid must use patch size 2"
            )

        def downsample(value: torch.Tensor) -> torch.Tensor:
            return F.avg_pool2d(
                value.float().reshape(
                    batch * frames,
                    1,
                    height,
                    width,
                ),
                kernel_size=2,
                stride=2,
            ).reshape(batch, frames, tokens_per_frame)

        features = F.normalize(
            source_features.float(),
            dim=-1,
        ).reshape(batch, frames, tokens_per_frame, -1)
        source_attention = source_attention.reshape(
            batch,
            frames,
            tokens_per_frame,
        ).clamp(0.0, 1.0)
        hand_proximity = hand_proximity.reshape(
            batch,
            frames,
            tokens_per_frame,
        ).clamp(0.0, 1.0)
        hand_probability = downsample(
            hand_mask.float().clamp(0.0, 1.0)
        )
        interaction = torch.maximum(
            hand_proximity,
            hand_probability,
        )
        edit_belief = downsample(belief.edit_belief)
        edit_precision = downsample(belief.edit_precision)
        trigger_precision = (
            edit_precision * interaction * edit_belief
        ).clamp(0.0, 1.0)
        trigger = (
            edit_belief * trigger_precision
        ).clamp(0.0, 1.0)

        transported_values = []
        transported_precisions = []
        anchor_values = []
        anchor_precisions = []
        semantic_presences = []
        semantic_absences = []
        commitments = []
        commitment_precisions = []
        state_precisions = []

        reference_features = self.previous_features
        reference_commitment = self.previous_commitment
        reference_precision = self.previous_precision
        anchor_features = self.anchor_features
        anchor_commitment = self.anchor_commitment
        anchor_precision = self.anchor_precision
        anchor_score = self.anchor_score
        for frame_index in range(frames):
            current_features = features[:, frame_index]
            current_attention = source_attention[:, frame_index]
            current_edit_belief = edit_belief[:, frame_index]
            current_trigger_precision = trigger_precision[:, frame_index]

            has_reference = not (
                reference_features is None
                or reference_commitment is None
                or reference_precision is None
            )
            if has_reference:
                (
                    transported,
                    transported_state_precision,
                    match_confidence,
                ) = self._transport(
                    current_features,
                    reference_features,
                    reference_commitment,
                    reference_precision,
                    spatial_shape=spatial_shape,
                    spatial_radius=spatial_radius,
                )
            else:
                transported = torch.zeros_like(current_edit_belief)
                transported_state_precision = torch.zeros_like(
                    current_trigger_precision
                )
                match_confidence = torch.zeros_like(current_attention)

            has_anchor = not (
                anchor_features is None
                or anchor_commitment is None
                or anchor_precision is None
            )
            if has_anchor:
                (
                    anchor_transport,
                    anchor_state_precision,
                    anchor_match_confidence,
                ) = self._transport(
                    current_features,
                    anchor_features,
                    anchor_commitment,
                    anchor_precision,
                    spatial_shape=spatial_shape,
                    spatial_radius=spatial_radius,
                )
            else:
                anchor_transport = torch.zeros_like(current_edit_belief)
                anchor_state_precision = torch.zeros_like(
                    current_trigger_precision
                )
                anchor_match_confidence = torch.zeros_like(
                    current_attention
                )

            temporal_absence = (
                match_confidence * (1.0 - current_attention)
            ).clamp(0.0, 1.0)
            anchor_absence = (
                anchor_match_confidence * (1.0 - current_attention)
            ).clamp(0.0, 1.0)
            temporal_localized_precision = (
                transported_state_precision * (1.0 - temporal_absence)
            ).clamp(0.0, 1.0)
            anchor_localized_precision = (
                anchor_state_precision * (1.0 - anchor_absence)
            ).clamp(0.0, 1.0)

            prior_precision_sum = (
                transported_state_precision + anchor_state_precision
            )
            prior_commitment = torch.where(
                prior_precision_sum > self.eps,
                (
                    transported * transported_state_precision
                    + anchor_transport * anchor_state_precision
                )
                / prior_precision_sum.clamp_min(self.eps),
                torch.zeros_like(current_edit_belief),
            )
            prior_state_precision = torch.maximum(
                transported_state_precision,
                anchor_state_precision,
            )
            localized_precision = torch.maximum(
                temporal_localized_precision,
                anchor_localized_precision,
            )
            semantic_presence = torch.maximum(
                torch.sqrt(
                    (
                        current_attention * match_confidence
                    ).clamp_min(0.0)
                ),
                torch.sqrt(
                    (
                        current_attention
                        * anchor_match_confidence
                    ).clamp_min(0.0)
                ),
            )
            semantic_absence = torch.where(
                prior_state_precision > self.eps,
                (
                    1.0
                    - localized_precision
                    / prior_state_precision.clamp_min(self.eps)
                ),
                torch.zeros_like(current_attention),
            ).clamp(0.0, 1.0)

            state_total_precision = (
                prior_state_precision + current_trigger_precision
            )
            commitment = torch.where(
                state_total_precision > self.eps,
                (
                    prior_commitment * prior_state_precision
                    + current_edit_belief
                    * current_trigger_precision
                )
                / state_total_precision.clamp_min(self.eps),
                torch.zeros_like(current_edit_belief),
            ).clamp(0.0, 1.0)
            state_precision = (
                1.0
                - (1.0 - prior_state_precision)
                * (1.0 - current_trigger_precision)
            ).clamp(0.0, 1.0)
            commitment_precision = (
                1.0
                - (1.0 - localized_precision)
                * (1.0 - current_trigger_precision)
            ).clamp(0.0, 1.0)

            transported_values.append(transported)
            transported_precisions.append(localized_precision)
            anchor_values.append(anchor_transport)
            anchor_precisions.append(anchor_localized_precision)
            semantic_presences.append(semantic_presence)
            semantic_absences.append(semantic_absence)
            commitments.append(commitment)
            commitment_precisions.append(commitment_precision)
            state_precisions.append(state_precision)

            reference_features = current_features
            reference_commitment = commitment
            reference_precision = state_precision

            current_anchor_score = current_trigger_precision.mean(
                dim=-1,
                keepdim=True,
            )
            if anchor_score is None:
                anchor_score = torch.zeros_like(current_anchor_score)
                anchor_features = torch.zeros_like(current_features)
                anchor_commitment = torch.zeros_like(commitment)
                anchor_precision = torch.zeros_like(state_precision)
            replace_anchor = current_anchor_score > anchor_score
            anchor_score = torch.where(
                replace_anchor,
                current_anchor_score,
                anchor_score,
            )
            anchor_features = torch.where(
                replace_anchor.unsqueeze(-1),
                current_features,
                anchor_features,
            )
            anchor_commitment = torch.where(
                replace_anchor,
                commitment,
                anchor_commitment,
            )
            anchor_precision = torch.where(
                replace_anchor,
                state_precision,
                anchor_precision,
            )

        self.previous_features = reference_features.detach()
        self.previous_commitment = reference_commitment.detach()
        self.previous_precision = reference_precision.detach()
        self.anchor_features = anchor_features.detach()
        self.anchor_commitment = anchor_commitment.detach()
        self.anchor_precision = anchor_precision.detach()
        self.anchor_score = anchor_score.detach()

        transported = torch.stack(transported_values, dim=1)
        transport_precision = torch.stack(
            transported_precisions,
            dim=1,
        )
        anchor_transport = torch.stack(
            anchor_values,
            dim=1,
        )
        anchor_precision = torch.stack(
            anchor_precisions,
            dim=1,
        )
        semantic_presence = torch.stack(
            semantic_presences,
            dim=1,
        )
        semantic_absence = torch.stack(
            semantic_absences,
            dim=1,
        )
        commitment = torch.stack(commitments, dim=1)
        commitment_precision = torch.stack(
            commitment_precisions,
            dim=1,
        )
        state_precision = torch.stack(
            state_precisions,
            dim=1,
        )
        effective_commitment = (
            commitment * commitment_precision
        ).clamp(0.0, 1.0)

        def token_map(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(
                batch,
                frames,
                token_height,
                token_width,
            )

        commitment_token_map = token_map(commitment)
        commitment_map = F.interpolate(
            commitment_token_map.reshape(
                batch * frames,
                1,
                token_height,
                token_width,
            ),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, frames, height, width).clamp(0.0, 1.0)
        effective_map = token_map(effective_commitment)
        effective_full = F.interpolate(
            effective_map.reshape(
                batch * frames,
                1,
                token_height,
                token_width,
            ),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, frames, height, width).clamp(0.0, 1.0)
        precision_map = F.interpolate(
            token_map(commitment_precision).reshape(
                batch * frames,
                1,
                token_height,
                token_width,
            ),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, frames, height, width).clamp(0.0, 1.0)

        hand_core = hand_mask.float().clamp(0.0, 1.0)
        base_edit_belief = belief.edit_belief.float()
        base_edit_precision = belief.edit_precision.float()
        unique_commitment = (
            commitment_map * (1.0 - base_edit_belief)
        )
        edit_belief_full = (
            base_edit_belief + unique_commitment
        ).clamp(0.0, 1.0)
        edit_precision_full = torch.where(
            edit_belief_full > self.eps,
            (
                base_edit_belief * base_edit_precision
                + unique_commitment * precision_map
            )
            / edit_belief_full.clamp_min(self.eps),
            base_edit_precision,
        ).clamp(0.0, 1.0)
        preserve_release = (
            effective_full * (1.0 - hand_core)
        )
        preserve_belief_full = (
            belief.preserve_belief.float()
            * (1.0 - preserve_release)
        ).clamp(0.0, 1.0)
        visibility_full = torch.maximum(
            belief.visibility.float(),
            effective_full,
        )
        conflict_full = (
            edit_belief_full * preserve_belief_full
        ).clamp(0.0, 1.0)
        responsibility = (
            edit_belief_full + preserve_belief_full
        ).clamp_min(self.eps)
        uncertainty_full = (
            edit_belief_full * (1.0 - edit_precision_full)
            + preserve_belief_full
            * (1.0 - belief.preserve_precision.float())
        ) / responsibility
        committed_belief = CausalControlBelief(
            edit_belief=edit_belief_full.float(),
            preserve_belief=preserve_belief_full.float(),
            edit_precision=edit_precision_full.float(),
            preserve_precision=belief.preserve_precision.float(),
            visibility=visibility_full.float(),
            uncertainty=uncertainty_full.clamp(0.0, 1.0).float(),
            conflict=conflict_full.float(),
        )
        committed_belief.validate()

        edit_strength = edit_belief_full * edit_precision_full
        preserve_strength = (
            preserve_belief_full
            * belief.preserve_precision.float()
        )
        edit_support = edit_strength > preserve_strength
        result = EditCommitmentResult(
            belief=committed_belief,
            trigger=token_map(trigger),
            transported=token_map(transported),
            transport_precision=token_map(transport_precision),
            anchor_transport=token_map(anchor_transport),
            anchor_precision=token_map(anchor_precision),
            semantic_presence=token_map(semantic_presence),
            semantic_absence=token_map(semantic_absence),
            commitment=token_map(commitment),
            commitment_precision=token_map(commitment_precision),
            state_precision=token_map(state_precision),
            effective_commitment=effective_map,
            edit_support=F.max_pool2d(
                edit_support.float().reshape(
                    batch * frames,
                    1,
                    height,
                    width,
                ),
                kernel_size=2,
                stride=2,
            ).reshape(
                batch,
                frames,
                token_height,
                token_width,
            ).bool(),
        )
        result.validate()
        return result
