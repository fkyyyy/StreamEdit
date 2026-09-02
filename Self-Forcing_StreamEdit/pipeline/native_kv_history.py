from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch


def validate_recent_entry_hand_only_contract(
    *,
    enabled: bool,
    routing_mode: str,
    hand_only_mask,
    oracle_object_mask,
    oracle_source_owner_mask,
    oracle_source_owner_full_mask,
) -> None:
    """Enforce the deployable hand-only input contract for entry memory."""
    if not enabled:
        return
    if (
        routing_mode != "hand_role_factorized_causal_owner_kv"
        or hand_only_mask is None
        or oracle_object_mask is not None
        or oracle_source_owner_mask is not None
        or oracle_source_owner_full_mask is not None
    ):
        raise ValueError(
            "Recent-entry native history requires hand-only causal owner "
            "routing with a hand mask and forbids external object/source-"
            "owner masks"
        )


@dataclass(frozen=True)
class NativeKVFrame:
    """One final-clean block or compact tier in native pre-RoPE KV space."""

    source_key: torch.Tensor
    source_value: torch.Tensor
    target_key: torch.Tensor
    target_value: torch.Tensor
    token_index: torch.Tensor
    support: torch.Tensor
    frame_count: int
    # Source addresses and target payloads have different trust contracts.
    # ``support`` says that a clean-source address may participate in
    # correspondence. ``payload_support`` says that the target K/V stored at
    # that address passed the tokenwise write transaction. Legacy entries use
    # one shared mask by leaving this unset.
    payload_support: torch.Tensor | None = None
    # True when target V at every authorized payload slot is represented as
    # current clean-source V plus a transported target-minus-source residual.
    # Target K remains current so attention follows current geometry. Readers
    # inject only the V residual instead of copying an older block's pose.
    residual_rebased_payload: bool = False

    def validate(self) -> None:
        if self.source_key.ndim != 4:
            raise ValueError("Native KV keys must have shape [B,L,H,D]")
        if not (
            self.source_key.shape
            == self.source_value.shape
            == self.target_key.shape
            == self.target_value.shape
        ):
            raise ValueError(
                "Native source/target K/V must share shape"
            )
        if self.token_index.shape != self.source_key.shape[:2]:
            raise ValueError("Native token indices must align with K/V")
        if self.support.shape != self.source_key.shape[:2]:
            raise ValueError("Native support must align with K/V")
        if (
            self.payload_support is not None
            and self.payload_support.shape != self.source_key.shape[:2]
        ):
            raise ValueError(
                "Native payload support must align with K/V"
            )
        if self.frame_count <= 0:
            raise ValueError("Native KV frame_count must be positive")
        if self.residual_rebased_payload and self.payload_support is None:
            raise ValueError(
                "Residual-rebased native payloads require token support"
            )


@dataclass(frozen=True)
class NativeFlowResidualFrame:
    """Target-minus-source V residual aligned to the current flow grid."""

    value_residual: torch.Tensor
    support: torch.Tensor
    confidence: torch.Tensor
    frame_count: int
    # ``appearance_trust`` is a property of the last verified target payload
    # and must not decay merely because its coordinate is transported.
    # ``transport_confidence`` describes only the current block-local flow
    # path.  ``confidence`` remains their effective product for readers.
    appearance_trust: torch.Tensor | None = None
    transport_confidence: torch.Tensor | None = None

    def validate(self) -> None:
        if self.value_residual.ndim != 4:
            raise ValueError(
                "Flow-indexed residuals must have shape [B,L,H,D]"
            )
        expected = self.value_residual.shape[:2]
        if self.support.shape != expected or self.confidence.shape != expected:
            raise ValueError(
                "Flow-indexed residual support/confidence must align with V"
            )
        if self.frame_count <= 0 or expected[1] % self.frame_count:
            raise ValueError(
                "Flow-indexed residual must contain complete frames"
            )
        if not torch.isfinite(self.value_residual.float()).all():
            raise ValueError("Flow-indexed residual contains non-finite values")
        if not torch.isfinite(self.confidence.float()).all():
            raise ValueError("Flow-indexed confidence contains non-finite values")
        if self.confidence.min() < 0 or self.confidence.max() > 1:
            raise ValueError("Flow-indexed confidence must lie in [0, 1]")
        for name, value in (
            ("appearance_trust", self.appearance_trust),
            ("transport_confidence", self.transport_confidence),
        ):
            if value is None:
                continue
            if value.shape != expected:
                raise ValueError(
                    f"Flow-indexed {name} must align with V"
                )
            if not torch.isfinite(value.float()).all():
                raise ValueError(f"Flow-indexed {name} is not finite")
            if value.min() < 0 or value.max() > 1:
                raise ValueError(
                    f"Flow-indexed {name} must lie in [0, 1]"
                )
        if (self.appearance_trust is None) != (
            self.transport_confidence is None
        ):
            raise ValueError(
                "Decoupled flow trust requires both appearance and "
                "transport confidence"
            )
        if self.appearance_trust is not None and not torch.allclose(
            self.confidence.float(),
            (
                self.appearance_trust.float()
                * self.transport_confidence.float()
            ),
            rtol=1e-5, atol=1e-6,
        ):
            raise ValueError(
                "Effective flow confidence must equal appearance trust "
                "times local transport confidence"
            )


@dataclass(frozen=True)
class TimestepCounterfactualFrame:
    """Immutable paired B0 K/V at one denoising timestep."""

    source_key: torch.Tensor
    source_value: torch.Tensor
    target_key: torch.Tensor
    target_value: torch.Tensor
    token_index: torch.Tensor
    support: torch.Tensor
    frame_count: int

    def validate(self) -> None:
        values = (
            self.source_key, self.source_value, self.target_key,
            self.target_value,
        )
        if any(value.ndim != 4 for value in values):
            raise ValueError(
                "Timestep counterfactual K/V must have shape [B,K,H,D]"
            )
        if len({value.shape for value in values}) != 1:
            raise ValueError(
                "Timestep counterfactual paired K/V must align"
            )
        if (
            self.token_index.shape != self.source_key.shape[:2]
            or self.support.shape != self.source_key.shape[:2]
        ):
            raise ValueError(
                "Timestep counterfactual metadata must align with K/V"
            )
        if self.frame_count <= 0:
            raise ValueError(
                "Timestep counterfactual frame count must be positive"
            )


@dataclass(frozen=True)
class NativeCanonicalCorrespondence:
    """Clean-source flow coordinates from frozen B0 slots to a block."""

    current_index: torch.Tensor
    support: torch.Tensor
    confidence: torch.Tensor
    frame_count: int

    def validate(self) -> None:
        if self.current_index.ndim != 3:
            raise ValueError(
                "Canonical correspondence must have shape [B,F,K]"
            )
        if not (
            self.current_index.shape
            == self.support.shape
            == self.confidence.shape
        ):
            raise ValueError(
                "Canonical correspondence metadata must align"
            )
        if self.current_index.shape[1] != self.frame_count:
            raise ValueError(
                "Canonical correspondence frame count is inconsistent"
            )
        if self.current_index.dtype != torch.long:
            raise ValueError(
                "Canonical correspondence indices must be long"
            )
        if not torch.isfinite(self.confidence.float()).all():
            raise ValueError(
                "Canonical correspondence confidence is not finite"
            )
        if self.confidence.min() < 0 or self.confidence.max() > 1:
            raise ValueError(
                "Canonical correspondence confidence must lie in [0, 1]"
            )


@dataclass(frozen=True)
class NativeSourceLineageFrame:
    """One source-only motion address block.

    This tier deliberately has no target K/V payload.  It may be replaced
    after every causal block to follow motion, while the edited appearance
    remains exclusively in the immutable canonical ``NativeKVFrame``.
    """

    source_key: torch.Tensor
    source_value: torch.Tensor
    token_index: torch.Tensor
    canonical_index: torch.Tensor
    support: torch.Tensor
    confidence: torch.Tensor
    frame_count: int

    def validate(self) -> None:
        if self.source_key.ndim != 4:
            raise ValueError(
                "Native source lineage keys must have shape [B,L,H,D]"
            )
        if self.source_value.shape != self.source_key.shape:
            raise ValueError(
                "Native source lineage K/V must share shape"
            )
        expected = self.source_key.shape[:2]
        for name, value in (
            ("token_index", self.token_index),
            ("canonical_index", self.canonical_index),
            ("support", self.support),
            ("confidence", self.confidence),
        ):
            if value.shape != expected:
                raise ValueError(
                    f"Native source lineage {name} must align with keys"
                )
        if not torch.isfinite(self.confidence.float()).all():
            raise ValueError(
                "Native source lineage confidence must be finite"
            )
        if self.confidence.min() < 0 or self.confidence.max() > 1:
            raise ValueError(
                "Native source lineage confidence must lie in [0, 1]"
            )
        if self.frame_count <= 0:
            raise ValueError(
                "Native source lineage frame_count must be positive"
            )
        if self.canonical_index.dtype != torch.long:
            raise ValueError(
                "Native source lineage canonical indices must be long"
            )
        if self.support.any() and self.canonical_index[self.support].min() < 0:
            raise ValueError(
                "Supported source lineage must reference canonical slots"
            )


@dataclass(frozen=True)
class NativeKVRead:
    canonical: NativeKVFrame
    recent: NativeKVFrame | None
    source_lineage: NativeSourceLineageFrame | None = None
    flow_residual: NativeFlowResidualFrame | None = None
    canonical_correspondence: NativeCanonicalCorrespondence | None = None
    # On the first read, both tiers were produced by the ignition commit.
    # They occupy the same physical frames even though canonical is compact
    # and recent is dense.  Consumers can use this provenance bit to avoid
    # assigning two consecutive temporal RoPE ranges to one source block.
    recent_shares_canonical_time: bool = False

    def temporal_origins(
        self, *, coalesce_bootstrap_alias: bool = False
    ) -> tuple[int, int]:
        """Return relative RoPE origins for recent and current tokens.

        The legacy layout always places ``recent`` after ``canonical``.  On
        the first post-ignition read those tiers alias the same source block;
        coalescing keeps both at time zero and starts the current block after
        the one real history block.
        """
        # Source lineage is an address table, not target temporal context.
        # It must therefore never push the current target farther away from
        # the canonical target on the RoPE time axis.
        if self.recent is None:
            return self.canonical.frame_count, self.canonical.frame_count
        if (
            coalesce_bootstrap_alias
            and self.recent_shares_canonical_time
        ):
            return 0, self.canonical.frame_count
        recent_start = self.canonical.frame_count
        return recent_start, recent_start + self.recent.frame_count


class RoleConditionedNativeKVHistory:
    """Short/long history with an optional payload-invariant mode.

    The first committed block contributes compact final-clean target tokens
    to the immutable canonical tier.  The latest committed block forms the
    short-term tier.  Both retain their corresponding clean-source keys as
    addresses and clean-source values as optional, read-only part signatures.
    In payload-invariant mode the mutable tier stores clean-source K/V and
    support only.  Generated target observations never enter that tier, so
    all cross-chunk target appearance comes from the immutable canonical
    target payload.
    """

    def __init__(
        self,
        *,
        layers: Iterable[int],
        tokens_per_frame: int,
        max_tokens_per_frame: int = 256,
        min_write_confidence: float = 0.5,
        payload_invariant_lineage: bool = False,
        transactional_compact_recent: bool = False,
        transactional_dense_recent: bool = False,
        token_atomic_dense_recent: bool = False,
        persistent_residual_upsert: bool = False,
        last_trusted_residual_lineage: bool = False,
        flow_indexed_residual_ledger: bool = False,
        decoupled_flow_trust: bool = False,
        multiframe_identity_sink: bool = False,
        timestep_counterfactual_memory: bool = False,
        source_flow_cache: SourceFlowCache | None = None,
        flow_min_confidence: float = 0.10,
        residual_update_min_cosine: float = 0.50,
        residual_update_min_magnitude_ratio: float = 0.90,
        dense_recent_min_residual_consensus: float = 0.05,
        min_lineage_similarity: float = 0.35,
    ):
        self.layers = tuple(int(layer) for layer in layers)
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("Native history layers must be unique/nonempty")
        if tokens_per_frame <= 0 or max_tokens_per_frame <= 0:
            raise ValueError("Native history token budgets must be positive")
        if not 0.0 <= min_write_confidence <= 1.0:
            raise ValueError(
                "Native history write confidence must lie in [0, 1]"
            )
        if not -1.0 < min_lineage_similarity < 1.0:
            raise ValueError(
                "Native source lineage similarity must lie in (-1, 1)"
            )
        if not 0.0 <= dense_recent_min_residual_consensus <= 1.0:
            raise ValueError(
                "Dense recent residual consensus must lie in [0, 1]"
            )
        self.tokens_per_frame = int(tokens_per_frame)
        self.max_tokens_per_frame = int(max_tokens_per_frame)
        self.min_write_confidence = float(min_write_confidence)
        self.payload_invariant_lineage = bool(payload_invariant_lineage)
        self.transactional_compact_recent = bool(
            transactional_compact_recent
        )
        self.transactional_dense_recent = bool(transactional_dense_recent)
        self.token_atomic_dense_recent = bool(token_atomic_dense_recent)
        if self.token_atomic_dense_recent and not self.transactional_dense_recent:
            raise ValueError(
                "Token-atomic recent payloads require transactional dense "
                "recent history"
            )
        self.persistent_residual_upsert = bool(
            persistent_residual_upsert
        )
        if self.persistent_residual_upsert and not (
            self.transactional_dense_recent
            and self.token_atomic_dense_recent
        ):
            raise ValueError(
                "Persistent residual upserts require token-atomic dense "
                "recent history"
            )
        self.last_trusted_residual_lineage = bool(
            last_trusted_residual_lineage
        )
        if self.last_trusted_residual_lineage and not (
            self.persistent_residual_upsert
        ):
            raise ValueError(
                "Last-trusted residual lineage requires persistent "
                "residual upserts"
            )
        self.flow_indexed_residual_ledger = bool(
            flow_indexed_residual_ledger
        )
        self.decoupled_flow_trust = bool(decoupled_flow_trust)
        self.multiframe_identity_sink = bool(multiframe_identity_sink)
        self.timestep_counterfactual_memory = bool(
            timestep_counterfactual_memory
        )
        if self.decoupled_flow_trust and not self.flow_indexed_residual_ledger:
            raise ValueError(
                "Decoupled flow trust requires a flow-indexed residual ledger"
            )
        if self.multiframe_identity_sink and not self.decoupled_flow_trust:
            raise ValueError(
                "Multi-frame identity sink requires decoupled flow trust"
            )
        if (
            self.timestep_counterfactual_memory
            and not self.multiframe_identity_sink
        ):
            raise ValueError(
                "Timestep counterfactual memory requires the multi-frame "
                "identity sink contract"
            )
        self.source_flow_cache = source_flow_cache
        if self.flow_indexed_residual_ledger and not (
            self.last_trusted_residual_lineage
            and self.source_flow_cache is not None
        ):
            raise ValueError(
                "Flow-indexed residual ledger requires last-trusted "
                "residual history and a clean-source flow cache"
            )
        if not 0.0 <= float(flow_min_confidence) <= 1.0:
            raise ValueError("Flow ledger confidence must lie in [0, 1]")
        self.flow_min_confidence = float(flow_min_confidence)
        if not -1.0 <= residual_update_min_cosine <= 1.0:
            raise ValueError(
                "Residual update cosine threshold must lie in [-1, 1]"
            )
        if not 0.0 <= residual_update_min_magnitude_ratio <= 1.0:
            raise ValueError(
                "Residual update magnitude ratio must lie in [0, 1]"
            )
        self.residual_update_min_cosine = float(
            residual_update_min_cosine
        )
        self.residual_update_min_magnitude_ratio = float(
            residual_update_min_magnitude_ratio
        )
        self.dense_recent_min_residual_consensus = float(
            dense_recent_min_residual_consensus
        )
        recent_modes = sum((
            self.transactional_compact_recent,
            self.transactional_dense_recent,
        ))
        if recent_modes > 1:
            raise ValueError(
                "Native history recent tier modes are mutually exclusive"
            )
        self.min_lineage_similarity = float(min_lineage_similarity)
        self._canonical: Dict[int, NativeKVFrame] = {}
        self._recent: Dict[int, NativeKVFrame] = {}
        self._source_lineage: Dict[int, NativeSourceLineageFrame] = {}
        self._flow_state: Dict[int, NativeFlowResidualFrame] = {}
        self._flow_state_index: int | None = None
        self._prepared_flow_read: Dict[int, NativeFlowResidualFrame] = {}
        self._prepared_flow_indices: tuple[int, ...] | None = None
        self._timestep_staging: Dict[
            tuple[int, int], dict[str, torch.Tensor]
        ] = {}
        self._timestep_bank: Dict[
            tuple[int, int], TimestepCounterfactualFrame
        ] = {}
        self._timestep_bank_frozen = False
        self._canonical_coordinate_state: Dict[int, dict] = {}
        self._prepared_canonical_coordinate_state: Dict[int, dict] = {}
        self._prepared_canonical_correspondence: Dict[
            int, NativeCanonicalCorrespondence
        ] = {}
        self._commit_count = 0

    def has_canonical(self) -> bool:
        return all(layer in self._canonical for layer in self.layers)

    @property
    def timestep_bank_frozen(self) -> bool:
        return self._timestep_bank_frozen

    @torch.no_grad()
    def stage_timestep_counterfactual(
        self, *, layer: int, timestep_index: int, source_key,
        source_value, target_key, target_value, selection_weight,
    ) -> bool:
        """Stage dense B0 state once; later commits cannot mutate it."""
        if (
            not self.timestep_counterfactual_memory
            or self._timestep_bank_frozen
            or self._commit_count > 0
        ):
            return False
        layer = int(layer)
        timestep_index = int(timestep_index)
        if layer not in self.layers or timestep_index < 0:
            raise ValueError(
                "Invalid layer/timestep for counterfactual memory"
            )
        tensors = {
            "source_key": source_key,
            "source_value": source_value,
            "target_key": target_key,
            "target_value": target_value,
        }
        if any(value.ndim != 4 for value in tensors.values()) or len({
            value.shape for value in tensors.values()
        }) != 1:
            raise ValueError(
                "Dense counterfactual B0 K/V must share [B,L,H,D]"
            )
        batch, length = source_key.shape[:2]
        if selection_weight.shape != (batch, length):
            raise ValueError(
                "Counterfactual staging weight must align with Q/K/V"
            )
        if length % self.tokens_per_frame:
            raise ValueError(
                "Counterfactual staging requires complete frames"
            )
        key = (timestep_index, layer)
        if key in self._timestep_staging:
            return False
        frame_count = length // self.tokens_per_frame
        per_frame_budget = min(
            self.max_tokens_per_frame, self.tokens_per_frame
        )
        confidence = selection_weight.detach().float()
        eligible = confidence >= self.min_write_confidence
        score = torch.where(
            eligible, confidence, torch.full_like(confidence, -torch.inf)
        )
        selected_score_frames = []
        selected_index_frames = []
        for frame_index in range(frame_count):
            left = frame_index * self.tokens_per_frame
            right = left + self.tokens_per_frame
            eligible_count = int(eligible[:, left:right].sum(dim=-1).max())
            # Keep one invalid sentinel slot when a frame has no authorized
            # token. This preserves a rectangular tensor while avoiding the
            # full per-frame budget for sparse automatic-owner writes.
            stage_budget = max(1, min(per_frame_budget, eligible_count))
            selected_score, selected_index = score[:, left:right].topk(
                stage_budget, dim=-1
            )
            selected_score_frames.append(selected_score)
            selected_index_frames.append(selected_index + left)
        selected_score = torch.cat(selected_score_frames, dim=-1)
        selected_index = torch.cat(selected_index_frames, dim=-1)

        def gather(value):
            gather_index = selected_index[:, :, None, None].expand(
                -1, -1, value.shape[2], value.shape[3]
            )
            return value.gather(1, gather_index).detach().to(
                device="cpu"
            ).clone()

        self._timestep_staging[key] = {
            name: gather(value)
            for name, value in tensors.items()
        }
        self._timestep_staging[key].update({
            "token_index": selected_index.detach().cpu().clone(),
            "support": torch.isfinite(selected_score).detach().cpu().clone(),
        })
        return True

    def read_timestep_counterfactual(
        self, layer: int, timestep_index: int
    ) -> TimestepCounterfactualFrame | None:
        return self._timestep_bank.get(
            (int(timestep_index), int(layer))
        )

    @torch.no_grad()
    def _freeze_timestep_counterfactual_bank(self) -> None:
        if not self.timestep_counterfactual_memory:
            return
        if self._timestep_bank_frozen:
            return
        if not self.has_canonical():
            raise RuntimeError(
                "Cannot freeze timestep memory before canonical commit"
            )
        for (timestep_index, layer), dense in self._timestep_staging.items():
            canonical = self._canonical[layer]
            token_index = canonical.token_index.to(device="cpu")
            if dense["source_key"].shape[0] != token_index.shape[0]:
                raise ValueError(
                    "Staged timestep state does not align with canonical B0"
                )
            staged_index = dense["token_index"]
            matches = token_index[:, :, None] == staged_index[:, None, :]
            found = matches.any(dim=-1)
            staged_slot = matches.float().argmax(dim=-1)

            def gather(value):
                gather_index = staged_slot[:, :, None, None].expand(
                    -1, -1, value.shape[2], value.shape[3]
                )
                return value.gather(1, gather_index).clone()

            frozen = TimestepCounterfactualFrame(
                source_key=gather(dense["source_key"]),
                source_value=gather(dense["source_value"]),
                target_key=gather(dense["target_key"]),
                target_value=gather(dense["target_value"]),
                token_index=token_index.clone(),
                support=(
                    canonical.support.to(device="cpu").bool()
                    & found
                    & dense["support"].gather(1, staged_slot).bool()
                ).clone(),
                frame_count=canonical.frame_count,
            )
            frozen.validate()
            self._timestep_bank[(timestep_index, layer)] = frozen
        missing_layers = set(self.layers).difference(
            layer for _, layer in self._timestep_bank
        )
        if missing_layers:
            raise RuntimeError(
                "No B0 timestep state was captured for native layers "
                f"{sorted(missing_layers)}"
            )
        self._timestep_staging.clear()
        self._timestep_bank_frozen = True

    @staticmethod
    def _sample_flow_at_coordinates(
        value: torch.Tensor, coordinates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bilinearly sample [B,C,H,W] at sparse XY coordinates."""
        if value.ndim != 4 or coordinates.ndim != 3:
            raise ValueError(
                "Sparse flow sampling expects [B,C,H,W] and [B,K,2]"
            )
        if value.shape[0] != coordinates.shape[0]:
            raise ValueError(
                "Sparse flow sampling batch dimensions must align"
            )
        height, width = value.shape[-2:]
        x = coordinates[..., 0]
        y = coordinates[..., 1]
        valid = (
            (x >= 0.0) & (x <= max(width - 1, 0))
            & (y >= 0.0) & (y <= max(height - 1, 0))
        )
        grid_x = (
            2.0 * x / float(width - 1) - 1.0
            if width > 1 else torch.zeros_like(x)
        )
        grid_y = (
            2.0 * y / float(height - 1) - 1.0
            if height > 1 else torch.zeros_like(y)
        )
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(2)
        sampled = torch.nn.functional.grid_sample(
            value.float(), grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        ).squeeze(-1).transpose(1, 2)
        return sampled, valid

    @torch.no_grad()
    def _advance_canonical_coordinates(
        self, state: dict, target_index: int,
        *, spatial_shape: tuple[int, int], device: torch.device,
    ) -> dict:
        source_index = int(state["frame_index"])
        target_index = int(target_index)
        if target_index != source_index + 1:
            raise ValueError(
                "Canonical coordinates require consecutive source flow: "
                f"state={source_index}, target={target_index}"
            )
        transition = self.source_flow_cache.transition(
            source_index, target_index, size=spatial_shape, device=device
        )
        coordinates = state["coordinates"].to(device=device).float()
        sampled_flow, valid = self._sample_flow_at_coordinates(
            transition.forward_flow, coordinates
        )
        sampled_confidence, confidence_valid = (
            self._sample_flow_at_coordinates(
                transition.forward_confidence, coordinates
            )
        )
        next_coordinates = coordinates + sampled_flow
        height, width = spatial_shape
        endpoint_valid = (
            (next_coordinates[..., 0] >= 0.0)
            & (next_coordinates[..., 0] <= width - 1)
            & (next_coordinates[..., 1] >= 0.0)
            & (next_coordinates[..., 1] <= height - 1)
        )
        support = (
            state["support"].to(device=device).bool()
            & valid & confidence_valid & endpoint_valid
        )
        # Confidence is the weakest verified edge on the B0-to-current path.
        # A minimum retains a meaningful long-range cycle score without the
        # artificial exponential decay caused by multiplying every edge.
        confidence = torch.minimum(
            state["confidence"].to(device=device).float(),
            sampled_confidence[..., 0].clamp(0.0, 1.0),
        ) * support.float()
        return {
            "coordinates": next_coordinates,
            "support": support,
            "confidence": confidence,
            "frame_index": target_index,
        }
    @torch.no_grad()
    def _initialize_canonical_coordinates(
        self, *, frame_indices: tuple[int, ...],
        spatial_shape: tuple[int, int], device: torch.device,
    ) -> None:
        if not self.timestep_counterfactual_memory:
            return
        if len(frame_indices) != next(iter(self._canonical.values())).frame_count:
            raise ValueError(
                "Canonical coordinate timestamps must align with B0"
            )
        height, width = spatial_shape
        if height * width != self.tokens_per_frame:
            raise ValueError(
                "Canonical coordinate grid differs from native token grid"
            )
        final_index = int(frame_indices[-1])
        for layer, canonical in self._canonical.items():
            token_index = canonical.token_index.to(device=device).long()
            source_frame = torch.div(
                token_index, self.tokens_per_frame, rounding_mode="floor"
            )
            spatial_index = token_index.remainder(self.tokens_per_frame)
            coordinates = torch.stack((
                spatial_index.remainder(width),
                torch.div(spatial_index, width, rounding_mode="floor"),
            ), dim=-1).float()
            support = canonical.support.to(device=device).bool()
            confidence = support.float()
            # Slots originate in different ignition views. Transport each
            # subset to the common final-B0 timestamp before the first read.
            final_coordinates = torch.zeros_like(coordinates)
            final_support = torch.zeros_like(support)
            final_confidence = torch.zeros_like(confidence)
            for local_frame, global_frame in enumerate(frame_indices):
                slot_mask = support & (source_frame == local_frame)
                state = {
                    "coordinates": coordinates,
                    "support": slot_mask,
                    "confidence": slot_mask.float(),
                    "frame_index": int(global_frame),
                }
                for target_index in range(int(global_frame) + 1, final_index + 1):
                    state = self._advance_canonical_coordinates(
                        state, target_index, spatial_shape=spatial_shape,
                        device=device,
                    )
                final_coordinates = torch.where(
                    slot_mask[..., None], state["coordinates"],
                    final_coordinates,
                )
                final_support |= state["support"]
                final_confidence = torch.where(
                    slot_mask, state["confidence"], final_confidence
                )
            self._canonical_coordinate_state[layer] = {
                "coordinates": final_coordinates.detach().cpu(),
                "support": final_support.detach().cpu(),
                "confidence": final_confidence.detach().cpu(),
                "frame_index": final_index,
            }

    @torch.no_grad()
    def prepare_canonical_correspondence(
        self, *, frame_indices: Iterable[int],
        spatial_shape: tuple[int, int], device: torch.device,
    ) -> Dict[int, NativeCanonicalCorrespondence]:
        """Transport every frozen B0 slot to each current frame."""
        if (
            not self.timestep_counterfactual_memory
            or not self._canonical_coordinate_state
        ):
            self._prepared_canonical_correspondence = {}
            self._prepared_canonical_coordinate_state = {}
            return {}
        indices = tuple(int(value) for value in frame_indices)
        if not indices:
            raise ValueError(
                "Canonical correspondence read requires frame indices"
            )
        prepared = {}
        final_states = {}
        for layer, persistent in self._canonical_coordinate_state.items():
            state = {
                name: (value.to(device=device) if torch.is_tensor(value) else value)
                for name, value in persistent.items()
            }
            mapped_frames = []
            support_frames = []
            confidence_frames = []
            for target_index in indices:
                state = self._advance_canonical_coordinates(
                    state, target_index, spatial_shape=spatial_shape,
                    device=device,
                )
                coordinates = state["coordinates"]
                height, width = spatial_shape
                rounded_x = coordinates[..., 0].round().long().clamp(
                    0, width - 1
                )
                rounded_y = coordinates[..., 1].round().long().clamp(
                    0, height - 1
                )
                mapped_frames.append(
                    len(mapped_frames) * self.tokens_per_frame
                    + rounded_y * width + rounded_x
                )
                hard_support = (
                    state["support"]
                    & (state["confidence"] >= self.flow_min_confidence)
                )
                support_frames.append(hard_support)
                confidence_frames.append(
                    state["confidence"] * hard_support.float()
                )
            correspondence = NativeCanonicalCorrespondence(
                current_index=torch.stack(mapped_frames, dim=1),
                support=torch.stack(support_frames, dim=1),
                confidence=torch.stack(confidence_frames, dim=1),
                frame_count=len(indices),
            )
            correspondence.validate()
            prepared[layer] = correspondence
            final_states[layer] = {
                name: (value.detach().cpu() if torch.is_tensor(value) else value)
                for name, value in state.items()
            }
        self._prepared_canonical_correspondence = prepared
        self._prepared_canonical_coordinate_state = final_states
        return prepared

    def timestep_bank_statistics(self) -> dict[str, int]:
        """Return lightweight, allocation-free diagnostics for TCCM."""
        return {
            "staged_entries": len(self._timestep_staging),
            "frozen_entries": len(self._timestep_bank),
            "timesteps": len({key[0] for key in self._timestep_bank}),
            "layers": len({key[1] for key in self._timestep_bank}),
            "supported_slots": sum(
                int(frame.support.sum().item())
                for frame in self._timestep_bank.values()
            ),
            "total_slots": sum(
                frame.support.numel()
                for frame in self._timestep_bank.values()
            ),
            "frozen": int(self._timestep_bank_frozen),
        }

    @staticmethod
    def _flow_spatial_view(
        value: torch.Tensor, spatial_shape: tuple[int, int]
    ) -> torch.Tensor:
        batch, length, heads, head_dim = value.shape
        height, width = spatial_shape
        if length != height * width:
            raise ValueError(
                "Flow ledger frame does not match its spatial grid"
            )
        return value.permute(0, 2, 3, 1).reshape(
            batch, heads * head_dim, height, width
        )

    @staticmethod
    def _flow_token_view(
        value: torch.Tensor, *, heads: int, head_dim: int
    ) -> torch.Tensor:
        batch, _, height, width = value.shape
        return value.reshape(batch, heads, head_dim, height * width).permute(
            0, 3, 1, 2
        ).contiguous()

    @torch.no_grad()
    def prepare_flow_read(
        self,
        *,
        frame_indices: Iterable[int],
        spatial_shape: tuple[int, int],
        device: torch.device,
    ) -> Dict[int, NativeFlowResidualFrame]:
        """Transport the last trusted V residual into a new causal block.

        The correspondence is fixed by clean-source optical flow, not by a
        generated target key.  This method is read-only: it prepares a block
        prediction but advances persistent state only after ``commit``.
        """
        from pipeline.motion.flow_geometry import (
            forward_splat,
            warp_with_backward_flow,
        )
        if not self.flow_indexed_residual_ledger or not self._flow_state:
            self._prepared_flow_read = {}
            self._prepared_flow_indices = None
            return {}
        indices = tuple(int(value) for value in frame_indices)
        if not indices:
            raise ValueError("Flow ledger read requires frame indices")
        if self._flow_state_index is None:
            raise RuntimeError("Flow ledger state has no source timestamp")
        if indices[0] != self._flow_state_index + 1:
            raise ValueError(
                "Flow ledger requires consecutive source latent frames: "
                f"state={self._flow_state_index}, read={indices[0]}"
            )
        if any(right != left + 1 for left, right in zip(indices, indices[1:])):
            raise ValueError("Flow ledger frame indices must be consecutive")

        prepared: Dict[int, NativeFlowResidualFrame] = {}
        for layer, state in self._flow_state.items():
            state.validate()
            if state.frame_count != 1:
                raise ValueError("Persistent flow state must contain one frame")
            heads, head_dim = state.value_residual.shape[2:]
            residual = self._flow_spatial_view(
                state.value_residual.to(device=device), spatial_shape
            ).float()
            support = state.support.to(device=device).float().reshape(
                state.support.shape[0], 1, *spatial_shape
            )
            confidence = state.confidence.to(device=device).float().reshape(
                state.confidence.shape[0], 1, *spatial_shape
            )
            if self.decoupled_flow_trust:
                appearance_trust = (
                    state.appearance_trust
                    if state.appearance_trust is not None
                    else state.confidence
                ).to(device=device).float().reshape(
                    state.confidence.shape[0], 1, *spatial_shape
                )
                # Transport reliability is local to this read transaction.
                # It restarts at the committed source coordinate instead of
                # becoming a recursively decaying property of appearance.
                transport_confidence = support.clone()
            else:
                appearance_trust = confidence
                transport_confidence = confidence
            source_index = self._flow_state_index
            residual_frames = []
            support_frames = []
            confidence_frames = []
            appearance_frames = []
            transport_frames = []
            for target_index in indices:
                transition = self.source_flow_cache.transition(
                    source_index, target_index, size=spatial_shape, device=device
                )
                pulled_residual_numerator, pull_valid = warp_with_backward_flow(
                    residual * support
                    if self.decoupled_flow_trust else residual,
                    transition.backward_flow,
                )
                pulled_support, _ = warp_with_backward_flow(
                    support, transition.backward_flow
                )
                pulled_confidence, _ = warp_with_backward_flow(
                    confidence, transition.backward_flow
                )
                pushed_residual, push_coverage = forward_splat(
                    residual, transition.forward_flow,
                    weight=transport_confidence * support,
                )
                pushed_support, _ = forward_splat(
                    support, transition.forward_flow,
                    weight=transition.forward_confidence,
                )
                if self.decoupled_flow_trust:
                    # Treat support occupancy and payload attributes as two
                    # different quantities.  A plain bilinear pull mixes the
                    # zero-filled exterior into residual/trust values at a
                    # moving boundary.  Persisting that attenuated value at
                    # every commit causes identity strength to shrink even
                    # when the flow coordinate remains correct.  Normalize
                    # attributes by transported support so interpolation only
                    # changes occupancy, not the appearance payload itself.
                    pulled_residual = (
                        pulled_residual_numerator
                        / pulled_support.clamp_min(1e-6)
                    )
                    pulled_appearance_numerator, _ = warp_with_backward_flow(
                        appearance_trust * support,
                        transition.backward_flow,
                    )
                    pulled_appearance = (
                        pulled_appearance_numerator
                        / pulled_support.clamp_min(1e-6)
                    )
                    pulled_transport_numerator, _ = warp_with_backward_flow(
                        transport_confidence * support,
                        transition.backward_flow,
                    )
                    pulled_transport = (
                        pulled_transport_numerator
                        / pulled_support.clamp_min(1e-6)
                    )
                    pushed_appearance, _ = forward_splat(
                        appearance_trust, transition.forward_flow,
                        weight=transport_confidence * support,
                    )
                    pushed_transport, _ = forward_splat(
                        transport_confidence, transition.forward_flow,
                        weight=support,
                    )
                    pushed_transition_confidence, _ = forward_splat(
                        transition.forward_confidence,
                        transition.forward_flow,
                        weight=support,
                    )
                    pull_transport_confidence = (
                        pulled_transport
                        * transition.backward_confidence
                        * pull_valid.float()
                    ).clamp(0.0, 1.0)
                    push_transport_confidence = (
                        pushed_transport * pushed_transition_confidence
                        * (push_coverage > 0.0).float()
                    ).clamp(0.0, 1.0)
                    pull_confidence = (
                        pulled_appearance * pull_transport_confidence
                    ).clamp(0.0, 1.0)
                    push_confidence = (
                        pushed_appearance * push_transport_confidence
                    ).clamp(0.0, 1.0)
                else:
                    pulled_residual = pulled_residual_numerator
                    pull_confidence = (
                        pulled_confidence
                        * transition.backward_confidence
                        * pull_valid.float()
                    ).clamp(0.0, 1.0)
                    push_confidence = torch.minimum(
                        push_coverage, torch.ones_like(push_coverage)
                    ).clamp(0.0, 1.0)
                    pulled_appearance = pulled_confidence
                    pushed_appearance = push_confidence
                    pull_transport_confidence = pull_confidence
                    push_transport_confidence = push_confidence
                use_push = (
                    pull_transport_confidence < self.flow_min_confidence
                    if self.decoupled_flow_trust
                    else pull_confidence < self.flow_min_confidence
                )
                residual = torch.where(
                    use_push.expand_as(pulled_residual),
                    pushed_residual, pulled_residual,
                )
                appearance_trust = torch.where(
                    use_push, pushed_appearance, pulled_appearance
                ).clamp(0.0, 1.0)
                transport_confidence = torch.where(
                    use_push,
                    push_transport_confidence,
                    pull_transport_confidence,
                ).clamp(0.0, 1.0)
                confidence = torch.where(
                    use_push, push_confidence, pull_confidence
                ).clamp(0.0, 1.0)
                support = torch.where(
                    use_push, pushed_support, pulled_support
                ).clamp(0.0, 1.0)
                hard_support = (
                    (support >= 0.25)
                    & (confidence >= self.flow_min_confidence)
                )
                residual = residual * hard_support.float()
                residual_frames.append(self._flow_token_view(
                    residual, heads=heads, head_dim=head_dim
                ))
                support_frames.append(hard_support.flatten(1))
                confidence_frames.append(
                    (confidence * hard_support.float()).flatten(1)
                )
                appearance_frames.append(
                    (appearance_trust * hard_support.float()).flatten(1)
                )
                transport_frames.append(
                    (transport_confidence * hard_support.float()).flatten(1)
                )
                source_index = target_index
            flow_read = NativeFlowResidualFrame(
                value_residual=torch.cat(residual_frames, dim=1),
                support=torch.cat(support_frames, dim=1).bool(),
                confidence=torch.cat(confidence_frames, dim=1),
                frame_count=len(indices),
                appearance_trust=(
                    torch.cat(appearance_frames, dim=1)
                    if self.decoupled_flow_trust else None
                ),
                transport_confidence=(
                    torch.cat(transport_frames, dim=1)
                    if self.decoupled_flow_trust else None
                ),
            )
            flow_read.validate()
            prepared[layer] = flow_read
        self._prepared_flow_read = prepared
        self._prepared_flow_indices = indices
        return prepared

    @staticmethod
    def _current_native(cache, layer):
        state = cache[layer]
        length = int(state.get("num_new_tokens", 0))
        end = int(state["local_end_index"].item())
        if length <= 0 or end < length:
            raise ValueError("Native KV commit requires a completed block")
        return (
            state["k"][:, end - length:end],
            state["v"][:, end - length:end],
        )

    def _select_frame(
        self, source_key, source_value, target_key, target_value,
        write_confidence,
    ) -> NativeKVFrame:
        batch, length = write_confidence.shape
        if source_key.shape[:2] != (batch, length):
            raise ValueError(
                "Native history confidence must align with current K/V"
            )
        if length % self.tokens_per_frame:
            raise ValueError(
                "Native history blocks must contain complete frames"
            )
        # Retain role-approved tokens across the complete ignition block.
        # Their original indices preserve within-block time and space.
        confidence = write_confidence.detach().float()
        eligible = confidence >= self.min_write_confidence
        frame_count = length // self.tokens_per_frame
        score = torch.where(
            eligible, confidence, torch.full_like(confidence, -torch.inf)
        )
        if self.multiframe_identity_sink:
            # The immutable sink represents every ignition view separately.
            # A global top-k can spend the complete budget on one easy frame,
            # defeating the intended multi-pose identity bank.  Per-frame
            # selection changes only the new 962 mode; prior baselines retain
            # their exact global selection semantics.
            per_frame_budget = min(
                self.max_tokens_per_frame, self.tokens_per_frame
            )
            selected_score_frames = []
            selected_index_frames = []
            for frame_index in range(frame_count):
                left = frame_index * self.tokens_per_frame
                right = left + self.tokens_per_frame
                frame_score, frame_index_local = score[:, left:right].topk(
                    per_frame_budget, dim=-1
                )
                selected_score_frames.append(frame_score)
                selected_index_frames.append(frame_index_local + left)
            selected_score = torch.cat(selected_score_frames, dim=-1)
            selected_index = torch.cat(selected_index_frames, dim=-1)
        else:
            budget = min(
                self.max_tokens_per_frame * frame_count, length
            )
            selected_score, selected_index = score.topk(budget, dim=-1)
        selected_support = torch.isfinite(selected_score)

        def gather(value):
            gather_index = selected_index[:, :, None, None].expand(
                -1, -1, value.shape[2], value.shape[3]
            )
            return value.gather(1, gather_index).detach().clone()

        frame = NativeKVFrame(
            source_key=gather(source_key),
            source_value=gather(source_value),
            target_key=gather(target_key),
            target_value=gather(target_value),
            token_index=selected_index.detach().clone(),
            support=selected_support.detach().clone(),
            frame_count=frame_count,
        )
        frame.validate()
        return frame

    def _complete_recent_block(
        self, source_key, source_value, target_key, target_value,
        payload_support=None,
    ) -> NativeKVFrame:
        length = source_key.shape[1]
        if length % self.tokens_per_frame:
            raise ValueError(
                "Recent native history must contain complete frames"
            )
        batch = source_key.shape[0]
        token_index = torch.arange(
            length, device=source_key.device, dtype=torch.long
        )[None].expand(batch, -1).clone()
        support = torch.ones(
            (batch, length),
            dtype=torch.bool,
            device=source_key.device,
        )
        if payload_support is not None and payload_support.shape != support.shape:
            raise ValueError(
                "Dense recent payload support must align with the block"
            )
        frame = NativeKVFrame(
            source_key=source_key.detach().clone(),
            source_value=source_value.detach().clone(),
            target_key=target_key.detach().clone(),
            target_value=target_value.detach().clone(),
            token_index=token_index,
            support=support,
            frame_count=length // self.tokens_per_frame,
            payload_support=(
                None
                if payload_support is None
                else payload_support.detach().bool().clone()
            ),
        )
        frame.validate()
        return frame

    @staticmethod
    def _hold_dense_recent_on_abstention(current, previous, commit_batch):
        """Commit a complete block only when the owner transaction wrote.

        Dense recent K/V is the local pose/scale state, not an object mask.
        The automatic owner evidence authorizes the transaction at batch level;
        if it abstains, the previous clean edited block is retained verbatim.
        Queries are still restricted by the owner and source correspondence at
        read time, so background tokens can never initiate this bridge.
        """
        current.validate()
        if commit_batch.shape != (current.source_key.shape[0],):
            raise ValueError(
                "Dense recent transaction decision must align with the batch"
            )
        held = torch.zeros_like(current.support)
        if previous is None:
            admitted = commit_batch.detach().bool()[:, None]
            initialized = NativeKVFrame(
                source_key=current.source_key,
                source_value=current.source_value,
                target_key=current.target_key,
                target_value=current.target_value,
                token_index=current.token_index,
                support=current.support & admitted,
                frame_count=current.frame_count,
                payload_support=(
                    None
                    if current.payload_support is None
                    else current.payload_support & admitted
                ),
            )
            initialized.validate()
            return initialized, held
        previous.validate()
        hold_batch = ~commit_batch.detach().bool()
        if not hold_batch.any():
            return current, held
        if current.source_key.shape != previous.source_key.shape:
            if bool(hold_batch.all()):
                return previous, previous.support.detach().clone()
            return current, held

        def aligned(value, reference):
            return value.to(device=reference.device, dtype=reference.dtype)

        row_mask = hold_batch[:, None]
        feature_mask = row_mask[:, :, None, None]
        retained = NativeKVFrame(
            source_key=torch.where(
                feature_mask,
                aligned(previous.source_key, current.source_key),
                current.source_key,
            ),
            source_value=torch.where(
                feature_mask,
                aligned(previous.source_value, current.source_value),
                current.source_value,
            ),
            target_key=torch.where(
                feature_mask,
                aligned(previous.target_key, current.target_key),
                current.target_key,
            ),
            target_value=torch.where(
                feature_mask,
                aligned(previous.target_value, current.target_value),
                current.target_value,
            ),
            token_index=torch.where(
                row_mask,
                previous.token_index.to(current.token_index.device),
                current.token_index,
            ),
            support=torch.where(
                row_mask,
                previous.support.to(current.support.device),
                current.support,
            ),
            frame_count=current.frame_count,
            payload_support=(
                None
                if current.payload_support is None
                else torch.where(
                    row_mask,
                    (
                        previous.support
                        if previous.payload_support is None
                        else previous.payload_support
                    ).to(current.payload_support.device),
                    current.payload_support,
                )
            ),
        )
        retained.validate()
        held = row_mask & (
            retained.support
            if retained.payload_support is None
            else retained.payload_support
        )
        return retained, held

    def _persistent_residual_upsert(
        self,
        current: NativeKVFrame,
        previous: NativeKVFrame | None,
        *,
        direct_write: torch.Tensor,
        retention_confidence: torch.Tensor,
        commit_batch: torch.Tensor,
    ) -> tuple[
        NativeKVFrame, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        """Rebase trusted appearance residuals onto current source tokens.

        This is a true token transaction.  A direct, write-approved target
        observation replaces the payload at its current source address.  An
        owner token without a direct write may retain the nearest previously
        authorized payload only after clean-source key correspondence.  The
        retained target V is reconstructed as current source V plus the
        previous target-minus-source residual; current target K is retained,
        so old pose and scale are not copied into the current block.

        ``retention_confidence`` is produced by the automatic hand/flow owner.
        It is read-only authorization: retained tokens never count as new
        writes and cannot expand the persistent state into background.
        """
        current.validate()
        if current.payload_support is None:
            raise ValueError(
                "Persistent residual upsert requires token payload support"
            )
        expected = current.support.shape
        if direct_write.shape != expected:
            raise ValueError(
                "Direct persistent writes must align with current tokens"
            )
        if retention_confidence.shape != expected:
            raise ValueError(
                "Persistent retention confidence must align with tokens"
            )
        if commit_batch.shape != (expected[0],):
            raise ValueError(
                "Persistent commit decisions must align with the batch"
            )

        direct_proposal = direct_write.detach().bool()
        direct = (
            direct_proposal
            & commit_batch.detach().bool()[:, None]
        )
        retained = torch.zeros_like(direct)
        guarded = torch.zeros_like(direct)
        match_similarity = retention_confidence.new_zeros(expected).float()
        residual_consistency = retention_confidence.new_zeros(
            expected
        ).float()
        target_key = current.target_key.detach().clone()
        target_value = current.target_value.detach().clone()

        if previous is not None:
            previous.validate()
            if previous.payload_support is None:
                raise ValueError(
                    "Persistent residual history cannot consume a legacy "
                    "payload tier"
                )
            if current.source_key.shape[0] != previous.source_key.shape[0]:
                raise ValueError(
                    "Persistent residual history batch size changed"
                )

            for batch_index in range(expected[0]):
                owner_index = torch.nonzero(
                    retention_confidence[batch_index].detach().float() > 0.0,
                    as_tuple=False,
                ).flatten()
                previous_index = torch.nonzero(
                    previous.payload_support[batch_index].detach().bool(),
                    as_tuple=False,
                ).flatten()
                if not owner_index.numel() or not previous_index.numel():
                    continue

                current_address = torch.nn.functional.normalize(
                    current.source_key[
                        batch_index, owner_index
                    ].detach().float().flatten(1),
                    dim=-1, eps=1e-6,
                )
                previous_address = torch.nn.functional.normalize(
                    previous.source_key[
                        batch_index, previous_index
                    ].detach().to(
                        device=current.source_key.device,
                        dtype=torch.float32,
                    ).flatten(1),
                    dim=-1, eps=1e-6,
                )
                similarity = current_address @ previous_address.transpose(0, 1)
                if self.last_trusted_residual_lineage:
                    # Mutual-nearest-neighbour matching used by 958 is safe
                    # but unnecessarily erases payload whenever a pose change
                    # makes several old addresses prefer the same new token.
                    # Build a deterministic one-to-one bipartite assignment
                    # from each old payload's best few current addresses.  No
                    # payload can clone, but a non-mutual second-best match can
                    # keep the lineage alive.
                    candidate_count = min(
                        4, int(owner_index.numel())
                    )
                    candidate_similarity, candidate_current = (
                        similarity.transpose(0, 1).topk(
                            candidate_count, dim=-1
                        )
                    )
                    flat_similarity = candidate_similarity.flatten()
                    flat_current = candidate_current.flatten()
                    flat_previous = torch.arange(
                        previous_index.numel(),
                        device=similarity.device,
                    )[:, None].expand(-1, candidate_count).flatten()
                    flat_similarity_cpu = flat_similarity.detach().cpu()
                    flat_current_cpu = flat_current.detach().cpu()
                    flat_previous_cpu = flat_previous.detach().cpu()
                    order = flat_similarity_cpu.argsort(descending=True)
                    assigned_current = set()
                    assigned_previous = set()
                    matched_current_local = []
                    matched_previous_local = []
                    matched_similarity = []
                    for candidate in order.tolist():
                        score = float(flat_similarity_cpu[candidate].item())
                        if score < self.min_lineage_similarity:
                            break
                        current_local = int(
                            flat_current_cpu[candidate].item()
                        )
                        previous_local = int(
                            flat_previous_cpu[candidate].item()
                        )
                        if (
                            current_local in assigned_current
                            or previous_local in assigned_previous
                        ):
                            continue
                        assigned_current.add(current_local)
                        assigned_previous.add(previous_local)
                        matched_current_local.append(current_local)
                        matched_previous_local.append(previous_local)
                        matched_similarity.append(score)
                    if not matched_current_local:
                        continue
                    accepted_current = owner_index[torch.tensor(
                        matched_current_local,
                        device=owner_index.device, dtype=torch.long,
                    )]
                    accepted_previous = previous_index[torch.tensor(
                        matched_previous_local,
                        device=previous_index.device, dtype=torch.long,
                    )]
                    accepted_similarity = torch.tensor(
                        matched_similarity,
                        device=match_similarity.device,
                        dtype=match_similarity.dtype,
                    )
                else:
                    best_similarity, best_local_index = similarity.max(dim=-1)
                    # 958 baseline: strict mutual nearest-neighbour matching.
                    previous_best_current = similarity.argmax(dim=0)
                    owner_local_index = torch.arange(
                        owner_index.numel(), device=owner_index.device
                    )
                    mutual = (
                        previous_best_current[best_local_index]
                        == owner_local_index
                    )
                    accepted = (
                        torch.isfinite(best_similarity)
                        & (best_similarity >= self.min_lineage_similarity)
                        & mutual
                    )
                    if not accepted.any():
                        continue
                    accepted_current = owner_index[accepted]
                    accepted_previous = previous_index[
                        best_local_index[accepted]
                    ]
                    accepted_similarity = best_similarity[accepted]

                if not accepted_current.numel():
                    continue
                match_similarity[
                    batch_index, accepted_current
                ] = accepted_similarity

                previous_value_residual = (
                    previous.target_value[
                        batch_index, accepted_previous
                    ].to(
                        device=current.target_value.device,
                        dtype=torch.float32,
                    )
                    - previous.source_value[
                        batch_index, accepted_previous
                    ].to(
                        device=current.target_value.device,
                        dtype=torch.float32,
                    )
                )
                current_value_residual = (
                    current.target_value[
                        batch_index, accepted_current
                    ].detach().float()
                    - current.source_value[
                        batch_index, accepted_current
                    ].detach().float()
                )
                previous_flat = previous_value_residual.flatten(1)
                current_flat = current_value_residual.flatten(1)
                previous_norm = previous_flat.norm(dim=-1)
                current_norm = current_flat.norm(dim=-1)
                residual_cosine = (
                    torch.nn.functional.normalize(
                        previous_flat, dim=-1, eps=1e-6
                    )
                    * torch.nn.functional.normalize(
                        current_flat, dim=-1, eps=1e-6
                    )
                ).sum(dim=-1)
                magnitude_ratio = (
                    current_norm / previous_norm.clamp_min(1e-6)
                )
                residual_consistency[
                    batch_index, accepted_current
                ] = (
                    residual_cosine.clamp(-1.0, 1.0)
                    * magnitude_ratio.clamp(max=1.0)
                )

                matched_direct_proposal = direct_proposal[
                    batch_index, accepted_current
                ]
                if self.last_trusted_residual_lineage:
                    trusted_update = (
                        matched_direct_proposal
                        & commit_batch[batch_index].detach().bool()
                        & (
                            residual_cosine
                            >= self.residual_update_min_cosine
                        )
                        & (
                            magnitude_ratio
                            >= self.residual_update_min_magnitude_ratio
                        )
                    )
                    guarded_match = (
                        matched_direct_proposal & ~trusted_update
                    )
                    guarded[batch_index, accepted_current] = guarded_match
                    direct[batch_index, accepted_current] = trusted_update
                else:
                    guarded_match = torch.zeros_like(
                        matched_direct_proposal
                    )

                retain_match = ~direct[batch_index, accepted_current]
                retained[batch_index, accepted_current] = retain_match
                # The current target K already carries current pose/geometry.
                # Only V appearance residual is transported. Copying an old K
                # residual would alter attention correspondence and can import
                # the previous block's pose or scale.
                retained_current = accepted_current[retain_match]
                retained_residual = previous_value_residual[retain_match]
                if retained_current.numel():
                    target_value[batch_index, retained_current] = (
                        current.source_value[
                            batch_index, retained_current
                        ].float()
                        + retained_residual
                    ).to(target_value.dtype)

        payload_support = direct | retained
        # Unauthorized slots are never readable, but rebasing them to source
        # makes the representation fail closed if a future caller mishandles
        # the support tensor.
        payload_mask = payload_support[:, :, None, None]
        target_key = torch.where(
            payload_mask, target_key, current.source_key
        )
        target_value = torch.where(
            payload_mask, target_value, current.source_value
        )
        result = NativeKVFrame(
            source_key=current.source_key,
            source_value=current.source_value,
            target_key=target_key.detach().clone(),
            target_value=target_value.detach().clone(),
            token_index=current.token_index,
            support=current.support,
            frame_count=current.frame_count,
            payload_support=payload_support.detach().clone(),
            residual_rebased_payload=True,
        )
        result.validate()
        return (
            result, direct, retained, match_similarity, guarded,
            residual_consistency,
        )

    def _flow_residual_upsert(
        self,
        current: NativeKVFrame,
        transported: NativeFlowResidualFrame | None,
        *,
        direct_write: torch.Tensor,
        retention_confidence: torch.Tensor,
        commit_batch: torch.Tensor,
    ) -> tuple[
        NativeKVFrame, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        """Update a residual ledger at RAFT-transported token addresses."""
        current.validate()
        expected = current.support.shape
        if direct_write.shape != expected or retention_confidence.shape != expected:
            raise ValueError(
                "Flow residual transaction must align with current tokens"
            )
        if commit_batch.shape != (expected[0],):
            raise ValueError(
                "Flow residual commit decisions must align with the batch"
            )
        if transported is not None:
            transported.validate()
            if transported.value_residual.shape != current.target_value.shape:
                raise ValueError(
                    "Transported flow residual must align with current V"
                )

        direct_proposal = direct_write.detach().bool()
        direct = direct_proposal & commit_batch.detach().bool()[:, None]
        retained = torch.zeros_like(direct)
        guarded = torch.zeros_like(direct)
        match_confidence = retention_confidence.new_zeros(expected).float()
        residual_consistency = retention_confidence.new_zeros(expected).float()
        target_value = current.target_value.detach().clone()

        if transported is not None:
            transported_support = (
                transported.support.to(direct.device).bool()
                & (retention_confidence.detach().float() > 0.0)
            )
            transported_confidence = transported.confidence.to(
                device=match_confidence.device, dtype=match_confidence.dtype
            )
            match_confidence = torch.where(
                transported_support, transported_confidence, match_confidence
            )
            previous_residual = transported.value_residual.to(
                device=current.target_value.device, dtype=torch.float32
            )
            current_residual = (
                current.target_value.detach().float()
                - current.source_value.detach().float()
            )
            previous_flat = previous_residual.flatten(2)
            current_flat = current_residual.flatten(2)
            previous_norm = previous_flat.norm(dim=-1)
            current_norm = current_flat.norm(dim=-1)
            residual_cosine = (
                torch.nn.functional.normalize(
                    previous_flat, dim=-1, eps=1e-6
                )
                * torch.nn.functional.normalize(
                    current_flat, dim=-1, eps=1e-6
                )
            ).sum(dim=-1).clamp(-1.0, 1.0)
            magnitude_ratio = current_norm / previous_norm.clamp_min(1e-6)
            residual_consistency = torch.where(
                transported_support,
                residual_cosine * magnitude_ratio.clamp(max=1.0),
                residual_consistency,
            )
            if self.last_trusted_residual_lineage:
                trusted_update = (
                    direct
                    & (
                        (~transported_support)
                        | (
                            (residual_cosine >= self.residual_update_min_cosine)
                            & (
                                magnitude_ratio
                                >= self.residual_update_min_magnitude_ratio
                            )
                        )
                    )
                )
                guarded = direct_proposal & transported_support & ~trusted_update
                direct = trusted_update
            retained = transported_support & ~direct
            retained_mask = retained[:, :, None, None]
            target_value = torch.where(
                retained_mask,
                (current.source_value.float() + previous_residual).to(
                    current.target_value.dtype
                ),
                target_value,
            )

        payload_support = direct | retained
        payload_mask = payload_support[:, :, None, None]
        target_value = torch.where(
            payload_mask, target_value, current.source_value
        )
        result = NativeKVFrame(
            source_key=current.source_key,
            source_value=current.source_value,
            # The flow-indexed reader never uses target K for identity
            # addressing. Keeping current K here preserves compatibility and
            # makes any accidental legacy read carry current geometry only.
            target_key=torch.where(
                payload_mask, current.target_key, current.source_key
            ).detach().clone(),
            target_value=target_value.detach().clone(),
            token_index=current.token_index,
            support=current.support,
            frame_count=current.frame_count,
            payload_support=payload_support.detach().clone(),
            residual_rebased_payload=True,
        )
        result.validate()
        return (
            result, direct, retained, match_confidence, guarded,
            residual_consistency,
        )

    def _commit_flow_state(
        self,
        *,
        layer: int,
        frame: NativeKVFrame,
        frame_indices: tuple[int, ...],
        spatial_shape: tuple[int, int],
        write_confidence: torch.Tensor,
        direct_support: torch.Tensor,
        retained_support: torch.Tensor,
    ) -> NativeFlowResidualFrame:
        frame.validate()
        if frame.frame_count != len(frame_indices):
            raise ValueError(
                "Flow state timestamps must align with the committed block"
            )
        if self.tokens_per_frame != spatial_shape[0] * spatial_shape[1]:
            raise ValueError(
                "Flow state spatial grid differs from native token grid"
            )
        start = (frame.frame_count - 1) * self.tokens_per_frame
        end = start + self.tokens_per_frame
        support = (
            frame.support if frame.payload_support is None
            else frame.payload_support
        )[:, start:end].detach().bool().clone()
        direct = direct_support[:, start:end].detach().bool()
        retained = retained_support[:, start:end].detach().bool()
        if direct.shape != support.shape or retained.shape != support.shape:
            raise ValueError(
                "Flow state transaction maps must align with the last frame"
            )
        current_confidence = (
            write_confidence[:, start:end].detach().float().clamp(0.0, 1.0)
        )
        confidence = torch.where(
            direct, current_confidence, torch.zeros_like(current_confidence)
        )
        prepared = self._prepared_flow_read.get(layer)
        if prepared is not None:
            transported_confidence = prepared.confidence[
                :, start:end
            ].to(confidence)
            # A retained payload was not observed or rewritten in the current
            # target.  Its confidence must therefore come from the exact
            # source-flow transport that supplied its value residual.  Using
            # the current soft write proposal here made confidence decay even
            # when the proposal was below the transactional write threshold.
            confidence = torch.where(
                retained,
                transported_confidence,
                confidence,
            )
        appearance_trust = None
        transport_confidence = None
        if self.decoupled_flow_trust:
            appearance_trust = torch.where(
                direct, current_confidence, torch.zeros_like(confidence)
            )
            if prepared is not None:
                prepared_appearance = (
                    prepared.appearance_trust
                    if prepared.appearance_trust is not None
                    else prepared.confidence
                )[:, start:end].to(confidence)
                appearance_trust = torch.where(
                    retained, prepared_appearance, appearance_trust
                )
            appearance_trust = appearance_trust * support.float()
            # At the committed source coordinate there is no outstanding
            # transport uncertainty. The next read starts a fresh local flow
            # path and combines that reliability with appearance_trust.
            transport_confidence = support.float()
            confidence = appearance_trust
        confidence = confidence * support.float()
        residual = (
            frame.target_value[:, start:end].detach().float()
            - frame.source_value[:, start:end].detach().float()
        ) * support[:, :, None, None].float()
        state = NativeFlowResidualFrame(
            value_residual=residual,
            support=support,
            confidence=confidence,
            frame_count=1,
            appearance_trust=appearance_trust,
            transport_confidence=transport_confidence,
        )
        state.validate()
        self._flow_state[layer] = state
        return state

    @staticmethod
    def _hold_recent_on_abstention(current, previous):
        """Abort an empty transaction without erasing the last commit.

        The dense native backbone already contains the complete causal
        context.  This tier is an object-appearance transaction, so an empty
        write means "no update" rather than "replace memory with empty".
        """
        held = torch.zeros_like(current.support)
        if previous is None:
            return current, held
        current.validate()
        previous.validate()
        current_has_evidence = current.support.any(dim=-1)
        previous_has_evidence = previous.support.any(dim=-1)
        hold_batch = ~current_has_evidence & previous_has_evidence
        if not hold_batch.any():
            return current, held
        if current.source_key.shape != previous.source_key.shape:
            if bool(hold_batch.all()):
                return previous, previous.support.detach().clone()
            return current, held

        previous = NativeKVFrame(
            source_key=previous.source_key.to(
                device=current.source_key.device,
                dtype=current.source_key.dtype,
            ),
            source_value=previous.source_value.to(
                device=current.source_value.device,
                dtype=current.source_value.dtype,
            ),
            target_key=previous.target_key.to(
                device=current.target_key.device,
                dtype=current.target_key.dtype,
            ),
            target_value=previous.target_value.to(
                device=current.target_value.device,
                dtype=current.target_value.dtype,
            ),
            token_index=previous.token_index.to(current.token_index.device),
            support=previous.support.to(current.support.device),
            frame_count=previous.frame_count,
            payload_support=(
                None
                if previous.payload_support is None
                else previous.payload_support.to(current.support.device)
            ),
        )
        row_mask = hold_batch[:, None]
        feature_mask = row_mask[:, :, None, None]
        retained = NativeKVFrame(
            source_key=torch.where(
                feature_mask, previous.source_key, current.source_key
            ),
            source_value=torch.where(
                feature_mask, previous.source_value, current.source_value
            ),
            target_key=torch.where(
                feature_mask, previous.target_key, current.target_key
            ),
            target_value=torch.where(
                feature_mask, previous.target_value, current.target_value
            ),
            token_index=torch.where(
                row_mask, previous.token_index, current.token_index
            ),
            support=torch.where(
                row_mask, previous.support, current.support
            ),
            frame_count=current.frame_count,
            payload_support=(
                None
                if current.payload_support is None
                else torch.where(
                    row_mask,
                    (
                        previous.support
                        if previous.payload_support is None
                        else previous.payload_support
                    ),
                    current.payload_support,
                )
            ),
        )
        retained.validate()
        held = row_mask & previous.support
        return retained, held

    def _complete_source_lineage(
        self, source_key, source_value, lineage_confidence, canonical,
        previous_lineage=None,
    ) -> tuple[
        NativeSourceLineageFrame,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        length = source_key.shape[1]
        if length % self.tokens_per_frame:
            raise ValueError(
                "Source lineage blocks must contain complete frames"
            )
        if lineage_confidence.shape != source_key.shape[:2]:
            raise ValueError(
                "Source lineage confidence must align with current K/V"
            )
        confidence = lineage_confidence.detach().float().clamp(0.0, 1.0)
        frame_count = length // self.tokens_per_frame
        budget = min(
            self.max_tokens_per_frame * frame_count, length
        )
        lineage_score = torch.where(
            confidence > 0.0,
            confidence,
            torch.full_like(confidence, -torch.inf),
        )
        selected_score, selected_index = lineage_score.topk(
            budget, dim=-1
        )
        selected_support = torch.isfinite(selected_score)

        def gather(value):
            gather_index = selected_index[:, :, None, None].expand(
                -1, -1, value.shape[2], value.shape[3]
            )
            return value.gather(1, gather_index).detach().clone()

        selected_source_key = gather(source_key)
        selected_source_value = gather(source_value)
        selected_confidence = torch.where(
            selected_support, selected_score, torch.zeros_like(selected_score)
        )

        def normalized(value):
            return torch.nn.functional.normalize(
                value.detach().float().flatten(2), dim=-1, eps=1e-6
            )

        current_address = normalized(selected_source_key)
        canonical_address = normalized(canonical.source_key)
        direct_similarity = torch.einsum(
            "bqd,bkd->bqk", current_address, canonical_address
        ).masked_fill(
            ~canonical.support.detach().bool()[:, None, :], -torch.inf
        )
        direct_score, direct_index = direct_similarity.max(dim=-1)
        direct_valid = (
            torch.isfinite(direct_score)
            & (direct_score >= self.min_lineage_similarity)
        )
        canonical_index = torch.where(
            direct_valid, direct_index, torch.full_like(direct_index, -1)
        )
        match_score = direct_score
        transported = torch.zeros_like(direct_valid)

        if previous_lineage is not None:
            previous_lineage.validate()
            previous_address = normalized(previous_lineage.source_key)
            transport_similarity = torch.einsum(
                "bqd,bkd->bqk", current_address, previous_address
            ).masked_fill(
                ~previous_lineage.support.detach().bool()[:, None, :],
                -torch.inf,
            )
            transport_score, transport_index = transport_similarity.max(
                dim=-1
            )
            inherited_index = previous_lineage.canonical_index.gather(
                1, transport_index
            )
            transport_valid = (
                torch.isfinite(transport_score)
                & (transport_score >= self.min_lineage_similarity)
                & (inherited_index >= 0)
            )
            # Prefer the adjacent clean-source address whenever it is valid:
            # this is the motion bridge.  Direct canonical matching remains
            # a non-recursive re-association fallback after a missed block.
            canonical_index = torch.where(
                transport_valid, inherited_index, canonical_index
            )
            match_score = torch.where(
                transport_valid, transport_score, match_score
            )
            transported = transport_valid

        support = (
            selected_support
            & (canonical_index >= 0)
            & torch.isfinite(match_score)
        )
        canonical_index = torch.where(
            support, canonical_index, torch.full_like(canonical_index, -1)
        )
        frame = NativeSourceLineageFrame(
            source_key=selected_source_key,
            source_value=selected_source_value,
            token_index=selected_index.detach().clone(),
            canonical_index=canonical_index.detach().clone(),
            support=support.detach().clone(),
            confidence=selected_confidence.detach().clone(),
            frame_count=frame_count,
        )
        frame.validate()
        return frame, transported, direct_valid, match_score

    @staticmethod
    def _payload_similarity_diagnostics(
        frame: NativeKVFrame,
        canonical: NativeKVFrame,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Measure a proposed target write without accepting its payload.

        Both measurements are diagnostic only.  They make recursive source
        drift observable in payload-invariant runs while the actual mutable
        state remains clean-source address metadata.
        """
        frame.validate()
        canonical.validate()

        def normalized(value):
            return torch.nn.functional.normalize(
                value.detach().float().flatten(2), dim=-1, eps=1e-6
            )

        target_value = normalized(frame.target_value)
        source_value = normalized(frame.source_value)
        target_source = (target_value * source_value).sum(dim=-1)

        current_address = normalized(frame.source_key)
        canonical_address = normalized(canonical.source_key)
        address_similarity = torch.einsum(
            "bqd,bkd->bqk", current_address, canonical_address
        ).masked_fill(
            ~canonical.support.detach().bool()[:, None, :], -torch.inf
        )
        best_address, best_index = address_similarity.max(dim=-1)
        gather_index = best_index[:, :, None].expand(
            -1, -1, target_value.shape[-1]
        )
        canonical_target = normalized(canonical.target_value).gather(
            1, gather_index
        )
        target_canonical = (target_value * canonical_target).sum(dim=-1)
        valid = (
            frame.support.detach().bool()
            & torch.isfinite(best_address)
        )

        def supported_mean(value):
            return (
                torch.where(valid, value, torch.zeros_like(value)).sum()
                / valid.float().sum().clamp_min(1.0)
            )

        return supported_mean(target_source), supported_mean(
            target_canonical
        )

    @staticmethod
    def _payload_similarity_by_batch(
        frame: NativeKVFrame, canonical: NativeKVFrame
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return source/canonical agreement for each proposed transaction."""
        frame.validate()
        canonical.validate()

        def normalized(value):
            return torch.nn.functional.normalize(
                value.detach().float().flatten(2), dim=-1, eps=1e-6
            )

        target_value = normalized(frame.target_value)
        source_value = normalized(frame.source_value)
        target_source = (target_value * source_value).sum(dim=-1)
        current_address = normalized(frame.source_key)
        canonical_address = normalized(canonical.source_key)
        address_similarity = torch.einsum(
            "bqd,bkd->bqk", current_address, canonical_address
        ).masked_fill(
            ~canonical.support.detach().bool()[:, None, :], -torch.inf
        )
        best_address, best_index = address_similarity.max(dim=-1)
        canonical_target = normalized(canonical.target_value).gather(
            1, best_index[:, :, None].expand(-1, -1, target_value.shape[-1])
        )
        target_canonical = (target_value * canonical_target).sum(dim=-1)
        valid = frame.support.detach().bool() & torch.isfinite(best_address)
        denominator = valid.float().sum(dim=-1).clamp_min(1.0)

        def supported_mean(value):
            return (
                torch.where(valid, value, torch.zeros_like(value)).sum(dim=-1)
                / denominator
            )

        return supported_mean(target_source), supported_mean(target_canonical)

    @staticmethod
    def _edit_residual_consensus_by_batch(
        frame: NativeKVFrame, canonical: NativeKVFrame
    ) -> torch.Tensor:
        """Measure whether a proposed edit still points toward ignition.

        This statistic gates a transaction only; it is never used as an
        attention payload.  Current and canonical target-minus-source vectors
        are compared after source-key correspondence, so a near-source proposal
        has zero confidence while pose is carried by the accepted full target
        K/V rather than by a linear residual.
        """
        frame.validate()
        canonical.validate()

        def flattened(value):
            return value.detach().float().flatten(2)

        current_address = torch.nn.functional.normalize(
            flattened(frame.source_key), dim=-1, eps=1e-6
        )
        canonical_address = torch.nn.functional.normalize(
            flattened(canonical.source_key), dim=-1, eps=1e-6
        )
        address_similarity = torch.einsum(
            "bqd,bkd->bqk", current_address, canonical_address
        ).masked_fill(
            ~canonical.support.detach().bool()[:, None, :], -torch.inf
        )
        best_address, best_index = address_similarity.max(dim=-1)
        gather_index = best_index[:, :, None, None].expand(
            -1, -1, canonical.target_value.shape[2],
            canonical.target_value.shape[3],
        )
        canonical_residual = flattened(
            canonical.target_value.gather(1, gather_index)
            - canonical.source_value.gather(1, gather_index)
        )
        current_residual = flattened(
            frame.target_value - frame.source_value
        )
        canonical_norm = canonical_residual.norm(dim=-1)
        current_norm = current_residual.norm(dim=-1)
        direction = (
            torch.nn.functional.normalize(
                current_residual, dim=-1, eps=1e-6
            )
            * torch.nn.functional.normalize(
                canonical_residual, dim=-1, eps=1e-6
            )
        ).sum(dim=-1).clamp_min(0.0)
        energy = (
            current_norm / canonical_norm.clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        valid = (
            frame.support.detach().bool()
            & torch.isfinite(best_address)
            & (canonical_norm > 1e-6)
        )
        score = direction * energy
        return (
            torch.where(valid, score, torch.zeros_like(score)).sum(dim=-1)
            / valid.float().sum(dim=-1).clamp_min(1.0)
        )

    @staticmethod
    def _edit_residual_consensus_by_token(
        frame: NativeKVFrame, canonical: NativeKVFrame
    ) -> torch.Tensor:
        """Audit every proposed target payload against ignition.

        Clean-source keys provide a dense pose/scale address.  They do not
        authorize the target value at that address.  Authorization requires
        the current target-minus-source residual to keep both the direction
        and non-zero energy of the source-aligned immutable edit residual.
        """
        frame.validate()
        canonical.validate()

        def flattened(value):
            return value.detach().float().flatten(2)

        current_address = torch.nn.functional.normalize(
            flattened(frame.source_key), dim=-1, eps=1e-6
        )
        canonical_address = torch.nn.functional.normalize(
            flattened(canonical.source_key), dim=-1, eps=1e-6
        )
        address_similarity = torch.einsum(
            "bqd,bkd->bqk", current_address, canonical_address
        ).masked_fill(
            ~canonical.support.detach().bool()[:, None, :], -torch.inf
        )
        best_address, best_index = address_similarity.max(dim=-1)
        gather_index = best_index[:, :, None, None].expand(
            -1, -1, canonical.target_value.shape[2],
            canonical.target_value.shape[3],
        )
        canonical_residual = flattened(
            canonical.target_value.gather(1, gather_index)
            - canonical.source_value.gather(1, gather_index)
        )
        current_residual = flattened(
            frame.target_value - frame.source_value
        )
        canonical_norm = canonical_residual.norm(dim=-1)
        current_norm = current_residual.norm(dim=-1)
        direction = (
            torch.nn.functional.normalize(
                current_residual, dim=-1, eps=1e-6
            )
            * torch.nn.functional.normalize(
                canonical_residual, dim=-1, eps=1e-6
            )
        ).sum(dim=-1).clamp_min(0.0)
        energy = (
            current_norm / canonical_norm.clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        valid = (
            torch.isfinite(best_address)
            & (best_address >= 0.0)
            & (canonical_norm > 1e-6)
        )
        return torch.where(
            valid, direction * energy, torch.zeros_like(direction)
        )

    @staticmethod
    def _hold_source_lineage_on_abstention(current, previous):
        """Keep the last valid address row when a transaction abstains.

        A zero-confidence owner observation is a no-op, not evidence that the
        object lineage ceased to exist.  In particular, replacing the mutable
        tier with an empty frame would make the following chunk fall back to
        source appearance.  This hold-last rule changes only source addresses;
        the immutable target payload is never copied or rewritten here.
        """
        held = torch.zeros_like(current.support)
        if previous is None:
            return current, held
        previous.validate()
        current_has_evidence = current.support.any(dim=-1)
        previous_has_evidence = previous.support.any(dim=-1)
        hold_batch = ~current_has_evidence & previous_has_evidence
        if not hold_batch.any():
            return current, held

        if current.source_key.shape != previous.source_key.shape:
            # Frame counts may differ for a partial terminal block.  A full
            # batch abstention can still retain the previous compact table; a
            # mixed batch cannot be represented by one rectangular tensor.
            if bool(hold_batch.all()):
                return previous, previous.support.detach().clone()
            return current, held

        previous = NativeSourceLineageFrame(
            source_key=previous.source_key.to(
                device=current.source_key.device,
                dtype=current.source_key.dtype,
            ),
            source_value=previous.source_value.to(
                device=current.source_value.device,
                dtype=current.source_value.dtype,
            ),
            token_index=previous.token_index.to(current.token_index.device),
            canonical_index=previous.canonical_index.to(
                current.canonical_index.device
            ),
            support=previous.support.to(current.support.device),
            confidence=previous.confidence.to(
                device=current.confidence.device,
                dtype=current.confidence.dtype,
            ),
            frame_count=previous.frame_count,
        )
        row_mask = hold_batch[:, None]
        feature_mask = row_mask[:, :, None, None]
        retained = NativeSourceLineageFrame(
            source_key=torch.where(
                feature_mask, previous.source_key, current.source_key
            ),
            source_value=torch.where(
                feature_mask, previous.source_value, current.source_value
            ),
            token_index=torch.where(
                row_mask, previous.token_index, current.token_index
            ),
            canonical_index=torch.where(
                row_mask, previous.canonical_index, current.canonical_index
            ),
            support=torch.where(
                row_mask, previous.support, current.support
            ),
            confidence=torch.where(
                row_mask, previous.confidence, current.confidence
            ),
            frame_count=current.frame_count,
        )
        retained.validate()
        held = row_mask & previous.support
        return retained, held

    @torch.no_grad()
    def commit(
        self, *, source_kv_cache, target_kv_cache, write_confidence,
        lineage_confidence=None, retention_confidence=None,
        frame_indices=None, spatial_shape=None,
    ):
        if write_confidence.ndim != 2:
            raise ValueError(
                "Native history write confidence must have shape [B,L]"
            )
        if self.payload_invariant_lineage:
            if lineage_confidence is None:
                raise ValueError(
                    "Payload-invariant history requires source lineage "
                    "confidence"
                )
            if lineage_confidence.shape != write_confidence.shape:
                raise ValueError(
                    "Source lineage and canonical write confidence must "
                    "share shape"
                )
        if self.persistent_residual_upsert:
            if retention_confidence is None:
                raise ValueError(
                    "Persistent residual upserts require automatic owner "
                    "retention confidence"
                )
            if retention_confidence.shape != write_confidence.shape:
                raise ValueError(
                    "Persistent retention and write confidence must share "
                    "shape"
                )
        flow_frame_indices = None
        if self.flow_indexed_residual_ledger:
            if frame_indices is None or spatial_shape is None:
                raise ValueError(
                    "Flow-indexed residual commits require source frame "
                    "indices and a spatial token grid"
                )
            flow_frame_indices = tuple(int(value) for value in frame_indices)
            if not flow_frame_indices:
                raise ValueError("Flow-indexed commit has no frame indices")
            if any(
                right != left + 1
                for left, right in zip(
                    flow_frame_indices, flow_frame_indices[1:]
                )
            ):
                raise ValueError(
                    "Flow-indexed commit frame indices must be consecutive"
                )
            expected_frames = write_confidence.shape[1] // self.tokens_per_frame
            if len(flow_frame_indices) != expected_frames:
                raise ValueError(
                    "Flow-indexed commit timestamps do not match KV frames"
                )
            if self._commit_count > 0 and (
                self._prepared_flow_indices != flow_frame_indices
            ):
                raise RuntimeError(
                    "Flow-indexed residual read was not prepared for this block"
                )
        diagnostics = {}
        new_recent = {}
        new_source_lineage = {}
        for layer in self.layers:
            source_key, source_value = self._current_native(
                source_kv_cache, layer
            )
            target_key, target_value = self._current_native(
                target_kv_cache, layer
            )
            frame = self._select_frame(
                source_key, source_value, target_key, target_value,
                write_confidence,
            )
            canonical_for_diagnostics = self._canonical.get(layer, frame)
            (
                candidate_target_source_similarity,
                candidate_target_canonical_similarity,
            ) = self._payload_similarity_diagnostics(
                frame, canonical_for_diagnostics
            )
            held_recent = frame.support.new_zeros(frame.support.shape)
            dense_recent_accepted = frame.support.new_zeros(
                (frame.support.shape[0],), dtype=torch.bool
            )
            dense_recent_residual_consensus = frame.support.new_zeros(
                (frame.support.shape[0],), dtype=torch.float32
            )
            dense_recent_written = frame.support.new_zeros(
                (), dtype=torch.float32
            )
            persistent_direct = write_confidence.new_zeros(
                write_confidence.shape, dtype=torch.bool
            )
            persistent_retained = torch.zeros_like(persistent_direct)
            persistent_similarity = write_confidence.new_zeros(
                write_confidence.shape, dtype=torch.float32
            )
            persistent_guarded = torch.zeros_like(persistent_direct)
            persistent_residual_consistency = write_confidence.new_zeros(
                write_confidence.shape, dtype=torch.float32
            )
            if self.payload_invariant_lineage:
                # The mutable tier carries only clean-source addresses.  In
                # particular, no generated target value from this block can
                # become the next block's appearance payload.
                (
                    new_source_lineage[layer],
                    transported_lineage,
                    direct_lineage,
                    lineage_similarity,
                ) = self._complete_source_lineage(
                    source_key,
                    source_value,
                    lineage_confidence,
                    self._canonical.get(layer, frame),
                    self._source_lineage.get(layer),
                )
                (
                    new_source_lineage[layer],
                    held_lineage,
                ) = self._hold_source_lineage_on_abstention(
                    new_source_lineage[layer],
                    self._source_lineage.get(layer),
                )
                if (
                    lineage_similarity.shape
                    != new_source_lineage[layer].support.shape
                ):
                    # A partial terminal block can have a different compact
                    # lineage width.  If that transaction abstained and held
                    # the prior table, there is no current-block similarity
                    # statistic to report for the held rows.
                    lineage_similarity = (
                        new_source_lineage[layer].confidence.new_zeros(
                            new_source_lineage[layer].support.shape
                        )
                    )
                else:
                    lineage_similarity = torch.where(
                        held_lineage,
                        torch.zeros_like(lineage_similarity),
                        lineage_similarity,
                    )
            else:
                if self.transactional_dense_recent:
                    token_payload_support = (
                        write_confidence.detach().float()
                        >= self.min_write_confidence
                    )
                    dense_recent = self._complete_recent_block(
                        source_key, source_value, target_key, target_value,
                        payload_support=(
                            token_payload_support
                            if self.token_atomic_dense_recent
                            else None
                        ),
                    )
                    owner_wrote = token_payload_support.any(dim=-1)
                    dense_recent_residual_consensus = (
                        self._edit_residual_consensus_by_batch(
                            frame, canonical_for_diagnostics
                        )
                    )
                    dense_recent_accepted = (
                        owner_wrote
                        & (
                            (self._commit_count == 0)
                            | (
                                dense_recent_residual_consensus
                                >= self.dense_recent_min_residual_consensus
                            )
                        )
                    )
                    if self.persistent_residual_upsert:
                        (
                            new_recent[layer],
                            persistent_direct,
                            persistent_retained,
                            persistent_similarity,
                            persistent_guarded,
                            persistent_residual_consistency,
                        ) = (
                            self._flow_residual_upsert(
                                dense_recent,
                                self._prepared_flow_read.get(layer),
                                direct_write=token_payload_support,
                                retention_confidence=(
                                    retention_confidence.detach().float()
                                ),
                                commit_batch=dense_recent_accepted,
                            )
                            if self.flow_indexed_residual_ledger
                            else self._persistent_residual_upsert(
                                dense_recent,
                                self._recent.get(layer),
                                direct_write=token_payload_support,
                                retention_confidence=(
                                    retention_confidence.detach().float()
                                ),
                                commit_batch=dense_recent_accepted,
                            )
                        )
                        held_recent = persistent_retained
                    else:
                        new_recent[layer], held_recent = (
                            self._hold_dense_recent_on_abstention(
                                dense_recent,
                                self._recent.get(layer),
                                dense_recent_accepted,
                            )
                        )
                    dense_recent_written = (
                        (
                            persistent_direct.float().sum(dim=-1)
                            if self.persistent_residual_upsert
                            else (
                                new_recent[layer].support
                                if new_recent[layer].payload_support is None
                                else new_recent[layer].payload_support
                            ).float().sum(dim=-1)
                            * dense_recent_accepted.float()
                        )
                    ).sum()
                elif self.transactional_compact_recent:
                    # A transactional target tier contains exactly the
                    # write-approved object tokens.  The complete block is
                    # already present in native causal attention and must not
                    # bypass the role-specific write contract here.
                    new_recent[layer], held_recent = (
                        self._hold_recent_on_abstention(
                            frame, self._recent.get(layer)
                        )
                    )
                else:
                    # Legacy InfinityEdit-style short tier: retain the
                    # complete immediately preceding final-clean block.
                    new_recent[layer] = self._complete_recent_block(
                        source_key, source_value, target_key, target_value
                    )
            if layer not in self._canonical:
                self._canonical[layer] = frame
            diagnostics[layer] = {
                "written": frame.support.float().sum(),
                "lineage_tokens": (
                    new_source_lineage[layer].support.float().sum()
                    if self.payload_invariant_lineage
                    else frame.support.new_zeros((), dtype=torch.float32)
                ),
                "lineage_transport_tokens": (
                    transported_lineage.float().sum()
                    if self.payload_invariant_lineage
                    else frame.support.new_zeros((), dtype=torch.float32)
                ),
                "lineage_held_tokens": (
                    held_lineage.float().sum()
                    if self.payload_invariant_lineage
                    else frame.support.new_zeros((), dtype=torch.float32)
                ),
                "lineage_direct_tokens": (
                    (direct_lineage & ~transported_lineage).float().sum()
                    if self.payload_invariant_lineage
                    else frame.support.new_zeros((), dtype=torch.float32)
                ),
                "lineage_similarity": (
                    torch.where(
                        new_source_lineage[layer].support,
                        lineage_similarity,
                        torch.zeros_like(lineage_similarity),
                    ).sum()
                    / new_source_lineage[layer].support.float().sum().clamp_min(1.0)
                    if self.payload_invariant_lineage
                    else frame.support.new_zeros((), dtype=torch.float32)
                ),
                "canonical_frozen": torch.tensor(
                    float(self._commit_count > 0),
                    device=write_confidence.device,
                ),
                "recent_held_tokens": held_recent.float().sum(),
                "dense_recent_accepted": (
                    dense_recent_accepted.float().mean()
                ),
                "dense_recent_residual_consensus": (
                    dense_recent_residual_consensus.mean()
                ),
                "candidate_target_source_similarity": (
                    candidate_target_source_similarity
                ),
                "candidate_target_canonical_similarity": (
                    candidate_target_canonical_similarity
                ),
                "mutable_target_payload_written": (
                    frame.support.new_zeros((), dtype=torch.float32)
                    if self.payload_invariant_lineage
                    else (
                        dense_recent_written
                    )
                    if self.transactional_dense_recent
                    else frame.support.float().sum()
                ),
                "mutable_target_payload_authorized": (
                    new_recent[layer].payload_support.float().sum()
                    if (
                        self.transactional_dense_recent
                        and new_recent[layer].payload_support is not None
                    )
                    else frame.support.new_zeros((), dtype=torch.float32)
                ),
                "persistent_direct_support": persistent_direct.detach(),
                "persistent_retained_support": (
                    persistent_retained.detach()
                ),
                "persistent_payload_support": (
                    new_recent[layer].payload_support.detach()
                    if (
                        self.persistent_residual_upsert
                        and new_recent[layer].payload_support is not None
                    )
                    else torch.zeros_like(persistent_direct)
                ),
                "persistent_residual_transport_tokens": (
                    persistent_retained.float().sum()
                ),
                "persistent_residual_transport_similarity": (
                    persistent_similarity.sum()
                    / persistent_retained.float().sum().clamp_min(1.0)
                ),
                "persistent_guarded_update_support": (
                    persistent_guarded.detach()
                ),
                "persistent_guarded_update_tokens": (
                    persistent_guarded.float().sum()
                ),
                "persistent_residual_consistency": (
                    persistent_residual_consistency.detach()
                ),
                "persistent_residual_consistency_on_match": (
                    persistent_residual_consistency.abs().sum()
                    / (
                        persistent_residual_consistency != 0.0
                    ).float().sum().clamp_min(1.0)
                ),
                "flow_indexed_residual_ledger": (
                    write_confidence.new_tensor(
                        float(self.flow_indexed_residual_ledger)
                    )
                ),
            }
            if self.flow_indexed_residual_ledger:
                flow_state = self._commit_flow_state(
                    layer=layer,
                    frame=new_recent[layer],
                    frame_indices=flow_frame_indices,
                    spatial_shape=tuple(int(v) for v in spatial_shape),
                    write_confidence=write_confidence,
                    direct_support=persistent_direct,
                    retained_support=persistent_retained,
                )
                diagnostics[layer].update({
                    "flow_indexed_state_support": (
                        torch.nn.functional.pad(
                            flow_state.support.detach(),
                            (
                                persistent_direct.shape[1]
                                - self.tokens_per_frame,
                                0,
                            ),
                        )
                    ),
                    "flow_indexed_state_confidence": (
                        torch.nn.functional.pad(
                            flow_state.confidence.detach(),
                            (
                                persistent_direct.shape[1]
                                - self.tokens_per_frame,
                                0,
                            ),
                        )
                    ),
                    "flow_indexed_appearance_trust": (
                        torch.nn.functional.pad(
                            (
                                flow_state.appearance_trust
                                if flow_state.appearance_trust is not None
                                else flow_state.confidence
                            ).detach(),
                            (
                                persistent_direct.shape[1]
                                - self.tokens_per_frame,
                                0,
                            ),
                        )
                    ),
                    "flow_indexed_local_transport_confidence": (
                        torch.nn.functional.pad(
                            (
                                flow_state.transport_confidence
                                if flow_state.transport_confidence is not None
                                else flow_state.support.float()
                            ).detach(),
                            (
                                persistent_direct.shape[1]
                                - self.tokens_per_frame,
                                0,
                            ),
                        )
                    ),
                    "flow_indexed_appearance_trust_on_support": (
                        (
                            (
                                flow_state.appearance_trust
                                if flow_state.appearance_trust is not None
                                else flow_state.confidence
                            )
                            * flow_state.support.float()
                        ).sum()
                        / flow_state.support.float().sum().clamp_min(1.0)
                    ),
                    "flow_indexed_local_transport_on_support": (
                        (
                            (
                                flow_state.transport_confidence
                                if flow_state.transport_confidence is not None
                                else flow_state.support.float()
                            )
                            * flow_state.support.float()
                        ).sum()
                        / flow_state.support.float().sum().clamp_min(1.0)
                    ),
                })
        self._recent = new_recent
        self._source_lineage = new_source_lineage
        if self.flow_indexed_residual_ledger:
            self._flow_state_index = flow_frame_indices[-1]
            self._prepared_flow_read = {}
            self._prepared_flow_indices = None
        if self.timestep_counterfactual_memory:
            if self._commit_count == 0:
                self._freeze_timestep_counterfactual_bank()
                self._initialize_canonical_coordinates(
                    frame_indices=flow_frame_indices,
                    spatial_shape=tuple(int(v) for v in spatial_shape),
                    device=write_confidence.device,
                )
            elif self._prepared_canonical_coordinate_state:
                self._canonical_coordinate_state = (
                    self._prepared_canonical_coordinate_state
                )
            self._prepared_canonical_coordinate_state = {}
            self._prepared_canonical_correspondence = {}
        self._commit_count += 1
        return diagnostics

    def read(self) -> Dict[int, NativeKVRead]:
        if not self.has_canonical():
            return {}

        def recent_without_duplicate_canonical(layer: int):
            recent = self._recent.get(layer)
            if self.transactional_dense_recent:
                # The entry bridge needs the complete adjacent edited block.
                # Canonical is a fallback in a disjoint per-query branch, so
                # duplicate suppression is neither necessary nor desirable.
                return recent
            if recent is None or self._commit_count > 1:
                return recent
            canonical = self._canonical[layer]
            support = recent.support.clone()
            canonical_index = canonical.token_index.to(support.device)
            canonical_valid = canonical.support.to(support.device).bool()
            # Token indices are source-block coordinates.  This works for
            # both the legacy dense recent tier and the compact transactional
            # tier, whose storage indices are not source coordinates.
            duplicated = (
                recent.token_index.to(support.device)[:, :, None]
                == canonical_index[:, None, :]
            ) & canonical_valid[:, None, :]
            support &= ~duplicated.any(dim=-1)
            deduplicated = NativeKVFrame(
                source_key=recent.source_key,
                source_value=recent.source_value,
                target_key=recent.target_key,
                target_value=recent.target_value,
                token_index=recent.token_index,
                support=support,
                frame_count=recent.frame_count,
                payload_support=(
                    None
                    if recent.payload_support is None
                    else recent.payload_support & support
                ),
            )
            deduplicated.validate()
            return deduplicated

        return {
            layer: NativeKVRead(
                canonical=self._canonical[layer],
                # The complete previous block preserves motion/occlusion. If
                # it is also the ignition block, mask only canonical tokens
                # from this short tier so they are not counted twice.
                recent=recent_without_duplicate_canonical(layer),
                source_lineage=self._source_lineage.get(layer),
                flow_residual=self._prepared_flow_read.get(layer),
                canonical_correspondence=(
                    self._prepared_canonical_correspondence.get(layer)
                ),
                recent_shares_canonical_time=(self._commit_count == 1),
            )
            for layer in self.layers
        }
