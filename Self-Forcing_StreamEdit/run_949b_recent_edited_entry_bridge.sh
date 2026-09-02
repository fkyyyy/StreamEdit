#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Hand-only dual-timescale KV.  The immediately preceding timestep-zero clean
# target block carries local pose/scale continuity into only the first latent
# frame of the next block.  The ignition block remains immutable and is used
# only if recent source correspondence fails.  No object/source-owner mask is
# accepted by the inherited 945 input contract.
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/949b_recent_edited_entry_bridge}"
export OUTPUT_NAME="${OUTPUT_NAME:-949b-recent-edited-entry-bridge.mp4}"
export ENTRY_BRIDGE_STRENGTH="${ENTRY_BRIDGE_STRENGTH:-1.0}"
export MIN_RESIDUAL_CONSENSUS="${MIN_RESIDUAL_CONSENSUS:-0.05}"
mkdir -p "$OUTDIR"
printf '%s\n' \
  'baseline=947a_verified_attention_authority' \
  'method=hand_conditioned_dual_timescale_entry_bridge' \
  'external_object_mask=disabled' \
  'external_source_owner_mask=disabled' \
  'external_hand_mask=enabled' \
  'short_term_payload=complete_previous_timestep_zero_clean_target_kv' \
  'short_term_scope=first_latent_frame_per_causal_block' \
  'short_term_address=current_clean_source_to_previous_clean_source' \
  'long_term_payload=immutable_ignition_target_kv_fallback_only' \
  'write=automatic_owner_and_edit_residual_consensus' \
  'failed_write=hold_last_trusted_recent' \
  'non_owner_and_non_entry=exact_native_fallback' \
  "entry_bridge_strength=$ENTRY_BRIDGE_STRENGTH" \
  "min_residual_consensus=$MIN_RESIDUAL_CONSENSUS" \
  > "$OUTDIR/949b_config.txt"

exec bash "$SCRIPT_DIR/run_947a_verified_attention_authority.sh" \
  --native_history_coalesce_bootstrap_time \
  --native_history_recent_entry_bridge \
  --native_history_entry_bridge_strength "$ENTRY_BRIDGE_STRENGTH" \
  --native_history_dense_recent_min_residual_consensus \
  "$MIN_RESIDUAL_CONSENSUS" \
  "$@"
