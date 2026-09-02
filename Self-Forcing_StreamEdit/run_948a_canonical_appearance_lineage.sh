#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 948 keeps the complete 947 hand-only owner, prompt, seed, routing, source
# arbitration, and verified attention authority.  It changes one state
# contract: generated target K/V may no longer become cross-chunk appearance
# memory.  The ignition block is the immutable appearance payload; subsequent
# blocks update only clean-source addresses and canonical-slot lineage.
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/948a_canonical_appearance_lineage}"
export OUTPUT_NAME="${OUTPUT_NAME:-948a-canonical-appearance-lineage.mp4}"
export PAYLOAD_BLEND_STRENGTH="${PAYLOAD_BLEND_STRENGTH:-0.35}"
mkdir -p "$OUTDIR"
printf '%s\n' \
  'baseline=947a_verified_attention_authority' \
  'method=canonical_appearance_source_lineage_transaction' \
  'external_object_mask=disabled' \
  'external_source_owner_mask=disabled' \
  'external_hand_mask=enabled' \
  'target_payload=immutable_first_chunk_native_kv_only' \
  'mutable_target_payload=disabled' \
  'mutable_state=clean_source_address_and_canonical_slot_lineage_only' \
  'appearance_read=shared_attention_target_value_minus_source_value' \
  'geometry_motion=current_native_stream' \
  'source_arbitration=verified_retrieval_antagonistic_projection' \
  'attention_authority=verified_cross_layer_read' \
  "payload_blend_strength=$PAYLOAD_BLEND_STRENGTH" \
  > "$OUTDIR/948a_config.txt"

exec bash "$SCRIPT_DIR/run_947a_verified_attention_authority.sh" \
  --native_history_payload_invariant_lineage \
  --native_history_payload_blend_strength \
  "$PAYLOAD_BLEND_STRENGTH" \
  "$@"
