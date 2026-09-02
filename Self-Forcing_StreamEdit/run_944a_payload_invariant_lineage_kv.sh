#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/944a_payload_invariant_lineage_kv}"
OUTPUT_NAME="${OUTPUT_NAME:-944a-payload-invariant-lineage-kv.mp4}"
PAYLOAD_BLEND_STRENGTH="${PAYLOAD_BLEND_STRENGTH:-0.35}"
TRG_PROMPT="${TRG_PROMPT:-A first-person egocentric kitchen video identical to the source. Only the screw cap of the seasoning bottle is changed to dark violet. The bottle body and everything else remain unchanged.}"

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=943a_transactional_owner_kv' \
    'method=payload_invariant_source_lineage_kv' \
    'target_scope=cap_only' \
    'canonical_payload=immutable_ignition_target_kv' \
    'mutable_target_payload=disabled' \
    'mutable_state=clean_source_address_and_canonical_slot_lineage_only' \
    'abstained_write=hold_last_valid_source_lineage' \
    'read_gate=complete_source_owner_times_source_correspondence' \
    'write_gate=visible_non_hand_owner_core_only' \
    'fallback=exact_native_when_source_owner_or_correspondence_abstains' \
    "payload_blend_strength=$PAYLOAD_BLEND_STRENGTH" \
    "target_prompt=$TRG_PROMPT"
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/944a_config.txt"

# The mutable tier advances source-coordinate lineage but carries no target
# K/V.  Appearance is read only from the frozen ignition payload, preventing
# a generated color/shape error from becoming the next chunk's identity.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" TRG_PROMPT="$TRG_PROMPT" \
  "$SCRIPT_DIR/run_943a_transactional_owner_kv.sh" \
  --native_history_payload_invariant_lineage \
  --native_history_payload_blend_strength "$PAYLOAD_BLEND_STRENGTH" \
  "$@"
