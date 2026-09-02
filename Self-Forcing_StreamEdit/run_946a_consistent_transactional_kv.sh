#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Keep every 945 input, prompt, seed, and inference setting unchanged.  The
# single method change is the end-to-end consistent transaction contract.
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/946a_consistent_transactional_kv}"
export OUTPUT_NAME="${OUTPUT_NAME:-946a-consistent-transactional-kv.mp4}"
# In 946 this scales only the source-residual component opposing the retrieved
# target edit. Orthogonal source motion/geometry is retained, so one is the
# natural no-leakage setting rather than a global source block.
export VERIFIED_SOURCE_SUPPRESSION="${VERIFIED_SOURCE_SUPPRESSION:-1.0}"
mkdir -p "$OUTDIR"
printf '%s\n' \
  'baseline=945a_hand_flow_transactional_kv' \
  'method=consistent_transactional_native_kv' \
  'external_object_mask=disabled' \
  'external_source_owner_mask=disabled' \
  'recent_payload=compact_write_approved' \
  'empty_write=hold_last_commit' \
  'read_gate=soft_owner_x_source_address_match' \
  'source_arbitration=same_retrieval_strength_antagonistic_projection' \
  'lifecycle=blockwise' \
  "verified_source_suppression=$VERIFIED_SOURCE_SUPPRESSION" \
  > "$OUTDIR/946a_config.txt"

exec bash "$SCRIPT_DIR/run_945a_hand_flow_transactional_kv.sh" \
  --native_history_consistent_transaction \
  "$@"
