#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Single-variable control for the 947 -> 948 comparison.  It keeps the 947
# hand-only KV semantics unchanged and only coalesces the ignition canonical
# and recent tiers onto their shared physical time range.
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/949a_947_rope_origin_control}"
export OUTPUT_NAME="${OUTPUT_NAME:-949a-947-rope-origin-control.mp4}"
mkdir -p "$OUTDIR"
printf '%s\n' \
  'baseline=947a_verified_attention_authority' \
  'ablation=bootstrap_rope_origin_only' \
  'external_object_mask=disabled' \
  'external_source_owner_mask=disabled' \
  'external_hand_mask=enabled' \
  'payload=unchanged_transactional_compact_target_kv' \
  'first_post_ignition_recent_origin=0' \
  'first_post_ignition_current_origin=3' \
  > "$OUTDIR/949a_config.txt"

exec bash "$SCRIPT_DIR/run_947a_verified_attention_authority.sh" \
  --native_history_coalesce_bootstrap_time \
  "$@"
