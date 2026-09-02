#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 947 changes one mechanism relative to 946: after all configured native-KV
# layers agree that an automatically inferred owner query retrieved target
# memory, the remaining transformer layers use factorized target-value
# attention for that query. Everything else, including velocity routing, is
# held fixed so this run isolates attention-side source-identity reflux.
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/947a_verified_attention_authority}"
export OUTPUT_NAME="${OUTPUT_NAME:-947a-verified-attention-authority.mp4}"
export ATTENTION_AUTHORITY_STRENGTH="${ATTENTION_AUTHORITY_STRENGTH:-1.0}"
mkdir -p "$OUTDIR"
printf '%s\n' \
  'baseline=946a_consistent_transactional_kv' \
  'method=verified_attention_authority_ablation' \
  'external_object_mask=disabled' \
  'external_source_owner_mask=disabled' \
  'external_hand_mask=enabled' \
  'authority_gate=automatic_owner_x_cross_layer_successful_kv_read' \
  'authority_scope=post_last_verified_kv_layer_only' \
  'velocity_routing=unchanged_from_946' \
  "attention_authority_strength=$ATTENTION_AUTHORITY_STRENGTH" \
  > "$OUTDIR/947a_config.txt"

exec bash "$SCRIPT_DIR/run_946a_consistent_transactional_kv.sh" \
  --native_history_verified_attention_authority \
  --native_history_attention_authority_strength \
  "$ATTENTION_AUTHORITY_STRENGTH" \
  "$@"
