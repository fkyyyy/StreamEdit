#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/943a_transactional_owner_kv}"
OUTPUT_NAME="${OUTPUT_NAME:-943a-transactional-owner-kv.mp4}"
OWNER_MAX_MISSING_FRAMES="${OWNER_MAX_MISSING_FRAMES:-1}"
VERIFIED_SOURCE_SUPPRESSION="${VERIFIED_SOURCE_SUPPRESSION:-0.35}"

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=937a_role_fixed_native_kv' \
    'method=transactional_causal_owner_native_kv' \
    'prompt=exact_937a' \
    'read_support=complete_source_owner_plus_bounded_source_feature_lifecycle' \
    'write_support=visible_non_hand_owner_core_only' \
    'contact_and_lifecycle=read_only' \
    'source_suppression=post_retrieval_layer_agreement_only' \
    'part_bias=disabled' \
    'fallback=exact_937a_when_native_kv_abstains' \
    "owner_max_missing_frames=$OWNER_MAX_MISSING_FRAMES" \
    "verified_source_suppression=$VERIFIED_SOURCE_SUPPRESSION"
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/943a_config.txt"

# 943 keeps the best 937a prompt, native K/V payload, layer set and write
# threshold. The only intervention repairs owner continuity at hand contact:
# contact/lifecycle tokens can query the frozen target memory but cannot write
# it, and source residuals are reduced only after a real KV match succeeds.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" \
  "$SCRIPT_DIR/run_937a_role_fixed_native_kv.sh" \
  --native_history_transactional_owner \
  --native_history_owner_max_missing_frames "$OWNER_MAX_MISSING_FRAMES" \
  --native_history_verified_source_suppression "$VERIFIED_SOURCE_SUPPRESSION" \
  "$@"
