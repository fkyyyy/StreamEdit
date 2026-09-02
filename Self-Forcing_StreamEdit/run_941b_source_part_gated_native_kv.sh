#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/941b_source_part_gated_native_kv}"
OUTPUT_NAME="${OUTPUT_NAME:-941b-source-part-gated-native-kv.mp4}"
MIN_PART_SIMILARITY="${MIN_PART_SIMILARITY:-0.45}"
PART_SIMILARITY_MARGIN="${PART_SIMILARITY_MARGIN:-0.08}"

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=941a_white_body_prompt_control' \
    'method=source_part_gated_native_kv' \
    'part_address=clean_source_value_signature' \
    'payload=unchanged_native_target_kv' \
    'unmatched_query=exact_native_fallback' \
    "min_part_similarity=$MIN_PART_SIMILARITY" \
    "part_similarity_margin=$PART_SIMILARITY_MARGIN"
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/941b_config.txt"

OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" \
  "$SCRIPT_DIR/run_941a_white_body_prompt_control.sh" \
  --native_history_source_part_consistency \
  --native_history_min_part_similarity "$MIN_PART_SIMILARITY" \
  --native_history_part_similarity_margin "$PART_SIMILARITY_MARGIN" \
  "$@"
