#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/937a_role_fixed_native_kv}"
OUTPUT_NAME="${OUTPUT_NAME:-937a-role-fixed-native-kv.mp4}"

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=926a_prompt_deconfounded_violet' \
    'method=role_conditioned_fixed_relative_native_kv_history' \
    'long_term=immutable_first_final_clean_target_kv' \
    'short_term=latest_final_clean_target_kv' \
    'address=clean_source_pre_rope_key' \
    'payload=native_target_kv_only' \
    'position=fixed_relative_native_3d_rope' \
    'fallback=exact_926_for_hand_boundary_background_unknown_or_unmatched' \
    'value_projection=disabled' \
    'output_residual=disabled'
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/937_config.txt"

# Strict one-variable successor to 926a.  It keeps the same prompt, role
# router, velocity routing and complete native clean-target cache.  The only
# addition is a query-gated short/long native-KV read in layers 8/12/16/20.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" \
  "$SCRIPT_DIR/run_926a_prompt_deconfounded_violet.sh" \
  --role_fixed_native_history \
  --native_history_layers 8 12 16 20 \
  --native_history_max_tokens_per_frame 256 \
  --native_history_topk 8 \
  --native_history_min_similarity 0.35 \
  --native_history_min_write_confidence 0.50 \
  --native_history_min_query_confidence 0.50 \
  --native_history_canonical_logit_bias 1.0 \
  "$@"
