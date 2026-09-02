#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/942a_trust_region_part_arbitration}"
OUTPUT_NAME="${OUTPUT_NAME:-942a-trust-region-part-arbitration.mp4}"
MIN_PART_SIMILARITY="${MIN_PART_SIMILARITY:-0.45}"
PART_SIMILARITY_MARGIN="${PART_SIMILARITY_MARGIN:-0.08}"
PART_BIAS_STRENGTH="${PART_BIAS_STRENGTH:-0.25}"
PART_REFINEMENT_RATIO="${PART_REFINEMENT_RATIO:-0.10}"

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=937a_role_fixed_native_kv' \
    'method=soft_source_part_canonical_arbitration_with_trust_region' \
    'prompt=exact_937a' \
    'owner_complement_source=disabled' \
    'recent_and_current_kv=never_part_pruned' \
    'canonical_candidates=source_part_soft_logit_bias_only' \
    'fallback=exact_937a_for_unmatched_queries' \
    "min_part_similarity=$MIN_PART_SIMILARITY" \
    "part_similarity_margin=$PART_SIMILARITY_MARGIN" \
    "part_bias_strength=$PART_BIAS_STRENGTH" \
    "part_refinement_ratio=$PART_REFINEMENT_RATIO"
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/942a_config.txt"

# Keep the best observed 937a path intact. Source-part evidence can only
# reweight its already admitted canonical keys; it cannot remove the dense
# recent/current support. The final refinement is clipped relative to the
# original 937a memory residual, preventing the sparse-softmax collapse seen
# in 941b.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" \
  "$SCRIPT_DIR/run_937a_role_fixed_native_kv.sh" \
  --native_history_source_part_consistency \
  --native_history_min_part_similarity "$MIN_PART_SIMILARITY" \
  --native_history_part_similarity_margin "$PART_SIMILARITY_MARGIN" \
  --native_history_part_bias_strength "$PART_BIAS_STRENGTH" \
  --native_history_part_refinement_ratio "$PART_REFINEMENT_RATIO" \
  "$@"
