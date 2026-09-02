#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/939a_uncertainty_abstaining_source_closure}"
OUTPUT_NAME="${OUTPUT_NAME:-939a-uncertainty-abstaining-source-closure.mp4}"
OWNER_MARGIN="${OWNER_MARGIN:-1}"
MIN_PRESERVE_CONFIDENCE="${MIN_PRESERVE_CONFIDENCE:-0.25}"

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=938a_owner_complement_source' \
    'method=uncertainty_abstaining_source_closure' \
    'owner=unchanged_937_native_kv_velocity' \
    'confident_hand_background=exact_clean_source_reconstruction_velocity' \
    'unknown_or_occluded=unchanged_937_native_kv_velocity' \
    "owner_margin=$OWNER_MARGIN" \
    "min_preserve_confidence=$MIN_PRESERVE_CONFIDENCE" \
    'kv_change=none'
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/939_config.txt"

# Only positive preservation evidence may trigger exact source closure.
# Unknown complement pixels retain 937 so owner dropout cannot restore the
# source object's appearance during hand occlusion or an image exit.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" OWNER_MARGIN="$OWNER_MARGIN" \
  "$SCRIPT_DIR/run_938a_owner_complement_source.sh" \
  --factorized_owner_complement_min_preserve_confidence \
  "$MIN_PRESERVE_CONFIDENCE" \
  "$@"
