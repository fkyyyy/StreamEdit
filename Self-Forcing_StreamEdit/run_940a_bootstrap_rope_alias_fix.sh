#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/940a_bootstrap_rope_alias_fix}"
OUTPUT_NAME="${OUTPUT_NAME:-940a-bootstrap-rope-alias-fix.mp4}"

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=939a_uncertainty_abstaining_source_closure' \
    'method=same_commit_bootstrap_rope_alias_fix' \
    'first_read=canonical_and_recent_share_temporal_origin' \
    'current_origin=one_real_history_block' \
    'later_reads=canonical_recent_current_sequential' \
    'prompt_change=none' \
    'owner_change=none' \
    'velocity_change=none'
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/940_config.txt"

# One-variable successor to 939a. On the first post-ignition read, compact
# canonical tokens and the dense recent tier both came from block 0, so they
# receive the same temporal RoPE range. Subsequent reads retain the normal
# canonical/recent/current ordering.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" \
  "$SCRIPT_DIR/run_939a_uncertainty_abstaining_source_closure.sh" \
  --native_history_coalesce_bootstrap_time \
  "$@"
