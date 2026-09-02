#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/938a_owner_complement_source}"
OUTPUT_NAME="${OUTPUT_NAME:-938a-owner-complement-source.mp4}"
OWNER_MARGIN="${OWNER_MARGIN:-1}"

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=937c_minimal_flat_cap_prompt' \
    'method=owner_complement_source_coordinate_velocity' \
    'background=exact_clean_source_reconstruction_velocity' \
    'owner_and_boundary=unchanged_937_native_kv_velocity' \
    "owner_margin=$OWNER_MARGIN" \
    'kv_change=none'
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/938_config.txt"

# 938 changes only the final velocity outside the causal source owner.  A
# one-cell owner-grid safety band keeps object/contact boundaries on the 937
# path; definite non-owner pixels cannot receive the target-prompt delta.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" \
  "$SCRIPT_DIR/run_937c_minimal_flat_cap_prompt.sh" \
  --factorized_owner_complement_source \
  --factorized_owner_complement_margin "$OWNER_MARGIN" \
  "$@"
