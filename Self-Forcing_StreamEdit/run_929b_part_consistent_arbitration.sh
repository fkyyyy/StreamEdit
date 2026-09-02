#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/929b_part_consistent_arbitration}"
OUTPUT_NAME="${OUTPUT_NAME:-929b-part-consistent-arbitration.mp4}"

# Single-variable follow-up to 929a. The only addition is soft suppression of
# the competing source residual under the same successful, interior paired
# read. A lower value than 928 keeps source geometry available near uncertain
# correspondences while exact zero-support fallback remains unchanged.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" \
  "$SCRIPT_DIR/run_929a_part_consistent_projection.sh" \
  --paired_memory_source_suppression 0.50 \
  "$@"
