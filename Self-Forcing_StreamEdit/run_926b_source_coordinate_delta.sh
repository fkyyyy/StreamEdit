#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/926b_source_coordinate_delta}"
OUTPUT_NAME="${OUTPUT_NAME:-926b-source-coordinate-delta.mp4}"

# Same violet prompt and controls as 926a; add only source-coordinate
# target-delta velocity routing.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" \
  "$SCRIPT_DIR/run_926a_prompt_deconfounded_violet.sh" \
  --factorized_source_coordinate_target_delta
