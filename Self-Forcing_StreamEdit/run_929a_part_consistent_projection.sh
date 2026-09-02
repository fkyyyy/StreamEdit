#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/929a_part_consistent_projection}"
OUTPUT_NAME="${OUTPUT_NAME:-929a-part-consistent-projection.mp4}"

# Mechanism test against 928a: make the source-addressed value projection
# part-consistent and apply it to the replayed first block as well as later
# blocks. Source-residual routing is intentionally unchanged in this run, so
# 929b can isolate that additional intervention.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" \
  "$SCRIPT_DIR/run_927a_asymmetric_paired_memory.sh" \
  --paired_memory_value_projection \
  --paired_memory_read_strength 0.75 \
  --paired_memory_coordinate_radius 0.25 \
  --paired_memory_min_residual_consensus 0.45 \
  --paired_memory_interior_projection \
  --paired_memory_first_block_replay \
  "$@"
