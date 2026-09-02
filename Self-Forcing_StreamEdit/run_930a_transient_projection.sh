#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/930a_transient_projection}"
OUTPUT_NAME="${OUTPUT_NAME:-930a-transient-projection.mp4}"

# Strict cache-write ablation against 929a. Keep the same canonical memory,
# retrieval, support, and current-block value projection, but do not rewrite
# the persistent clean target KV. Any 929a -> 930a improvement therefore
# isolates recurrent propagation through historical projected values.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" \
  "$SCRIPT_DIR/run_929a_part_consistent_projection.sh" \
  --paired_memory_disable_persistent_projection \
  "$@"
