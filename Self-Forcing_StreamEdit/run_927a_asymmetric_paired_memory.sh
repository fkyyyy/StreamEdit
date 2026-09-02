#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/927a_asymmetric_paired_memory}"
OUTPUT_NAME="${OUTPUT_NAME:-927a-asymmetric-paired-memory.mp4}"

# Single-variable ablation against 926a. Native dense target history remains
# the short-term/failure path. A sparse canonical memory is addressed by
# clean-source K and stores only target-minus-source V. Reads are object-only;
# later writes require source correspondence and residual agreement.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" \
  "$SCRIPT_DIR/run_926a_prompt_deconfounded_violet.sh" \
  --causal_paired_edit_memory \
  --paired_memory_layers 8 12 16 20 \
  --paired_memory_max_tokens 1536 \
  --paired_memory_max_tokens_per_block 192 \
  --paired_memory_topk 8 \
  --paired_memory_min_similarity 0.35 \
  --paired_memory_min_commit_confidence 0.20 \
  --paired_memory_coordinate_bias 1.0 \
  --paired_memory_read_strength 0.35 \
  "$@"
