#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

CUDA_DEVICE="${CUDA_DEVICE:-7}"
VIDEO="${VIDEO:-$REPO_ROOT/source_video1.mp4}"
CHECKPOINT="${CHECKPOINT:-$SCRIPT_DIR/checkpoints/optical_flow/raft_large_C_T_SKHT_V2-ff5fadd5.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/952b_phone_to_cardboard_box/source_flow}"
BATCH_SIZE="${BATCH_SIZE:-2}"

exec env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python "$SCRIPT_DIR/tools/precompute_source_flow.py" \
  --video "$VIDEO" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --height 480 \
  --width 832 \
  --device cuda \
  --batch-size "$BATCH_SIZE" \
  --preview-pairs 8
