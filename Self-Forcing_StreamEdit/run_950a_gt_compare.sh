#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/950a_high_recall_hand_roles}"
OUTPUT_NAME="${OUTPUT_NAME:-950a-high-recall-hand-roles.mp4}"
SOURCE_VIDEO="${DATA_PATH:-$REPO_ROOT/source_video.mp4}"
GT_MASK="${GT_MASK:-$REPO_ROOT/bottle_mask.mp4}"
HAND_INPUT="$OUTDIR/${OUTPUT_NAME%.mp4}.hand_role_input.npz"

# This script is evaluation-only. The GT path is intentionally absent from
# every generation script argument and every inference function signature.
python "$SCRIPT_DIR/tools/compare_roles_to_gt.py" \
  --roles_dir "$OUTDIR/roles" \
  --gt_mask_video "$GT_MASK" \
  --source_video "$SOURCE_VIDEO" \
  --hand_input_npz "$HAND_INPUT" \
  --output_dir "$OUTDIR/gt_audit"
