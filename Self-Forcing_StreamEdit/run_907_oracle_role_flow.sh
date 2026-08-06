#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video.mp4}"
OBJECT_MASK="${OBJECT_MASK:-$REPO_ROOT/bottle_mask.mp4}"
HAND_MASK="${HAND_MASK:-$REPO_ROOT/hand_mask.mp4}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/egoedit_907_oracle_role_flow_kv}"
STEP="${STEP:-15}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
ROUTING_MODE="${ROUTING_MODE:-oracle_role_flow_kv}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
# #region debug-point B:runtime-log-routing
export DEBUG_SERVER_URL="${DEBUG_SERVER_URL:-http://10.74.55.101:7777/event}"
export DEBUG_SESSION_ID="${DEBUG_SESSION_ID:-oracle-edit-collapse}"
export DEBUG_RUN_ID="${DEBUG_RUN_ID:-post-fix}"
# #endregion
mkdir -p "$OUTDIR/roles"
cd "$SCRIPT_DIR"

python inference_edit_streamedit.py \
  --data_path "$DATA_PATH" \
  --object_mask_video "$OBJECT_MASK" \
  --hand_mask_video "$HAND_MASK" \
  --save_path "$OUTDIR/907-oracle-role-flow-kv.mp4" \
  --save_role_dir "$OUTDIR/roles" \
  --routing_mode "$ROUTING_MODE" \
  --role_boundary_radius 1 \
  --mask_white_threshold 245 \
  --object_min_latent_coverage 0.001 \
  --src_prompt "A person is holding a white plastic seasoning shaker with a blue cap in a kitchen." \
  --trg_prompt "A person is holding a ribbed silver aluminum can with horizontal ribs in a kitchen." \
  --src_word "white plastic seasoning shaker" \
  --trg_word "ribbed silver aluminum can" \
  --fg_boost_factor 4 \
  --blend_power 2 \
  --step "$STEP" \
  --rollout_chunk_size 21 \
  --rollout_overlap_block_num 1 \
  2>&1 | tee "$OUTDIR/run.log"
