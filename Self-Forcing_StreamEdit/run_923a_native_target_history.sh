#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video.mp4}"
HAND_MASK="${HAND_MASK:-$REPO_ROOT/hand_mask.mp4}"
SOURCE_OWNER_MASK="${SOURCE_OWNER_MASK:-$REPO_ROOT/bottle_mask.mp4}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/923a_native_target_history}"
OUTPUT_NAME="${OUTPUT_NAME:-923a-native-target-history.mp4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
STEP="${STEP:-15}"
HAND_MASK_MODE="${HAND_MASK_MODE:-overlay_white}"
HAND_MASK_OVERLAY_DIFF_THRESHOLD="${HAND_MASK_OVERLAY_DIFF_THRESHOLD:-24}"
SOURCE_OWNER_MASK_MODE="${SOURCE_OWNER_MASK_MODE:-overlay_white}"
SOURCE_OWNER_OVERLAY_DIFF_THRESHOLD="${SOURCE_OWNER_OVERLAY_DIFF_THRESHOLD:-24}"
SRC_PROMPT="${SRC_PROMPT:-A first-person egocentric kitchen video. One hand holds and moves one white cylindrical plastic seasoning bottle with an opaque blue screw cap near a stovetop and kitchen drawers. Preserve the hand anatomy, fingers, grasp, bottle position, scale, silhouette, orientation, motion, occlusions, camera motion, counter, stovetop, drawers, lighting, and background.}"
TRG_PROMPT="${TRG_PROMPT:-A first-person egocentric kitchen video. The exact same cylindrical plastic seasoning bottle uniformly recolored saturated opaque red, including its existing screw cap, remains consistent across every view. Every visible bottle surface remains the same red while it is held, tilted, poured, moved, occluded, and placed into a drawer. Keep exactly the same bottle dimensions, silhouette, cap geometry, material structure, position, scale, orientation, motion, hand anatomy, hand grasp, hand-object occlusions, camera motion, counter, stovetop, drawers, lighting, and background.}"
SRC_WORD="${SRC_WORD:-white cylindrical plastic seasoning bottle with an opaque blue screw cap}"
TRG_WORD="${TRG_WORD:-cylindrical plastic seasoning bottle uniformly recolored saturated opaque red}"

for required_path in "$DATA_PATH" "$HAND_MASK" "$SOURCE_OWNER_MASK"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
mkdir -p "$OUTDIR/roles"
cd "$SCRIPT_DIR"

"$PYTHON_BIN" inference_edit_streamedit.py \
  --data_path "$DATA_PATH" \
  --hand_mask_video "$HAND_MASK" \
  --source_owner_mask_video "$SOURCE_OWNER_MASK" \
  --source_owner_mask_mode "$SOURCE_OWNER_MASK_MODE" \
  --source_owner_overlay_diff_threshold "$SOURCE_OWNER_OVERLAY_DIFF_THRESHOLD" \
  --save_path "$OUTDIR/$OUTPUT_NAME" \
  --save_role_dir "$OUTDIR/roles" \
  --routing_mode hand_role_factorized_causal_owner_kv \
  --factorized_native_target_history \
  --contact_graph_mode no_graph \
  --hand_query_layers 8 12 16 20 \
  --mask_white_threshold 245 \
  --hand_mask_mode "$HAND_MASK_MODE" \
  --hand_mask_overlay_diff_threshold "$HAND_MASK_OVERLAY_DIFF_THRESHOLD" \
  --src_prompt "$SRC_PROMPT" \
  --trg_prompt "$TRG_PROMPT" \
  --src_word "$SRC_WORD" \
  --trg_word "$TRG_WORD" \
  --fg_boost_factor 4 \
  --blend_power 2 \
  --step "$STEP" \
  --seed 0 \
  --rollout_chunk_size 21 \
  --rollout_overlap_block_num 1 \
  2>&1 | tee "$OUTDIR/run.log"
