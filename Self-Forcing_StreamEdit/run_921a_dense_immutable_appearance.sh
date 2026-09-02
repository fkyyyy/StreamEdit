#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video.mp4}"
HAND_MASK="${HAND_MASK:-$REPO_ROOT/hand_mask.mp4}"
SOURCE_OWNER_MASK="${SOURCE_OWNER_MASK:-$REPO_ROOT/bottle_mask.mp4}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/921a_dense_immutable_appearance}"
OUTPUT_NAME="${OUTPUT_NAME:-921a-dense-immutable-appearance.mp4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
STEP="${STEP:-15}"
HAND_MASK_MODE="${HAND_MASK_MODE:-overlay_white}"
HAND_MASK_OVERLAY_DIFF_THRESHOLD="${HAND_MASK_OVERLAY_DIFF_THRESHOLD:-24}"
SOURCE_OWNER_MASK_MODE="${SOURCE_OWNER_MASK_MODE:-overlay_white}"
SOURCE_OWNER_OVERLAY_DIFF_THRESHOLD="${SOURCE_OWNER_OVERLAY_DIFF_THRESHOLD:-24}"
IDENTITY_CORRECTION_STRENGTH="${IDENTITY_CORRECTION_STRENGTH:-1.0}"
IDENTITY_SUPPORT_FLOOR="${IDENTITY_SUPPORT_FLOOR:-1.0}"
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
  --first_chunk_identity_replay \
  --factorized_immutable_target_memory \
  --immutable_target_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --immutable_target_num_prototypes 16 \
  --immutable_target_value_mode absolute \
  --immutable_target_hard_owner \
  --identity_correction_strength "$IDENTITY_CORRECTION_STRENGTH" \
  --identity_support_floor "$IDENTITY_SUPPORT_FLOOR" \
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
