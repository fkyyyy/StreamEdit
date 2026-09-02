#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video.mp4}"
HAND_MASK="${HAND_MASK:-$REPO_ROOT/hand_mask.mp4}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/915a_prompt_control}"
OUTPUT_NAME="${OUTPUT_NAME:-915a-prompt-control.mp4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
STEP="${STEP:-15}"
HAND_MASK_MODE="${HAND_MASK_MODE:-overlay_white}"
HAND_MASK_OVERLAY_DIFF_THRESHOLD="${HAND_MASK_OVERLAY_DIFF_THRESHOLD:-24}"
IDENTITY_CORRECTION_STRENGTH="${IDENTITY_CORRECTION_STRENGTH:-0.35}"
IDENTITY_SOURCE_SUPPRESSION="${IDENTITY_SOURCE_SUPPRESSION:-0.35}"
IDENTITY_SUPPORT_FLOOR="${IDENTITY_SUPPORT_FLOOR:-0.40}"
IDENTITY_RESIDUAL_CARRY_STRENGTH="${IDENTITY_RESIDUAL_CARRY_STRENGTH:-0.25}"
IGNITION_HAND_EXCLUSION_RADIUS="${IGNITION_HAND_EXCLUSION_RADIUS:-1}"
IGNITION_CONTACT_RADIUS="${IGNITION_CONTACT_RADIUS:-3}"
SRC_PROMPT="${SRC_PROMPT:-A first-person egocentric kitchen video. A person holds one white cylindrical plastic seasoning bottle with an opaque blue screw cap. Preserve the hand, fingers, arm, grasp, object position, object size, object orientation, object motion, occlusions, camera motion, pan, stovetop, counter, lighting, and kitchen background.}"
TRG_PROMPT="${TRG_PROMPT:-A first-person egocentric kitchen video. The exact same cylindrical plastic seasoning bottle uniformly recolored opaque red, including its existing screw cap, is held by the person. Its silhouette, dimensions, cap geometry, material structure, position, orientation, motion, hand grasp, and occlusions remain exactly unchanged. Preserve the hand, fingers, arm, camera motion, pan, stovetop, counter, lighting, and kitchen background unchanged.}"
SRC_WORD="${SRC_WORD:-white cylindrical plastic seasoning bottle with an opaque blue screw cap}"
TRG_WORD="${TRG_WORD:-cylindrical plastic seasoning bottle uniformly recolored opaque red}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
mkdir -p "$OUTDIR/roles"
cd "$SCRIPT_DIR"

"$PYTHON_BIN" inference_edit_streamedit.py \
  --data_path "$DATA_PATH" \
  --hand_mask_video "$HAND_MASK" \
  --save_path "$OUTDIR/$OUTPUT_NAME" \
  --save_role_dir "$OUTDIR/roles" \
  --routing_mode hand_role_bayes_flow_tokenprop_kv \
  --first_chunk_identity_replay \
  --appearance_leakage_decomposition \
  --source_coordinate_identity \
  --source_identity_residual_carry \
  --identity_correction_strength "$IDENTITY_CORRECTION_STRENGTH" \
  --identity_source_suppression "$IDENTITY_SOURCE_SUPPRESSION" \
  --identity_support_floor "$IDENTITY_SUPPORT_FLOOR" \
  --identity_residual_carry_strength "$IDENTITY_RESIDUAL_CARRY_STRENGTH" \
  --ignition_hand_exclusion_radius "$IGNITION_HAND_EXCLUSION_RADIUS" \
  --ignition_contact_radius "$IGNITION_CONTACT_RADIUS" \
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
