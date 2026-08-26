#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video.mp4}"
HAND_MASK="${HAND_MASK:-$REPO_ROOT/hand_mask.mp4}"
ROUTING_MODE="${ROUTING_MODE:-hand_role_bayes_flow_tokenprop_kv}"
OUTPUT_NAME="${OUTPUT_NAME:-907-bayes-object-wise-anchor-reset.mp4}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/907_object_wise_anchor_reset}"
STEP="${STEP:-15}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
HAND_MASK_MODE="${HAND_MASK_MODE:-overlay_white}"
HAND_MASK_OVERLAY_DIFF_THRESHOLD="${HAND_MASK_OVERLAY_DIFF_THRESHOLD:-24}"
IDENTITY_TOKENPROP_MIN_SIMILARITY="${IDENTITY_TOKENPROP_MIN_SIMILARITY:-0.55}"
IDENTITY_TOKENPROP_GATE_STRENGTH="${IDENTITY_TOKENPROP_GATE_STRENGTH:-0.85}"
IDENTITY_TOKENPROP_MAX_CANDIDATES="${IDENTITY_TOKENPROP_MAX_CANDIDATES:-512}"
COMMITTED_MEMORY_FEEDBACK_STRENGTH="${COMMITTED_MEMORY_FEEDBACK_STRENGTH:-0.75}"
SRC_PROMPT="${SRC_PROMPT:-A first-person egocentric kitchen video of a person holding a white plastic seasoning shaker with a blue cap. The same hand, hand pose, camera motion, stovetop, pan, counter, and kitchen background are visible.}"
TRG_PROMPT="${TRG_PROMPT:-A first-person egocentric kitchen video of a person holding a red bottle. The same hand, hand motion, camera motion, stovetop, pan, counter, and kitchen background remain unchanged.}"
SRC_WORD="${SRC_WORD:-white plastic seasoning shaker}"
TRG_WORD="${TRG_WORD:-red bottle}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
mkdir -p "$OUTDIR/roles"
cd "$SCRIPT_DIR"

python inference_edit_streamedit.py \
  --data_path "$DATA_PATH" \
  --hand_mask_video "$HAND_MASK" \
  --save_path "$OUTDIR/$OUTPUT_NAME" \
  --save_role_dir "$OUTDIR/roles" \
  --routing_mode "$ROUTING_MODE" \
  --identity_first_latent_bootstrap \
  --object_wise_anchor_reset \
  --contact_graph_mode no_graph \
  --hand_query_layers 8 12 16 20 \
  --mask_white_threshold 245 \
  --hand_mask_mode "$HAND_MASK_MODE" \
  --hand_mask_overlay_diff_threshold "$HAND_MASK_OVERLAY_DIFF_THRESHOLD" \
  --identity_tokenprop_min_similarity "$IDENTITY_TOKENPROP_MIN_SIMILARITY" \
  --identity_tokenprop_gate_strength "$IDENTITY_TOKENPROP_GATE_STRENGTH" \
  --identity_tokenprop_max_candidates "$IDENTITY_TOKENPROP_MAX_CANDIDATES" \
  --committed_memory_feedback_strength "$COMMITTED_MEMORY_FEEDBACK_STRENGTH" \
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
