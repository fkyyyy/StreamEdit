#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video.mp4}"
HAND_MASK="${HAND_MASK:-$REPO_ROOT/hand_mask.mp4}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/907_hand_role_query_visibility}"
STEP="${STEP:-15}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
POSTERIOR_THRESHOLD="${POSTERIOR_THRESHOLD:-0.20}"
MAX_OBJECT_COVERAGE="${MAX_OBJECT_COVERAGE:-0.18}"
HAND_PROXIMITY_RADIUS="${HAND_PROXIMITY_RADIUS:-3}"
PROPAGATION_STEPS="${PROPAGATION_STEPS:-2}"
VISIBILITY_RATIO="${VISIBILITY_RATIO:-0.40}"
TEMPORAL_WEIGHT="${TEMPORAL_WEIGHT:-0.45}"
QUERY_SIMILARITY_THRESHOLD="${QUERY_SIMILARITY_THRESHOLD:-0.65}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
mkdir -p "$OUTDIR/roles"
cd "$SCRIPT_DIR"

python inference_edit_streamedit.py \
  --data_path "$DATA_PATH" \
  --hand_mask_video "$HAND_MASK" \
  --save_path "$OUTDIR/907-hand-role-query-visibility.mp4" \
  --save_role_dir "$OUTDIR/roles" \
  --routing_mode hand_role_residual_kv \
  --contact_target_weight 1.0 \
  --contact_graph_mode no_graph \
  --hand_posterior_threshold "$POSTERIOR_THRESHOLD" \
  --hand_max_object_coverage "$MAX_OBJECT_COVERAGE" \
  --hand_proximity_radius "$HAND_PROXIMITY_RADIUS" \
  --hand_propagation_steps "$PROPAGATION_STEPS" \
  --hand_visibility_ratio "$VISIBILITY_RATIO" \
  --hand_temporal_weight "$TEMPORAL_WEIGHT" \
  --hand_query_similarity_threshold "$QUERY_SIMILARITY_THRESHOLD" \
  --hand_query_layers 8 12 16 20 \
  --mask_white_threshold 245 \
  --src_prompt "A person is holding a white plastic seasoning shaker with a blue cap in a kitchen." \
  --trg_prompt "A person is holding a ribbed silver aluminum can with horizontal ribs in a kitchen." \
  --src_word "white plastic seasoning shaker" \
  --trg_word "ribbed silver aluminum can" \
  --fg_boost_factor 4 \
  --blend_power 2 \
  --step "$STEP" \
  --seed 0 \
  --rollout_chunk_size 21 \
  --rollout_overlap_block_num 1 \
  2>&1 | tee "$OUTDIR/run.log"
