#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video.mp4}"
HAND_MASK="${HAND_MASK:-$REPO_ROOT/hand_mask.mp4}"
POSTERIOR_FLOW_MODE="${POSTERIOR_FLOW_MODE:-soft}"
POSTERIOR_FLOW_USE_FIELD="${POSTERIOR_FLOW_USE_FIELD:-0}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/907_posterior_role_flow_${POSTERIOR_FLOW_MODE}}"
STEP="${STEP:-15}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

if [[ "$POSTERIOR_FLOW_MODE" != "soft" && "$POSTERIOR_FLOW_MODE" != "hard" ]]; then
  echo "POSTERIOR_FLOW_MODE must be soft or hard" >&2
  exit 2
fi
if [[ "$POSTERIOR_FLOW_USE_FIELD" != "0" && "$POSTERIOR_FLOW_USE_FIELD" != "1" ]]; then
  echo "POSTERIOR_FLOW_USE_FIELD must be 0 or 1" >&2
  exit 2
fi

FIELD_ARGS=()
if [[ "$POSTERIOR_FLOW_USE_FIELD" == "1" ]]; then
  FIELD_ARGS+=(--posterior_flow_use_field)
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
mkdir -p "$OUTDIR/roles"
cd "$SCRIPT_DIR"

python inference_edit_streamedit.py \
  --data_path "$DATA_PATH" \
  --hand_mask_video "$HAND_MASK" \
  --save_path "$OUTDIR/907-posterior-role-flow-${POSTERIOR_FLOW_MODE}.mp4" \
  --save_role_dir "$OUTDIR/roles" \
  --routing_mode hand_role_posterior_flow_kv \
  --posterior_flow_mode "$POSTERIOR_FLOW_MODE" \
  "${FIELD_ARGS[@]}" \
  --contact_graph_mode no_graph \
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
