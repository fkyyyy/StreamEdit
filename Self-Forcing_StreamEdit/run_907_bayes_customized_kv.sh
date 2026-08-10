#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video.mp4}"
HAND_MASK="${HAND_MASK:-$REPO_ROOT/hand_mask.mp4}"
REFERENCE_IMAGE="${REFERENCE_IMAGE:-}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/907_bayes_customized_kv}"
STEP="${STEP:-15}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

if [[ -z "$REFERENCE_IMAGE" ]]; then
  echo "Set REFERENCE_IMAGE to an aligned edited source first frame." >&2
  exit 2
fi
if [[ ! -f "$REFERENCE_IMAGE" ]]; then
  echo "Reference image does not exist: $REFERENCE_IMAGE" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
mkdir -p "$OUTDIR/roles"
cd "$SCRIPT_DIR"

python inference_edit_streamedit.py \
  --data_path "$DATA_PATH" \
  --hand_mask_video "$HAND_MASK" \
  --first_frame_edit "$REFERENCE_IMAGE" \
  --save_path "$OUTDIR/907-bayes-customized-kv.mp4" \
  --save_role_dir "$OUTDIR/roles" \
  --routing_mode hand_role_bayes_flow_customized_kv \
  --contact_graph_mode no_graph \
  --hand_query_layers 8 12 16 20 \
  --mask_white_threshold 245 \
  --src_prompt "A person is holding a white plastic seasoning shaker with a blue cap in a kitchen." \
  --trg_prompt "A person is holding a red Coca-Cola aluminum soda can with a white Coca-Cola logo in a kitchen." \
  --src_word "white plastic seasoning shaker" \
  --trg_word "red Coca-Cola aluminum soda can" \
  --fg_boost_factor 4 \
  --blend_power 2 \
  --step "$STEP" \
  --seed 0 \
  --rollout_chunk_size 21 \
  --rollout_overlap_block_num 1 \
  2>&1 | tee "$OUTDIR/run.log"
