#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video.mp4}"
HAND_MASK="${HAND_MASK:-$REPO_ROOT/hand_mask.mp4}"
BACKBONE="${BACKBONE:-causal_forcing_plus_plus_2step}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
ROUTING_MODE="${ROUTING_MODE:-hand_role_bayes_flow_identity_kv}"
REFERENCE_IMAGE="${REFERENCE_IMAGE:-}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/907_bayes_causal_forcing}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

case "$BACKBONE" in
  causal_forcing_framewise|causal_forcing_plus_plus_2step)
    ;;
  *)
    echo "Unsupported Causal Forcing backbone: $BACKBONE" >&2
    exit 2
    ;;
esac

EXTRA_ARGS=()
if [[ -n "$CHECKPOINT_PATH" ]]; then
  EXTRA_ARGS+=(--checkpoint_path "$CHECKPOINT_PATH")
fi
if [[ "$ROUTING_MODE" == "hand_role_bayes_flow_customized_kv" ]]; then
  if [[ -z "$REFERENCE_IMAGE" || ! -f "$REFERENCE_IMAGE" ]]; then
    echo "Customized mode requires a valid REFERENCE_IMAGE." >&2
    exit 2
  fi
  EXTRA_ARGS+=(--first_frame_edit "$REFERENCE_IMAGE")
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
mkdir -p "$OUTDIR/roles"
cd "$SCRIPT_DIR"

python inference_edit_streamedit.py \
  --backbone "$BACKBONE" \
  --data_path "$DATA_PATH" \
  --hand_mask_video "$HAND_MASK" \
  --save_path "$OUTDIR/907-bayes-causal-forcing.mp4" \
  --save_role_dir "$OUTDIR/roles" \
  --routing_mode "$ROUTING_MODE" \
  --contact_graph_mode no_graph \
  --hand_query_layers 8 12 16 20 \
  --mask_white_threshold 245 \
  --src_prompt "A person is holding a white plastic seasoning shaker with a blue cap in a kitchen." \
  --trg_prompt "A person is holding a ribbed silver aluminum can with horizontal ribs in a kitchen." \
  --src_word "white plastic seasoning shaker" \
  --trg_word "ribbed silver aluminum can" \
  --fg_boost_factor 4 \
  --blend_power 2 \
  --seed 0 \
  --rollout_chunk_size 21 \
  --rollout_overlap_block_num 1 \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$OUTDIR/run.log"
