#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video.mp4}"
HAND_MASK="${HAND_MASK:-$REPO_ROOT/hand_mask.mp4}"
REFERENCE_IMAGE="${REFERENCE_IMAGE:-}"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/outputs/907_reference_causal_ablation}"
ABLATION="${ABLATION:-all}"
STEP="${STEP:-15}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

case "$ABLATION" in
  all|prompt_only|prefill_only)
    ;;
  *)
    echo "ABLATION must be all, prompt_only, or prefill_only." >&2
    exit 2
    ;;
esac

if [[ "$ABLATION" != "prompt_only" ]]; then
  if [[ -z "$REFERENCE_IMAGE" ]]; then
    echo "Set REFERENCE_IMAGE for the prefill-only ablation." >&2
    exit 2
  fi
  if [[ ! -f "$REFERENCE_IMAGE" ]]; then
    echo "Reference image does not exist: $REFERENCE_IMAGE" >&2
    exit 2
  fi
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
cd "$SCRIPT_DIR"

run_case() {
  local name="$1"
  shift
  local outdir="$OUT_ROOT/$name"
  mkdir -p "$outdir/roles"

  python inference_edit_streamedit.py \
    --data_path "$DATA_PATH" \
    --hand_mask_video "$HAND_MASK" \
    --save_path "$outdir/907-${name}.mp4" \
    --save_role_dir "$outdir/roles" \
    --routing_mode hand_role_bayes_flow_identity_kv \
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
    "$@" \
    2>&1 | tee "$outdir/run.log"
}

if [[ "$ABLATION" == "all" || "$ABLATION" == "prompt_only" ]]; then
  run_case prompt_only
fi

if [[ "$ABLATION" == "all" || "$ABLATION" == "prefill_only" ]]; then
  run_case prefill_only --first_frame_edit "$REFERENCE_IMAGE"
fi
