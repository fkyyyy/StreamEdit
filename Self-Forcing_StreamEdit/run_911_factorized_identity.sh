#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

VARIANT="${VARIANT:-c}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video.mp4}"
HAND_MASK="${HAND_MASK:-$REPO_ROOT/hand_mask.mp4}"
ROUTING_MODE="${ROUTING_MODE:-hand_role_bayes_flow_tokenprop_kv}"
STEP="${STEP:-15}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
HAND_MASK_MODE="${HAND_MASK_MODE:-overlay_white}"
HAND_MASK_OVERLAY_DIFF_THRESHOLD="${HAND_MASK_OVERLAY_DIFF_THRESHOLD:-24}"
IDENTITY_TOKENPROP_MIN_SIMILARITY="${IDENTITY_TOKENPROP_MIN_SIMILARITY:-0.55}"
IDENTITY_TOKENPROP_GATE_STRENGTH="${IDENTITY_TOKENPROP_GATE_STRENGTH:-0.85}"
IDENTITY_TOKENPROP_MAX_CANDIDATES="${IDENTITY_TOKENPROP_MAX_CANDIDATES:-512}"
COMMITTED_MEMORY_FEEDBACK_STRENGTH="${COMMITTED_MEMORY_FEEDBACK_STRENGTH:-0.75}"
IDENTITY_CORRECTION_STRENGTH="${IDENTITY_CORRECTION_STRENGTH:-0.35}"
IDENTITY_MAX_OCCLUDED_BLOCKS="${IDENTITY_MAX_OCCLUDED_BLOCKS:-1}"
SRC_PROMPT="${SRC_PROMPT:-A first-person egocentric kitchen video of a person holding a white plastic seasoning shaker with a blue cap. The same hand, hand pose, camera motion, stovetop, pan, counter, and kitchen background are visible.}"
TRG_PROMPT="${TRG_PROMPT:-A first-person egocentric kitchen video of a person holding a red bottle. The same hand, hand motion, camera motion, stovetop, pan, counter, and kitchen background remain unchanged.}"
SRC_WORD="${SRC_WORD:-white plastic seasoning shaker}"
TRG_WORD="${TRG_WORD:-red bottle}"

case "$VARIANT" in
  a)
    EXPERIMENT="911a_first_chunk_replay"
    OUTPUT_NAME="911a-first-chunk-replay.mp4"
    EXTRA_ARGS=()
    ;;
  b)
    EXPERIMENT="911b_factorized_identity"
    OUTPUT_NAME="911b-factorized-identity.mp4"
    EXTRA_ARGS=(
      --factorized_target_identity
      --identity_correction_strength "$IDENTITY_CORRECTION_STRENGTH"
    )
    ;;
  c)
    EXPERIMENT="911c_visibility_lifecycle"
    OUTPUT_NAME="911c-visibility-lifecycle.mp4"
    EXTRA_ARGS=(
      --factorized_target_identity
      --identity_correction_strength "$IDENTITY_CORRECTION_STRENGTH"
      --identity_visibility_lifecycle
      --identity_max_occluded_blocks "$IDENTITY_MAX_OCCLUDED_BLOCKS"
    )
    ;;
  *)
    echo "VARIANT must be one of: a, b, c" >&2
    exit 2
    ;;
esac

OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/$EXPERIMENT}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
mkdir -p "$OUTDIR/roles"
cd "$SCRIPT_DIR"

"$PYTHON_BIN" inference_edit_streamedit.py \
  --data_path "$DATA_PATH" \
  --hand_mask_video "$HAND_MASK" \
  --save_path "$OUTDIR/$OUTPUT_NAME" \
  --save_role_dir "$OUTDIR/roles" \
  --routing_mode "$ROUTING_MODE" \
  --first_chunk_identity_replay \
  "${EXTRA_ARGS[@]}" \
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
