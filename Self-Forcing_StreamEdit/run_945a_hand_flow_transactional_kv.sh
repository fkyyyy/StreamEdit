#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video.mp4}"
HAND_MASK="${HAND_MASK:-$REPO_ROOT/hand_mask.mp4}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/945a_hand_flow_transactional_kv}"
OUTPUT_NAME="${OUTPUT_NAME:-945a-hand-flow-transactional-kv.mp4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
STEP="${STEP:-15}"
HAND_MASK_MODE="${HAND_MASK_MODE:-overlay_white}"
HAND_MASK_OVERLAY_DIFF_THRESHOLD="${HAND_MASK_OVERLAY_DIFF_THRESHOLD:-24}"
OWNER_MAX_MISSING_FRAMES="${OWNER_MAX_MISSING_FRAMES:-1}"
VERIFIED_SOURCE_SUPPRESSION="${VERIFIED_SOURCE_SUPPRESSION:-0.35}"
SRC_PROMPT="${SRC_PROMPT:-A first-person egocentric kitchen video. A hand holds and moves the same cylindrical plastic seasoning bottle with a white body and blue screw cap. Preserve the bottle dimensions, cap seam, pose, scale, motion, hand grasp, occlusions, camera motion, cooking pan, food, countertop, stovetop, drawers, lighting, and background.}"
TRG_PROMPT="${TRG_PROMPT:-A first-person egocentric kitchen video identical to the source. Only the exterior color of the same cylindrical plastic seasoning bottle is changed: its body is matte violet and its flat-topped screw cap is dark violet. Preserve the exact cylindrical proportions, cap seam, pose, scale, motion, hand anatomy, grasp, and occlusions. The cooking pan, food, utensils, countertop, stovetop, drawers, lighting, and all background retain exactly their source appearance and colors.}"
SRC_WORD="${SRC_WORD:-seasoning bottle}"
TRG_WORD="${TRG_WORD:-seasoning bottle}"

for required_path in "$DATA_PATH" "$HAND_MASK"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

# The deployable experiment accepts hand information only.  Reject both
# environment-based and forwarded oracle object/source-owner inputs before
# Python starts, so this script cannot silently become an oracle ablation.
if [[ -n "${SOURCE_OWNER_MASK:-}" || -n "${OBJECT_MASK:-}" ]]; then
  echo "945a forbids SOURCE_OWNER_MASK and OBJECT_MASK" >&2
  exit 2
fi
for argument in "$@"; do
  case "$argument" in
    --source_owner_mask_video|--source_owner_mask_video=*|\
    --object_mask_video|--object_mask_video=*)
      echo "945a accepts hand information only; forbidden: $argument" >&2
      exit 2
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
mkdir -p "$OUTDIR/roles"
{
  printf '%s\n' \
    'baseline=919_causal_source_owner' \
    'method=hand_flow_transactional_native_kv' \
    'external_object_mask=disabled' \
    'external_source_owner_mask=disabled' \
    'external_hand_mask=enabled' \
    'token_roles=object,boundary,hand,background,unknown' \
    'owner_source=hand_proximity+source_attention+clean_source_transport+velocity_field' \
    'write=visible_non_hand_high_confidence_object_core' \
    'read=write_core+contact_boundary+bounded_lifecycle' \
    'uncertainty=abstain' \
    "owner_max_missing_frames=$OWNER_MAX_MISSING_FRAMES" \
    "verified_source_suppression=$VERIFIED_SOURCE_SUPPRESSION"
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/945a_config.txt"

cd "$SCRIPT_DIR"
echo "HAND_FLOW_INPUT_CONTRACT external_object_mask=disabled external_source_owner_mask=disabled external_hand_mask=enabled owner_source=hand_attention_source_transport_flow"

"$PYTHON_BIN" inference_edit_streamedit.py \
  --data_path "$DATA_PATH" \
  --hand_mask_video "$HAND_MASK" \
  --save_path "$OUTDIR/$OUTPUT_NAME" \
  --save_role_dir "$OUTDIR/roles" \
  --routing_mode hand_role_factorized_causal_owner_kv \
  --factorized_native_target_history \
  --role_fixed_native_history \
  --native_history_transactional_owner \
  --hand_flow_transactional_owner \
  --native_history_layers 8 12 16 20 \
  --native_history_max_tokens_per_frame 256 \
  --native_history_topk 8 \
  --native_history_min_similarity 0.35 \
  --native_history_min_write_confidence 0.50 \
  --native_history_min_query_confidence 0.50 \
  --native_history_canonical_logit_bias 1.0 \
  --native_history_owner_max_missing_frames "$OWNER_MAX_MISSING_FRAMES" \
  --native_history_verified_source_suppression "$VERIFIED_SOURCE_SUPPRESSION" \
  --contact_graph_mode no_graph \
  --hand_query_layers 8 12 16 20 \
  --hand_field_update_mode posterior \
  --mask_white_threshold 245 \
  --hand_mask_mode "$HAND_MASK_MODE" \
  --hand_mask_overlay_diff_threshold "$HAND_MASK_OVERLAY_DIFF_THRESHOLD" \
  --src_prompt "$SRC_PROMPT" \
  --trg_prompt "$TRG_PROMPT" \
  --src_word "$SRC_WORD" \
  --trg_word "$TRG_WORD" \
  --fg_boost_factor 4 \
  --blend_power 2 \
  --identity_max_occluded_blocks 1 \
  --identity_tokenprop_min_similarity 0.55 \
  --step "$STEP" \
  --seed 0 \
  --rollout_chunk_size 21 \
  --rollout_overlap_block_num 1 \
  "$@" \
  2>&1 | tee "$OUTDIR/run.log"
