#!/usr/bin/env bash
set -euo pipefail

# 967e: wallet edit with automatic region + frozen identity anchor.
# Combines soft region modulation (967d) with first-block identity anchor.
# Tests whether the anchor prevents ID drift and glare artifacts.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video1.mp4}"
HAND_MASK="${HAND_MASK:-$REPO_ROOT/hand_mask1.mp4}"
FLOW_CACHE="${FLOW_CACHE:-$SCRIPT_DIR/outputs/952b_phone_to_cardboard_box/source_flow/raft_large_bidirectional_flow.pt}"
PHONE_GT="${PHONE_GT:-$REPO_ROOT/phone_mask.mp4}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/967e5_raw_posterior_region}"
OUTPUT_NAME="${OUTPUT_NAME:-967e5-raw-posterior-region.mp4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-7}"
DRY_RUN="${DRY_RUN:-0}"
RUN_PHONE_GT_EVAL="${RUN_PHONE_GT_EVAL:-1}"

STEP="${STEP:-15}"
HAND_MASK_MODE="${HAND_MASK_MODE:-overlay_white}"
HAND_MASK_OVERLAY_DIFF_THRESHOLD="${HAND_MASK_OVERLAY_DIFF_THRESHOLD:-24}"
HAND_PERSISTENT_OCCUPANCY="${HAND_PERSISTENT_OCCUPANCY:-1.0}"
CONNECTED_GROWTH_STEPS="${CONNECTED_GROWTH_STEPS:-3}"
CONNECTED_CANDIDATE_RATIO="${CONNECTED_CANDIDATE_RATIO:-0.75}"
FLOW_ROLE_WEIGHT="${FLOW_ROLE_WEIGHT:-0.75}"
VERIFIED_OWNER_RADIUS="${VERIFIED_OWNER_RADIUS:-2}"
BACKGROUND_VETO_THRESHOLD="${BACKGROUND_VETO_THRESHOLD:-0.55}"
BACKGROUND_VETO_MIN_CONFIDENCE="${BACKGROUND_VETO_MIN_CONFIDENCE:-0.50}"
SOFT_REGION_BLEND_STRENGTH="${SOFT_REGION_BLEND_STRENGTH:-1.0}"
IDENTITY_ANCHOR_SCALE="${IDENTITY_ANCHOR_SCALE:-1.5}"

readonly SRC_PROMPT='First-person POV shot, wide-angle lens. A person is relaxing on a beige sofa, holding a smartphone with both hands and actively scrolling through a calendar app interface. An open silver laptop rests on their lap, displaying a screen filled with blue technical diagrams or data charts. In the mid-ground, a pair of feet wearing white socks are propped up on a dark wooden coffee table, which is scattered with small white puzzle pieces. The background features a spacious, modern living room with a dark staircase on the left, a large black TV on a wooden cabinet, and two blue armchairs near a bright glass door on the right. Bright natural daylight, realistic 4k video style, slight fish-eye effect.'
readonly TRG_PROMPT='First-person POV shot, wide-angle lens. A person is relaxing on a beige sofa, holding a brown leather wallet with both hands, casually flipping it open. The wallet is a classic bifold design made of rich, dark brown genuine leather with visible grain texture and neat stitching along the edges. It has a slightly worn, warm patina on the surface. Beneath the wallet, an open silver laptop rests on their lap, displaying a screen filled with blue technical diagrams. In the mid-ground, a pair of feet wearing white socks are propped up on a dark wooden coffee table scattered with small white puzzle pieces. The background features a spacious, modern living room with a dark staircase on the left, a large black TV on a wooden cabinet, and two blue armchairs near a bright glass door on the right. Bright natural daylight, realistic 4k video style, slight fish-eye effect.'
readonly SRC_WORD='smartphone'
readonly TRG_WORD='brown leather wallet'

for required_path in "$DATA_PATH" "$HAND_MASK" "$FLOW_CACHE"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done
if [[ -n "${SOURCE_OWNER_MASK:-}" || -n "${OBJECT_MASK:-}" ]]; then
  echo "967e forbids SOURCE_OWNER_MASK and OBJECT_MASK" >&2
  exit 2
fi

mkdir -p "$OUTDIR/roles"

COMMAND=(
  "$PYTHON_BIN" "$SCRIPT_DIR/inference_edit_streamedit.py"
  --data_path "$DATA_PATH"
  --hand_mask_video "$HAND_MASK"
  --save_path "$OUTDIR/$OUTPUT_NAME"
  --save_role_dir "$OUTDIR/roles"

  --routing_mode hand_role_factorized_causal_owner_kv
  --contact_graph_mode no_graph
  --hand_query_layers 8 12 16 20
  --hand_field_update_mode posterior
  --mask_white_threshold 245
  --hand_mask_mode "$HAND_MASK_MODE"
  --hand_mask_overlay_diff_threshold "$HAND_MASK_OVERLAY_DIFF_THRESHOLD"
  --hand_causal_evidence
  --hand_persistent_occupancy "$HAND_PERSISTENT_OCCUPANCY"
  --hand_connected_hysteresis
  --hand_connected_growth_steps "$CONNECTED_GROWTH_STEPS"
  --hand_connected_candidate_ratio "$CONNECTED_CANDIDATE_RATIO"
  --motion_geometry_owner
  --source_flow_cache "$FLOW_CACHE"
  --source_flow_role_fusion
  --source_flow_role_weight "$FLOW_ROLE_WEIGHT"
  --source_flow_verified_region
  --source_flow_verified_owner_radius "$VERIFIED_OWNER_RADIUS"
  --source_flow_background_veto_threshold "$BACKGROUND_VETO_THRESHOLD"
  --source_flow_background_veto_min_confidence "$BACKGROUND_VETO_MIN_CONFIDENCE"

  # Soft region modulation (no hard routing)
  --soft_region_modulation
  --soft_region_blend_strength "$SOFT_REGION_BLEND_STRENGTH"

  # Identity anchor: freeze block 0 target KV with amplified keys
  --first_block_identity_anchor
  --identity_anchor_scale "$IDENTITY_ANCHOR_SCALE"

  # Keep native StreamGVE dense clean-target history
  --factorized_native_target_history

  --src_prompt "$SRC_PROMPT"
  --trg_prompt "$TRG_PROMPT"
  --src_word "$SRC_WORD"
  --trg_word "$TRG_WORD"
  --fg_boost_factor 4
  --blend_power 2
  --step "$STEP"
  --seed 0
  --rollout_chunk_size 21
  --rollout_overlap_block_num 1
  "$@"
)

{
  printf '%s\n' \
    'experiment=967e_wallet_anchor' \
    'edit=smartphone_to_brown_leather_wallet' \
    'semantic_region=token_proposal_only' \
    'region_usage=soft_velocity_modulation' \
    'identity_anchor=frozen_first_block_amplified_keys' \
    'kv_metadata=legacy_cross_attention_union' \
    'external_hand_mask=enabled' \
    'external_object_mask=disabled' \
    "identity_anchor_scale=$IDENTITY_ANCHOR_SCALE" \
    "soft_region_blend_strength=$SOFT_REGION_BLEND_STRENGTH" \
    "data_path=$DATA_PATH" \
    "hand_mask=$HAND_MASK" \
    "source_flow_cache=$FLOW_CACHE"
  printf 'command='
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
} > "$OUTDIR/967e5_config.txt"

echo "967E_EDIT smartphone → brown leather wallet"
echo "967E_REGION auto flow-verified (no object mask)"
echo "967E_ANCHOR frozen_first_block scale=$IDENTITY_ANCHOR_SCALE"
echo "967E_OUTPUT $OUTDIR/$OUTPUT_NAME"

if [[ "$DRY_RUN" == 1 ]]; then
  echo '967E_DRY_RUN resolved command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${COMMAND[@]}" 2>&1 | tee "$OUTDIR/run.log"

"$PYTHON_BIN" "$SCRIPT_DIR/tools/visualize_inference_edit_regions.py" \
  --run-dir "$OUTDIR"
if [[ "$RUN_PHONE_GT_EVAL" == 1 ]]; then
  if [[ ! -f "$PHONE_GT" ]]; then
    echo "Missing offline-only phone GT: $PHONE_GT" >&2
    exit 2
  fi
  mkdir -p "$OUTDIR/.matplotlib"
  MPLCONFIGDIR="$OUTDIR/.matplotlib" \
  "$PYTHON_BIN" "$SCRIPT_DIR/tools/replay_flow_verified_region_phone_gt.py" \
    --run-dir "$OUTDIR" \
    --source-video "$DATA_PATH" \
    --phone-mask "$PHONE_GT" \
    --owner-radius "$VERIFIED_OWNER_RADIUS" \
    --background-veto-threshold "$BACKGROUND_VETO_THRESHOLD" \
    --background-veto-min-confidence "$BACKGROUND_VETO_MIN_CONFIDENCE"
fi
