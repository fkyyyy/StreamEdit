#!/usr/bin/env bash
set -euo pipefail

# Standalone 964b: keep 964a's automatic owner and frozen first-block memory,
# but constrain only the target-minus-source direction shared by independent
# first-block prototypes. Current chunks retain pose/view/boundary/occlusion.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video1.mp4}"
HAND_MASK="${HAND_MASK:-$REPO_ROOT/hand_mask1.mp4}"
# Generated from the complete clean source RGB video; it is not an object mask.
FLOW_CACHE="${FLOW_CACHE:-$SCRIPT_DIR/outputs/952b_phone_to_cardboard_box/source_flow/raft_large_bidirectional_flow.pt}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/964b_pose_preserving_immutable_memory}"
OUTPUT_NAME="${OUTPUT_NAME:-964b-pose-preserving-immutable-memory.mp4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-7}"
DRY_RUN="${DRY_RUN:-0}"

STEP="${STEP:-15}"
# Subspace correction is already attenuated by cross-prototype coherence, so
# full strength applies only to the verified shared direction, not full KV.
IDENTITY_CORRECTION_STRENGTH="${IDENTITY_CORRECTION_STRENGTH:-1.0}"
IDENTITY_SUPPORT_FLOOR="${IDENTITY_SUPPORT_FLOOR:-1.0}"
IMMUTABLE_TARGET_NUM_PROTOTYPES="${IMMUTABLE_TARGET_NUM_PROTOTYPES:-16}"
HAND_MASK_MODE="${HAND_MASK_MODE:-overlay_white}"
HAND_MASK_OVERLAY_DIFF_THRESHOLD="${HAND_MASK_OVERLAY_DIFF_THRESHOLD:-24}"
HAND_PERSISTENT_OCCUPANCY="${HAND_PERSISTENT_OCCUPANCY:-1.0}"
CONNECTED_GROWTH_STEPS="${CONNECTED_GROWTH_STEPS:-3}"
CONNECTED_CANDIDATE_RATIO="${CONNECTED_CANDIDATE_RATIO:-0.75}"
FLOW_ROLE_WEIGHT="${FLOW_ROLE_WEIGHT:-0.75}"

readonly SRC_PROMPT='First-person POV shot, wide-angle lens. A person is relaxing on a beige sofa, holding a smartphone with both hands and actively scrolling through a calendar app interface. An open silver laptop rests on their lap, displaying a screen filled with blue technical diagrams or data charts. In the mid-ground, a pair of feet wearing white socks are propped up on a dark wooden coffee table, which is scattered with small white puzzle pieces. The background features a spacious, modern living room with a dark staircase on the left, a large black TV on a wooden cabinet, and two blue armchairs near a bright glass door on the right. Bright natural daylight, realistic 4k video style, slight fish-eye effect.'
readonly TRG_PROMPT='First-person POV shot, wide-angle lens. A person is relaxing on a beige sofa, holding a handheld calculator with both hands and actively pressing the buttons. The calculator has a compact rectangular body with rounded corners, molded in light gray matte plastic. It features a slightly glossy, dark LCD display window with a small reddish-brown solar strip above it. The keypad has raised round and rectangular buttons in darker gray and black with white numerals and symbols, creating a two-tone contrast. The surface is smooth plastic with mild reflections on the display. Beneath the calculator, an open silver laptop rests on their lap, displaying a screen filled with blue technical diagrams. In the mid-ground, a pair of feet wearing white socks are propped up on a dark wooden coffee table scattered with small white puzzle pieces. The background features a spacious, modern living room with a dark staircase on the left, a large black TV on a wooden cabinet, and two blue armchairs near a bright glass door on the right. Bright natural daylight, realistic 4k video style, slight fish-eye effect.'
readonly SRC_WORD='smartphone'
readonly TRG_WORD='handheld calculator'

for required_path in "$DATA_PATH" "$HAND_MASK" "$FLOW_CACHE"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

if [[ -n "${SOURCE_OWNER_MASK:-}" || -n "${OBJECT_MASK:-}" ]]; then
  echo "964b forbids SOURCE_OWNER_MASK and OBJECT_MASK" >&2
  exit 2
fi
for argument in "$@"; do
  case "$argument" in
    --object_mask_video|--object_mask_video=*|\
    --source_owner_mask_video|--source_owner_mask_video=*|\
    --src_prompt|--src_prompt=*|\
    --trg_prompt|--trg_prompt=*|\
    --src_word|--src_word=*|\
    --trg_word|--trg_word=*|\
    --factorized_native_target_history|\
    --role_fixed_native_history|\
    --native_history_*|\
    --immutable_target_layers|--immutable_target_layers=*|\
    --immutable_target_num_prototypes|--immutable_target_num_prototypes=*|\
    --immutable_target_value_mode|--immutable_target_value_mode=*|\
    --immutable_target_hard_owner|\
    --identity_correction_strength|--identity_correction_strength=*|\
    --identity_support_floor|--identity_support_floor=*)
      echo "964b forbids mask, prompt, or identity-ablation override: $argument" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUTDIR/roles"

COMMAND=(
  "$PYTHON_BIN" "$SCRIPT_DIR/inference_edit_streamedit.py"
  --data_path "$DATA_PATH"
  --hand_mask_video "$HAND_MASK"
  --save_path "$OUTDIR/$OUTPUT_NAME"
  --save_role_dir "$OUTDIR/roles"

  --routing_mode hand_role_factorized_causal_owner_kv
  --first_chunk_identity_replay
  --factorized_immutable_target_memory
  --immutable_target_layers 8 12 16 20
  --immutable_target_num_prototypes "$IMMUTABLE_TARGET_NUM_PROTOTYPES"
  --immutable_target_value_mode subspace
  --identity_correction_strength "$IDENTITY_CORRECTION_STRENGTH"
  --identity_support_floor "$IDENTITY_SUPPORT_FLOOR"

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
  --causal_owner_consistent_kv_metadata
  --factorized_owner_complement_source
  --factorized_owner_complement_margin 1
  --factorized_owner_complement_min_preserve_confidence 0.8

  --src_prompt "$SRC_PROMPT"
  --trg_prompt "$TRG_PROMPT"
  --src_word "$SRC_WORD"
  --trg_word "$TRG_WORD"
  --fg_boost_factor 4
  --blend_power 2
  --identity_max_occluded_blocks 1
  --identity_tokenprop_min_similarity 0.55
  --step "$STEP"
  --seed 0
  --rollout_chunk_size 21
  --rollout_overlap_block_num 1
  "$@"
)

{
  printf '%s\n' \
    'experiment=964b_pose_preserving_immutable_memory' \
    'entrypoint=direct_python_no_inherited_experiment_shells' \
    'identity_payload=first_three_frame_target_minus_source_residual_frozen' \
    'identity_operator=cross_prototype_coherent_residual_subspace' \
    'geometry_policy=current_chunk_owns_pose_view_boundary_occlusion' \
    'identity_address=current_clean_source_kv' \
    'identity_gate=automatic_hand_conditioned_clean_source_rgb_flow_owner' \
    'identity_layers=8,12,16,20' \
    'identity_value_mode=pose_preserving_subspace' \
    "identity_correction_strength=$IDENTITY_CORRECTION_STRENGTH" \
    "identity_support_floor=$IDENTITY_SUPPORT_FLOOR" \
    "identity_prototypes=$IMMUTABLE_TARGET_NUM_PROTOTYPES" \
    'prototype_assignment_diagnostics=entropy,peak,margin,per_layer_top1' \
    'native_target_history=disabled' \
    'multiframe_identity_sink=disabled' \
    'timestep_counterfactual_memory=disabled' \
    'external_hand_mask=enabled' \
    'external_object_mask=disabled' \
    'external_source_owner_mask=disabled' \
    "data_path=$DATA_PATH" \
    "hand_mask=$HAND_MASK" \
    "source_flow_cache=$FLOW_CACHE" \
    "src_word=$SRC_WORD" \
    "trg_word=$TRG_WORD" \
    "src_prompt=$SRC_PROMPT" \
    "trg_prompt=$TRG_PROMPT"
  printf 'command='
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
} > "$OUTDIR/964b_config.txt"

echo "964B_PROMPT src_word=$SRC_WORD trg_word=$TRG_WORD"
echo "964B_TARGET $TRG_PROMPT"
echo "964B_IDENTITY mode=subspace strength=$IDENTITY_CORRECTION_STRENGTH layers=8,12,16,20 prototypes=$IMMUTABLE_TARGET_NUM_PROTOTYPES"
echo "964B_OUTPUT $OUTDIR/$OUTPUT_NAME"

if [[ "$DRY_RUN" == 1 ]]; then
  echo '964B_DRY_RUN resolved command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${COMMAND[@]}" 2>&1 | tee "$OUTDIR/run.log"

# Read-only diagnostics; none of these artifacts are fed back to inference.
"$PYTHON_BIN" "$SCRIPT_DIR/tools/visualize_inference_edit_regions.py" \
  --run-dir "$OUTDIR"
