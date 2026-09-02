#!/usr/bin/env bash
set -euo pipefail

# Hybrid ablation requested after 965a:
#   keep: automatic token-role localization + clean-source RGB optical flow
#   KV:   original StreamGVE dense clean generated-target history
#   drop: every later identity/memory/sink/prototype experiment
# This is a standalone Python entrypoint and never calls an older run script.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video1.mp4}"
HAND_MASK="${HAND_MASK:-$REPO_ROOT/hand_mask1.mp4}"
FLOW_CACHE="${FLOW_CACHE:-$SCRIPT_DIR/outputs/952b_phone_to_cardboard_box/source_flow/raft_large_bidirectional_flow.pt}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/965c_flow_token_streamgve_kv}"
OUTPUT_NAME="${OUTPUT_NAME:-965c-flow-token-streamgve-kv.mp4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-7}"
DRY_RUN="${DRY_RUN:-0}"

STEP="${STEP:-15}"
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
  echo "965c forbids SOURCE_OWNER_MASK and OBJECT_MASK" >&2
  exit 2
fi

# Keep this ablation one-dimensional. Appended CLI arguments may tune generic
# sampling parameters, but cannot silently re-enable another memory/operator.
for argument in "$@"; do
  case "$argument" in
    --object_mask_video|--object_mask_video=*|\
    --source_owner_mask_video|--source_owner_mask_video=*|\
    --src_prompt|--src_prompt=*|\
    --trg_prompt|--trg_prompt=*|\
    --src_word|--src_word=*|\
    --trg_word|--trg_word=*|\
    --routing_mode|--routing_mode=*|\
    --first_chunk_identity_replay|\
    --identity_first_latent_bootstrap|\
    --object_wise_anchor_reset|\
    --target_owned_object_handoff|\
    --factorized_target_identity|\
    --factorized_immutable_target_memory|\
    --factorized_native_target_history|\
    --factorized_owner_source_block|\
    --factorized_source_coordinate_target_delta|\
    --factorized_owner_complement_source|\
    --factorized_owner_complement_margin|--factorized_owner_complement_margin=*|\
    --factorized_owner_complement_min_preserve_confidence|--factorized_owner_complement_min_preserve_confidence=*|\
    --causal_paired_edit_memory|\
    --paired_memory_*|\
    --role_fixed_native_history|\
    --native_history_*|\
    --factorized_orthogonal_geometry|\
    --appearance_leakage_decomposition|\
    --target_semantic_competition|\
    --hand_flow_transactional_owner|\
    --source_coordinate_identity|\
    --source_identity_residual_carry|\
    --source_owner_residual_constraint|\
    --source_owner_geometry_envelope|\
    --motion_geometry_owner|\
    --source_flow_cache|--source_flow_cache=*|\
    --source_flow_role_fusion|\
    --source_flow_role_weight|--source_flow_role_weight=*|\
    --causal_owner_consistent_kv_metadata|\
    --rollout_chunk_size|--rollout_chunk_size=*|\
    --rollout_overlap_block_num|--rollout_overlap_block_num=*)
      echo "965c forbids mask, prompt, region, KV, or memory override: $argument" >&2
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

  # First novelty: token-role classification plus clean-source flow transport.
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
  --causal_owner_consistent_kv_metadata
  --factorized_owner_complement_source
  --factorized_owner_complement_margin 1
  --factorized_owner_complement_min_preserve_confidence 0.8

  # KV ablation: select the native StreamGVE dense clean target history.
  # role_fixed_native_history and every descendant history operator stay off.
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
    'experiment=965c_flow_token_streamgve_kv' \
    'entrypoint=direct_python_no_inherited_experiment_shells' \
    'editable_region=hand_token_roles_plus_clean_source_rgb_raft' \
    'region_transport=automatic_causal_motion_geometry_owner' \
    'region_flow_fusion=enabled' \
    'velocity_routing=factorized_role_conditioned' \
    'outside_region=clean_source_when_preserve_confidence_ge_0.8' \
    'self_attention=streamgve_native_dense_clean_target_history' \
    'target_kv_commit=clean_context_rerun_after_each_3_latent_frame_block' \
    'factorized_attention=counterfactual_diagnostic_only' \
    'role_fixed_native_history=disabled' \
    'paired_memory=disabled' \
    'immutable_target_memory=disabled' \
    'prototype_memory=disabled' \
    'multiframe_identity_sink=disabled' \
    'timestep_counterfactual_memory=disabled' \
    'external_hand_mask=enabled' \
    'external_object_mask=disabled' \
    'external_source_owner_mask=disabled' \
    "data_path=$DATA_PATH" \
    "hand_mask=$HAND_MASK" \
    "source_flow_cache=$FLOW_CACHE" \
    "source_flow_role_weight=$FLOW_ROLE_WEIGHT" \
    "src_word=$SRC_WORD" \
    "trg_word=$TRG_WORD" \
    "src_prompt=$SRC_PROMPT" \
    "trg_prompt=$TRG_PROMPT"
  printf 'command='
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
} > "$OUTDIR/965c_config.txt"

echo "965C_PROMPT src_word=$SRC_WORD trg_word=$TRG_WORD"
echo '965C_REGION token_roles=enabled clean_source_rgb_flow=enabled'
echo '965C_KV streamgve_dense_clean_target_history=enabled later_memory=disabled'
echo "965C_OUTPUT $OUTDIR/$OUTPUT_NAME"

if [[ "$DRY_RUN" == 1 ]]; then
  echo '965C_DRY_RUN resolved command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${COMMAND[@]}" 2>&1 | tee "$OUTDIR/run.log"

# Offline visualization only. No diagnostic map is fed back into inference.
"$PYTHON_BIN" "$SCRIPT_DIR/tools/visualize_inference_edit_regions.py" \
  --run-dir "$OUTDIR"
