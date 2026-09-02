#!/usr/bin/env bash
set -euo pipefail

# Standalone calculator-prompt entry point. This script invokes Python
# directly and never delegates to an earlier experiment shell script.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video1.mp4}"
HAND_MASK="${HAND_MASK:-$REPO_ROOT/hand_mask1.mp4}"
# This cache contains optical flow computed only from the clean source RGB.
FLOW_CACHE="${FLOW_CACHE:-$SCRIPT_DIR/outputs/952b_phone_to_cardboard_box/source_flow/raft_large_bidirectional_flow.pt}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/963c_handheld_calculator_standalone}"
OUTPUT_NAME="${OUTPUT_NAME:-963c-handheld-calculator-standalone.mp4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-7}"
DRY_RUN="${DRY_RUN:-0}"

STEP="${STEP:-15}"
HAND_MASK_MODE="${HAND_MASK_MODE:-overlay_white}"
HAND_MASK_OVERLAY_DIFF_THRESHOLD="${HAND_MASK_OVERLAY_DIFF_THRESHOLD:-24}"
OWNER_MAX_MISSING_FRAMES="${OWNER_MAX_MISSING_FRAMES:-1}"
VERIFIED_SOURCE_SUPPRESSION="${VERIFIED_SOURCE_SUPPRESSION:-1.0}"
ATTENTION_AUTHORITY_STRENGTH="${ATTENTION_AUTHORITY_STRENGTH:-1.0}"
ENTRY_BRIDGE_STRENGTH="${ENTRY_BRIDGE_STRENGTH:-1.0}"
MIN_RESIDUAL_CONSENSUS="${MIN_RESIDUAL_CONSENSUS:-0.05}"
HAND_PERSISTENT_OCCUPANCY="${HAND_PERSISTENT_OCCUPANCY:-1.0}"
CONNECTED_GROWTH_STEPS="${CONNECTED_GROWTH_STEPS:-3}"
CONNECTED_CANDIDATE_RATIO="${CONNECTED_CANDIDATE_RATIO:-0.75}"
FLOW_ROLE_WEIGHT="${FLOW_ROLE_WEIGHT:-0.75}"
FLOW_MIN_CONFIDENCE="${FLOW_MIN_CONFIDENCE:-0.10}"
RESIDUAL_UPDATE_MIN_COSINE="${RESIDUAL_UPDATE_MIN_COSINE:-0.50}"
RESIDUAL_UPDATE_MIN_MAGNITUDE_RATIO="${RESIDUAL_UPDATE_MIN_MAGNITUDE_RATIO:-0.90}"
SINK_TOPK_PER_FRAME="${SINK_TOPK_PER_FRAME:-8}"
SINK_SOURCE_LOGIT_BIAS="${SINK_SOURCE_LOGIT_BIAS:-1.0}"
SINK_STRENGTH="${SINK_STRENGTH:-1.0}"
TCCM_FLOW_RADIUS="${TCCM_FLOW_RADIUS:-2.0}"
TCCM_STRENGTH="${TCCM_STRENGTH:-1.0}"
TCCM_MAX_ERROR_RATIO="${TCCM_MAX_ERROR_RATIO:-1.0}"

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
  echo "963c forbids SOURCE_OWNER_MASK and OBJECT_MASK" >&2
  exit 2
fi
for argument in "$@"; do
  case "$argument" in
    --object_mask_video|--object_mask_video=*|\
    --source_owner_mask_video|--source_owner_mask_video=*|\
    --src_prompt|--src_prompt=*|\
    --trg_prompt|--trg_prompt=*|\
    --src_word|--src_word=*|\
    --trg_word|--trg_word=*)
      echo "963c forbids mask or prompt identity override: $argument" >&2
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
  --factorized_native_target_history
  --role_fixed_native_history
  --native_history_transactional_owner
  --hand_flow_transactional_owner
  --native_history_layers 8 12 16 20
  --native_history_max_tokens_per_frame 256
  --native_history_topk 8
  --native_history_min_similarity 0.35
  --native_history_min_write_confidence 0.50
  --native_history_min_query_confidence 0.50
  --native_history_canonical_logit_bias 1.0
  --native_history_owner_max_missing_frames "$OWNER_MAX_MISSING_FRAMES"
  --native_history_verified_source_suppression "$VERIFIED_SOURCE_SUPPRESSION"
  --contact_graph_mode no_graph
  --hand_query_layers 8 12 16 20
  --hand_field_update_mode posterior
  --mask_white_threshold 245
  --hand_mask_mode "$HAND_MASK_MODE"
  --hand_mask_overlay_diff_threshold "$HAND_MASK_OVERLAY_DIFF_THRESHOLD"
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

  --native_history_consistent_transaction
  --native_history_verified_attention_authority
  --native_history_attention_authority_strength "$ATTENTION_AUTHORITY_STRENGTH"
  --native_history_coalesce_bootstrap_time
  --native_history_recent_entry_bridge
  --native_history_entry_bridge_strength "$ENTRY_BRIDGE_STRENGTH"
  --native_history_dense_recent_min_residual_consensus "$MIN_RESIDUAL_CONSENSUS"

  --hand_causal_evidence
  --hand_persistent_occupancy "$HAND_PERSISTENT_OCCUPANCY"
  --hand_connected_hysteresis
  --hand_connected_growth_steps "$CONNECTED_GROWTH_STEPS"
  --hand_connected_candidate_ratio "$CONNECTED_CANDIDATE_RATIO"

  --motion_geometry_owner
  --source_flow_cache "$FLOW_CACHE"
  --native_history_motion_owner_dense_read
  --factorized_owner_complement_source
  --factorized_owner_complement_margin 1
  --factorized_owner_complement_min_preserve_confidence 0.8

  --native_history_token_atomic_payload
  --native_history_topology_complete_read
  --native_history_persistent_residual_upsert
  --native_history_last_trusted_appearance
  --native_history_residual_update_min_cosine "$RESIDUAL_UPDATE_MIN_COSINE"
  --native_history_residual_update_min_magnitude_ratio "$RESIDUAL_UPDATE_MIN_MAGNITUDE_RATIO"

  --source_flow_role_fusion
  --source_flow_role_weight "$FLOW_ROLE_WEIGHT"
  --native_history_flow_indexed_residual
  --native_history_flow_min_confidence "$FLOW_MIN_CONFIDENCE"
  --native_history_decoupled_flow_trust

  --native_history_multiframe_identity_sink
  --native_history_multiframe_sink_topk_per_frame "$SINK_TOPK_PER_FRAME"
  --native_history_multiframe_sink_source_logit_bias "$SINK_SOURCE_LOGIT_BIAS"
  --native_history_multiframe_sink_strength "$SINK_STRENGTH"

  --native_history_timestep_counterfactual_memory
  --native_history_tccm_flow_radius "$TCCM_FLOW_RADIUS"
  --native_history_tccm_strength "$TCCM_STRENGTH"
  --native_history_tccm_max_error_ratio "$TCCM_MAX_ERROR_RATIO"
  "$@"
)

{
  printf '%s\n' \
    'experiment=963c_handheld_calculator_standalone' \
    'entrypoint=direct_python_no_inherited_experiment_shells' \
    'baseline=963a_timestep_counterfactual_memory' \
    'change=prompt_only' \
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
} > "$OUTDIR/963c_config.txt"

echo "963C_PROMPT src_word=$SRC_WORD trg_word=$TRG_WORD"
echo "963C_TARGET $TRG_PROMPT"
echo "963C_OUTPUT $OUTDIR/$OUTPUT_NAME"

if [[ "$DRY_RUN" == 1 ]]; then
  echo '963C_DRY_RUN resolved command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${COMMAND[@]}" 2>&1 | tee "$OUTDIR/run.log"

# Read-only post-processing; neither visualization is fed back to inference.
"$PYTHON_BIN" "$SCRIPT_DIR/tools/visualize_inference_edit_regions.py" \
  --run-dir "$OUTDIR"
"$PYTHON_BIN" "$SCRIPT_DIR/tools/visualize_tccm_diagnostics.py" \
  --run-dir "$OUTDIR"
