#!/usr/bin/env bash
set -euo pipefail

# Cook F1V: velocity-native counterfactual causal provenance tracking.
# No RAFT and no object/source-owner mask. Clean-source query correspondence
# transports the owner; the model's first-step target-minus-source diffusion
# velocity verifies it. StreamGVE's native dense clean-target KV is unchanged.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/cook.mp4}"
HAND_MASK="${HAND_MASK:-$REPO_ROOT/cook_handmask.mp4}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/cook_F1V_velocity_native_owner}"
OUTPUT_NAME="${OUTPUT_NAME:-cook-F1V-velocity-native-owner.mp4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-7}"
DRY_RUN="${DRY_RUN:-0}"

STEP="${STEP:-15}"
HAND_MASK_MODE="${HAND_MASK_MODE:-overlay_white}"
HAND_MASK_OVERLAY_DIFF_THRESHOLD="${HAND_MASK_OVERLAY_DIFF_THRESHOLD:-24}"
HAND_PERSISTENT_OCCUPANCY="${HAND_PERSISTENT_OCCUPANCY:-1.0}"
CONNECTED_GROWTH_STEPS="${CONNECTED_GROWTH_STEPS:-3}"
CONNECTED_CANDIDATE_RATIO="${CONNECTED_CANDIDATE_RATIO:-0.75}"
VELOCITY_MIN_RESPONSE="${VELOCITY_MIN_RESPONSE:-0.10}"
VELOCITY_MIN_SIGNATURE_SIMILARITY="${VELOCITY_MIN_SIGNATURE_SIMILARITY:-0.0}"
VELOCITY_TRANSPORT_FLOOR="${VELOCITY_TRANSPORT_FLOOR:-0.25}"
VELOCITY_SIGNATURE_MOMENTUM="${VELOCITY_SIGNATURE_MOMENTUM:-0.80}"
GEOMETRY_STRENGTH="${GEOMETRY_STRENGTH:-0.35}"

readonly SRC_PROMPT='First-person POV shot from above, wide-angle lens. A person is standing at a kitchen counter, cooking on an induction stovetop. The right hand holds a metal spatula, actively stirring and flipping diced ingredients in a large dark non-stick frying pan. The left hand grips the pan handle to steady it. Diced potatoes, onions, and small meat cubes are being stir-fried in the pan. A bowl of beaten eggs sits to the lower left. The granite countertop is cluttered with wine bottles, a stainless steel kettle, a white colander, glass jars, condiment bottles, and a small yellow cup. A second dark pan sits on the adjacent burner. Warm indoor lighting, realistic 4k video style, slight overhead fish-eye effect.'

readonly TRG_PROMPT='First-person POV shot from above, wide-angle lens. A person is standing at a kitchen counter, cooking on an induction stovetop. The right hand holds a wooden spatula with a flat, wide paddle head made of smooth light-colored natural wood with visible grain, actively stirring and flipping diced ingredients in a large dark non-stick frying pan. The left hand grips the pan handle to steady it. Diced potatoes, onions, and small meat cubes are being stir-fried in the pan. A bowl of beaten eggs sits to the lower left. The granite countertop is cluttered with wine bottles, a stainless steel kettle, a white colander, glass jars, condiment bottles, and a small yellow cup. A second dark pan sits on the adjacent burner. Warm indoor lighting, realistic 4k video style, slight overhead fish-eye effect.'

readonly SRC_WORD='metal spatula'
readonly TRG_WORD='wooden spatula'

for required_path in "$DATA_PATH" "$HAND_MASK"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

if [[ -n "${SOURCE_OWNER_MASK:-}" || -n "${OBJECT_MASK:-}" || -n "${FLOW_CACHE:-}" ]]; then
  echo 'Cook F1V forbids SOURCE_OWNER_MASK, OBJECT_MASK, and FLOW_CACHE' >&2
  exit 2
fi

for argument in "$@"; do
  case "$argument" in
    --object_mask_video|--object_mask_video=*|\
    --source_owner_mask_video|--source_owner_mask_video=*|\
    --source_flow_cache|--source_flow_cache=*|\
    --motion_geometry_owner|\
    --source_flow_role_fusion|\
    --source_flow_verified_region|\
    --factorized_owner_complement_source|\
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
    --causal_paired_edit_memory|\
    --role_fixed_native_history|\
    --native_history_*|\
    --immutable_delta_v_bank|\
    --closed_loop_delta_v_error|\
    --first_block_identity_anchor|\
    --suppress_source_bg_value|\
    --counterfactual_source_bg_output|\
    --projected_source_residual|\
    --drop_source_bg_kv|\
    --soft_region_modulation|\
    --owner_query_drop_current_source_bg|\
    --rollout_chunk_size|--rollout_chunk_size=*|\
    --rollout_overlap_block_num|--rollout_overlap_block_num=*)
      echo "Cook F1V forbids flow, mask, memory, or prompt override: $argument" >&2
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

  --velocity_native_owner
  --velocity_owner_min_response "$VELOCITY_MIN_RESPONSE"
  --velocity_owner_min_signature_similarity "$VELOCITY_MIN_SIGNATURE_SIMILARITY"
  --velocity_owner_transport_floor "$VELOCITY_TRANSPORT_FLOOR"
  --velocity_owner_signature_momentum "$VELOCITY_SIGNATURE_MOMENTUM"

  # Closed owner permissions: raw source appearance is blocked. A bounded,
  # edit-orthogonal source residual remains as a geometry/lighting carrier.
  --factorized_native_target_history
  --factorized_owner_source_block
  --factorized_orthogonal_geometry
  --factorized_geometry_strength "$GEOMETRY_STRENGTH"
  --causal_owner_consistent_kv_metadata

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
    'experiment=cook_F1V_velocity_native_owner' \
    'comparison=cook_L0_baseline,cook_F1_flow_token_streamgve_kv' \
    'edit=metal_spatula_to_wooden_spatula' \
    'novelty=velocity_native_counterfactual_causal_provenance' \
    'owner_ignition=hand_token_semantics_x_counterfactual_velocity_response' \
    'owner_transport=clean_source_query_correspondence' \
    'owner_verification=velocity_signature_plus_query_cycle_consistency' \
    'external_rgb_flow=disabled' \
    'external_hand_mask=enabled' \
    'external_object_mask=disabled' \
    'external_source_owner_mask=disabled' \
    'owner_raw_source_fallback=blocked' \
    'owner_complement_policy=native_streamgve_fallback' \
    'owner_complement_source=disabled' \
    'self_attention=streamgve_native_dense_clean_target_history' \
    'M1_M2=disabled' \
    "geometry_safe_residual_strength=$GEOMETRY_STRENGTH" \
    "velocity_min_response=$VELOCITY_MIN_RESPONSE" \
    "velocity_min_signature_similarity=$VELOCITY_MIN_SIGNATURE_SIMILARITY" \
    "velocity_transport_floor=$VELOCITY_TRANSPORT_FLOOR" \
    "velocity_signature_momentum=$VELOCITY_SIGNATURE_MOMENTUM" \
    "data_path=$DATA_PATH" \
    "hand_mask=$HAND_MASK" \
    "src_word=$SRC_WORD" \
    "trg_word=$TRG_WORD"
  printf 'command='
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
} > "$OUTDIR/cook_F1V_config.txt"

echo "COOK_F1V_PROMPT src_word=$SRC_WORD trg_word=$TRG_WORD"
echo 'COOK_F1V_OWNER velocity_native=enabled raft=disabled object_mask=disabled'
echo "COOK_F1V_PERMISSION owner_source_fallback=blocked outside_owner=native_streamgve geometry_safe=$GEOMETRY_STRENGTH complement_source=disabled"
echo 'COOK_F1V_KV streamgve_dense_clean_target_history=enabled M1_M2=disabled'
echo "COOK_F1V_OUTPUT $OUTDIR/$OUTPUT_NAME"

if [[ "$DRY_RUN" == 1 ]]; then
  echo 'DRY_RUN resolved command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${COMMAND[@]}" 2>&1 | tee "$OUTDIR/run.log"

# Offline visualization only; nothing below is fed back into inference.
"$PYTHON_BIN" "$SCRIPT_DIR/tools/visualize_cook_f1v.py" \
  --source "$DATA_PATH" \
  --output "$OUTDIR/$OUTPUT_NAME" \
  --roles-dir "$OUTDIR/roles" \
  --output-dir "$OUTDIR/analysis"
