#!/usr/bin/env bash
set -euo pipefail

# Cook S1+M2: interaction-conditioned token roles jointly control
# counterfactual-velocity routing and closed-loop appearance memory.
#
# S1 roles (soft): object / hand-object contact / hand / background.
# M2 read: object + discounted contact, attenuated by role entropy.
# M2 write: adaptive two-frame consensus core; recovery/contact/uncertain
# tokens are read-only and never enter the immutable bank.
# No RGB optical flow and no external object mask are used.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/cook.mp4}"
HAND_MASK="${HAND_MASK:-$REPO_ROOT/cook_handmask.mp4}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/cook_S1M2_adaptive}"
OUTPUT_NAME="${OUTPUT_NAME:-cook-S1M2-adaptive-role-aware.mp4}"
CONFIG_NAME="${CONFIG_NAME:-cook_S1M2_config.txt}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cook_S1M2_reliability_adaptive}"
EDIT_NAME="${EDIT_NAME:-metal_spatula_to_wooden_spatula}"
RUN_LABEL="${RUN_LABEL:-Cook S1+M2}"
CONFIG_PATH="${CONFIG_PATH:-$SCRIPT_DIR/configs/self_forcing_dmd.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$SCRIPT_DIR/checkpoints/self_forcing_dmd.pt}"
WAN_MODELS_ROOT="${WAN_MODELS_ROOT:-$SCRIPT_DIR/wan_models}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
DRY_RUN="${DRY_RUN:-0}"
STEP="${STEP:-15}"
S1M2_ATTENTION_MODE="${S1M2_ATTENTION_MODE:-full}"

case "$S1M2_ATTENTION_MODE" in
  legacy)
    SPATIAL_ATTENTION_ENABLED=0
    M2_ATTENTION_ENABLED=0
    ;;
  m2)
    SPATIAL_ATTENTION_ENABLED=0
    M2_ATTENTION_ENABLED=1
    ;;
  spatial)
    SPATIAL_ATTENTION_ENABLED=1
    M2_ATTENTION_ENABLED=0
    ;;
  full)
    SPATIAL_ATTENTION_ENABLED=1
    M2_ATTENTION_ENABLED=1
    ;;
  *)
    echo "Invalid S1M2_ATTENTION_MODE: $S1M2_ATTENTION_MODE" >&2
    exit 2
    ;;
esac
for extra_arg in "$@"; do
  case "$extra_arg" in
    --s1m2_attention_mode|--s1m2_attention_mode=*)
      echo "Set S1M2_ATTENTION_MODE instead of overriding the mode in extra arguments" >&2
      exit 2
      ;;
  esac
done

OBJECT_RESIDUAL_STRENGTH="${OBJECT_RESIDUAL_STRENGTH:-0.10}"
CONTACT_RESIDUAL_STRENGTH="${CONTACT_RESIDUAL_STRENGTH:-0.35}"
BACKGROUND_ANCHOR_STRENGTH="${BACKGROUND_ANCHOR_STRENGTH:-0.50}"
BACKGROUND_OWNER_VETO_STRENGTH="${BACKGROUND_OWNER_VETO_STRENGTH:-0.0}"
CONTACT_READ_WEIGHT="${CONTACT_READ_WEIGHT:-0.50}"
MIN_READ_PROBABILITY="${MIN_READ_PROBABILITY:-0.20}"
OBJECT_WRITE_THRESHOLD="${OBJECT_WRITE_THRESHOLD:-0.50}"
M2_STRENGTH="${M2_STRENGTH:-0.20}"
M2_CANONICAL_IDENTITY_COMMIT="${M2_CANONICAL_IDENTITY_COMMIT:-$M2_ATTENTION_ENABLED}"
M2_CANONICAL_COMMIT_STRENGTH="${M2_CANONICAL_COMMIT_STRENGTH:-0.50}"
CONNECTED_GROWTH_STEPS="${CONNECTED_GROWTH_STEPS:-6}"
CONNECTED_CANDIDATE_RATIO="${CONNECTED_CANDIDATE_RATIO:-0.65}"
MAX_OBJECT_COVERAGE="${MAX_OBJECT_COVERAGE:-0.18}"

SRC_PROMPT="${SRC_PROMPT:-First-person POV shot from above, wide-angle lens. A person is standing at a kitchen counter, cooking on an induction stovetop. The right hand holds a metal spatula, actively stirring and flipping diced ingredients in a large dark non-stick frying pan. The left hand grips the pan handle to steady it. Diced potatoes, onions, and small meat cubes are being stir-fried in the pan. A bowl of beaten eggs sits to the lower left. The granite countertop is cluttered with wine bottles, a stainless steel kettle, a white colander, glass jars, condiment bottles, and a small yellow cup. A second dark pan sits on the adjacent burner. Warm indoor lighting, realistic 4k video style, slight overhead fish-eye effect.}"
TRG_PROMPT="${TRG_PROMPT:-First-person POV shot from above, wide-angle lens. A person is standing at a kitchen counter, cooking on an induction stovetop. The right hand holds a wooden spatula with a flat, wide paddle head made of smooth light-colored natural wood with visible grain, actively stirring and flipping diced ingredients in a large dark non-stick frying pan. The left hand grips the pan handle to steady it. Diced potatoes, onions, and small meat cubes are being stir-fried in the pan. A bowl of beaten eggs sits to the lower left. The granite countertop is cluttered with wine bottles, a stainless steel kettle, a white colander, glass jars, condiment bottles, and a small yellow cup. A second dark pan sits on the adjacent burner. Warm indoor lighting, realistic 4k video style, slight overhead fish-eye effect.}"
SRC_WORD="${SRC_WORD:-metal spatula}"
TRG_WORD="${TRG_WORD:-wooden spatula}"
readonly SRC_PROMPT TRG_PROMPT SRC_WORD TRG_WORD

for required_path in \
  "$DATA_PATH" \
  "$HAND_MASK" \
  "$CONFIG_PATH" \
  "$CHECKPOINT_PATH" \
  "$WAN_MODELS_ROOT/Wan2.1-T2V-1.3B/config.json"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done
export WAN_MODELS_ROOT

mkdir -p "$OUTDIR/roles"

M2_CANONICAL_ARGS=()
if [[ "$M2_CANONICAL_IDENTITY_COMMIT" == 1 ]]; then
  if [[ "$M2_ATTENTION_ENABLED" != 1 ]]; then
    echo "M2_CANONICAL_IDENTITY_COMMIT requires an m2/full attention mode" >&2
    exit 2
  fi
  M2_CANONICAL_ARGS+=(
    --m2_canonical_identity_commit
    --m2_canonical_commit_strength "$M2_CANONICAL_COMMIT_STRENGTH"
  )
elif [[ "$M2_CANONICAL_IDENTITY_COMMIT" != 0 ]]; then
  echo "M2_CANONICAL_IDENTITY_COMMIT must be 0 or 1" >&2
  exit 2
fi

COMMAND=(
  "$PYTHON_BIN" "$SCRIPT_DIR/inference_edit_streamedit.py"
  --data_path "$DATA_PATH"
  --hand_mask_video "$HAND_MASK"
  --save_path "$OUTDIR/$OUTPUT_NAME"
  --save_role_dir "$OUTDIR/roles"
  --config_path "$CONFIG_PATH"
  --checkpoint_path "$CHECKPOINT_PATH"

  # S1: automatic hand-conditioned counterfactual-velocity roles.
  --routing_mode hand_role_factorized_causal_owner_kv
  --contact_graph_mode no_graph
  --hand_query_layers 8 12 16 20
  --hand_field_update_mode posterior
  --mask_white_threshold 245
  --hand_mask_mode overlay_white
  --hand_mask_overlay_diff_threshold 24
  --hand_causal_evidence
  --hand_persistent_occupancy 1.0
  --hand_connected_hysteresis
  --hand_connected_growth_steps "$CONNECTED_GROWTH_STEPS"
  --hand_connected_candidate_ratio "$CONNECTED_CANDIDATE_RATIO"
  --hand_max_object_coverage "$MAX_OBJECT_COVERAGE"
  --soft_region_modulation
  --soft_region_blend_strength 1.0
  --role_object_residual_strength "$OBJECT_RESIDUAL_STRENGTH"
  --role_contact_residual_strength "$CONTACT_RESIDUAL_STRENGTH"
  --role_background_anchor_strength "$BACKGROUND_ANCHOR_STRENGTH"
  --role_background_owner_veto_strength "$BACKGROUND_OWNER_VETO_STRENGTH"

  # Native target history remains intact. M2 is an additive residual channel.
  --factorized_native_target_history
  --s1m2_attention_mode "$S1M2_ATTENTION_MODE"
  --role_memory_contact_read_weight "$CONTACT_READ_WEIGHT"
  --role_memory_min_read_probability "$MIN_READ_PROBABILITY"
  --role_memory_object_write_threshold "$OBJECT_WRITE_THRESHOLD"
  --immutable_delta_v_bank
  --closed_loop_delta_v_error
  "${M2_CANONICAL_ARGS[@]}"
  --immutable_delta_v_layers 8 12 16 20
  --immutable_delta_v_topk 8
  --immutable_delta_v_min_similarity 0.35
  --immutable_delta_v_strength "$M2_STRENGTH"
  --immutable_delta_v_max_rms_ratio 1.0
  --closed_loop_delta_v_max_error_ratio 1.0

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
    "experiment=$EXPERIMENT_NAME" \
    "edit=$EDIT_NAME" \
    'S1=hand_semantics_plus_model_native_counterfactual_velocity' \
    'roles=object,contact,hand,background' \
    'uncertainty_policy=native_velocity_fallback_and_memory_abstention' \
    'owner_extent=reliability_adaptive_causal_area_budget' \
    'M2_base_write=two_frame_consensus_recovery_read_only' \
    'M2_canonical_write=verified_owner_plus_local_role_boundary' \
    "object_residual_strength=$OBJECT_RESIDUAL_STRENGTH" \
    "contact_residual_strength=$CONTACT_RESIDUAL_STRENGTH" \
    'background_residual_gate=native_target_source_response' \
    "background_anchor_strength=$BACKGROUND_ANCHOR_STRENGTH" \
    "background_owner_veto_strength=$BACKGROUND_OWNER_VETO_STRENGTH" \
    "M2_enabled=$M2_ATTENTION_ENABLED" \
    "spatial_qk_enabled=$SPATIAL_ATTENTION_ENABLED" \
    "s1m2_attention_mode=$S1M2_ATTENTION_MODE" \
    "contact_read_weight=$CONTACT_READ_WEIGHT" \
    "min_read_probability=$MIN_READ_PROBABILITY" \
    "object_write_threshold=$OBJECT_WRITE_THRESHOLD" \
    "M2_strength=$M2_STRENGTH" \
    "M2_canonical_identity_commit=$M2_CANONICAL_IDENTITY_COMMIT" \
    "M2_canonical_commit_strength=$M2_CANONICAL_COMMIT_STRENGTH" \
    "connected_growth_steps=$CONNECTED_GROWTH_STEPS" \
    "connected_candidate_ratio=$CONNECTED_CANDIDATE_RATIO" \
    "max_object_coverage=$MAX_OBJECT_COVERAGE" \
    'attention_order=scalar_or_spatial_qk_then_native_attention_then_optional_M2' \
    'target_key=native_unchanged' \
    'target_value=canonical_owner_only_when_enabled' \
    'non_owner_target_value=native_unchanged' \
    'rgb_optical_flow=disabled' \
    'external_object_mask=disabled' \
    "data_path=$DATA_PATH" \
    "hand_mask=$HAND_MASK" \
    "config_path=$CONFIG_PATH" \
    "checkpoint_path=$CHECKPOINT_PATH" \
    "wan_models_root=$WAN_MODELS_ROOT"
  printf 'command='
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
} > "$OUTDIR/$CONFIG_NAME"

echo "$RUN_LABEL: mode=$S1M2_ATTENTION_MODE role-aware velocity routing + owner-indexed memory"
echo "OUTPUT $OUTDIR/$OUTPUT_NAME"
echo "ROLES $OUTDIR/roles"

if [[ "$DRY_RUN" == 1 ]]; then
  echo 'DRY_RUN resolved command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${COMMAND[@]}" 2>&1 | tee "$OUTDIR/run.log"
