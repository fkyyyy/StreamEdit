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
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-7}"
DRY_RUN="${DRY_RUN:-0}"
STEP="${STEP:-15}"

OBJECT_RESIDUAL_STRENGTH="${OBJECT_RESIDUAL_STRENGTH:-0.10}"
CONTACT_RESIDUAL_STRENGTH="${CONTACT_RESIDUAL_STRENGTH:-0.35}"
CONTACT_READ_WEIGHT="${CONTACT_READ_WEIGHT:-0.50}"
MIN_READ_PROBABILITY="${MIN_READ_PROBABILITY:-0.20}"
OBJECT_WRITE_THRESHOLD="${OBJECT_WRITE_THRESHOLD:-0.50}"
M2_STRENGTH="${M2_STRENGTH:-0.20}"
CONNECTED_GROWTH_STEPS="${CONNECTED_GROWTH_STEPS:-6}"
CONNECTED_CANDIDATE_RATIO="${CONNECTED_CANDIDATE_RATIO:-0.65}"
MAX_OBJECT_COVERAGE="${MAX_OBJECT_COVERAGE:-0.18}"

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

mkdir -p "$OUTDIR/roles"

COMMAND=(
  "$PYTHON_BIN" "$SCRIPT_DIR/inference_edit_streamedit.py"
  --data_path "$DATA_PATH"
  --hand_mask_video "$HAND_MASK"
  --save_path "$OUTDIR/$OUTPUT_NAME"
  --save_role_dir "$OUTDIR/roles"

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

  # Native target history remains intact. M2 is an additive residual channel.
  --factorized_native_target_history
  --role_memory_contact_read_weight "$CONTACT_READ_WEIGHT"
  --role_memory_min_read_probability "$MIN_READ_PROBABILITY"
  --role_memory_object_write_threshold "$OBJECT_WRITE_THRESHOLD"
  --immutable_delta_v_bank
  --closed_loop_delta_v_error
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
    'experiment=cook_S1M2_reliability_adaptive' \
    'edit=metal_spatula_to_wooden_spatula' \
    'S1=hand_semantics_plus_model_native_counterfactual_velocity' \
    'roles=object,contact,hand,background' \
    'uncertainty_policy=native_velocity_fallback_and_memory_abstention' \
    'owner_extent=reliability_adaptive_causal_area_budget' \
    'M2_write=two_frame_consensus_recovery_read_only' \
    "object_residual_strength=$OBJECT_RESIDUAL_STRENGTH" \
    "contact_residual_strength=$CONTACT_RESIDUAL_STRENGTH" \
    'M2=owner_indexed_closed_loop_deltaV_error' \
    "contact_read_weight=$CONTACT_READ_WEIGHT" \
    "min_read_probability=$MIN_READ_PROBABILITY" \
    "object_write_threshold=$OBJECT_WRITE_THRESHOLD" \
    "M2_strength=$M2_STRENGTH" \
    "connected_growth_steps=$CONNECTED_GROWTH_STEPS" \
    "connected_candidate_ratio=$CONNECTED_CANDIDATE_RATIO" \
    "max_object_coverage=$MAX_OBJECT_COVERAGE" \
    'native_attention_and_kv=unchanged' \
    'rgb_optical_flow=disabled' \
    'external_object_mask=disabled' \
    "data_path=$DATA_PATH" \
    "hand_mask=$HAND_MASK"
  printf 'command='
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
} > "$OUTDIR/cook_S1M2_config.txt"

echo 'Cook S1+M2: role-aware velocity routing + owner-indexed memory'
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
