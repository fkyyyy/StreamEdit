#!/usr/bin/env bash
set -euo pipefail

# Ball S1+M2: ivory stress ball → basketball
# Based on cook_S1M2_full configuration.
# S1 roles (soft): object / hand-object contact / hand / background.
# M2: adaptive two-frame consensus + closed-loop delta-V error correction.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DATA_PATH="${DATA_PATH:-/root/CASE8/01_ball_ivory_stress_0da77f69400be699d452104ef5fe301c.mp4}"
HAND_MASK="${HAND_MASK:-$SCRIPT_DIR/hand_mask_ball.mp4}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/ball_S1M2_basketball}"
OUTPUT_NAME="${OUTPUT_NAME:-ball-S1M2-ivory-to-basketball.mp4}"
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

readonly SRC_PROMPT='First-person POV shot from above, wide-angle lens. Two hands hold a smooth round ivory-white foam stress ball over a dark surface on a cluttered crafting table. The left hand cups the bottom of the ball while the right hand grips a piece of dark red fabric near the ball. A patterned tablecloth with blue circles and cartoon prints is visible in the background. Scattered on the table are a pair of black scissors, an orange container, a purple knitted stuffed toy, a spool of thread, a green pen, and folded yellow fabric. A red children chair and striped cushion are visible to the right. Warm indoor lighting, realistic 4k video style, slight overhead fish-eye effect.'
readonly TRG_PROMPT='First-person POV shot from above, wide-angle lens. Two hands hold a round orange basketball with black seam lines over a dark surface on a cluttered crafting table. The left hand cups the bottom of the basketball while the right hand grips a piece of dark red fabric near the basketball. A patterned tablecloth with blue circles and cartoon prints is visible in the background. Scattered on the table are a pair of black scissors, an orange container, a purple knitted stuffed toy, a spool of thread, a green pen, and folded yellow fabric. A red children chair and striped cushion are visible to the right. Warm indoor lighting, realistic 4k video style, slight overhead fish-eye effect.'
readonly SRC_WORD='ivory-white foam stress ball'
readonly TRG_WORD='orange basketball with black seam lines'

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

  # M2: owner-indexed closed-loop delta-V memory.
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
    'experiment=ball_S1M2_ivory_to_basketball' \
    'edit=ivory_stress_ball_to_basketball' \
    'S1=hand_semantics_plus_model_native_counterfactual_velocity' \
    'roles=object,contact,hand,background' \
    'M2=owner_indexed_closed_loop_deltaV_error' \
    "object_residual_strength=$OBJECT_RESIDUAL_STRENGTH" \
    "contact_residual_strength=$CONTACT_RESIDUAL_STRENGTH" \
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
} > "$OUTDIR/ball_S1M2_config.txt"

echo 'Ball S1+M2: ivory stress ball -> basketball'
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
