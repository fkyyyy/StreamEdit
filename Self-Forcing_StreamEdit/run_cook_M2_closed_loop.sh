#!/usr/bin/env bash
set -euo pipefail

# Cook M2: closed-loop delta-V error correction on cook video.
# Edit: metal spatula → wooden spatula
# Uses the same M2 mechanism as the wallet experiments.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/cook.mp4}"
HAND_MASK="${HAND_MASK:-$REPO_ROOT/cook_handmask.mp4}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/cook_M2_closed_loop}"
OUTPUT_NAME="${OUTPUT_NAME:-cook-M2-closed-loop.mp4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-7}"
DRY_RUN="${DRY_RUN:-0}"
STEP="${STEP:-15}"

readonly SRC_PROMPT='First-person POV shot from above, wide-angle lens. A person is standing at a kitchen counter, cooking on an induction stovetop. The right hand holds a metal spatula, actively stirring and flipping diced ingredients in a large dark non-stick frying pan. The left hand grips the pan handle to steady it. Diced potatoes, onions, and small meat cubes are being stir-fried in the pan. A bowl of beaten eggs sits to the lower left. The granite countertop is cluttered with wine bottles, a stainless steel kettle, a white colander, glass jars, condiment bottles, and a small yellow cup. A second dark pan sits on the adjacent burner. Warm indoor lighting, realistic 4k video style, slight overhead fish-eye effect.'

readonly TRG_PROMPT='First-person POV shot from above, wide-angle lens. A person is standing at a kitchen counter, cooking on an induction stovetop. The right hand holds a wooden spatula with a flat, wide paddle head made of smooth light-colored natural wood with visible grain, actively stirring and flipping diced ingredients in a large dark non-stick frying pan. The left hand grips the pan handle to steady it. Diced potatoes, onions, and small meat cubes are being stir-fried in the pan. A bowl of beaten eggs sits to the lower left. The granite countertop is cluttered with wine bottles, a stainless steel kettle, a white colander, glass jars, condiment bottles, and a small yellow cup. A second dark pan sits on the adjacent burner. Warm indoor lighting, realistic 4k video style, slight overhead fish-eye effect.'

readonly SRC_WORD='metal spatula'
readonly TRG_WORD='wooden spatula'

if [[ ! -f "$DATA_PATH" ]]; then
  echo "Missing: $DATA_PATH" >&2
  exit 2
fi

mkdir -p "$OUTDIR"

COMMAND=(
  "$PYTHON_BIN" "$SCRIPT_DIR/inference_edit_streamedit.py"
  --data_path "$DATA_PATH"
  --save_path "$OUTDIR/$OUTPUT_NAME"

  --immutable_delta_v_bank
  --closed_loop_delta_v_error
  --immutable_delta_v_layers 8 12 16 20
  --immutable_delta_v_topk 8
  --immutable_delta_v_min_similarity 0.35
  --immutable_delta_v_strength 0.20
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
    'experiment=cook_M2_closed_loop' \
    'edit=metal_spatula_to_wooden_spatula' \
    'baseline=cook_L0_baseline' \
    'single_variable=closed_loop_delta_v_error' \
    'routing=default_dynamic_sog' \
    'owner=clean_source_trigger_word_cross_attention' \
    'bank_write=first_generated_block_once' \
    'desired_response=frozen_B0_deltaV_source_addressed' \
    'current_response=current_deltaV_current_source_addressed' \
    'correction=desired_response_minus_current_response' \
    'native_attention_unchanged=true' \
    'native_kv_unchanged=true' \
    'hand_mask=disabled' \
    'object_mask=disabled' \
    'flow=disabled' \
    'projected_residual=disabled' \
    'factorized=disabled' \
    'soft_modulation=disabled' \
    'source_bg_suppression=disabled' \
    'old_anchor=disabled' \
    'layers=8,12,16,20' \
    'topk=8' \
    'min_similarity=0.35' \
    'strength=0.20' \
    'max_error_ratio=1.0' \
    "data_path=$DATA_PATH"
  printf 'command='
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
} > "$OUTDIR/cook_M2_config.txt"

echo "Cook M2: metal spatula → wooden spatula, closed-loop delta-V error"
echo "OUTPUT $OUTDIR/$OUTPUT_NAME"

if [[ "$DRY_RUN" == 1 ]]; then
  echo 'DRY_RUN resolved command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${COMMAND[@]}" 2>&1 | tee "$OUTDIR/run.log"
