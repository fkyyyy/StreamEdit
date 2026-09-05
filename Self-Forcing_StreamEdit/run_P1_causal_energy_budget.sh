#!/usr/bin/env bash
set -euo pipefail

# P1: P0 projected source residual + causal first-block energy budget.
#
# Single change from P0: for each denoising timestep, the first causal block's
# removed-energy fraction is frozen as a budget. Later blocks retain P0's
# projection direction but scale its magnitude down if it exceeds that budget.
#
# NO object/hand mask, NO flow region, NO factorized routing, NO suppress, NO
# anchor. Attention, KV, and the native StreamGVE soft SOG mask are unchanged.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video1.mp4}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/P1_causal_energy_budget}"
OUTPUT_NAME="${OUTPUT_NAME:-P1-causal-energy-budget.mp4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-7}"
DRY_RUN="${DRY_RUN:-0}"
STEP="${STEP:-15}"

readonly SRC_PROMPT='First-person POV shot, wide-angle lens. A person is relaxing on a beige sofa, holding a smartphone with both hands and actively scrolling through a calendar app interface. An open silver laptop rests on their lap, displaying a screen filled with blue technical diagrams or data charts. In the mid-ground, a pair of feet wearing white socks are propped up on a dark wooden coffee table, which is scattered with small white puzzle pieces. The background features a spacious, modern living room with a dark staircase on the left, a large black TV on a wooden cabinet, and two blue armchairs near a bright glass door on the right. Bright natural daylight, realistic 4k video style, slight fish-eye effect.'
readonly TRG_PROMPT='First-person POV shot, wide-angle lens. A person is relaxing on a beige sofa, holding a brown leather wallet with both hands, casually flipping it open. The wallet is a classic bifold design made of rich, dark brown genuine leather with visible grain texture and neat stitching along the edges. It has a slightly worn, warm patina on the surface. Beneath the wallet, an open silver laptop rests on their lap, displaying a screen filled with blue technical diagrams. In the mid-ground, a pair of feet wearing white socks are propped up on a dark wooden coffee table scattered with small white puzzle pieces. The background features a spacious, modern living room with a dark staircase on the left, a large black TV on a wooden cabinet, and two blue armchairs near a bright glass door on the right. Bright natural daylight, realistic 4k video style, slight fish-eye effect.'
readonly SRC_WORD='smartphone'
readonly TRG_WORD='brown leather wallet'

if [[ ! -f "$DATA_PATH" ]]; then
  echo "Missing: $DATA_PATH" >&2
  exit 2
fi

mkdir -p "$OUTDIR"

COMMAND=(
  "$PYTHON_BIN" "$SCRIPT_DIR/inference_edit_streamedit.py"
  --data_path "$DATA_PATH"
  --save_path "$OUTDIR/$OUTPUT_NAME"

  # P0 projection plus the single P1 change:
  --projected_source_residual
  --projected_source_residual_energy_budget

  # Everything else remains the native StreamGVE baseline.
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
    'experiment=P1_causal_energy_budget' \
    'baseline=P0_projected_residual' \
    'single_variable=causal_first_block_projection_energy_budget' \
    'budget_reference=first_causal_block_per_denoising_step' \
    'budget_scope=global_per_sample_no_spatial_region' \
    'routing=default_dynamic_sog' \
    'attention=native_streamgve' \
    'kv=native_streamgve' \
    'object_mask=disabled' \
    'hand_mask=disabled' \
    'flow=disabled' \
    'factorized=disabled' \
    'soft_modulation=disabled' \
    'suppress=disabled' \
    'anchor=disabled' \
    "data_path=$DATA_PATH"
  printf 'command='
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
} > "$OUTDIR/P1_config.txt"

echo "P1: P0 + causal first-block projection-energy budget"
echo "P1 keeps native StreamGVE attention/KV/SOG and adds no spatial region"
echo "P1_OUTPUT $OUTDIR/$OUTPUT_NAME"

if [[ "$DRY_RUN" == 1 ]]; then
  echo 'P1_DRY_RUN resolved command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${COMMAND[@]}" 2>&1 | tee "$OUTDIR/run.log"
