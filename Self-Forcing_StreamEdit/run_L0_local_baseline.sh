#!/usr/bin/env bash
set -euo pipefail

# L0: baseline — current StreamEdit code with all experimental flags OFF.
# Uses inference_edit_streamedit.py entry point (not the original StreamGVE),
# but passes NO custom flags (no hand mask, no flow, no routing override,
# no suppress, no anchor, no soft modulation).
#
# This should behave identically to 965a if the code base has not diverged.
# If L0 differs from 965a, there is a hidden code change in the default path.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video1.mp4}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/L0_local_baseline}"
OUTPUT_NAME="${OUTPUT_NAME:-L0-local-baseline.mp4}"
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

  # NO experimental flags:
  # - NO --hand_mask_video
  # - NO --routing_mode override (defaults to dynamic_sog)
  # - NO --source_flow_cache
  # - NO --motion_geometry_owner
  # - NO --soft_region_modulation
  # - NO --suppress_source_bg_value
  # - NO --first_block_identity_anchor
  # - NO --factorized_native_target_history

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
    'experiment=L0_local_baseline' \
    'code_entry=current_StreamEdit_inference_edit_streamedit.py' \
    'routing=default_dynamic_sog' \
    'hand_mask=disabled' \
    'flow=disabled' \
    'factorized=disabled' \
    'soft_modulation=disabled' \
    'suppress=disabled' \
    'anchor=disabled' \
    'comparison_target=965a_wallet_streamgve' \
    "data_path=$DATA_PATH"
  printf 'command='
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
} > "$OUTDIR/L0_config.txt"

echo "L0 local baseline: current code, all experimental flags OFF"
echo "L0 should match 965a if code has not diverged"
echo "L0_OUTPUT $OUTDIR/$OUTPUT_NAME"

if [[ "$DRY_RUN" == 1 ]]; then
  echo 'L0_DRY_RUN resolved command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${COMMAND[@]}" 2>&1 | tee "$OUTDIR/run.log"
