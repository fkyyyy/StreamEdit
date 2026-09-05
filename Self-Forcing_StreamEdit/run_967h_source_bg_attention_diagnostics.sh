#!/usr/bin/env bash
set -euo pipefail

# 967h: 967g generation, plus read-only attention-energy diagnostics.
# Generation behavior stays unchanged: original StreamGVE dynamic_sog with
# source-background keys retained and their values replaced by target values.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video1.mp4}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/967h_source_bg_attention_diagnostics}"
OUTPUT_NAME="${OUTPUT_NAME:-967h-source-bg-attention-diagnostics.mp4}"
DIAGNOSTIC_PATH="${DIAGNOSTIC_PATH:-$OUTDIR/source_bg_attention.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-7}"
DRY_RUN="${DRY_RUN:-0}"
STEP="${STEP:-15}"
QUERY_SAMPLES="${QUERY_SAMPLES:-4}"

readonly SRC_PROMPT='First-person POV shot, wide-angle lens. A person is relaxing on a beige sofa, holding a smartphone with both hands and actively scrolling through a calendar app interface. An open silver laptop rests on their lap, displaying a screen filled with blue technical diagrams or data charts. In the mid-ground, a pair of feet wearing white socks are propped up on a dark wooden coffee table, which is scattered with small white puzzle pieces. The background features a spacious, modern living room with a dark staircase on the left, a large black TV on a wooden cabinet, and two blue armchairs near a bright glass door on the right. Bright natural daylight, realistic 4k video style, slight fish-eye effect.'
readonly TRG_PROMPT='First-person POV shot, wide-angle lens. A person is relaxing on a beige sofa, holding a brown leather wallet with both hands, casually flipping it open. The wallet is a classic bifold design made of rich, brown genuine leather with visible grain texture and neat stitching along the edges. It has a slightly worn, warm patina on the surface. Beneath the wallet, an open silver laptop rests on their lap, displaying a screen filled with blue technical diagrams. In the mid-ground, a pair of feet wearing white socks are propped up on a dark wooden coffee table scattered with small white puzzle pieces. The background features a spacious, modern living room with a dark staircase on the left, a large black TV on a wooden cabinet, and two blue armchairs near a bright glass door on the right. Bright natural daylight, realistic 4k video style, slight fish-eye effect.'
readonly SRC_WORD='smartphone'
readonly TRG_WORD='brown leather wallet'

if [[ ! -f "$DATA_PATH" ]]; then
  echo "Missing: $DATA_PATH" >&2
  exit 2
fi
if ! [[ "$QUERY_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "QUERY_SAMPLES must be a positive integer: $QUERY_SAMPLES" >&2
  exit 2
fi

mkdir -p "$OUTDIR"

COMMAND=(
  "$PYTHON_BIN" "$SCRIPT_DIR/inference_edit_streamedit.py"
  --data_path "$DATA_PATH"
  --save_path "$OUTDIR/$OUTPUT_NAME"

  --suppress_source_bg_value
  --source_bg_attention_diagnostics
  --source_bg_attention_diagnostic_query_samples "$QUERY_SAMPLES"
  --source_bg_attention_diagnostic_path "$DIAGNOSTIC_PATH"

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
    'experiment=967h_source_bg_attention_diagnostics' \
    'baseline=967g_minimal_suppress' \
    'generation_delta=none' \
    'diagnostic=source_bg_attention_energy' \
    'routing=dynamic_sog_default' \
    'hand_mask=disabled' \
    'flow=disabled' \
    'object_mask=disabled' \
    "query_samples_per_role=$QUERY_SAMPLES" \
    "diagnostic_path=$DIAGNOSTIC_PATH" \
    "data_path=$DATA_PATH"
  printf 'command='
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
} > "$OUTDIR/967h_config.txt"

echo "967H diagnostic rerun: generation path is identical to 967g"
echo "967H no hand mask, no flow, no object mask, no anchor"
echo "967H_OUTPUT $OUTDIR/$OUTPUT_NAME"
echo "967H_DIAGNOSTICS $DIAGNOSTIC_PATH"

if [[ "$DRY_RUN" == 1 ]]; then
  echo '967H_DRY_RUN resolved command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${COMMAND[@]}" 2>&1 | tee "$OUTDIR/run.log"

"$PYTHON_BIN" "$SCRIPT_DIR/tools/summarize_source_bg_attention.py" \
  "$DIAGNOSTIC_PATH" \
  --csv "$OUTDIR/source_bg_attention_by_block.csv" \
  | tee "$OUTDIR/source_bg_attention_summary.txt"
