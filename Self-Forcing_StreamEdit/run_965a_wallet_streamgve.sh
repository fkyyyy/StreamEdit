#!/usr/bin/env bash
set -euo pipefail

# 965a_wallet: pure StreamGVE baseline editing smartphone → wallet.
# Uses the UNTOUCHED StreamGVE entrypoint (no hand mask, no flow, no
# factorized routing). Direct comparison target for 967f.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
STREAMGVE_ROOT="${STREAMGVE_ROOT:-/opt/tiger/CausalForcing/StreamGVE/Self-Forcing_StreamEdit}"
STREAMGVE_ENTRYPOINT="$STREAMGVE_ROOT/inference_edit_streamedit.py"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video1.mp4}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/965a_wallet_streamgve}"
OUTPUT_NAME="${OUTPUT_NAME:-965a-wallet-streamgve.mp4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-7}"
DRY_RUN="${DRY_RUN:-0}"
STEP="${STEP:-15}"

CONFIG_PATH="${CONFIG_PATH:-$SCRIPT_DIR/configs/self_forcing_dmd.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$SCRIPT_DIR/checkpoints/self_forcing_dmd.pt}"

readonly SRC_PROMPT='First-person POV shot, wide-angle lens. A person is relaxing on a beige sofa, holding a smartphone with both hands and actively scrolling through a calendar app interface. An open silver laptop rests on their lap, displaying a screen filled with blue technical diagrams or data charts. In the mid-ground, a pair of feet wearing white socks are propped up on a dark wooden coffee table, which is scattered with small white puzzle pieces. The background features a spacious, modern living room with a dark staircase on the left, a large black TV on a wooden cabinet, and two blue armchairs near a bright glass door on the right. Bright natural daylight, realistic 4k video style, slight fish-eye effect.'
readonly TRG_PROMPT='First-person POV shot, wide-angle lens. A person is relaxing on a beige sofa, holding a brown leather wallet with both hands, casually flipping it open. The wallet is a classic bifold design made of rich, dark brown genuine leather with visible grain texture and neat stitching along the edges. It has a slightly worn, warm patina on the surface. Beneath the wallet, an open silver laptop rests on their lap, displaying a screen filled with blue technical diagrams. In the mid-ground, a pair of feet wearing white socks are propped up on a dark wooden coffee table scattered with small white puzzle pieces. The background features a spacious, modern living room with a dark staircase on the left, a large black TV on a wooden cabinet, and two blue armchairs near a bright glass door on the right. Bright natural daylight, realistic 4k video style, slight fish-eye effect.'
readonly SRC_WORD='smartphone'
readonly TRG_WORD='brown leather wallet'

for required_path in \
  "$STREAMGVE_ENTRYPOINT" \
  "$DATA_PATH" \
  "$CONFIG_PATH" \
  "$CHECKPOINT_PATH" \
  "$SCRIPT_DIR/configs/default_config.yaml" \
  "$SCRIPT_DIR/wan_models/Wan2.1-T2V-1.3B/config.json"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

mkdir -p "$OUTDIR"

COMMAND=(
  "$PYTHON_BIN" "$STREAMGVE_ENTRYPOINT"
  --data_path "$DATA_PATH"
  --save_path "$OUTDIR/$OUTPUT_NAME"
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
  --config_path "$CONFIG_PATH"
  --checkpoint_path "$CHECKPOINT_PATH"
  "$@"
)

{
  printf '%s\n' \
    'experiment=965a_wallet_streamgve' \
    "streamgve_root=$STREAMGVE_ROOT" \
    "streamgve_entrypoint=$STREAMGVE_ENTRYPOINT" \
    'implementation=untouched_streamgve_python_modules' \
    'edit=smartphone_to_brown_leather_wallet' \
    'target_kv=ordinary_generated_target_history' \
    'routing=dynamic_sog' \
    'external_hand_mask=disabled' \
    'external_object_mask=disabled' \
    "data_path=$DATA_PATH"
  printf 'command='
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
} > "$OUTDIR/965a_wallet_config.txt"

echo "965A_WALLET pure StreamGVE baseline"
echo "965A_WALLET edit=smartphone→wallet routing=dynamic_sog"
echo "965A_WALLET no hand mask, no flow, no factorized routing"
echo "965A_WALLET_OUTPUT $OUTDIR/$OUTPUT_NAME"

if [[ "$DRY_RUN" == 1 ]]; then
  echo '965A_WALLET_DRY_RUN resolved command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

cd "$SCRIPT_DIR"
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
  "${COMMAND[@]}" 2>&1 | tee "$OUTDIR/run.log"
