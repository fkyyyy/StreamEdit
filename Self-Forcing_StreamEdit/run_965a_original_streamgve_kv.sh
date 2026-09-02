#!/usr/bin/env bash
set -euo pipefail

# Strict StreamGVE KV-cache baseline. This launches the untouched StreamGVE
# inference entrypoint directly; none of this checkout's experimental identity
# memory, sink, owner, flow, or mask paths are imported.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
STREAMGVE_ROOT="${STREAMGVE_ROOT:-/opt/tiger/CausalForcing/StreamGVE/Self-Forcing_StreamEdit}"
STREAMGVE_ENTRYPOINT="$STREAMGVE_ROOT/inference_edit_streamedit.py"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video1.mp4}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/965a_original_streamgve_kv}"
OUTPUT_NAME="${OUTPUT_NAME:-965a-original-streamgve-kv.mp4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-7}"
DRY_RUN="${DRY_RUN:-0}"
STEP="${STEP:-15}"

# These assets are shared with the local StreamEdit checkout, while all model
# and pipeline Python modules come from STREAMGVE_ROOT. The two YAML files are
# byte-identical to the copies in the local StreamGVE checkout.
CONFIG_PATH="${CONFIG_PATH:-$SCRIPT_DIR/configs/self_forcing_dmd.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$SCRIPT_DIR/checkpoints/self_forcing_dmd.pt}"

readonly SRC_PROMPT='First-person POV shot, wide-angle lens. A person is relaxing on a beige sofa, holding a smartphone with both hands and actively scrolling through a calendar app interface. An open silver laptop rests on their lap, displaying a screen filled with blue technical diagrams or data charts. In the mid-ground, a pair of feet wearing white socks are propped up on a dark wooden coffee table, which is scattered with small white puzzle pieces. The background features a spacious, modern living room with a dark staircase on the left, a large black TV on a wooden cabinet, and two blue armchairs near a bright glass door on the right. Bright natural daylight, realistic 4k video style, slight fish-eye effect.'
readonly TRG_PROMPT='First-person POV shot, wide-angle lens. A person is relaxing on a beige sofa, holding a handheld calculator with both hands and actively pressing the buttons. The calculator has a compact rectangular body with rounded corners, molded in light gray matte plastic. It features a slightly glossy, dark LCD display window with a small reddish-brown solar strip above it. The keypad has raised round and rectangular buttons in darker gray and black with white numerals and symbols, creating a two-tone contrast. The surface is smooth plastic with mild reflections on the display. Beneath the calculator, an open silver laptop rests on their lap, displaying a screen filled with blue technical diagrams. In the mid-ground, a pair of feet wearing white socks are propped up on a dark wooden coffee table scattered with small white puzzle pieces. The background features a spacious, modern living room with a dark staircase on the left, a large black TV on a wooden cabinet, and two blue armchairs near a bright glass door on the right. Bright natural daylight, realistic 4k video style, slight fish-eye effect.'
readonly SRC_WORD='smartphone'
readonly TRG_WORD='handheld calculator'

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

if [[ -n "${HAND_MASK:-}" || -n "${SOURCE_OWNER_MASK:-}" || -n "${OBJECT_MASK:-}" ]]; then
  echo "965a is the original StreamGVE path and accepts no external masks" >&2
  exit 2
fi
for argument in "$@"; do
  case "$argument" in
    --src_prompt|--src_prompt=*|\
    --trg_prompt|--trg_prompt=*|\
    --src_word|--src_word=*|\
    --trg_word|--trg_word=*|\
    --config_path|--config_path=*|\
    --checkpoint_path|--checkpoint_path=*|\
    --rollout_chunk_size|--rollout_chunk_size=*|\
    --rollout_overlap_block_num|--rollout_overlap_block_num=*)
      echo "965a forbids baseline-defining override: $argument" >&2
      exit 2
      ;;
  esac
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
    'experiment=965a_original_streamgve_kv' \
    "streamgve_root=$STREAMGVE_ROOT" \
    "streamgve_entrypoint=$STREAMGVE_ENTRYPOINT" \
    'implementation=untouched_streamgve_python_modules' \
    'target_kv=ordinary_generated_target_history' \
    'target_kv_commit=clean_context_rerun_after_each_3_latent_frame_block' \
    'identity_memory=disabled' \
    'prototype_memory=disabled' \
    'multiframe_identity_sink=disabled' \
    'timestep_counterfactual_memory=disabled' \
    'motion_owner=disabled' \
    'external_hand_mask=disabled' \
    'external_object_mask=disabled' \
    'external_source_owner_mask=disabled' \
    "data_path=$DATA_PATH" \
    "config_path=$CONFIG_PATH" \
    "checkpoint_path=$CHECKPOINT_PATH" \
    "src_word=$SRC_WORD" \
    "trg_word=$TRG_WORD" \
    "src_prompt=$SRC_PROMPT" \
    "trg_prompt=$TRG_PROMPT"
  printf 'command='
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
} > "$OUTDIR/965a_config.txt"

echo "965A_IMPLEMENTATION $STREAMGVE_ENTRYPOINT"
echo "965A_PROMPT src_word=$SRC_WORD trg_word=$TRG_WORD"
echo "965A_TARGET $TRG_PROMPT"
echo '965A_KV mode=original_streamgve_generated_target_history'
echo "965A_OUTPUT $OUTDIR/$OUTPUT_NAME"

if [[ "$DRY_RUN" == 1 ]]; then
  echo '965A_DRY_RUN resolved command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

# StreamGVE resolves configs/, checkpoints/, and wan_models/ from cwd. Keep cwd
# here for the shared assets, while Python resolves imports from its own script
# directory (STREAMGVE_ROOT), guaranteeing the original implementation.
cd "$SCRIPT_DIR"
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
  "${COMMAND[@]}" 2>&1 | tee "$OUTDIR/run.log"
