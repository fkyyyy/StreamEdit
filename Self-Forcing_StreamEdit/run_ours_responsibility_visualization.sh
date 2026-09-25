#!/usr/bin/env bash
set -euo pipefail

# End-to-end white-mug -> red-plate run using the full cook S1+M2 runner.
#
# Coordination contract with the inference hook:
#   --responsibility_diagnostics_dir DIR
# writes the nested hook dictionaries as DIR/responsibility/block_NNN.pt.  The
# exact field contract is documented and enforced by the offline visualizer.
# This wrapper deliberately reuses the canonical full cook runner instead of
# copying its inference configuration.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DATA_PATH="${DATA_PATH:-/mnt/bn/public-lf4/fky/resources/CASE8/09_mug_white_0cf372ad96ca71749a1a81bfd330d77e.mp4}"
HAND_MASK="${HAND_MASK:-/mnt/bn/public-lf4/fky/mug_hand/09_mug_white_0cf372ad96ca71749a1a81bfd330d77e_20260924_175331_321157/mask.mp4}"
SRC_PROMPT="${SRC_PROMPT:-A person is washing a white mug.}"
TRG_PROMPT="${TRG_PROMPT:-A person is washing a red plate.}"
SRC_WORD="${SRC_WORD:-white mug}"
TRG_WORD="${TRG_WORD:-red plate}"
CONFIG_PATH="${CONFIG_PATH:-$SCRIPT_DIR/configs/self_forcing_dmd.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/bn/public-lf4/fky/checkpoints/checkpoints/self_forcing_dmd.pt}"
WAN_MODELS_ROOT="${WAN_MODELS_ROOT:-/mnt/bn/public-lf4/fky/checkpoints/wan_models}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
DRY_RUN="${DRY_RUN:-0}"

ARTIFACT_ROOT="$SCRIPT_DIR/artifacts/ours_responsibility_visualization"
RUN_DIR="$ARTIFACT_ROOT/run"
RESPONSIBILITY_DIR="$RUN_DIR/responsibility"
INFERENCE_VIDEO="$RUN_DIR/ours_edited_hook_output.mp4"
HAND_ROLE_INPUT="$RUN_DIR/ours_edited_hook_output.hand_role_input.npz"

if (($#)); then
  echo "This reproducibility entrypoint accepts no positional/extra arguments." >&2
  echo "Use DATA_PATH, HAND_MASK, SRC_PROMPT, TRG_PROMPT, PYTHON_BIN, or CUDA_DEVICE environment variables." >&2
  exit 2
fi

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

mkdir -p "$RUN_DIR" "$ARTIFACT_ROOT/.matplotlib"

# Reuse the complete full S1+M2 method configuration while supplying the
# actual mug sample and its edit prompts.  The runner hard-codes seed 0; STEP
# is fixed here to 15 and its full S1+M2 attention mode is fixed explicitly.
env \
  DATA_PATH="$DATA_PATH" \
  HAND_MASK="$HAND_MASK" \
  SRC_PROMPT="$SRC_PROMPT" \
  TRG_PROMPT="$TRG_PROMPT" \
  SRC_WORD="$SRC_WORD" \
  TRG_WORD="$TRG_WORD" \
  OUTDIR="$RUN_DIR" \
  OUTPUT_NAME="$(basename -- "$INFERENCE_VIDEO")" \
  CONFIG_NAME="ours_responsibility_inference_config.txt" \
  CONFIG_PATH="$CONFIG_PATH" \
  CHECKPOINT_PATH="$CHECKPOINT_PATH" \
  WAN_MODELS_ROOT="$WAN_MODELS_ROOT" \
  PYTHON_BIN="$PYTHON_BIN" \
  CUDA_DEVICE="$CUDA_DEVICE" \
  DRY_RUN="$DRY_RUN" \
  STEP=15 \
  S1M2_ATTENTION_MODE=full \
  bash "$SCRIPT_DIR/run_cook_S1M2_full.sh" \
    --responsibility_diagnostics_dir "$RUN_DIR"

if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY_RUN: inference was not executed; offline visualization skipped."
  exit 0
fi

MPLCONFIGDIR="$ARTIFACT_ROOT/.matplotlib" "$PYTHON_BIN" \
  "$SCRIPT_DIR/tools/visualize_ours_responsibility.py" \
  --artifacts "$RESPONSIBILITY_DIR" \
  --source-video "$DATA_PATH" \
  --edited-video "$INFERENCE_VIDEO" \
  --hand-role-input "$HAND_ROLE_INPUT" \
  --output-dir "$ARTIFACT_ROOT"

echo "OURS_RESPONSIBILITY_OUTPUT $ARTIFACT_ROOT"
