#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export DATA_PATH="${DATA_PATH:-/mnt/bn/public-lf4/fky/resources/CASE8/09_mug_white_0cf372ad96ca71749a1a81bfd330d77e.mp4}"
export HAND_MASK="${HAND_MASK:-/mnt/bn/public-lf4/fky/mug_hand/09_mug_white_0cf372ad96ca71749a1a81bfd330d77e_20260924_175331_321157/mask.mp4}"
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/mug_white_to_red_plate_S1M2_full}"
export OUTPUT_NAME="${OUTPUT_NAME:-mug-white-to-red-plate-S1M2-full.mp4}"
export CONFIG_NAME="${CONFIG_NAME:-mug_white_to_red_plate_S1M2_config.txt}"
export EXPERIMENT_NAME="mug_white_to_red_plate_S1M2_full"
export EDIT_NAME="white_mug_to_red_plate"
export RUN_LABEL="White mug to red plate S1+M2"
export S1M2_ATTENTION_MODE="${S1M2_ATTENTION_MODE:-full}"
# Preserve the behavior of the existing mug baselines. The dedicated
# m2_identity entrypoint exports 1 before reaching this wrapper.
export M2_CANONICAL_IDENTITY_COMMIT="${M2_CANONICAL_IDENTITY_COMMIT:-0}"
export WAN_MODELS_ROOT="${WAN_MODELS_ROOT:-/mnt/bn/public-lf4/fky/checkpoints/wan_models}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/bn/public-lf4/fky/checkpoints/checkpoints/self_forcing_dmd.pt}"

export SRC_PROMPT="${MUG_SRC_PROMPT:-A person is washing a white mug.}"
export TRG_PROMPT="${MUG_TRG_PROMPT:-A person is washing a red plate.}"
export SRC_WORD="${MUG_SRC_WORD:-white mug}"
export TRG_WORD="${MUG_TRG_WORD:-red plate}"

exec "$SCRIPT_DIR/run_cook_S1M2_full.sh" "$@"
