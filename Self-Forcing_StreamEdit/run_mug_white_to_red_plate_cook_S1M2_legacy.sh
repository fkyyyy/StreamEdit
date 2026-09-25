#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Reproduce the original cook S1M2 execution semantics on the mug input.
# In that historical path the factorized native-target branch did not consume
# spatial Q/K or the M2 attention correction, although the delta-V bank was
# still constructed. Keep every later stabilization mechanism disabled so
# this remains a useful baseline rather than another full-method variant.
export DATA_PATH="${DATA_PATH:-/mnt/bn/public-lf4/fky/resources/CASE8/09_mug_white_0cf372ad96ca71749a1a81bfd330d77e.mp4}"
export HAND_MASK="${HAND_MASK:-/mnt/bn/public-lf4/fky/mug_hand/09_mug_white_0cf372ad96ca71749a1a81bfd330d77e_20260924_175331_321157/mask.mp4}"
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/mug_white_to_red_plate_cook_S1M2_legacy_washing}"
export OUTPUT_NAME="${OUTPUT_NAME:-mug-white-to-red-plate-cook-S1M2-legacy-washing.mp4}"
export CONFIG_NAME="${CONFIG_NAME:-mug_white_to_red_plate_cook_S1M2_legacy_washing_config.txt}"
export EXPERIMENT_NAME="mug_white_to_red_plate_cook_S1M2_legacy"
export EDIT_NAME="white_mug_to_red_plate"
export RUN_LABEL="White mug to red plate using original cook S1M2 path"

export S1M2_ATTENTION_MODE=legacy
export BACKGROUND_ANCHOR_STRENGTH=0.0
export BACKGROUND_OWNER_VETO_STRENGTH=0.0
export M2_CANONICAL_IDENTITY_COMMIT=0
export CUDA_DEVICE="${CUDA_DEVICE:-0}"

export WAN_MODELS_ROOT="${WAN_MODELS_ROOT:-/mnt/bn/public-lf4/fky/checkpoints/wan_models}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/bn/public-lf4/fky/checkpoints/checkpoints/self_forcing_dmd.pt}"
export SRC_PROMPT="${SRC_PROMPT:-A person is washing a white mug.}"
export TRG_PROMPT="${TRG_PROMPT:-A person is washing a red plate.}"
export SRC_WORD="${SRC_WORD:-white mug}"
export TRG_WORD="${TRG_WORD:-red plate}"

exec "$SCRIPT_DIR/run_cook_S1M2_full.sh" "$@"
