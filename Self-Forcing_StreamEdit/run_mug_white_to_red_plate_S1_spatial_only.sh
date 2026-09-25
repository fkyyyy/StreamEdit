#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Hybrid diagnostic: retain the full method's spatial Q/K and background
# routing for motion/scene quality, but remove both forms of M2 correction so
# the generator's native target history remains the sole identity state.
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/mug_white_to_red_plate_S1_spatial_only_washing}"
export OUTPUT_NAME="${OUTPUT_NAME:-mug-white-to-red-plate-S1-spatial-only-washing.mp4}"
export CONFIG_NAME="${CONFIG_NAME:-mug_white_to_red_plate_S1_spatial_only_washing_config.txt}"
export EXPERIMENT_NAME="mug_white_to_red_plate_S1_spatial_only"
export RUN_LABEL="White mug to red plate S1 spatial-only identity diagnostic"

export S1M2_ATTENTION_MODE=spatial
export M2_CANONICAL_IDENTITY_COMMIT=0
export BACKGROUND_ANCHOR_STRENGTH="${BACKGROUND_ANCHOR_STRENGTH:-0.50}"
export BACKGROUND_OWNER_VETO_STRENGTH="${BACKGROUND_OWNER_VETO_STRENGTH:-0.75}"
export CUDA_DEVICE="${CUDA_DEVICE:-0}"

export MUG_SRC_PROMPT="${MUG_SRC_PROMPT:-A person is washing a white mug.}"
export MUG_TRG_PROMPT="${MUG_TRG_PROMPT:-A person is washing a red plate.}"

exec "$SCRIPT_DIR/run_mug_white_to_red_plate_S1M2_full_owner_veto.sh" "$@"
