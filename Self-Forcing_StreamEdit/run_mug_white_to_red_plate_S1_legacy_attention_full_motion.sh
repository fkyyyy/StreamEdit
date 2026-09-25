#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Keep the identity-friendlier native attention of the cook legacy run while
# retaining the full run's response-gated S1 background/motion routing.
# Spatial Q/K, M2 attention, and canonical V commit are all disabled.
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/mug_white_to_red_plate_S1_legacy_attention_full_motion_washing}"
export OUTPUT_NAME="${OUTPUT_NAME:-mug-white-to-red-plate-S1-legacy-attention-full-motion-washing.mp4}"
export CONFIG_NAME="${CONFIG_NAME:-mug_white_to_red_plate_S1_legacy_attention_full_motion_washing_config.txt}"
export EXPERIMENT_NAME="mug_white_to_red_plate_S1_legacy_attention_full_motion"
export RUN_LABEL="White mug to red plate with legacy attention and full S1 motion routing"

export S1M2_ATTENTION_MODE=legacy
export M2_CANONICAL_IDENTITY_COMMIT=0
export BACKGROUND_ANCHOR_STRENGTH="${BACKGROUND_ANCHOR_STRENGTH:-0.50}"
export BACKGROUND_OWNER_VETO_STRENGTH="${BACKGROUND_OWNER_VETO_STRENGTH:-0.75}"
export CUDA_DEVICE="${CUDA_DEVICE:-0}"

export MUG_SRC_PROMPT="${MUG_SRC_PROMPT:-A person is washing a white mug.}"
export MUG_TRG_PROMPT="${MUG_TRG_PROMPT:-A person is washing a red plate.}"

exec "$SCRIPT_DIR/run_mug_white_to_red_plate_S1M2_full_owner_veto.sh" "$@"
