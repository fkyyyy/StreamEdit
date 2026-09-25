#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Isolated ablation for the turn/head-motion failure: keep the response gate,
# but reduce its source-preservation authority inside persistent causal-owner
# support. The continuous owner confidence softens the effect at uncertain
# edges. A strength of 0.75 retains at least 25% of the original background
# route, avoiding the all-target hard switch that previously caused detail
# loss.
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/mug_white_to_red_plate_S1M2_full_owner_veto_washing}"
export OUTPUT_NAME="${OUTPUT_NAME:-mug-white-to-red-plate-S1M2-full-owner-veto-washing.mp4}"
export CONFIG_NAME="${CONFIG_NAME:-mug_white_to_red_plate_S1M2_full_owner_veto_washing_config.txt}"
export BACKGROUND_OWNER_VETO_STRENGTH="${BACKGROUND_OWNER_VETO_STRENGTH:-0.75}"
export BACKGROUND_ANCHOR_STRENGTH="${BACKGROUND_ANCHOR_STRENGTH:-0.50}"
export CUDA_DEVICE="${CUDA_DEVICE:-0}"

export MUG_SRC_PROMPT="${MUG_SRC_PROMPT:-A person is washing a white mug.}"
export MUG_TRG_PROMPT="${MUG_TRG_PROMPT:-A person is washing a red plate.}"

exec "$SCRIPT_DIR/run_mug_white_to_red_plate_S1M2_full_response_gated_residual.sh" "$@"
