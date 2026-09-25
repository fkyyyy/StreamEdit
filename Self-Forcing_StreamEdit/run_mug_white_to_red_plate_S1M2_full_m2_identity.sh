#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Full S1+M2 with two complementary protections:
#   1. owner-veto prevents source appearance from washing the plate back out;
#   2. canonical M2 commits the first successful plate delta-V into later
#      owner-only clean target V writes, preserving the same plate instance.
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/mug_white_to_red_plate_S1M2_full_m2_identity_washing}"
export OUTPUT_NAME="${OUTPUT_NAME:-mug-white-to-red-plate-S1M2-full-m2-identity-washing.mp4}"
export CONFIG_NAME="${CONFIG_NAME:-mug_white_to_red_plate_S1M2_full_m2_identity_washing_config.txt}"
export BACKGROUND_OWNER_VETO_STRENGTH="${BACKGROUND_OWNER_VETO_STRENGTH:-0.75}"
export M2_CANONICAL_IDENTITY_COMMIT="${M2_CANONICAL_IDENTITY_COMMIT:-1}"
export M2_CANONICAL_COMMIT_STRENGTH="${M2_CANONICAL_COMMIT_STRENGTH:-0.50}"
export CUDA_DEVICE="${CUDA_DEVICE:-0}"

export MUG_SRC_PROMPT="${MUG_SRC_PROMPT:-A person is washing a white mug.}"
export MUG_TRG_PROMPT="${MUG_TRG_PROMPT:-A person is washing a red plate.}"

exec "$SCRIPT_DIR/run_mug_white_to_red_plate_S1M2_full_owner_veto.sh" "$@"
