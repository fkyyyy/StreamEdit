#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ICLR_CASES_ROOT="${ICLR_CASES_ROOT:-/mnt/bn/public-lf4/fky/files/iclr_cases}"

export DATA_PATH="${DATA_PATH:-$ICLR_CASES_ROOT/resources/case9_5/combined (10).mp4}"
export HAND_MASK="${HAND_MASK:-$ICLR_CASES_ROOT/apple_hand/mask.mp4}"
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/apple_to_tennis_ball_S1M2_full_reviewed_baseline}"
export OUTPUT_NAME="${OUTPUT_NAME:-apple-to-tennis-ball-S1M2-full-reviewed-baseline.mp4}"
export CONFIG_NAME="${CONFIG_NAME:-apple_to_tennis_ball_S1M2_full_reviewed_baseline_config.txt}"
export EXPERIMENT_NAME="apple_to_tennis_ball_S1M2_full_reviewed_baseline"
export EDIT_NAME="apple_to_tennis_ball"
export RUN_LABEL="Apple to tennis ball S1+M2 Full reviewed baseline"

# Use the current complete method without changing its default parameters.
export S1M2_ATTENTION_MODE="${S1M2_ATTENTION_MODE:-full}"
export M2_CANONICAL_IDENTITY_COMMIT="${M2_CANONICAL_IDENTITY_COMMIT:-1}"
export WAN_MODELS_ROOT="${WAN_MODELS_ROOT:-/mnt/bn/public-lf4/fky/checkpoints/wan_models}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/bn/public-lf4/fky/checkpoints/checkpoints/self_forcing_dmd.pt}"
export CUDA_DEVICE="${CUDA_DEVICE:-0}"

export SRC_PROMPT="${APPLE_SRC_PROMPT:-A first-person view of a person holding and releasing a red-and-green apple above a wooden office desk.}"
export TRG_PROMPT="${APPLE_TRG_PROMPT:-A first-person view of a person holding and releasing a green tennis ball above a wooden office desk.}"
export SRC_WORD="${APPLE_SRC_WORD:-apple}"
export TRG_WORD="${APPLE_TRG_WORD:-tennis ball}"

exec "$SCRIPT_DIR/run_cook_S1M2_full.sh" "$@"
