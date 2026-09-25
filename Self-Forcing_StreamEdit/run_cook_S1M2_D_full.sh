#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export S1M2_ATTENTION_MODE=full
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/cook_S1M2_D_full}"
export OUTPUT_NAME="${OUTPUT_NAME:-cook-S1M2-D-full.mp4}"
exec "$SCRIPT_DIR/run_cook_S1M2_full.sh" "$@"
