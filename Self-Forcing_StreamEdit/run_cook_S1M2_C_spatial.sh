#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export S1M2_ATTENTION_MODE=spatial
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/cook_S1M2_C_spatial}"
export OUTPUT_NAME="${OUTPUT_NAME:-cook-S1M2-C-spatial.mp4}"
exec "$SCRIPT_DIR/run_cook_S1M2_full.sh" "$@"
