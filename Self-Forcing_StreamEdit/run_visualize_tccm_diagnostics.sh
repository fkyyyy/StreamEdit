#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:?Usage: bash run_visualize_tccm_diagnostics.sh OUTPUT_DIR}"

exec python "$SCRIPT_DIR/tools/visualize_tccm_diagnostics.py" \
  --run-dir "$RUN_DIR"
