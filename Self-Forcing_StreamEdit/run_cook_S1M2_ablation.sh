#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for variant in A_legacy B_m2 C_spatial D_full; do
  "$SCRIPT_DIR/run_cook_S1M2_${variant}.sh" "$@"
done
