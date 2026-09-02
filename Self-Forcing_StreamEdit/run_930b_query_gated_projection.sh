#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/930b_query_gated_projection}"
OUTPUT_NAME="${OUTPUT_NAME:-930b-query-gated-projection.mp4}"

# Output-side access-control ablation against 930a. Compute an untouched
# native attention output and a paired-projected counterfactual, then expose
# their difference only to object-interior queries with successful reads.
# Persistent projection stays disabled because historical projected values do
# not yet carry owner metadata that could enforce the same access policy.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" \
  "$SCRIPT_DIR/run_930a_transient_projection.sh" \
  --paired_memory_query_gated_projection \
  "$@"
