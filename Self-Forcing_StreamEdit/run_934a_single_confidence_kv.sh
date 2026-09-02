#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/934a_single_confidence_kv}"
OUTPUT_NAME="${OUTPUT_NAME:-934a-single-confidence-kv.mp4}"
METHOD="${METHOD:-single_confidence_source_transported_kv}"
CONFIDENCE_POLICY="${CONFIDENCE_POLICY:-continuous_value_binary_query_single_write}"
PROJECTION_GATE="${PROJECTION_GATE:-strict_object_interior}"

# Single-variable correction to 933a. Continuous source-match confidence is
# consumed exactly once by the current-value projection. Query arbitration is
# binary access control, and transactional promotion no longer squares the
# already role-weighted read support. Geometry/boundary policy is unchanged.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" METHOD="$METHOD" \
  CONFIDENCE_POLICY="$CONFIDENCE_POLICY" \
  PROJECTION_GATE="$PROJECTION_GATE" \
  "$SCRIPT_DIR/run_933a_source_transported_counterfactual_kv.sh" \
  --paired_memory_single_confidence \
  "$@"
