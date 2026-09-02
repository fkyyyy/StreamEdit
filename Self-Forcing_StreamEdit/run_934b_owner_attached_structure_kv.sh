#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/934b_owner_attached_structure_kv}"
OUTPUT_NAME="${OUTPUT_NAME:-934b-owner-attached-structure-kv.mp4}"
METHOD="${METHOD:-owner_attached_structure_counterfactual_kv}"
CONFIDENCE_POLICY="${CONFIDENCE_POLICY:-continuous_value_binary_query_single_write}"
PROJECTION_GATE="${PROJECTION_GATE:-owner_attached_object_structure}"

# Geometry ablation on top of 934a. The source-addressed residual may reach
# owner-attached object structure, including object-dominant contact/boundary
# cells, only after the normal clean-source correspondence, part-consistency,
# and cycle checks. Hand-dominant/background/unknown tokens remain native.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" METHOD="$METHOD" \
  CONFIDENCE_POLICY="$CONFIDENCE_POLICY" \
  PROJECTION_GATE="$PROJECTION_GATE" \
  "$SCRIPT_DIR/run_934a_single_confidence_kv.sh" \
  --paired_memory_owner_attached_boundary \
  "$@"
