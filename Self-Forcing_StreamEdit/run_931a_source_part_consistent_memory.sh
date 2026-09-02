#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/931a_source_part_consistent_memory}"
OUTPUT_NAME="${OUTPUT_NAME:-931a-source-part-consistent-memory.mp4}"

# Object-internal part-consistency ablation against 930b. Canonical residuals
# remain source-addressed, local, uncertainty-gated, transient, and query-
# gated. The sole additional mechanism is a normalized clean-source V
# signature per memory slot, used to prevent cross-part top-k mixing without
# semantic labels or object-specific color rules.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" \
  "$SCRIPT_DIR/run_930b_query_gated_projection.sh" \
  --paired_memory_source_part_consistency \
  --paired_memory_min_part_similarity 0.45 \
  --paired_memory_part_similarity_margin 0.08 \
  "$@"
