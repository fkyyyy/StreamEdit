#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/935a_dual_timescale_anchor}"
OUTPUT_NAME="${OUTPUT_NAME:-935a-dual-timescale-anchor.mp4}"
METHOD="${METHOD:-source_addressed_role_aware_dual_timescale_kv}"
CONFIDENCE_POLICY="${CONFIDENCE_POLICY:-continuous_anchor_value_binary_owner_query}"
PROJECTION_GATE="${PROJECTION_GATE:-strict_object_interior}"

# EditaLive-inspired but training-free ablation against 934a.  The untouched
# dense target history remains the motion/occlusion expert.  A separate
# current-source Q/K branch carries only the immutable canonical appearance
# residual, so it cannot be diluted by the growing recent-history softmax.
# Only source-matched owner/interior queries receive its delta; unsupported
# hand, boundary, background, and unknown queries remain exactly native.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" METHOD="$METHOD" \
  CONFIDENCE_POLICY="$CONFIDENCE_POLICY" \
  PROJECTION_GATE="$PROJECTION_GATE" \
  "$SCRIPT_DIR/run_934a_single_confidence_kv.sh" \
  --paired_memory_dual_timescale_anchor \
  "$@"
