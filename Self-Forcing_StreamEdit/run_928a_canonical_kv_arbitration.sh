#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/928a_canonical_kv_arbitration}"
OUTPUT_NAME="${OUTPUT_NAME:-928a-canonical-kv-arbitration.mp4}"

# Controlled follow-up to 927a. The sparse paired bank and retrieval are
# unchanged. A successful read now (1) materializes the canonical edit in
# current-source value coordinates before attention and in the persistent
# target KV, and (2) softly arbitrates the competing source residual.
# Unsupported hand/background/unknown tokens remain exact 927a fallbacks.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" \
  "$SCRIPT_DIR/run_927a_asymmetric_paired_memory.sh" \
  --paired_memory_value_projection \
  --paired_memory_read_strength 0.75 \
  --paired_memory_source_suppression 0.75 \
  "$@"
