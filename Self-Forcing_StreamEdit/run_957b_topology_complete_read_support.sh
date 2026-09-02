#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# One-variable read-support ablation on top of 957a. Only background cells
# fully enclosed by the automatic transported owner can ask for KV, and their
# strength comes from source semantics plus clean-source flow affinity. The
# hand mask is exclusion-only. Writes stay exactly the 957a transactional core.
export CUDA_DEVICE="${CUDA_DEVICE:-7}"
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/957b_topology_complete_read_support}"
export OUTPUT_NAME="${OUTPUT_NAME:-957b-topology-complete-read-support.mp4}"

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=957a_token_atomic_payload_commit' \
    'change=read_only_enclosed_owner_hole_completion' \
    'external_hand_mask=enabled_exclusion_only' \
    'external_object_mask=disabled' \
    'external_source_owner_mask=disabled' \
    'topology=eight_connected_exterior_flood_fill' \
    'open_contours=no_growth' \
    'hole_read_strength=sqrt_source_semantic_x_clean_source_flow_affinity' \
    'appearance_admission=token_atomic_payload_plus_source_address' \
    'kv_write=unchanged_from_957a'
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/957b_config.txt"

bash "$SCRIPT_DIR/run_957a_token_atomic_payload_commit.sh" \
  --native_history_topology_complete_read \
  "$@"

# Post-hoc visualization only: consumes the debug maps written by inference
# and never feeds any region or mask back into generation.
bash "$SCRIPT_DIR/run_visualize_inference_edit_regions.sh" "$OUTDIR"
