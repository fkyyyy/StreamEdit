#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# One-variable KV ablation on the 954p prompt and 954a spatial-routing
# baseline.  Clean-source K remains the motion/address signal.  Mutable recent
# target V is trusted only when its target-minus-source residual agrees with
# the source-aligned immutable first-chunk residual.  No object/source-owner
# mask is accepted by the inherited hand-only input contract.
export CUDA_DEVICE="${CUDA_DEVICE:-7}"
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/956a_dual_evidence_kv_arbitration}"
export OUTPUT_NAME="${OUTPUT_NAME:-956a-dual-evidence-kv-arbitration.mp4}"
MIN_PAYLOAD_CONSISTENCY="${MIN_PAYLOAD_CONSISTENCY:-0.15}"

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=954p_streamgve_prompt_ablation' \
    'change=kv_read_arbitration_only' \
    'prompt=unchanged_streamgve_minimal_difference' \
    'spatial_routing=unchanged_from_954a' \
    'external_hand_mask=enabled' \
    'external_object_mask=disabled' \
    'external_source_owner_mask=disabled' \
    'address_evidence=clean_source_pre_rope_key' \
    'payload_evidence=recent_residual_vs_immutable_canonical_residual' \
    'accepted_recent=convex_canonical_to_recent_output' \
    'rejected_recent=immutable_canonical_fallback' \
    'kv_key_sparsification=disabled' \
    "min_payload_consistency=$MIN_PAYLOAD_CONSISTENCY"
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/956a_config.txt"

exec bash "$SCRIPT_DIR/run_954p_streamgve_prompt_ablation.sh" \
  --native_history_dual_evidence_arbitration \
  --native_history_min_payload_consistency "$MIN_PAYLOAD_CONSISTENCY" \
  "$@"
