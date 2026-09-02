#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/936a_immutable_canonical_key_anchor}"
OUTPUT_NAME="${OUTPUT_NAME:-936a-immutable-canonical-key-anchor.mp4}"
METHOD="${METHOD:-immutable_source_key_target_residual_dual_timescale_kv}"
CONFIDENCE_POLICY="${CONFIDENCE_POLICY:-evidence_logit_prior_binary_query_admission}"
PROJECTION_GATE="${PROJECTION_GATE:-strict_object_interior}"
mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    "method=$METHOD" \
    "confidence_policy=$CONFIDENCE_POLICY" \
    "projection_gate=$PROJECTION_GATE" \
    'canonical_address=replay_verified_first_block_pre_rope_source_k' \
    'canonical_payload=immutable_first_block_target_minus_source_delta_v' \
    'motion_memory=native_dense_target_history' \
    'write_policy=target_critic_filters_evidence_only' \
    'read_policy=source_transport_lineage_plus_local_part_neighborhood'
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/936_config.txt"

# 936 fixes the structural flaw in 935 instead of changing thresholds.  The
# proposal pass freezes first-block pre-RoPE clean-source K and target-minus-
# source delta-V in a separate ignition bank.  Replay can only invalidate
# lineages.  Later blocks transport lineage/admission through clean-source
# observations, then cross-attend current pre-RoPE source Q directly to the
# frozen bank.  Native target history remains the motion/occlusion path and
# unsupported hand, boundary, background, and unknown queries remain exact
# native fallback.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" METHOD="$METHOD" \
  CONFIDENCE_POLICY="$CONFIDENCE_POLICY" \
  PROJECTION_GATE="$PROJECTION_GATE" \
  "$SCRIPT_DIR/run_935a_dual_timescale_anchor.sh" \
  --paired_memory_canonical_key_anchor \
  "$@"
