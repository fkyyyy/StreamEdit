#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Single-variable successor to 961a. Editable-region inference, prompts, the
# automatic hand/flow owner, and the conservative write transaction remain
# unchanged. The only algorithmic change is the read operator: identity comes
# from the immutable three-frame ignition canonical. Clean-source keys choose
# a bounded candidate set in every frame, then current target Q selects only
# among those candidates. Recent flow state supplies local transport trust but
# never overwrites or replaces the frozen identity payload.
export CUDA_DEVICE="${CUDA_DEVICE:-7}"
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/962a_multiframe_identity_sink}"
export OUTPUT_NAME="${OUTPUT_NAME:-962a-multiframe-identity-sink.mp4}"
SINK_TOPK_PER_FRAME="${SINK_TOPK_PER_FRAME:-8}"
SINK_SOURCE_LOGIT_BIAS="${SINK_SOURCE_LOGIT_BIAS:-1.0}"
SINK_STRENGTH="${SINK_STRENGTH:-1.0}"

if [[ -n "${SOURCE_OWNER_MASK:-}" || -n "${OBJECT_MASK:-}" ]]; then
  echo "962a accepts hand information only; unset SOURCE_OWNER_MASK/OBJECT_MASK" >&2
  exit 2
fi
for argument in "$@"; do
  case "$argument" in
    --object_mask_video|--object_mask_video=*|\
    --source_owner_mask_video|--source_owner_mask_video=*)
      echo "962a forbids external object/source-owner masks: $argument" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=961a_decoupled_flow_trust' \
    'change=frozen_query_aware_multiframe_identity_sink' \
    'external_hand_mask=enabled' \
    'external_object_mask=disabled' \
    'external_source_owner_mask=disabled' \
    'editable_region=unchanged_from_961a' \
    'identity_payload=immutable_ignition_target_minus_source_value_residual' \
    'identity_address=per_frame_clean_source_key_topk' \
    'identity_selection=current_target_query_inside_source_candidates_only' \
    'frame_diversity=equal_candidate_budget_per_ignition_frame' \
    'recent_flow_role=local_transport_trust_only' \
    'write=unchanged_uncertainty_gated_transaction' \
    'low_confidence_or_occlusion=exact_native_abstention' \
    "sink_topk_per_frame=$SINK_TOPK_PER_FRAME" \
    "sink_source_logit_bias=$SINK_SOURCE_LOGIT_BIAS" \
    "sink_strength=$SINK_STRENGTH"
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/962a_config.txt"

bash "$SCRIPT_DIR/run_961a_decoupled_flow_trust.sh" \
  --native_history_multiframe_identity_sink \
  --native_history_multiframe_sink_topk_per_frame \
  "$SINK_TOPK_PER_FRAME" \
  --native_history_multiframe_sink_source_logit_bias \
  "$SINK_SOURCE_LOGIT_BIAS" \
  --native_history_multiframe_sink_strength "$SINK_STRENGTH" \
  "$@"
