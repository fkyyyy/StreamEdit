#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Independent successor to 959a. It keeps the hand-only input contract and
# 959a prompts/settings. Clean-source RAFT now has two explicit roles:
# (1) camera-compensated evidence is fused into the automatic token roles;
# (2) last-trusted target-minus-source V residuals are transported directly
#     to current source coordinates, with no target-K identity addressing.
export CUDA_DEVICE="${CUDA_DEVICE:-7}"
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/960a_flow_indexed_identity}"
export OUTPUT_NAME="${OUTPUT_NAME:-960a-flow-indexed-identity.mp4}"
FLOW_ROLE_WEIGHT="${FLOW_ROLE_WEIGHT:-0.75}"
FLOW_MIN_CONFIDENCE="${FLOW_MIN_CONFIDENCE:-0.10}"

if [[ -n "${SOURCE_OWNER_MASK:-}" || -n "${OBJECT_MASK:-}" ]]; then
  echo "960a accepts hand information only; unset SOURCE_OWNER_MASK/OBJECT_MASK" >&2
  exit 2
fi
for argument in "$@"; do
  case "$argument" in
    --object_mask_video|--object_mask_video=*|\
    --source_owner_mask_video|--source_owner_mask_video=*)
      echo "960a forbids external object/source-owner masks: $argument" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=959a_last_trusted_appearance' \
    'external_hand_mask=enabled' \
    'external_object_mask=disabled' \
    'external_source_owner_mask=disabled' \
    'role_evidence=hand_conditioned_counterfactual_response_plus_camera_compensated_source_flow' \
    'flow_magnitude_alone_is_not_object_evidence=true' \
    'flow_role_fusion=positive_owner_recovery_without_background_cropping' \
    'identity_address=clean_source_bidirectional_flow_coordinate' \
    'identity_payload=last_trusted_target_minus_source_value_residual' \
    'target_key_identity_addressing=disabled' \
    'write=unchanged_uncertainty_gated_transaction' \
    'retained_confidence=source_flow_transport_not_soft_write_proposal' \
    'low_confidence_or_occlusion=exact_native_abstention' \
    "flow_role_weight=$FLOW_ROLE_WEIGHT" \
    "flow_min_confidence=$FLOW_MIN_CONFIDENCE"
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/960a_config.txt"

bash "$SCRIPT_DIR/run_959a_last_trusted_appearance.sh" \
  --source_flow_role_fusion \
  --source_flow_role_weight "$FLOW_ROLE_WEIGHT" \
  --native_history_flow_indexed_residual \
  --native_history_flow_min_confidence "$FLOW_MIN_CONFIDENCE" \
  "$@"
