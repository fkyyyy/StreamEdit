#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# One-variable successor to 960a. Editable-region inference, prompts, source
# flow, and the uncertainty-gated KV write transaction remain unchanged.
# Only confidence semantics change: durable appearance trust is transported as
# a payload attribute, while flow reliability is local to the current block.
# Their product gates the current read; committing a retained payload resets
# local transport instead of recursively weakening appearance trust.
export CUDA_DEVICE="${CUDA_DEVICE:-7}"
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/961a_decoupled_flow_trust}"
export OUTPUT_NAME="${OUTPUT_NAME:-961a-decoupled-flow-trust.mp4}"

if [[ -n "${SOURCE_OWNER_MASK:-}" || -n "${OBJECT_MASK:-}" ]]; then
  echo "961a accepts hand information only; unset SOURCE_OWNER_MASK/OBJECT_MASK" >&2
  exit 2
fi
for argument in "$@"; do
  case "$argument" in
    --object_mask_video|--object_mask_video=*|\
    --source_owner_mask_video|--source_owner_mask_video=*)
      echo "961a forbids external object/source-owner masks: $argument" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=960a_flow_indexed_identity' \
    'change=decoupled_persistent_appearance_and_local_transport_trust' \
    'external_hand_mask=enabled' \
    'external_object_mask=disabled' \
    'external_source_owner_mask=disabled' \
    'editable_region=unchanged_from_960a' \
    'kv_payload=source_flow_indexed_target_minus_source_value_residual' \
    'appearance_trust=last_verified_payload_non_accumulating' \
    'transport_reliability=current_block_local_path' \
    'effective_read_confidence=appearance_trust_times_local_transport' \
    'transported_attributes=support_normalized_no_boundary_amplitude_decay' \
    'commit=local_transport_reset_at_committed_source_coordinate' \
    'low_confidence_or_occlusion=exact_native_abstention'
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/961a_config.txt"

bash "$SCRIPT_DIR/run_960a_flow_indexed_identity.sh" \
  --native_history_decoupled_flow_trust \
  "$@"

# Read-only post-hoc diagnostics. No visualization map is fed to inference.
bash "$SCRIPT_DIR/run_visualize_inference_edit_regions.sh" "$OUTDIR"
