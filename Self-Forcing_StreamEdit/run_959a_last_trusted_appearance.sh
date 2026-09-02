#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Independent successor to 958a. Geometry and editable-token inference are
# unchanged. The conservative direct-write core remains unchanged. Only the
# target-appearance transaction changes: source-addressed one-to-one matching
# carries the last trusted target-minus-source V residual, a source-regressed
# direct proposal cannot overwrite it, and the same successful KV read closes
# the competing source-appearance route. No object/source-owner mask is used.
export CUDA_DEVICE="${CUDA_DEVICE:-7}"
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/959a_last_trusted_appearance}"
export OUTPUT_NAME="${OUTPUT_NAME:-959a-last-trusted-appearance.mp4}"
RESIDUAL_UPDATE_MIN_COSINE="${RESIDUAL_UPDATE_MIN_COSINE:-0.50}"
RESIDUAL_UPDATE_MIN_MAGNITUDE_RATIO="${RESIDUAL_UPDATE_MIN_MAGNITUDE_RATIO:-0.90}"

if [[ -n "${SOURCE_OWNER_MASK:-}" || -n "${OBJECT_MASK:-}" ]]; then
  echo "959a accepts hand information only; unset SOURCE_OWNER_MASK/OBJECT_MASK" >&2
  exit 2
fi
for argument in "$@"; do
  case "$argument" in
    --object_mask_video|--object_mask_video=*|\
    --source_owner_mask_video|--source_owner_mask_video=*)
      echo "959a forbids external object/source-owner masks: $argument" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=958a_persistent_residual_upsert' \
    'change=last_trusted_appearance_transaction' \
    'external_hand_mask=enabled' \
    'external_object_mask=disabled' \
    'external_source_owner_mask=disabled' \
    'geometry_owner=unchanged_hand_conditioned_source_rgb_flow' \
    'editable_region=unchanged_from_958a' \
    'direct_write=unchanged_uncertainty_gated_core' \
    'lineage=one_to_one_top4_source_address_assignment' \
    'update=accept_only_non_regressed_target_residual' \
    'guarded_update=retain_last_trusted_target_residual' \
    'read_strength=geometric_mean_owner_x_source_address' \
    'source_appearance=close_only_after_successful_target_kv_read' \
    'failed_read=exact_native_abstention' \
    "residual_update_min_cosine=$RESIDUAL_UPDATE_MIN_COSINE" \
    "residual_update_min_magnitude_ratio=$RESIDUAL_UPDATE_MIN_MAGNITUDE_RATIO"
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/959a_config.txt"

bash "$SCRIPT_DIR/run_958a_persistent_residual_upsert.sh" \
  --native_history_last_trusted_appearance \
  --native_history_residual_update_min_cosine \
  "$RESIDUAL_UPDATE_MIN_COSINE" \
  --native_history_residual_update_min_magnitude_ratio \
  "$RESIDUAL_UPDATE_MIN_MAGNITUDE_RATIO" \
  "$@"
