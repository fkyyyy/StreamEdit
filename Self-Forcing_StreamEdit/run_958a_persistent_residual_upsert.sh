#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# General hand-only memory fix on top of 957b.  The editable region and the
# conservative write core are unchanged.  A current direct write updates one
# token; an unwritten owner token may retain a mutually source-address-matched
# target-minus-source value residual from the previous block. Retained values
# are rebased on current clean-source V while current target K keeps geometry,
# so the memory cannot directly copy an older block's pose or scale.
export CUDA_DEVICE="${CUDA_DEVICE:-7}"
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/958a_persistent_residual_upsert}"
export OUTPUT_NAME="${OUTPUT_NAME:-958a-persistent-residual-upsert.mp4}"

if [[ -n "${SOURCE_OWNER_MASK:-}" || -n "${OBJECT_MASK:-}" ]]; then
  echo "958a accepts hand information only; unset SOURCE_OWNER_MASK/OBJECT_MASK" >&2
  exit 2
fi
for argument in "$@"; do
  case "$argument" in
    --object_mask_video|--object_mask_video=*|\
    --source_owner_mask_video|--source_owner_mask_video=*)
      echo "958a forbids external object/source-owner masks: $argument" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=957b_topology_complete_read_support' \
    'change=source_addressed_token_persistent_residual_upsert' \
    'external_hand_mask=enabled' \
    'external_object_mask=disabled' \
    'external_source_owner_mask=disabled' \
    'editable_region=unchanged_from_957b' \
    'direct_write=existing_uncertainty_gated_transactional_core' \
    'retention=automatic_motion_owner_x_mutual_source_key_match' \
    'payload=target_minus_source_value_residual_rebased_on_current_source' \
    'transport=one_to_one_mutual_nearest_neighbour' \
    'retained_payload_is_not_a_new_write=true' \
    'failed_match=exact_native_abstention'
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/958a_config.txt"

bash "$SCRIPT_DIR/run_957b_topology_complete_read_support.sh" \
  --native_history_persistent_residual_upsert \
  "$@"
