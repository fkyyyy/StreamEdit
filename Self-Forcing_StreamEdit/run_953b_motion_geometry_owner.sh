#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

export CUDA_DEVICE="${CUDA_DEVICE:-7}"
export DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video1.mp4}"
export HAND_MASK="${HAND_MASK:-$REPO_ROOT/hand_mask1.mp4}"
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/953b_motion_geometry_owner}"
export OUTPUT_NAME="${OUTPUT_NAME:-953b-motion-geometry-owner.mp4}"
FLOW_CACHE="${FLOW_CACHE:-$SCRIPT_DIR/outputs/952b_phone_to_cardboard_box/source_flow/raft_large_bidirectional_flow.pt}"

if [[ -n "${SOURCE_OWNER_MASK:-}" || -n "${OBJECT_MASK:-}" ]]; then
  echo "953b accepts hand information only; unset SOURCE_OWNER_MASK/OBJECT_MASK" >&2
  exit 2
fi
for argument in "$@"; do
  case "$argument" in
    --object_mask_video|--object_mask_video=*|\
    --source_owner_mask_video|--source_owner_mask_video=*)
      echo "953b forbids external object/source-owner masks: $argument" >&2
      exit 2
      ;;
  esac
done
for required_path in "$DATA_PATH" "$HAND_MASK" "$FLOW_CACHE"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    echo "Precompute flow with: VIDEO=$DATA_PATH OUTPUT_DIR=$(dirname -- "$FLOW_CACHE") bash $SCRIPT_DIR/run_953a_precompute_source_flow.sh" >&2
    exit 2
  fi
done

mkdir -p "$OUTDIR"
printf '%s\n' \
  'baseline=952b_phone_to_cardboard_box' \
  'method=hand_conditioned_causal_motion_geometry_owner' \
  'external_hand_mask=enabled' \
  'external_object_mask=disabled' \
  'external_source_owner_mask=disabled' \
  'motion=bidirectional_raft_on_clean_source_rgb' \
  'geometry_state=full_soft_owner_flow_transport' \
  'appearance_state=uncertainty_gated_transactional_kv_write_core' \
  'geometry_write_feedback=disabled' \
  'diffusion_velocity_role=counterfactual_edit_response_only' \
  "source_flow_cache=$FLOW_CACHE" \
  > "$OUTDIR/953b_config.txt"

exec bash "$SCRIPT_DIR/run_952b_phone_to_cardboard_box.sh" \
  --motion_geometry_owner \
  --source_flow_cache "$FLOW_CACHE" \
  "$@"
