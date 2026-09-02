#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Single-variable successor to 962a. It keeps the same hand-only automatic
# owner and flow trust. B0 now captures paired source/target K/V at every
# denoising timestep. Later chunks compare the frozen desired target-minus-
# source response against their current paired response at a clean-source
# flow coordinate, then inject only the bounded feedback error.
export CUDA_DEVICE="${CUDA_DEVICE:-7}"
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/963a_timestep_counterfactual_memory}"
export OUTPUT_NAME="${OUTPUT_NAME:-963a-timestep-counterfactual-memory.mp4}"
TCCM_FLOW_RADIUS="${TCCM_FLOW_RADIUS:-2.0}"
TCCM_STRENGTH="${TCCM_STRENGTH:-1.0}"
TCCM_MAX_ERROR_RATIO="${TCCM_MAX_ERROR_RATIO:-1.0}"

if [[ -n "${SOURCE_OWNER_MASK:-}" || -n "${OBJECT_MASK:-}" ]]; then
  echo "963a accepts hand information only; unset SOURCE_OWNER_MASK/OBJECT_MASK" >&2
  exit 2
fi
for argument in "$@"; do
  case "$argument" in
    --object_mask_video|--object_mask_video=*|\
    --source_owner_mask_video|--source_owner_mask_video=*)
      echo "963a forbids external object/source-owner masks: $argument" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=962a_multiframe_identity_sink' \
    'change=timestep_synchronous_closed_loop_counterfactual_memory' \
    'external_hand_mask=enabled' \
    'external_object_mask=disabled' \
    'external_source_owner_mask=disabled' \
    'identity_state=frozen_B0_paired_source_target_kv_per_timestep' \
    'identity_address=clean_source_flow_coordinate_then_local_source_key' \
    'desired_response=full_paired_B0_target_attention_minus_source_attention' \
    'current_response=full_paired_current_target_attention_minus_source_attention' \
    'feedback=bounded_desired_minus_current_response' \
    'geometry_state=mutable_clean_source_flow_coordinates' \
    'low_confidence_or_occlusion=bit_exact_native_abstention' \
    'storage=detached_cpu_compact_after_B0_commit' \
    "tccm_flow_radius=$TCCM_FLOW_RADIUS" \
    "tccm_strength=$TCCM_STRENGTH" \
    "tccm_max_error_ratio=$TCCM_MAX_ERROR_RATIO"
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/963a_config.txt"

# Override 962a's exported defaults before delegation so all inherited
# artifacts land in the independent 963a directory.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" \
  bash "$SCRIPT_DIR/run_962a_multiframe_identity_sink.sh" \
    --native_history_timestep_counterfactual_memory \
    --native_history_tccm_flow_radius "$TCCM_FLOW_RADIUS" \
    --native_history_tccm_strength "$TCCM_STRENGTH" \
    --native_history_tccm_max_error_ratio "$TCCM_MAX_ERROR_RATIO" \
    "$@"

bash "$SCRIPT_DIR/run_visualize_tccm_diagnostics.sh" "$OUTDIR"
