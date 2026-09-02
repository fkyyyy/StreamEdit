#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# 954 addresses two independent 953b failures without using object masks:
# (1) motion ownership may request target K/V on every latent frame, while
#     clean-source key correspondence remains the appearance admission gate;
# (2) high-confidence pixels outside the dilated motion owner are reconstructed
#     from the clean source to close target-prompt leakage into the background.
export CUDA_DEVICE="${CUDA_DEVICE:-7}"
export DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video1.mp4}"
export HAND_MASK="${HAND_MASK:-$REPO_ROOT/hand_mask1.mp4}"
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/954a_motion_owner_authority_closure}"
export OUTPUT_NAME="${OUTPUT_NAME:-954a-motion-owner-authority-closure.mp4}"

DEFAULT_SRC_PROMPT="A first-person egocentric indoor video. Two hands hold and operate the same vertically oriented smartphone in front of an open silver laptop on a table. The smartphone has a dark front bezel, a brown protective case, and a bright calendar interface. The right thumb taps and swipes across the touchscreen. Preserve the two hands, fingers, grasp, hand-object occlusions, camera motion, open laptop and its display, table, chairs, television, cabinet, walls, window, lighting, and background."
DEFAULT_TRG_PROMPT="A first-person egocentric indoor video. Two hands hold and handle the same vertically oriented rectangular cardboard package in front of an open silver laptop on a table. The package is a rigid matte pale-yellow paper box with an unmarked uninterrupted front surface, clean straight edges, and shallow thickness. The right thumb moves naturally across the plain front surface while both hands maintain a stable grasp around the package. Preserve the two hands, fingers, grasp, hand-object occlusions, camera motion, open silver laptop and its unchanged display, table, chairs, television, cabinet, walls, window, lighting, and background."
export SRC_PROMPT="${SRC_PROMPT:-$DEFAULT_SRC_PROMPT}"
export TRG_PROMPT="${TRG_PROMPT:-$DEFAULT_TRG_PROMPT}"
export SRC_WORD="${SRC_WORD:-smartphone}"
export TRG_WORD="${TRG_WORD:-cardboard package}"

if [[ -n "${SOURCE_OWNER_MASK:-}" || -n "${OBJECT_MASK:-}" ]]; then
  echo "954a accepts hand information only; unset SOURCE_OWNER_MASK/OBJECT_MASK" >&2
  exit 2
fi
for argument in "$@"; do
  case "$argument" in
    --object_mask_video|--object_mask_video=*|\
    --source_owner_mask_video|--source_owner_mask_video=*)
      echo "954a forbids external object/source-owner masks: $argument" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=953b_motion_geometry_owner' \
    'method=motion_owner_dense_source_addressed_read_plus_owner_complement_source_closure' \
    'external_hand_mask=enabled' \
    'external_object_mask=disabled' \
    'external_source_owner_mask=disabled' \
    'owner=hand_conditioned_bidirectional_source_rgb_flow' \
    'kv_read=all_motion_owner_latent_frames_then_clean_source_address_verification' \
    'kv_write=unchanged_uncertainty_gated_transactional_core' \
    'background=clean_source_closure_outside_one_token_owner_margin_when_preserve_confidence_ge_0.8' \
    'target_prompt=parallel_scene_description'
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/954a_config.txt"

exec bash "$SCRIPT_DIR/run_953b_motion_geometry_owner.sh" \
  --native_history_motion_owner_dense_read \
  --factorized_owner_complement_source \
  --factorized_owner_complement_margin 1 \
  --factorized_owner_complement_min_preserve_confidence 0.8 \
  "$@"
