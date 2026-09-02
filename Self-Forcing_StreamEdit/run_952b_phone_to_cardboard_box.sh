#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Pre-951 hand-only baseline.  The failed competitive semantic hard gate is
# disabled by run_952a_950a_baseline_new_video.sh.  No object/bottle mask or
# source-owner mask is accepted anywhere in this run.
export CUDA_DEVICE="${CUDA_DEVICE:-7}"
export DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video1.mp4}"
export HAND_MASK="${HAND_MASK:-$REPO_ROOT/hand_mask1.mp4}"
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/952b_phone_to_cardboard_box}"
export OUTPUT_NAME="${OUTPUT_NAME:-952b-phone-to-cardboard-box.mp4}"

DEFAULT_SRC_PROMPT='A first-person egocentric indoor video. Two hands hold and operate the same vertically oriented smartphone in front of an open silver laptop on a table. The smartphone has a dark front bezel, a brown protective case, and a bright calendar interface. The right thumb taps and swipes across the touchscreen. Preserve the two hands, fingers, grasp, hand-object occlusions, camera motion, open laptop and its display, table, chairs, television, cabinet, walls, window, lighting, and background.'
DEFAULT_TRG_PROMPT="A first-person egocentric indoor video identical to the source. The smartphone held between the two hands is replaced by the same small vertically oriented rectangular cardboard package throughout the entire video. It is a rigid matte pale-yellow paper box with a plain front surface, clean straight edges, shallow thickness, and no text, logo, label, pattern, screen, buttons, camera, or electronic components. The cardboard box follows exactly the source smartphone's position, scale, orientation, motion, and hand occlusions. Both hands naturally hold and touch the box. Preserve the hands, fingers, original grasp and gestures, camera motion, the open silver laptop and its display, table, chairs, television, cabinet, walls, window, lighting, and all background exactly as in the source."
export SRC_PROMPT="${SRC_PROMPT:-$DEFAULT_SRC_PROMPT}"
export TRG_PROMPT="${TRG_PROMPT:-$DEFAULT_TRG_PROMPT}"

export SRC_WORD="${SRC_WORD:-smartphone}"
export TRG_WORD="${TRG_WORD:-cardboard package}"

exec bash "$SCRIPT_DIR/run_952a_950a_baseline_new_video.sh" "$@"
