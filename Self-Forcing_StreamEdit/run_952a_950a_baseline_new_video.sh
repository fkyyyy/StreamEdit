#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# A clean new-video entry point for the pre-951 baseline.  All case-specific
# inputs are required explicitly so a new experiment cannot silently reuse the
# old bottle video, hand mask, or prompt.
: "${DATA_PATH:?Set DATA_PATH to the new source video}"
: "${HAND_MASK:?Set HAND_MASK to the matching hand-mask video}"
: "${SRC_PROMPT:?Set SRC_PROMPT for the new source video}"
: "${TRG_PROMPT:?Set TRG_PROMPT for the requested edit}"
: "${SRC_WORD:?Set SRC_WORD to the source object phrase}"
: "${TRG_WORD:?Set TRG_WORD to the target/edit phrase}"

OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/952a_950a_baseline_new_video}"
OUTPUT_NAME="${OUTPUT_NAME:-952a-950a-baseline-new-video.mp4}"

for required_path in "$DATA_PATH" "$HAND_MASK"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

# Preserve the deployable hand-only contract and keep 951's failed semantic
# hard gate out of this control run.  Bottle/object GT remains evaluation-only.
if [[ -n "${SOURCE_OWNER_MASK:-}" || -n "${OBJECT_MASK:-}" ]]; then
  echo "952a accepts hand information only; unset SOURCE_OWNER_MASK/OBJECT_MASK" >&2
  exit 2
fi
for argument in "$@"; do
  case "$argument" in
    --target_semantic_*|\
    --object_mask_video|--object_mask_video=*|\
    --source_owner_mask_video|--source_owner_mask_video=*)
      echo "952a forbids semantic hard gating and external object masks: $argument" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUTDIR"
printf '%s\n' \
  'baseline=950a_high_recall_hand_roles' \
  'target_semantic_competition=disabled' \
  'external_object_mask=disabled' \
  'external_source_owner_mask=disabled' \
  'external_hand_mask=enabled' \
  'purpose=new_video_case_difficulty_control' \
  "data_path=$DATA_PATH" \
  "hand_mask=$HAND_MASK" \
  "src_word=$SRC_WORD" \
  "trg_word=$TRG_WORD" \
  > "$OUTDIR/952a_config.txt"

DATA_PATH="$DATA_PATH" \
HAND_MASK="$HAND_MASK" \
SRC_PROMPT="$SRC_PROMPT" \
TRG_PROMPT="$TRG_PROMPT" \
SRC_WORD="$SRC_WORD" \
TRG_WORD="$TRG_WORD" \
OUTDIR="$OUTDIR" \
OUTPUT_NAME="$OUTPUT_NAME" \
exec bash "$SCRIPT_DIR/run_950a_high_recall_hand_roles.sh" "$@"
