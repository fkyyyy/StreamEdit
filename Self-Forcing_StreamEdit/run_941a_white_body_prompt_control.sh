#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/941a_white_body_prompt_control}"
OUTPUT_NAME="${OUTPUT_NAME:-941a-white-body-prompt-control.mp4}"
TRG_PROMPT="${TRG_PROMPT:-A first-person egocentric kitchen video identical to the source. Only the flat screw cap of the same cylindrical plastic seasoning bottle is changed to uniform dark violet. The bottle body remains exactly its original matte white color, including its original printed label markings and relief, without violet tint or newly generated patterns. Preserve a sharp stable seam between the white body and violet cap, and preserve the exact bottle proportions, cap shape, pose, scale, motion, hand anatomy, grasp, and occlusions. The cooking pan, reddish-brown food, utensils, countertop, stovetop, drawers, lighting, and all background retain exactly their source appearance and colors.}"

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=940a_bootstrap_rope_alias_fix' \
    'ablation=white_body_violet_cap_prompt_control' \
    'kv_change=none' \
    "target_prompt=$TRG_PROMPT"
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/941a_config.txt"

OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" TRG_PROMPT="$TRG_PROMPT" \
  "$SCRIPT_DIR/run_940a_bootstrap_rope_alias_fix.sh" \
  "$@"
