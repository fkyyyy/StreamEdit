#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/937c_minimal_flat_cap_prompt}"
OUTPUT_NAME="${OUTPUT_NAME:-937c-minimal-flat-cap-prompt.mp4}"

# Keep the proven 937a wording and change only the cap-shape adjective.  The
# longer 937b topology/negative list shifted the first-block body appearance,
# which the immutable canonical tier then correctly preserved.
TRG_PROMPT="${TRG_PROMPT:-A first-person egocentric kitchen video identical to the source. Only the exterior color of the same cylindrical plastic seasoning bottle is changed: its body is matte violet and its flat-topped screw cap is dark violet. Preserve the exact cylindrical proportions, cap seam, label relief, pose, scale, motion, hand anatomy, grasp, and occlusions. The cooking pan, reddish-brown food, utensils, countertop, stovetop, drawers, lighting, and all background retain exactly their source appearance and colors.}"

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=937a_role_fixed_native_kv' \
    'ablation=minimal_flat_cap_lexical_constraint' \
    'kv_change=none' \
    'removed=937b_topology_and_negative_attribute_list' \
    "target_prompt=$TRG_PROMPT"
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/937c_config.txt"

OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" TRG_PROMPT="$TRG_PROMPT" \
  "$SCRIPT_DIR/run_937a_role_fixed_native_kv.sh" \
  "$@"
