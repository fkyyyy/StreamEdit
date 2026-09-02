#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/937b_source_topology_prompt}"
OUTPUT_NAME="${OUTPUT_NAME:-937b-source-topology-prompt.mp4}"

# The source cap is geometrically flat.  State that topology explicitly so the
# ignition block does not invent a tiny relief that a coarse video-latent token
# cannot represent reliably in later views.  This is a prompt-only ablation:
# role routing, native KV admission/read, seed, and all sampling controls remain
# identical to 937a.
TRG_PROMPT="${TRG_PROMPT:-A first-person egocentric kitchen video identical to the source. Only the exterior color of the same cylindrical plastic seasoning bottle is changed: its body is matte violet and its screw cap is dark violet. Preserve the exact source bottle topology and cylindrical proportions, including the cap seam; the cap top remains flat and smooth. Do not introduce any new protrusion, indentation, nozzle, button, ornament, marking, or relief. Preserve the pose, scale, motion, hand anatomy, grasp, and occlusions. The cooking pan, reddish-brown food, utensils, countertop, stovetop, drawers, lighting, and all background retain exactly their source appearance and colors.}"

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=937a_role_fixed_native_kv' \
    'ablation=source_topology_constrained_target_prompt' \
    'kv_change=none' \
    'reason=avoid_ignition_only_subtoken_geometry_hallucination' \
    "target_prompt=$TRG_PROMPT"
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/937b_config.txt"

OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" TRG_PROMPT="$TRG_PROMPT" \
  "$SCRIPT_DIR/run_937a_role_fixed_native_kv.sh" \
  "$@"
