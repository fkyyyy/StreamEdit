#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/933a_source_transported_counterfactual_kv}"
OUTPUT_NAME="${OUTPUT_NAME:-933a-source-transported-counterfactual-kv.mp4}"
METHOD="${METHOD:-source_transported_counterfactual_kv}"
CONFIDENCE_POLICY="${CONFIDENCE_POLICY:-legacy_soft_projection_and_arbitration}"
PROJECTION_GATE="${PROJECTION_GATE:-strict_object_interior}"

# Controlled successor to 932a/931b. The prompt specifies the intended edit,
# while the mechanism itself remains object- and color-agnostic. The white
# body makes source-blue leakage and cap-to-body leakage directly visible.
TRG_PROMPT="${TRG_PROMPT:-A first-person egocentric kitchen video identical to the source. The same cylindrical plastic seasoning bottle keeps a plain matte white body without patterns, markings, or color tint, while only its flat screw cap is changed to a uniform dark violet. Preserve a sharp stable seam between the white bottle body and violet cap, and preserve the exact bottle proportions, cap shape, pose, scale, motion, hand anatomy, grasp, and occlusions. The cooking pan, reddish-brown food, utensils, countertop, stovetop, drawers, lighting, and all background retain exactly their source appearance and colors.}"
mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    "method=$METHOD" \
    "confidence_policy=$CONFIDENCE_POLICY" \
    "projection_gate=$PROJECTION_GATE" \
    'address=adjacent_clean_source_kv' \
    'payload=immutable_canonical_target_minus_source_residual' \
    'target_observation=transactional_critic_only' \
    "target_prompt=$TRG_PROMPT" \
    'transport_min_similarity=0.10' \
    'transport_coordinate_radius=0.60' \
    'transport_cycle_radius=0.20' \
    'transport_min_confidence=0.05'
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/config.txt"

# 933 adds a source-only moving address frontier to the strict immutable
# canonical bank. Adjacent clean-source K/V and object-relative coordinates
# transport an already accepted residual lineage across pose changes. The
# generated target is a critic for transactional promotion only and can never
# replace the inherited payload. Unsupported roles use exact native fallback.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" TRG_PROMPT="$TRG_PROMPT" \
  "$SCRIPT_DIR/run_931a_source_part_consistent_memory.sh" \
  --paired_memory_source_transport \
  --paired_memory_transport_min_similarity 0.10 \
  --paired_memory_transport_coordinate_radius 0.60 \
  --paired_memory_transport_cycle_radius 0.20 \
  --paired_memory_transport_min_confidence 0.05 \
  "$@"
