#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/951a_competitive_semantic_authority}"
export OUTPUT_NAME="${OUTPUT_NAME:-951a-competitive-semantic-authority.mp4}"
SEMANTIC_MARGIN="${SEMANTIC_MARGIN:-0.10}"
SEMANTIC_MIN_CONFIDENCE="${SEMANTIC_MIN_CONFIDENCE:-0.20}"

# Keep the prompt literal and part-local.  In particular, the bottle body is
# explicitly unchanged instead of being described as violet.  Every phrase
# below occurs verbatim so semantic grounding fails loudly rather than being
# silently disabled by tokenizer mismatch.
export SRC_PROMPT="${SRC_PROMPT:-A first-person egocentric kitchen video. A hand holds and moves the same cylindrical plastic seasoning bottle with a white bottle body and a blue screw cap. The cooking pan, food, countertop, stovetop, drawers, hand, lighting, and background retain their source appearance.}"
export TRG_PROMPT="${TRG_PROMPT:-A first-person egocentric kitchen video. A hand holds and moves the same cylindrical plastic seasoning bottle with an unchanged white bottle body and a dark violet screw cap. Only the screw cap color changes. The cooking pan, food, countertop, stovetop, drawers, hand, lighting, and background retain their source appearance.}"
export SRC_WORD="${SRC_WORD:-seasoning bottle}"
export TRG_WORD="${TRG_WORD:-screw cap}"

mkdir -p "$OUTDIR"
printf '%s\n' \
  'baseline=950a_high_recall_hand_roles' \
  'method=competitive_target_semantic_authority' \
  'external_object_mask=disabled' \
  'external_source_owner_mask=disabled' \
  'external_hand_mask=enabled' \
  'semantic_probe=clean_source_latent_with_target_prompt' \
  'ownership=whole_automatic_hand_flow_object_for_geometry' \
  'edit_authority=edit_phrase_minus_preserve_phrase_inside_owner' \
  'target_kv_query=local_edit_authority_only' \
  'target_kv_write=local_edit_authority_only' \
  'target_velocity=local_edit_authority_only' \
  'preserve_velocity=clean_source_reconstruction' \
  'edit_phrases=dark violet screw cap|screw cap|cap color' \
  'preserve_phrases=white bottle body|cooking pan|food|countertop|stovetop|drawers|hand|lighting|background' \
  "semantic_margin=$SEMANTIC_MARGIN" \
  "semantic_min_confidence=$SEMANTIC_MIN_CONFIDENCE" \
  > "$OUTDIR/951a_config.txt"

exec bash "$SCRIPT_DIR/run_950a_high_recall_hand_roles.sh" \
  --target_semantic_competition \
  --target_edit_phrases \
    "dark violet screw cap" \
    "screw cap" \
    "cap color" \
  --target_preserve_phrases \
    "white bottle body" \
    "cooking pan" \
    "food" \
    "countertop" \
    "stovetop" \
    "drawers" \
    "hand" \
    "lighting" \
    "background" \
  --target_semantic_margin "$SEMANTIC_MARGIN" \
  --target_semantic_min_confidence "$SEMANTIC_MIN_CONFIDENCE" \
  "$@"
