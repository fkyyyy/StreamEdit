#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/950a_high_recall_hand_roles}"
export OUTPUT_NAME="${OUTPUT_NAME:-950a-high-recall-hand-roles.mp4}"
HAND_PERSISTENT_OCCUPANCY="${HAND_PERSISTENT_OCCUPANCY:-1.0}"
CONNECTED_GROWTH_STEPS="${CONNECTED_GROWTH_STEPS:-3}"
CONNECTED_CANDIDATE_RATIO="${CONNECTED_CANDIDATE_RATIO:-0.75}"

mkdir -p "$OUTDIR"
printf '%s\n' \
  'baseline=949b_recent_edited_entry_bridge' \
  'method=causal_hand_evidence_plus_connected_hysteresis' \
  'external_object_mask=disabled' \
  'external_source_owner_mask=disabled' \
  'external_hand_mask=enabled' \
  'gt_mask=inference_disabled_offline_evaluation_only' \
  'hand_union=proximity_only' \
  'hand_occupancy=soft_contact_only' \
  'hand_persistent=hard_owner_exclusion_only' \
  'region_growth=high_confidence_seed_through_low_threshold_connected_corridor' \
  "hand_persistent_occupancy=$HAND_PERSISTENT_OCCUPANCY" \
  "connected_growth_steps=$CONNECTED_GROWTH_STEPS" \
  "connected_candidate_ratio=$CONNECTED_CANDIDATE_RATIO" \
  > "$OUTDIR/950a_config.txt"

bash "$SCRIPT_DIR/run_949b_recent_edited_entry_bridge.sh" \
  --hand_causal_evidence \
  --hand_persistent_occupancy "$HAND_PERSISTENT_OCCUPANCY" \
  --hand_connected_hysteresis \
  --hand_connected_growth_steps "$CONNECTED_GROWTH_STEPS" \
  --hand_connected_candidate_ratio "$CONNECTED_CANDIDATE_RATIO" \
  "$@"
