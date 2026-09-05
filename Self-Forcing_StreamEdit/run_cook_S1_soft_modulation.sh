#!/usr/bin/env bash
set -euo pipefail

# Cook S1: Hand-conditioned role inference + soft region modulation.
#
# Novelty 1 demonstration: hand mask → automatic token role discovery
# → soft spatial modulation on native StreamGVE velocity blend.
#
# Role inference pipeline is fully enabled (HandRoleInferencer with
# velocity field refinement), but the downstream intervention is SOFT:
# the role posterior modulates the native SOG blend strength instead of
# hard-switching velocity routing or KV metadata.
#
# NO factorized owner source block.
# NO causal owner consistent KV metadata.
# NO orthogonal geometry injection.
# NO optical flow (velocity-only owner transport).
# NO M1/M2 delta-V bank.
# NO projected residual / source bg suppression.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/cook.mp4}"
HAND_MASK="${HAND_MASK:-$REPO_ROOT/cook_handmask.mp4}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/cook_S1_soft_modulation}"
OUTPUT_NAME="${OUTPUT_NAME:-cook-S1-soft-modulation.mp4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-7}"
DRY_RUN="${DRY_RUN:-0}"
STEP="${STEP:-15}"
BLEND_STRENGTH="${BLEND_STRENGTH:-0.5}"

readonly SRC_PROMPT='First-person POV shot from above, wide-angle lens. A person is standing at a kitchen counter, cooking on an induction stovetop. The right hand holds a metal spatula, actively stirring and flipping diced ingredients in a large dark non-stick frying pan. The left hand grips the pan handle to steady it. Diced potatoes, onions, and small meat cubes are being stir-fried in the pan. A bowl of beaten eggs sits to the lower left. The granite countertop is cluttered with wine bottles, a stainless steel kettle, a white colander, glass jars, condiment bottles, and a small yellow cup. A second dark pan sits on the adjacent burner. Warm indoor lighting, realistic 4k video style, slight overhead fish-eye effect.'

readonly TRG_PROMPT='First-person POV shot from above, wide-angle lens. A person is standing at a kitchen counter, cooking on an induction stovetop. The right hand holds a wooden spatula with a flat, wide paddle head made of smooth light-colored natural wood with visible grain, actively stirring and flipping diced ingredients in a large dark non-stick frying pan. The left hand grips the pan handle to steady it. Diced potatoes, onions, and small meat cubes are being stir-fried in the pan. A bowl of beaten eggs sits to the lower left. The granite countertop is cluttered with wine bottles, a stainless steel kettle, a white colander, glass jars, condiment bottles, and a small yellow cup. A second dark pan sits on the adjacent burner. Warm indoor lighting, realistic 4k video style, slight overhead fish-eye effect.'

readonly SRC_WORD='metal spatula'
readonly TRG_WORD='wooden spatula'

if [[ ! -f "$DATA_PATH" ]]; then
  echo "Missing: $DATA_PATH" >&2
  exit 2
fi
if [[ ! -f "$HAND_MASK" ]]; then
  echo "Missing: $HAND_MASK" >&2
  exit 2
fi

mkdir -p "$OUTDIR"

COMMAND=(
  "$PYTHON_BIN" "$SCRIPT_DIR/inference_edit_streamedit.py"
  --data_path "$DATA_PATH"
  --hand_mask_video "$HAND_MASK"
  --save_path "$OUTDIR/$OUTPUT_NAME"
  --save_role_dir "$OUTDIR/roles"

  # Hand role inference (Novelty 1 core)
  --routing_mode hand_role_factorized_causal_owner_kv
  --contact_graph_mode no_graph
  --hand_query_layers 8 12 16 20
  --hand_field_update_mode posterior
  --mask_white_threshold 245
  --hand_mask_mode overlay_white
  --hand_mask_overlay_diff_threshold 24
  --hand_causal_evidence
  --hand_persistent_occupancy 1.0
  --hand_connected_hysteresis
  --hand_connected_growth_steps 3
  --hand_connected_candidate_ratio 0.75

  # SOFT region modulation — the key difference from F1/F1V
  --soft_region_modulation
  --soft_region_blend_strength "$BLEND_STRENGTH"

  # Native target history (needed for clean target KV context)
  --factorized_native_target_history

  # Everything aggressive is OFF:
  # NO --factorized_owner_source_block
  # NO --causal_owner_consistent_kv_metadata
  # NO --factorized_orthogonal_geometry
  # NO --source_flow_cache (no optical flow)
  # NO --motion_geometry_owner
  # NO --factorized_owner_complement_source
  # NO --immutable_delta_v_bank
  # NO --projected_source_residual
  # NO --suppress_source_bg_value

  --src_prompt "$SRC_PROMPT"
  --trg_prompt "$TRG_PROMPT"
  --src_word "$SRC_WORD"
  --trg_word "$TRG_WORD"
  --fg_boost_factor 4
  --blend_power 2
  --step "$STEP"
  --seed 0
  --rollout_chunk_size 21
  --rollout_overlap_block_num 1
  "$@"
)

{
  printf '%s\n' \
    'experiment=cook_S1_soft_modulation' \
    'edit=metal_spatula_to_wooden_spatula' \
    'baseline=cook_L0_baseline' \
    'novelty1=hand_conditioned_role_inference_plus_soft_modulation' \
    'role_inference=hand_mask_plus_velocity_field_posterior' \
    'region_application=soft_spatial_modulation_on_native_sog' \
    "soft_region_blend_strength=$BLEND_STRENGTH" \
    'routing=hand_role_factorized_causal_owner_kv' \
    'native_target_history=enabled' \
    'factorized_owner_source_block=disabled' \
    'causal_owner_consistent_kv_metadata=disabled' \
    'factorized_orthogonal_geometry=disabled' \
    'optical_flow=disabled' \
    'owner_complement_source=disabled' \
    'M1_M2=disabled' \
    'projected_residual=disabled' \
    'source_bg_suppression=disabled' \
    'external_hand_mask=enabled' \
    'external_object_mask=disabled' \
    "data_path=$DATA_PATH" \
    "hand_mask=$HAND_MASK"
  printf 'command='
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
} > "$OUTDIR/cook_S1_config.txt"

echo "Cook S1: hand role inference + soft region modulation (blend=$BLEND_STRENGTH)"
echo "OUTPUT $OUTDIR/$OUTPUT_NAME"

if [[ "$DRY_RUN" == 1 ]]; then
  echo 'DRY_RUN resolved command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${COMMAND[@]}" 2>&1 | tee "$OUTDIR/run.log"
