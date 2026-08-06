#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="${1:-}"
if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: bash run_907_contact_graph.sh <config.env>" >&2
  exit 2
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  CONFIG_PATH="$SCRIPT_DIR/$CONFIG_PATH"
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$CONFIG_PATH"
: "${EXPERIMENT_NAME:?Missing EXPERIMENT_NAME in config}"
: "${CONTACT_GRAPH_MODE:?Missing CONTACT_GRAPH_MODE in config}"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video.mp4}"
OBJECT_MASK="${OBJECT_MASK:-$REPO_ROOT/bottle_mask.mp4}"
HAND_MASK="${HAND_MASK:-$REPO_ROOT/hand_mask.mp4}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/907_contact_graph/$EXPERIMENT_NAME}"
STEP="${STEP:-15}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
mkdir -p "$OUTDIR/roles"
cd "$SCRIPT_DIR"

echo "CONTACT_GRAPH_EXPERIMENT name=$EXPERIMENT_NAME mode=$CONTACT_GRAPH_MODE"
python inference_edit_streamedit.py \
  --data_path "$DATA_PATH" \
  --object_mask_video "$OBJECT_MASK" \
  --hand_mask_video "$HAND_MASK" \
  --save_path "$OUTDIR/907-$EXPERIMENT_NAME.mp4" \
  --save_role_dir "$OUTDIR/roles" \
  --routing_mode oracle_role_residual_kv \
  --role_boundary_radius 1 \
  --contact_target_weight 1.0 \
  --contact_graph_mode "$CONTACT_GRAPH_MODE" \
  --contact_graph_topk "$CONTACT_GRAPH_TOPK" \
  --contact_graph_radius "$CONTACT_GRAPH_RADIUS" \
  --contact_graph_min_confidence "$CONTACT_GRAPH_MIN_CONFIDENCE" \
  --contact_graph_strength "$CONTACT_GRAPH_STRENGTH" \
  --contact_graph_layer_start "$CONTACT_GRAPH_LAYER_START" \
  --contact_graph_layer_end "$CONTACT_GRAPH_LAYER_END" \
  --contact_graph_seed "$CONTACT_GRAPH_SEED" \
  --mask_white_threshold 245 \
  --object_min_latent_coverage 0.001 \
  --src_prompt "A person is holding a white plastic seasoning shaker with a blue cap in a kitchen." \
  --trg_prompt "A person is holding a ribbed silver aluminum can with horizontal ribs in a kitchen." \
  --src_word "white plastic seasoning shaker" \
  --trg_word "ribbed silver aluminum can" \
  --fg_boost_factor 4 \
  --blend_power 2 \
  --step "$STEP" \
  --seed 0 \
  --rollout_chunk_size 21 \
  --rollout_overlap_block_num 1 \
  2>&1 | tee "$OUTDIR/run.log"
