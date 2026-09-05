#!/usr/bin/env bash
set -euo pipefail

# M1: native StreamGVE plus one immutable target-minus-source value bank.
#
# The first generated block runs exactly as L0.  After its clean target commit,
# M1 freezes automatic-SOG foreground (source K, target V - source V) pairs.
# Later blocks use current clean-source Q to retrieve that frozen residual in a
# parallel output channel.  It never changes native attention K/V, softmax, or
# cache writes, and the bank is never updated.
#
# NO object/hand mask, NO optical flow, NO P0-P3, NO factorized routing,
# NO source-background suppression, NO soft modulation, NO old anchor.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video1.mp4}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/M1_immutable_deltaV_bank}"
OUTPUT_NAME="${OUTPUT_NAME:-M1-immutable-deltaV-bank.mp4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-7}"
DRY_RUN="${DRY_RUN:-0}"
STEP="${STEP:-15}"

readonly SRC_PROMPT='First-person POV shot, wide-angle lens. A person is relaxing on a beige sofa, holding a smartphone with both hands and actively scrolling through a calendar app interface. An open silver laptop rests on their lap, displaying a screen filled with blue technical diagrams or data charts. In the mid-ground, a pair of feet wearing white socks are propped up on a dark wooden coffee table, which is scattered with small white puzzle pieces. The background features a spacious, modern living room with a dark staircase on the left, a large black TV on a wooden cabinet, and two blue armchairs near a bright glass door on the right. Bright natural daylight, realistic 4k video style, slight fish-eye effect.'
readonly TRG_PROMPT='First-person POV shot, wide-angle lens. A person is relaxing on a beige sofa, holding a brown leather wallet with both hands, casually flipping it open. The wallet is a classic bifold design made of rich, dark brown genuine leather with visible grain texture and neat stitching along the edges. It has a slightly worn, warm patina on the surface. Beneath the wallet, an open silver laptop rests on their lap, displaying a screen filled with blue technical diagrams. In the mid-ground, a pair of feet wearing white socks are propped up on a dark wooden coffee table scattered with small white puzzle pieces. The background features a spacious, modern living room with a dark staircase on the left, a large black TV on a wooden cabinet, and two blue armchairs near a bright glass door on the right. Bright natural daylight, realistic 4k video style, slight fish-eye effect.'
readonly SRC_WORD='smartphone'
readonly TRG_WORD='brown leather wallet'

if [[ ! -f "$DATA_PATH" ]]; then
  echo "Missing: $DATA_PATH" >&2
  exit 2
fi

mkdir -p "$OUTDIR"

COMMAND=(
  "$PYTHON_BIN" "$SCRIPT_DIR/inference_edit_streamedit.py"
  --data_path "$DATA_PATH"
  --save_path "$OUTDIR/$OUTPUT_NAME"

  # THE SINGLE CHANGE from L0:
  --immutable_delta_v_bank
  --immutable_delta_v_layers 8 12 16 20
  --immutable_delta_v_topk 8
  --immutable_delta_v_min_similarity 0.35
  --immutable_delta_v_strength 0.20
  --immutable_delta_v_max_rms_ratio 1.0

  # Everything else = L0 protocol (native StreamGVE / dynamic_sog).
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
    'experiment=M1_immutable_deltaV_bank' \
    'baseline=L0_local_baseline' \
    'single_variable=immutable_delta_v_parallel_retrieval' \
    'routing=default_dynamic_sog' \
    'bank_write=first_generated_block_once' \
    'bank_payload=clean_target_v_minus_clean_source_v' \
    'bank_address=clean_source_pre_rope_q_to_first_block_source_k' \
    'native_attention_unchanged=true' \
    'native_kv_unchanged=true' \
    'hand_mask=disabled' \
    'object_mask=disabled' \
    'flow=disabled' \
    'projected_residual=disabled' \
    'factorized=disabled' \
    'soft_modulation=disabled' \
    'suppress=disabled' \
    'old_anchor=disabled' \
    'layers=8,12,16,20' \
    'topk=8' \
    'min_similarity=0.35' \
    'strength=0.20' \
    'max_rms_ratio=1.0' \
    "data_path=$DATA_PATH"
  printf 'command='
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
} > "$OUTDIR/M1_config.txt"

echo "M1: L0 + immutable first-block delta-V parallel retrieval"
echo "M1 first block is exact native; later blocks read the frozen bank"
echo "M1_OUTPUT $OUTDIR/$OUTPUT_NAME"

if [[ "$DRY_RUN" == 1 ]]; then
  echo 'M1_DRY_RUN resolved command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${COMMAND[@]}" 2>&1 | tee "$OUTDIR/run.log"
