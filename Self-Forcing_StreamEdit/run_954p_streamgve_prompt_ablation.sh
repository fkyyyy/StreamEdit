#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Prompt-only ablation on top of 954a.  The source and target prompts have the
# same syntax and scene description; only the held-object phrase changes.
# All spatial routing, optical-flow ownership, and KV settings come unchanged
# from run_954a_motion_owner_authority_closure.sh.
export CUDA_DEVICE="${CUDA_DEVICE:-7}"
export DATA_PATH="${DATA_PATH:-$REPO_ROOT/source_video1.mp4}"
export HAND_MASK="${HAND_MASK:-$REPO_ROOT/hand_mask1.mp4}"
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/954p_streamgve_prompt_ablation}"
export OUTPUT_NAME="${OUTPUT_NAME:-954p-streamgve-prompt-ablation.mp4}"

export SRC_PROMPT='A first-person egocentric indoor video. Two hands hold a vertically oriented smartphone in front of an open silver laptop on a table, with chairs, a television, a cabinet, walls, and a window in the background. The camera follows the hands, focusing on the held object.'
export TRG_PROMPT='A first-person egocentric indoor video. Two hands hold a vertically oriented matte pale-yellow cardboard package in front of an open silver laptop on a table, with chairs, a television, a cabinet, walls, and a window in the background. The camera follows the hands, focusing on the held object.'
export SRC_WORD='smartphone'
export TRG_WORD='cardboard package'

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=954a_motion_owner_authority_closure' \
    'ablation=streamgve_style_minimal_prompt_difference' \
    'core_code=unchanged' \
    'spatial_routing=unchanged_from_954a' \
    'kv_read_write=unchanged_from_954a' \
    'external_hand_mask=enabled' \
    'external_object_mask=disabled' \
    'external_source_owner_mask=disabled' \
    "src_prompt=$SRC_PROMPT" \
    "trg_prompt=$TRG_PROMPT" \
    "src_word=$SRC_WORD" \
    "trg_word=$TRG_WORD"
} > "$OUTDIR/954p_config.txt"

exec bash "$SCRIPT_DIR/run_954a_motion_owner_authority_closure.sh" "$@"
