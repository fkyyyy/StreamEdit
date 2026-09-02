#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/940b_block1_native_bypass}"
OUTPUT_NAME="${OUTPUT_NAME:-940b-block1-native-bypass.mp4}"

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=939a_uncertainty_abstaining_source_closure' \
    'ablation=block1_role_fixed_native_kv_bypass' \
    'block1_attention=exact_native_926' \
    'block2_plus=role_fixed_native_kv_restored' \
    'bootstrap_rope_fix=disabled_for_independent_ablation' \
    'prompt_change=none' \
    'owner_change=none' \
    'velocity_change=none'
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/940_config.txt"

# Diagnostic control: do not replace native attention in causal block 1.
# Memory writes continue normally and role-fixed reads resume at block 2.
OUTDIR="$OUTDIR" OUTPUT_NAME="$OUTPUT_NAME" \
  "$SCRIPT_DIR/run_939a_uncertainty_abstaining_source_closure.sh" \
  --native_history_bypass_blocks 1 \
  "$@"
