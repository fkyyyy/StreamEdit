#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# One-variable KV write-contract ablation from the untouched 954p baseline.
# Clean-source K remains a complete address table, while recent target K/V is
# readable only at tokens accepted by the existing conservative hand-flow
# write transaction. No target payload is inferred from an object mask.
export CUDA_DEVICE="${CUDA_DEVICE:-7}"
export OUTDIR="${OUTDIR:-$SCRIPT_DIR/outputs/957a_token_atomic_payload_commit}"
export OUTPUT_NAME="${OUTPUT_NAME:-957a-token-atomic-payload-commit.mp4}"

mkdir -p "$OUTDIR"
{
  printf '%s\n' \
    'baseline=954p_streamgve_prompt_ablation' \
    'change=token_atomic_recent_target_payload_commit' \
    'prompt=unchanged_streamgve_minimal_difference' \
    'spatial_routing=unchanged_from_954a' \
    'external_hand_mask=enabled' \
    'external_object_mask=disabled' \
    'external_source_owner_mask=disabled' \
    'source_address_table=dense_clean_source_pre_rope_k' \
    'target_payload_support=transactional_write_approved_tokens_only' \
    'unauthorized_payload_read=exact_native_abstention' \
    'immutable_canonical_fallback=disabled_for_token_atomic_queries' \
    'write_threshold=existing_native_history_min_write_confidence'
  printf 'extra_args='
  printf ' %q' "$@"
  printf '\n'
} > "$OUTDIR/957a_config.txt"

exec bash "$SCRIPT_DIR/run_954p_streamgve_prompt_ablation.sh" \
  --native_history_token_atomic_payload \
  "$@"
