#!/usr/bin/env bash
set -euo pipefail

# Run one or all of the four main ablations from the current Full recipe.
# The editing case and model roots are intentionally fixed for comparability.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
FULL_RUNNER="${MAIN_ABLATION_FULL_RUNNER:-$SCRIPT_DIR/run_cook_S1M2_full.sh}"
FINALIZER="$SCRIPT_DIR/tools/finalize_main_ablation.py"
ARTIFACT_ROOT="${MAIN_ABLATION_ROOT:-$SCRIPT_DIR/artifacts/main_ablation}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FFPROBE_BIN="${FFPROBE_BIN:-ffprobe}"

readonly DATA_PATH="/mnt/bn/public-lf4/fky/resources/case2/08_wooden_spoon_b5a69994a99b4d01e4d6c306ac011477.mp4"
readonly HAND_MASK="/mnt/bn/public-lf4/fky/wooden_spoon_hand/mask.mp4"
readonly CHECKPOINT_PATH="/mnt/bn/public-lf4/fky/checkpoints/checkpoints/self_forcing_dmd.pt"
readonly WAN_MODELS_ROOT="/mnt/bn/public-lf4/fky/checkpoints/wan_models"
readonly SRC_PROMPT='a person is cooking with a wooden spoon.'
readonly TRG_PROMPT='a person is cooking with a iron black spatula'
readonly SRC_WORD='wooden spoon'
readonly TRG_WORD='iron black spatula'

usage() {
  echo "Usage: $0 {B1|B2|B3|B4|all}" >&2
}

variant_details() {
  case "$1" in
    B1) printf '%s\n%s\n' 'no_interface_role' '--ablation_no_interface_role' ;;
    B2) printf '%s\n%s\n' 'no_direction_filtering' '--ablation_no_direction_filtering' ;;
    B3) printf '%s\n%s\n' 'no_attention_control' '--ablation_no_attention_control' ;;
    B4) printf '%s\n%s\n' 'no_appearance_anchor' '--ablation_no_appearance_anchor' ;;
    *) return 2 ;;
  esac
}

run_variant() {
  local variant="$1"
  local details variant_name variant_slug ablation_flag video_dir output_path
  local config_path metadata_path log_path generated_config generated_log
  local started_at ended_at start_epoch end_epoch runtime_seconds status exit_code

  details="$(variant_details "$variant")"
  variant_name="$(sed -n '1p' <<<"$details")"
  ablation_flag="$(sed -n '2p' <<<"$details")"
  variant_slug="${variant}_${variant_name}"
  video_dir="$ARTIFACT_ROOT/videos/$variant_slug"
  output_path="$video_dir/edited.mp4"
  config_path="$ARTIFACT_ROOT/configs/$variant_slug.txt"
  metadata_path="$ARTIFACT_ROOT/configs/$variant_slug.json"
  log_path="$ARTIFACT_ROOT/logs/$variant_slug.log"
  generated_config="$video_dir/.resolved_config.txt"
  generated_log="$video_dir/run.log"
  mkdir -p "$video_dir" "$ARTIFACT_ROOT/configs" "$ARTIFACT_ROOT/logs" "$ARTIFACT_ROOT/metrics"

  started_at="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  start_epoch="$(date +%s)"
  echo "MAIN_ABLATION_START variant=$variant flag=$ablation_flag output=$output_path"

  if env \
    DATA_PATH="$DATA_PATH" \
    HAND_MASK="$HAND_MASK" \
    OUTDIR="$video_dir" \
    OUTPUT_NAME="$(basename -- "$output_path")" \
    CONFIG_NAME="$(basename -- "$generated_config")" \
    EXPERIMENT_NAME="main_ablation_${variant}_${variant_name}" \
    EDIT_NAME="08_wooden_spoon_to_iron_black_spatula" \
    RUN_LABEL="Main ablation $variant" \
    CHECKPOINT_PATH="$CHECKPOINT_PATH" \
    WAN_MODELS_ROOT="$WAN_MODELS_ROOT" \
    STEP=15 \
    S1M2_ATTENTION_MODE=full \
    SRC_PROMPT="$SRC_PROMPT" \
    TRG_PROMPT="$TRG_PROMPT" \
    SRC_WORD="$SRC_WORD" \
    TRG_WORD="$TRG_WORD" \
    bash "$FULL_RUNNER" "$ablation_flag"; then
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
      status=dry_run
    else
      status=success
    fi
    exit_code=0
  else
    exit_code=$?
    status=failed
  fi

  if [[ -f "$generated_config" ]]; then
    mv -f -- "$generated_config" "$config_path"
  fi
  if [[ -f "$generated_log" ]]; then
    mv -f -- "$generated_log" "$log_path"
  fi

  end_epoch="$(date +%s)"
  ended_at="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  runtime_seconds="$((end_epoch - start_epoch))"
  if "$PYTHON_BIN" "$FINALIZER" record \
    --root "$ARTIFACT_ROOT" \
    --repo-root "$REPO_ROOT" \
    --variant "$variant" \
    --variant-name "$variant_name" \
    --ablation-flag="$ablation_flag" \
    --status "$status" \
    --exit-code "$exit_code" \
    --started-at "$started_at" \
    --ended-at "$ended_at" \
    --runtime-seconds "$runtime_seconds" \
    --runner "$FULL_RUNNER" \
    --metadata-path "$metadata_path" \
    --output-path "$output_path" \
    --roles-path "$video_dir/roles" \
    --config-path "$config_path" \
    --log-path "$log_path" \
    --ffprobe-bin "$FFPROBE_BIN"; then
    :
  else
    exit_code=$?
    status=failed
  fi

  if [[ "$status" == success ]]; then
    "$PYTHON_BIN" "$FINALIZER" finalize --root "$ARTIFACT_ROOT"
    echo "MAIN_ABLATION_DONE variant=$variant runtime_seconds=$runtime_seconds"
    return 0
  fi

  if [[ "$status" == dry_run ]]; then
    echo "MAIN_ABLATION_DRY_RUN_DONE variant=$variant runtime_seconds=$runtime_seconds"
    return 0
  fi

  echo "MAIN_ABLATION_FAILED variant=$variant exit_code=$exit_code" >&2
  return "$exit_code"
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

case "$1" in
  B1|B2|B3|B4)
    run_variant "$1"
    ;;
  all)
    for variant in B1 B2 B3 B4; do
      run_variant "$variant"
    done
    ;;
  *)
    usage
    exit 2
    ;;
esac
