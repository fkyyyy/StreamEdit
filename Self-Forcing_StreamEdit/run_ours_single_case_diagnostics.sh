#!/usr/bin/env bash
set -euo pipefail

# Two-pass, Full-only diagnostic run.  Pass 1 selects the frame exclusively
# from source-side responsibility/hand evidence and is also the no-attention-
# hook reference.  Pass 2 captures the selected attention rows and verifies
# that diagnostics do not alter the decoded result.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$SCRIPT_DIR/artifacts/ours_single_case_diagnostics}"
BASELINE_DIR="$ARTIFACT_ROOT/baseline"
RUN_DIR="${CAPTURE_DIR:-$ARTIFACT_ROOT/capture}"
SELECTION_JSON="$ARTIFACT_ROOT/selection.json"
RUN_MANIFEST="$ARTIFACT_ROOT/run_manifest.json"
VIDEO_COMPARISON="$ARTIFACT_ROOT/decoded_video_comparison.json"

DATA_PATH="${DATA_PATH:-$DATA_ROOT/cook.mp4}"
HAND_MASK="${HAND_MASK:-$DATA_ROOT/cook_handmask.mp4}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/bn/public-lf4/fky/checkpoints/checkpoints/self_forcing_dmd.pt}"
WAN_MODELS_ROOT="${WAN_MODELS_ROOT:-/mnt/bn/public-lf4/fky/checkpoints/wan_models}"
CONFIG_PATH="${CONFIG_PATH:-$SCRIPT_DIR/configs/self_forcing_dmd.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
PHASE="${PHASE:-all}"

if (($#)); then
  echo "This entrypoint accepts no positional arguments; use environment variables." >&2
  exit 2
fi

if [[ "$PHASE" != baseline && "$PHASE" != capture && "$PHASE" != all ]]; then
  echo "PHASE must be baseline, capture, or all (got: $PHASE)" >&2
  exit 2
fi

if [[ "$PHASE" == all && -d "$ARTIFACT_ROOT" ]] &&
   [[ -n "$(find "$ARTIFACT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty artifact root: $ARTIFACT_ROOT" >&2
  exit 2
fi

run_full() {
  local outdir="$1"
  local output_name="$2"
  shift 2
  env \
    DATA_PATH="$DATA_PATH" \
    HAND_MASK="$HAND_MASK" \
    OUTDIR="$outdir" \
    OUTPUT_NAME="$output_name" \
    CONFIG_NAME="full_resolved_config.txt" \
    CONFIG_PATH="$CONFIG_PATH" \
    CHECKPOINT_PATH="$CHECKPOINT_PATH" \
    WAN_MODELS_ROOT="$WAN_MODELS_ROOT" \
    PYTHON_BIN="$PYTHON_BIN" \
    CUDA_DEVICE="$CUDA_DEVICE" \
    STEP=15 \
    S1M2_ATTENTION_MODE=full \
    bash "$SCRIPT_DIR/run_cook_S1M2_full.sh" "$@"
}

mkdir -p "$ARTIFACT_ROOT"

if [[ "$PHASE" == baseline || "$PHASE" == all ]]; then
  if [[ -e "$BASELINE_DIR" || -e "$SELECTION_JSON" ]]; then
    echo "Refusing to overwrite existing baseline/selection under: $ARTIFACT_ROOT" >&2
    exit 2
  fi
  mkdir "$BASELINE_DIR"
  run_full "$BASELINE_DIR" baseline_full.mp4 \
    --responsibility_diagnostics_dir "$BASELINE_DIR"
  "$PYTHON_BIN" "$SCRIPT_DIR/tools/select_single_case_diagnostics.py" \
    --responsibility-dir "$BASELINE_DIR/responsibility" \
    --output "$SELECTION_JSON"
fi

if [[ "$PHASE" == capture || "$PHASE" == all ]]; then
  if [[ ! -f "$SELECTION_JSON" ]]; then
    echo "Missing evidence-only selection: $SELECTION_JSON" >&2
    exit 2
  fi
  if [[ ! -f "$BASELINE_DIR/baseline_full.mp4" ]]; then
    echo "Missing baseline video: $BASELINE_DIR/baseline_full.mp4" >&2
    exit 2
  fi
  if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing capture directory: $RUN_DIR" >&2
    exit 2
  fi
  for owned_output in \
    cross_attention self_attention responsibility permissions \
    update_T0 update_T1 update_Tm anchor \
    ours_mechanism_dashboard.png ours_mechanism_dashboard.pdf \
    interface_token_card.png interface_token_card.json \
    edited_video.mp4 metrics.json metadata.json run_manifest.json; do
    if [[ -e "$ARTIFACT_ROOT/$owned_output" ]]; then
      echo "Refusing to overwrite existing visualization output: $ARTIFACT_ROOT/$owned_output" >&2
      exit 2
    fi
  done
  mkdir "$RUN_DIR"
  read -r BLOCK FRAME QUERY_INTERFACE QUERY_OBJECT QUERY_HAND < <(
    "$PYTHON_BIN" -c \
      'import json,sys; d=json.load(open(sys.argv[1])); print(d["block_index"], d["global_latent_frame"], *d["query_indices"])' \
      "$SELECTION_JSON"
  )
  run_full "$RUN_DIR" diagnostic_full.mp4 \
    --responsibility_diagnostics_dir "$RUN_DIR" \
    --mechanism_diagnostics_dir "$RUN_DIR" \
    --mechanism_diagnostics_block "$BLOCK" \
    --mechanism_diagnostics_latent_frame "$FRAME" \
    --mechanism_diagnostics_steps 0 1 7 \
    --mechanism_diagnostics_query_indices \
      "$QUERY_INTERFACE" "$QUERY_OBJECT" "$QUERY_HAND"

  # Separate GPU executions are not bitwise deterministic in this stack, and
  # MP4 container bytes can differ even for identical decoded frames.  Compare
  # decoded RGB frames and record the discrepancy without misrepresenting it as
  # a causal hook-effect test.  Shape/frame mismatches remain fatal.
  "$PYTHON_BIN" -c '
import av
import json
import math
import sys
from pathlib import Path

def frames(path):
    with av.open(path) as container:
        for frame in container.decode(video=0):
            yield frame.to_rgb().to_ndarray()

left_path, right_path, output_path = sys.argv[1:]
left = list(frames(left_path))
right = list(frames(right_path))
if len(left) != len(right):
    raise SystemExit(f"Decoded frame-count mismatch: {len(left)} != {len(right)}")
pixel_count = 0
abs_sum = 0.0
sq_sum = 0.0
max_abs = 0
different = 0
shapes = []
for index, (a, b) in enumerate(zip(left, right)):
    if a.shape != b.shape:
        raise SystemExit(f"Decoded shape mismatch at frame {index}: {a.shape} != {b.shape}")
    if index == 0:
        shapes = [len(left), *a.shape]
    delta = a.astype("float64") - b.astype("float64")
    absolute = abs(delta)
    pixel_count += delta.size
    abs_sum += float(absolute.sum())
    sq_sum += float((delta * delta).sum())
    max_abs = max(max_abs, int(absolute.max()))
    different += int((absolute != 0).sum())
payload = {
    "comparison_scope": "two independent seeded GPU executions",
    "causal_claim": "not sufficient by itself to attribute differences to diagnostics",
    "decoded_shape": shapes,
    "exact_decoded_match": different == 0,
    "different_channel_values": different,
    "max_abs_channel_difference": max_abs,
    "mean_abs_channel_difference": abs_sum / pixel_count,
    "rmse_channel_difference": math.sqrt(sq_sum / pixel_count),
}
Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
' "$BASELINE_DIR/baseline_full.mp4" "$RUN_DIR/diagnostic_full.mp4" "$VIDEO_COMPARISON"

  GIT_COMMIT="$(git -C "$SCRIPT_DIR" rev-parse HEAD)"
  "$PYTHON_BIN" -c '
import json
import shlex
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
baseline_path = Path(sys.argv[2])
capture_path = Path(sys.argv[3])
git_commit = sys.argv[4]

def resolved(path):
    entries = {}
    command = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("command="):
            command = line[len("command="):].strip()
        elif "=" in line:
            key, value = line.split("=", 1)
            entries[key] = value
    if not command:
        raise SystemExit(f"Missing command= in {path}")
    return entries, command, shlex.split(command)

def option(argv, name):
    positions = [index for index, value in enumerate(argv) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise SystemExit(f"Expected exactly one {name} in resolved command")
    return argv[positions[0] + 1]

baseline_entries, baseline_command, baseline_argv = resolved(baseline_path)
capture_entries, capture_command, capture_argv = resolved(capture_path)
case_fields = {
    "source_video": option(capture_argv, "--data_path"),
    "hand_mask_video": option(capture_argv, "--hand_mask_video"),
    "source_prompt": option(capture_argv, "--src_prompt"),
    "target_prompt": option(capture_argv, "--trg_prompt"),
    "source_phrase": option(capture_argv, "--src_word"),
    "target_phrase": option(capture_argv, "--trg_word"),
    "seed": int(option(capture_argv, "--seed")),
    "steps": int(option(capture_argv, "--step")),
    "rollout_chunk_size": int(option(capture_argv, "--rollout_chunk_size")),
    "checkpoint_path": option(capture_argv, "--checkpoint_path"),
    "config_path": option(capture_argv, "--config_path"),
    "wan_models_root": capture_entries["wan_models_root"],
}
for flag in ("--data_path", "--hand_mask_video", "--src_prompt", "--trg_prompt", "--src_word", "--trg_word", "--seed", "--step", "--rollout_chunk_size", "--checkpoint_path", "--config_path"):
    if option(baseline_argv, flag) != option(capture_argv, flag):
        raise SystemExit(f"Baseline/capture provenance mismatch for {flag}")
if baseline_entries.get("wan_models_root") != capture_entries.get("wan_models_root"):
    raise SystemExit("Baseline/capture provenance mismatch for wan_models_root")
payload = {
    "schema_version": "ours-single-case-run-manifest-v1",
    "git_commit": git_commit,
    "case": case_fields,
    "baseline": {
        "resolved_config": str(baseline_path.resolve()),
        "command": baseline_command,
    },
    "capture": {
        "resolved_config": str(capture_path.resolve()),
        "command": capture_command,
    },
    "diagnostic_updates": {"T0": 0, "T1": 1, "Tm": 7},
}
manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
' "$RUN_MANIFEST" \
    "$BASELINE_DIR/full_resolved_config.txt" \
    "$RUN_DIR/full_resolved_config.txt" \
    "$GIT_COMMIT"

  mkdir -p "$ARTIFACT_ROOT/.matplotlib"
  MPLCONFIGDIR="$ARTIFACT_ROOT/.matplotlib" "$PYTHON_BIN" \
    "$SCRIPT_DIR/tools/visualize_single_case_diagnostics.py" \
    --responsibility-dir "$RUN_DIR/responsibility" \
    --attention-dir "$RUN_DIR/raw_attention" \
    --source-video "$DATA_PATH" \
    --edited-video "$RUN_DIR/diagnostic_full.mp4" \
    --baseline-video "$BASELINE_DIR/baseline_full.mp4" \
    --selection "$SELECTION_JSON" \
    --hand-role-input "$RUN_DIR/diagnostic_full.hand_role_input.npz" \
    --run-manifest "$RUN_MANIFEST" \
    --output-dir "$ARTIFACT_ROOT"
fi

echo "OURS_SINGLE_CASE_DIAGNOSTICS $ARTIFACT_ROOT"
