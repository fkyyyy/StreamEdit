from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "finalize_main_ablation.py"
RUNNER_PATH = REPO_ROOT / "run_main_ablation.sh"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "finalize_main_ablation_standalone", TOOL_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


finalizer = load_module()


def test_finalize_emits_exactly_four_new_rows_with_blank_metrics(tmp_path):
    assert finalizer.METRIC_COLUMNS == (
        "target_edit_score",
        "source_appearance_removal",
        "hand_fidelity",
        "interaction_consistency",
        "background_preservation",
        "long_term_appearance_consistency",
    )
    root = tmp_path / "main_ablation"
    for variant, name, flag in finalizer.VARIANTS:
        config_dir = root / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / f"{variant}_{name}.json").write_text(
            json.dumps(
                {
                    "schema_version": finalizer.SCHEMA_VERSION,
                    "case_id": finalizer.CASE_ID,
                    "variant": variant,
                    "variant_name": name,
                    "ablation_flag": flag,
                    "status": "success",
                    "config_hash": f"sha256:{variant.lower()}",
                    "commit": "deadbeef",
                    "runtime_seconds": 12.5,
                }
            ),
            encoding="utf-8",
        )

    outputs = finalizer.finalize(root)

    assert len(outputs) == 9
    for stem in (
        root / "metrics" / "per_case_new_ablations",
        root / "metrics" / "summary_new_ablations",
        root / "main_ablation_table",
    ):
        for suffix in (".csv", ".md", ".tex"):
            assert stem.with_suffix(suffix).is_file()

    with (root / "main_ablation_table.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["variant"] for row in rows] == ["B1", "B2", "B3", "B4"]
    assert all(row["status"] == "success" for row in rows)
    assert all(
        row[metric] == "" for row in rows for metric in finalizer.METRIC_COLUMNS
    )
    assert all(row["historical_rows_merged"] == "false" for row in rows)
    assert "Historical Full/baseline results were not merged" in (
        root / "main_ablation_table.md"
    ).read_text(encoding="utf-8")
    assert "% Historical Full/baseline results were not merged" in (
        root / "main_ablation_table.tex"
    ).read_text(encoding="utf-8")


def test_all_runs_in_order_with_fixed_case_and_isolated_outputs(tmp_path):
    fake_runner = tmp_path / "fake_full.sh"
    fake_ffprobe = tmp_path / "fake_ffprobe.sh"
    call_log = tmp_path / "calls.txt"
    fake_runner.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$OUTDIR/roles"
printf '%s|%s|%s|%s|%s|%s\\n' "$1" "$OUTDIR" "$DATA_PATH" "$HAND_MASK" "$CHECKPOINT_PATH" "$WAN_MODELS_ROOT" >> "$FAKE_CALL_LOG"
printf 'command= python inference_edit_streamedit.py --data_path %q --save_path %q %q\\n' "$DATA_PATH" "$OUTDIR/$OUTPUT_NAME" "$1" > "$OUTDIR/$CONFIG_NAME"
printf 'video' > "$OUTDIR/$OUTPUT_NAME"
case "$1" in
  --ablation_no_interface_role)
    printf '%s\\n' 'ABLATION_INTERFACE_ROLE block=0 interface_mass_after=0.000000 role_sum_max_error=0.000e+00' > "$OUTDIR/run.log" ;;
  --ablation_no_direction_filtering)
    printf '%s\\n' 'ABLATION_DIRECTION_FILTERING block=0 max_abs_rho_safe_minus_rho=0.000e+00 role_allocation=1' > "$OUTDIR/run.log" ;;
  --ablation_no_attention_control)
    printf '%s\\n' 'inference started' 'ABLATION_ATTENTION_CONTROL block=0 spatial_qk=0 base_scalar_blending=1 velocity_update=1 appearance_anchor=1' 'IMMUTABLE_DELTA_V_FREEZE block=0 write_once=1' 'IMMUTABLE_DELTA_V_READ block=1' > "$OUTDIR/run.log" ;;
  --ablation_no_appearance_anchor)
    printf '%s\\n' 'ABLATION_APPEARANCE_ANCHOR bank_enabled=0 construction=0 write=0 retrieval=0 correction=0' 'NATIVE_TARGET_HISTORY block=0 selected=1' 'ROLE_AWARE_S1M2 block=0 effective_residual=0.8' 'S1M2_ATTENTION block=0 spatial=30 m2=0 bank_frozen=0' > "$OUTDIR/run.log" ;;
esac
""",
        encoding="utf-8",
    )
    fake_runner.chmod(0o755)
    fake_ffprobe.write_text(
        """#!/usr/bin/env bash
printf '%s\\n' '{"streams":[{"codec_name":"h264","width":832,"height":480,"duration":"5.4","nb_frames":"81"}],"format":{"duration":"5.4","size":"5"}}'
""",
        encoding="utf-8",
    )
    fake_ffprobe.chmod(0o755)
    artifact_root = tmp_path / "artifacts" / "main_ablation"
    environment = os.environ.copy()
    environment.update(
        {
            "MAIN_ABLATION_FULL_RUNNER": str(fake_runner),
            "MAIN_ABLATION_ROOT": str(artifact_root),
            "FAKE_CALL_LOG": str(call_log),
            "PYTHON_BIN": sys.executable,
            "FFPROBE_BIN": str(fake_ffprobe),
            "DATA_PATH": "/must/not/be/used.mp4",
            "HAND_MASK": "/must/not/be/used-mask.mp4",
            "CHECKPOINT_PATH": "/must/not/be/used.pt",
            "WAN_MODELS_ROOT": "/must/not/be/used-wan",
        }
    )

    result = subprocess.run(
        ["bash", str(RUNNER_PATH), "all"],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert [line.split("|", 1)[0] for line in calls] == [
        "--ablation_no_interface_role",
        "--ablation_no_direction_filtering",
        "--ablation_no_attention_control",
        "--ablation_no_appearance_anchor",
    ]
    for index, variant in enumerate(("B1", "B2", "B3", "B4")):
        fields = calls[index].split("|")
        name = finalizer.VARIANTS[index][1]
        slug = f"{variant}_{name}"
        assert fields[1] == str(artifact_root / "videos" / slug)
        assert fields[2] == "/mnt/bn/public-lf4/fky/resources/case2/08_wooden_spoon_b5a69994a99b4d01e4d6c306ac011477.mp4"
        assert fields[3] == "/mnt/bn/public-lf4/fky/wooden_spoon_hand/mask.mp4"
        assert fields[4] == "/mnt/bn/public-lf4/fky/checkpoints/checkpoints/self_forcing_dmd.pt"
        assert fields[5] == "/mnt/bn/public-lf4/fky/checkpoints/wan_models"
        metadata = json.loads(
            (artifact_root / "configs" / f"{slug}.json").read_text(encoding="utf-8")
        )
        assert metadata["status"] == "success"
        assert metadata["ablation_flag"] == fields[0]
        assert metadata["config_hash"].startswith("sha256:")
        assert metadata["command"]
        assert metadata["commit"]
        assert metadata["runtime_seconds"] >= 0
        assert metadata["case_id"] == "08_wooden_spoon_to_iron_black_spatula"
        assert metadata["seed"] == 0
        assert metadata["steps"] == 15
        assert metadata["chunk_size"] == 21
        assert metadata["overlap"] == 1
        assert metadata["source_prompt"] == "a person is cooking with a wooden spoon."
        assert metadata["target_prompt"] == "a person is cooking with a iron black spatula"
        assert metadata["output_validation"] == {
            "path": str(
                artifact_root
                / "videos"
                / slug
                / "edited.mp4"
            ),
            "size_bytes": 5,
            "status": "valid",
            "valid": True,
            "expected_structure": {
                "frames": 81,
                "height": 480,
                "width": 832,
            },
            "checks": {
                "frame_count_matches": True,
                "resolution_matches": True,
            },
            "video_stream": {
                "codec_name": "h264",
                "duration_seconds": 5.4,
                "height": 480,
                "nb_frames": 81,
                "width": 832,
            },
        }
        assert metadata["smoke_checks"]["status"] == "passed"
        assert metadata["artifacts"]["run_log"] == str(
            artifact_root / "logs" / f"{slug}.log"
        )

    with (artifact_root / "metrics" / "per_case_new_ablations.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 4


def test_invalid_selector_is_rejected_without_running(tmp_path):
    result = subprocess.run(
        ["bash", str(RUNNER_PATH), "B5"],
        cwd=REPO_ROOT,
        env={**os.environ, "MAIN_ABLATION_ROOT": str(tmp_path)},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Usage:" in result.stderr


def test_successful_process_without_video_is_recorded_as_failed(tmp_path):
    fake_runner = tmp_path / "fake_full_without_video.sh"
    fake_runner.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$OUTDIR/roles"
printf 'command= python inference_edit_streamedit.py %q\\n' "$1" > "$OUTDIR/$CONFIG_NAME"
""",
        encoding="utf-8",
    )
    artifact_root = tmp_path / "main_ablation"

    result = subprocess.run(
        ["bash", str(RUNNER_PATH), "B1"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "MAIN_ABLATION_FULL_RUNNER": str(fake_runner),
            "MAIN_ABLATION_ROOT": str(artifact_root),
            "PYTHON_BIN": sys.executable,
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 3
    metadata = json.loads(
        (
            artifact_root / "configs" / "B1_no_interface_role.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["status"] == "failed"
    assert metadata["output_validation"]["status"] == "missing"
    assert not (artifact_root / "main_ablation_table.csv").exists()


def test_numeric_smoke_does_not_match_the_word_inference(tmp_path):
    log_path = tmp_path / "b4.log"
    log_path.write_text(
        "inference complete\n"
        "ABLATION_APPEARANCE_ANCHOR bank_enabled=0 construction=0 "
        "write=0 retrieval=0 correction=0\n"
        "NATIVE_TARGET_HISTORY block=0 selected=1\n"
        "ROLE_AWARE_S1M2 block=0 effective_residual=0.8\n"
        "S1M2_ATTENTION block=0 spatial=30 m2=0 bank_frozen=0\n",
        encoding="utf-8",
    )

    result = finalizer.smoke_check_log("B4", log_path, dry_run=False)

    assert result["status"] == "passed"
    assert result["checks"]["numeric_nan_inf_absent"]
