#!/usr/bin/env python3
"""Record and summarize the four main S1M2 ablations.

This utility deliberately does not compute visual-quality metrics.  It keeps
the six evaluation columns empty until an independent evaluator provides
them, while still producing stable, reviewable tables for the new runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 2
CASE_ID = "08_wooden_spoon_to_iron_black_spatula"
RUN_CONFIG = {
    "seed": 0,
    "steps": 15,
    "chunk_size": 21,
    "overlap": 1,
    "source_video": "/mnt/bn/public-lf4/fky/resources/case2/08_wooden_spoon_b5a69994a99b4d01e4d6c306ac011477.mp4",
    "hand_mask": "/mnt/bn/public-lf4/fky/wooden_spoon_hand/mask.mp4",
    "source_prompt": "a person is cooking with a wooden spoon.",
    "target_prompt": "a person is cooking with a iron black spatula",
    "source_word": "wooden spoon",
    "target_word": "iron black spatula",
    "checkpoint": "/mnt/bn/public-lf4/fky/checkpoints/checkpoints/self_forcing_dmd.pt",
    "wan_root": "/mnt/bn/public-lf4/fky/checkpoints/wan_models",
}
VARIANTS = (
    ("B1", "no_interface_role", "--ablation_no_interface_role"),
    ("B2", "no_direction_filtering", "--ablation_no_direction_filtering"),
    ("B3", "no_attention_control", "--ablation_no_attention_control"),
    ("B4", "no_appearance_anchor", "--ablation_no_appearance_anchor"),
)
METRIC_COLUMNS = (
    "target_edit_score",
    "source_appearance_removal",
    "hand_fidelity",
    "interaction_consistency",
    "background_preservation",
    "long_term_appearance_consistency",
)
EXPECTED_VIDEO_WIDTH = 832
EXPECTED_VIDEO_HEIGHT = 480
EXPECTED_VIDEO_FRAMES = 81
HISTORY_EXCLUSION = (
    "Historical Full/baseline results were not merged because their metadata "
    "does not match this main-ablation run contract."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _resolved_command(config_path: Path, runner: Path, flag: str) -> str:
    if config_path.is_file():
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("command="):
                return line.removeprefix("command=").strip()
    return f"bash {runner} {flag}"


def validate_video(
    output_path: Path, *, dry_run: bool, ffprobe_bin: str
) -> dict[str, Any]:
    """Validate that an output contains one usable, non-empty video stream."""
    if dry_run:
        return {
            "status": "skipped_dry_run",
            "valid": None,
            "path": str(output_path),
        }
    if not output_path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "path": str(output_path),
        }
    size_bytes = output_path.stat().st_size
    if size_bytes <= 0:
        return {
            "status": "empty",
            "valid": False,
            "path": str(output_path),
            "size_bytes": size_bytes,
        }

    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,nb_frames,duration",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True
        )
    except OSError as error:
        return {
            "status": "probe_unavailable",
            "valid": False,
            "path": str(output_path),
            "size_bytes": size_bytes,
            "error": str(error),
        }
    if result.returncode != 0:
        return {
            "status": "probe_failed",
            "valid": False,
            "path": str(output_path),
            "size_bytes": size_bytes,
            "error": result.stderr.strip(),
        }
    try:
        probe = json.loads(result.stdout)
        stream = probe["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        duration = float(
            stream.get("duration") or probe.get("format", {}).get("duration")
        )
        frame_count = int(stream["nb_frames"])
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "status": "invalid_structure",
            "valid": False,
            "path": str(output_path),
            "size_bytes": size_bytes,
            "error": str(error),
        }
    structure_matches = (
        width == EXPECTED_VIDEO_WIDTH
        and height == EXPECTED_VIDEO_HEIGHT
        and frame_count == EXPECTED_VIDEO_FRAMES
    )
    if duration <= 0 or not structure_matches:
        return {
            "status": "structure_mismatch",
            "valid": False,
            "path": str(output_path),
            "size_bytes": size_bytes,
            "probe": probe,
            "expected": {
                "width": EXPECTED_VIDEO_WIDTH,
                "height": EXPECTED_VIDEO_HEIGHT,
                "frames": EXPECTED_VIDEO_FRAMES,
            },
        }
    return {
        "status": "valid",
        "valid": True,
        "path": str(output_path),
        "size_bytes": size_bytes,
        "video_stream": {
            "codec_name": stream.get("codec_name", ""),
            "width": width,
            "height": height,
            "duration_seconds": duration,
            "nb_frames": frame_count,
        },
        "expected_structure": {
            "width": EXPECTED_VIDEO_WIDTH,
            "height": EXPECTED_VIDEO_HEIGHT,
            "frames": EXPECTED_VIDEO_FRAMES,
        },
        "checks": {
            "resolution_matches": True,
            "frame_count_matches": True,
        },
    }


def smoke_check_log(
    variant: str, log_path: Path, *, dry_run: bool
) -> dict[str, Any]:
    """Check variant-specific runtime evidence without invoking an evaluator."""
    import re

    if dry_run:
        return {"status": "skipped_dry_run", "valid": None, "checks": {}}
    if not log_path.is_file():
        return {
            "status": "missing_log",
            "valid": False,
            "checks": {},
            "errors": [f"missing log: {log_path}"],
        }
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    errors: list[str] = []
    checks: dict[str, Any] = {}

    numeric_nonfinite = re.compile(
        r"(?<![A-Za-z0-9_])(?:nan|[+-]?inf)(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )
    nonfinite_matches = numeric_nonfinite.findall(text)
    checks["numeric_nan_inf_absent"] = not nonfinite_matches
    if nonfinite_matches:
        errors.append("numeric NaN/Inf token found in log")

    if variant == "B1":
        evidence = [line for line in lines if "ABLATION_INTERFACE_ROLE" in line]
        valid_values = []
        for line in evidence:
            after = re.search(r"interface_mass_after=([^ ]+)", line)
            role_error = re.search(r"role_sum_max_error=([^ ]+)", line)
            try:
                valid_values.append(
                    after is not None
                    and role_error is not None
                    and float(after.group(1)) == 0.0
                    and float(role_error.group(1)) <= 1e-5
                )
            except ValueError:
                valid_values.append(False)
        checks.update(
            {
                "ablation_interface_role_lines": len(evidence),
                "all_interface_mass_after_zero": bool(evidence) and all(valid_values),
                "all_role_sum_max_error_lte_1e_5": bool(evidence)
                and all(valid_values),
            }
        )
        if not evidence or not all(valid_values):
            errors.append("B1 interface-role evidence missing or invalid")
    elif variant == "B2":
        evidence = [
            line for line in lines if "max_abs_rho_safe_minus_rho=" in line
        ]
        valid_values = []
        for line in evidence:
            difference = re.search(r"max_abs_rho_safe_minus_rho=([^ ]+)", line)
            allocation = re.search(r"role_allocation=([^ ]+)", line)
            try:
                valid_values.append(
                    "ABLATION_DIRECTION_FILTERING" in line
                    and difference is not None
                    and allocation is not None
                    and float(difference.group(1)) == 0.0
                    and allocation.group(1) == "1"
                )
            except ValueError:
                valid_values.append(False)
        checks.update(
            {
                "ablation_direction_filtering_lines": len(evidence),
                "all_max_abs_rho_safe_minus_rho_zero": bool(evidence)
                and all(valid_values),
                "all_role_allocation_one": bool(evidence) and all(valid_values),
            }
        )
        if not evidence or not all(valid_values):
            errors.append("B2 direction-filtering evidence missing or invalid")
    elif variant == "B3":
        control_lines = [
            line for line in lines if "ABLATION_ATTENTION_CONTROL" in line
        ]
        control_valid = any(
            "spatial_qk=0" in line
            and (
                "base_scalar=1" in line
                or "base_scalar_blending=1" in line
            )
            and "velocity_update=1" in line
            and "appearance_anchor=1" in line
            for line in control_lines
        )
        freeze = any(
            "IMMUTABLE_DELTA_V_FREEZE" in line and "write_once=1" in line
            for line in lines
        )
        read = any("IMMUTABLE_DELTA_V_READ" in line for line in lines)
        checks.update(
            {
                "attention_control": control_valid,
                "spatial_qk": 0 if control_valid else None,
                "base_scalar": 1 if control_valid else None,
                "velocity_update": 1 if control_valid else None,
                "appearance_anchor": 1 if control_valid else None,
                "immutable_delta_v_freeze": freeze,
                "immutable_delta_v_read": read,
            }
        )
        if not control_valid or not freeze or not read:
            errors.append("B3 attention-control or bank freeze/read evidence missing")
    elif variant == "B4":
        appearance = any("ABLATION_APPEARANCE_ANCHOR" in line for line in lines)
        spatial_qk = any(
            "S1M2_ATTENTION" in line
            and re.search(r"spatial=[1-9][0-9]*", line)
            for line in lines
        )
        velocity_update = any("ROLE_AWARE_S1M2" in line for line in lines)
        native_history = any("NATIVE_TARGET_HISTORY " in line for line in lines)
        forbidden = [
            marker
            for marker in (
                "IMMUTABLE_DELTA_V_FREEZE",
                "IMMUTABLE_DELTA_V_READ",
                "M2_CANONICAL_KV_COMMIT",
            )
            if marker in text
        ]
        checks.update(
            {
                "ablation_appearance_anchor": appearance,
                "spatial_qk_active": spatial_qk,
                "velocity_update_active": velocity_update,
                "native_causal_history_active": native_history,
                "forbidden_anchor_events_absent": not forbidden,
                "forbidden_anchor_events_found": forbidden,
            }
        )
        if (
            not appearance
            or not spatial_qk
            or not velocity_update
            or not native_history
            or forbidden
        ):
            errors.append("B4 appearance-anchor evidence missing or forbidden event found")

    return {
        "status": "passed" if not errors else "failed",
        "valid": not errors,
        "checks": checks,
        "errors": errors,
    }


def record_metadata(args: argparse.Namespace) -> Path:
    root = args.root.resolve()
    expected_name, expected_flag = next(
        (name, flag) for key, name, flag in VARIANTS if key == args.variant
    )
    if args.variant_name != expected_name or args.ablation_flag != expected_flag:
        raise ValueError(
            f"{args.variant} requires {expected_name} and {expected_flag}"
        )
    metadata_path = args.metadata_path.resolve()
    config_path = args.config_path.resolve()
    runner = args.runner.resolve()
    runner_hash = _sha256_bytes(runner.read_bytes()) if runner.is_file() else "missing"
    hash_payload = {
        **RUN_CONFIG,
        "case_id": CASE_ID,
        "variant": args.variant,
        "ablation_flag": args.ablation_flag,
        "full_runner_sha256": runner_hash,
    }
    encoded = json.dumps(
        hash_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    config_hash = _sha256_bytes(encoded)

    output_path = args.output_path.resolve()
    if args.status == "failed":
        output_validation = {
            "status": "not_checked_run_failed",
            "valid": False,
            "path": str(output_path),
        }
    else:
        output_validation = validate_video(
            output_path,
            dry_run=args.status == "dry_run",
            ffprobe_bin=args.ffprobe_bin,
        )
    if args.status == "failed":
        smoke_checks = {
            "status": "not_checked_run_failed",
            "valid": False,
            "checks": {},
        }
    else:
        smoke_checks = smoke_check_log(
            args.variant,
            args.log_path.resolve(),
            dry_run=args.status == "dry_run",
        )
    final_status = args.status
    final_exit_code = args.exit_code
    if args.status == "success" and not output_validation["valid"]:
        final_status = "failed"
        final_exit_code = args.exit_code or 3
    if (
        args.status == "success"
        and output_validation["valid"]
        and not smoke_checks["valid"]
    ):
        final_status = "failed"
        final_exit_code = args.exit_code or 4

    payload = {
        "schema_version": SCHEMA_VERSION,
        "case_id": CASE_ID,
        "variant": args.variant,
        "variant_name": args.variant_name,
        "ablation_flag": args.ablation_flag,
        "command": _resolved_command(
            config_path, runner, args.ablation_flag
        ),
        "config_hash": config_hash,
        "config_hash_source": "canonical_run_config_and_full_runner",
        **RUN_CONFIG,
        "full_runner_sha256": runner_hash,
        "commit": _git_commit(args.repo_root.resolve()),
        "runtime_seconds": round(args.runtime_seconds, 6),
        "status": final_status,
        "exit_code": final_exit_code,
        "started_at": args.started_at,
        "ended_at": args.ended_at or _utc_now(),
        "artifacts": {
            "output": str(output_path),
            "roles": str(args.roles_path.resolve()),
            "resolved_config": str(config_path),
            "run_log": str(args.log_path.resolve()),
        },
        "output_validation": output_validation,
        "smoke_checks": smoke_checks,
    }
    _atomic_write(
        metadata_path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return metadata_path


def _load_variant_metadata(root: Path, variant: str) -> dict[str, Any] | None:
    variant_name = next(name for key, name, _ in VARIANTS if key == variant)
    path = root / "configs" / f"{variant}_{variant_name}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("case_id") != CASE_ID
        or payload.get("variant") != variant
    ):
        return None
    return payload


def build_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_case: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for variant, name, flag in VARIANTS:
        metadata = _load_variant_metadata(root, variant)
        status = metadata.get("status", "not_run") if metadata else "not_run"
        common = {
            "variant": variant,
            "ablation": name,
            "ablation_flag": flag,
            "status": status,
            "config_hash": metadata.get("config_hash", "") if metadata else "",
            "commit": metadata.get("commit", "") if metadata else "",
            "runtime_seconds": (
                metadata.get("runtime_seconds", "") if metadata else ""
            ),
            "output_validation_status": (
                metadata.get("output_validation", {}).get("status", "")
                if metadata
                else ""
            ),
            "smoke_status": (
                metadata.get("smoke_checks", {}).get("status", "")
                if metadata
                else ""
            ),
            **{metric: "" for metric in METRIC_COLUMNS},
            "historical_rows_merged": "false",
            "historical_exclusion_reason": (
                "historical_full_baseline_metadata_mismatch"
            ),
        }
        per_case.append({"case_id": CASE_ID, **common})
        summary.append(
            {
                "variant": variant,
                "ablation": name,
                "status": status,
                "successful_cases": 1 if status == "success" else 0,
                "total_cases": 1,
                "output_validation_status": (
                    metadata.get("output_validation", {}).get("status", "")
                    if metadata
                    else ""
                ),
                "smoke_status": (
                    metadata.get("smoke_checks", {}).get("status", "")
                    if metadata
                    else ""
                ),
                **{metric: "" for metric in METRIC_COLUMNS},
                "historical_rows_merged": "false",
                "historical_exclusion_reason": (
                    "historical_full_baseline_metadata_mismatch"
                ),
            }
        )
    return per_case, summary


def _csv_text(rows: list[dict[str, Any]]) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _display(value: Any) -> str:
    return "" if value is None else str(value)


def _markdown_text(rows: list[dict[str, Any]], title: str) -> str:
    fields = list(rows[0])
    lines = [f"# {title}", "", "| " + " | ".join(fields) + " |"]
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        values = [_display(row[field]).replace("|", "\\|") for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", f"> {HISTORY_EXCLUSION}", ""])
    return "\n".join(lines)


def _tex_escape(value: Any) -> str:
    text = _display(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _tex_text(rows: list[dict[str, Any]]) -> str:
    fields = list(rows[0])
    columns = "l" * len(fields)
    lines = [
        "% " + HISTORY_EXCLUSION,
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(_tex_escape(field) for field in fields) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(_tex_escape(row[field]) for field in fields) + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _write_formats(base: Path, rows: list[dict[str, Any]], title: str) -> None:
    _atomic_write(base.with_suffix(".csv"), _csv_text(rows))
    _atomic_write(base.with_suffix(".md"), _markdown_text(rows, title))
    _atomic_write(base.with_suffix(".tex"), _tex_text(rows))


def finalize(root: Path) -> list[Path]:
    root = root.resolve()
    per_case, summary = build_rows(root)
    if not any(row["status"] == "success" for row in per_case):
        raise ValueError("no successful B1-B4 metadata found; refusing to finalize")

    metrics_dir = root / "metrics"
    outputs = [
        metrics_dir / "per_case_new_ablations.csv",
        metrics_dir / "per_case_new_ablations.md",
        metrics_dir / "per_case_new_ablations.tex",
        metrics_dir / "summary_new_ablations.csv",
        metrics_dir / "summary_new_ablations.md",
        metrics_dir / "summary_new_ablations.tex",
        root / "main_ablation_table.csv",
        root / "main_ablation_table.md",
        root / "main_ablation_table.tex",
    ]
    _write_formats(
        metrics_dir / "per_case_new_ablations",
        per_case,
        "Per-case new main ablations",
    )
    _write_formats(
        metrics_dir / "summary_new_ablations",
        summary,
        "Summary of new main ablations",
    )
    _write_formats(
        root / "main_ablation_table",
        summary,
        "Main ablation table",
    )
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    record = subparsers.add_parser("record", help="write one variant metadata file")
    record.add_argument("--root", type=Path, required=True)
    record.add_argument("--repo-root", type=Path, required=True)
    record.add_argument("--variant", choices=[item[0] for item in VARIANTS], required=True)
    record.add_argument("--variant-name", required=True)
    record.add_argument("--ablation-flag", required=True)
    record.add_argument(
        "--status", choices=("success", "failed", "dry_run"), required=True
    )
    record.add_argument("--exit-code", type=int, required=True)
    record.add_argument("--started-at", required=True)
    record.add_argument("--ended-at", required=True)
    record.add_argument("--runtime-seconds", type=float, required=True)
    record.add_argument("--runner", type=Path, required=True)
    record.add_argument("--metadata-path", type=Path, required=True)
    record.add_argument("--output-path", type=Path, required=True)
    record.add_argument("--roles-path", type=Path, required=True)
    record.add_argument("--config-path", type=Path, required=True)
    record.add_argument("--log-path", type=Path, required=True)
    record.add_argument("--ffprobe-bin", default="ffprobe")

    finish = subparsers.add_parser("finalize", help="write CSV, Markdown, and TeX tables")
    finish.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "record":
        metadata_path = record_metadata(args)
        print(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if args.status == "success" and metadata["status"] != "success":
            return int(metadata["exit_code"])
    else:
        for output in finalize(args.root):
            print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
