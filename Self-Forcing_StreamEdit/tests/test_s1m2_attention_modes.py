from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shlex
import subprocess

import pytest
import torch


ROOT = Path(__file__).parents[1]


def load_attention_module():
    spec = importlib.util.spec_from_file_location(
        "streamedit_attention_modes_standalone",
        ROOT / "wan/modules/attention.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ATTENTION = load_attention_module()


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("legacy", (False, False)),
        ("m2", (False, True)),
        ("spatial", (True, False)),
        ("full", (True, True)),
    ],
)
def test_attention_mode_feature_matrix(mode, expected):
    assert ATTENTION.resolve_s1m2_attention_features(mode) == expected


def test_unknown_attention_mode_is_rejected():
    with pytest.raises(ValueError, match="legacy, m2, spatial, or full"):
        ATTENTION.resolve_s1m2_attention_features("unknown")


def test_spatial_qk_blend_is_tokenwise_and_non_mutating():
    target_query = torch.full((2, 1, 2), 10.0)
    source_query = torch.zeros_like(target_query)
    target_key = torch.full_like(target_query, 20.0)
    source_key = torch.full_like(target_query, 4.0)
    rate = torch.tensor([0.0, 0.75])
    originals = tuple(
        tensor.clone()
        for tensor in (
            target_query, source_query, target_key, source_key
        )
    )

    query, key = ATTENTION.blend_s1m2_spatial_qk(
        target_query,
        source_query,
        target_key,
        source_key,
        rate,
    )

    torch.testing.assert_close(query[0], source_query[0])
    torch.testing.assert_close(key[0], source_key[0])
    torch.testing.assert_close(query[1], torch.full((1, 2), 7.5))
    torch.testing.assert_close(key[1], torch.full((1, 2), 16.0))
    for tensor, original in zip(
        (target_query, source_query, target_key, source_key), originals
    ):
        torch.testing.assert_close(tensor, original)


def test_spatial_qk_blend_rejects_misaligned_rates():
    tensor = torch.zeros(2, 1, 2)
    with pytest.raises(ValueError, match="align with current Q/K tokens"):
        ATTENTION.blend_s1m2_spatial_qk(
            tensor, tensor, tensor, tensor, torch.zeros(3)
        )


def test_factorized_branch_consumes_spatial_qk_and_m2_before_continue():
    source = (ROOT / "wan/modules/causal_model.py").read_text(
        encoding="utf-8"
    )
    branch = source.index("if factorized_bayes_kv:")
    fallback = source.index(
        "paired_read is None\n                                and not role_fixed_native_history",
        branch,
    )

    spatial = source.index("blend_s1m2_spatial_qk(", branch)
    m2 = source.index("closed_loop_delta_v_memory_attention(", branch)
    assert branch < spatial < fallback
    assert branch < m2 < fallback
    assert "branch_batch_size = b // 2" in source
    assert "M2 clean-source Q/K must align" in source
    assert 's1m2_call_counts["spatial"] += 1' in source
    assert 's1m2_call_counts["m2"] += 1' in source


def test_cli_pipeline_and_runtime_diagnostic_are_wired():
    cli = (ROOT / "inference_edit_streamedit.py").read_text(
        encoding="utf-8"
    )
    pipeline = (ROOT / "pipeline/edit_causal_inference.py").read_text(
        encoding="utf-8"
    )
    assert '"--s1m2_attention_mode"' in cli
    assert 'choices=["legacy", "m2", "spatial", "full"]' in cli
    assert "s1m2_attention_mode=args.s1m2_attention_mode" in cli
    assert '"--role_background_anchor_strength"' in cli
    assert '"--role_background_owner_veto_strength"' in cli
    assert '"--m2_canonical_identity_commit"' in cli
    assert (
        "role_background_anchor_strength=(\n"
        "            args.role_background_anchor_strength"
    ) in cli
    assert pipeline.count(
        "s1m2_attention_mode=s1m2_attention_mode"
    ) == 3
    assert pipeline.count("role_background_anchor_strength=(") == 3
    assert pipeline.count("role_background_owner_veto_strength=(") == 3
    assert (
        "background_anchor_strength=(\n"
        "                                        role_background_anchor_strength"
    ) in pipeline
    assert (
        "background_anchor_gate=(\n"
        "                                        None\n"
        "                                        if (\n"
        '                                            s1m2_attention_mode == "legacy"'
    ) in pipeline
    assert "else background_routing_action" in pipeline
    assert "apply_background_owner_veto(" in pipeline
    assert "_materialize_immutable_delta_v_kv(" in pipeline
    assert "build_canonical_m2_gates(" in pipeline
    assert "m2_canonical_read_tokens = None" in pipeline
    assert "owner_gate=m2_canonical_read_tokens" in pipeline
    assert "Canonical M2 write gate was not built" in pipeline
    assert "role_uncertainty\n                                * background_routing_action" in pipeline
    assert '"s1m2_attention_mode": s1m2_attention_mode' in pipeline
    assert '"S1M2_ATTENTION "' in pipeline
    assert '"IMMUTABLE_DELTA_V_FREEZE state=never_frozen "' in pipeline
    assert '"current_role_memory_read_gate"' in pipeline
    assert '"current_role_memory_read_gate"' in (
        ROOT / "wan/modules/causal_model.py"
    ).read_text(encoding="utf-8")
    assert "region_flat = (\n                                role_object_posterior_tokens" in pipeline


def test_main_ablation_switches_are_independent_and_default_off():
    cli = (ROOT / "inference_edit_streamedit.py").read_text(
        encoding="utf-8"
    )
    pipeline = (ROOT / "pipeline/edit_causal_inference.py").read_text(
        encoding="utf-8"
    )
    switches = (
        "ablation_no_interface_role",
        "ablation_no_direction_filtering",
        "ablation_no_attention_control",
        "ablation_no_appearance_anchor",
    )
    for switch in switches:
        assert f'"--{switch}"' in cli
        assert f"{switch}: bool = False" in pipeline
        assert f"{switch}=args.{switch}" in cli or (
            f"{switch}=(\n            args.{switch}" in cli
        )
    assert "merge_interface_into_object(" in pipeline
    assert "safe_source_residual = source_residual" in pipeline
    assert "and not ablation_no_attention_control" in pipeline
    assert "if ablation_no_appearance_anchor:" in pipeline


@pytest.mark.parametrize(
    ("variant", "mode"),
    [
        ("A_legacy", "legacy"),
        ("B_m2", "m2"),
        ("C_spatial", "spatial"),
        ("D_full", "full"),
    ],
)
def test_ablation_entrypoints_select_exactly_one_mode(variant, mode):
    path = ROOT / f"run_cook_S1M2_{variant}.sh"
    source = path.read_text(encoding="utf-8")
    assert path.stat().st_mode & 0o111
    assert f"S1M2_ATTENTION_MODE={mode}" in source
    assert 'exec "$SCRIPT_DIR/run_cook_S1M2_full.sh" "$@"' in source


def test_full_entrypoint_defaults_to_full_mode():
    source = (ROOT / "run_cook_S1M2_full.sh").read_text(
        encoding="utf-8"
    )
    assert 'S1M2_ATTENTION_MODE="${S1M2_ATTENTION_MODE:-full}"' in source
    assert '--s1m2_attention_mode "$S1M2_ATTENTION_MODE"' in source
    assert (
        'BACKGROUND_ANCHOR_STRENGTH="${BACKGROUND_ANCHOR_STRENGTH:-0.50}"'
        in source
    )
    assert (
        '--role_background_anchor_strength "$BACKGROUND_ANCHOR_STRENGTH"'
        in source
    )
    assert (
        '--role_background_owner_veto_strength '
        '"$BACKGROUND_OWNER_VETO_STRENGTH"'
        in source
    )
    assert '--m2_canonical_identity_commit' in source
    assert '--m2_canonical_commit_strength' in source


def test_mug_to_red_plate_entrypoint_uses_full_mode_and_simple_prompts():
    path = ROOT / "run_mug_white_to_red_plate_S1M2_full.sh"
    source = path.read_text(encoding="utf-8")
    assert path.stat().st_mode & 0o111
    assert "09_mug_white_0cf372ad96ca71749a1a81bfd330d77e.mp4" in source
    assert "_20260924_175331_321157/mask.mp4" in source
    assert (
        "WAN_MODELS_ROOT=\"${WAN_MODELS_ROOT:-/mnt/bn/public-lf4/fky/"
        "checkpoints/wan_models}\""
    ) in source
    assert (
        "CHECKPOINT_PATH=\"${CHECKPOINT_PATH:-/mnt/bn/public-lf4/fky/"
        "checkpoints/checkpoints/self_forcing_dmd.pt}\""
    ) in source
    assert 'S1M2_ATTENTION_MODE="${S1M2_ATTENTION_MODE:-full}"' in source
    assert (
        'M2_CANONICAL_IDENTITY_COMMIT="${M2_CANONICAL_IDENTITY_COMMIT:-0}"'
        in source
    )
    assert (
        'SRC_PROMPT="${MUG_SRC_PROMPT:-A person is washing a white mug.}"'
        in source
    )
    assert (
        'TRG_PROMPT="${MUG_TRG_PROMPT:-A person is washing a red plate.}"'
        in source
    )
    assert 'SRC_WORD="${MUG_SRC_WORD:-white mug}"' in source
    assert 'TRG_WORD="${MUG_TRG_WORD:-red plate}"' in source


def test_mug_response_gated_residual_entrypoint_is_isolated():
    path = ROOT / (
        "run_mug_white_to_red_plate_S1M2_full_"
        "response_gated_residual.sh"
    )
    source = path.read_text(encoding="utf-8")
    assert path.stat().st_mode & 0o111
    assert "response_gated_residual_washing" in source
    assert 'BACKGROUND_ANCHOR_STRENGTH="${BACKGROUND_ANCHOR_STRENGTH:-0.50}"' in source
    assert 'CUDA_DEVICE="${CUDA_DEVICE:-0}"' in source
    assert 'exec "$SCRIPT_DIR/run_mug_white_to_red_plate_S1M2_full.sh" "$@"' in source


def test_mug_owner_veto_entrypoint_is_isolated_and_soft():
    path = ROOT / "run_mug_white_to_red_plate_S1M2_full_owner_veto.sh"
    source = path.read_text(encoding="utf-8")
    assert path.stat().st_mode & 0o111
    assert "full_owner_veto_washing" in source
    assert (
        'BACKGROUND_OWNER_VETO_STRENGTH="${BACKGROUND_OWNER_VETO_STRENGTH:-0.75}"'
        in source
    )
    assert 'BACKGROUND_ANCHOR_STRENGTH="${BACKGROUND_ANCHOR_STRENGTH:-0.50}"' in source
    assert 'CUDA_DEVICE="${CUDA_DEVICE:-0}"' in source
    assert (
        'exec "$SCRIPT_DIR/run_mug_white_to_red_plate_S1M2_full_'
        'response_gated_residual.sh" "$@"'
        in source
    )


def test_mug_m2_identity_entrypoint_is_isolated():
    path = ROOT / "run_mug_white_to_red_plate_S1M2_full_m2_identity.sh"
    source = path.read_text(encoding="utf-8")
    assert path.stat().st_mode & 0o111
    assert "full_m2_identity_washing" in source
    assert (
        'M2_CANONICAL_IDENTITY_COMMIT="${M2_CANONICAL_IDENTITY_COMMIT:-1}"'
        in source
    )
    assert (
        'M2_CANONICAL_COMMIT_STRENGTH="${M2_CANONICAL_COMMIT_STRENGTH:-0.50}"'
        in source
    )
    assert (
        'exec "$SCRIPT_DIR/run_mug_white_to_red_plate_S1M2_full_'
        'owner_veto.sh" "$@"'
        in source
    )


def test_mug_cook_legacy_entrypoint_disables_later_stabilizers():
    path = ROOT / "run_mug_white_to_red_plate_cook_S1M2_legacy.sh"
    source = path.read_text(encoding="utf-8")
    assert path.stat().st_mode & 0o111
    assert "cook_S1M2_legacy_washing" in source
    assert "S1M2_ATTENTION_MODE=legacy" in source
    assert "BACKGROUND_ANCHOR_STRENGTH=0.0" in source
    assert "BACKGROUND_OWNER_VETO_STRENGTH=0.0" in source
    assert "M2_CANONICAL_IDENTITY_COMMIT=0" in source
    assert "A person is washing a white mug." in source
    assert "A person is washing a red plate." in source


def test_mug_spatial_only_entrypoint_keeps_full_motion_without_m2():
    path = ROOT / "run_mug_white_to_red_plate_S1_spatial_only.sh"
    source = path.read_text(encoding="utf-8")
    assert path.stat().st_mode & 0o111
    assert "S1_spatial_only_washing" in source
    assert "S1M2_ATTENTION_MODE=spatial" in source
    assert "M2_CANONICAL_IDENTITY_COMMIT=0" in source
    assert "BACKGROUND_ANCHOR_STRENGTH" in source
    assert "BACKGROUND_OWNER_VETO_STRENGTH" in source


def test_mug_legacy_attention_full_motion_entrypoint_is_factorized():
    path = ROOT / (
        "run_mug_white_to_red_plate_S1_legacy_attention_full_motion.sh"
    )
    source = path.read_text(encoding="utf-8")
    assert path.stat().st_mode & 0o111
    assert "S1_legacy_attention_full_motion_washing" in source
    assert "S1M2_ATTENTION_MODE=legacy" in source
    assert "M2_CANONICAL_IDENTITY_COMMIT=0" in source
    assert 'BACKGROUND_ANCHOR_STRENGTH="${BACKGROUND_ANCHOR_STRENGTH:-0.50}"' in source
    assert (
        'BACKGROUND_OWNER_VETO_STRENGTH="${BACKGROUND_OWNER_VETO_STRENGTH:-0.75}"'
        in source
    )


def test_ablation_dry_runs_differ_only_by_mode_and_output(tmp_path):
    source_video = tmp_path / "source.mp4"
    hand_mask = tmp_path / "hand.mp4"
    source_video.touch()
    hand_mask.touch()
    checkpoint = tmp_path / "self_forcing_dmd.pt"
    checkpoint.touch()
    wan_root = tmp_path / "wan_models"
    model_root = wan_root / "Wan2.1-T2V-1.3B"
    model_root.mkdir(parents=True)
    (model_root / "config.json").touch()
    commands = {}
    for variant, mode, spatial_enabled, m2_enabled in (
        ("A_legacy", "legacy", 0, 0),
        ("B_m2", "m2", 0, 1),
        ("C_spatial", "spatial", 1, 0),
        ("D_full", "full", 1, 1),
    ):
        environment = os.environ.copy()
        environment.update(
            {
                "DRY_RUN": "1",
                "DATA_PATH": str(source_video),
                "HAND_MASK": str(hand_mask),
                "OUTDIR": str(tmp_path / variant),
                "OUTPUT_NAME": "output.mp4",
                "CHECKPOINT_PATH": str(checkpoint),
                "WAN_MODELS_ROOT": str(wan_root),
            }
        )
        result = subprocess.run(
            [str(ROOT / f"run_cook_S1M2_{variant}.sh")],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        command_line = result.stdout.split(
            "DRY_RUN resolved command:\n", maxsplit=1
        )[1].strip()
        command = shlex.split(command_line)
        mode_index = command.index("--s1m2_attention_mode")
        assert command.count("--s1m2_attention_mode") == 1
        assert command[mode_index + 1] == mode
        command[mode_index + 1] = "MODE"
        has_canonical_commit = "--m2_canonical_identity_commit" in command
        assert has_canonical_commit is bool(m2_enabled)
        if has_canonical_commit:
            command.remove("--m2_canonical_identity_commit")
            canonical_strength_index = command.index(
                "--m2_canonical_commit_strength"
            )
            del command[
                canonical_strength_index : canonical_strength_index + 2
            ]
        save_index = command.index("--save_path")
        role_index = command.index("--save_role_dir")
        command[save_index + 1] = "OUTPUT"
        command[role_index + 1] = "ROLES"
        commands[variant] = command
        manifest = (
            tmp_path / variant / "cook_S1M2_config.txt"
        ).read_text(encoding="utf-8")
        assert f"s1m2_attention_mode={mode}" in manifest
        assert f"spatial_qk_enabled={spatial_enabled}" in manifest
        assert f"M2_enabled={m2_enabled}" in manifest
        assert (
            f"M2_canonical_identity_commit={m2_enabled}" in manifest
        )

    reference = commands["A_legacy"]
    assert all(command == reference for command in commands.values())


def test_wan_wrapper_supports_an_absolute_model_root():
    source = (ROOT / "utils/wan_wrapper.py").read_text(encoding="utf-8")
    assert 'os.environ.get("WAN_MODELS_ROOT", "wan_models")' in source
    assert "_wan_model_path(model_name)" in source
