from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "visualize_single_case_diagnostics.py"
RUNNER_PATH = REPO_ROOT / "run_ours_single_case_diagnostics.sh"
PIPELINE_PATH = REPO_ROOT / "pipeline" / "edit_causal_inference.py"


def test_rollout_forwards_mechanism_diagnostics_to_active_chunk_inference():
    tree = ast.parse(PIPELINE_PATH.read_text(encoding="utf-8"))
    active_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Tuple)
            and len(target.elts) == 2
            and isinstance(target.elts[1], ast.Name)
            and target.elts[1].id == "rollout_latent"
            and isinstance(node.value, ast.Call)
        ):
            continue
        active_calls.append(node.value)
    assert len(active_calls) == 1
    forwarded = {keyword.arg for keyword in active_calls[0].keywords}
    assert {
        "mechanism_diagnostics_dir",
        "mechanism_diagnostics_block",
        "mechanism_diagnostics_latent_frame",
        "mechanism_diagnostics_steps",
        "mechanism_diagnostics_query_indices",
    } <= forwarded


def test_mechanism_config_records_real_chunk_latent_frame_indices():
    source = PIPELINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    matching_dicts = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "mechanism_config"
            for target in node.targets
        ) or not isinstance(node.value, ast.Dict):
            continue
        keys = {
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if "output_dir" in keys and "grid_shape" in keys:
            matching_dicts.append(keys)
    assert len(matching_dicts) == 1
    assert "latent_frame_indices" in matching_dicts[0]


def load_module():
    name = "visualize_single_case_diagnostics_standalone"
    spec = importlib.util.spec_from_file_location(name, TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


viz = load_module()


def test_fast_heatmap_is_true_rgb_and_preserves_declared_color_scale(tmp_path):
    output = tmp_path / "heatmap.png"
    viz._save_heatmap(
        output,
        np.asarray([[0.0, 0.5, 1.0]], dtype=np.float32),
        "color test",
        0.0,
        1.0,
        cmap="turbo",
    )
    image = np.asarray(viz.Image.open(output).convert("RGB"))
    assert image.shape[-1] == 3
    assert np.any(image[..., 0] != image[..., 1])
    assert np.any(image[..., 1] != image[..., 2])


def make_responsibility_block(*, with_anchor: bool = True) -> dict:
    frames, height, width = 2, 2, 3
    common = (1, frames, height, width)
    q_object = torch.full(common, 0.15)
    q_interface = torch.full(common, 0.10)
    q_hand = torch.full(common, 0.05)
    q_background = torch.full(common, 0.70)
    q_interface[0, 1, 0, 1] = 0.65
    q_background[0, 1, 0, 1] = 0.15
    connected = torch.zeros(common)
    connected[0, 1, 0, 1] = 1.0
    hand = torch.zeros(common)
    hand[0, 1, 0, 1] = 0.8
    base = torch.arange(frames * 3 * height * width, dtype=torch.float32).reshape(
        1, frames, 3, height, width
    )
    velocity_step0 = {
        name: base * (0.01 * index) + index
        for index, name in enumerate(viz.responsibility_viz.VELOCITY_VECTOR_FIELDS, start=1)
    }
    velocity_predictions = {
        prediction: {
            **{
                name: base * (0.01 * field_index) + update_index + field_index
                for field_index, name in enumerate(viz.VELOCITY_VECTOR_FIELDS, start=1)
            },
            "noise_level": torch.tensor((0.9, 0.8, 0.4)[update_index]),
        }
        for update_index, (prediction, _) in enumerate(viz.VELOCITY_UPDATES)
    }
    flat = frames * height * width
    appearance = {
        f"prediction_{index:03d}": {
            "source": "synthetic-real-tensor",
            "map": {
                "layers": torch.tensor([0.0]),
                "by_layer": {"layer_00": torch.full((1, flat), 0.2 + index / 100.0)},
                "mean": torch.full((1, flat), 0.2 + index / 100.0),
            },
        }
        for index in (0, 7, 14)
    }
    retrieval = {}
    if with_anchor:
        retrieval["prediction_014"] = {
            "layers": torch.tensor([0.0]),
            "by_layer": {},
            "mean": {
                "admitted": torch.full(common, 0.8),
                "gate": torch.full(common, 0.6),
                "correction_rms": torch.full(common, 0.15),
                "native_rms": torch.full(common, 0.3),
                "joint_similarity": torch.full(common, 0.7),
                "reference_response_rms": torch.full(common, 0.4),
                "reference_topk_index": torch.arange(flat * 2).reshape(1, flat, 2),
                "reference_topk_similarity": torch.full((1, flat, 2), 0.75),
                "current_topk_index": torch.arange(flat * 2).reshape(1, flat, 2),
                "current_topk_similarity": torch.full((1, flat, 2), 0.55),
            },
        }
    compact_anchor = {
        "layer_00": {
            "stored_token_indices": torch.tensor([0, 3, 7]),
            "stored_source_keys": torch.arange(12, dtype=torch.float32).reshape(3, 2, 2),
            "stored_target_minus_source_values": torch.arange(12, dtype=torch.float32).reshape(3, 2, 2) / 10,
            "current_clean_source_queries": torch.arange(12, dtype=torch.float32).reshape(3, 2, 2) / 20,
        }
    } if with_anchor else None
    return {
        "schema_version": "ours-responsibility-v1",
        "block_index": 0,
        "latent_frame_indices": torch.tensor([0, 1]),
        "token_spatial_shape": (height, width),
        "metadata": {"timesteps": [900, 500, 100], "fixture": True},
        "initial": {
            "source_attention": torch.full(common, 0.4),
            "hand_occupancy": hand,
            "hand_proximity": torch.full(common, 0.3),
            "temporal_confidence": torch.full(common, 0.9),
            "temporal_posterior": torch.full(common, 0.4),
            "connected_support": connected,
            "object_posterior": torch.full(common, 0.25),
        },
        "first_response": {
            "object_posterior_prior": torch.full(common, 0.3),
            "field_score": torch.full(common, 0.5),
            "field_observation": torch.full(common, 0.45),
            "refined_object_posterior": torch.full(common, 0.4),
            "q_object": q_object,
            "q_interface": q_interface,
            "q_hand": q_hand,
            "q_background": q_background,
            "role_entropy": torch.full(common, 0.2),
        },
        "velocity_step0": {
            **velocity_step0,
            "safe_residual_coefficient": torch.full(common, 0.25),
            "full_residual_coefficient": torch.full(common, 0.65),
            "effective_background_gated_coefficient": torch.full(common, 0.55),
        },
        "velocity_predictions": velocity_predictions,
        "appearance_access": appearance,
        "m2": {
            "final_attention_request": torch.full((1, flat), 0.4),
            "m2_canonical_read_gate": torch.full(common, 0.4),
            "m2_canonical_write_gate": torch.full(common, 0.2),
            "write_eligibility": torch.full(common, 0.2),
            "selected_write_tokens": torch.full((1, flat), 0.2),
            "bank_initialized": torch.tensor(1.0),
            "bank_initialized_before_block": torch.tensor(1.0),
            "bank_initialization_chunk": torch.tensor(0.0),
            "retrieval_maps": retrieval,
            **({"compact_anchor_state": compact_anchor} if compact_anchor is not None else {}),
        },
    }


def save_fixture(tmp_path: Path, *, with_anchor: bool = True):
    responsibility = tmp_path / "responsibility_input" / "responsibility"
    responsibility.mkdir(parents=True)
    block = make_responsibility_block(with_anchor=with_anchor)
    torch.save(block, responsibility / "block_000.pt")

    attention = tmp_path / "attention_input" / "raw_attention"
    for kind in ("cross", "self"):
        (attention / kind / "step_000").mkdir(parents=True)
    cross = torch.linspace(0.01, 0.99, 2 * 12 * 4).reshape(2, 12, 4)
    cross_payload = {
        "raw_attention": cross,
        "metadata": {
            "step_index": 0,
            "layer_index": 0,
            "tensor_layout": "heads_query_key",
            "query_spatial_shape": [2, 2, 3],
            "latent_frame_indices": [0, 1],
            "selected_key_index": 2,
            "selected_key_label": "red mug",
        },
    }
    torch.save(cross_payload, attention / "cross" / "step_000" / "layer_00.pt")
    (attention / "cross" / "step_007").mkdir(parents=True)
    torch.save(
        {
            **cross_payload,
            "raw_attention": cross * 0.5,
            "metadata": {**cross_payload["metadata"], "step_index": 7},
        },
        attention / "cross" / "step_007" / "layer_00.pt",
    )

    raw_self = torch.softmax(
        torch.arange(2 * 3 * 18, dtype=torch.float32).reshape(2, 3, 18) / 50,
        dim=-1,
    )
    self_payload = {
        "raw_attention": raw_self,
        "metadata": {
            "step_index": 0,
            "layer_index": 0,
            "tensor_layout": "heads_query_key",
            "queries": [
                {"name": "interface_query", "index": 0, "original_query_index": 7},
                {"name": "object_query", "index": 1, "original_query_index": 6},
                {"name": "hand_query", "index": 2, "original_query_index": 8},
            ],
            "current_key_range": [6, 18],
            "current_key_spatial_shape": [2, 2, 3],
            "history_key_ranges": [
                {"name": "target_history", "range": [0, 3]},
                {"name": "source_history", "range": [3, 6]},
            ],
            "history_frame_ranges": [
                {"name": "target_history_frame_000", "range": [0, 3]},
            ],
            "history_chunk_ranges": [
                {"name": "previous_chunk_000", "range": [0, 3]},
            ],
        },
    }
    torch.save(self_payload, attention / "self" / "step_000" / "layer_00.pt")
    (attention / "self" / "step_007").mkdir(parents=True)
    torch.save(
        {
            **self_payload,
            "raw_attention": raw_self * 0.5,
            "metadata": {**self_payload["metadata"], "step_index": 7},
        },
        attention / "self" / "step_007" / "layer_00.pt",
    )

    video_paths = []
    for name in ("source.mp4", "edited.mp4", "baseline.mp4"):
        path = tmp_path / name
        path.write_bytes((name + "-fixture").encode())
        video_paths.append(path)

    hand_role_input = tmp_path / "edited.hand_role_input.npz"
    np.savez_compressed(
        hand_role_input,
        causal_temporal_groups=np.asarray(((0, 1), (1, 5)), dtype=np.int64),
    )
    selection = {
        "selection_rule": "fixture evidence only",
        "block_index": 0,
        "global_latent_frame": 1,
        "local_frame": 1,
        "token_grid": [2, 3],
        "queries": {
            "interface_query": {"query_index": 7, "local_frame": 1, "row": 0, "column": 1},
            "object_query": {"query_index": 6, "local_frame": 1, "row": 0, "column": 0},
            "hand_query": {"query_index": 8, "local_frame": 1, "row": 0, "column": 2},
        },
        "query_indices": [7, 6, 8],
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    manifest = {
        "schema_version": "ours-single-case-run-manifest-v1",
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "case": {
            "source_video": str(video_paths[0].resolve()),
            "hand_mask_video": "/data/hand_mask.mp4",
            "source_prompt": "a white mug",
            "target_prompt": "a red plate",
            "source_phrase": "white mug",
            "target_phrase": "red plate",
            "seed": 0,
            "steps": 15,
            "rollout_chunk_size": 21,
            "checkpoint_path": "/models/self_forcing_dmd.pt",
            "config_path": "/repo/configs/self_forcing_dmd.yaml",
            "wan_models_root": "/models/wan",
        },
        "baseline": {
            "resolved_config": "/artifacts/baseline/full_resolved_config.txt",
            "command": "python inference_edit_streamedit.py --step 15 --seed 0",
        },
        "capture": {
            "resolved_config": "/artifacts/run/full_resolved_config.txt",
            "command": "python inference_edit_streamedit.py --step 15 --seed 0 --mechanism_diagnostics_steps 0 1 7",
        },
        "diagnostic_updates": {"T0": 0, "T1": 1, "Tm": 7},
    }
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return (
        responsibility.parent,
        attention.parent,
        video_paths,
        hand_role_input,
        selection_path,
        manifest_path,
        cross,
        raw_self,
    )


def test_full_synthetic_fixture_preserves_raw_tensors_and_output_contract(tmp_path, monkeypatch):
    responsibility, attention, videos, hand_input, selection, manifest, raw_cross, raw_self = save_fixture(tmp_path)
    frames = np.arange(5 * 24 * 32 * 3, dtype=np.uint8).reshape(5, 24, 32, 3)
    monkeypatch.setattr(viz, "read_video", lambda path: viz.responsibility_viz.Video(frames, 12.0))
    output = tmp_path / "output"
    # Existing non-owned content is permitted because the runner stages raw
    # artifacts below the same root.
    (output / "baseline").mkdir(parents=True)
    args = argparse.Namespace(
        responsibility_dir=responsibility,
        attention_dir=attention,
        source_video=videos[0],
        edited_video=videos[1],
        baseline_video=videos[2],
        selection=selection,
        hand_role_input=hand_input,
        run_manifest=manifest,
        output_dir=output,
    )

    paths = viz.run(args)

    assert all(path.is_file() for path in paths.values())
    assert (output / "ours_mechanism_dashboard.png").is_file()
    assert (output / "ours_mechanism_dashboard.pdf").is_file()
    assert (output / "edited_video.mp4").read_bytes() == videos[1].read_bytes()
    assert (output / "responsibility/q_interface.png").is_file()
    assert (output / "permissions/reference_write_gate.png").is_file()
    assert (output / "interface_token_card.png").is_file()
    assert (output / "interface_token_card.json").is_file()
    for label in ("T0", "T1", "Tm"):
        update = output / f"update_{label}"
        for name in viz.DERIVED_VELOCITY_FORMULAS:
            assert (update / f"{name}.png").is_file()
        assert (update / "velocity_joint_pca.png").is_file()
        assert (update / "velocity_pca_interface_zoom.png").is_file()
        assert (update / "pca_metadata.json").is_file()
    assert (output / "cross_attention/step_000/layer_00/raw_scale/head_00.png").is_file()
    assert (output / "cross_attention/step_007/layer_00/normalized_for_display/montage_all_heads.png").is_file()
    self_dir = output / "self_attention/step_000/layer_00/query_interface_query"
    assert (self_dir / "raw_scale/head_mean.png").is_file()
    assert (self_dir / "normalized_for_display/head_std.png").is_file()
    assert not list((output / "self_attention").glob("**/*history*.png"))
    assert (output / "anchor/correction_by_chunk.png").is_file()
    raw_anchor = torch.load(output / "anchor/raw_anchor.pt", weights_only=False)
    torch.testing.assert_close(
        raw_anchor["compact_anchor_state"]["layer_00"]["stored_target_minus_source_values"],
        make_responsibility_block()["m2"]["compact_anchor_state"]["layer_00"]["stored_target_minus_source_values"],
    )
    assert tuple(
        raw_anchor["topk_match_tables"]["prediction_014"]["current_topk_index"].shape
    ) == (12, 2)
    assert (
        output
        / "anchor/retrieval_prediction_014_current_topk_similarity_mean_over_k.png"
    ).is_file()

    copied_cross = torch.load(
        output / "cross_attention/step_000/layer_00/raw_attention.pt", weights_only=False
    )
    copied_self = torch.load(
        output / "self_attention/step_000/layer_00/raw_attention.pt", weights_only=False
    )
    torch.testing.assert_close(copied_cross["raw_attention"], raw_cross)
    torch.testing.assert_close(copied_self["raw_attention"], raw_self)

    history = json.loads((self_dir / "history_mass.json").read_text())
    np.testing.assert_allclose(
        history["current_chunk"]["total_mass_per_head"],
        raw_self[:, 0, 6:18].sum(dim=-1).numpy(),
    )
    np.testing.assert_allclose(
        history["target_history"]["total_mass_per_head"],
        raw_self[:, 0, 0:3].sum(dim=-1).numpy(),
    )
    np.testing.assert_allclose(
        history["source_history"]["total_mass_per_head"],
        raw_self[:, 0, 3:6].sum(dim=-1).numpy(),
    )
    np.testing.assert_allclose(
        history["previous_chunks"]["total_mass_per_head"],
        raw_self[:, 0, 0:3].sum(dim=-1).numpy(),
    )
    assert history["spatial_visualization_generated"] is False

    roles = torch.load(output / "responsibility/raw_roles.pt", weights_only=False)
    assert set(roles["common_grid"]) == set(viz.responsibility_viz.ROLE_FIELDS)
    permissions = torch.load(output / "permissions/raw_permissions.pt", weights_only=False)
    assert set(permissions["permissions"]) == set(viz.PERMISSION_FIELDS)
    card = json.loads((output / "interface_token_card.json").read_text())
    assert (card["token_row"], card["token_column"]) == (0, 1)

    pca = json.loads((output / "update_Tm/pca_metadata.json").read_text())
    assert pca["input_fields"] == list(viz.VELOCITY_VECTOR_FIELDS)
    assert pca["prediction"] == "prediction_007"
    assert pca["update_label"] == "Tm"
    assert pca["arrows"] == "per-token joint-PCA v_trg to v_controlled"
    assert pca["interface_velocity_flat_indices"] == [1]
    assert set(pca["excluded_inputs"]) >= set(viz.DERIVED_VELOCITY_FORMULAS)
    derived = torch.load(output / "update_Tm/derived_maps.pt", weights_only=False)
    vectors = make_responsibility_block()["velocity_predictions"]["prediction_007"]
    expected_removed = torch.linalg.vector_norm(
        vectors["rho"][0, 1] - vectors["rho_safe"][0, 1], dim=0
    )
    torch.testing.assert_close(derived["maps"]["removed_component"], expected_removed)
    assert viz.ROLE_COLORS == {
        "q_object": "#4c78a8",
        "q_interface": "#9c6ade",
        "q_hand": "#f5857a",
        "q_background": "#9d9da1",
    }
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["selection"]["pixel_frame_range"] == [1, 5]
    assert metadata["selection"]["representative_pixel_frame"] == 2
    assert metadata["anchor"]["generated"] is True
    assert metadata["velocity"]["updates"]["Tm"]["prediction"] == "prediction_007"
    assert metadata["case_provenance"]["seed"] == 0
    assert metadata["case_provenance"]["steps"] == 15
    assert metadata["case_provenance"]["rollout_chunk_size"] == 21
    assert metadata["case_provenance"]["diagnostic_updates"] == {
        "T0": {"prediction_index": 0, "noise_level": pytest.approx(0.9)},
        "T1": {"prediction_index": 1, "noise_level": pytest.approx(0.8)},
        "Tm": {"prediction_index": 7, "noise_level": pytest.approx(0.4)},
    }
    assert metadata["attention_visualization"]["normalized_color_scale"] == [0.0, 1.0]
    assert json.loads((output / "metrics.json").read_text())["schema_version"] == viz.SCHEMA_VERSION

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        viz.run(args)


def test_strict_absence_and_cli_runner_contracts(tmp_path):
    responsibility, attention, _, hand_input, selection_path, _, _, _ = save_fixture(
        tmp_path, with_anchor=False
    )
    selected_case = viz.load_selection(selection_path)
    _, block, _, local, _ = viz.select_responsibility(responsibility, selected_case)

    raw_block, raw_path = viz._raw_responsibility_block(responsibility, 0)
    anchor, outputs = viz.render_anchor(
        raw_block, block, local, raw_path, tmp_path / "anchor_output"
    )
    assert outputs == {}
    assert anchor == {
        "available": False,
        "generated": False,
        "reason": "selected responsibility block contains neither real retrieval maps nor compact anchor state",
    }

    bad_path = attention / "raw_attention" / "cross" / "step_000" / "layer_00.pt"
    payload = torch.load(bad_path, weights_only=False)
    del payload["metadata"]["tensor_layout"]
    torch.save(payload, bad_path)
    artifact = [item for item in viz.discover_attention_artifacts(attention) if item.kind == "cross"][0]
    with pytest.raises(viz.SchemaError, match="tensor_layout"):
        viz.cross_attention_maps(artifact, 1)

    groups = viz.load_temporal_groups(hand_input)
    assert viz.selected_pixel_frame(
        groups,
        1,
        {"source": viz.responsibility_viz.Video(np.zeros((5, 1, 1, 3)), 1.0)},
    ) == (2, (1, 5))
    bad_groups = tmp_path / "bad_groups.npz"
    np.savez_compressed(bad_groups, causal_temporal_groups=np.asarray(((0, 1), (2, 5))))
    with pytest.raises(viz.SchemaError, match="contiguous"):
        viz.load_temporal_groups(bad_groups)

    runner = RUNNER_PATH.read_text(encoding="utf-8")
    assert "--mechanism_diagnostics_steps 0 1 7" in runner
    assert '--selection "$SELECTION_JSON"' in runner
    assert '--hand-role-input "$RUN_DIR/diagnostic_full.hand_role_input.npz"' in runner
    assert '--run-manifest "$RUN_MANIFEST"' in runner
    assert 'VIDEO_COMPARISON="$ARTIFACT_ROOT/decoded_video_comparison.json"' in runner
    assert 'comparison_scope' in runner
    assert 'cmp -s' not in runner
