from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "visualize_ours_responsibility.py"
RUN_SCRIPT = REPO_ROOT / "run_ours_responsibility_visualization.sh"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "visualize_ours_responsibility_standalone", TOOL_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


viz = load_module()


def make_block(*, direct_token_roles: bool = False) -> dict:
    frames = 3
    role_height, role_width = 60, 104
    token_height, token_width = 30, 52
    posterior = np.zeros(
        (4, 1, frames, role_height, role_width), dtype=np.float32
    )
    posterior[3] = 1.0
    # One exact 2x2 native patch maps to token (4,5) under the production
    # factor-two bilinear downsample.  It is deliberately pure Interface=1.
    posterior[3, 0, 1, 8:10, 10:12] = 0.0
    posterior[1, 0, 1, 8:10, 10:12] = 1.0
    prior = np.full_like(posterior, 0.25)
    common_shape = (1, frames, token_height, token_width)
    valid = np.ones(common_shape, dtype=np.uint8)
    connected = np.zeros(common_shape, dtype=np.uint8)
    connected[0, 1, 4, 5] = 1
    connected[0, 0, 1, 1] = 1
    hand = np.zeros(common_shape, dtype=np.float32)
    hand[0, 1, 4, 5] = 0.7
    hand[0, 0, 1, 1] = 0.8
    block = {
        "schema_version": viz.SCHEMA_VERSION,
        "block_index": 0,
        "latent_frame_indices": np.arange(frames, dtype=np.int64),
        "spatial_shape": np.asarray((token_height, token_width), dtype=np.int64),
        "valid_token": valid,
        "connected_object_support": connected,
        "hand_probability": hand,
        "initial_object_posterior": np.full(common_shape, 0.25, dtype=np.float32),
        "object_posterior_prior": np.full(common_shape, 0.3, dtype=np.float32),
        "object_posterior": np.full(common_shape, 0.4, dtype=np.float32),
        "field_score": np.full(common_shape, 0.5, dtype=np.float32),
        "field_observation": np.full(common_shape, 0.35, dtype=np.float32),
        "role_entropy": np.full(common_shape, 0.2, dtype=np.float32),
        "safe_residual_coefficient": np.full(common_shape, 0.1, dtype=np.float32),
        "full_residual_coefficient": np.full(common_shape, 0.6, dtype=np.float32),
        "effective_background_gated_coefficient": np.full(common_shape, 0.4, dtype=np.float32),
        "appearance_access": np.full(common_shape, 0.7, dtype=np.float32),
        "reference_read_gate": np.full(common_shape, 0.3, dtype=np.float32),
        "reference_write_gate": np.full(common_shape, 0.2, dtype=np.float32),
        "rho_magnitude": np.full(common_shape, 1.0, dtype=np.float32),
        "d_magnitude": np.full(common_shape, 2.0, dtype=np.float32),
        "rho_safe_magnitude": np.full(common_shape, 3.0, dtype=np.float32),
        "removed_magnitude": np.full(common_shape, 4.0, dtype=np.float32),
        "rho_role_magnitude": np.full(common_shape, 5.0, dtype=np.float32),
    }
    for index, name in enumerate(viz.ROLE_FIELDS):
        block[name] = posterior[index]
    if direct_token_roles:
        token_posterior = np.zeros((4, *common_shape), dtype=np.float32)
        token_posterior[3] = 1.0
        token_posterior[3, 0, 1, 4, 5] = 0.0
        token_posterior[1, 0, 1, 4, 5] = 1.0
        token_posterior[3, 0, 0, 1, 1] = 0.5
        token_posterior[1, 0, 0, 1, 1] = 0.5
        for index, name in enumerate(viz.ROLE_FIELDS):
            block[f"{name}_token_grid"] = token_posterior[index]
    return block


def make_nested_block() -> dict:
    flat = make_block()
    frames, role_height, role_width = 3, 60, 104
    common_height, common_width = 30, 52
    common = (1, frames, common_height, common_width)
    vectors = {}
    for index, name in enumerate(viz.VELOCITY_VECTOR_FIELDS, start=1):
        vectors[name] = torch.full(
            (1, frames, 2, role_height, role_width), float(index)
        )
    coefficients = {
        name: torch.full((1, frames, 1, role_height, role_width), 0.25)
        for name in viz.COEFFICIENT_FIELDS
    }
    appearance_map = torch.full(
        (1, frames * role_height * role_width), 0.7
    )
    appearance = {
        f"prediction_{index:03d}": {
            "source": "spatial_blender_rate",
            "map": {
                "layers": torch.tensor([8.0, 12.0]),
                "by_layer": {"layer_08": appearance_map.clone()},
                "mean": appearance_map.clone(),
            },
        }
        for index in (0, 7, 14)
    }
    return {
        "global_latent_frame_indices": torch.tensor([0.0, 1.0, 2.0]),
        "metadata": {"timesteps": list(range(15))},
        "initial": {
            "source_attention": torch.full(common, 0.4),
            "hand_occupancy": torch.from_numpy(flat["hand_probability"]),
            "hand_proximity": torch.full(common, 0.3),
            "temporal_confidence": torch.full(common, 0.8),
            "temporal_posterior": torch.full(common, 0.25),
            "connected_support": torch.from_numpy(flat["connected_object_support"]),
            "object_posterior": torch.from_numpy(flat["initial_object_posterior"]),
        },
        "first_response": {
            "object_posterior_prior": torch.from_numpy(flat["object_posterior_prior"]),
            "field_score": torch.from_numpy(flat["field_score"]),
            "field_observation": torch.from_numpy(flat["field_observation"]),
            "refined_object_posterior": torch.from_numpy(flat["object_posterior"]),
            **{name: torch.from_numpy(flat[name]) for name in viz.ROLE_FIELDS},
            "role_entropy": torch.full((1, frames, role_height, role_width), 0.2),
        },
        "velocity_step0": {**vectors, **coefficients},
        "appearance_access": appearance,
        "m2": {
            "final_attention_request": torch.full(
                (1, frames * common_height * common_width), 0.3
            ),
            "m2_canonical_read_gate": torch.full(common, 0.4),
            "m2_canonical_write_gate": torch.full(common, 0.2),
            "write_eligibility": torch.full(common, 0.2),
            "selected_write_tokens": torch.full(
                (1, frames * common_height * common_width), 0.2
            ),
            "bank_initialized": torch.tensor(1.0),
            "bank_initialized_before_block": torch.tensor(0.0),
            "bank_initialization_chunk": torch.tensor(0.0),
            "retrieval_maps": {},
        },
    }


def save_artifact(tmp_path: Path, block: dict) -> Path:
    raw = tmp_path / "artifacts" / "raw"
    raw.mkdir(parents=True)
    torch.save(block, raw / "block_000.pt")
    return raw.parent


def test_real_dual_grid_shape_uses_exact_router_resize(tmp_path):
    artifact_dir = save_artifact(tmp_path, make_nested_block())

    block = viz.load_blocks(artifact_dir)[0]

    assert block.raw_role_spatial_shape == (60, 104)
    assert block.spatial_shape == (30, 52)
    assert block.maps["q_interface"].shape == (3, 30, 52)
    assert block.role_alignment.startswith("PosteriorResidualFlowRouter")
    assert block.velocity_spatial_shape == (60, 104)
    assert block.maps["appearance_access"].shape == (3, 30, 52)
    assert block.maps["reference_read_gate"].shape == (3, 30, 52)
    raw = torch.from_numpy(
        np.stack([block.raw_role_maps[name] for name in viz.ROLE_FIELDS], axis=1)
    ).float()
    expected = F.interpolate(
        raw, size=(30, 52), mode="bilinear", align_corners=False
    ).clamp_min(0.0)
    expected = expected / expected.sum(dim=1, keepdim=True).clamp_min(1e-6)
    np.testing.assert_allclose(
        block.maps["q_interface"], expected[:, 1].numpy(), atol=0, rtol=0
    )


def test_saved_token_grid_is_preferred_and_pure_interface_argmax_is_honest(tmp_path):
    artifact_dir = save_artifact(tmp_path, make_block(direct_token_roles=True))
    blocks = viz.load_blocks(artifact_dir)

    selected = viz.select_interface_token(blocks)
    card = viz.token_card_values(blocks, selected)
    _, mixed = viz.role_composite(blocks[0], selected.local_frame)

    assert blocks[0].role_alignment == "hook_saved_token_grid"
    assert (selected.local_frame, selected.row, selected.column) == (1, 4, 5)
    assert selected.score == 1.0
    assert card["values"]["q_interface"] == 1.0
    assert not mixed[selected.row, selected.column]


def test_selection_requires_intersection_of_support_hand_and_valid(tmp_path):
    block = make_block(direct_token_roles=True)
    block["hand_probability"][:] = 0.0
    artifact_dir = save_artifact(tmp_path, block)

    with pytest.raises(viz.SchemaError, match="No token satisfies"):
        viz.select_interface_token(viz.load_blocks(artifact_dir))


def test_schema_rejects_missing_or_partial_semantic_fields(tmp_path):
    missing = make_block()
    del missing["object_posterior"]
    with pytest.raises(viz.SchemaError, match="object_posterior"):
        viz.validate_block(missing, tmp_path / "missing.pt")

    partial = make_block()
    partial["q_interface_token_grid"] = np.zeros((1, 3, 30, 52), dtype=np.float32)
    with pytest.raises(viz.SchemaError, match="Partial token-grid role tuple"):
        viz.validate_block(partial, tmp_path / "partial.pt")

    nested = make_nested_block()
    del nested["appearance_access"]["prediction_014"]
    with pytest.raises(viz.SchemaError, match="prediction_014"):
        viz.validate_block(nested, tmp_path / "block_000.pt")


def test_velocity_scale_is_one_pooled_run_p99(tmp_path):
    block = make_block()
    block["rho_magnitude"].fill(1.0)
    block["rho_safe_magnitude"].fill(5.0)
    block["rho_role_magnitude"].fill(9.0)
    loaded = viz.validate_block(block, tmp_path / "block.pt")

    p99, statistics = viz.velocity_statistics([loaded])

    pooled = np.concatenate(
        [viz._valid_values([loaded], name) for name in viz.VELOCITY_FIELDS]
    )
    assert p99 == pytest.approx(float(np.quantile(pooled, 0.99)))
    assert statistics["by_field"]["rho_magnitude"]["min"] == 1.0
    assert statistics["by_field"]["rho_role_magnitude"]["max"] == 9.0
    assert "p99" in statistics["pooled"]["quantiles"]


def test_region_metrics_keep_hard_argmax_and_soft_weighted(tmp_path):
    loaded = viz.validate_block(make_block(direct_token_roles=True), tmp_path / "block.pt")
    selected = viz.select_interface_token([loaded])

    metrics = viz.compute_metrics([loaded], selected)

    assert set(metrics["regions"]) == set(viz.ROLE_LABELS)
    assert metrics["region_assignment"].startswith("argmax")
    interface = metrics["regions"]["Interface"]
    # The second token is a 0.5 Interface/0.5 Background tie.  np.argmax
    # follows the declared role order and assigns it to Interface.
    assert interface["hard_argmax"]["token_count"] == 2
    assert interface["soft_weighted"]["weight_sum"] > 1.0
    assert "rho_role_magnitude" in interface["soft_weighted"]["fields"]


def test_full_output_contract_and_raw_diagnostics(tmp_path, monkeypatch):
    nested = make_nested_block()
    token = make_block(direct_token_roles=True)
    for name in viz.ROLE_FIELDS:
        nested["first_response"][f"{name}_token_grid"] = torch.from_numpy(
            token[f"{name}_token_grid"]
        )
    artifact_dir = save_artifact(tmp_path, nested)
    hand_input = tmp_path / "edited.hand_role_input.npz"
    np.savez_compressed(
        hand_input,
        causal_temporal_groups=np.asarray(((0, 1), (1, 5), (5, 9)), dtype=np.int64),
    )
    source_path = tmp_path / "source.mp4"
    edited_path = tmp_path / "edited.mp4"
    source_path.write_bytes(b"source-placeholder")
    edited_path.write_bytes(b"edited-placeholder")
    frames = np.arange(9 * 32 * 48 * 3, dtype=np.uint8).reshape(9, 32, 48, 3)
    monkeypatch.setattr(viz, "read_video", lambda path: viz.Video(frames, 16.0))
    output_dir = tmp_path / "run"
    args = argparse.Namespace(
        artifacts=artifact_dir,
        source_video=source_path,
        edited_video=edited_path,
        hand_role_input=hand_input,
        output_dir=output_dir,
        max_keyframes=3,
        crop_radius_tokens=2,
    )

    paths = viz.run(args)

    assert all(path.is_file() for path in paths.values())
    assert paths["edited_video"].read_bytes() == edited_path.read_bytes()
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["velocity_scaling"]["shared_robust_p99"] > 0
    assert metadata["blocks"][0]["native_role_spatial_shape"] == [60, 104]
    assert metadata["blocks"][0]["spatial_shape"] == [30, 52]
    assert metadata["selected_interface_token"]["values"]["q_interface"] == 1.0
    with np.load(paths["diagnostics"]) as diagnostics:
        assert diagnostics["native_q_interface"].shape == (3, 60, 104)
        assert diagnostics["q_interface"].shape == (3, 30, 52)
        assert diagnostics["native_rho_role_magnitude"].shape == (3, 60, 104)
        assert diagnostics["causal_temporal_groups"].shape == (3, 2)
    assert (output_dir / "ours_selected_keyframes").is_dir()


def test_run_script_uses_full_cook_runner_with_fixed_mug_sample():
    text = RUN_SCRIPT.read_text(encoding="utf-8")

    assert 'bash "$SCRIPT_DIR/run_cook_S1M2_full.sh"' in text
    assert "--responsibility_diagnostics_dir" in text
    assert 'ARTIFACT_ROOT="$SCRIPT_DIR/artifacts/ours_responsibility_visualization"' in text
    assert 'RUN_DIR="$ARTIFACT_ROOT/run"' in text
    assert "/mnt/bn/public-lf4/fky/checkpoints/wan_models" in text
    assert "/mnt/bn/public-lf4/fky/checkpoints/checkpoints/self_forcing_dmd.pt" in text
    assert "STEP=15" in text
    assert "S1M2_ATTENTION_MODE=full" in text
    assert 'CUDA_DEVICE="${CUDA_DEVICE:-1}"' in text
    assert '--artifacts "$RESPONSIBILITY_DIR"' in text
    assert '--output-dir "$ARTIFACT_ROOT"' in text
    assert "09_mug_white_0cf372ad96ca71749a1a81bfd330d77e.mp4" in text
    assert "A person is washing a white mug." in text
    assert "A person is washing a red plate." in text
    assert 'SRC_WORD="${SRC_WORD:-white mug}"' in text
    assert 'TRG_WORD="${TRG_WORD:-red plate}"' in text
    assert "ours_edited_video.mp4" in TOOL_PATH.read_text(encoding="utf-8")
