from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]


def load_attention_module():
    spec = importlib.util.spec_from_file_location(
        "streamedit_responsibility_attention",
        ROOT / "wan/modules/attention.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pipeline_diagnostic_helpers():
    source = (ROOT / "pipeline/edit_causal_inference.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    pipeline_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "EditCausalInferencePipeline"
    )
    wanted = {
        "_responsibility_cpu_tree",
        "_aggregate_responsibility_layer_maps",
        "_save_responsibility_diagnostics",
    }
    functions = []
    for node in pipeline_class.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            node.decorator_list = []
            functions.append(node)
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)

    class Helpers:
        pass

    namespace = {
        "torch": torch,
        "os": os,
        "EditCausalInferencePipeline": Helpers,
    }
    exec(compile(module, "diagnostic_helpers", "exec"), namespace)
    for name in wanted:
        setattr(Helpers, name, staticmethod(namespace[name]))
    return Helpers


def run_closed_loop(*, return_maps=False):
    attention = load_attention_module()
    native = torch.ones(1, 2, 1, 2)
    query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    source_value = torch.zeros_like(query)
    current_target = torch.tensor(
        [[[[0.5, 0.0]], [[0.0, 1.0]]]]
    )
    canonical_delta = torch.tensor(
        [[[[2.0, 0.0]], [[0.0, 3.0]]]]
    )
    return attention.closed_loop_delta_v_memory_attention(
        native_output=native,
        current_source_query=query,
        current_source_key=query,
        current_source_value=source_value,
        current_target_value=current_target,
        canonical_source_key=query,
        canonical_delta_value=canonical_delta,
        canonical_support=torch.ones(1, 2, dtype=torch.bool),
        owner_gate=torch.tensor([[1.0, 0.5]]),
        topk=1,
        min_similarity=0.0,
        strength=0.4,
        max_error_ratio=10.0,
        return_maps=return_maps,
    )


def test_closed_loop_default_contract_is_unchanged():
    result = run_closed_loop()
    assert len(result) == 2


def test_closed_loop_optional_maps_are_exact_per_token_values():
    output, diagnostics, maps = run_closed_loop(return_maps=True)

    assert set(maps) == {
        "admitted",
        "gate",
        "correction_rms",
        "native_rms",
        "joint_similarity",
        "reference_response_rms",
        "current_response_rms",
        "response_discrepancy_rms",
        "reference_topk_index",
        "reference_topk_similarity",
        "current_topk_index",
        "current_topk_similarity",
    }
    torch.testing.assert_close(maps["admitted"], torch.ones(1, 2))
    torch.testing.assert_close(maps["gate"], torch.tensor([[0.4, 0.2]]))
    torch.testing.assert_close(
        maps["native_rms"], torch.ones(1, 2)
    )
    torch.testing.assert_close(
        maps["correction_rms"],
        (output - torch.ones_like(output)).square().mean((2, 3)).sqrt(),
    )
    torch.testing.assert_close(
        diagnostics["applied_correction_rms"],
        maps["correction_rms"].mean(),
    )
    assert maps["reference_response_rms"].shape == maps["admitted"].shape
    assert maps["current_response_rms"].shape == maps["admitted"].shape
    assert maps["response_discrepancy_rms"].shape == maps["admitted"].shape
    assert torch.isfinite(maps["response_discrepancy_rms"]).all()
    assert maps["reference_topk_index"].dtype == torch.long
    assert maps["current_topk_index"].dtype == torch.long
    assert maps["reference_topk_index"].shape == maps["reference_topk_similarity"].shape
    assert maps["current_topk_index"].shape == maps["current_topk_similarity"].shape


def test_layer_aggregation_and_artifact_save_force_float_cpu(tmp_path):
    helpers = load_pipeline_diagnostic_helpers()
    layer_maps = {
        8: [
            {"gate": torch.tensor([[0.2, 0.4]])},
            {"gate": torch.tensor([[0.6, 0.8]])},
        ],
        12: [
            {"gate": torch.tensor([[0.4, 0.6]])},
            {"gate": torch.tensor([[0.8, 1.0]])},
        ],
    }
    aggregated = helpers._aggregate_responsibility_layer_maps(layer_maps)
    torch.testing.assert_close(
        aggregated["mean"]["gate"],
        torch.tensor([[0.3, 0.5], [0.7, 0.9]]),
    )

    artifact = {
        "schema_version": "ours-responsibility-v1",
        "block_index": 7,
        "latent_frame_indices": torch.tensor([3, 4]),
        "valid_token": torch.ones(1, 2, 3, 4, dtype=torch.bool),
        "metadata": {"variable_sources": {"x": "unit"}},
        "integer_tensor": torch.tensor([1, 2], dtype=torch.int64),
        "nested": aggregated,
    }
    helpers._save_responsibility_diagnostics(tmp_path, 7, artifact)
    path = tmp_path / "responsibility" / "block_007.pt"
    saved = torch.load(path, weights_only=False)

    assert path.is_file()
    assert saved["integer_tensor"].dtype == torch.float32
    assert saved["integer_tensor"].device.type == "cpu"
    assert not saved["integer_tensor"].requires_grad
    assert saved["schema_version"] == "ours-responsibility-v1"
    assert saved["block_index"] == 7
    assert saved["latent_frame_indices"].dtype == torch.int64
    assert saved["valid_token"].dtype == torch.bool
    assert saved["metadata"]["variable_sources"]["x"] == "unit"


def test_cli_pipeline_and_model_diagnostic_wiring():
    cli = (ROOT / "inference_edit_streamedit.py").read_text(
        encoding="utf-8"
    )
    pipeline = (ROOT / "pipeline/edit_causal_inference.py").read_text(
        encoding="utf-8"
    )
    model = (ROOT / "wan/modules/causal_model.py").read_text(
        encoding="utf-8"
    )

    assert '"--responsibility_diagnostics_dir"' in cli
    assert "args.responsibility_diagnostics_dir" in cli
    assert pipeline.count(
        "responsibility_diagnostics_dir: Optional[str] = None"
    ) == 2
    assert "responsibility_diagnostics_dir=None" in pipeline
    assert '"global_latent_frame_indices"' in pipeline
    assert '"schema_version": "ours-responsibility-v1"' in pipeline
    assert '"block_index": int(responsibility_block_index)' in pipeline
    assert '"latent_frame_indices"' in pipeline
    assert '"token_spatial_shape"' in pipeline
    assert '"velocity_spatial_shape"' in pipeline
    assert '"valid_token"' in pipeline
    assert '"variable_sources"' in pipeline
    assert '"source_variable_paths"' in pipeline
    assert '"prediction_indices"' in pipeline
    assert '"timesteps"' in pipeline
    assert '"object_posterior_prior"' in pipeline
    assert '"refined_object_posterior"' in pipeline
    assert '"safe_residual_coefficient"' in pipeline
    assert '"full_residual_coefficient"' in pipeline
    assert '"effective_background_gated_coefficient"' in pipeline
    assert '"m2_canonical_read_gate"' in pipeline
    assert '"m2_canonical_write_gate"' in pipeline
    assert '"final_attention_request"' in pipeline
    assert '"uniform_base_scalar"' in pipeline
    assert '"spatial_blender_rate"' in pipeline
    canonical_commit = pipeline.index(
        "canonical_commit_diagnostics = ("
    )
    bank_freeze = pipeline.index(
        "freeze_diagnostics = immutable_delta_v_state.freeze("
    )
    bank_refresh = pipeline.index(
        'responsibility_artifact["m2"].update({'
    )
    artifact_save = pipeline.index(
        "self._save_responsibility_diagnostics("
    )
    assert canonical_commit < bank_refresh < artifact_save
    assert bank_freeze < bank_refresh < artifact_save
    assert model.count("return_maps=bool(") == 2
    assert model.count('"responsibility_m2_maps"') == 2

