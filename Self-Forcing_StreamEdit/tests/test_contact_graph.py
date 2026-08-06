import importlib.util
from pathlib import Path
import sys
import types

import torch


ROOT = Path(__file__).parents[1]
pipeline_package = types.ModuleType("pipeline")
pipeline_package.__path__ = [str(ROOT / "pipeline")]
sys.modules.setdefault("pipeline", pipeline_package)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


role_router = _load(
    "pipeline.role_router",
    ROOT / "pipeline" / "role_router.py",
)
contact_graph = _load(
    "pipeline.contact_graph",
    ROOT / "pipeline" / "contact_graph.py",
)
graph_attention = _load(
    "contact_graph_attention",
    ROOT / "wan" / "modules" / "contact_graph_attention.py",
)


def _synthetic_roles():
    shape = (1, 1, 8, 8)
    boundary = torch.zeros(shape)
    hand = torch.zeros(shape)
    boundary[:, :, 2:4, 4:6] = 1.0
    boundary[:, :, 4:6, 4:6] = 1.0
    hand[:, :, 2:4, 2:4] = 1.0
    hand[:, :, 4:6, 2:4] = 1.0
    background = 1.0 - boundary - hand
    return role_router.RoleState(
        object=torch.zeros(shape),
        boundary=boundary,
        hand=hand,
        background=background,
    )


def test_oracle_contact_graph_has_bounded_same_frame_edges():
    graph = contact_graph.build_oracle_contact_graphs(
        _synthetic_roles(),
        mode="source_qk",
        topk=2,
        radius=2.5,
    )[0]

    assert graph["object_indices"].numel() == 2
    assert graph["hand_indices"].shape == (2, 2)
    assert graph["edge_valid"].all()
    assert graph["object_indices"].max() < 16
    assert graph["hand_indices"].max() < 16


def test_shuffled_graph_preserves_edge_budget_and_endpoints():
    true_graph = contact_graph.build_oracle_contact_graphs(
        _synthetic_roles(),
        mode="source_qk",
        topk=2,
        radius=2.5,
    )[0]
    shuffled_graph = contact_graph.build_oracle_contact_graphs(
        _synthetic_roles(),
        mode="shuffled",
        topk=2,
        radius=2.5,
    )[0]

    assert torch.equal(
        true_graph["object_indices"],
        shuffled_graph["object_indices"],
    )
    assert torch.equal(
        true_graph["edge_valid"],
        shuffled_graph["edge_valid"],
    )
    assert torch.equal(
        true_graph["edge_confidence"],
        shuffled_graph["edge_confidence"],
    )
    true_hand = true_graph["hand_indices"][true_graph["edge_valid"]]
    shuffled_hand = shuffled_graph["hand_indices"][
        shuffled_graph["edge_valid"]
    ]
    assert torch.equal(
        true_hand.sort().values,
        shuffled_hand.sort().values,
    )
    assert not torch.equal(true_hand, shuffled_hand)


def test_relation_residual_only_updates_object_contact_tokens():
    sequence_length = 4
    shape = (sequence_length, 1, 2)
    source_value = torch.zeros(shape)
    source_value[0:2, 0, 0] = 1.0
    graph = {
        "object_indices": torch.tensor([2]),
        "hand_indices": torch.tensor([[0, 1]]),
        "edge_confidence": torch.tensor([[1.0, 1.0]]),
        "edge_valid": torch.tensor([[True, True]]),
        "object_confidence": torch.tensor([1.0]),
    }

    output = graph_attention.apply_contact_graph_residual(
        target_output=torch.zeros((1, *shape)),
        source_query=torch.zeros(shape),
        target_query=torch.zeros(shape),
        source_key=torch.zeros(shape),
        target_key=torch.zeros(shape),
        source_value=source_value,
        target_value=torch.zeros(shape),
        graph=graph,
        mode="distance_only",
        strength=0.25,
    )

    assert torch.allclose(output[0, 2, 0], torch.tensor([0.25, 0.0]))
    assert torch.count_nonzero(output[0, :2]) == 0
    assert torch.count_nonzero(output[0, 3]) == 0


def test_no_graph_is_exact_noop_for_bfloat16():
    output = torch.randn((1, 4, 1, 2), dtype=torch.bfloat16)
    result = graph_attention.apply_contact_graph_residual(
        target_output=output,
        source_query=output[0],
        target_query=output[0],
        source_key=output[0],
        target_key=output[0],
        source_value=output[0],
        target_value=output[0],
        graph={},
        mode="no_graph",
        strength=0.0,
    )
    assert result is output
