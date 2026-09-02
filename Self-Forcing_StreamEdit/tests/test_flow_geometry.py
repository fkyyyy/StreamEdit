import torch
from pathlib import Path
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[1]
if "pipeline" not in sys.modules:
    pipeline_package = types.ModuleType("pipeline")
    pipeline_package.__path__ = [str(REPO_ROOT / "pipeline")]
    sys.modules["pipeline"] = pipeline_package

from pipeline.motion.flow_geometry import (
    compose_forward_flow,
    forward_splat,
    forward_backward_confidence,
    resize_flow,
    warp_with_backward_flow,
)


def test_backward_warp_moves_map_right() -> None:
    previous = torch.zeros(1, 1, 5, 6)
    previous[0, 0, 2, 1] = 1.0
    # Current x=2 samples previous x=1, so the content moves right by one.
    backward = torch.zeros(1, 2, 5, 6)
    backward[:, 0] = -1.0
    warped, valid = warp_with_backward_flow(previous, backward)
    assert torch.isclose(warped[0, 0, 2, 2], torch.tensor(1.0))
    assert not valid[0, 0, 2, 0]


def test_inverse_translation_is_forward_backward_consistent() -> None:
    forward = torch.zeros(1, 2, 8, 8)
    backward = torch.zeros_like(forward)
    forward[:, 0] = 1.0
    backward[:, 0] = -1.0
    confidence, occlusion, error = forward_backward_confidence(
        forward, backward
    )
    assert torch.allclose(error[..., :, :-1], torch.zeros_like(error[..., :, :-1]))
    assert torch.allclose(confidence[..., :, :-1], torch.ones_like(confidence[..., :, :-1]))
    assert not occlusion[..., :, :-1].any()
    assert occlusion[..., :, -1].all()


def test_flow_resize_scales_pixel_displacement() -> None:
    flow = torch.ones(1, 2, 4, 8)
    resized = resize_flow(flow, (8, 4))
    assert resized.shape == (1, 2, 8, 4)
    assert torch.allclose(resized[:, 0], torch.full_like(resized[:, 0], 0.5))
    assert torch.allclose(resized[:, 1], torch.full_like(resized[:, 1], 2.0))


def test_constant_flow_composition_adds_displacements() -> None:
    flow_ab = torch.zeros(1, 2, 8, 8)
    flow_bc = torch.zeros_like(flow_ab)
    flow_ab[:, 0] = 1.0
    flow_bc[:, 1] = 2.0
    composed, valid = compose_forward_flow(flow_ab, flow_bc)
    assert torch.allclose(composed[:, 0, :, :-1], torch.ones_like(composed[:, 0, :, :-1]))
    assert torch.allclose(composed[:, 1, :, :-1], torch.full_like(composed[:, 1, :, :-1], 2.0))
    assert valid[..., :, :-1].all()


def test_forward_splat_moves_map_right() -> None:
    value = torch.zeros(1, 1, 5, 6)
    value[0, 0, 2, 1] = 1.0
    forward = torch.zeros(1, 2, 5, 6)
    forward[:, 0] = 1.0
    moved, coverage = forward_splat(value, forward)
    assert torch.isclose(moved[0, 0, 2, 2], torch.tensor(1.0))
    assert coverage[0, 0, 2, 2] > 0.0
    assert coverage[0, 0, 2, 0] == 0.0
