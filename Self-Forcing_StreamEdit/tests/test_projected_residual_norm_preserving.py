import importlib.util
from pathlib import Path
import sys

import pytest
import torch


def load_appearance_leakage_module():
    path = Path(__file__).parents[1] / "pipeline" / "appearance_leakage.py"
    spec = importlib.util.spec_from_file_location(
        "streamedit_appearance_leakage_norm_standalone", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


appearance_leakage = load_appearance_leakage_module()
remove_antagonistic_source_residual = (
    appearance_leakage.remove_antagonistic_source_residual
)
restore_projected_residual_norm = (
    appearance_leakage.restore_projected_residual_norm
)


def weighted_energy(residual, weight):
    return (weight.float() * residual.float()).square().sum(dim=(2, 3, 4))


def test_restores_weighted_norm_per_sample_and_timestep():
    source = torch.ones(2, 3, 2, 2, 2)
    factors = torch.tensor(
        [[0.5, 0.25, 1.0], [0.4, 0.8, 0.4]]
    ).view(2, 3, 1, 1, 1)
    projected = source * factors
    weight = torch.ones(2, 3, 1, 2, 2)
    weight[..., 0, 0] = 0.25

    output, diagnostics = restore_projected_residual_norm(
        source, projected, weight
    )

    torch.testing.assert_close(
        weighted_energy(output, weight),
        weighted_energy(source, weight),
    )
    torch.testing.assert_close(
        diagnostics["norm_scale"], factors.squeeze((-1, -2, -3)).reciprocal()
    )
    assert not diagnostics["scale_capped"].any()
    assert not diagnostics["degenerate"].any()


def test_positive_rescaling_does_not_restore_antagonistic_direction():
    source = torch.tensor([-1.0, 1.0]).view(1, 1, 2, 1, 1)
    direction = torch.tensor([1.0, 0.0]).view(1, 1, 2, 1, 1)
    core = torch.ones(1, 1, 1, 1, dtype=torch.bool)
    weight = torch.ones(1, 1, 1, 1, 1)
    projected, _ = remove_antagonistic_source_residual(
        source, direction, core
    )

    output, _ = restore_projected_residual_norm(
        source, projected, weight
    )

    dot = (output.float() * direction).sum(dim=2)
    # P0 divides by direction_energy + eps, so it can retain an O(eps)
    # numerical remainder. Positive norm rescaling cannot change its sign or
    # recreate a finite antagonistic component.
    assert torch.all(dot >= -2e-6)
    torch.testing.assert_close(
        weighted_energy(output, weight),
        weighted_energy(source, weight),
    )


def test_degenerate_safe_direction_is_not_amplified():
    source = torch.ones(1, 2, 2, 1, 1)
    projected = torch.zeros_like(source)
    weight = torch.ones(1, 2, 1, 1, 1)

    output, diagnostics = restore_projected_residual_norm(
        source, projected, weight
    )

    assert torch.equal(output, projected)
    torch.testing.assert_close(
        diagnostics["norm_scale"], torch.ones(1, 2)
    )
    assert diagnostics["degenerate"].all()


def test_scale_guard_caps_near_degenerate_amplification():
    source = torch.ones(1, 1, 1, 1, 1)
    projected = source * 0.01
    weight = torch.ones(1, 1, 1, 1, 1)

    output, diagnostics = restore_projected_residual_norm(
        source, projected, weight, max_scale=4.0
    )

    torch.testing.assert_close(output, torch.full_like(output, 0.04))
    torch.testing.assert_close(
        diagnostics["norm_scale"], torch.full((1, 1), 4.0)
    )
    assert diagnostics["scale_capped"].all()


def test_preserves_projected_dtype():
    source = torch.ones(1, 1, 2, 1, 1, dtype=torch.bfloat16)
    projected = source * 0.5
    weight = torch.ones(1, 1, 1, 1, 1)

    output, _ = restore_projected_residual_norm(
        source, projected, weight
    )

    assert output.dtype == torch.bfloat16


def test_rejects_invalid_weight_shape_and_scale_guard():
    source = torch.ones(1, 1, 2, 1, 1)
    projected = source.clone()

    with pytest.raises(ValueError, match="application_weight"):
        restore_projected_residual_norm(
            source, projected, torch.ones(1, 1, 1, 1)
        )
    with pytest.raises(ValueError, match="max_scale"):
        restore_projected_residual_norm(
            source, projected, torch.ones(1, 1, 1, 1, 1), max_scale=0.5
        )
