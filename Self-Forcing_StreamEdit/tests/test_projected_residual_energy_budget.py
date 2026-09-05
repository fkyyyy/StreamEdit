import importlib.util
from pathlib import Path
import sys

import torch


def load_appearance_leakage_module():
    path = Path(__file__).parents[1] / "pipeline" / "appearance_leakage.py"
    spec = importlib.util.spec_from_file_location(
        "streamedit_appearance_leakage_standalone", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CausalResidualEnergyBudget = (
    load_appearance_leakage_module().CausalResidualEnergyBudget
)


def fields(scale: float):
    source = torch.ones(1, 2, 2, 2, 2)
    projected = source * (1.0 - scale)
    weight = torch.ones(1, 2, 1, 2, 2)
    return source, projected, weight


def test_first_observation_is_preserved_and_frozen():
    budget = CausalResidualEnergyBudget()
    source, projected, weight = fields(0.25)

    output, diagnostics = budget.apply(
        source, projected, weight, denoising_step_index=0
    )

    torch.testing.assert_close(output, projected)
    torch.testing.assert_close(
        diagnostics["projection_scale"], torch.ones(1)
    )
    assert diagnostics["reference_initialized"]
    assert set(budget.reference_fractions) == {0}


def test_later_excess_removal_is_scaled_to_first_block_budget():
    budget = CausalResidualEnergyBudget()
    source, first, weight = fields(0.25)
    budget.apply(source, first, weight, denoising_step_index=3)
    _, later, _ = fields(0.50)

    output, diagnostics = budget.apply(
        source, later, weight, denoising_step_index=3
    )

    # Removed energy is quadratic in vector magnitude: 0.50 is scaled by
    # sqrt(0.25^2 / 0.50^2) = 0.5, yielding the first-block 0.25 removal.
    torch.testing.assert_close(output, first)
    torch.testing.assert_close(
        diagnostics["projection_scale"], torch.tensor([0.5])
    )
    torch.testing.assert_close(
        diagnostics["applied_removed_fraction"],
        diagnostics["reference_removed_fraction"],
    )
    assert not diagnostics["reference_initialized"]


def test_later_smaller_removal_is_not_amplified():
    budget = CausalResidualEnergyBudget()
    source, first, weight = fields(0.50)
    budget.apply(source, first, weight, denoising_step_index=1)
    _, later, _ = fields(0.20)

    output, diagnostics = budget.apply(
        source, later, weight, denoising_step_index=1
    )

    torch.testing.assert_close(output, later)
    torch.testing.assert_close(
        diagnostics["projection_scale"], torch.ones(1)
    )


def test_uncapped_low_precision_projection_is_bit_exact():
    budget = CausalResidualEnergyBudget()
    source, projected, weight = fields(0.25)
    source = source.to(torch.bfloat16)
    projected = projected.to(torch.bfloat16)

    output, _ = budget.apply(
        source, projected, weight, denoising_step_index=0
    )

    assert torch.equal(output, projected)


def test_steps_have_independent_references():
    budget = CausalResidualEnergyBudget()
    source, step_zero, weight = fields(0.10)
    _, step_one, _ = fields(0.40)

    budget.apply(source, step_zero, weight, denoising_step_index=0)
    budget.apply(source, step_one, weight, denoising_step_index=1)

    assert set(budget.reference_fractions) == {0, 1}
    assert budget.reference_fractions[0] < budget.reference_fractions[1]


def test_application_weight_defines_effective_energy_budget():
    budget = CausalResidualEnergyBudget()
    source, projected, weight = fields(0.30)
    weight[:, 1] = 0.0

    output, diagnostics = budget.apply(
        source, projected, weight, denoising_step_index=0
    )

    torch.testing.assert_close(output, projected)
    torch.testing.assert_close(
        diagnostics["raw_removed_fraction"], torch.tensor([0.09])
    )
