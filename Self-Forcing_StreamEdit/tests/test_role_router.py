import importlib.util
from pathlib import Path
import sys

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "pipeline" / "role_router.py"
SPEC = importlib.util.spec_from_file_location("role_router", MODULE_PATH)
role_router = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = role_router
SPEC.loader.exec_module(role_router)


def _roles():
    return role_router.RoleState(
        object=torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
        boundary=torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]]),
        hand=torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]]]),
        background=torch.tensor([[[[0.0, 0.0], [0.0, 1.0]]]]),
    )


def test_residual_router_applies_role_specific_source_correction():
    target = torch.full((1, 1, 1, 2, 2), 10.0)
    source = torch.full_like(target, 3.0)
    source_reconstruction = torch.full_like(target, 5.0)

    routed, correction_weight = role_router.ResidualRoleFlowRouter()(
        target_velocity=target,
        source_velocity=source,
        source_reconstruction_velocity=source_reconstruction,
        roles=_roles(),
        contact_target_weight=0.75,
    )

    expected_weight = torch.tensor(
        [[[[[0.0, 0.25], [1.0, 1.0]]]]]
    )
    assert torch.allclose(correction_weight, expected_weight)
    assert torch.allclose(routed, target + expected_weight * 2.0)


def test_residual_router_supports_bfloat16():
    target = torch.full(
        (1, 1, 1, 2, 2), 10.0, dtype=torch.bfloat16
    )
    source = torch.full_like(target, 3.0)
    source_reconstruction = torch.full_like(target, 5.0)

    routed, correction_weight = role_router.ResidualRoleFlowRouter()(
        target_velocity=target,
        source_velocity=source,
        source_reconstruction_velocity=source_reconstruction,
        roles=_roles(),
    )

    assert routed.dtype == torch.bfloat16
    assert correction_weight.dtype == torch.bfloat16
    assert torch.isfinite(routed.float()).all()


def test_legacy_role_router_behavior_is_unchanged():
    target = torch.full((1, 1, 1, 2, 2), 10.0)
    source_reconstruction = torch.full_like(target, 5.0)

    routed, edit_weight, preserve_weight = role_router.RoleFlowRouter()(
        target_velocity=target,
        source_reconstruction_velocity=source_reconstruction,
        roles=_roles(),
    )

    expected_edit = torch.tensor(
        [[[[[1.0, 1.0], [0.0, 0.0]]]]]
    )
    expected_preserve = 1.0 - expected_edit
    assert torch.equal(edit_weight, expected_edit)
    assert torch.equal(preserve_weight, expected_preserve)
    assert torch.equal(
        routed,
        expected_edit * target
        + expected_preserve * source_reconstruction,
    )


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_residual_router_rejects_invalid_contact_weight(weight):
    velocity = torch.zeros((1, 1, 1, 2, 2))
    with pytest.raises(ValueError, match="contact_target_weight"):
        role_router.ResidualRoleFlowRouter()(
            target_velocity=velocity,
            source_velocity=velocity,
            source_reconstruction_velocity=velocity,
            roles=_roles(),
            contact_target_weight=weight,
        )
