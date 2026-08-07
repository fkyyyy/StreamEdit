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


def _soft_roles():
    return role_router.RoleState(
        object=torch.tensor([[[[0.4]]]]),
        boundary=torch.tensor([[[[0.3]]]]),
        hand=torch.tensor([[[[0.2]]]]),
        background=torch.tensor([[[[0.1]]]]),
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


def test_posterior_router_matches_closed_form_soft_mixture():
    target = torch.full((1, 1, 1, 1, 1), 10.0)
    source = torch.full_like(target, 3.0)
    source_reconstruction = torch.full_like(target, 5.0)

    routed, debug = role_router.PosteriorResidualFlowRouter()(
        target_velocity=target,
        source_velocity=source,
        source_reconstruction_velocity=source_reconstruction,
        roles=_soft_roles(),
    )

    # p_preserve=0.3, rho_contact=0.3/(0.3+0.3)=0.5.
    expected_residual_weight = torch.full_like(target, 0.45)
    assert torch.allclose(
        debug["residual_expert_weight"],
        expected_residual_weight,
    )
    assert torch.allclose(
        debug["target_expert_weight"],
        1.0 - expected_residual_weight,
    )
    assert torch.allclose(
        routed,
        target + expected_residual_weight * 2.0,
    )


def test_posterior_router_expert_weights_sum_to_one():
    target = torch.zeros((1, 1, 2, 3, 3))
    roles = role_router.RoleState(
        object=torch.rand((1, 1, 3, 3)),
        boundary=torch.rand((1, 1, 3, 3)),
        hand=torch.rand((1, 1, 3, 3)),
        background=torch.rand((1, 1, 3, 3)),
    )
    total = sum(roles.as_dict().values())
    roles = role_router.RoleState(
        **{
            name: value / total
            for name, value in roles.as_dict().items()
        }
    )

    _, debug = role_router.PosteriorResidualFlowRouter()(
        target_velocity=target,
        source_velocity=target,
        source_reconstruction_velocity=target,
        roles=roles,
    )

    expert_sum = (
        debug["target_expert_weight"]
        + debug["residual_expert_weight"]
    )
    assert torch.allclose(
        expert_sum,
        torch.ones_like(expert_sum),
        atol=1e-6,
    )
    assert debug["role_entropy"].min() >= 0
    assert debug["role_entropy"].max() <= 1


def test_posterior_router_hard_mode_uses_argmax_role():
    target = torch.full((1, 1, 1, 1, 1), 10.0)
    source = torch.full_like(target, 3.0)
    source_reconstruction = torch.full_like(target, 5.0)

    routed, debug = role_router.PosteriorResidualFlowRouter()(
        target_velocity=target,
        source_velocity=source,
        source_reconstruction_velocity=source_reconstruction,
        roles=_soft_roles(),
        hard_roles=True,
    )

    # Object has the largest posterior, so hard routing is pure target.
    assert torch.equal(routed, target)
    assert torch.count_nonzero(
        debug["residual_expert_weight"]
    ) == 0
    assert torch.count_nonzero(debug["role_entropy"]) == 0


def test_posterior_router_pure_preservation_roles_use_full_residual():
    target = torch.full((1, 1, 1, 2, 2), 10.0)
    source = torch.full_like(target, 3.0)
    source_reconstruction = torch.full_like(target, 5.0)
    roles = role_router.RoleState(
        object=torch.zeros((1, 1, 2, 2)),
        boundary=torch.zeros((1, 1, 2, 2)),
        hand=torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]]),
        background=torch.tensor(
            [[[[0.0, 1.0], [0.0, 1.0]]]]
        ),
    )

    routed, debug = role_router.PosteriorResidualFlowRouter()(
        target_velocity=target,
        source_velocity=source,
        source_reconstruction_velocity=source_reconstruction,
        roles=roles,
    )

    assert torch.equal(
        debug["residual_expert_weight"],
        torch.ones_like(target),
    )
    assert torch.equal(routed, target + 2.0)


def test_posterior_router_pure_contact_uses_target_field():
    target = torch.full((1, 1, 1, 1, 1), 10.0)
    source = torch.full_like(target, 3.0)
    source_reconstruction = torch.full_like(target, 5.0)
    zero = torch.zeros((1, 1, 1, 1))
    roles = role_router.RoleState(
        object=zero,
        boundary=torch.ones_like(zero),
        hand=zero,
        background=zero,
    )

    routed, debug = role_router.PosteriorResidualFlowRouter()(
        target_velocity=target,
        source_velocity=source,
        source_reconstruction_velocity=source_reconstruction,
        roles=roles,
    )

    assert torch.equal(routed, target)
    assert torch.equal(
        debug["contact_target_weight"],
        torch.ones_like(target),
    )


def test_posterior_router_supports_bfloat16():
    target = torch.full(
        (1, 1, 1, 1, 1),
        10.0,
        dtype=torch.bfloat16,
    )
    source = torch.full_like(target, 3.0)
    source_reconstruction = torch.full_like(target, 5.0)

    routed, debug = role_router.PosteriorResidualFlowRouter()(
        target_velocity=target,
        source_velocity=source,
        source_reconstruction_velocity=source_reconstruction,
        roles=_soft_roles(),
    )

    assert routed.dtype == torch.bfloat16
    assert debug["target_expert_weight"].dtype == torch.bfloat16
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
