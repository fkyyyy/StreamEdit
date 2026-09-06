import pytest
import torch

from tests._pipeline_imports import load_pipeline_module


role_router = load_pipeline_module("role_router")


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


class _ControlBelief:
    def __init__(
        self,
        edit_belief,
        preserve_belief,
        edit_precision,
        preserve_precision,
    ):
        self.edit_belief = edit_belief
        self.preserve_belief = preserve_belief
        self.edit_precision = edit_precision
        self.preserve_precision = preserve_precision

    def validate(self):
        values = (
            self.edit_belief,
            self.preserve_belief,
            self.edit_precision,
            self.preserve_precision,
        )
        assert len({tuple(value.shape) for value in values}) == 1


def _control_belief(
    edit_belief=0.8,
    preserve_belief=0.6,
    edit_precision=0.5,
    preserve_precision=1.0,
):
    def value(x):
        return torch.full((1, 1, 1, 1), x)

    return _ControlBelief(
        value(edit_belief),
        value(preserve_belief),
        value(edit_precision),
        value(preserve_precision),
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


def test_posterior_router_explicit_role_residual_policy():
    target = torch.full((1, 1, 1, 2, 2), 10.0)
    source = torch.full_like(target, 3.0)
    source_reconstruction = torch.full_like(target, 5.0)
    safe_residual = torch.full_like(target, 4.0)

    routed, debug = role_router.PosteriorResidualFlowRouter()(
        target_velocity=target,
        source_velocity=source,
        source_reconstruction_velocity=source_reconstruction,
        roles=_roles(),
        editable_source_residual=safe_residual,
        object_residual_strength=0.10,
        contact_residual_strength=0.35,
    )

    expected = torch.tensor([[[[[10.4, 11.4], [12.0, 12.0]]]]])
    expected_weight = torch.tensor([[[[[0.10, 0.35], [1.0, 1.0]]]]])
    assert torch.allclose(routed, expected)
    assert torch.allclose(
        debug["residual_expert_weight"], expected_weight
    )


def test_role_memory_gates_read_contact_but_write_object_core_only():
    read, write, debug = role_router.build_role_memory_gates(
        _roles(),
        spatial_size=(2, 2),
        contact_read_weight=0.5,
        object_write_threshold=0.5,
    )

    assert torch.equal(
        read, torch.tensor([[[[1.0, 0.5], [0.0, 0.0]]]])
    )
    assert torch.equal(
        write, torch.tensor([[[[True, False], [False, False]]]])
    )
    assert torch.equal(debug["role_memory_write_gate"], write.float())


def test_role_memory_gates_abstain_on_high_entropy_tokens():
    read, write, debug = role_router.build_role_memory_gates(
        _soft_roles(),
        spatial_size=(1, 1),
        contact_read_weight=0.5,
        object_write_threshold=0.35,
    )

    assert 0.0 < read.item() < 0.55
    assert not write.item()
    assert debug["role_memory_entropy"].item() > 0.5


def test_hand_connected_memory_gate_rejects_disconnected_distractor():
    object_probability = torch.zeros(1, 2, 5, 8)
    object_probability[0, 0, 2, 1:4] = 0.9
    object_probability[0, 0, 0, 5:8] = 0.95
    # The hand temporarily disappears in frame 1. Ownership must follow the
    # preceding connected component instead of jumping to the distractor.
    object_probability[0, 1, 2, 2:5] = 0.9
    object_probability[0, 1, 0, 5:8] = 0.95
    background = 1.0 - object_probability
    roles = role_router.RoleState(
        object=object_probability,
        boundary=torch.zeros_like(object_probability),
        hand=torch.zeros_like(object_probability),
        background=background,
    )
    hand_anchor = torch.zeros_like(object_probability)
    hand_anchor[0, 0, 2, 0] = 1.0

    read, write, debug = role_router.build_role_memory_gates(
        roles,
        spatial_size=(5, 8),
        object_write_threshold=0.5,
        hand_anchor=hand_anchor,
    )

    assert write[0, 0, 2, 1:4].all()
    assert write[0, 1, 2, 2:5].all()
    assert not write[0, :, 0, 5:8].any()
    assert not (read[0, :, 0, 5:8] > 0).any()
    assert debug["role_memory_raw_write_gate"][
        0, :, 0, 5:8
    ].bool().all()


def test_memory_gate_can_share_precomputed_s1_owner_support():
    object_probability = torch.full((1, 1, 2, 3), 0.9)
    roles = role_router.RoleState(
        object=object_probability,
        boundary=torch.zeros_like(object_probability),
        hand=torch.zeros_like(object_probability),
        background=1.0 - object_probability,
    )
    owner_support = torch.tensor(
        [[[[True, True, False], [False, False, False]]]]
    )

    read, write, debug = role_router.build_role_memory_gates(
        roles,
        spatial_size=(2, 3),
        object_write_threshold=0.5,
        owner_support=owner_support,
    )

    assert torch.equal(write, owner_support)
    assert torch.equal(read > 0, owner_support)
    assert torch.equal(
        debug["role_memory_connected_support"].bool(),
        owner_support,
    )


def test_adaptive_write_is_sparse_consistent_and_recovery_read_only():
    object_probability = torch.full((1, 3, 3, 4), 0.9)
    roles = role_router.RoleState(
        object=object_probability,
        boundary=torch.zeros_like(object_probability),
        hand=torch.zeros_like(object_probability),
        background=1.0 - object_probability,
    )
    support = torch.ones_like(object_probability, dtype=torch.bool)
    recovery = torch.zeros(1, 3, 1, 1, dtype=torch.bool)
    recovery[:, 1] = True

    read, write, debug = role_router.build_role_memory_gates(
        roles,
        spatial_size=(3, 4),
        object_write_threshold=0.5,
        owner_support=support,
        evidence_reliability=torch.ones(1, 3, 1, 1),
        temporal_recovery=recovery,
        adaptive_write=True,
        temporal_write_consensus=True,
    )

    assert (read[:, 1] > 0).any()
    assert not write[:, 0].any()  # first frame establishes candidates
    assert not write[:, 1].any()  # recovery frames are read-only
    assert not write[:, 2].any()  # previous frame had no trusted write
    assert debug["role_memory_raw_write_gate"][:, 0].sum() == 12
    assert debug["role_memory_connected_write_gate"][:, 0].sum() == 12
    assert debug["role_memory_candidate_write_gate"][:, 0].sum() == 12
    assert debug["role_memory_preconsensus_write_gate"][:, 0].sum() == 4


def test_connected_area_limit_respects_budget_and_preserves_path():
    support = torch.zeros(1, 1, 5, 8, dtype=torch.bool)
    support[0, 0, 2, 1:8] = True
    support[0, 0, 1:4, 6:8] = True
    weight = torch.linspace(1.0, 0.1, 40).reshape(1, 1, 5, 8)
    anchor = torch.zeros_like(support)
    anchor[0, 0, 2, 0] = True

    limited = role_router.limit_connected_support_area(
        support,
        weight,
        anchor,
        torch.tensor([[[[0.10]]]]),
    )

    assert limited.sum().item() == 4
    # Every selected token is reachable from the hand-side root through the
    # selected support; the distant high-area blob cannot survive alone.
    reached = torch.zeros_like(limited)
    reached[0, 0, 2, 1] = limited[0, 0, 2, 1]
    for _ in range(4):
        reached = reached | (
            role_router._dilate(reached, 1) & limited
        )
    assert torch.equal(reached, limited)


def test_bayes_router_matches_precision_weighted_closed_form():
    target = torch.full((1, 1, 1, 1, 1), 10.0)
    source = torch.full_like(target, 3.0)
    source_reconstruction = torch.full_like(target, 5.0)

    routed, debug = role_router.BayesResidualFlowRouter()(
        target_velocity=target,
        source_velocity=source,
        source_reconstruction_velocity=source_reconstruction,
        belief=_control_belief(),
    )

    # edit strength=0.8*0.5=0.4, preserve strength=0.6*1=0.6.
    assert torch.allclose(
        debug["preserve_action_weight"],
        torch.full_like(target, 0.6),
    )
    assert torch.allclose(routed, torch.full_like(target, 11.2))
    assert debug["action_sum_error"].max().item() < 1e-6


def test_bayes_router_preserves_when_both_beliefs_are_absent():
    target = torch.full((1, 1, 1, 1, 1), 10.0)
    source = torch.full_like(target, 3.0)
    source_reconstruction = torch.full_like(target, 5.0)

    routed, debug = role_router.BayesResidualFlowRouter()(
        target_velocity=target,
        source_velocity=source,
        source_reconstruction_velocity=source_reconstruction,
        belief=_control_belief(
            edit_belief=0.0,
            preserve_belief=0.0,
        ),
    )

    assert torch.equal(routed, torch.full_like(target, 12.0))
    assert torch.equal(
        debug["no_evidence"],
        torch.ones_like(target),
    )


def test_bayes_router_uses_fp32_weights_with_bfloat16_velocity():
    target = torch.full(
        (1, 1, 1, 1, 1),
        10.0,
        dtype=torch.bfloat16,
    )
    source = torch.full_like(target, 3.0)
    source_reconstruction = torch.full_like(target, 5.0)

    routed, debug = role_router.BayesResidualFlowRouter()(
        target_velocity=target,
        source_velocity=source,
        source_reconstruction_velocity=source_reconstruction,
        belief=_control_belief(),
    )

    assert routed.dtype == torch.bfloat16
    assert debug["preserve_action_weight"].dtype == torch.float32
    assert debug["action_sum_error"].max().item() < 1e-6


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
