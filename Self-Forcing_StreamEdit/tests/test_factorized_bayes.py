import unittest
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

from tests._pipeline_imports import load_pipeline_module


role_router = load_pipeline_module("role_router")
factorized_bayes = load_pipeline_module("factorized_bayes")
RoleState = role_router.RoleState
FactorizedBayesOperatorBuilder = (
    factorized_bayes.FactorizedBayesOperatorBuilder
)
route_factorized_velocity = factorized_bayes.route_factorized_velocity


def load_attention_module():
    path = Path(__file__).resolve().parents[1] / "wan/modules/attention.py"
    spec = importlib.util.spec_from_file_location(
        "streamedit_factorized_bayes_attention", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fuse_factorized_aligned_memory = (
    load_attention_module().fuse_factorized_aligned_memory
)
blend_factorized_with_native_fallback = (
    load_attention_module().blend_factorized_with_native_fallback
)
build_factorized_history_read_mask = (
    load_attention_module().build_factorized_history_read_mask
)
materialize_immutable_target_value = (
    load_attention_module().materialize_immutable_target_value
)
resolve_target_identity_correction_strength = (
    load_attention_module().resolve_target_identity_correction_strength
)


def make_evidence(
    *,
    object_posterior=1.0,
    source_attention=1.0,
    reliability=1.0,
    hand_proximity=0.0,
    visible=1.0,
):
    shape = (1, 1, 1, 1)

    def full(value):
        return torch.full(shape, float(value))

    return {
        "object_posterior": full(object_posterior),
        "posterior_threshold": full(0.2),
        "source_attention": full(source_attention),
        "hand_proximity": full(hand_proximity),
        "adaptive_attention_reliability": full(reliability),
        "object_visible": full(visible),
        "temporal_confidence": full(object_posterior),
    }


def one_hot_roles(name):
    values = {
        key: torch.zeros(1, 1, 2, 2)
        for key in ("object", "boundary", "hand", "background")
    }
    values[name].fill_(1.0)
    return RoleState(**values)


class FactorizedBayesTest(unittest.TestCase):
    def test_immutable_absolute_identity_honors_configured_strength(self):
        self.assertEqual(
            resolve_target_identity_correction_strength(
                0.6,
                immutable_factorized_identity=True,
                prototype_value_is_residual=False,
            ),
            0.6,
        )

    def test_legacy_absolute_identity_keeps_hard_replacement(self):
        self.assertEqual(
            resolve_target_identity_correction_strength(
                0.6,
                prototype_value_is_residual=False,
            ),
            1.0,
        )

    def test_immutable_residual_changes_only_owner_tokens(self):
        correspondence = torch.tensor(
            [[[[1.0, 0.0]], [[1.0, 0.0]]]]
        )
        target = torch.tensor(
            [[[[2.0, 2.0]], [[7.0, 7.0]]]]
        )
        source = torch.tensor(
            [[[[1.0, 1.0]], [[5.0, 5.0]]]]
        )
        owner = torch.tensor([[1.0, 0.0]])

        materialized, support, _ = (
            materialize_immutable_target_value(
                correspondence_key=correspondence,
                target_value=target,
                source_value=source,
                prototype_key=torch.tensor([[[[1.0, 0.0]]]]),
                prototype_value=torch.tensor([[[[3.0, 4.0]]]]),
                prototype_evidence=torch.ones(1, 1),
                owner_weight=owner,
                tokens_per_frame=2,
                support_floor=1.0,
                correction_strength=1.0,
            )
        )

        torch.testing.assert_close(
            materialized[:, 0], source[:, 0] + torch.tensor([[[3.0, 4.0]]])
        )
        torch.testing.assert_close(
            materialized[:, 1], target[:, 1], rtol=0.0, atol=0.0
        )
        self.assertEqual(support[0, 0].item(), 1.0)
        self.assertEqual(support[0, 1].item(), 0.0)

    def test_immutable_absolute_value_does_not_add_source_appearance(self):
        correspondence = torch.tensor(
            [[[[1.0, 0.0]], [[1.0, 0.0]]]]
        )
        target = torch.zeros_like(correspondence)
        source = torch.full_like(correspondence, 100.0)
        owner = torch.tensor([[1.0, 0.0]])

        materialized, support, _ = materialize_immutable_target_value(
            correspondence_key=correspondence,
            target_value=target,
            source_value=source,
            prototype_key=torch.tensor([[[[1.0, 0.0]]]]),
            prototype_value=torch.tensor([[[[3.0, 4.0]]]]),
            prototype_evidence=torch.ones(1, 1),
            owner_weight=owner,
            tokens_per_frame=2,
            prototype_value_is_residual=False,
            support_floor=1.0,
            correction_strength=1.0,
        )

        torch.testing.assert_close(
            materialized[:, 0], torch.tensor([[[3.0, 4.0]]])
        )
        torch.testing.assert_close(
            materialized[:, 1], target[:, 1], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(support, owner)

    def test_residual_subspace_preserves_current_orthogonal_structure(self):
        correspondence = torch.tensor([[[[1.0, 0.0]]]])
        source = torch.zeros_like(correspondence)
        target = torch.tensor([[[[0.0, 5.0]]]])

        corrected, support, diagnostics = (
            materialize_immutable_target_value(
                correspondence_key=correspondence,
                target_value=target,
                source_value=source,
                prototype_key=torch.tensor(
                    [[[[1.0, 0.0]], [[1.0, 0.0]]]]
                ),
                prototype_value=torch.tensor(
                    [[[[2.0, 0.0]], [[4.0, 0.0]]]]
                ),
                prototype_evidence=torch.ones(1, 2),
                owner_weight=torch.ones(1, 1),
                tokens_per_frame=1,
                prototype_value_is_residual=True,
                residual_subspace=True,
                support_floor=1.0,
                correction_strength=1.0,
            )
        )

        torch.testing.assert_close(
            corrected, torch.tensor([[[[3.0, 5.0]]]])
        )
        torch.testing.assert_close(support, torch.ones_like(support))
        torch.testing.assert_close(
            diagnostics["appearance_subspace_coherence"],
            torch.ones_like(support),
        )

    def test_prototype_retrieval_reports_assignment_stability(self):
        correspondence = torch.tensor([
            [[[1.0, 0.0]], [[0.0, 1.0]]],
        ])
        target = torch.zeros_like(correspondence)
        prototypes = torch.tensor([
            [[[1.0, 0.0]], [[0.0, 1.0]]],
        ])

        _, _, diagnostics = materialize_immutable_target_value(
            correspondence_key=correspondence,
            target_value=target,
            source_value=target,
            prototype_key=prototypes,
            prototype_value=prototypes,
            prototype_evidence=torch.ones(1, 2),
            owner_weight=torch.ones(1, 2),
            tokens_per_frame=2,
            prototype_value_is_residual=True,
            support_floor=1.0,
            correction_strength=1.0,
        )

        torch.testing.assert_close(
            diagnostics["selected_prototype"],
            torch.tensor([[0.0, 1.0]]),
        )
        self.assertTrue(
            (diagnostics["prototype_assignment_peak"] > 0.5).all()
        )
        self.assertTrue(
            (diagnostics["prototype_assignment_margin"] > 0.0).all()
        )
        self.assertTrue(
            (diagnostics["prototype_assignment_entropy"] >= 0.0).all()
        )
        self.assertTrue(
            (diagnostics["prototype_assignment_entropy"] <= 1.0).all()
        )

    def test_incoherent_residual_prototypes_cannot_create_blob(self):
        correspondence = torch.tensor([[[[1.0, 0.0]]]])
        source = torch.zeros_like(correspondence)
        target = torch.tensor([[[[0.5, 5.0]]]])

        corrected, support, diagnostics = (
            materialize_immutable_target_value(
                correspondence_key=correspondence,
                target_value=target,
                source_value=source,
                prototype_key=torch.tensor(
                    [[[[1.0, 0.0]], [[1.0, 0.0]]]]
                ),
                prototype_value=torch.tensor(
                    [[[[2.0, 0.0]], [[-2.0, 0.0]]]]
                ),
                prototype_evidence=torch.ones(1, 2),
                owner_weight=torch.ones(1, 1),
                tokens_per_frame=1,
                prototype_value_is_residual=True,
                residual_subspace=True,
                support_floor=1.0,
                correction_strength=1.0,
            )
        )

        torch.testing.assert_close(corrected, target)
        torch.testing.assert_close(support, torch.zeros_like(support))
        torch.testing.assert_close(
            diagnostics["appearance_subspace_action"],
            torch.zeros_like(support),
        )

    def test_residual_subspace_requires_residual_memory(self):
        value = torch.zeros(1, 1, 1, 2)
        with self.assertRaisesRegex(
            ValueError, "requires residual prototypes"
        ):
            materialize_immutable_target_value(
                correspondence_key=torch.ones_like(value),
                target_value=value,
                source_value=value,
                prototype_key=torch.ones_like(value),
                prototype_value=torch.ones_like(value),
                prototype_evidence=torch.ones(1, 1),
                owner_weight=torch.ones(1, 1),
                tokens_per_frame=1,
                prototype_value_is_residual=False,
                residual_subspace=True,
            )

    def test_orthogonal_geometry_preserves_motion_not_source_appearance(self):
        operators = FactorizedBayesOperatorBuilder()(
            roles=one_hot_roles("object"),
            evidence=make_evidence(),
            expected_token_length=1,
        )
        target = torch.tensor([[[[[1.0]], [[0.0]]]]])
        source = torch.zeros_like(target)
        # The first channel opposes the target edit and must be removed;
        # the second is orthogonal geometry/motion and must survive.
        reconstruction = torch.tensor([[[[[-1.0]], [[2.0]]]]])
        routed, diagnostics = route_factorized_velocity(
            target,
            source,
            reconstruction,
            operators,
            torch.zeros(1, 1, 1, 1, 1),
            geometry_owner_weight=torch.ones(1, 1, 1, 1),
            geometry_strength=1.0,
            denoising_fraction=1.0,
        )

        torch.testing.assert_close(
            routed, torch.tensor([[[[[1.0]], [[2.0]]]]])
        )
        torch.testing.assert_close(
            diagnostics["orthogonal_geometry_action"],
            torch.ones(1, 1, 1, 1, 1),
        )

    def test_orthogonal_geometry_is_zero_outside_owner(self):
        operators = FactorizedBayesOperatorBuilder()(
            roles=one_hot_roles("object"),
            evidence=make_evidence(),
            expected_token_length=1,
        )
        target = torch.tensor([[[[[1.0]], [[0.0]]]]])
        source = torch.zeros_like(target)
        reconstruction = torch.tensor([[[[[-1.0]], [[2.0]]]]])
        routed, diagnostics = route_factorized_velocity(
            target,
            source,
            reconstruction,
            operators,
            torch.zeros(1, 1, 1, 1, 1),
            geometry_owner_weight=torch.zeros(1, 1, 1, 1),
            geometry_strength=1.0,
            denoising_fraction=1.0,
        )

        torch.testing.assert_close(routed, target, rtol=0.0, atol=0.0)
        self.assertFalse(
            diagnostics["orthogonal_geometry_action"].any()
        )

    def test_object_owns_target_value_and_rejects_source_appearance(self):
        operators = FactorizedBayesOperatorBuilder()(
            roles=one_hot_roles("object"),
            evidence=make_evidence(),
            expected_token_length=1,
        )
        self.assertTrue(
            torch.equal(
                operators.source_value_action,
                torch.zeros_like(operators.source_value_action),
            )
        )
        self.assertTrue(
            torch.equal(
                operators.source_residual_action,
                torch.zeros_like(operators.source_residual_action),
            )
        )
        self.assertTrue(
            torch.equal(
                operators.unknown_action,
                torch.zeros_like(operators.unknown_action),
            )
        )
        self.assertTrue(
            torch.equal(
                operators.target_memory_action,
                torch.ones_like(operators.target_memory_action),
            )
        )
        self.assertTrue(
            torch.equal(
                operators.target_memory_action,
                torch.ones_like(operators.target_memory_action),
            )
        )
        self.assertTrue(
            torch.equal(
                operators.source_key_action,
                torch.ones_like(operators.source_key_action),
            )
        )

    def test_soft_object_core_does_not_leak_background_value(self):
        object_role = torch.full((1, 1, 2, 2), 0.6)
        operators = FactorizedBayesOperatorBuilder()(
            roles=RoleState(
                object=object_role,
                boundary=torch.zeros_like(object_role),
                hand=torch.zeros_like(object_role),
                background=1.0 - object_role,
            ),
            evidence=make_evidence(object_posterior=0.6),
            expected_token_length=1,
        )
        self.assertTrue(
            torch.equal(
                operators.source_value_action,
                torch.zeros_like(operators.source_value_action),
            )
        )
        self.assertTrue(
            torch.equal(
                operators.source_residual_action,
                torch.zeros_like(operators.source_residual_action),
            )
        )

    def test_partial_object_patch_is_target_owned_at_token_level(self):
        object_role = torch.zeros(1, 1, 2, 2)
        object_role[..., 0, 0] = 0.6
        operators = FactorizedBayesOperatorBuilder()(
            roles=RoleState(
                object=object_role,
                boundary=torch.zeros_like(object_role),
                hand=torch.zeros_like(object_role),
                background=1.0 - object_role,
            ),
            evidence=make_evidence(object_posterior=0.6),
            expected_token_length=1,
        )
        self.assertEqual(operators.source_value_action.item(), 0.0)
        self.assertEqual(operators.source_residual_action.item(), 0.0)
        self.assertEqual(operators.unknown_action.item(), 0.0)
        self.assertEqual(operators.target_memory_action.item(), 1.0)
        self.assertEqual(operators.source_key_action.item(), 1.0)

    def test_hand_and_background_read_source_state(self):
        for role_name, evidence in (
            ("hand", make_evidence(hand_proximity=1.0)),
            (
                "background",
                make_evidence(
                    object_posterior=0.0,
                    source_attention=0.0,
                    visible=0.0,
                ),
            ),
        ):
            with self.subTest(role=role_name):
                operators = FactorizedBayesOperatorBuilder()(
                    roles=one_hot_roles(role_name),
                    evidence=evidence,
                    expected_token_length=1,
                )
                self.assertTrue(
                    torch.allclose(operators.source_key_action, torch.ones(1, 1))
                )
                self.assertTrue(
                    torch.allclose(operators.source_value_action, torch.ones(1, 1))
                )
                self.assertTrue(
                    torch.allclose(operators.source_residual_action, torch.ones(1, 1))
                )
                self.assertTrue(
                    torch.allclose(operators.target_memory_action, torch.zeros(1, 1))
                )

    def test_missing_evidence_is_unknown_not_source_owned(self):
        operators = FactorizedBayesOperatorBuilder()(
            roles=one_hot_roles("background"),
            evidence=make_evidence(
                object_posterior=0.0,
                source_attention=0.5,
                reliability=0.0,
                visible=0.0,
            ),
            expected_token_length=1,
        )
        self.assertTrue(
            torch.allclose(operators.unknown_action, torch.ones(1, 1))
        )
        self.assertTrue(
            torch.allclose(operators.source_value_action, torch.zeros(1, 1))
        )
        self.assertTrue(
            torch.allclose(operators.source_residual_action, torch.zeros(1, 1))
        )

    def test_boundary_is_soft_and_provenance_is_normalized(self):
        operators = FactorizedBayesOperatorBuilder(
            boundary_source_fraction=0.25
        )(
            roles=one_hot_roles("boundary"),
            evidence=make_evidence(hand_proximity=1.0),
            expected_token_length=1,
        )
        self.assertTrue(
            torch.allclose(operators.source_value_action, torch.full((1, 1), 0.25))
        )
        self.assertTrue(
            torch.allclose(operators.target_memory_action, torch.full((1, 1), 0.75))
        )
        provenance = (
            operators.source_value_action
            + operators.target_memory_action
            + operators.unknown_action
        )
        self.assertTrue(torch.allclose(provenance, torch.ones_like(provenance)))

    def test_unknown_velocity_uses_native_fallback(self):
        operators = FactorizedBayesOperatorBuilder()(
            roles=one_hot_roles("background"),
            evidence=make_evidence(
                object_posterior=0.0,
                source_attention=0.5,
                reliability=0.0,
                visible=0.0,
            ),
            expected_token_length=1,
        )
        target = torch.zeros(1, 1, 1, 2, 2)
        source = torch.zeros_like(target)
        reconstruction = torch.ones_like(target)
        fallback = torch.full_like(target, 0.4)
        routed, diagnostics = route_factorized_velocity(
            target, source, reconstruction, operators, fallback
        )
        self.assertTrue(torch.allclose(routed, fallback))
        self.assertTrue(
            torch.allclose(
                diagnostics["effective_source_residual_action"],
                fallback,
            )
        )

    def test_owner_source_block_is_orthogonal_to_native_fallback(self):
        operators = FactorizedBayesOperatorBuilder()(
            roles=one_hot_roles("object"),
            evidence=make_evidence(object_posterior=1.0),
            expected_token_length=1,
        )
        target = torch.zeros(1, 1, 1, 2, 2)
        source = torch.zeros_like(target)
        reconstruction = torch.ones_like(target)
        fallback = torch.full(
            (1, 1, 1, 2, 2), 0.4, dtype=target.dtype
        )
        owner = torch.ones(1, 1, 2, 2)

        unblocked, unblocked_debug = route_factorized_velocity(
            target,
            source,
            reconstruction,
            operators,
            fallback,
            target_owned_weight=owner,
            block_target_owned_source=False,
        )
        blocked, blocked_debug = route_factorized_velocity(
            target,
            source,
            reconstruction,
            operators,
            fallback,
            target_owned_weight=owner,
            block_target_owned_source=True,
        )

        torch.testing.assert_close(unblocked, fallback)
        torch.testing.assert_close(blocked, target)
        torch.testing.assert_close(
            unblocked_debug[
                "target_owned_native_fallback_action"
            ],
            fallback,
        )
        torch.testing.assert_close(
            blocked_debug[
                "target_owned_native_fallback_action"
            ],
            torch.zeros_like(fallback),
        )

    def test_paired_support_softly_arbitrates_source_fallback(self):
        operators = FactorizedBayesOperatorBuilder()(
            roles=one_hot_roles("object"),
            evidence=make_evidence(object_posterior=1.0),
            expected_token_length=1,
        )
        target = torch.zeros(1, 1, 1, 2, 2)
        source = torch.zeros_like(target)
        reconstruction = torch.ones_like(target)
        fallback = torch.full_like(target, 0.8)
        paired_support = torch.full((1, 1, 1, 1), 0.5)
        routed, diagnostics = route_factorized_velocity(
            target,
            source,
            reconstruction,
            operators,
            fallback,
            target_owned_weight=torch.ones(1, 1, 1, 1),
            block_target_owned_source=False,
            paired_memory_support_weight=paired_support,
            paired_memory_source_suppression=0.5,
        )
        # Native owner fallback is 0.8. A 0.5 support at strength 0.5
        # suppresses one quarter of it, leaving 0.6.
        torch.testing.assert_close(routed, torch.full_like(target, 0.6))
        torch.testing.assert_close(
            diagnostics[
                "paired_memory_source_suppression_action"
            ],
            torch.full_like(fallback, 0.25),
        )

    def test_paired_abstention_preserves_native_velocity_exactly(self):
        operators = FactorizedBayesOperatorBuilder()(
            roles=one_hot_roles("object"),
            evidence=make_evidence(object_posterior=1.0),
            expected_token_length=1,
        )
        target = torch.randn(1, 1, 1, 2, 2)
        source = torch.zeros_like(target)
        reconstruction = torch.ones_like(target)
        fallback = torch.full_like(target, 0.4)
        baseline, _ = route_factorized_velocity(
            target, source, reconstruction, operators, fallback,
            target_owned_weight=torch.ones(1, 1, 1, 1),
            block_target_owned_source=False,
        )
        abstained, _ = route_factorized_velocity(
            target, source, reconstruction, operators, fallback,
            target_owned_weight=torch.ones(1, 1, 1, 1),
            block_target_owned_source=False,
            paired_memory_support_weight=torch.zeros(1, 1, 1, 1),
            paired_memory_source_suppression=1.0,
        )
        torch.testing.assert_close(
            abstained, baseline, rtol=0.0, atol=0.0
        )

    def test_verified_native_history_softly_suppresses_source_fallback(self):
        operators = FactorizedBayesOperatorBuilder()(
            roles=one_hot_roles("object"),
            evidence=make_evidence(object_posterior=1.0),
            expected_token_length=1,
        )
        target = torch.zeros(1, 1, 1, 2, 2)
        source = torch.zeros_like(target)
        reconstruction = torch.ones_like(target)
        fallback = torch.full_like(target, 0.8)
        verified = torch.full((1, 1, 1, 1), 0.5)
        routed, diagnostics = route_factorized_velocity(
            target, source, reconstruction, operators, fallback,
            target_owned_weight=torch.ones(1, 1, 1, 1),
            block_target_owned_source=False,
            verified_native_history_support_weight=verified,
            verified_native_history_source_suppression=0.5,
        )

        torch.testing.assert_close(routed, torch.full_like(target, 0.6))
        torch.testing.assert_close(
            diagnostics[
                "verified_native_history_source_suppression_action"
            ],
            torch.full_like(fallback, 0.25),
        )

    def test_native_history_abstention_preserves_velocity_exactly(self):
        operators = FactorizedBayesOperatorBuilder()(
            roles=one_hot_roles("object"),
            evidence=make_evidence(object_posterior=1.0),
            expected_token_length=1,
        )
        target = torch.randn(1, 1, 1, 2, 2)
        source = torch.zeros_like(target)
        reconstruction = torch.ones_like(target)
        fallback = torch.full_like(target, 0.4)
        baseline, _ = route_factorized_velocity(
            target, source, reconstruction, operators, fallback,
            target_owned_weight=torch.ones(1, 1, 1, 1),
            block_target_owned_source=False,
        )
        abstained, _ = route_factorized_velocity(
            target, source, reconstruction, operators, fallback,
            target_owned_weight=torch.ones(1, 1, 1, 1),
            block_target_owned_source=False,
            verified_native_history_support_weight=(
                torch.zeros(1, 1, 1, 1)
            ),
            verified_native_history_source_suppression=1.0,
        )

        torch.testing.assert_close(
            abstained, baseline, rtol=0.0, atol=0.0
        )

    def test_verified_retrieval_projects_only_antagonistic_source_appearance(self):
        operators = FactorizedBayesOperatorBuilder()(
            roles=one_hot_roles("object"),
            evidence=make_evidence(object_posterior=1.0),
            expected_token_length=1,
        )
        target = torch.ones(1, 1, 1, 1, 1)
        source = torch.zeros_like(target)
        # This source reconstruction residual points exactly against the edit.
        reconstruction = -torch.ones_like(target)
        fallback = torch.ones_like(target)
        routed, diagnostics = route_factorized_velocity(
            target, source, reconstruction, operators, fallback,
            target_owned_weight=torch.ones(1, 1, 1, 1),
            block_target_owned_source=False,
            verified_native_history_support_weight=torch.ones(1, 1, 1, 1),
            verified_native_history_source_suppression=1.0,
            verified_native_history_appearance_projection=True,
        )
        torch.testing.assert_close(routed, target)
        self.assertGreater(
            diagnostics[
                "verified_native_history_appearance_removed_energy"
            ].item(),
            0.0,
        )

    def test_source_coordinate_target_delta_is_spatially_gated(self):
        target = torch.full((1, 1, 1, 2, 2), 5.0)
        source = torch.full_like(target, 2.0)
        reconstruction = torch.full_like(target, 10.0)
        fallback = torch.zeros_like(target)

        object_operators = FactorizedBayesOperatorBuilder()(
            roles=one_hot_roles("object"),
            evidence=make_evidence(object_posterior=1.0),
            expected_token_length=1,
        )
        object_routed, object_debug = route_factorized_velocity(
            target,
            source,
            reconstruction,
            object_operators,
            fallback,
            source_coordinate_target_delta=True,
        )
        torch.testing.assert_close(
            object_routed,
            torch.full_like(target, 13.0),
        )
        torch.testing.assert_close(
            object_debug[
                "source_coordinate_target_delta_action"
            ],
            torch.ones_like(fallback),
        )

        background_operators = FactorizedBayesOperatorBuilder()(
            roles=one_hot_roles("background"),
            evidence=make_evidence(),
            expected_token_length=1,
        )
        background_routed, background_debug = (
            route_factorized_velocity(
                target,
                source,
                reconstruction,
                background_operators,
                fallback,
                source_coordinate_target_delta=True,
            )
        )
        torch.testing.assert_close(background_routed, reconstruction)
        torch.testing.assert_close(
            background_debug[
                "source_coordinate_target_delta_action"
            ],
            torch.zeros_like(fallback),
        )

        hand_operators = FactorizedBayesOperatorBuilder()(
            roles=one_hot_roles("hand"),
            evidence=make_evidence(),
            expected_token_length=1,
        )
        hand_routed, _ = route_factorized_velocity(
            target,
            source,
            reconstruction,
            hand_operators,
            fallback,
            source_coordinate_target_delta=True,
        )
        torch.testing.assert_close(hand_routed, reconstruction)

        boundary_operators = FactorizedBayesOperatorBuilder()(
            roles=one_hot_roles("boundary"),
            evidence=make_evidence(object_posterior=1.0),
            expected_token_length=1,
        )
        boundary_routed, boundary_debug = route_factorized_velocity(
            target,
            source,
            reconstruction,
            boundary_operators,
            fallback,
            source_coordinate_target_delta=True,
        )
        # The default boundary split keeps 25% source provenance and
        # applies 75% of the target-minus-source delta.
        torch.testing.assert_close(
            boundary_routed,
            torch.full_like(target, 12.25),
        )
        torch.testing.assert_close(
            boundary_debug[
                "source_coordinate_target_delta_action"
            ],
            torch.full_like(fallback, 0.75),
        )

    def test_owner_complement_is_exact_clean_source(self):
        operators = FactorizedBayesOperatorBuilder()(
            roles=one_hot_roles("object"),
            evidence=make_evidence(object_posterior=1.0),
            expected_token_length=1,
        )
        target = torch.full((1, 1, 1, 5, 5), 5.0)
        source = torch.full_like(target, 2.0)
        reconstruction = torch.full_like(target, 10.0)
        owner = torch.zeros(1, 1, 5, 5)
        owner[:, :, 2, 2] = 1.0

        routed, diagnostics = route_factorized_velocity(
            target,
            source,
            reconstruction,
            operators,
            torch.zeros(1, 1, 1, 5, 5),
            owner_complement_source_weight=owner,
            owner_complement_margin=0,
        )

        # The edited owner keeps the normal target path. Every definite
        # non-owner pixel is exactly the clean-source reconstruction.
        self.assertEqual(routed[0, 0, 0, 2, 2].item(), 5.0)
        outside = torch.ones(5, 5, dtype=torch.bool)
        outside[2, 2] = False
        torch.testing.assert_close(
            routed[0, 0, 0][outside],
            torch.full((24,), 10.0),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(
            diagnostics["owner_complement_source_action"][
                0, 0, 0, 2, 2
            ].item(),
            0.0,
        )

    def test_owner_complement_margin_keeps_object_boundary_native(self):
        operators = FactorizedBayesOperatorBuilder()(
            roles=one_hot_roles("object"),
            evidence=make_evidence(object_posterior=1.0),
            expected_token_length=1,
        )
        target = torch.full((1, 1, 1, 5, 5), 5.0)
        source = torch.full_like(target, 2.0)
        reconstruction = torch.full_like(target, 10.0)
        owner = torch.zeros(1, 1, 5, 5)
        owner[:, :, 2, 2] = 1.0

        routed, diagnostics = route_factorized_velocity(
            target,
            source,
            reconstruction,
            operators,
            torch.zeros(1, 1, 1, 5, 5),
            owner_complement_source_weight=owner,
            owner_complement_margin=1,
        )

        expected_source_action = torch.ones(1, 1, 1, 5, 5)
        expected_source_action[:, :, :, 1:4, 1:4] = 0.0
        torch.testing.assert_close(
            diagnostics["owner_complement_source_action"],
            expected_source_action,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            routed[:, :, :, 1:4, 1:4],
            torch.full((1, 1, 1, 3, 3), 5.0),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(routed[0, 0, 0, 0, 0].item(), 10.0)

    def test_owner_complement_accepts_flat_causal_owner_tokens(self):
        operators = FactorizedBayesOperatorBuilder()(
            roles=one_hot_roles("object"),
            evidence=make_evidence(object_posterior=1.0),
            expected_token_length=1,
        )
        target = torch.full((1, 1, 1, 5, 5), 5.0)
        source = torch.full_like(target, 2.0)
        reconstruction = torch.full_like(target, 10.0)
        # CausalObjectOwnership.owner_support is stored as [B,L].  Its
        # factorized grid in this test is [1,1,2,2].
        flat_owner = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

        routed, diagnostics = route_factorized_velocity(
            target,
            source,
            reconstruction,
            operators,
            torch.zeros(1, 1, 1, 5, 5),
            owner_complement_source_weight=flat_owner,
            owner_complement_margin=0,
        )

        expected_owner = torch.zeros(1, 1, 1, 5, 5, dtype=torch.bool)
        expected_owner[:, :, :, 3:, 3:] = True
        torch.testing.assert_close(
            routed[expected_owner],
            torch.full((4,), 5.0),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            routed[~expected_owner],
            torch.full((21,), 10.0),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            diagnostics["owner_complement_source_action"],
            (~expected_owner).float(),
            rtol=0.0,
            atol=0.0,
        )

    def test_owner_complement_abstains_on_unknown_pixels(self):
        roles = RoleState(
            object=torch.zeros(1, 1, 2, 2),
            boundary=torch.zeros(1, 1, 2, 2),
            hand=torch.zeros(1, 1, 2, 2),
            background=torch.ones(1, 1, 2, 2),
        )
        evidence = make_evidence(
            object_posterior=0.0,
            source_attention=0.0,
            reliability=1.0,
        )
        # This token is ambiguous rather than confirmed background.
        evidence["source_attention"] = torch.tensor(
            [[[[0.0, 1.0], [0.0, 0.0]]]]
        )
        operators = FactorizedBayesOperatorBuilder()(
            roles=roles,
            evidence=evidence,
            expected_token_length=1,
        )
        target = torch.full((1, 1, 1, 2, 2), 5.0)
        source = torch.full_like(target, 2.0)
        reconstruction = torch.full_like(target, 10.0)

        routed, diagnostics = route_factorized_velocity(
            target,
            source,
            reconstruction,
            operators,
            torch.zeros_like(target),
            owner_complement_source_weight=torch.zeros(1, 1, 2, 2),
            owner_complement_margin=0,
            owner_complement_min_preserve_confidence=0.25,
        )

        self.assertEqual(routed[0, 0, 0, 0, 0].item(), 10.0)
        self.assertEqual(routed[0, 0, 0, 0, 1].item(), 5.0)
        self.assertEqual(
            diagnostics["owner_complement_source_action"][
                0, 0, 0, 0, 1
            ].item(),
            0.0,
        )
        self.assertEqual(
            diagnostics["owner_complement_abstain_action"][
                0, 0, 0, 0, 1
            ].item(),
            1.0,
        )

    def test_mixed_roles_keep_actions_finite_and_bounded(self):
        raw = torch.tensor(
            [
                [
                    [[0.4, 0.1], [0.2, 0.3]],
                    [[0.2, 0.2], [0.1, 0.1]],
                    [[0.1, 0.6], [0.2, 0.1]],
                    [[0.3, 0.1], [0.5, 0.5]],
                ]
            ]
        )
        roles = RoleState(
            object=raw[:, 0:1],
            boundary=raw[:, 1:2],
            hand=raw[:, 2:3],
            background=raw[:, 3:4],
        )
        operators = FactorizedBayesOperatorBuilder()(
            roles=roles,
            evidence=make_evidence(
                object_posterior=0.5,
                source_attention=0.6,
                reliability=0.8,
                hand_proximity=0.4,
            ),
            expected_token_length=1,
        )
        for value in (
            operators.source_key_action,
            operators.source_value_action,
            operators.source_residual_action,
            operators.target_memory_action,
            operators.unknown_action,
        ):
            self.assertTrue(torch.isfinite(value).all())
            self.assertGreaterEqual(value.min().item(), 0.0)
            self.assertLessEqual(value.max().item(), 1.0)

    def test_unknown_attention_output_is_exact_native_fallback(self):
        torch.manual_seed(0)
        target_query = torch.randn(1, 3, 2, 4)
        source_query = torch.randn_like(target_query)
        target_key = torch.randn(1, 5, 2, 4)
        source_key = torch.randn_like(target_key)
        target_value = torch.randn(1, 5, 2, 4)
        source_value = torch.randn_like(target_value)
        key, value = fuse_factorized_aligned_memory(
            target_key,
            target_value,
            source_key,
            source_value,
            torch.zeros(1, 5),
            torch.zeros(1, 5),
            torch.zeros(1, 5),
            torch.ones(1, 5),
        )
        factorized = F.scaled_dot_product_attention(
            target_query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        ).transpose(1, 2)
        native = F.scaled_dot_product_attention(
            source_query.transpose(1, 2),
            source_key.transpose(1, 2),
            source_value.transpose(1, 2),
        ).transpose(1, 2)
        output = blend_factorized_with_native_fallback(
            factorized, native, torch.ones(1, 3)
        )
        self.assertTrue(torch.equal(output, native))

    def test_unknown_history_is_not_read_as_long_term_memory(self):
        source = torch.tensor([[0.0, 1.0, 0.25]])
        target = torch.tensor([[0.0, 0.0, 0.50]])
        readable = build_factorized_history_read_mask(source, target)
        self.assertTrue(
            torch.equal(
                readable,
                torch.tensor([[False, True, True]]),
            )
        )

    def test_attention_geometry_and_value_provenance_are_independent(self):
        target_key = torch.zeros(1, 2, 1, 1)
        source_key = torch.full_like(target_key, 10.0)
        target_value = torch.full_like(target_key, 3.0)
        source_value = torch.full_like(target_key, 9.0)
        source_key_action = torch.tensor([[1.0, 1.0]])
        source_value_action = torch.tensor([[0.0, 1.0]])
        target_memory_action = torch.tensor([[1.0, 0.0]])
        unknown_action = torch.zeros(1, 2)
        key, value = fuse_factorized_aligned_memory(
            target_key,
            target_value,
            source_key,
            source_value,
            source_key_action,
            source_value_action,
            target_memory_action,
            unknown_action,
        )
        self.assertTrue(torch.equal(key, source_key))
        self.assertEqual(value[0, 0].item(), 3.0)
        self.assertEqual(value[0, 1].item(), 9.0)


if __name__ == "__main__":
    unittest.main()
