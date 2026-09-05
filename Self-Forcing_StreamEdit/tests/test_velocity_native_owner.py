import importlib.util
from pathlib import Path
import sys
import types

import torch


ROOT = Path(__file__).parents[1]
PACKAGE = "streamedit_f1v_pipeline"


def load_module(name: str, relative_path: str):
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(ROOT / "pipeline")]
        sys.modules[PACKAGE] = package
    full_name = f"{PACKAGE}.{name}"
    spec = importlib.util.spec_from_file_location(
        full_name, ROOT / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


load_module("control_belief", "pipeline/control_belief.py")
load_module("target_identity_memory", "pipeline/target_identity_memory.py")
ownership_module = load_module(
    "causal_ownership", "pipeline/causal_ownership.py"
)
load_module("appearance_leakage", "pipeline/appearance_leakage.py")
role_module = load_module("role_router", "pipeline/role_router.py")
factorized_module = load_module(
    "factorized_bayes", "pipeline/factorized_bayes.py"
)
CausalObjectOwnershipTracker = (
    ownership_module.CausalObjectOwnershipTracker
)
FactorizedBayesOperators = factorized_module.FactorizedBayesOperators
FactorizedRolePosterior = factorized_module.FactorizedRolePosterior
route_factorized_velocity = factorized_module.route_factorized_velocity


def source_features(order=(0, 1, 2, 3)):
    eye = torch.eye(4)
    return eye[list(order)].unsqueeze(0)


def response_at(token: int, direction=(1.0, 0.0)):
    response = torch.zeros(1, 1, 2, 2, 2)
    row, col = divmod(token, 2)
    response[0, 0, :, row, col] = torch.tensor(direction)
    return response


def propose(tracker, features, observation):
    return tracker(
        source_features=features,
        observation_weight=observation,
        source_semantic=torch.ones_like(observation),
        hand_mask=torch.zeros_like(observation, dtype=torch.bool),
        hand_proximity=torch.ones_like(observation),
        tokens_per_frame=4,
        detector_visible=torch.ones(1, 1, dtype=torch.bool),
        spatial_shape=(2, 2),
        update_state=False,
    )


def refine(tracker, ownership, features, response, *, update_state=True):
    return tracker.refine_with_velocity(
        ownership=ownership,
        source_features=features,
        edit_response=response,
        hand_mask=torch.zeros(1, 4, dtype=torch.bool),
        tokens_per_frame=4,
        spatial_shape=(2, 2),
        min_response=0.10,
        min_signature_similarity=0.0,
        transport_floor=0.25,
        signature_momentum=0.80,
        update_state=update_state,
    )


def test_velocity_response_ignites_only_the_counterfactual_token():
    tracker = CausalObjectOwnershipTracker(
        min_similarity=0.0, min_owner_weight=0.05
    )
    features = source_features()
    observation = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    ownership = refine(
        tracker, propose(tracker, features, observation), features,
        response_at(0),
    )

    assert ownership.owner_support.tolist() == [[True, False, False, False]]
    assert ownership.diagnostics[
        "velocity_owner_response_active"
    ].tolist() == [[1.0, 0.0, 0.0, 0.0]]
    assert tracker._velocity_signature_live.tolist() == [True]


def test_query_transport_moves_owner_and_velocity_signature_verifies_it():
    tracker = CausalObjectOwnershipTracker(
        min_similarity=0.0, min_owner_weight=0.05
    )
    first_features = source_features()
    first_observation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    refine(
        tracker,
        propose(tracker, first_features, first_observation),
        first_features,
        response_at(0),
    )

    # The old token-0 source feature now appears at token 1. Current semantic
    # observation deliberately points elsewhere; it cannot respawn ownership.
    moved_features = source_features((1, 0, 2, 3))
    contradictory_observation = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    ownership = refine(
        tracker,
        propose(tracker, moved_features, contradictory_observation),
        moved_features,
        response_at(1),
    )

    assert ownership.owner_support.tolist() == [[False, True, False, False]]
    assert ownership.match_confidence[0, 1] > 0.99
    assert ownership.diagnostics[
        "velocity_owner_query_cycle_confidence"
    ][0, 1] > 0.99
    assert ownership.diagnostics[
        "velocity_owner_signature_similarity"
    ][0, 1] > 0.99


def test_first_chunk_switches_from_ignition_to_query_transport():
    tracker = CausalObjectOwnershipTracker(
        min_similarity=0.0, min_owner_weight=0.05
    )
    first = source_features()
    second = source_features((1, 0, 2, 3))
    features = torch.cat([first, second], dim=1)
    # Frame 0 ignites token 0. Frame 1's semantic proposal is deliberately
    # wrong (token 2), while source-query correspondence moves token 0 to 1.
    observation = torch.tensor([[
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]])
    proposal = tracker(
        source_features=features,
        observation_weight=observation,
        source_semantic=torch.ones_like(observation),
        hand_mask=torch.zeros_like(observation, dtype=torch.bool),
        hand_proximity=torch.ones_like(observation),
        tokens_per_frame=4,
        detector_visible=torch.ones(1, 2, dtype=torch.bool),
        spatial_shape=(2, 2),
        update_state=False,
    )
    response = torch.zeros(1, 2, 2, 2, 2)
    response[0, 0, :, 0, 0] = torch.tensor([1.0, 0.0])
    response[0, 1, :, 0, 1] = torch.tensor([1.0, 0.0])

    ownership = tracker.refine_with_velocity(
        ownership=proposal,
        source_features=features,
        edit_response=response,
        hand_mask=torch.zeros(1, 8, dtype=torch.bool),
        tokens_per_frame=4,
        spatial_shape=(2, 2),
        min_response=0.10,
        min_signature_similarity=0.0,
        transport_floor=0.25,
        signature_momentum=0.80,
    )

    assert ownership.owner_support.tolist() == [[
        True, False, False, False,
        False, True, False, False,
    ]]
    # Every velocity diagnostic must span all frames, not accidentally retain
    # only the final frame-local tensor.  Inference immediately reshapes these
    # maps for the saved role diagnostics.
    assert all(
        value.shape == ownership.owner_weight.shape
        for value in ownership.diagnostics.values()
    )
    debug = ownership.as_debug_maps((1, 2, 2, 2))
    assert debug["velocity_owner_response_active"].shape == (1, 2, 2, 2)
    assert debug["velocity_owner_response_active"].sum() == 2


def test_opposite_velocity_signature_rejects_transported_owner():
    tracker = CausalObjectOwnershipTracker(
        min_similarity=0.0, min_owner_weight=0.05
    )
    features = source_features()
    observation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    refine(
        tracker, propose(tracker, features, observation), features,
        response_at(0),
    )

    moved_features = source_features((1, 0, 2, 3))
    ownership = refine(
        tracker,
        propose(tracker, moved_features, torch.zeros_like(observation)),
        moved_features,
        response_at(1, direction=(-1.0, 0.0)),
    )

    assert not ownership.owner_support.any()
    assert ownership.state_code.item() == 2


def test_diagnostic_refinement_does_not_mutate_velocity_state():
    tracker = CausalObjectOwnershipTracker(
        min_similarity=0.0, min_owner_weight=0.05
    )
    features = source_features()
    proposal = propose(
        tracker, features, torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    )

    refine(
        tracker, proposal, features, response_at(0), update_state=False
    )

    assert tracker._velocity_signature is None
    assert tracker._velocity_signature_live is None
    assert tracker.transport.previous_weight is None


def test_closed_permissions_keep_native_fallback_only_outside_owner():
    shape = (1, 1, 2, 2)
    zeros = torch.zeros(shape)
    ones = torch.ones(shape)
    roles = FactorizedRolePosterior(
        object=zeros,
        boundary=zeros,
        hand=zeros,
        background=ones,
        unknown=zeros,
        target_owned=zeros,
    )
    operators = FactorizedBayesOperators(
        roles=roles,
        source_key_action_map=ones,
        source_value_action_map=ones,
        source_residual_action_map=ones,
        target_memory_action_map=zeros,
        unknown_action_map=zeros,
        source_key_action=ones.reshape(1, -1),
        source_value_action=ones.reshape(1, -1),
        source_residual_action=ones.reshape(1, -1),
        target_memory_action=zeros.reshape(1, -1),
        unknown_action=zeros.reshape(1, -1),
    )
    source = torch.zeros(1, 1, 2, 2, 2)
    target = torch.zeros_like(source)
    target[:, :, 0] = 1.0
    reconstruction = torch.zeros_like(source)
    reconstruction[:, :, 0] = -1.0
    reconstruction[:, :, 1] = 1.0
    native_action = torch.full((1, 1, 1, 2, 2), 0.60)
    owner = torch.zeros(shape)
    owner[:, :, 0, 0] = 1.0

    routed, diagnostics = route_factorized_velocity(
        target_velocity=target,
        source_velocity=source,
        source_reconstruction_velocity=reconstruction,
        operators=operators,
        native_fallback_action=native_action,
        target_owned_weight=owner,
        block_target_owned_source=True,
        native_outside_target_owned=True,
        geometry_owner_weight=owner,
        geometry_strength=0.35,
        denoising_fraction=1.0,
    )

    # On the owner, the raw source residual is closed. Only the orthogonal
    # geometry component survives at bounded strength 0.35.
    torch.testing.assert_close(
        routed[0, 0, :, 0, 0], torch.tensor([1.0, 0.35])
    )
    assert diagnostics["source_residual_action"][0, 0, 0, 0, 0] == 0
    assert (
        diagnostics["owner_source_residual_blocked_action"][
            0, 0, 0, 0, 0
        ]
        == 1
    )

    # Outside the owner, F1V is exactly the native StreamGVE residual path,
    # not the factorized background posterior.
    expected_native = target[0, 0, :, 1, 1] + 0.60 * (
        reconstruction[0, 0, :, 1, 1] - source[0, 0, :, 1, 1]
    )
    torch.testing.assert_close(routed[0, 0, :, 1, 1], expected_native)
    assert (
        diagnostics["owner_native_complement_action"][0, 0, 0, 1, 1]
        == 0.60
    )


def test_velocity_owner_forbids_factorized_source_appearance_actions():
    shape = (1, 1, 4, 4)
    owner = torch.zeros(shape)
    owner[:, :, 1:3, 1:3] = 0.80
    roles = role_module.RoleState(
        object=owner,
        boundary=torch.zeros(shape),
        hand=torch.zeros(shape),
        background=1.0 - owner,
    )
    evidence = {
        "object_posterior": owner,
        "posterior_threshold": torch.full(shape, 0.20),
        "source_attention": owner,
        "hand_proximity": torch.zeros(shape),
        "adaptive_attention_reliability": torch.ones(shape),
        "object_visible": torch.ones(shape),
        "temporal_confidence": owner,
        "causal_owner_weight": owner,
        "causal_owner_support": owner > 0.0,
        "hand_hard_exclusion": torch.zeros(shape),
    }

    operators = factorized_module.FactorizedBayesOperatorBuilder()(
        roles=roles, evidence=evidence, expected_token_length=4
    )
    owner_tokens = torch.nn.functional.max_pool2d(
        (owner > 0.0).float().reshape(1, 1, 4, 4),
        kernel_size=2,
        stride=2,
    ).reshape(1, -1).bool()

    assert not operators.source_value_action[owner_tokens].any()
    assert not operators.source_residual_action[owner_tokens].any()
    assert operators.target_memory_action[owner_tokens].eq(1.0).all()
