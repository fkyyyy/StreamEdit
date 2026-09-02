from __future__ import annotations

import unittest

import numpy as np
import torch

from tests._pipeline_imports import REPO_ROOT, load_pipeline_module


target_identity_module = load_pipeline_module("target_identity_memory")
CausalIdentityReadCorridor = (
    target_identity_module.CausalIdentityReadCorridor
)


def make_corridor(**overrides):
    options = {
        "min_evidence": 0.5,
        "hand_threshold": 0.5,
        "hand_anchor_radius": 1,
        "hand_search_radius": 4,
        "hand_continuity_radius": 1,
        "temporal_radius": 2,
        "max_unobserved_frames": 2,
        "exit_hand_area_ratio": 0.5,
        "max_area_fraction": 1.0,
    }
    options.update(overrides)
    return CausalIdentityReadCorridor(**options)


def empty_inputs(*, batch=1, frames=1, height=7, width=12):
    shape = (batch, frames, height, width)
    return (
        torch.zeros(shape, dtype=torch.bool),
        torch.zeros(shape, dtype=torch.float32),
        torch.ones(shape, dtype=torch.float32),
        torch.zeros(shape, dtype=torch.float32),
    )


def run_corridor(
    corridor,
    observation,
    transported_commitment,
    transport_precision,
    hand_mask,
    *,
    update_state=True,
):
    return corridor(
        observation_mask=observation.reshape(observation.shape[0], -1),
        transported_commitment=transported_commitment,
        transport_precision=transport_precision,
        pooled_hand_mask=hand_mask,
        update_state=update_state,
    )


def clone_object_state(corridor):
    return {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in vars(corridor).items()
    }


class CausalIdentityReadCorridorTests(unittest.TestCase):
    def test_never_activated_corridor_can_bootstrap_later(self):
        corridor = make_corridor()
        observation, transported, precision, hand = empty_inputs(
            frames=2, height=7, width=10
        )
        hand[0, :, 3, 1] = 1.0
        observation[0, 1, 2:4, 2:4] = True

        result = run_corridor(
            corridor, observation, transported, precision, hand
        )

        torch.testing.assert_close(
            result.active.flatten(), torch.tensor([False, True])
        )
        torch.testing.assert_close(
            result.anchored.flatten(), torch.tensor([False, True])
        )
        self.assertFalse(result.read_mask.reshape_as(observation)[0, 0].any())
        self.assertEqual(result.connected_mask[0, 1].sum().item(), 4)

    def test_observation_selects_hand_anchored_component(self):
        corridor = make_corridor(hand_search_radius=3)
        observation, transported, precision, hand = empty_inputs()
        hand[0, 0, 3, 1] = 1.0

        expected_anchor = torch.zeros_like(observation)
        expected_anchor[0, 0, 2:4, 2] = True
        observation |= expected_anchor
        observation[0, 0, 1:4, 8:10] = True

        result = run_corridor(
            corridor, observation, transported, precision, hand
        )

        torch.testing.assert_close(result.anchor_mask, expected_anchor)
        torch.testing.assert_close(result.connected_mask, expected_anchor)
        torch.testing.assert_close(result.candidate_mask, expected_anchor)
        torch.testing.assert_close(
            result.read_mask, expected_anchor.reshape(1, -1)
        )
        self.assertTrue(result.anchored.item())
        self.assertTrue(result.active.item())
        self.assertFalse(result.connected_mask[0, 0, :, 8:10].any())

    def test_observation_without_contacting_hand_cannot_start_track(self):
        corridor = make_corridor(hand_anchor_radius=1)
        observation, transported, precision, hand = empty_inputs(
            height=7, width=12
        )
        hand[0, 0, 3, 1] = 1.0
        observation[0, 0, 2:4, 9:11] = True

        result = run_corridor(
            corridor, observation, transported, precision, hand
        )

        self.assertFalse(result.anchored.item())
        self.assertFalse(result.active.item())
        self.assertFalse(result.read_mask.any())
        self.assertFalse(corridor.ever_activated.item())
        self.assertFalse(corridor.terminated.item())

    def test_area_cap_preserves_one_connected_component(self):
        corridor = make_corridor(
            hand_search_radius=6,
            max_area_fraction=0.1,
        )
        observation, transported, precision, hand = empty_inputs(
            height=5, width=5
        )
        hand[0, 0, 4, 2] = 1.0
        observation[0, 0, 1:4, 1:4] = True

        result = run_corridor(
            corridor, observation, transported, precision, hand
        )

        selected = result.connected_mask[0, 0]
        self.assertEqual(selected.sum().item(), 3)
        self.assertEqual(len(corridor._components(selected)), 1)

    def test_dropout_expands_single_evidence_seed_to_target_area(self):
        corridor = make_corridor()
        observation, transported, precision, hand = empty_inputs(
            height=7, width=10
        )
        hand[0, 0, 3, 1] = 1.0
        observation[0, 0, 2:4, 2:4] = True
        anchored = run_corridor(
            corridor, observation, transported, precision, hand
        )
        self.assertEqual(anchored.target_area.item(), 4)

        dropout, transported, precision, hand = empty_inputs(
            height=7, width=10
        )
        hand[0, 0, 3, 1] = 1.0
        transported[0, 0, 2, 3] = 0.9
        result = run_corridor(
            corridor, dropout, transported, precision, hand
        )

        self.assertEqual(result.candidate_mask.sum().item(), 1)
        self.assertEqual(result.connected_mask.sum().item(), 4)
        self.assertTrue(result.connected_mask[0, 0, 2, 3].item())
        torch.testing.assert_close(
            result.track_only_mask, result.connected_mask
        )
        self.assertEqual(result.target_area.item(), 4)
        self.assertEqual(result.unobserved_age.item(), 1)

    def test_third_dropout_clears_and_terminal_state_does_not_reactivate(self):
        corridor = make_corridor(max_unobserved_frames=2)
        observation, transported, precision, hand = empty_inputs(
            height=7, width=10
        )
        hand[0, 0, 3, 1] = 1.0
        observation[0, 0, 2:4, 2:4] = True
        run_corridor(corridor, observation, transported, precision, hand)

        dropout, transported, precision, hand = empty_inputs(
            frames=3, height=7, width=10
        )
        hand[0, :, 3, 1] = 1.0
        transported[0, :, 2, 3] = 0.9
        result = run_corridor(
            corridor, dropout, transported, precision, hand
        )

        torch.testing.assert_close(
            result.active.flatten(),
            torch.tensor([True, True, False]),
        )
        torch.testing.assert_close(
            result.unobserved_age.flatten(),
            torch.tensor([1, 2, 0]),
        )
        self.assertEqual(result.connected_mask[0, 0].sum().item(), 4)
        self.assertEqual(result.connected_mask[0, 1].sum().item(), 4)
        self.assertFalse(result.connected_mask[0, 2].any())
        self.assertEqual(result.target_area[0, 2].item(), 0)

        observation, transported, precision, hand = empty_inputs(
            height=7, width=10
        )
        hand[0, 0, 3, 1] = 1.0
        observation[0, 0, 2:4, 2:4] = True
        attempted_reactivation = run_corridor(
            corridor, observation, transported, precision, hand
        )

        self.assertFalse(attempted_reactivation.anchored.item())
        self.assertFalse(attempted_reactivation.active.item())
        self.assertFalse(attempted_reactivation.read_mask.any())

    def test_hand_exit_is_terminal_and_cannot_switch_to_another_hand(self):
        corridor = make_corridor(hand_continuity_radius=1)
        observation, transported, precision, hand = empty_inputs(
            height=7, width=12
        )
        hand[0, 0, 3, 1] = 1.0
        observation[0, 0, 2:4, 2:4] = True
        anchored = run_corridor(
            corridor, observation, transported, precision, hand
        )
        self.assertTrue(anchored.active.item())

        dropout, transported, precision, other_hand = empty_inputs(
            height=7, width=12
        )
        other_hand[0, 0, 3, 9] = 1.0
        transported[0, 0, 3, 8] = 1.0
        exited = run_corridor(
            corridor, dropout, transported, precision, other_hand
        )
        self.assertFalse(exited.active.item())
        self.assertFalse(exited.read_mask.any())

        observation, transported, precision, other_hand = empty_inputs(
            height=7, width=12
        )
        other_hand[0, 0, 3, 9] = 1.0
        observation[0, 0, 2:4, 8:10] = True
        attempted_switch = run_corridor(
            corridor, observation, transported, precision, other_hand
        )

        self.assertFalse(attempted_switch.anchored.item())
        self.assertFalse(attempted_switch.active.item())
        self.assertFalse(attempted_switch.read_mask.any())

    def test_prediction_is_side_effect_free_and_matches_commit(self):
        corridor = make_corridor()
        observation, transported, precision, hand = empty_inputs(
            height=7, width=10
        )
        hand[0, 0, 3, 1] = 1.0
        observation[0, 0, 2:4, 2:4] = True
        run_corridor(corridor, observation, transported, precision, hand)

        dropout, transported, precision, hand = empty_inputs(
            height=7, width=10
        )
        hand[0, 0, 3, 1] = 1.0
        transported[0, 0, 2, 3] = 0.9
        state_before = clone_object_state(corridor)

        prediction = run_corridor(
            corridor,
            dropout,
            transported,
            precision,
            hand,
            update_state=False,
        )

        self.assertEqual(set(vars(corridor)), set(state_before))
        for name, expected in state_before.items():
            actual = vars(corridor)[name]
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(actual, expected)
            else:
                self.assertEqual(actual, expected)

        committed = run_corridor(
            corridor, dropout, transported, precision, hand
        )
        for name in prediction.__dataclass_fields__:
            torch.testing.assert_close(
                getattr(prediction, name), getattr(committed, name)
            )
        self.assertEqual(corridor.unobserved_age.item(), 1)

    def test_dropout_rejects_remote_scattered_evidence(self):
        corridor = make_corridor(
            hand_search_radius=4, temporal_radius=2
        )
        observation, transported, precision, hand = empty_inputs(
            height=8, width=14
        )
        hand[0, 0, 4, 1] = 1.0
        observation[0, 0, 3:5, 2:4] = True
        run_corridor(corridor, observation, transported, precision, hand)

        dropout, transported, precision, hand = empty_inputs(
            height=8, width=14
        )
        hand[0, 0, 4, 1] = 1.0
        transported[0, 0, 3, 3] = 0.6
        remote_points = ((0, 12), (2, 9), (6, 11))
        for row, column in remote_points:
            transported[0, 0, row, column] = 1.0

        result = run_corridor(
            corridor, dropout, transported, precision, hand
        )

        for row, column in remote_points:
            self.assertEqual(
                result.track_evidence[0, 0, row, column].item(), 1.0
            )
            self.assertFalse(
                result.candidate_mask[0, 0, row, column].item()
            )
            self.assertFalse(
                result.connected_mask[0, 0, row, column].item()
            )
        self.assertTrue(result.connected_mask[0, 0, 3, 3].item())
        self.assertEqual(result.connected_mask.sum().item(), 4)

    def test_batch_items_are_isolated_and_output_shapes_are_stable(self):
        corridor = make_corridor()
        batch, frames, height, width = 2, 2, 6, 9
        observation, transported, precision, hand = empty_inputs(
            batch=batch, frames=frames, height=height, width=width
        )
        hand[0, :, 3, 1] = 1.0
        observation[0, 0, 2:4, 2] = True
        transported[0, 1, 2, 2] = 0.9

        hand[1, :, 3, 6] = 1.0
        transported[1, :, 3, 5] = 1.0
        result = run_corridor(
            corridor, observation, transported, precision, hand
        )

        self.assertEqual(
            result.read_mask.shape, (batch, frames * height * width)
        )
        for name in (
            "track_evidence",
            "candidate_mask",
            "connected_mask",
            "track_only_mask",
            "anchor_mask",
            "tracked_hand_mask",
        ):
            self.assertEqual(
                getattr(result, name).shape,
                (batch, frames, height, width),
            )
        for name in (
            "anchored",
            "active",
            "terminated",
            "unobserved_age",
            "target_area",
        ):
            self.assertEqual(
                getattr(result, name).shape, (batch, frames, 1, 1)
            )
        torch.testing.assert_close(
            result.read_mask, result.connected_mask.reshape(batch, -1)
        )
        torch.testing.assert_close(
            result.active[0].flatten(), torch.tensor([True, True])
        )
        self.assertFalse(result.active[1].any())
        self.assertFalse(result.connected_mask[1].any())
        self.assertFalse(corridor.previous_mask[1].any())

    def test_rejects_invalid_shapes_and_incompatible_saved_state(self):
        observation = torch.zeros(1, 20, dtype=torch.bool)
        transported = torch.zeros(1, 1, 4, 5)
        precision = torch.ones_like(transported)
        hand = torch.zeros_like(transported)

        invalid_calls = (
            (
                "observation rank",
                observation.reshape(1, 1, 20),
                transported,
                precision,
                hand,
                r"observation_mask must have shape \[B,L\]",
            ),
            (
                "transport rank",
                observation,
                transported[:, 0],
                precision[:, 0],
                hand[:, 0],
                r"transported_commitment must have shape \[B,T,H,W\]",
            ),
            (
                "precision shape",
                observation,
                transported,
                precision[:, :, :, :-1],
                hand,
                "transport_precision must match transported_commitment",
            ),
            (
                "hand shape",
                observation,
                transported,
                precision,
                hand[:, :, :-1],
                "pooled_hand_mask must match transported_commitment",
            ),
            (
                "flattened length",
                observation[:, :-1],
                transported,
                precision,
                hand,
                "observation_mask and transported commitment must align",
            ),
        )
        for (
            label,
            invalid_observation,
            invalid_transport,
            invalid_precision,
            invalid_hand,
            message,
        ) in invalid_calls:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    make_corridor()(
                        observation_mask=invalid_observation,
                        transported_commitment=invalid_transport,
                        transport_precision=invalid_precision,
                        pooled_hand_mask=invalid_hand,
                    )

        corridor = make_corridor()
        corridor(
            observation_mask=observation,
            transported_commitment=transported,
            transport_precision=precision,
            pooled_hand_mask=hand,
        )
        larger = torch.zeros(1, 1, 5, 5)
        with self.assertRaisesRegex(
            ValueError, "state is incompatible with input"
        ):
            corridor(
                observation_mask=torch.zeros(1, 25, dtype=torch.bool),
                transported_commitment=larger,
                transport_precision=torch.ones_like(larger),
                pooled_hand_mask=torch.zeros_like(larger),
            )


class CausalIdentityReadCorridorReplayTests(unittest.TestCase):
    def test_replays_908_debug_artifacts(self):
        artifact_root = (
            REPO_ROOT / "outputs" / "908_persistent_target_track" / "roles"
        )
        paths = [
            artifact_root / f"block_{block_index:03d}_hand_role_debug.npz"
            for block_index in range(7)
        ]
        if not all(path.is_file() for path in paths):
            self.skipTest("908 hand-role debug artifacts are not available")

        required_keys = (
            "identity_read_gate_observation",
            "commitment_transport",
            "commitment_transport_precision",
            "hand_probability",
        )
        corridor = CausalIdentityReadCorridor()
        results = []
        for path in paths:
            with np.load(path) as artifact:
                missing = [key for key in required_keys if key not in artifact]
                if missing:
                    self.skipTest(
                        f"908 artifact {path.name} lacks {missing}"
                    )
                observation, transported, precision, hand = (
                    torch.from_numpy(artifact[key]) for key in required_keys
                )
                results.append(
                    run_corridor(
                        corridor,
                        observation,
                        transported,
                        precision,
                        hand,
                    )
                )

        self.assertEqual(
            results[4].connected_mask[0, 2].sum().item(), 18
        )
        torch.testing.assert_close(
            results[5].connected_mask.sum(dim=(-2, -1))[0],
            torch.tensor([18, 18, 0]),
        )
        torch.testing.assert_close(
            results[6].connected_mask.sum(dim=(-2, -1))[0],
            torch.tensor([0, 0, 0]),
        )

        early_dropout = results[5].connected_mask[0, :2]
        coordinates = torch.nonzero(early_dropout, as_tuple=False)
        minimum_x = int(coordinates[:, -1].min().item())
        maximum_x = int(coordinates[:, -1].max().item())
        self.assertGreaterEqual(minimum_x, 31)
        self.assertLessEqual(minimum_x, 33)
        self.assertGreaterEqual(maximum_x, 38)
        self.assertLessEqual(maximum_x, 40)


if __name__ == "__main__":
    unittest.main()
