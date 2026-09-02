from __future__ import annotations

import unittest

import torch

from tests._pipeline_imports import load_pipeline_module


causal_vae_frame_groups = (
    load_pipeline_module("mask_alignment").causal_vae_frame_groups
)
project_visible_owner_to_causal_latents = (
    load_pipeline_module("mask_alignment")
    .project_visible_owner_to_causal_latents
)
project_hand_evidence_to_causal_latents = (
    load_pipeline_module("mask_alignment")
    .project_hand_evidence_to_causal_latents
)


class CausalVaeFrameGroupTests(unittest.TestCase):
    def test_wan_eighty_one_frames_map_to_twenty_one_latents(self):
        groups = causal_vae_frame_groups(81, 21)

        self.assertEqual(groups[0], (0, 1))
        self.assertEqual(groups[1], (1, 5))
        self.assertEqual(groups[2], (5, 9))
        self.assertEqual(groups[3], (9, 13))
        self.assertEqual(groups[-1], (77, 81))

    def test_groups_cover_each_pixel_frame_exactly_once(self):
        groups = causal_vae_frame_groups(13, 4)
        covered = [
            frame
            for left, right in groups
            for frame in range(left, right)
        ]

        self.assertEqual(covered, list(range(13)))

    def test_incompatible_shapes_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "causal VAE mapping"):
            causal_vae_frame_groups(81, 20, temporal_stride=4)


class VisibleOwnerProjectionTests(unittest.TestCase):
    def test_removes_hand_before_temporal_pooling(self):
        owner = torch.zeros(5, 1, 2, 2, dtype=torch.bool)
        owner[1:, 0, 0, :] = True
        hand = torch.zeros_like(owner)
        hand[1, 0, 0, 0] = True
        hand[2, 0, 0, 1] = True

        projected = project_visible_owner_to_causal_latents(
            owner,
            hand,
            latent_frames=2,
            latent_spatial_shape=(2, 2),
        )

        # Each owner pixel is visible in part of the four-frame causal group.
        # Pooling object and hand separately would incorrectly erase both.
        self.assertTrue(projected[1, 0, 0])
        self.assertTrue(projected[1, 0, 1])

    def test_first_causal_frame_keeps_exact_visible_owner(self):
        owner = torch.zeros(5, 1, 2, 2, dtype=torch.bool)
        owner[0, 0, 0, :] = True
        hand = torch.zeros_like(owner)
        hand[0, 0, 0, 0] = True

        projected = project_visible_owner_to_causal_latents(
            owner,
            hand,
            latent_frames=2,
            latent_spatial_shape=(2, 2),
        )

        self.assertFalse(projected[0, 0, 0])
        self.assertTrue(projected[0, 0, 1])


class CausalHandEvidenceTests(unittest.TestCase):
    def test_moving_hand_union_is_not_hard_exclusion(self):
        hand = torch.zeros(5, 1, 2, 4)
        hand[0, 0, 1, 0] = 1.0
        for frame_index in range(1, 5):
            hand[frame_index, 0, 0, frame_index - 1] = 1.0

        evidence = project_hand_evidence_to_causal_latents(
            hand,
            latent_frames=2,
            latent_spatial_shape=(2, 4),
            persistent_occupancy=1.0,
        )

        # The first latent represents one exact frame.
        self.assertTrue(evidence.persistent[0, 1, 0])
        # The later latent sees the whole hand sweep as proximity, while each
        # location is only 1/4 occupied and none is a hard exclusion.
        self.assertTrue(evidence.union[1, 0].all())
        torch.testing.assert_close(
            evidence.occupancy[1, 0], torch.full((4,), 0.25)
        )
        self.assertFalse(evidence.persistent[1].any())

    def test_persistent_threshold_can_retain_stable_hand_core(self):
        hand = torch.zeros(5, 1, 2, 2)
        hand[1:, 0, 0, 0] = 1.0
        hand[1:4, 0, 0, 1] = 1.0

        evidence = project_hand_evidence_to_causal_latents(
            hand,
            latent_frames=2,
            latent_spatial_shape=(2, 2),
            persistent_occupancy=0.75,
        )

        self.assertTrue(evidence.persistent[1, 0, 0])
        self.assertTrue(evidence.persistent[1, 0, 1])


if __name__ == "__main__":
    unittest.main()
