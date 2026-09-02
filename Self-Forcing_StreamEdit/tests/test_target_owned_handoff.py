from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch

from tests._pipeline_imports import REPO_ROOT, load_pipeline_module


def load_attention_module():
    path = REPO_ROOT / "wan" / "modules" / "attention.py"
    spec = importlib.util.spec_from_file_location(
        "streamedit_attention_helpers",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


attention_module = load_attention_module()
identity_module = load_pipeline_module("target_identity_memory")
role_router_module = load_pipeline_module("role_router")

blend_target_owned_tensor = (
    attention_module.blend_target_owned_tensor
)
build_target_owned_source_background_mask = (
    attention_module.build_target_owned_source_background_mask
)
fuse_aligned_memory = attention_module.fuse_aligned_memory
scatter_target_owned_output = (
    attention_module.scatter_target_owned_output
)
suppress_source_preserve_on_target_owned_history = (
    attention_module.suppress_source_preserve_on_target_owned_history
)
SlowTargetIdentityMemory = identity_module.SlowTargetIdentityMemory
BayesResidualFlowRouter = role_router_module.BayesResidualFlowRouter
CausalControlBelief = (
    load_pipeline_module("control_belief").CausalControlBelief
)


class TargetOwnedAttentionTests(unittest.TestCase):
    def setUp(self):
        self.target = torch.tensor(
            [[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]]
        )
        self.source = torch.tensor(
            [[[[10.0]], [[20.0]], [[30.0]], [[40.0]]]]
        )
        self.owned = torch.tensor([[True, False, True, False]])
        self.target_weight = 0.25

    def test_owned_query_and_current_key_are_source_invariant(self):
        source_perturbed = self.source + 1000.0

        first = blend_target_owned_tensor(
            self.target,
            self.source,
            self.target_weight,
            self.owned,
        )
        second = blend_target_owned_tensor(
            self.target,
            source_perturbed,
            self.target_weight,
            self.owned,
        )

        torch.testing.assert_close(
            first[self.owned],
            self.target[self.owned],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            second[self.owned],
            self.target[self.owned],
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(
            torch.equal(first[~self.owned], second[~self.owned])
        )

    def test_missing_ownership_mask_is_exact_legacy_blend(self):
        expected = (
            self.target * self.target_weight
            + self.source * (1.0 - self.target_weight)
        )

        actual = blend_target_owned_tensor(
            self.target,
            self.source,
            self.target_weight,
        )

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_owned_history_cannot_select_source_kv(self):
        target_key = self.target.clone()
        target_value = self.target + 0.5
        source_key = self.source.clone()
        source_value = self.source + 0.5
        preserve = torch.tensor([[1.0, 0.6, 0.8, 0.2]])

        owned_preserve = (
            suppress_source_preserve_on_target_owned_history(
                preserve,
                self.owned,
            )
        )
        key, value = fuse_aligned_memory(
            target_key,
            target_value,
            source_key,
            source_value,
            owned_preserve,
        )

        torch.testing.assert_close(
            key[self.owned],
            target_key[self.owned],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            value[self.owned],
            target_value[self.owned],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            owned_preserve[~self.owned],
            preserve[~self.owned],
            rtol=0.0,
            atol=0.0,
        )

    def test_source_background_append_excludes_owned_tokens(self):
        selected = build_target_owned_source_background_mask(
            current_edit_mask=torch.tensor(
                [False, False, True, False]
            ),
            current_target_owned_mask=self.owned[0],
        )

        torch.testing.assert_close(
            selected,
            torch.tensor([False, True, False, True]),
        )

    def test_background_output_is_bitwise_legacy(self):
        legacy = torch.arange(4.0).reshape(1, 4, 1, 1)
        isolated = torch.tensor([[[[100.0]], [[102.0]]]])

        selected = scatter_target_owned_output(
            legacy,
            isolated,
            self.owned,
        )

        torch.testing.assert_close(
            selected[~self.owned],
            legacy[~self.owned],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            selected[self.owned],
            isolated.flatten(0, 1),
            rtol=0.0,
            atol=0.0,
        )


class TargetOwnedStateTests(unittest.TestCase):
    def test_only_anchor_matches_become_owned(self):
        memory = SlowTargetIdentityMemory(layers=(0,))
        anchor_features = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]
        )
        memory.commit_target_owned_anchor(
            anchor_mask=torch.tensor([[True, False, False]]),
            anchor_features=anchor_features,
        )
        current_features = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [1.0, 0.0]]]
        )
        candidates = torch.tensor([[True, True, True, False]])

        owned = memory.match_target_owned_tokens(
            current_features,
            candidates,
            min_similarity=0.9,
        )

        torch.testing.assert_close(
            owned,
            torch.tensor([[True, False, True, False]]),
        )

    def test_rollout_overlap_reads_only_recent_target_history(self):
        memory = SlowTargetIdentityMemory(layers=(0,))
        memory.record_target_owned_tokens(
            torch.tensor([[True, False, False]])
        )
        memory.record_target_owned_tokens(
            torch.tensor([[False, True, True]])
        )

        recent = memory.recent_target_owned_tokens(
            4,
            batch_size=1,
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(
            recent,
            torch.tensor([[False, False, True, True]]),
        )


class TargetOwnedRouterTests(unittest.TestCase):
    def test_owned_region_cannot_receive_source_residual(self):
        shape = (1, 1, 1, 2)
        zeros = torch.zeros(shape)
        belief = CausalControlBelief(
            edit_belief=zeros.clone(),
            preserve_belief=torch.ones(shape),
            edit_precision=zeros.clone(),
            preserve_precision=torch.ones(shape),
            visibility=torch.ones(shape),
            uncertainty=zeros.clone(),
            conflict=zeros.clone(),
        )
        target = torch.full((1, 1, 1, 2, 4), 3.0)
        source = torch.full_like(target, 10.0)
        reconstruction = torch.full_like(target, 14.0)

        routed, diagnostics = BayesResidualFlowRouter()(
            target_velocity=target,
            source_velocity=source,
            source_reconstruction_velocity=reconstruction,
            belief=belief,
            target_owned_mask=torch.tensor(
                [[[[True, False]]]]
            ),
        )

        expected = target.clone()
        expected[..., 2:] += 4.0
        torch.testing.assert_close(routed, expected)
        torch.testing.assert_close(
            diagnostics["preserve_action_weight"],
            torch.tensor(
                [[[[[0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0]]]]]
            ),
        )

    def test_missing_owned_mask_keeps_legacy_router_exact(self):
        shape = (1, 1, 1, 1)
        belief = CausalControlBelief(
            edit_belief=torch.full(shape, 0.25),
            preserve_belief=torch.full(shape, 0.75),
            edit_precision=torch.ones(shape),
            preserve_precision=torch.ones(shape),
            visibility=torch.ones(shape),
            uncertainty=torch.zeros(shape),
            conflict=torch.zeros(shape),
        )
        target = torch.full((1, 1, 1, 2, 2), 3.0)
        source = torch.full_like(target, 10.0)
        reconstruction = torch.full_like(target, 14.0)
        expected = target + 0.75 * (reconstruction - source)

        routed, _ = BayesResidualFlowRouter()(
            target_velocity=target,
            source_velocity=source,
            source_reconstruction_velocity=reconstruction,
            belief=belief,
        )

        torch.testing.assert_close(routed, expected)


if __name__ == "__main__":
    unittest.main()
