from __future__ import annotations

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "run_966a_flow_verified_region_streamgve_kv.sh"
PIPELINE_PATH = REPO_ROOT / "pipeline" / "edit_causal_inference.py"


class FlowVerifiedStreamGVEEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT_PATH.read_text(encoding="utf-8")
        cls.pipeline = PIPELINE_PATH.read_text(encoding="utf-8")
        match = re.search(
            r"^COMMAND=\(\n(?P<body>.*?)^\)\n",
            cls.text,
            flags=re.MULTILINE | re.DOTALL,
        )
        if match is None:
            raise AssertionError("966a script has no COMMAND array")
        cls.command = match.group("body")

    def test_invokes_python_directly(self):
        self.assertIn(
            '"$PYTHON_BIN" "$SCRIPT_DIR/inference_edit_streamedit.py"',
            self.command,
        )
        self.assertNotIn("exec bash", self.text)
        self.assertNotRegex(self.text, r"\bbash\s+[^\n]*run_[0-9]")

    def test_changes_only_generic_region_fusion_from_965c(self):
        required = (
            "--routing_mode hand_role_factorized_causal_owner_kv",
            "--motion_geometry_owner",
            "--source_flow_role_fusion",
            "--source_flow_verified_region",
            '--source_flow_verified_owner_radius "$VERIFIED_OWNER_RADIUS"',
            '--source_flow_background_veto_threshold "$BACKGROUND_VETO_THRESHOLD"',
            '--source_flow_background_veto_min_confidence "$BACKGROUND_VETO_MIN_CONFIDENCE"',
            "--factorized_native_target_history",
            "--rollout_chunk_size 21",
            "--rollout_overlap_block_num 1",
        )
        for fragment in required:
            self.assertIn(fragment, self.command)

    def test_has_generic_defaults_selected_offline(self):
        self.assertIn('VERIFIED_OWNER_RADIUS="${VERIFIED_OWNER_RADIUS:-1}"', self.text)
        self.assertIn('BACKGROUND_VETO_THRESHOLD="${BACKGROUND_VETO_THRESHOLD:-0.55}"', self.text)
        self.assertIn('BACKGROUND_VETO_MIN_CONFIDENCE="${BACKGROUND_VETO_MIN_CONFIDENCE:-0.50}"', self.text)

    def test_uses_no_external_object_or_source_owner_mask(self):
        self.assertIn('--hand_mask_video "$HAND_MASK"', self.command)
        self.assertNotIn("--object_mask_video", self.command)
        self.assertNotIn("--source_owner_mask_video", self.command)
        self.assertNotIn("$PHONE_GT", self.command)

    def test_phone_gt_occurs_only_after_inference(self):
        command_end = self.text.index("\n)\n", self.text.index("COMMAND=("))
        inference_call = self.text.index("CUDA_VISIBLE_DEVICES=")
        gt_replay_call = self.text.index(
            "tools/replay_flow_verified_region_phone_gt.py"
        )
        self.assertLess(command_end, inference_call)
        self.assertLess(inference_call, gt_replay_call)

    def test_keeps_calculator_prompt(self):
        self.assertIn("readonly TRG_WORD='handheld calculator'", self.text)
        self.assertIn("holding a handheld calculator", self.text)

    def test_disables_later_identity_and_memory_operators(self):
        forbidden = (
            "--causal_paired_edit_memory",
            "--role_fixed_native_history",
            "--native_history_",
            "--factorized_immutable_target_memory",
            "--target_semantic_competition",
        )
        for fragment in forbidden:
            self.assertNotIn(fragment, self.command)

    def test_pipeline_does_not_reunion_owner_in_verified_mode(self):
        self.assertIn(
            "if not source_flow_verified_region:\n"
            "                        role_edit_tokens = (",
            self.pipeline,
        )


if __name__ == "__main__":
    unittest.main()
