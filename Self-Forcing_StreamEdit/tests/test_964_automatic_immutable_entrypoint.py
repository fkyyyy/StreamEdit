from __future__ import annotations

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "run_964a_automatic_immutable_target_memory.sh"


class AutomaticImmutableEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT_PATH.read_text(encoding="utf-8")
        match = re.search(
            r"^COMMAND=\(\n(?P<body>.*?)^\)\n",
            cls.text,
            flags=re.MULTILINE | re.DOTALL,
        )
        if match is None:
            raise AssertionError("964a script has no COMMAND array")
        cls.command = match.group("body")

    def test_invokes_python_directly(self):
        self.assertIn(
            '"$PYTHON_BIN" "$SCRIPT_DIR/inference_edit_streamedit.py"',
            self.command,
        )
        self.assertNotIn("exec bash", self.text)

    def test_uses_one_frozen_identity_path(self):
        for required in (
            "--first_chunk_identity_replay",
            "--factorized_immutable_target_memory",
            "--immutable_target_layers 8 12 16 20",
            "--immutable_target_value_mode absolute",
            "--motion_geometry_owner",
            "--source_flow_cache",
        ):
            self.assertIn(required, self.command)
        for competing in (
            "--factorized_native_target_history",
            "--role_fixed_native_history",
            "--native_history_multiframe_identity_sink",
            "--native_history_timestep_counterfactual_memory",
        ):
            self.assertNotIn(competing, self.command)
        self.assertNotIn("--immutable_target_hard_owner", self.command)

    def test_inference_command_contains_no_object_owner_mask(self):
        self.assertNotIn("--object_mask_video", self.command)
        self.assertNotIn("--source_owner_mask_video", self.command)
        self.assertIn("--hand_mask_video", self.command)

    def test_calculator_prompt_and_trigger_are_fixed(self):
        self.assertIn(
            "readonly TRG_WORD='handheld calculator'",
            self.text,
        )
        self.assertIn("holding a handheld calculator", self.text)
        self.assertIn("dark LCD display window", self.text)


if __name__ == "__main__":
    unittest.main()
