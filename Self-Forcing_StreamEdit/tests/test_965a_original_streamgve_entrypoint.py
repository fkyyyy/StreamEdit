from __future__ import annotations

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "run_965a_original_streamgve_kv.sh"


class OriginalStreamGVEEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT_PATH.read_text(encoding="utf-8")
        match = re.search(
            r"^COMMAND=\(\n(?P<body>.*?)^\)\n",
            cls.text,
            flags=re.MULTILINE | re.DOTALL,
        )
        if match is None:
            raise AssertionError("965a script has no COMMAND array")
        cls.command = match.group("body")

    def test_invokes_original_streamgve_python_directly(self):
        self.assertIn(
            "/opt/tiger/CausalForcing/StreamGVE/Self-Forcing_StreamEdit",
            self.text,
        )
        self.assertIn(
            '"$PYTHON_BIN" "$STREAMGVE_ENTRYPOINT"',
            self.command,
        )
        self.assertNotIn("$SCRIPT_DIR/inference_edit_streamedit.py", self.command)
        self.assertNotIn("exec bash", self.text)

    def test_uses_original_rollout_contract(self):
        for required in (
            "--fg_boost_factor 4",
            "--blend_power 2",
            "--rollout_chunk_size 21",
            "--rollout_overlap_block_num 1",
        ):
            self.assertIn(required, self.command)

    def test_contains_no_experimental_memory_or_mask_arguments(self):
        forbidden_fragments = (
            "--hand_mask_video",
            "--object_mask_video",
            "--source_owner_mask_video",
            "--routing_mode",
            "--factorized_immutable_target_memory",
            "--factorized_native_target_history",
            "--role_fixed_native_history",
            "--native_history_",
            "--motion_geometry_owner",
        )
        for forbidden in forbidden_fragments:
            self.assertNotIn(forbidden, self.command)

    def test_calculator_prompt_and_trigger_are_fixed(self):
        self.assertIn(
            "readonly TRG_WORD='handheld calculator'",
            self.text,
        )
        self.assertIn("holding a handheld calculator", self.text)
        self.assertIn("dark LCD display window", self.text)


if __name__ == "__main__":
    unittest.main()
