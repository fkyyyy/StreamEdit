from __future__ import annotations

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "run_965c_flow_token_streamgve_kv.sh"
CAUSAL_MODEL_PATH = REPO_ROOT / "wan" / "modules" / "causal_model.py"
PIPELINE_PATH = REPO_ROOT / "pipeline" / "edit_causal_inference.py"


class FlowTokenStreamGVEKVEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT_PATH.read_text(encoding="utf-8")
        cls.causal_model = CAUSAL_MODEL_PATH.read_text(encoding="utf-8")
        cls.pipeline = PIPELINE_PATH.read_text(encoding="utf-8")
        match = re.search(
            r"^COMMAND=\(\n(?P<body>.*?)^\)\n",
            cls.text,
            flags=re.MULTILINE | re.DOTALL,
        )
        if match is None:
            raise AssertionError("965c script has no COMMAND array")
        cls.command = match.group("body")

    def test_invokes_local_python_directly(self):
        self.assertIn(
            '"$PYTHON_BIN" "$SCRIPT_DIR/inference_edit_streamedit.py"',
            self.command,
        )
        self.assertNotIn("exec bash", self.text)
        self.assertNotRegex(self.text, r"\bbash\s+[^\n]*run_[0-9]")

    def test_keeps_flow_and_token_region_classifier(self):
        required = (
            "--routing_mode hand_role_factorized_causal_owner_kv",
            "--hand_query_layers 8 12 16 20",
            "--hand_field_update_mode posterior",
            "--hand_causal_evidence",
            "--hand_connected_hysteresis",
            "--motion_geometry_owner",
            '--source_flow_cache "$FLOW_CACHE"',
            "--source_flow_role_fusion",
            '--source_flow_role_weight "$FLOW_ROLE_WEIGHT"',
            "--causal_owner_consistent_kv_metadata",
            "--factorized_owner_complement_source",
            "--factorized_owner_complement_margin 1",
            "--factorized_owner_complement_min_preserve_confidence 0.8",
        )
        for fragment in required:
            self.assertIn(fragment, self.command)

    def test_selects_only_native_streamgve_target_history(self):
        self.assertIn("--factorized_native_target_history", self.command)
        forbidden = (
            "--first_chunk_identity_replay",
            "--identity_first_latent_bootstrap",
            "--object_wise_anchor_reset",
            "--target_owned_object_handoff",
            "--factorized_target_identity",
            "--factorized_immutable_target_memory",
            "--causal_paired_edit_memory",
            "--role_fixed_native_history",
            "--native_history_",
            "--factorized_orthogonal_geometry",
            "--source_coordinate_identity",
            "--source_identity_residual_carry",
            "--source_owner_residual_constraint",
            "--source_owner_geometry_envelope",
        )
        # Inspect only the command array: forbidden spellings are intentionally
        # present in the argument-rejection guard and experiment manifest.
        for fragment in forbidden:
            self.assertNotIn(fragment, self.command)

    def test_uses_no_external_object_or_source_owner_mask(self):
        self.assertIn('--hand_mask_video "$HAND_MASK"', self.command)
        self.assertNotIn("--object_mask_video", self.command)
        self.assertNotIn("--source_owner_mask_video", self.command)

    def test_keeps_calculator_prompt_and_original_rollout_contract(self):
        for fragment in (
            "readonly TRG_WORD='handheld calculator'",
            "holding a handheld calculator",
            "--fg_boost_factor 4",
            "--blend_power 2",
            "--rollout_chunk_size 21",
            "--rollout_overlap_block_num 1",
        ):
            self.assertIn(fragment, self.text)

    def test_native_flag_uses_dense_clean_target_output(self):
        # Under factorized_native_target_history with no role-fixed or paired
        # memory, the model appends the native StreamGVE output and continues.
        self.assertIn(
            '"factorized_native_target_history",\n'
            "                                False,",
            self.causal_model,
        )
        self.assertIn(
            "paired_read is None\n"
            "                                and not role_fixed_native_history",
            self.causal_model,
        )
        self.assertIn("x_list.append(native_output)", self.causal_model)
        self.assertIn(
            "attention=native_streamgve_dense_clean_target", self.pipeline
        )
        self.assertIn("immutable_kv_write=disabled", self.pipeline)

    def test_clean_target_kv_is_committed_after_each_block(self):
        self.assertIn(
            "# Step 3.3: rerun with timestep zero to update KV cache using clean context",
            self.pipeline,
        )
        self.assertIn("kv_cache=kv_cache_trg,", self.pipeline)
        self.assertIn("NATIVE_TARGET_KV_COMMIT", self.pipeline)
        self.assertIn('"immutable_write=0"', self.pipeline)


if __name__ == "__main__":
    unittest.main()
