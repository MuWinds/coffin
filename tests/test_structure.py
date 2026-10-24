"""Engine package structure: imports, compatibility shim, architecture dispatch.

No weights are read and no kernel runs, so these tests run on a box without a
DCU too -- run them first after a code change, before paying the model load.

Port of ``diag/check_modules.py`` (now covered here).
"""

from __future__ import annotations

import json
import unittest

import support

import engine_api
import qwen35_engine

#: Flat names the older code imports from ``qwen35_engine`` (``import
#: qwen35_engine as E`` then ``E.chunked_delta`` / ``E.Qwen35`` / ...).  Listed
#: here rather than in the shim so that dropping one is a test failure.
LEGACY_NAMES = (
    "D", "WF", "NEG_INF", "CKPT",
    "PREFILL_ATTN_TILE", "PREFILL_ATTN_KB", "PREFILL_MTP_CHUNK", "ATTN_SPLIT",
    "GEMV", "GEMVGroup", "TiedEmbedding", "RMSNorm", "RMSNormGated",
    "make_rope_tables", "rotate_half", "apply_rope",
    "_upper_mask", "chunked_delta",
    "MLP", "Layer", "LinearAttn", "FullAttn", "MTP", "Qwen35", "load_weights",
    "hip_ops",
)

QWEN_CONFIG = {"architectures": ["Qwen3_5ForConditionalGeneration"],
               "text_config": {"model_type": "qwen3_5_text"}}


class ShimTest(unittest.TestCase):
    """The compatibility module must keep the old flat surface working."""

    def test_legacy_names_are_exported(self):
        missing = [name for name in LEGACY_NAMES if not hasattr(qwen35_engine, name)]
        self.assertEqual(missing, [], f"shim is missing {missing}")

    def test_hip_ops_module_is_reachable_as_E_hip_ops(self):
        import hip_ops
        self.assertIs(qwen35_engine.hip_ops, hip_ops)

    def test_reload_rebinds_shim_to_the_reloaded_registry(self):
        engine_api.reload()
        qwen35_engine.reload()
        self.assertIs(qwen35_engine.Qwen35,
                      engine_api._REGISTRY["qwen3_5"].model)


class RegistryTest(unittest.TestCase):
    """Checkpoint config -> implementation; the only model-specific switchboard."""

    def test_qwen35_is_registered(self):
        self.assertIn("qwen3_5", engine_api.registered())

    def test_detect_from_model_type_and_architectures(self):
        self.assertEqual(engine_api.detect_architecture(QWEN_CONFIG), "qwen3_5")

    def test_detect_from_text_geometry_when_metadata_is_missing(self):
        self.assertEqual(
            engine_api.detect_architecture({"text_config": {"linear_num_value_heads": 16}}),
            "qwen3_5")

    def test_unknown_architecture_raises_instead_of_guessing(self):
        with self.assertRaises(RuntimeError):
            engine_api.detect_architecture({"model_type": "llama"})

    def test_describe_reports_name_and_family(self):
        info = engine_api.describe(support.FIXTURE_MODEL_DIR)
        self.assertEqual(info["model"], "qwen3_5")
        self.assertEqual(info["arch"], "qwen3_5")
        self.assertEqual(info["model_path"], str(support.FIXTURE_MODEL_DIR))

    def test_create_model_dispatches_and_forwards_arguments(self):
        seen: dict = {}

        class _Dummy:
            def __init__(self, model_dir, max_seq=4096, bench=True,
                         weights=None, paged=False):
                seen["ctor"] = (model_dir, max_seq, bench, weights, paged)

        real = engine_api._REGISTRY["qwen3_5"]
        engine_api._REGISTRY["qwen3_5"] = engine_api._Entry(_Dummy, real.load_weights)
        try:
            model = engine_api.create_model(support.FIXTURE_MODEL_DIR, max_seq=8192,
                                            bench=False, paged=True)
        finally:
            engine_api._REGISTRY["qwen3_5"] = real
        self.assertIsInstance(model, _Dummy)
        self.assertEqual(seen["ctor"][1:], (8192, False, None, True))

    def test_load_weights_dispatches_to_the_family_loader(self):
        seen: dict = {}

        # The family loader contract: positional ``model_dir`` plus the text
        # quantisation knobs and the vision ones (defaults so a text-only family
        # can ignore them).  The vision kwargs are new -- the multimodal work
        # added them, and this test is what keeps the dispatch and the contract
        # in step for every family, not just qwen35.
        def _fake_loader(model_dir, quant=False, weight_dtype=None,
                         vision=None, vision_quant=False):
            seen["call"] = (model_dir, quant, weight_dtype, vision, vision_quant)
            return {"ok": True}

        real = engine_api._REGISTRY["qwen3_5"]
        engine_api._REGISTRY["qwen3_5"] = engine_api._Entry(real.model, _fake_loader)
        try:
            weights = engine_api.load_weights(support.FIXTURE_MODEL_DIR, quant=True)
        finally:
            engine_api._REGISTRY["qwen3_5"] = real
        self.assertEqual(weights, {"ok": True})
        self.assertEqual(seen["call"][1:], (True, None, None, False))


@unittest.skipUnless(support.has_model(), f"no checkpoint at {support.MODEL_DIR}")
class RealCheckpointTest(unittest.TestCase):
    def test_dispatch_real_checkpoint(self):
        config = json.loads((support.MODEL_DIR / "config.json").read_text())
        self.assertEqual(engine_api.detect_architecture(config), "qwen3_5")
        self.assertEqual(engine_api.describe(support.MODEL_DIR)["arch"], "qwen3_5")


if __name__ == "__main__":
    unittest.main()
