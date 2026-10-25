"""Engine package structure, architecture dispatch and reload behavior.

Module: ``coffin/tests``. Imports ``engine_api`` and ``support`` only; it is the
one test module that must run with no DCU, so it may not import ``hip_ops`` or a
model family at module scope.

Constraints:
    * No weights are read and no kernel runs, so this file is the gate to run
      first after a code change, before paying the model load.
    * Model-family imports stay behind ``engine_api``; importing this test must
      not load the compiled kernel extension on a CPU-only host.

The tests pin the package registry directly. There is no flat compatibility
module in the production import graph.
"""

from __future__ import annotations

import json
import unittest

import support

import engine_api

QWEN_CONFIG = {"architectures": ["Qwen3_5ForConditionalGeneration"],
               "text_config": {"model_type": "qwen3_5_text"}}


class PackageTest(unittest.TestCase):
    """The registry exposes the model family through the new package layout."""

    def test_registered_model_is_the_qwen35_package_model(self):
        self.assertEqual(engine_api._REGISTRY["qwen3_5"].model.__module__,
                         "models.qwen35.model")

    def test_reload_rebuilds_the_registered_package_objects(self):
        """Reloading the package order must leave the registry bound to new classes."""
        before = engine_api._REGISTRY["qwen3_5"].model
        engine_api.reload()
        after = engine_api._REGISTRY["qwen3_5"].model
        self.assertEqual(after.__module__, "models.qwen35.model")
        self.assertIsNot(before, after)


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
        """Dispatching an unknown family must fail loudly; a silent fallback would load the wrong weights."""
        with self.assertRaises(RuntimeError):
            engine_api.detect_architecture({"model_type": "llama"})

    def test_describe_reports_name_and_family(self):
        """The startup log line and ``/v1/models`` both read ``describe``, so its keys are a small public contract."""
        info = engine_api.describe(support.FIXTURE_MODEL_DIR)
        self.assertEqual(info["model"], "qwen3_5")
        self.assertEqual(info["arch"], "qwen3_5")
        self.assertEqual(info["model_path"], str(support.FIXTURE_MODEL_DIR))

    def test_create_model_dispatches_and_forwards_arguments(self):
        """``create_model`` must pass every knob through to the family constructor, not re-apply its own defaults."""
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
        """Pins the loader signature every family must accept: a new kwarg added to the dispatch and not to a family breaks loading, not this test's own call."""
        seen: dict = {}

        # The family loader contract: positional ``model_dir`` plus the text
        # quantisation knobs and the vision ones, with defaults so a text-only
        # family can ignore them. The vision kwargs came with the multimodal
        # work, and this test is what keeps the dispatch and the contract in step
        # for every family, not just qwen35.
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
    """The checked-in fixture is config-shaped, so one test must read a real ``config.json``."""

    def test_dispatch_real_checkpoint(self):
        """Dispatch must recognise the deployed checkpoint's own metadata, not just the fixture."""
        config = json.loads((support.MODEL_DIR / "config.json").read_text())
        self.assertEqual(engine_api.detect_architecture(config), "qwen3_5")
        self.assertEqual(engine_api.describe(support.MODEL_DIR)["arch"], "qwen3_5")


if __name__ == "__main__":
    unittest.main()
