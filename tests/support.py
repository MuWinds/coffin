"""Shared fixtures, skip gates and path setup for the engine test suite.

Module: ``coffin/tests``. Imported first by every other test module, so it may
not import any of them; it is the only test file that touches ``sys.path`` and
the only one that loads a checkpoint.

Constraints:
    * ``MKL_THREADING_LAYER=GNU`` has to be exported before torch initialises
      MKL, hence it is set between the imports rather than in a fixture.
    * ``support`` is imported by flat name, so every test module must be run
      through the ``discover -s tests`` layout that puts ``tests/`` on
      ``sys.path``; ``coffin/`` is then added here so ``ops`` / ``models`` /
      ``engine_api`` import as top-level packages.
    * At most one model is resident at a time: the card is 16 GB and a second
      copy of the 4B weights plus both KV caches does not fit, so a test
      requests its configuration through the fixture instead of building its
      own model.
    * The skip gates are evaluated at import time, so a test module that needs
      the checkpoint must import ``support`` before decorating its classes.

The suite runs in two places, and the skip rules are what make that work:

* a dev box without a DCU -- the kernel/model tests skip, the package-structure
  tests still run;
* the SCNet container -- everything runs against the real HIP kernels and the
  checkpoint at ``$SCNET_MODEL`` (default ``/root/private_data/models/Qwen3.5-4B``).

Run it with::

    cd /root/private_data/engine
    MKL_THREADING_LAYER=GNU PYTHONPATH=/root/private_data/engine \
        python3 -m unittest discover -s tests -t . -v
"""

from __future__ import annotations

import os

# Must be set before torch initialises MKL: the container ships only the GNU
# thread layer, and the default dispatch aborts with a FATAL ERROR (NOTES.md).
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import gc
import sys
import unittest
from pathlib import Path

import torch

#: ``coffin/`` is the ``sys.path`` root for everything, so ``ops`` and ``models``
#: are top-level packages.
ENGINE_DIR = Path(__file__).resolve().parent.parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

#: Checkpoint under test.  ``$SCNET_MODEL`` wins; the default is where the
#: container keeps it.
MODEL_DIR = Path(os.environ.get("SCNET_MODEL")
                 or "/root/private_data/models/Qwen3.5-4B")

#: Checked-in config-shaped fixture: lets the dispatch tests run without a
#: checkpoint (and without a writable system temp dir).
FIXTURE_MODEL_DIR = Path(__file__).resolve().parent / "fixtures" / "qwen3_5"


def has_cuda() -> bool:
    """Whether a DCU is visible; the skip gates are built from this at import time."""
    return torch.cuda.is_available()


def has_hip() -> bool:
    """True when the HIP extension is built/loadable (never builds on a CPU box).

    The import is deferred inside the function: importing ``hip_ops`` loads the
    compiled extension, which aborts on a box with no DCU driver.
    """
    if not torch.cuda.is_available():
        return False
    import hip_ops
    return hip_ops.available()


def has_model() -> bool:
    """Whether the checkpoint exists on disk; cheap enough to call at import time."""
    return (MODEL_DIR / "config.json").is_file()


#: Needs a DCU.  Pure-torch kernels still need one to hold the tensors.
needs_gpu = unittest.skipUnless(has_cuda(), "needs a DCU (torch.cuda unavailable)")
#: Needs the hand-written HIP kernels (gemv / attn / delta / norm).
needs_hip = unittest.skipUnless(has_hip(), "needs the HIP extension")
#: Needs the checkpoint on disk (also implies a DCU).
needs_model = unittest.skipUnless(
    has_cuda() and has_model(), f"needs a DCU and the checkpoint at {MODEL_DIR}")


def rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Max relative error against ``ref`` (same metric the old probes printed).

    Normalised by ``ref``'s largest magnitude so a tolerance reads the same at
    every layer depth; an absolute tolerance would have to be re-tuned per layer.
    """
    return ((got - ref).abs().max() / ref.abs().max()).item()


class ModelFixture:
    """The one resident model, rebuilt only when the requested configuration changes.

    Rebuilding is expensive (weights load, graph capture), so the key is the full
    constructor tuple: two tests sharing a configuration share the model, and a
    test that changes one knob pays one reload rather than running against a
    model configured for somebody else.
    """

    _key: tuple | None = None
    _model = None
    _weights: dict | None = None

    @classmethod
    def get(cls, *, quant: bool = True, max_seq: int = 2048,
            paged: bool = False, bench: bool = False, weight_dtype=None):
        """Return the shared model, reloading it first if the configuration differs.

        Args:
            quant: Store weights as INT8 (W8A16) rather than the checkpoint dtype.
            max_seq: KV capacity; a longer request than this is truncated by the
                model, so a test that probes a long context must raise it.
            paged: Enable the paged KV cache. The paged and flat paths allocate
                differently, so they cannot share one instance.
            bench: Allocate the benchmark-shaped buffers.
            weight_dtype: Override the weight storage dtype (``torch.float16`` /
                ``torch.bfloat16``); ``None`` keeps the checkpoint's own dtype.

        Returns:
            The live model. It is owned by this class, not by the caller: do not
            close it, call :meth:`close` only when you need the VRAM back.
        """
        key = (quant, max_seq, paged, bench, weight_dtype)
        if cls._key != key:
            cls.close()
            import engine_api
            kwargs = {} if weight_dtype is None else {"weight_dtype": weight_dtype}
            cls._weights = engine_api.load_weights(MODEL_DIR, quant=quant, **kwargs)
            cls._model = engine_api.create_model(
                MODEL_DIR, max_seq=max_seq, bench=bench,
                weights=cls._weights, paged=paged)
            cls._key = key
        return cls._model

    @classmethod
    def close(cls) -> None:
        """Drop the model and return its VRAM, so the next :meth:`get` reloads.

        Skipping this between differently-configured models would leave the
        previous one resident and the 16 GB card over-subscribed.
        """
        if cls._model is not None:
            try:
                cls._model.close()
            finally:
                cls._model = None
        cls._weights = None
        cls._key = None
        gc.collect()
        if torch.cuda.is_available():
            # A captured graph pins a private memory pool; the pool only goes back
            # to the driver once the graph objects are gone, which needs a sync +
            # collect before empty_cache can actually release the segment.
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def ensure_captured(model):
    """Capture the decode graph once; ``decode_step`` needs a replayed graph."""
    if not model.graph_ready:
        model.capture()
    return model


class ModelTestCase(unittest.TestCase):
    """Base class for tests that drive a loaded model.

    Subclasses set ``QUANT`` / ``MAX_SEQ`` / ``PAGED``; ``setUp`` gets the shared
    fixture and rewinds it so every test starts from the empty context.
    """

    QUANT = True
    MAX_SEQ = 2048
    PAGED = False
    WEIGHT_DTYPE = None

    @classmethod
    def setUpClass(cls) -> None:
        """Skip the class with a readable reason instead of failing in ``setUp``."""
        if not (has_cuda() and has_model()):
            raise unittest.SkipTest(f"needs a DCU and the checkpoint at {MODEL_DIR}")

    def setUp(self) -> None:
        """Take the shared model and rewind its context to empty."""
        self.model = ModelFixture.get(quant=self.QUANT, max_seq=self.MAX_SEQ,
                                      paged=self.PAGED,
                                      weight_dtype=self.WEIGHT_DTYPE)
        self.model.reset()

    def tearDown(self) -> None:
        """Rewind the context so the next test cannot inherit this request's state."""
        self.model.reset()
