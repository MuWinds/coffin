"""The serving-facing surface of the engine: one checkpoint in, one session out.

The engine used to be a single module.  It is split by responsibility now:

    ops/                 device primitives shared by any model (GEMV, norms,
                         RoPE, the chunked delta rule, the tied embedding)
    models/<family>/     one architecture family: blocks, MTP head, weight
                         loader, and the model class that owns the graphs
    engine_api.py        this file: the ``Model`` contract, checkpoint -> model
                         dispatch, and the reload used by ``serve.py``

``qwen35_engine.py`` is only a compatibility shim for the old flat import; the
``diag/`` probes still use it, new code should not.

Adding a model family is a new directory under ``models/`` plus one
``register()`` call at the bottom of this file.  What the family has to provide
is the ``Model`` protocol -- the methods the server and the prefix-cache engine
actually call -- plus ``PagedModel`` when its layers carry a recurrent state.

Dispatch is by checkpoint config, not by a flag: :func:`create_model` reads
``config.json`` and matches a registered key against ``model_type`` /
``architectures``.  The key ``"qwen3_5"`` therefore matches both
``Qwen3_5ForConditionalGeneration`` and a bare ``qwen3_5``.  ``architecture=``
overrides the detection for a checkpoint whose metadata lies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


class Model(Protocol):
    """One loaded model, driven one request at a time.

    The server owns the tokenizer, the socket and the request loop; everything
    below the token ids is the model's business.  Attributes that the serving
    layer reads directly:

    ``max_seq``
        Context window the KV cache and rope tables were built for.
    ``pos1``
        Length-1 device tensor holding the current position.  The server and
        ``PrefixCacheEngine`` read and write it to move the window, which is
        why it is part of the contract rather than a private detail.
    ``mtp``
        Drafter object, or ``None`` when the checkpoint has no MTP head.
    ``verify_c``
        Verify-graph width: ``spec_generate`` can accept at most ``verify_c - 1``
        drafts per step.
    """

    max_seq: int
    pos1: "torch.Tensor"
    verify_c: int
    mtp: Any | None

    def reset(self) -> None:
        """Rewind every piece of per-sequence state to the empty context."""

    def prefill(self, tokens: list[int], chunk: int = 2048) -> "torch.Tensor":
        """Process a prompt; returns the logits for the next token."""

    def decode_step(self, token: int) -> "torch.Tensor":
        """Advance one token through the captured decode graph."""

    def logits(self, token: int) -> "torch.Tensor":
        """Un-captured single step (prefill fallback / debugging)."""

    def capture(self, warmup: int = 3) -> None:
        """Capture the decode step (and, when present, the drafter) once."""

    @property
    def graph_ready(self) -> bool:
        """True once :meth:`capture` has produced a replayable decode graph."""

    def capture_verify(self, c: int = 4, warmup: int = 3) -> None:
        """Capture the ``c``-token verification forward."""

    def spec_generate(self, prompt: list[int], max_new: int, K: int = 3,
                      eos: set[int] | None = None) -> list[int]:
        """Greedy speculative decode from a prompt."""

    @property
    def spec_stats(self) -> dict:
        """Acceptance report for the last :meth:`spec_generate` run."""

    def close(self) -> None:
        """Drop the graphs and the layer objects so a reload can rebuild."""


class PagedModel(Model, Protocol):
    """``Model`` with the extra capabilities the prefix cache drives.

    Hybrid (linear-attention) models do not hold their prefix as KV alone: 24
    layers carry it forward as a fixed-size recurrent state, so a cached block
    prefix is only reusable together with the state that produced it.  These
    three methods are what ``PrefixCacheEngine`` needs on top of ``Model``.
    """

    def enable_paging(self, num_blocks: int = 512, block_size: int = 64):
        """Attach a shared paged KV pool; must run before :meth:`capture`."""

    def snapshot_linear_states(self) -> dict[int, tuple["torch.Tensor", "torch.Tensor"]]:
        """Clone every linear-attention layer's (recurrent, conv) state."""

    def restore_linear_states(self, states) -> None:
        """Adopt a snapshot taken by :meth:`snapshot_linear_states`."""


@dataclass(frozen=True)
class _Entry:
    model: type
    load_weights: Callable[..., dict]


#: Checkpoint-config key -> implementation.  Keys are matched as lowercase
#: substrings of the joined ``model_type`` / ``architectures`` strings.
_REGISTRY: dict[str, _Entry] = {}


def register(keys: tuple[str, ...], model: type,
             load_weights: Callable[..., dict]) -> None:
    """Make ``model`` the implementation for checkpoints whose config says ``keys``."""
    for key in keys:
        _REGISTRY[key] = _Entry(model, load_weights)


def registered() -> list[str]:
    """The architecture keys this engine build can serve."""
    return sorted(_REGISTRY)


def _arch_strings(config: dict) -> list[str]:
    out: list[str] = []
    for section in (config, config.get("text_config") or {}):
        if not isinstance(section, dict):
            continue
        if section.get("model_type"):
            out.append(str(section["model_type"]))
        out.extend(str(a) for a in (section.get("architectures") or []))
    return [s.lower() for s in out]


def detect_architecture(config: dict) -> str:
    """Return the registry key for a checkpoint ``config.json``.

    Raises instead of guessing: serving a checkpoint through the wrong
    implementation would produce garbage rather than an error.
    """
    text = " ".join(_arch_strings(config))
    for key in _REGISTRY:
        if key in text:
            return key
    # A multimodal checkpoint can hide the text model behind a wrapper config;
    # the Qwen3.5 text geometry is distinctive enough to use as a last resort.
    text_config = config.get("text_config") or {}
    if "linear_num_value_heads" in text_config:
        return "qwen3_5"
    raise RuntimeError(
        "no engine implementation matches this checkpoint: "
        f"model_type/architectures={text!r}; registered={registered()}")


def _entry_for(model_dir: str | Path, architecture: str | None = None) -> _Entry:
    config = json.loads((Path(model_dir) / "config.json").read_text())
    key = architecture or detect_architecture(config)
    try:
        return _REGISTRY[key]
    except KeyError:
        raise RuntimeError(
            f"unknown architecture {key!r}; registered={registered()}") from None


def create_model(model_dir: str | Path, max_seq: int = 4096, bench: bool = True,
                 weights: dict | None = None, paged: bool = False,
                 architecture: str | None = None) -> Model:
    """Load the model implementation this checkpoint asks for.

    ``weights`` is the resident tensor dict from :func:`load_weights`, passed
    through on a reload so rebuilding skips the ~1 min disk read.
    """
    entry = _entry_for(model_dir, architecture)
    return entry.model(model_dir, max_seq=max_seq, bench=bench,
                       weights=weights, paged=paged)


def load_weights(model_dir: str | Path, quant: bool = False,
                 weight_dtype: "torch.dtype | None" = None,
                 architecture: str | None = None) -> dict:
    """Read this checkpoint's weights onto the GPU, for a resident server.

    Kept separate from :func:`create_model` because a reload rebuilds the model
    from the tensors already in VRAM instead of reading 9 GB from disk again.
    """
    entry = _entry_for(model_dir, architecture)
    kwargs: dict[str, Any] = {} if weight_dtype is None else {"weight_dtype": weight_dtype}
    return entry.load_weights(Path(model_dir), quant=quant, **kwargs)


def describe(model_dir: str | Path, architecture: str | None = None) -> dict:
    """What to log about a checkpoint: path, directory name and architecture.

    The server prints this at startup and returns it from ``ping``, and the
    OpenAI shell reads it back, so neither of them has to assume which model is
    loaded -- the checkpoint config is the only source of truth.
    """
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    return {"model_path": str(model_dir), "model": model_dir.name,
            "arch": architecture or detect_architecture(config)}


#: Dependency order: a module comes after everything it imports from, so that a
#: ``from ops.gemv import GEMV`` at the top of a model module picks up the class
#: object the just-reloaded ops module defines.
_RELOAD_ORDER = (
    "ops.dtypes", "ops.gemv", "ops.embed", "ops.norm", "ops.rope", "ops.delta",
    "ops",
    "models.qwen35.tuning", "models.qwen35.weights", "models.qwen35.batch",
    "models.qwen35.linear_attn",
    "models.qwen35.full_attn", "models.qwen35.blocks", "models.qwen35.mtp",
    "models.qwen35.model", "models.qwen35", "models",
    "config", "sequence", "paged_cache", "radix_cache", "block_manager",
    "prefix_engine", "model_runner", "scheduler", "llm_engine",
    "engine_api",
)


def _engine_module(name: str) -> bool:
    return name in ("ops", "models") or name.startswith(("ops.", "models.")) or name in _RELOAD_ORDER


def reload() -> None:
    """Re-import every engine module, dependencies first.

    ``serve.py`` used to call ``importlib.reload`` on the single engine module;
    now that the engine is a package, reloading only the top module would leave
    the implementation modules stale.  The caller's module object is updated in
    place, so ``engine_api.create_model`` is the new function afterwards.
    """
    import importlib
    import sys

    stale = {name for name in sys.modules if _engine_module(name)} - set(_RELOAD_ORDER)
    if stale:
        raise RuntimeError(
            f"engine_api.reload(): add {sorted(stale)} to _RELOAD_ORDER")
    for name in _RELOAD_ORDER:
        module = sys.modules.get(name)
        if module is not None:
            importlib.reload(module)


from models.qwen35 import Qwen35, load_weights as _qwen35_load_weights  # noqa: E402

register(("qwen3_5", "qwen35"), Qwen35, _qwen35_load_weights)
