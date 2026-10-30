"""The serving-facing surface of the engine: one checkpoint in, one session out.

Module: ``coffin/`` (top of the dependency graph, model-agnostic). This is the
only module that knows which model families exist; every other engine module
talks to a family through the ``Model`` / ``PagedModel`` protocols declared
here. Runtime and diagnostic code therefore never imports a model family through
the dispatch layer, and a second family is added by one
``register()`` call at the bottom of this file rather than by editing callers.

The engine is split by responsibility:

    ops/                 device primitives shared by any model (GEMV, norms,
                         RoPE, the chunked delta rule, the tied embedding)
    models/<family>/     one architecture family: blocks, MTP head, weight
                         loader, and the model class that owns the graphs
    engine_api.py        this file: the ``Model`` contract, checkpoint -> model
                         dispatch, and the reload used by the runtime service

Dispatch is by checkpoint config, not by a flag: :func:`create_model` reads
``config.json`` and matches a registered key against ``model_type`` /
``architectures``. The key ``"qwen3_5"`` therefore matches both
``Qwen3_5ForConditionalGeneration`` and a bare ``qwen3_5``. ``architecture=``
overrides the detection for a checkpoint whose metadata lies.

Constraints:
    * Imports are absolute (``from models.qwen35 import ...``) because the entry
      script puts ``coffin/`` itself on ``sys.path``; ``ops`` and ``models`` are
      top-level packages beside ``hip_ops``, not subpackages of this module.
    * Every engine module must be listed in ``_RELOAD_ORDER``. :func:`reload`
      refuses to run otherwise, because a module missing from the list would
      keep serving the class objects from before the reload while its importers
      pick up new ones -- a type mismatch that only shows up at run time.
    * The model class is imported at the *bottom* of the file (hence the
      ``# noqa: E402``): the ``register()`` call that follows it needs
      ``_Entry`` and ``_REGISTRY`` to exist, so this is the one module allowed
      to import a family, and only after its own dispatch surface is defined.
    * Nothing here allocates model memory. Weights arrive from
      :func:`load_weights` and are passed back on a reload, so a rebuild is
      seconds instead of a 9 GB disk read.
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
    below the token ids is the model's business. Attributes that the serving
    layer reads directly:

    ``max_seq``
        Context window the KV cache and rope tables were built for.
    ``pos1``
        Length-1 device tensor holding the current position. The server and
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
        """Process a prompt; returns the logits for the next token.

        Args:
            tokens: Prompt tokens to run, starting at the current ``pos1``.
            chunk: GEMM M to slice the prefill into; rocBLAS on this card tops
                out around 2048 tokens per call.

        Returns:
            Logits of the last position, the distribution the next token is
            drawn from.

        Raises:
            RuntimeError: the model has no paged pool, so it has nowhere to put
                the KV this prefill produces. Call ``enable_paging`` first.
            ValueError: the prompt is empty, longer than ``max_seq``, or would
                start past a reserved row.
        """

    def decode_step(self, token: int) -> "torch.Tensor":
        """Advance one token through the captured decode graph.

        Args:
            token: The token to append; the graph writes its KV at ``pos1`` and
                advances the counter.

        Returns:
            Logits for the position after ``token``.

        Raises:
            RuntimeError: the graphs were never captured, so there is no replay
                to run; call ``capture`` before serving.
        """

    def logits(self, token: int) -> "torch.Tensor":
        """Un-captured single step (prefill fallback / debugging).

        Args:
            token: The token to append, exactly as ``decode_step`` would.

        Returns:
            The same logits ``decode_step`` would produce, without a graph. Used
            where capture cannot apply: a shape the graphs do not cover, or a
            numerical comparison against the captured path.

        Raises:
            RuntimeError: the paged pool is missing, so the step has no slots to
                write.
        """

    def capture(self, warmup: int = 3) -> None:
        """Capture the decode step (and, when present, the drafter) once.

        Args:
            warmup: Untimed replays before capture, to settle allocator and
                kernel state; 3 is enough to stop the first real step from
                allocating inside an already-captured region.

        Raises:
            RuntimeError: paging was never enabled. The captured kernel bakes in
                the pool tensors and the slot map, so capture without a pool
                would produce a graph writing nowhere.
        """

    @property
    def graph_ready(self) -> bool:
        """True once :meth:`capture` has produced a replayable decode graph."""

    def capture_verify(self, c: int = 4, warmup: int = 3) -> None:
        """Capture the ``c``-token verification forward.

        Args:
            c: Verify width, i.e. the drafts plus the anchor token. Bounds the
                ``K`` that :meth:`spec_generate` accepts.
            warmup: Untimed replays before capture for the same reason as
                :meth:`capture`.
        """

    def spec_generate(self, prompt: list[int], max_new: int, K: int = 3,
                      eos: set[int] | None = None) -> list[int]:
        """Greedy speculative decode from a prompt.

        Args:
            prompt: Prompt tokens, processed from the empty context.
            max_new: Upper bound on generated tokens.
            K: Drafts per step, at most ``verify_c - 1``. Higher is only worth
                it when the drafter's acceptance rate stays high.
            eos: Stop token ids; an EOS ends generation without being returned.

        Returns:
            The generated token ids. Identical to non-speculative greedy output,
            because every draft is verified by the main model.
        """

    @property
    def spec_stats(self) -> dict:
        """Acceptance report for the last :meth:`spec_generate` run."""

    def close(self) -> None:
        """Drop the graphs and the layer objects so a reload can rebuild."""


class PagedModel(Model, Protocol):
    """``Model`` with the extra capabilities the prefix cache drives.

    Hybrid (linear-attention) models do not hold their prefix as KV alone: 24
    layers carry it forward as a fixed-size recurrent state, so a cached block
    prefix is only reusable together with the state that produced it. These
    three methods are what ``PrefixCacheEngine`` needs on top of ``Model``.
    """

    def enable_paging(self, num_blocks: int = 512, block_size: int = 64):
        """Attach a shared paged KV pool; must run before :meth:`capture`.

        Args:
            num_blocks: Blocks in the pool, the KV budget of every live request
                combined.
            block_size: Tokens per block; also the granularity at which the
                radix tree can share a prefix.

        Returns:
            The pool, which the caller drives through a
            :class:`~radix_cache.RadixCache`.

        Raises:
            RuntimeError: paging was already enabled; a second pool would leave
                the first one's blocks unreachable.
        """

    def snapshot_linear_states(self) -> dict[int, tuple["torch.Tensor", "torch.Tensor"]]:
        """Clone every linear-attention layer's (recurrent, conv) state.

        Returns:
            Layer index -> (recurrent, conv) tensors, detached clones, so the
            live state may keep advancing while the snapshot stays a record of
            the boundary it was taken at.
        """

    def restore_linear_states(self, states) -> None:
        """Adopt a snapshot taken by :meth:`snapshot_linear_states`.

        Args:
            states: The snapshot to install; it must describe this model's own
                linear-attention layers, since the indices are used as keys.

        Raises:
            KeyError: ``states`` names a layer index this model does not have,
                i.e. a snapshot from another architecture. Reject it rather than
                restoring a partial state that silently pairs shared KV with the
                wrong recurrent state.
        """


@dataclass(frozen=True)
class _Entry:
    model: type
    load_weights: Callable[..., dict]


#: Checkpoint-config key -> implementation. Keys are matched as lowercase
#: substrings of the joined ``model_type`` / ``architectures`` strings.
_REGISTRY: dict[str, _Entry] = {}


def register(keys: tuple[str, ...], model: type,
             load_weights: Callable[..., dict]) -> None:
    """Make ``model`` the implementation for checkpoints whose config says ``keys``.

    Called once per family at the bottom of this file. A later registration for
    the same key wins, which is what makes a reload of this module take effect.

    Args:
        keys: Lowercase substrings of ``model_type`` / ``architectures``. Keep
            them short but distinctive: matching is a plain substring test, so
            an over-general key would claim other families' checkpoints.
        model: The family's model class; its constructor signature is the one
            :func:`create_model` calls.
        load_weights: The family's weight loader, called by
            :func:`load_weights` with the checkpoint path.
    """
    for key in keys:
        _REGISTRY[key] = _Entry(model, load_weights)


def registered() -> list[str]:
    """The architecture keys this engine build can serve.

    Returns:
        Sorted keys, i.e. what ``architecture=`` accepts and what goes into an
        error message when a checkpoint matches nothing.
    """
    return sorted(_REGISTRY)


def _arch_strings(config: dict) -> list[str]:
    """Lowercased architecture hints from a checkpoint config.

    Both the top level and ``text_config`` are read, because a multimodal
    checkpoint wraps the text model and names it only inside the nested section.

    Args:
        config: Parsed ``config.json``.

    Returns:
        ``model_type`` and every ``architectures`` entry from both sections, in
        that order, lowercased for substring matching.
    """
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

    Args:
        config: Parsed ``config.json``.

    Returns:
        A key present in the registry, suitable for ``architecture=`` and for
        ``create_model``.

    Raises:
        RuntimeError: nothing registered matches the config and the geometry
            fallback does not apply either. The caller must serve a different
            checkpoint or register the family; there is no safe default.
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
    """Resolve a checkpoint directory to its registry entry.

    Args:
        model_dir: Directory containing ``config.json``.
        architecture: Explicit key overriding detection, for a checkpoint whose
            metadata lies.

    Returns:
        The entry holding the model class and weight loader for that key.

    Raises:
        RuntimeError: the resolved key is not registered. The message lists what
            is, since a bare "unknown architecture" leaves the caller with
            nothing to correct.
    """
    config = json.loads((Path(model_dir) / "config.json").read_text())
    key = architecture or detect_architecture(config)
    try:
        return _REGISTRY[key]
    except KeyError:
        raise RuntimeError(
            f"unknown architecture {key!r}; registered={registered()}") from None


def create_model(model_dir: str | Path, max_seq: int = 4096, bench: bool = True,
                 weights: dict | None = None, paged: bool = False,
                 architecture: str | None = None, quant: bool = False) -> Model:
    """Load the model implementation this checkpoint asks for.

    ``weights`` is the resident tensor dict from :func:`load_weights`, passed
    through on a reload so rebuilding skips the ~1 min disk read.

    ``quant`` applies only when ``weights`` is ``None``, i.e. when this call loads the
    checkpoint itself. It exists because a caller that owns the dict has already chosen
    the storage form, while a caller that does not would otherwise be stuck with the
    loader's default of bf16 -- twice the resident size, which is the difference between
    a 27B fitting a card and not fitting it. Passing ``quant`` together with ``weights``
    is not an error and not a conversion: the dict is used as it is.

    Args:
        model_dir: Checkpoint directory.
        max_seq: Context window the KV pool and rope tables are sized for. A
            VRAM commitment, not a runtime check.
        bench: Run the family's kernel-variant tuner at load time. Costs a few
            seconds of start-up; tests turn it off.
        weights: Already-loaded tensors to build from. ``None`` makes the family
            read them from disk.
        paged: Build the paged-KV variant. The prefix-cache and batched paths
            need it; ``capture`` without paging bakes in the dense layout.
        architecture: Registry key overriding config detection.
        quant: Store the text projections as int8 when this call loads the checkpoint
            itself. Ignored when ``weights`` is supplied.

    Returns:
        A ``Model``; cast it to ``PagedModel`` only when ``paged`` was set.

    Raises:
        RuntimeError: the checkpoint matches no registered family, or names one
            that is not registered. Serve a supported checkpoint rather than
            guessing an implementation.
    """
    entry = _entry_for(model_dir, architecture)
    return entry.model(model_dir, max_seq=max_seq, bench=bench,
                       weights=weights, paged=paged, quant=quant)


def load_weights(model_dir: str | Path, quant: bool = False,
                 weight_dtype: "torch.dtype | None" = None,
                 architecture: str | None = None, vision: bool | None = None,
                 vision_quant: bool = False) -> dict:
    """Read this checkpoint's weights onto the GPU, for a resident server.

    Kept separate from :func:`create_model` because a reload rebuilds the model
    from the tensors already in VRAM instead of reading 9 GB from disk again.

    ``vision``/``vision_quant`` control the tower's tensors only: ``None`` means
    "load them iff the checkpoint declares a ``vision_config``", and
    ``vision_quant`` stores the tower's GEMM weights as int8 (a separate memory
    policy from the text model's, which ``quant`` governs).

    Args:
        model_dir: Checkpoint directory.
        quant: Store the text model's 2-D projection weights as int8. On a 16 GB
            card this is the difference between fitting the KV pool or not.
        weight_dtype: Compute dtype for the tensors that stay dense. ``None``
            uses the family's own default.
        architecture: Registry key overriding config detection.
        vision: Load the vision tower at all. ``False`` keeps it unloaded, so an
            image request has to be rejected instead of silently mis-prefilled.
        vision_quant: Store the tower's GEMM weights as int8 (~0.31 GiB instead
            of ~0.62 GiB), at a small per-request cost.

    Returns:
        The resident tensor dict, keyed by checkpoint parameter name, to hand to
        :func:`create_model` now and on every later reload.

    Raises:
        RuntimeError: the checkpoint matches no registered family, so there is
            no loader that knows its layout.
    """
    entry = _entry_for(model_dir, architecture)
    kwargs: dict[str, Any] = {} if weight_dtype is None else {"weight_dtype": weight_dtype}
    return entry.load_weights(Path(model_dir), quant=quant, vision=vision,
                              vision_quant=vision_quant, **kwargs)


def describe(model_dir: str | Path, architecture: str | None = None) -> dict:
    """What to log about a checkpoint: path, directory name and architecture.

    The server prints this at startup and returns it from ``ping``, and the
    OpenAI shell reads it back, so neither of them has to assume which model is
    loaded -- the checkpoint config is the only source of truth.

    Args:
        model_dir: Checkpoint directory.
        architecture: Registry key overriding detection.

    Returns:
        ``model_path`` (as given), ``model`` (the directory's basename, which is
        what ``/v1/models`` reports by default) and ``arch`` (the registry key
        actually in use).

    Raises:
        RuntimeError: the checkpoint matches no registered family, so there is
            no ``arch`` to report.
    """
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    return {"model_path": str(model_dir), "model": model_dir.name,
            "arch": architecture or detect_architecture(config)}


#: Dependency order: a module comes after everything it imports from, so that a
#: ``from ops.gemv import GEMV`` at the top of a model module picks up the class
#: object the just-reloaded ops module defines.
_RELOAD_ORDER = (
    "backends.protocol", "backends.reference", "backends.vendor",
    "backends.discovery", "backends.hip", "backends.corex",
    "backends.vendors", "backends.registry", "backends",
    "ops.dtypes", "ops.runtime", "ops.gemv", "ops.embed", "ops.norm", "ops.rope",
    "ops.mrope", "ops.delta", "ops.suffix",
    "ops",
    "models.qwen35.tuning", "models.qwen35.weights", "models.qwen35.batch",
    "models.qwen35.vision",
    "models.qwen35.linear_attn",
    "models.qwen35.full_attn", "models.qwen35.blocks", "models.qwen35.mtp",
    "models.qwen35.model", "models.qwen35", "models",
    "utils.mm_input", "utils.tool_call", "utils",
    "cache.paged_cache", "cache.radix_cache", "cache",
    "engine.config", "engine.sequence", "engine.block_manager",
    "engine.prefix_engine", "engine.model_runner", "engine.scheduler",
    "engine.llm_engine", "engine",
    "engine_api",
)


def _engine_module(name: str) -> bool:
    """Whether a ``sys.modules`` entry belongs to this engine.

    Args:
        name: Fully qualified module name.

    Returns:
        ``True`` for the engine's own packages, so :func:`reload` can detect a
        module that was added without being added to ``_RELOAD_ORDER``.
    """
    return (
        name in ("ops", "models", "engine", "cache", "utils")
        or name.startswith(("ops.", "models.", "engine.", "cache.", "utils."))
        or name in _RELOAD_ORDER
    )


def reload() -> None:
    """Re-import every engine module, dependencies first.

    The runtime calls this after unloading a model. Reloading only the top module
    would leave
    the implementation modules stale. The caller's module object is updated in
    place, so ``engine_api.create_model`` is the new function afterwards.

    Raises:
        RuntimeError: a loaded engine module is missing from ``_RELOAD_ORDER``,
            which would leave it serving pre-reload class objects while its
            importers use the new ones. Add it to the list; the message names
            the module.
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


# The one permitted upward edge: engine_api reaches into a family, never the
# reverse. Kept below the dispatch code so the module's own surface is defined
# before a family is pulled in, and because the registration right after it needs
# ``register`` and ``_Entry`` to exist. Files below it trigger ``# noqa: E402``.
from models.qwen35 import Qwen35, load_weights as _qwen35_load_weights  # noqa: E402

register(("qwen3_5", "qwen35"), Qwen35, _qwen35_load_weights)
