"""Runtime orchestration package for the inference engine.

Module: ``coffin.engine``. This package owns request lifecycle, scheduling,
execution and block ownership; it does not know about HTTP or a model family.

Constraints:
    * Submodules are imported explicitly to keep compatibility shims free of
      package-initialisation cycles.
    * Model-family code is reached through ``engine_api`` contracts.
"""

__all__ = [
    "block_manager",
    "config",
    "llm_engine",
    "model_runner",
    "prefix_engine",
    "scheduler",
    "sequence",
]


def __getattr__(name: str):
    """Resolve the public runtime class without eager package imports."""
    modules = {
        "BlockManager": "block_manager",
        "EngineConfig": "config",
        "LLMEngine": "llm_engine",
        "ModelRunner": "model_runner",
        "PrefixCacheEngine": "prefix_engine",
        "Scheduler": "scheduler",
        "SamplingParams": "sequence",
        "Sequence": "sequence",
        "SequenceStatus": "sequence",
    }
    module_name = modules.get(name)
    if module_name is None:
        raise AttributeError(name)
    from importlib import import_module

    return getattr(import_module(f"{__name__}.{module_name}"), name)
