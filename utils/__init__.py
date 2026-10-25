"""Shared request and multimodal helpers.

Module: ``coffin.utils``. Helpers here are leaf modules used by the engine and
serving layers; they do not import the HTTP server or a model implementation.

The package stays lazy because the HTTP adapter only needs ``tool_call`` and
must remain importable on a machine without torch or PIL.
"""

__all__ = [
    "MMInput",
    "ToolCall",
    "has_tool_call",
    "parse_tool_calls",
    "split_thinking",
    "strip_thinking",
    "tool_result_message",
]


def __getattr__(name: str):
    """Load a helper module only when its public symbol is requested."""
    if name == "MMInput":
        from .mm_input import MMInput

        return MMInput
    if name in {"ToolCall", "has_tool_call", "parse_tool_calls",
                "split_thinking", "strip_thinking", "tool_result_message"}:
        from . import tool_call

        return getattr(tool_call, name)
    raise AttributeError(name)
