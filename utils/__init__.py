"""请求与多模态的公共辅助函数。

Module: ``coffin.utils``。这里是叶子模块，供引擎和服务层使用；不许 import
HTTP 服务器或具体模型实现。

Constraints:
    * 包保持惰性导入：HTTP 适配器只需要 ``tool_call``，而它必须在没有
      torch 或 PIL 的机器上也能导入。
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
    """只在真正取用某个公开符号时才加载对应的辅助模块。

    惰性导入让没有 torch 或 PIL 的机器也能 import 本包：``tool_call`` 是纯
    字符串处理，``mm_input`` 才依赖图像库。

    Args:
        name: 属性名。

    Returns:
        对应的类或函数。

    Raises:
        AttributeError: 没有这个符号。必须抛 AttributeError 而不是
            ImportError，否则 ``hasattr`` 的探测会把依赖缺失误报成符号缺失。
    """
    if name == "MMInput":
        from .mm_input import MMInput

        return MMInput
    if name in {"ToolCall", "has_tool_call", "parse_tool_calls",
                "split_thinking", "strip_thinking", "tool_result_message"}:
        from . import tool_call

        return getattr(tool_call, name)
    raise AttributeError(name)
