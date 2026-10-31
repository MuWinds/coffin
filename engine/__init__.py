"""推理引擎的运行时编排包。

Module: ``coffin.engine``。请求生命周期、调度、执行和块归属都归本包管；
它既不知道 HTTP，也不知道具体模型族。

Constraints:
    * 子模块一律显式导入，避免兼容垫片在包初始化阶段形成循环导入。
    * 访问模型族代码只走 ``engine_api`` 的契约。
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
    """按需解析公开的运行时类，不在导包时把整棵依赖树拉起来。

    延迟导入是为了让 ``import coffin.engine`` 只付出包本身的代价：上层做
    后端探测或跑纯 CPU 测试时，不碰 torch 也能拿到符号。

    Args:
        name: 属性名，必须是 ``__all__`` 之外映射表里登记的类名。

    Returns:
        对应模块里的类对象。

    Raises:
        AttributeError: ``name`` 不在映射表里。包装成 AttributeError 才能让
            ``hasattr`` 和 ``from ... import`` 的探测语义保持正常。
    """
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
