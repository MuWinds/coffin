"""Qwen4-Exp（Qwen3.8-Flash-Next）的纯 PyTorch 参考实现与它的引擎包装。

这一族是"正确性地基"：所有算子都是可直接在 CPU 上跑、可对着
``docs/qwen4/modeling_qwen4_exp.py`` 逐行核对的朴素实现，没有任何融合内核、
没有量化、没有并行。W4A8 与多卡版本以后在此基础上替换算子，数值基准就是这里。

与 Qwen3.5 族的结构差异是根本性的，不要照搬那个族的 ``Layer``：

* 残差流宽度是 ``hc_count * hidden_size``，不是 ``hidden_size``；
* 层内没有 pre-norm，归一化全在 ``GatedResidual`` 里；
* mixer 的输入是读门塌缩后的一路激活，输出按逐流标量门写回 4 路。

Module: ``coffin/models/qwen4exp``。族内自包含：只依赖 ``torch`` 与标准库，
不 import ``transformers``。``engine.py`` 是唯一往上走的那一层——它把
``CausalLM`` 接到 ``engine_api.Model`` 契约上，并且（经 ``proj`` / ``qsa`` /
``gdn``）间接接触到 ``backends``：布局与派发是平台事实，这条依赖早就存在，只是
以前藏在族的内部。

Constraints:
    * 不许 import ``transformers``。参考实现只是 oracle，测试通过
      ``docs/qwen4/oracle/*.pt`` 里的 golden fixture 对齐。
    * 所有形状从 ``config`` 推导，不许写死真实 checkpoint 的常数：小配置
      （hidden 64、4 头）和真实配置（hidden 2560、48 层）跑同一份代码。
    * 每个模块都必须能单独构造、单独前向，因为 fixture 是按模块 dump 的。
    * ``models.qwen4exp.engine`` 必须排在 ``engine_api._RELOAD_ORDER`` 里去 reload
      的每一个同族模块之后：它的类持有各层的类对象，顺序反了会绑到旧类上。
"""

from models.qwen4exp.engine import Qwen4Engine, load_weights

__all__ = ["Qwen4Engine", "load_weights"]
