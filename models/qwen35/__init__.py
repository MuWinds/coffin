"""Qwen3.5-4B 族入口：模型类、权重加载器、调优开关。

checkpoint 是混合注意力模型——24 层 Gated-DeltaNet 加 8 层 GQA，另有一个
MTP 层当投机草稿模型——所以这里的部件都是这个形状专属的。数学和它们跑的
内核放在 ``ops/``。

Module: ``coffin/models/qwen35``。族的顶层：它可以 import ``ops`` 和自己
的同级模块（``blocks``、``batch``、``linear_attn``、``full_attn``、``mtp``、
``vision``、``weights``、``tuning``），而它自己只能经 ``engine_api`` 到达，
后者按 checkpoint 的 ``config.json`` 分派。

Constraints:
    * 只用绝对导入（``from models.qwen35.model import Qwen35``）：入口脚本把
      ``coffin/`` 放进 ``sys.path``，所以 ``models`` 是和 ``ops``、``backends``
      平级的顶层包。
    * 族里任何东西都不许 import ``engine_api`` 或更上层——那个模块是唯一
      允许知道存在哪些族的地方。
    * 不写硬件细节：设备、dtype 和 split-K 宽度都来自 ``backends``，换一颗
      加速器不需要改这个包里的任何东西。
"""

from models.qwen35.model import Qwen35
from models.qwen35.weights import load_weights

__all__ = ["Qwen35", "load_weights"]
