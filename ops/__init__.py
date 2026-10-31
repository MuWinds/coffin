"""设备侧原语，刻意不绑定任何一套 checkpoint 族。

模型族决定形状和接线；这些模块决定投影、归一化、RoPE 或 delta 递推在这颗
芯片上*怎么执行*。每一个都是被基准测到当前形态的，所以实测记录跟着它所辩护
的代码一起走。

Module: ``coffin/ops``。依赖图上位于 ``backends`` 之上的最底层：
``models/*`` 和引擎从这里导入，反方向不允许。下面每个模块都再导出一次，
调用方写 ``from ops import GEMV`` 即可，不必伸手进子模块。

Constraints:
    * 不许 import ``models/`` 或图里更上层的东西；这一层必须能在没有
      checkpoint 的情况下加载、单测。
    * 平台细节存放在 ``backends``：这里任何模块都不许出现设备 API、内核源码
      路径、arch 字符串或设备字面量。DCU 和 CoreX 之间不同的东西一律通过
      ``backends.ops``、``backends.caps()`` 或 ``ops.runtime`` 取得。
    * 用绝对导入（``from ops.gemv import GEMV``），因为入口脚本把 ``coffin/``
      本身放进 ``sys.path``，``ops`` 于是成了和 ``models``、``cache`` 平级的
      顶层包——见 ``entrypoints/cli/runtime.py`` 和 ``run_chat.py``。
    * 模块 docstring 里测得的 TFLOP/s 和 GB/s 数字要保留：它们是当前内核形态
      为什么胜出的唯一记录，针对的是本引擎最早调优的那颗芯片；新平台的数字
      要拿来和它们对比。
"""

from ops.delta import chunked_delta
from ops.dtypes import D, NEG_INF, WF
from ops.embed import TiedEmbedding
from ops.gemv import GEMV, GEMVGroup, _proj
from ops.norm import RMSNorm, RMSNormGated
from ops.rope import apply_rope, make_rope_tables, rotate_half

__all__ = [
    "D", "NEG_INF", "WF",
    "GEMV", "GEMVGroup", "_proj",
    "TiedEmbedding",
    "RMSNorm", "RMSNormGated",
    "make_rope_tables", "rotate_half", "apply_rope",
    "chunked_delta",
]
