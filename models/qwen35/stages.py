"""预填充的分阶段计时，只在 ``COFFIN_STAGE_TIMING=1`` 时启用。

存在的理由：Occamy 在 4 卡 TP 上预填充 321 token 要 6.2 s，也就是**每层 156 ms**；
而同一台机器上 dp4a 分组核按 ``moe_grouped_i8.cu`` 里记的 0.96 TMAC/s 算，每卡每层
的专家投影只要 2 ms 量级、40 层合计约 80 ms。张量核与 dp4a 两档实测毫秒级一样
（6235 vs 6237 ms），说明这 6 s 既不在专家投影里、也不在两条路的差别里。猜到这里
就该上计时器：每层拆成「输入归一化 + 投影 / 注意力主体 / 输出投影 / MLP」，MLP 再拆
「路由 / 专家 / 共享专家」。

刻意做成叶子模块：只 import ``os`` / ``json`` / ``time`` / ``torch``，被 ``blocks``、
``moe``、``model`` 引用，因此不能反向依赖它们（否则预填充路径会出现导入环）。

代价必须写在明面上：每个阶段边界一次 ``torch.cuda.synchronize()``，会把原本能重叠的
阶段压平。所以这里的数字是「各阶段独占」的墙钟时间，加起来会**略大于**端到端的
``prefill_ms``，只能用来判断谁占大头，不能当作优化后的预期值。
"""

from __future__ import annotations

import json
import os
import time

import torch

#: 是否启用。关掉时 :class:`span` 的进出只剩一次属性读取，热路径几乎零成本。
ON = os.environ.get("COFFIN_STAGE_TIMING") == "1"

#: 每这么多次「一层结束」打一份报告并清零；``0`` 表示不在层边界打（只有显式调用
#: :func:`report` 的地方才有输出，比如 ``model.prefill`` 的分块循环）。
#:
#: 用它的理由是真实请求走的是调度器的预填充路径，而不是 ``model.prefill`` 的分块
#: 循环——第一次只加 :func:`report` 时整份日志里一条 ``[stages]`` 都没有，所以
#: 计数必须挂在**层**上，不能挂在某一个上层调用点上。
EVERY = int(os.environ.get("COFFIN_STAGE_EVERY", "0"))

_acc: dict[str, float] = {}
_cnt: dict[str, int] = {}
_layers = 0

if ON:
    print(f"[stages] on every={EVERY}", flush=True)


def reset() -> None:
    """清空累加器；一次预填充之前调一次。"""
    _acc.clear()
    _cnt.clear()


def done_layer() -> None:
    """一层跑完时调一次；到 ``EVERY`` 就打一份报告并清零。"""
    global _layers
    if not ON:
        return
    _layers += 1
    if EVERY and _layers % EVERY == 0:
        report(f"layers={_layers}")
        reset()


class span:
    """累计某个阶段的墙钟时间（毫秒）。

    Args:
        name: 阶段名，直接进报告。
    """

    __slots__ = ("name", "t0")

    def __init__(self, name: str):
        self.name = name
        self.t0 = 0.0

    def __enter__(self) -> "span":
        if ON:
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        if ON:
            torch.cuda.synchronize()
            _acc[self.name] = _acc.get(self.name, 0.0) + (time.perf_counter() - self.t0) * 1e3
            _cnt[self.name] = _cnt.get(self.name, 0) + 1
        return False


def report(tag: str = "") -> None:
    """把累计结果打一行 JSON；未启用时什么都不做。

    Args:
        tag: 打在前面的标签（比如这次分块的行区间）。
    """
    if not ON or not _acc:
        return
    total = sum(_acc.values())
    rows = {k: {"ms": round(v, 2), "n": _cnt.get(k, 0)}
            for k, v in sorted(_acc.items(), key=lambda kv: -kv[1])}
    print(f"[stages] tag={tag} sum_ms={total:.1f} " + json.dumps(rows, ensure_ascii=False),
          flush=True)
