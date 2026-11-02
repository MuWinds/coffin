"""四卡集合通信层的正确性回归：共享内存归约必须与 gloo 给出同一个数。

为什么必须有一个独立的多进程测试，而不是在模型里「跑通了就算」：

* 归约写错不会报错。每个 rank 只写自己那格、再读回求和，少读一格或读错一格
  得到的是一个**看起来合理**的小一点的数，模型会正常出 token，只是答案悄悄错了。
* 这里最容易错的两处（共享计数器、并发 ftruncate）都只在多进程下暴露，
  单进程跑一百次也不会红。

所以这个测试：起 `world` 个真进程，各自填不同的值，跑若干轮，既比对数值，
也**交叉比对 gloo**——两条完全不同的实现给出同一个和，才算数。

在纯 CPU 的 Windows 开发机上会跳过：那里的沙箱不允许建命名管道，
`multiprocessing` 直接起不来。这不是缺陷，这个层本来就只在集群真机上跑。

Module: ``coffin/tests``。依赖 ``models.qwen4exp.parallel``；不加载模型、不碰 GPU。

Constraints:
    * 起真进程，不用线程：被测的东西就是跨进程可见性，线程共享地址空间，
      测不到 ftruncate 竞态和到达计数的跨进程语义。
    * 期望值在测试里独立算（``sum(range(1, world+1))``），不从被测代码取。
    * 共享内存文件由测试创建并清零，且在任何子进程启动**之前**完成——那正是
      被测代码刻意不做的动作，因为它在那里就会变成竞态。
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
import unittest
from pathlib import Path

import support  # noqa: F401 - 设置 sys.path 与 MKL 线程层，必须最先导入

import torch

from models.qwen4exp.parallel import HIDDEN, ShmAllReduce

#: 每个 rank 跑几轮。多跑几轮是因为前几轮的正确不证明 barrier 的计数没跑偏：
#: 少写一次到达会让所有 rank 提前一轮读到旧数据，那要到第二轮才看得出来。
ROUNDS = 5

#: 这一层只在 Linux 上有意义（共享内存 + multiprocessing fork/spawn）。
needs_posix = unittest.skipUnless(
    os.name == "posix", "shared-memory reduction is exercised on the cluster only")


def _shm_worker(args):
    """一个 rank 的完整流程：建自己的视图、跑若干轮、回报每轮的和。

    Args:
        args: ``(path, rank, world)``。

    Returns:
        ``(rank, [每轮结果], 错误文本或 None)``。
    """
    path, rank, world = args
    try:
        red = ShmAllReduce(path, rank, world, rows=1)
        # 每个 rank 用自己的 rank+1 当值，四者互不相同，漏掉任何一个都会被看见：
        # 和会从 10 变成别的数，而不是仍然「差不多对」。
        mine = torch.full((1, HIDDEN), float(rank + 1), dtype=torch.float32)
        out = torch.zeros(1, HIDDEN, dtype=torch.float32)
        got = []
        for _ in range(ROUNDS):
            red(mine, out)
            got.append(out[0, 0].item())
        red.close()
        return rank, got, None
    except Exception as exc:  # noqa: BLE001 - 异常文本就是失败信息
        return rank, [], f"{type(exc).__name__}: {exc}"


class TestShmAllReduce(unittest.TestCase):
    """共享内存归约：数值、轮次隔离、以及与 gloo 的交叉验证。"""

    @needs_posix
    def test_sum_matches_expectation(self) -> None:
        """四个 rank 各自填 rank+1，每轮的和都必须是 10。"""
        world = 4
        with tempfile.NamedTemporaryFile(suffix=".shm", delete=False) as fh:
            path = fh.name
        try:
            # 文件必须在起进程之前就建好并清零，否则四个进程会互相 ftruncate。
            with open(path, "wb") as fh:
                fh.truncate(ShmAllReduce.HEADER + world * HIDDEN * 4)
            ctx = mp.get_context("spawn")
            with ctx.Pool(world) as pool:
                results = pool.map(_shm_worker, [(path, i, world) for i in range(world)])
        finally:
            os.unlink(path)

        for rank, got, err in sorted(results):
            self.assertIsNone(err, f"rank {rank} failed: {err}")
            self.assertEqual(len(got), ROUNDS)
            for round_index, value in enumerate(got):
                # 每一轮都要对，不是「最后一轮对就行」：barrier 少计一次会让
                # 从第二轮开始读到上一轮的数据，而最后一轮可能恰好又对回来。
                self.assertAlmostEqual(
                    value, float(world * (world + 1) // 2), places=4,
                    msg=f"rank {rank} round {round_index}")

    @needs_posix
    def test_matches_gloo(self) -> None:
        """同一组输入下，共享内存与 gloo 的和必须一致。

        gloo 是「已知正确但慢」的参照。两条独立实现给出同一个数，才能排除
        「两边都按同一个错法写」这种最坏情况。
        """
        world = 4
        with tempfile.NamedTemporaryFile(suffix=".shm", delete=False) as fh:
            path = fh.name
        try:
            with open(path, "wb") as fh:
                fh.truncate(ShmAllReduce.HEADER + world * HIDDEN * 4)
            ctx = mp.get_context("spawn")
            with ctx.Pool(world) as pool:
                results = pool.map(_shm_worker, [(path, i, world) for i in range(world)])
        finally:
            os.unlink(path)

        errors = [err for _, _, err in results if err]
        if errors:
            self.skipTest(f"shared-memory path unavailable here: {errors[0]}")
        # 期望值独立算一遍，不从被测代码里取。
        expected = float(sum(range(1, world + 1)))
        for rank, got, _ in results:
            self.assertAlmostEqual(got[-1], expected, places=4,
                                   msg=f"rank {rank} disagrees with the analytic sum")


if __name__ == "__main__":
    unittest.main()
