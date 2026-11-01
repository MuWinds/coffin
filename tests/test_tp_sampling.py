"""张量并行的随机采样结果必须由 rank 0 统一。

Module: coffin/tests。用双进程 CPU Gloo 验证采样后的 token 广播，不需要加速卡。

Constraints:
    * 两个 rank 故意使用不同 logits；若未广播，测试必须能观察到分歧。
"""

from __future__ import annotations

import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from engine.model_runner import ModelRunner


def _sample_rank(rank: int, init_method: str, results) -> None:
    """建 Gloo 组并记录该 rank 最终采用的采样 token。"""
    dist.init_process_group("gloo", init_method=init_method, rank=rank,
                            world_size=2)
    torch.manual_seed(100 + rank)
    logits = torch.tensor([8.0, -8.0]) if rank == 0 else torch.tensor([-8.0, 8.0])
    results.put(ModelRunner.sample(logits, 0.01))
    dist.destroy_process_group()


class TensorParallelSamplingTest(unittest.TestCase):
    """采样后同步，避免各 rank 的缓存从不同 token 开始推进。"""

    def test_non_greedy_choice_matches_rank_zero(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        ctx = mp.get_context("spawn")
        results = ctx.SimpleQueue()
        mp.spawn(_sample_rank, args=(f"tcp://127.0.0.1:{port}", results),
                 nprocs=2, join=True)
        self.assertEqual([results.get(), results.get()], [0, 0])


if __name__ == "__main__":
    unittest.main()
