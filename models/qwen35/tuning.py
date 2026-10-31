"""在这个模型和这颗芯片上实测出来的调优参数。

默认值就是每个常量旁边点名的探针找到的最优点；环境变量覆盖是为了让一次运行
能做 A/B 对比而不必改引擎。

Module: ``coffin/models/qwen35``。族里的叶子模块——只 import ``os`` 和
``backends``——被 ``blocks``、``full_attn``、``linear_attn`` 和 ``mtp`` 用来
确定内核和缓冲区的尺寸。

Constraints:
    * 这里不许出现*后端*的数字。``ATTN_SPLIT`` 原来就住在这个文件里，注释写着
      它必须等于注意力内核里的常量，那等于让后端 ABI 漏进模型族：在根本不切分的
      后端上，调用方仍按 8 去定分片缓冲区，内核就会写越界。现在它从
      ``caps.attn_split`` 读取、再以旧名字导出，缓冲区尺寸因此跟着当前后端实际
      做什么走。
    * ``CKPT`` 决定投机回滚状态的尺寸，并限定校验宽度，所以它必须和
      ``model.py`` 里捕获的校验图一致。
    * 这些是实测值，不是自由参数；改默认值之前先重跑它点名的那支探针。
"""

from __future__ import annotations

import os

from backends import caps as _caps

#: 每个投机步校验的 token 数；校验图就是按这个尺寸捕获的。
CKPT = 8

#: 预填充时因果注意力的查询切块宽度。每块只读到自己的末尾，score 块的上三角
#: 因此根本不会被构造：pos=0 时省掉一半 score/softmax/PV 计算，pos=2048 时省
#: 四分之一。更小的块能省更多计算，但会增加 launch（整套注意力是 8 层、每层 2
#: 次 einsum）；在这颗芯片上 512 实测最优（diag/probe_prefill_split.py）。
PREFILL_ATTN_TILE = int(os.environ.get("PREFILL_ATTN_TILE", "512"))

#: 一个查询块内部每次流式读取的 key 数。score 块原来是 ``[nkv, rep, nq, p1]``
#: ——每个块都要 O(上下文长度) 的内存，256k 时是 8.6 GB，和权重一起在 16 GB 卡上
#: 放不下。现在按运行最大值（online softmax）累积，块大小无论上下文多长都保持
#: ``[nkv, rep, nq, PREFILL_ATTN_KB]``，代价是每块对累加器做一次重缩放。
#:
#: 在 BI-V100 上实测后从 1024 提到 4096。块更大意味着 online softmax 重缩放更少，
#: 而那里的预填充是计算受限的，45-51 TFLOP/s，所以重缩放才是成本，多出来的内存
#: 负担得起：按实际使用的 512 token 查询块算，活跃 score 块多占 128 MB，对照的是
#: 32 GB 的卡和约 4.5 GB 的 int8 权重。n=8192 时每一步上调都值——512 -> 2048
#: 快 10.5%，2048 -> 4096 再快 4.9%——而且这个方向在每个块宽、两轮独立扫描里都
#: 成立（``diag/corex/probe_prefill_sweep.py``、../../../PERF_COREX.md §13.7）。
#: 2048 token 的 prompt 毫无收益（720.0 -> 720.2 ms），因为它只有一个查询块，块
#: 循环几乎不跑；收益在长 prompt 上。
PREFILL_ATTN_KB = int(os.environ.get("PREFILL_ATTN_KB", "4096"))

#: ``MTP.prefill`` 里每个分块的 token 数。它的注意力和主层一样是分块的那种，但
#: 喂进去的是整个 prompt（为了让草稿缓存不掉队），所以它自己把输入切块、分块，
#: 而不是依赖调用方的预填充分块——2048 让它的 score 块无论 prompt 多长都维持在
#: ~134 MB。
PREFILL_MTP_CHUNK = int(os.environ.get("PREFILL_MTP_CHUNK", "2048"))

#: 解码注意力内核里每个查询头的 key 区间切分数。这是内核 ABI，不是调优选择，
#: 所以它是*后端*的值：这一侧持有内核要写的 ``part_acc``/``part_ml`` 缓冲区，
#: 宽度必须和内核索引它们时用的一致。值从 ``caps.attn_split`` 读取；名字保留旧的，
#: 所有定缓冲区尺寸的调用点因此一行未改。
ATTN_SPLIT = _caps().attn_split
