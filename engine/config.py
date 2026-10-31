"""引擎配置：服务器、调度器和 runner 共用的全部开关。

Module: ``coffin/``（顶层，不 import 引擎里的任何其他模块）。一个
``EngineConfig`` 从命令行或测试里构造出来，再往下传给 ``llm_engine`` /
``scheduler`` / ``model_runner``，它们因此都不需要知道服务是怎么被拉起来的。
``entrypoints/cli/runtime.py`` 是唯一解析运行时开关的地方。

Constraints:
    * 字段在被配置的那几层构造时读一次；少数几个（``num_blocks``）在
      ``__post_init__`` 里定值。引擎建好之后再改字段，不会重新分配任何按它
      计算过尺寸的东西。
    * 默认值面向 16 GB 的 BI-V100。调大 ``max_seq``、``num_blocks`` 或开启
      ``vision`` 都会改变常驻显存，必须拿这个预算核对。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EngineConfig:
    """一个已加载引擎的静态配置。

    默认值就是 ``NOTES.md`` 里那些基准数字的测量条件；为别的卡调出来的值应该
    放在 ``entrypoints/cli/runtime.py`` 那一层，而不是这里。
    """

    model_dir: str
    #: 上下文窗口。KV 池和 RoPE 表都按它预先分配，所以这是一笔显存承诺，不是
    #: 运行时才检查的上限。
    max_seq: int = 262144
    #: 模型的调优开关：加载时实测几种内核变体，留下最快的那个。要多花几秒启动
    #: 时间，所以测试里关掉。
    bench: bool = True
    #: 二维权重保持 int8（常驻 4 GiB）而不是 fp16（8 GiB）。在 16 GB 的卡上，
    #: 这就是 KV 池放得下和放不下的区别。
    quant: bool = True
    #: 全注意力层从共享的分页 int8 池里取 KV。
    prefix_cache: bool = True
    #: 每个 KV 块的 token 数。要让 ``max_seq`` 能按它整除，基数树才落得在块边界
    #: 上；64 是快照间隔的下限。
    block_size: int = 64
    #: 池里的物理块数；``-1`` 表示「一条序列的量」，在 ``__post_init__`` 里按
    #: ``max_seq`` 和 ``block_size`` 算出来。
    num_blocks: int = -1
    #: 基数树最多保留多少个线性注意力循环状态快照。一个快照约 50 MB，所以这
    #: 其实是缓存那笔显存预算的另一种写法。
    snapshots: int = 8
    #: 每 N 个 token 打一次快照（向上取整到整块）。
    snapshot_interval: int = 512
    #: 每个投机步的草稿数（0 = 关闭）。需要 checkpoint 里带 MTP 头。
    spec_k: int = 0
    #: 允许在*批处理*引擎里做投机。默认关闭：批处理的验证每个候选 token 都要
    #: 过一遍权重，在这张卡上比普通批解码步还慢（见 NOTES.md），尽管输出完全
    #: 一致。单流投机不看这个开关。
    spec_batch: bool = False
    #: 批验证用的近似 KV 选择：每行只保留得分最高的 k 个历史 KV 块（由草稿模型的
    #: 前瞻和主模型最后一次注意力挑出）再加一个近期窗口，而不是读整个上下文。
    #: 0 = 精确全注意力。隐含开启 ``spec_batch``。
    spec_sparse_k: int = 0
    #: 一条序列的缓存 token 达到这个数之后才在批处理引擎里投机：低于它时贪心
    #: 步更便宜（批验证要多付一遍权重），高于它共享 KV 的验证才划算。0 = 总是。
    spec_min_ctx: int = 0
    #: 预填充 GEMM 的 M 维。这张卡上 rocBLAS 在 2048 token 附近到顶。
    chunk: int = 2048
    #: 一个请求在一个调度步里最多能预填充多少 prompt token。这是*调度*开关，
    #: 故意不复用 ``chunk``：``chunk`` 是模型调优时定下的 GEMM 分块，而这个决定
    #: 单个请求能占着卡多久才轮到别人。两者耦合的话，重新调 GEMM 就会改变谁
    #: 先被调度。
    prefill_quantum: int = 2048
    #: 一步里解码的序列数。调度器不会放进超过它能服务的数量，且第
    #: ``capacity-1`` 行留给预填充。
    max_num_seqs: int = 1
    #: 一个调度步的 token 预算（预填充分块 + 解码 token）。
    max_num_batched_tokens: int = 2048
    #: 一次生成的硬墙钟上限，单位秒（0 = 不限）。
    max_seconds: float = 0.0
    #: 把解码步 capture 成设备图。``False`` 表示每个 token 都跑 eager 步，更慢，
    #: 但这是服务一个图回放不可信的后端的唯一办法：图如果回放了过期输出，是
    #: 静默失败，所以这个决定关乎正确性，不是调优项。传到这之前，服务端已经
    #: 把它和 ``caps.graph`` 取了与。
    graph: bool = True

    # 多模态（图像）服务。
    #: 一个可读的 503，而不是让图像占位符走纯文本预填充、产出一堆没人看得懂的
    #: 是否支持图像。``False``（``--no-vision``）不加载视觉塔，带图请求直接回
    #: 垃圾输出。
    vision: bool = True
    #: 视觉塔的二维权重保持 int8（外加每行 fp32 scale）而不是 fp16：常驻约
    #: 0.31 GiB 而不是 0.62 GiB，代价是每个请求多一点点开销。
    vision_int8: bool = False
    #: 唯一真正影响显存和延迟的开关：更大的图会被处理器降采样。1 MP 约等于
    #: 1024 个图像 token，预填充代价与 1k 文本 token 相当。
    image_max_pixels: int = 1048576
    #: 官方下限：更小的图会被*放大*到这个尺寸，所以一张图最少也要 64 个 token。
    image_min_pixels: int = 65536
    #: 每个请求的图像数（四张就是 4k 图像 token，已经是一次很大的预填充）。
    max_images: int = 4
    #: 单张图像解码后的字节数（请求里 base64 表示约为它的 4/3）。
    max_image_bytes: int = 8388608
    #: 一个请求里所有图像片段的 base64 字符数（解码前）。
    max_image_total_bytes: int = 16777216

    def __post_init__(self) -> None:
        """把哨兵默认值换算成具体尺寸。

        ``num_blocks = -1`` 表示「一条序列的量」，单流引擎也就只需要这么多。
        批处理服务要的是一个多序列共用的池，所以这个字段保持显式，而不是到
        分配现场再推。
        """
        if self.num_blocks < 0:
            self.num_blocks = (self.max_seq + self.block_size - 1) // self.block_size
