"""常驻推理运行时。

Module: ``coffin.serving``。这个类负责把模型准备好，并把生命周期、生成和对话
行为分派给各自专门的 mixin。

Constraints:
    * 模型线程始终是引擎状态的唯一写者。
    * socket 协议保持按行，且不改变。
    * 设备与图相关的决定归后端，不归本文件：这里唯一涉及设备的读取是下面的空闲
      显存保护，而在没有该查询的后端上它必须回答「未知」而不是「0」，否则这段
      保护会在一台从来没缺过预算的主机上把批投机关掉。
"""

from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path

import backends
import engine_api
from engine import llm_engine
from engine.config import EngineConfig
from engine.sequence import SamplingParams

from .runtime_parts import (
    RuntimeChatMixin,
    RuntimeGenerationMixin,
    RuntimeLifecycleMixin,
)
from .runtime_support import (MAX_SEQ, MODEL, MMError, SOCK, _eos_ids, free_bytes)


class Server(
    RuntimeLifecycleMixin,
    RuntimeGenerationMixin,
    RuntimeChatMixin,
):
        def __init__(self, model_dir: str, max_seq: int, bench: bool, spec_k: int = 0,
                     int8: bool = True, prefix_cache: bool = True, block_size: int = 64,
                     snapshots: int = 8, snapshot_interval: int = 512,
                     max_seconds: float = 0.0, max_num_seqs: int = 1,
                     num_blocks: int = 0, spec_batch: bool = False,
                     spec_sparse_k: int = 0, spec_min_ctx: int = 0,
                     vision: bool = True, vision_int8: bool = False,
                     image_max_pixels: int = 1048576, image_min_pixels: int = 65536,
                     max_images: int = 4, max_image_bytes: int = 8388608,
                     graph: bool = True, prefill_quantum: int = 2048) -> None:
            """加载 checkpoint、定下 ``EngineConfig``，再把引擎建起来。

            Args:
                model_dir: checkpoint 目录；它的 ``config.json`` 决定架构，它的
                    分词器负责渲染对话模板。
                max_seq: prompt 和输出共享的上下文窗口。KV 缓存按它预先分配，所以
                    它是最大的一项显存开销。
                bench: 保留旧的单流 bench 路径可用。
                spec_k: 每个投机步的草稿数，0 表示关闭。前缀缓存开着时会被静默
                    改成 0，因为快照不携带草稿模型的缓存。
                int8: 把二维权重量化成 int8。作为默认值，是因为 256k 窗口和
                    fp16 权重没法同时塞进 16 GB。
                prefix_cache: 启用分页基数前缀缓存。
                block_size: KV 块的 token 粒度；前缀复用只在整块边界上生效。
                snapshots: 循环状态快照预算，每份约 50 MB。
                snapshot_interval: 每 N 个 token 打一次快照，取整到块。
                max_seconds: 一次生成的墙钟上限，0 表示不限。
                max_num_seqs: 批解码步里的行数；1 就是原来的单流引擎。
                num_blocks: KV 池的块数，0 表示用
                    ``max_seq / block_size * max_num_seqs``。
                spec_batch: 在批处理引擎里投机。要显式打开，因为它的验证每个候选
                    token 都要过一遍权重。
                spec_sparse_k: 近似的批验证：每行只保留历史 KV 块里的 top-k。
                    隐含 ``spec_batch``。
                spec_min_ctx: 缓存 token 数超过它才在批处理引擎里投机，0 表示总是。
                vision: checkpoint 有视觉塔时加载它。
                vision_int8: 视觉塔的二维权重保持 int8，每个请求一次反量化一个
                    GEMM。
                image_max_pixels: 处理器服务的最大图像；真正影响显存和延迟的开关，
                    更大的都会被降采样。
                image_min_pixels: 处理器放大之后的最小图像，一张图因此最少也要
                    64 个 token。
                max_images: 一个请求接受的图像数。
                max_image_bytes: 单张图像允许的解码字节数。
                graph: 把解码步 capture 成设备图。调用方会拿用户的 ``--no-graph``
                    和后端自己的 ``caps.graph`` 算出实际结论再传进来，因为这个值
                    必须在 ``build()`` capture 之前到达引擎；报告
                    ``caps.graph=False`` 的后端，不管要什么都是 eager。
                prefill_quantum: 一个请求每步最多预填充多少 prompt token。配合
                    调度器的 token 预算，它才挡住一个长 prompt 把后面每个短请求
                    都拖慢；0 表示「一个分块可以用掉整份预算」。

            Raises:
                RuntimeError: checkpoint 的 ``config.json`` 没有点名任何已登记的
                    架构，或者权重加载不起来。两者都是启动期故障：去修
                    ``--model`` 路径，不要在循环里重试。
            """
            from transformers import AutoTokenizer

            self.model_dir = model_dir
            # 这个服务在服务什么，供启动日志和 ``ping`` 使用
            self.info = engine_api.describe(model_dir)
            self.max_seq = max_seq
            self.bench = bench
            self.spec_k = spec_k
            self.int8 = int8
            self.prefix_cache = prefix_cache
            self.block_size = block_size
            self.snapshots = snapshots
            self.snapshot_interval = snapshot_interval
            # 每次生成的硬墙钟上限（0 = 不限）；单个请求可以把它调低
            self.max_seconds = float(max_seconds)
            #: 实际的图决定：调用方的要求与后端能 capture 的能力取与。不能回放图
            #: 的后端会静默产出过期输出，所以能力优先于开关。
            self.graph = bool(graph) and bool(backends.caps().graph)
            self.pc = None
            self.n_req = 0
            self.n_reload = 0

            # 图像相关
            self.vision = bool(vision)
            self.vision_int8 = bool(vision_int8)
            self.image_max_pixels = int(image_max_pixels)
            self.image_min_pixels = int(image_min_pixels)
            self.max_images = int(max_images)
            self.max_image_bytes = int(max_image_bytes)
            #: checkpoint 自带的处理器（``Qwen3VLProcessor``：图像处理器 + 分词器）。
            #: 第一次带图请求时才加载，不在启动时：它要花约 2 s，纯文本服务不该为
            #: 它付钱——而且一个从没见过图像的纯文本容器压根不需要 transformers
            #: 的图像栈。``_processor_failed`` 记住加载失败，坏掉的 checkpoint
            #: 因此不会在每个请求上重试并卡住。
            self.processor = None
            self._processor_failed: str | None = None
            self.n_image_req = 0
            #: 所有前向都由唯一的引擎线程驱动；各连接只提交请求并排空各自队列，
            #: 所以可以同时有多个请求在飞，由调度器把它们解码步凑成批。
            self._lock = threading.RLock()
            self._cv = threading.Condition(self._lock)
            self._outputs: dict[int, "queue.Queue"] = {}
            self._stop = False
            self._paused = False
            self._loop_thread: threading.Thread | None = None
            batched = max(1, int(max_num_seqs)) > 1
            #: 批投机需要单独选择开启：它的验证对每个候选 token 都要过一遍权重，
            #: 所以在这张卡上比普通批解码步还慢，尽管输出完全一致。
            self.spec_batch = bool((spec_batch or spec_sparse_k > 0) and batched)
            self.spec_sparse_k = int(spec_sparse_k) if self.spec_batch else 0
            self.spec_min_ctx = int(spec_min_ctx)
            if prefix_cache and spec_k and not self.spec_batch:
                # 前缀快照不携带 MTP 草稿模型的缓存，命中之后草稿模型看到的是过期
                # 上下文（输出仍然精确，接受率下降）。与其静默降级，不如在前缀缓存
                # 开着时把投机关掉。
                print("[serve] --spec-k is ignored while the prefix cache is on")
                self.spec_k = 0
            if spec_k and batched and not self.spec_batch:
                print("[serve] batched speculation is off; pass --spec-batch to enable it "
                      "(measured 0.79x plain batch decode on this card)")
            if spec_k and not self.graph:
                # 解码步和验证步都是图回放；关了 capture 就没有可回放的东西，投机
                # 会在第一步就抛异常。与其让请求跑到一半才失败，不如现在说清楚。
                print("[serve] --spec-k is ignored with graph capture off")
                self.spec_k = 0
            #: 引擎开关的唯一事实来源；LLM 引擎、它的调度器和 runner 都从这里读。
            blocks = int(num_blocks) if num_blocks else (max_seq // block_size) * max(1, max_num_seqs)
            self.cfg = EngineConfig(
                model_dir=model_dir, max_seq=max_seq, bench=bench, quant=int8,
                prefix_cache=prefix_cache, block_size=block_size,
                num_blocks=blocks, snapshots=snapshots,
                snapshot_interval=snapshot_interval,
                spec_k=self.spec_k, spec_batch=self.spec_batch,
                spec_sparse_k=self.spec_sparse_k,
                spec_min_ctx=self.spec_min_ctx,
                max_seconds=self.max_seconds,
                max_num_seqs=max(1, int(max_num_seqs)),
                # 0 表示「不另设上限」：一个分块可以用掉整步预算，也就是分块之前的
                # 行为。
                prefill_quantum=(int(prefill_quantum) if prefill_quantum > 0
                                 else EngineConfig.__dataclass_fields__[
                                     "max_num_batched_tokens"].default),
                graph=self.graph)
            t0 = time.time()
            # 常驻不释放；int8 把它减半（4.0 GiB 而不是 8.1），16 GB 的卡上才腾得出
            # 空间放下长上下文 KV 缓存
            self.weights = engine_api.load_weights(
                model_dir, quant=int8, **self._load_weight_kwargs())
            self.load_s = time.time() - t0
            self.tok = AutoTokenizer.from_pretrained(model_dir)
            self.eos = _eos_ids(self.tok)
            if self.cfg.spec_k and self.cfg.spec_batch:
                # 批 MTP 要多占约 1.25 GiB 的回滚快照加上草稿模型的分页 KV；宁可
                # 在这里提前拒绝，也不要到 capture 里面才 OOM。报告不了空闲显存的
                # 后端回答 -1，这个值绝不能读成「空闲 0 字节」——那会把一个预算充足的
                # 主机上的功能关掉。
                free = free_bytes()
                if 0 <= free < 1.5 * 2**30:
                    print(f"[serve] --spec-batch disabled: only {free / 2**30:.2f} GiB free "
                          f"(batched MTP needs ~1.5 GiB)")
                    self.cfg.spec_k = 0
                    self.cfg.spec_batch = False
            self._build()
