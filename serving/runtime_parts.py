"""常驻推理服务的运行时 mixin。

Module: ``coffin.serving``。每个 mixin 管一个生命周期关注点，操作的是
``runtime_service.Server`` 建好的状态。

Constraints:
    * mixin 不持有进程状态；模型、分词器和请求队列的准备工作仍然只在
      ``Server.__init__`` 一处完成。
    * 这里不写任何平台事实。设备、显存读数、计时屏障以及「有没有内核」全部来自
      ``backends``；读数辅助函数放在 ``runtime_support``，因为它们必须在没有
      加速卡的主机上退化成空操作，而那里 ``COFFIN_BACKEND=reference`` 仍要把
      启动横幅和诊断跑起来。
    * 图像预处理是这里唯一需要 checkpoint 处理器的路径，而且只从带 ``images``
      的请求进入：纯文本部署绝不能 import 视觉栈。
"""

from __future__ import annotations

import gc
import json
import queue
import threading
import time
from pathlib import Path

import torch

import backends
import engine_api
from engine import llm_engine
from engine.config import EngineConfig
from engine.sequence import SamplingParams
from utils.tool_call import parse_tool_calls, split_thinking

from .runtime_support import (MMError, _decode_image, allocated_gib, empty_cache,
                              profiler_activities, reserved_gib, sync)



class RuntimeLifecycleMixin:
    """启动、拆解和 reload：模型、引擎和视觉塔。

    它管服务生命里的三种转换——首次构建、``reload`` 之后的整体重建、一次一步的
    引擎循环——关于请求内容一概不管。之所以不放进 ``Server.__init__``，是因为
    reload 重跑的恰好就是构建那一步，两者不能各写一份。

    Constraints:
        * ``_build`` 是唯一构造模型或引擎的地方，reload 因此不可能跳过首次启动
          做过的任何一步。
        * 先拆再建：池加上 captured 的图已经超过这张卡的容量，一边建新的还一边
          留着旧的，结果不是「不整洁」而是直接 OOM。
    """

    def _load_weight_kwargs(self) -> dict:
        """本服务的权重怎么加载：文本量化加视觉策略。

        视觉塔和文本权重共用一个常驻字典（reload 才能从显存重建），但它有自己的
        内存策略：默认 fp16，``--vision-int8`` 时 int8，``--no-vision`` 时压根
        不加载。只有 checkpoint 声明了视觉配置才转发这些参数，纯文本 checkpoint
        因此仍走最朴素的调用。

        Returns:
            给 ``engine_api.load_weights`` 的额外关键字参数；没有
            ``vision_config`` 的 checkpoint 返回空字典。
        """
        raw = json.loads((Path(self.model_dir) / "config.json").read_text())
        if not raw.get("vision_config"):
            return {}
        return {"vision": self.vision, "vision_quant": self.vision_int8}

    def _build(self) -> None:
        """建模型、建引擎，然后拉起视觉塔。

        从 ``__init__`` 里拆出来，是因为 ``_reload`` 会拿已经在显存里的张量把
        这一步原样再跑一遍。``build_s`` 由 ``ping`` 和启动日志上报，构建慢了
        不用 profiler 也看得见。
        """
        t0 = time.time()
        # 建哪个模型族由 checkpoint 配置决定；``serve`` 自己只认识
        # engine_api.Model 这个契约。
        self.model = engine_api.create_model(self.model_dir, max_seq=self.max_seq,
                                            bench=self.bench, weights=self.weights,
                                            paged=self.prefix_cache)
        # 前缀缓存、块池、调度器和 captured 的图都归引擎；``serve`` 只留传输和渲染。
        self.engine = llm_engine.LLMEngine(self.cfg, self.model)
        self.engine.eos = set(self.eos)
        self.engine.build()
        self.pc = self.engine.pc
        # 视觉塔放在这里建，在引擎的池和图都就绪之后：它是 eager 模块，从不进
        # 解码图，所以最后起来也仍能在第一个带图请求时用上。``--no-vision`` 跳过
        # 它，这是把那 0.62 GiB 拿回来的唯一办法。
        if self.vision and self.model.vision_sd and self.model.vision is None:
            try:
                t_v = time.time()
                self.model.enable_vision(quant=self.vision_int8)
                print(f"[serve] vision tower ready in {time.time() - t_v:.1f}s "
                      f"(int8={self.vision_int8}, "
                      f"{len(self.model.vision_sd)} tensors)", flush=True)
            except Exception as exc:  # noqa: BLE001 - text serving must still work
                print(f"[serve] vision tower unavailable: {type(exc).__name__}: {exc}",
                      flush=True)
                self.vision = False
        elif not self.vision:
            self.model.vision_sd = {}
        self.build_s = time.time() - t0

    def _reload(self) -> None:
        """用显存里的内容重建模型、引擎和后端算子表。

        老模型（缓冲区加上它 capture 的图池）先拆掉，否则两个模型在 16 GB 里放
        不下。图是在 ``close()`` 里显式置空的：活着的 ``CUDAGraph`` 会把它私有的
        内存池钉住，而那个池约占模型体积的一半（所有分组投影的副本），所以光把
        ``self.model`` 丢掉是不够的。
        """
        import gc

        def _mem(tag: str) -> None:
            print(f"[reload] {tag}: alloc={allocated_gib():.2f} "
                  f"reserved={reserved_gib():.2f} GiB", flush=True)

        _mem("before")
        old = self.model
        self.engine = None
        self.model = None
        if old is not None:
            old.close()          # 置空图和层（图会钉住自己的池）
        del old
        gc.collect()
        sync()
        empty_cache()
        _mem("after teardown")

        # 只重新解析后端这一层，不逐模块 reload：内核构建在后端模块上是记忆化的，
        # 重新 import 算子表会让引擎继续调用闭包里的已拆解上下文。
        backends.reload()

        # 每个引擎模块按依赖先后 reload：顺序反了会让旧类绑到新张量上。
        engine_api.reload()
        self.processor = None    # transformers 的类重新导入过：惰性重建
        self._processor_failed = None
        self.n_reload += 1
        self._build()
        _mem("after build")

    def start(self) -> None:
        """在自己的线程上跑服务循环。

        幂等：第二次调用是空操作，``serve`` 因此可以在开始接受连接之前就把循环
        起好，不必自己记状态。
        """
        if self._loop_thread is not None:
            return
        self._loop_thread = threading.Thread(target=self._engine_loop,
                                             name="engine-loop", daemon=True)
        self._loop_thread.start()

    def stop(self) -> None:
        """请求引擎线程退出，并最多等它 5 s。

        线程本身也按 0.2 s 超时轮询条件变量，所以漏掉一次通知的代价是一步，而不是
        卡死。join 超时不算错误：它是守护线程，进程本来也要退了。
        """
        self._stop = True
        with self._cv:
            self._cv.notify_all()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)
            self._loop_thread = None

    def _engine_loop(self) -> None:
        """每次迭代跑一次 ``engine.step()``，把 token 分发给各请求。

        0.2 s 的等待给漏掉的通知兜底：循环还必须能注意到 ``_paused``（有诊断代码
        正在动模型）和 ``_stop``，这两个都不能无限等下去。
        """
        while not self._stop:
            with self._cv:
                while not self._stop:
                    engine = self.engine
                    if engine is not None and not self._paused \
                            and engine.scheduler.has_work():
                        break
                    self._cv.wait(0.2)
                if self._stop:
                    return
            try:
                with self._lock:
                    events = self.engine.step()
            # 服务要继续活着，所以把错误报给每个在途请求，而不是把线程弄挂。
            except Exception as exc:  # noqa: BLE001
                print(f"[serve] engine step failed: {exc}", flush=True)
                self._fail(str(exc))
                self.engine.reset()
                continue
            for seq, tok, done in events:
                with self._lock:
                    out_q = self._outputs.get(seq.seq_id)
                if out_q is not None:
                    out_q.put({"token": tok, "done": done, "error": None,
                               "prefill_s": seq.prefill_s})
                if done:
                    with self._lock:
                        self._outputs.pop(seq.seq_id, None)

    def _fail(self, message: str) -> None:
        """把一次引擎故障报告给每个在途请求，并丢掉所有队列。

        失败的 ``step()` 已经把这一批的状态弄丢了，所以每个等待中的连接都直接收
        到错误，而不是继续阻塞在一个再也没有人来喂的队列上。字典在锁内整体换掉，
        并发提交的请求因此不可能落进被丢弃的那一份。

        Args:
            message: 故障文本，原样发给每个在等待的连接。
        """
        with self._lock:
            pending, self._outputs = self._outputs, {}
        for out_q in pending.values():
            out_q.put({"token": None, "done": True, "error": message,
                       "prefill_s": 0.0})


class RuntimeGenerationMixin:
    """文本生成：提交请求、收回 token、夹紧预算。

    这里的一切都与引擎的请求队列打交道，所以一个请求和其他所有请求一样由同一套批
    处理步进循环服务；唯一例外是 ``_spec_generate``，它直接驱动模型，因为投机循环
    需要整个验证窗口的 logits。

    Constraints:
        * 输出预算在请求提交之前就要按上下文窗口夹紧：解码步写 KV 时不做越界检查，
          RoPE 表也是按 ``max_seq`` 定尺寸的，预算不夹就会撞穿两者。
        * ``_spec_generate`` 要求验证图是按 ``k + 1`` 个 token capture 的，而且只支持
          贪心：接受判断是拿主模型的 argmax 与草稿比对。
    """

    def _submit(self, ids: list[int], params: SamplingParams, mm=None):
        """把一个请求排进队列，返回它的序列和 token 队列。

        ``mm`` 挂在 :class:`~sequence.Sequence` 上（``seq.mm``），runner、块管理器和
        基数树读的都是它：树拿 ``mm.digest`` 给块键加盐，两张不同的图因此永远无法
        共用同一段缓存 KV。队列是在引擎循环 ``step()`` 所用的同一把锁*内部*注册的，
        所以一条预填充中的序列总有一个等着收 token 的队列。

        Args:
            ids: prompt 的 token id，已经渲染好，带图请求的话也已经展开成处理器
                的布局。
            params: 这个请求的采样与停止策略。
            mm: 请求的 ``MMInput``，文本 prompt 传 ``None``。

        Returns:
            ``(seq, out_q)``：引擎侧的序列，以及将收到
            ``{"token", "done", "error", "prefill_s"}`` 事件的队列。
        """
        out_q: queue.Queue = queue.Queue()
        with self._lock:
            seq = self.engine.add_request(ids, params, mm=mm)
            self._outputs[seq.seq_id] = out_q
            self._cv.notify()
        return seq, out_q

    def _blocking_generate(self, prompt_ids: list[int], max_new: int,
                           temperature: float, deadline: float = 0.0, mm=None):
        """提交一个请求，一直收 token 到引擎说完成。

        Args:
            prompt_ids: prompt 的 token id。
            max_new: 以 token 计的输出预算，已经由 ``_room`` 夹紧。
            temperature: 0 表示贪心。
            deadline: 以 epoch 秒计的截止时间，0 表示不限。
            mm: 请求的 ``MMInput``，或 ``None``。

        Returns:
            ``(tokens, prefill_s, decode_s)``。``decode_s`` 不含预填充，且下界
            取 0，因为两段计时读的是不同的时钟，短回答上可能倒挂几微秒。

        Raises:
            RuntimeError: 引擎为这个请求报了错。失败的那一步已经被重置，调用方
                可以重新发起一个新请求，但这条序列不能再用。
        """
        seq, out_q = self._submit(prompt_ids,
                                  self._sampling(max_new, temperature, deadline), mm=mm)
        t0 = time.time()
        out: list[int] = []
        while True:
            ev = out_q.get()
            if ev["error"]:
                raise RuntimeError(ev["error"])
            if ev["token"] is not None:
                out.append(ev["token"])
            if ev["done"]:
                break
        total = time.time() - t0
        pre = seq.prefill_s
        return out, pre, max(total - pre, 0.0)

    def _room(self, n_prompt: int, max_new) -> tuple[int, bool]:
        """把输出预算夹到上下文窗口还剩的空间以内。

        窗口由 prompt *和* 输出共享（总共 262144 个 token），而解码步自身没有越界
        检查：它按 ``pos`` 写 KV 缓存，RoPE 表也只有 ``[max_seq]`` 那么长，预算
        超过剩余空间就会同时撞穿两者。长上下文客户端经常看都不看 prompt 长度就要一
        整个窗口，所以引擎把它夹紧，而不是让它越界。

        Args:
            n_prompt: 图像展开之后的 prompt token 长度。
            max_new: 请求的输出预算；``None`` 或负数表示「能放多少写多少」，也就是
                调用方没有表态。

        Returns:
            ``(budget, window_full)``。``budget`` 是这个请求实际能生成多少，永远
            不为负——刚好占满窗口的 prompt 一个 token 也生成不了。``window_full``
            表示这个*请求*是被窗口截断的，而不是自己跑完的。两者都随回复返回，
            客户端才能分清「完成」和「没地方了」。
        """
        room = max(0, self.max_seq - n_prompt)
        if max_new is None or int(max_new) < 0:
            return room, False
        return max(0, min(int(max_new), room)), int(max_new) > room

    def _sampling(self, max_new: int, temperature: float,
                  deadline: float = 0.0) -> SamplingParams:
        """构造引擎调度器看到的、按请求生效的解码策略。

        Args:
            max_new: 输出预算，已经过 ``_room`` 夹紧。
            temperature: 0 表示贪心，也正是投机路径的开启条件。
            deadline: 以 epoch 秒计的截止时间，0 表示不限。

        Returns:
            采样参数，附上本服务完整的停止 id 集合，这样每条路径都会停在模板自己的
            轮次结束 token 上。
        """
        return SamplingParams(temperature=temperature, max_tokens=max_new,
                              eos=set(self.eos), deadline=deadline)

    def _deadline(self, req: dict) -> float:
        """这次生成的墙钟上限（epoch 秒，0 = 不限）。

        请求只能把服务级的 ``--max-seconds`` 改短，永远不能改长：这一刀是为了兜住
        静默下来却不关 socket 的客户端，所以客户端没法把自己排除在外。

        Args:
            req: 请求字典；``max_seconds`` 覆盖服务默认值。

        Returns:
            绝对的截止时刻；没有上限时返回 0。
        """
        secs = float(req.get("max_seconds") or self.max_seconds or 0.0)
        return time.time() + secs if secs > 0 else 0.0

    def _cached_tokens(self) -> int:
        """上一个 prompt 有多少是从前缀缓存里直接来的。

        Returns:
            匹配到的前缀长度，什么都没复用时是 0。在请求结束之后读，因为批处理步
            是按序列更新它的，而只有回复需要这个数。
        """
        if self.pc is not None:
            return self.pc.last_matched
        return int(getattr(self.engine, "last_matched", 0))

    @torch.inference_mode()
    def _generate(self, prompt_ids: list[int], max_new: int, temperature: float,
                  deadline: float = 0.0, mm=None) -> tuple[list[int], float, float]:
        """走引擎的批处理路径做非流式生成。

        Args:
            prompt_ids: prompt 的 token id。
            max_new: 输出预算，已经过 ``_room`` 夹紧。
            temperature: 0 表示贪心。
            deadline: 以 epoch 秒计的截止时间，0 表示不限。
            mm: 请求的 ``MMInput``，或 ``None``。

        Returns:
            ``(tokens, prefill_s, decode_s)``，与 ``_blocking_generate`` 返回的
            一致。

        Raises:
            RuntimeError: 引擎为这个请求报了错。
        """
        return self._blocking_generate(prompt_ids, max_new, temperature, deadline, mm=mm)

    @torch.inference_mode()
    def _spec_generate(self, prompt_ids: list[int], max_new: int, k: int,
                       deadline: float = 0.0,
                       mm=None) -> tuple[list[int], float, float]:
        """贪心投机解码，直接驱动模型而不是走调度器。

        它绕过引擎的请求队列：先重置模型、预填充，然后直接在 ``spec_step`` 上循环。
        ``spec`` 操作因此能拿它和普通路径做 A/B；代价是验证图必须已经按 ``k + 1``
        个 token capture 过（``capture_verify``），否则第一步就会抛异常。

        只支持贪心：接受/拒绝判断是拿主模型的 argmax 与草稿比对，这对贪心成立，对
        采样不成立。

        Args:
            prompt_ids: prompt 的 token id。
            max_new: 输出预算，已经过 ``_room`` 夹紧。
            k: 每步的草稿数；验证图必须已按 ``k + 1`` 个 token capture。
            deadline: 以 epoch 秒计的截止时间，0 表示不限。
            mm: 请求的 ``MMInput``，或 ``None``。支持而不是拒绝：
                ``prefill(mm=...)`` 会把请求的 ``rope_delta`` 写进去，草稿模型和
                验证图每步都读它（``spec_step`` 会把它拷进 ``mtp.rope_delta``），
                所以带图 prompt 和文本 prompt 投机的行为完全一样。

        Returns:
            ``(tokens, prefill_s, decode_s)``。``decode_s`` 只覆盖投机循环，所以
            可以直接和普通路径的解码时间对比。
        """
        m = self.model
        m.reset()
        t0 = time.time()
        logits = m.prefill(prompt_ids, mm=mm)
        t_pre = time.time() - t0
        m.n_drafted = m.n_accepted = m.n_spec_steps = 0
        out: list[int] = []
        t1 = time.time()
        while len(out) < max_new:
            if int(m.pos1) + m.verify_c > m.max_seq:
                break
            if deadline and time.time() >= deadline:
                break
            toks, logits = m.spec_step(logits, k)
            stop = False
            for t in toks:
                if t in self.eos:
                    stop = True
                    break
                out.append(t)
                if len(out) >= max_new:
                    break
            if stop:
                break
        return out[:max_new], t_pre, time.time() - t1


class RuntimeChatMixin:
    """对话渲染、图像预处理和操作分派。

    它把 OpenAI 形状的请求变成引擎能跑的 token id，再把引擎的 token 流变回回复或
    SSE 序列。诊断类操作也归它管，这也是它们在这里而不是在传输层被拒绝的原因：
    它们直接动模型，所以必须和解码步拿同一把锁。

    Constraints:
        * 文本和图像共用一条渲染规则：模板照常渲染 prompt，图像路径只是之后把同一
          段文本交给处理器，所以加入视觉能力不可能改变纯文本 prompt 的渲染结果。
        * 图像路径完全跑在预填充之前，所以每一次拒绝都是一个干净的状态码，而不是
          一个生成到一半的回答。
    """

    def _vision_ready(self) -> bool:
        """这个服务到底能不能处理带图请求。

        从 checkpoint 配置读，而不是看 ``self.vision``，因为视觉塔只在第一个带图
        请求时才惰性构建：光看那个开关会对纯文本 checkpoint 回答「能」，把一个本该
        清楚的 503 变成预填充跑到一半才失败。

        Returns:
            没传 ``--no-vision`` 且 checkpoint 声明了 ``vision_config`` 时返回
            ``True``。
        """
        if not self.vision:
            return False
        raw = json.loads((Path(self.model_dir) / "config.json").read_text())
        return bool(raw.get("vision_config"))

    def _ensure_processor(self):
        """首次使用时才加载 checkpoint 的图像处理器。

        惰性而不是启动时加载，因为除了分词器，它是 transformers 里唯一导入开销很大
        的部件，而纯文本部署根本碰不到它。那约 2 s 落在第一个带图请求上，那里相对
        一次本来就要几百毫秒的预填充只多一秒，而不是摊到每一次冷启动。

        Returns:
            加载好的 ``AutoProcessor``。

        Raises:
            MMError: 处理器加载不了（503）。坏掉的 checkpoint 不能每个请求都重试，
                所以失败会被记住并上报，而不是反复尝试。
        """
        if self.processor is not None:
            return self.processor
        if self._processor_failed is not None:
            raise MMError(f"the image processor is unavailable on this server: "
                          f"{self._processor_failed}", 503)
        try:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.model_dir)
        except Exception as exc:  # noqa: BLE001 - reported as a 503, not a crash
            self._processor_failed = f"{type(exc).__name__}: {exc}"
            raise MMError(f"could not load the image processor: "
                          f"{self._processor_failed}", 503) from None
        print(f"[serve] image processor ready: {type(self.processor).__name__} "
              f"({type(self.processor.image_processor).__name__})", flush=True)
        return self.processor

    def _prepare_mm(self, images: list[dict], text: str):
        """把带图请求的片段变成引擎用的 ``MMInput``。

        整条图像路径都跑在 prompt 预填充*之前*，所以每次拒绝都是干净的 400，而不是
        一个生成到一半的回答：

        1. 每张图做 PIL 解码 + 摆正 + 转 RGB（``_decode_image``）；
        2. checkpoint 的处理器处理 ``text``（渲染好的 prompt，其中的
           ``{"type": "image"}`` 标记已经变成
           ``<|vision_start|><|image_pad|><|vision_end|>``）——它把每张图缩到
           ``[image_min_pixels, image_max_pixels]`` 区间、切 patch，并把
           ``<|image_pad|>`` 段展开成每个合并 patch 一个 token；
        3. 自检展开后的数量恰好是 ``Σ grid.prod() / 4``，正是这一步把处理器版本
           漂移变成 400，而不是模型深处的一次下标错位；
        4. 视觉塔 -> 嵌入，``mm_input`` -> 三维位置。

        Args:
            images: 请求的 ``images`` 数组，每张图一个 ``{"b64", "mime"}`` 字典，
                按 prompt 顺序排列。
            text: 渲染好的 prompt，也就是处理器展开图像占位符之前的对话模板输出。

        Returns:
            ``(token_ids, mm)``。``token_ids`` 用的是处理器自己给的 id：带图
            prompt 上分词器和处理器本来一致，但从处理器取才能保证占位符和
            ``pixel_values`` 对得上（数量是断言过的，分歧不可能悄悄过去）。

        Raises:
            MMError: 服务没有视觉塔（503）；``images`` 为空或不是列表；请求带的
                图像超过 ``--max-images``；某张图解不开；处理器失败；或者占位符段
                与网格对不上。每一条都是客户端错误，带着传输层可以直接转发的状态码。
        """
        from utils.mm_input import (IMAGE_TOKEN_ID, MMInput, find_image_spans,
                                    image_digest, mrope_positions)

        proc = self._ensure_processor()
        if not self._vision_ready():
            raise MMError("this server cannot serve images (no vision tower was "
                          "loaded; see --no-vision)", 503)
        if not isinstance(images, list) or not images:
            raise MMError("`images` must be a non-empty array")
        if len(images) > self.max_images:
            raise MMError(f"too many images: {len(images)} > --max-images "
                          f"{self.max_images}")

        pil: list = []
        chunks: list[bytes] = []
        for i, item in enumerate(images):
            if not isinstance(item, dict) or "b64" not in item:
                raise MMError(f"image {i}: expected {{'b64': ..., 'mime': ...}}")
            img, raw = _decode_image(str(item["b64"]), str(item.get("mime") or ""),
                                     self.max_image_bytes)
            pil.append(img)
            chunks.append(raw)

        t0 = time.time()
        try:
            out = proc(text=[text], images=pil, return_tensors="pt",
                       size={"shortest_edge": self.image_min_pixels,
                             "longest_edge": self.image_max_pixels})
        except MMError:
            raise
        except Exception as exc:  # noqa: BLE001 - a bad image must not 500 the server
            raise MMError(f"image preprocessing failed: {type(exc).__name__}: {exc}") from None
        prep_ms = (time.time() - t0) * 1000

        ids = out["input_ids"][0].tolist()
        grid = out["image_grid_thw"].long()
        pixel_values = out["pixel_values"]
        # 每张图里一个合并后的视觉 patch 对应一个 prompt token。
        want = int((grid.prod(-1) // 4).sum())

        # 三个廉价检查合起来把整套对齐钉死，处理器或模板漂移因此在这里就失败，而
        # 不是变成模型内部的一次嵌入/KV 下标错位：
        #
        # 1. 处理器自带的分词器必须和服务端渲染 prompt 用的是同一个——若两者不同，
        #    每个 id 都会错，模型答出来的东西会完全不相干；
        # 2. 展开量必须*恰好*是「每个图像标记变成 grid.prod()/4 个 token」；未展开
        #    的渲染每张图只有一个 ``<|image_pad|>``，所以
        #    ``len(ids) == len(unexpanded) - n_images + Σ grid.prod()/4`` 正是
        #    处理器自己的契约；
        # 3. 由此得到的占位符段是模型要覆写的地方，它们的数量和顺序必须与视觉塔编码
        #    的网格一致。
        plain_ids = self.tok.encode(text)
        n_plain_pads = sum(1 for t in plain_ids if t == IMAGE_TOKEN_ID)
        if n_plain_pads != len(images):
            raise MMError(
                f"the rendered prompt has {n_plain_pads} image placeholders but the "
                f"request carries {len(images)} images; the template and the image "
                f"parts disagree")
        if len(ids) != len(plain_ids) - len(images) + want:
            raise MMError(
                f"image expansion does not match the processor's grid: {len(ids)} "
                f"prompt tokens vs {len(plain_ids)} unexpanded - {len(images)} "
                f"placeholders + {want} image tokens")
        spans = find_image_spans(ids)
        got = sum(e - s for s, e in spans)
        if len(spans) != len(images) or got != want:
            raise MMError(
                f"image placeholder mismatch: the prompt has {len(spans)} image runs "
                f"covering {got} tokens, the processor produced {len(images)} images "
                f"needing {want} tokens (image_pad count must equal "
                f"sum(grid.prod()/4))")

        embeds = self.model.image_embeds(pixel_values, grid)
        positions, rope_delta = mrope_positions(spans, grid, len(ids),
                                               device=backends.device)
        mm = MMInput(embeds=embeds, grid_thw=grid, positions=positions,
                     rope_delta=int(rope_delta),
                     digest=image_digest(chunks), spans=spans)
        self.n_image_req += 1
        print(f"[serve] mm: {len(images)} img grid={grid.tolist()} "
              f"img_tokens={want} n_prompt={len(ids)} rope_delta={rope_delta} "
              f"prep={prep_ms:.0f}ms ", flush=True)
        return ids, mm

    def _render_with_text(self, messages: list[dict], tools, enable_thinking: bool,
                          forced_prefix: str = "") -> tuple[list[int], str]:
        """用 checkpoint 自带的 jinja 模板渲染一段对话。

        用 checkpoint 的模板而不是手拼字符串，才能让工具 schema、思考模式和特殊
        token 与模型训练时见过的完全一致。

        Args:
            messages: OpenAI 对话格式的轮次。
            tools: 工具 schema 或 ``None``；模板会把它们渲染进 system 轮次，所以
                带工具的文本 prompt 并不是一个朴素 prompt。
            enable_thinking: 模板是否打开推理段。
            forced_prefix: 追加到 assistant 轮次的模板开头，用于实现
                ``tool_choice: required``。

        Returns:
            ``(token_ids, text)``。保留文本，是因为图像路径要把*渲染好的* prompt
            交给处理器，也因为调用方会看它的结尾判断思考段有没有被打开。
        """
        text = self.tok.apply_chat_template(
            messages, tools=tools, enable_thinking=enable_thinking,
            tokenize=False, add_generation_prompt=True)
        if forced_prefix:
            # ``tool_choice: required``（OpenAI 的 ``ChatCompletionToolChoiceOption``）
            # 是靠续写 assistant 轮次实现的：渲染好的 prompt 已经停在 assistant 头
            # 上，追加模板的开头之后，模型除了把这个调用写完没有别的出路。
            text += forced_prefix
        return self.tok.encode(text), text

    def _render(self, messages: list[dict], tools, enable_thinking: bool,
                forced_prefix: str = "") -> list[int]:
        """渲染好的对话的 token id；不需要文本形式。

        Args:
            messages: OpenAI 对话格式的轮次。
            tools: 工具 schema 或 ``None``。
            enable_thinking: 模板是否打开推理段。
            forced_prefix: 追加到 assistant 轮次的模板开头。

        Returns:
            可以直接提交的 prompt token id。
        """
        return self._render_with_text(messages, tools, enable_thinking, forced_prefix)[0]

    @staticmethod
    def _split(raw: str, enable_thinking: bool) -> tuple[str, str]:
        """在知道有没有要求思考的前提下，把推理段和答案分开。

        ``split_thinking`` 分不清「没有推理」和「推理在闭合标记之前就被截断了」：
        开着思考而文本里没有 ``</think>`` 时，到目前为止生成的全部*都是*推理。
        把它当答案返回，思考模型的内心独白就会变成一条看起来像回复的东西。

        Args:
            raw: 完整的生成文本，含强制开头。
            enable_thinking: 调用方是否向模板要求过推理。

        Returns:
            ``(thinking, answer)``。模型什么都没产出时两者都是空串；推理段被截断
            时答案是空串。
        """
        thinking, answer = split_thinking(raw)
        if enable_thinking and not thinking and "</think>" not in raw:
            thinking, answer = answer, ""
        return thinking, answer

    def _render_request(self, req: dict, thinking_on: bool, prefix: str):
        """渲染一个请求的 prompt -> ``(token_ids, text, MMInput|None)``。

        纯文本路径就是渲染、编码、不带 ``mm``。带 ``images`` 的请求用*同样的*
        方式渲染（模板把它的 ``{"type": "image"}`` 标记变成视觉分隔符），然后把
        渲染好的文本交给处理器，区别仅此而已——两条路径共用一条渲染规则，加入视觉
        因此不可能改变纯文本 prompt 的渲染结果。

        Args:
            req: 请求字典：``messages``、可选的 ``tools`` 和 ``images``。
            thinking_on: 模板是否应打开推理段。
            prefix: 强制工具调用的开头，调用方没要求 ``tool_choice: required`` 时
                为空。

        Returns:
            ``(token_ids, text, mm)``；文本请求的 ``mm`` 是 ``None``。带图请求
            的 token id 是处理器展开后的，不是分词器的。

        Raises:
            MMError: 强制工具调用生效时请求带了 ``images``（400），或者图像路径
                拒绝（状态码来自 ``_prepare_mm``）。两者互斥，因为强制调用需要
                assistant 轮次直接开口就是调用，没有留给视觉分隔符的位置。
        """
        messages = req.get("messages", [])
        images = req.get("images") or []
        ids, prompt = self._render_with_text(messages, req.get("tools"), thinking_on,
                                             prefix)
        if not images:
            return ids, prompt, None
        if prefix:
            raise MMError("tool_choice=required forces the assistant turn to open "
                          "with a call; it cannot be combined with images")
        mm_ids, mm = self._prepare_mm(images, prompt)
        return mm_ids, prompt, mm

    @torch.inference_mode()
    def stream_chat(self, req: dict):
        """把 ``chat`` 做成生成器：每步一行回复，供 SSE 客户端使用。

        先产出 ``type: "start"``（prompt 长度，以及 prompt 自身是否已经打开了
        推理段），然后每个解码出的片段产出一行 ``type: "token"``，最后是一个
        ``type: "done"``，字段与 ``chat`` 回复完全一致。这三行就是 OpenAI 入口
        转发的全部契约，所以新字段归 ``done``，绝不能新加一种行类型。

        token 行上的 ``text`` 是*增量*的：每步只解码新 token 的 id，流式开销因此
        与回答长度成线性。上一步的 id 只在它停在一个多字节字符中间时才留着，而这种
        情况最多也就一两个 id。

        被拒绝的请求以 ``type: "error"`` 连同状态码产出，而且在 ``start`` 行
        *之前*：那时 SSE 的 200 已经发出去了，OpenAI 入口会把这一行的状态码映射进
        流事件，而不是假装请求成功了。

        这里刻意不用投机解码：投机循环要等一整个 K+1 验证窗口才会返回，这正好与流式
        想要的相反。

        Args:
            req: 请求字典：``messages``、可选的 ``tools``、``images``、
                ``max_new``、``temperature``、``enable_thinking``、
                ``tool_prefix`` 和 ``max_seconds``。

        Raises:
            RuntimeError: 引擎为这个请求报了错。它是在 ``start`` 行之后抛的，
                所以传输层会关掉这条流，而不是发一个宣称成功的 ``done``。
        """
        prefix = str(req.get("tool_prefix") or "")

        # 强制调用必须开在答案最前面，所以这一轮关掉推理：否则模板会在调用该出现
        # 的位置先打开 ``<think>``。
        thinking_on = bool(req.get("enable_thinking", False)) and not prefix
        try:
            ids, prompt, mm = self._render_request(req, thinking_on, prefix)
        except MMError as exc:
            yield {"type": "error", "ok": False, "error": str(exc),
                   "status": exc.status}
            return
        n, window_full = self._room(len(ids), req.get("max_new"))
        temp = float(req.get("temperature", 0.0))
        deadline = self._deadline(req)
        yield {"type": "start", "n_prompt": len(ids), "max_new": n,
               "prompt_opens_think": prompt.rstrip().endswith("<think>"),
               "forced_tool": bool(prefix),
               "n_images": mm.num_images if mm is not None else 0,
               "image_tokens": len(mm) if mm is not None else 0}
        t0 = time.time()
        seq, out_q = self._submit(ids, self._sampling(n, temp, deadline), mm=mm)
        out: list[int] = []
        hold: list[int] = []             # 暂扣的 token id：UTF-8 编码不完整
        t1 = time.time()
        while True:
            ev = out_q.get()
            if ev["error"]:
                raise RuntimeError(ev["error"])
            nxt = ev["token"]
            if nxt is not None:
                out.append(nxt)

                # 增量解码：每步解码整个列表是 O(n^2)，长回答里它会成为主要开销
                # （一条 20k token 的回复要在分词器里耗掉几分钟）。单个 token 可能
                # 停在 UTF-8 序列中间，所以它的 id 先扣住，和接下来的几个一起解码，
                # 直到那个字符补齐。
                held = hold + [nxt] if hold else [nxt]
                text = self.tok.decode(held, skip_special_tokens=True)
                if text.endswith("\ufffd"):
                    hold = held
                else:
                    hold = []
                    if text:
                        yield {"type": "token", "id": nxt, "text": text}
            if ev["done"]:
                break
        if hold:                         # 这个字符始终没补全（被截断）
            text = self.tok.decode(hold, skip_special_tokens=True)
            if text:
                yield {"type": "token", "id": hold[-1], "text": text}
        t_dec = time.time() - t1
        t_pre = seq.prefill_s
        raw = self.tok.decode(out, skip_special_tokens=False)

        # 强制开头是在 *prompt* 里的，所以解析调用之前要先把它拼回生成文本前面——
        # 否则那个块看起来就是一段没有 `<tool_call>` 包着的 `<function=...>`。
        full = prefix + raw
        thinking, answer = self._split(full, thinking_on)
        calls = parse_tool_calls(answer)
        yield {"type": "done", "ok": True, "text": answer, "thinking": thinking,
               "raw": full, "tokens": out,
               "tool_calls": [c.to_openai_dict() for c in calls],
               "n_prompt": len(ids), "n_new": len(out), "max_new": n,
               "window_full": window_full, "cached_tokens": self._cached_tokens(),
               "prefill_ms": round(t_pre * 1000, 1),
               "decode_ms": round(t_dec * 1000, 1),
               "decode_tok_s": round(len(out) / t_dec, 2) if t_dec else None,
               # 生成为什么停下（仅当是引擎自己判定的）
               "stopped": ("deadline" if deadline and time.time() >= deadline else None),
               "spec_k": 0, "spec": None}

    def handle(self, req: dict) -> dict:
        """处理一个非流式操作。

        诊断类操作（bench/profile/ops/spec/reload/reset）直接动模型，所以有请求
        在飞时会被拒绝，跑的时候还会暂停引擎循环：``step`` 和诊断拿的是同一把锁。
        暂停正是它们安全的前提——reload 会把模型拆掉，profiler 抓取否则会和活着的
        一步交错在一起。

        Args:
            req: 一个请求字典；``op`` 默认是 ``ping``。

        Returns:
            回复字典，总是带 ``ok``。拒绝是 ``ok: False`` 加一个 ``error`` 字符串，
            不是异常：传输层会把它变成一行客户端能读的普通 JSON。
        """
        op = req.get("op", "ping")
        debug = op in ("reload", "reset", "bench", "profile", "ops", "spec")
        if not debug:
            return self._handle_impl(req)
        engine = self.engine
        if op in ("bench", "profile", "ops", "spec") and engine is not None \
                and engine.batching:
            return {"ok": False, "error": "diagnostic ops need the single-stream "
                                          "engine (run without --max-num-seqs)"}
        if engine is not None and (engine.scheduler.waiting or engine.scheduler.running):
            return {"ok": False, "error": "engine busy: diagnostic ops need an idle engine"}
        with self._lock:
            self._paused = True
            try:
                return self._handle_impl(req)
            finally:
                self._paused = False
                self._cv.notify_all()

    def _decode_one(self, logits: torch.Tensor) -> torch.Tensor:
        """把 ``logits`` 的 argmax 喂回去，走一次解码步。

        它把诊断同样需要的图/eager 选择包了一层：关掉 capture 时模型没有图可回放，
        ``decode_step`` 会抛异常。两个入口产出的数字一样，基准因此测的是它实际
        拥有的启动方式，而不是因为缺一张图就失败。

        Args:
            logits: 来自预填充或上一步的 ``[vocab]`` logits。

        Returns:
            下一个位置的 logits；两种路径下都是同一个缓冲区。
        """
        token = int(logits.argmax())
        return self.model.decode_step(token) if self.graph else self.model.logits(token)

    def _handle_impl(self, req: dict) -> dict:
        """分派一个操作并构造它的回复。

        每个回复都带同一组计时字段（``prefill_ms``、``decode_ms``、
        ``decode_tok_s``、``cached_tokens``），探针比较各条路径时因此不用特判。
        这里不加锁：调用方已经持有。

        Args:
            req: 请求字典；它的键取决于 ``op``。

        Returns:
            回复字典；未知操作为 ``{"ok": False, "error": ...}``。
        """
        op = req.get("op", "ping")
        self.n_req += 1
        if op == "ping":
            caps = backends.caps()
            out = {"ok": True, **self.info,
                   # 报的是实际决定，而不只是模型就绪与否：capture 关掉时根本没有
                   # 图，只报 ``graph_ready`` 会被读成 capture 失败。
                   "graph": self.graph and self.model.graph_ready,
                   "max_seq": self.max_seq,
                   "load_s": round(self.load_s, 1), "build_s": round(self.build_s, 1),
                   "n_req": self.n_req, "n_reload": self.n_reload,
                   # 后端横幅原样带出：HTTP 外壳会把 ``ping`` 的结果并进
                   # ``/health``，运维分辨「平台不对」和「平台慢」需要的字段就随它
                   # 一起走。
                   "backend": caps.name, "device": str(backends.device),
                   "native_ops": len(caps.native), "total_ops": len(backends.Ops.NAMES),
                   "int8": self.int8,
                   "vram_gib": round(allocated_gib(), 2),
                   "batch": bool(self.engine is not None and self.engine.batching),
                   "decode_capacity": int(getattr(self.model, "decode_capacity", 1)),
                   # 本服务接受什么样的带图请求；OpenAI 外壳读 ``vision`` 提前回
                   # 503，而不是把一个引擎反正也会拒绝的请求转发下去
                   "vision": bool(self.vision and self.model.vision_sd),
                   "vision_int8": self.vision_int8,
                   "image_max_pixels": self.image_max_pixels,
                   "image_min_pixels": self.image_min_pixels,
                   "max_images": self.max_images,
                   "n_image_req": self.n_image_req,
                   "processor": ("loaded" if self.processor is not None
                                 else ("failed" if self._processor_failed else "lazy"))}

            if self.engine is not None:
                out["queue"] = {"waiting": len(self.engine.scheduler.waiting),
                                "running": len(self.engine.scheduler.running)}
            if self.pc is not None:
                out["prefix"] = self.pc.stats()
            elif self.engine is not None and self.engine.bm is not None \
                    and self.engine.bm.radix is not None:
                out["prefix"] = self.engine.bm.stats()
            return out
        if op == "reload":
            self._reload()
            return {"ok": True, "build_s": round(self.build_s, 1), "n_reload": self.n_reload,
                    "backend": backends.caps().name,
                    "reason": backends.backend().reason()}
        if op == "reset":
            self.engine.reset()
            return {"ok": True}
        if op == "stop":
            return {"ok": True, "stopping": True}
        if op == "generate":
            ids = req.get("tokens") or self.tok.encode(req.get("prompt", ""))
            n, window_full = self._room(len(ids), req.get("max_new"))
            temp = float(req.get("temperature", 0.0))
            deadline = self._deadline(req)
            k = int(req.get("spec_k", self.spec_k))
            spec = None
            if k and temp == 0:
                out, t_pre, t_dec = self._spec_generate(ids, n, k, deadline)
                spec = self.model.spec_stats
            else:
                out, t_pre, t_dec = self._generate(ids, n, temp, deadline)
            return {"ok": True, "tokens": out, "text": self.tok.decode(out),
                    "n_prompt": len(ids), "n_new": len(out), "max_new": n,
                    "window_full": window_full, "cached_tokens": self._cached_tokens(),
                    "prefill_ms": round(t_pre * 1000, 1), "decode_ms": round(t_dec * 1000, 1),
                    "decode_tok_s": round(len(out) / t_dec, 2) if t_dec else None,
                    "stopped": ("deadline" if deadline and time.time() >= deadline else None),
                    "spec_k": k if spec else 0, "spec": spec}
        if op == "chat":
            prefix = str(req.get("tool_prefix") or "")
            thinking_on = bool(req.get("enable_thinking", False)) and not prefix
            try:
                ids, _prompt, mm = self._render_request(req, thinking_on, prefix)
            except MMError as exc:
                return {"ok": False, "error": str(exc), "status": exc.status}
            n, window_full = self._room(len(ids), req.get("max_new"))
            temp = float(req.get("temperature", 0.0))
            deadline = self._deadline(req)
            k = int(req.get("spec_k", self.spec_k))
            spec = None
            if k and temp == 0:
                out, t_pre, t_dec = self._spec_generate(ids, n, k, deadline, mm=mm)
                spec = self.model.spec_stats
            else:
                out, t_pre, t_dec = self._generate(ids, n, temp, deadline, mm=mm)
            raw = self.tok.decode(out, skip_special_tokens=False)
            full = prefix + raw
            thinking, answer = self._split(full, thinking_on)
            calls = parse_tool_calls(answer)
            return {"ok": True, "text": answer, "thinking": thinking, "raw": full,
                    "tool_calls": [c.to_openai_dict() for c in calls], "forced_tool": bool(prefix),
                    "n_prompt": len(ids), "n_new": len(out), "max_new": n,
                    "window_full": window_full, "cached_tokens": self._cached_tokens(),
                    "prefill_ms": round(t_pre * 1000, 1),
                    "decode_tok_s": round(len(out) / t_dec, 2) if t_dec else None,
                    "stopped": ("deadline" if deadline and time.time() >= deadline else None),
                    "spec_k": k if spec else 0, "spec": spec,
                    # 图像路径实际做了什么，供探针上报
                    "n_images": mm.num_images if mm is not None else 0,
                    "image_tokens": len(mm) if mm is not None else 0}
        if op == "spec":
            # 在同一段 prompt 上把两条解码路径做 A/B。计时之前两者都先预热；投机
            # 输出拿普通的贪心输出去比，因为一个「差不多一样好」的投机循环不算解码器。
            n = int(req.get("n", 48))
            ids = self.tok.encode(req.get("prompt") or "The quick brown fox")
            k = int(req.get("spec_k", self.spec_k or 1))
            self.model.capture_verify(k + 1)
            self._generate(ids, 8, 0.0)
            plain, _, t_a = self._generate(ids, n, 0.0)
            self._spec_generate(ids, 8, k)
            spec, _, t_b = self._spec_generate(ids, n, k)
            same = sum(a == b for a, b in zip(plain, spec))
            return {"ok": True, "n": n, "k": k, "n_prompt": len(ids),
                    "plain_tok_s": round(len(plain) / t_a, 2),
                    "spec_tok_s": round(len(spec) / t_b, 2),
                    "speedup": round((len(spec) / t_b) / (len(plain) / t_a), 3),
                    "exact": f"{same}/{min(len(plain), len(spec))}",
                    "accept": self.model.spec_stats}
        if op == "bench":
            n = int(req.get("n", 30))
            ids = self.tok.encode(req.get("prompt") or "The quick brown fox")
            self.model.reset()
            logits = self.model.prefill(ids)
            for _ in range(3):  # 不计时：第一次回放要付图缓存预热的代价
                logits = self._decode_one(logits)
            sync()
            t0 = time.time()
            for _ in range(n):
                logits = self._decode_one(logits)
            sync()
            dt = time.time() - t0
            return {"ok": True, "n": n, "ms_per_step": round(dt / n * 1000, 2),
                    "decode_tok_s": round(n / dt, 2)}
        if op == "profile":
            n = int(req.get("n", 30))
            ids = self.tok.encode(req.get("prompt") or "The quick brown fox")
            self.model.reset()
            logits = self.model.prefill(ids)
            for _ in range(3):
                logits = self._decode_one(logits)
            sync()
            with torch.profiler.profile(
                    activities=profiler_activities(device=True)) as prof:
                for _ in range(n):
                    logits = self._decode_one(logits)
                sync()
            ka = prof.key_averages()
            total = sum(r.self_device_time_total for r in ka)
            rows = sorted(ka, key=lambda r: r.self_device_time_total, reverse=True)[:18]
            return {"ok": True, "total_us": round(total / n, 1),
                    "rows": [{"kernel": r.key[:60], "calls": r.count // n,
                              "us": round(r.self_device_time_total / n, 1),
                              "pct": round(100 * r.self_device_time_total / total, 1)}
                             for r in rows]}
        if op == "ops":
            # 走 eager（不经过图）的前向，每个 aten 算子才各自可见；按算子名归组
            # 并统计它的自身设备时间 -> 逐算子归因
            n = int(req.get("n", 5))
            ids = self.tok.encode(req.get("prompt") or "The quick brown fox")
            self.model.reset()
            tok = int(self.model.prefill(ids).argmax())
            with torch.inference_mode(), torch.profiler.profile(
                    activities=profiler_activities(device=True)) as prof:
                for _ in range(n):
                    self.model.logits(tok)
                sync()
            ka = prof.key_averages()
            total = sum(r.self_device_time_total for r in ka)
            rows = sorted(ka, key=lambda r: r.self_device_time_total, reverse=True)[:25]
            return {"ok": True, "n": n, "total_us": round(total / n, 1),
                    "rows": [{"op": r.key[:70], "calls": r.count // n,
                              "us": round(r.self_device_time_total / n, 1),
                              "pct": round(100 * r.self_device_time_total / total, 1)}
                             for r in rows]}
        return {"ok": False, "error": f"unknown op {op!r}"}
