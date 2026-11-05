"""把 ``models.qwen4exp`` 的参考实现接成引擎能调度的模型类。

``CausalLM`` 那一面只有「整段前向 + 一份 ``ModelStates``」，而引擎按「预填充一段、
再一格一格解码」驱动模型，还要读 ``pos1`` 来记账。这个文件就是那条缝：它不实现任何
数值，只把状态、位置和一个 token 的搬运接到 ``engine_api`` 的 ``Model`` 契约上。

Module: ``coffin/models/qwen4exp``。族内最上层，也是 ``engine_api`` 注册的那个类。
它 import 同族的 ``config`` / ``model`` / ``weights`` / ``checkpoint`` 和
``backends.device``；同族其余模块不 import 它，族以上的东西一律不许进来。

Constraints:
    * 平台硬约束：**不用设备图**（``platform/scow/AGENTS.md`` 第 1 条）。所以
      ``capture()`` 是空实现、``graph_ready`` 恒 ``False``，``decode_step`` 与
      ``logits`` 是同一条 eager 前向。契约里「没有图就抛异常」和「图把缓冲区地址
      烘进去」这两条在这里都不适用，硬编一张图只会得到回放过期输出的静默错误答案。
    * ``pos1`` 是位置的**唯一权威**，不是缓存：``PrefixCacheEngine`` 会直接写它来把
      窗口移到命中边界上（``model.pos1.fill_(matched)``）。所以每个入口都先把它读回
      主机侧的 ``_pos``，任何「自己在家记账」的写法都会在那条路径上错位。
    * 状态整份活在 ``ModelStates`` 里，和 KV 同行。GDN 的递推/卷积、PLE 的 n-gram
      上下文/短卷积、QSA 的 KV 与 indexer 池、全位置表是**一个**对象，拆开搬任何
      一部分都会把共享的 KV 配上一个从没见过它的状态。
    * 单序列、单行：``prefill_row == 0``、``decode_capacity == 1``。批处理与分页的
      入口一律抛 ``NotImplementedError`` 并说明下一步，绝不提供一个能跑但算错的
      实现。
"""

from __future__ import annotations

import torch

from backends import device as _device
from models.qwen4exp.checkpoint import StreamingSource
from models.qwen4exp.config import TextConfig
from models.qwen4exp.model import CausalLM, ModelStates, causal_mask
from models.qwen4exp.weights import load_into, read_config


def load_weights(model_dir, quant: bool = False, weight_dtype=None,
                 vision: bool | None = None, vision_quant: bool = False) -> dict:
    """把 checkpoint 打开成引擎常驻的加载凭据。

    qwen4exp 常驻的**不是**张量：参考 checkpoint 是 360 GB（专家 246 GB、n-gram 表
    102 GB），任何单机内存都装不下，所以这一族走的是 ``weights.load_into`` 的流式
    装载。这个字典里装的是「从哪读」和「按什么几何读」，``create_model`` 把它原样
    交给 :class:`Qwen4Engine`，重建时因此只需要重扫一遍分片 header，而不是再读一遍
    360 GB。

    Args:
        model_dir: checkpoint 目录。
        quant: 是否期望 W4A8 量化产物。本 lane 不接量化装载，所以 ``True`` 直接抛
            ``NotImplementedError``——静默按 bf16 读会先花几十分钟，然后在
            ``Tried to allocate`` 上失败，而那个错看不出根因是「量化没接上」。
        weight_dtype: 稠密张量的计算 dtype。本族的装载器按目标参数自己的 dtype
            落盘，所以这里只接受 ``None``。
        vision: 是否加载视觉塔。本 lane 只做文本，``True`` 抛
            ``NotImplementedError``；``False``/``None`` 表示不加载。
        vision_quant: 视觉塔的 int8 开关，本 lane 不加载视觉塔，忽略。

    Returns:
        ``{"source": StreamingSource, "config": dict, "model_dir": str}``。
        ``source`` 只持有分片 header 索引，一个张量都没读。

    Raises:
        NotImplementedError: 要求量化权重、要求视觉塔，或者要求了一个非 ``None``
            的 ``weight_dtype``。
        ImportError: 环境里没有 ``safetensors``（本机开发机就是这样），由
            ``checkpoint.build_index`` 抛出。
        FileNotFoundError: 目录里一个 safetensors 分片都没有。
    """
    if quant:
        raise NotImplementedError(
            "qwen4exp 的 W4A8 装载还没接：量化产物由 proj.set_quantized 供货，"
            "走的是量化那条路（M0/M2），本 lane 只有 bf16 流式装载。"
            "真机部署请先用 --no-quant 跑通，再等量化产物接上。")
    if vision:
        raise NotImplementedError(
            "本 lane 只接文本塔：视觉塔要 enable_vision + image_embeds，"
            "多模态不在 Q4EXP-SERVING-PLAN 的 M1 范围内（那里明确不做视觉）。")
    if weight_dtype is not None:
        raise NotImplementedError(
            "本族的流式装载按目标参数自己的 dtype 落盘，不提供全局 dtype 覆盖；"
            f"收到 weight_dtype={weight_dtype!r}。")
    return {"source": StreamingSource(model_dir), "config": read_config(model_dir),
            "model_dir": str(model_dir)}


class Qwen4Engine:
    """qwen4exp 的单序列、eager、不分页引擎实现。

    一次只服务一条序列：位置由 ``pos1`` 记账，状态整份放在 :class:`ModelStates` 里，
    解码一格就把这份状态往前推一格。批处理、分页与前缀缓存的入口都存在，但一律抛
    ``NotImplementedError``——它们要等 KV 池化和状态快照（M3），现在给出能跑的假实现
    只会产出「形状全对、数值错」的结果。

    Attributes:
        max_seq: 上下文窗口，预填充越过它就抛 ``ValueError``。
        pos1: 长度为 1 的设备张量，装着当前位置；外部可以直接写它来移动窗口。
        pos: 一维位置张量，``pos1`` 是它第一个元素的视图。分开两样是为了对齐
            ``model_runner``：分块路径按行下标写 ``model.pos[row]``，单流路径只用
            ``pos1``。
        verify_c: 验证图宽度。本族没有验证图，恒为 ``0``。
        mtp: 草稿模型。本族这条路上没有，恒为 ``None``。
        prefill_row: 单流前向写的那一行，恒为 ``0``。
        decode_capacity: 解码行数，恒为 ``1``（单序列）。
        vision: 永不接上的视觉塔，恒为 ``None``。
        vision_sd: 视觉塔张量。恒为空：服务层的 ``_build`` 会无条件读这两个属性。
        load_report: 从 checkpoint 建时装载器的账目；注入模型时为 ``None``。
    """

    def __init__(self, model_dir_or_model, max_seq: int = 4096, bench: bool = True,
                 weights: dict | None = None, paged: bool = False,
                 quant: bool = False, stream_ngram: bool | None = None,
                 ple_tables: dict[int, object] | None = None) -> None:
        """接上一个已经建好的 ``CausalLM``，或者从一个 checkpoint 目录建一个。

        Args:
            model_dir_or_model: checkpoint 目录（真机），或者一个已经建好的
                :class:`CausalLM`（测试直接注入，省掉 360 GB 的加载）。
            max_seq: 引擎会跑的最长序列。预填充和每次解码都按它做上界检查。
            bench: 本族没有内核变体可调（参考实现全是朴素 PyTorch），这个开关只
                为满足 ``engine_api.create_model`` 的调用签名而存在。
            weights: :func:`load_weights` 的返回值。给了就不再去磁盘扫分片；``None``
                时由本构造器自己调一次。
            paged: 分页 KV 变体。本 lane 没有池，``True`` 直接抛
                ``NotImplementedError``。
            quant: 只在 ``weights`` 为 ``None``、也就是本次调用自己去加载时生效，
                与 ``engine_api.create_model`` 的约定一致；给了 ``weights`` 时忽略。
            stream_ngram: PLE 的 n-gram 表是否流式。``None`` 表示按 checkpoint 决定：
                有 PLE 层就流式（真实表 95 GiB，分配出来必然 OOM），没有就普通建表。
            ple_tables: PLE 的 n-gram 表，按层号索引，交给 ``CausalLM.forward`` 的
                ``ple_tables``。只在 ``stream_ngram`` 为真时需要；那张表由 PLE 那条路
                （M4）从 checkpoint 分片 mmap 提供。

        Raises:
            NotImplementedError: ``paged`` 为真、checkpoint 绑定了词表嵌入、PLE 需要
                流式表但没给 ``ple_tables``，或者请求的量化/视觉装载还没接。
            ValueError: ``max_seq`` 非正，或者注入的模型没有参数（拿不到设备）。
        """
        if int(max_seq) <= 0:
            raise ValueError(f"max_seq 必须是正数，收到 {max_seq}")
        if paged:
            raise NotImplementedError(
                "qwen4exp 还没有分页 KV 池：QSA 的 KV 与 indexer 池现在跟着 "
                "ModelStates 一起按 torch.cat 增长（qsa.AttentionCache）。"
                "下一步是把两个池改成 [blocks, block_size, ...] 并按 block_table "
                "寻址（Q4EXP-SERVING-PLAN §3.2），那之前 enable_paging 无池可挂。")
        self.max_seq = int(max_seq)
        self.prefill_row = 0
        self.decode_capacity = 1
        self.batch_capacity = 1
        self.verify_c = 0
        self.mtp = None
        self.vision = None
        self.vision_sd: dict = {}
        self._ple_tables = ple_tables
        self.load_report: dict | None = None

        if isinstance(model_dir_or_model, torch.nn.Module):
            # 注入路径：模型归调用方，设备与 train/eval 状态都由它定，这里不改。
            self.model = model_dir_or_model
        else:
            self.model = self._build_from_checkpoint(model_dir_or_model, weights,
                                                    quant, stream_ngram)

        # 位置缓冲区的设备跟着模型走：注入的模型可能在任何设备上（测试里是 CPU）。
        try:
            device = next(self.model.parameters()).device
        except StopIteration:                       # pragma: no cover - 零参数配置不存在
            raise ValueError("注入的模型没有参数，推不出它所在的设备") from None
        self._states = None
        self._pos = 0
        self.pos = torch.zeros(1, dtype=torch.long, device=device)
        self.pos1 = self.pos[:1]

    def _build_from_checkpoint(self, model_dir, weights: dict | None, quant: bool,
                               stream_ngram: bool | None) -> CausalLM:
        """按 checkpoint 的配置建模型，再用 ``weights.plan`` / ``load_into`` 装权重。

        装载一律走 :func:`models.qwen4exp.weights.load_into`，本方法只决定「装哪些」：
        这就是「不重写加载逻辑」的意思——映射表、专家切片、n-gram 分片拼接全在那边。

        Args:
            model_dir: checkpoint 目录。
            weights: :func:`load_weights` 的返回值；``None`` 时自己调一次。
            quant: 透传给 :func:`load_weights`；给了 ``weights`` 时不起作用。
            stream_ngram: 见构造器。

        Returns:
            装好权重、放在引擎设备上的模型。

        Raises:
            NotImplementedError: checkpoint 绑定词表嵌入（``tie_word_embeddings``），
                或者 PLE 需要流式 n-gram 表却没有 ``ple_tables``。
            KeyError: 源里缺模型要的张量，或计划里出现了不是参数的键。
            ValueError: 某个张量形状不符。
        """
        if weights is None:
            weights = load_weights(model_dir, quant=quant)
        raw = weights.get("config") or read_config(model_dir)
        cfg = TextConfig.from_dict(raw.get("text_config", raw))
        if cfg.tie_word_embeddings:
            # ``CausalLM`` 永远建一个独立的 lm_head，没有绑定分支；装作支持会让
            # 1.27 GB 的 lm_head 停在 ``normal_(0.02)`` 的随机值上，logits 看着
            # 像模像样却与 checkpoint 无关。
            raise NotImplementedError(
                "这个 checkpoint 把 lm_head 绑在输入嵌入上，而 CausalLM 没有绑定分支："
                "这样建出来的 lm_head 会是一份随机初始化矩阵，形状全对、答案全错。")
        stream = bool(cfg.ple_layer_ids) if stream_ngram is None else bool(stream_ngram)
        if stream and not self._ple_tables:
            raise NotImplementedError(
                "PLE 的 n-gram 表是 102.4 GB，必须流式（stream_ngram=True）；"
                "这张表由 PLE 那条路（M4）从 checkpoint 分片 mmap 之后按层传进来，"
                "本 lane 不提供。要么传 ple_tables={层号: HostNGramTable}"
                "（表本身由 weights.load_host_ngram_table 建），"
                "要么用一个 ple_layer_ids 为空的配置。")
        model = CausalLM(cfg, stream_ngram=stream)
        model.to(_device)
        self.load_report = dict(load_into(model, weights["source"], cfg))
        return model.eval()

    def _live(self) -> CausalLM:
        """返回活着的模型；``close()`` 之后再用它就是编程错误。"""
        if self.model is None:
            raise RuntimeError("这个引擎已经 close()，请重建一个")
        return self.model

    def _require_row(self, row: int) -> None:
        """确认 ``row`` 是单序列引擎唯一的那一行。"""
        if int(row) != 0:
            raise NotImplementedError(
                f"这个引擎只有第 0 行，收到 row={row}；"
                "按行的状态快照与搬移属于批处理路径（enable_batching）。")

    def _sync_pos(self) -> int:
        """把 ``pos1`` 读回主机侧的位置。

        ``pos1`` 是权威而不是缓存：``PrefixCacheEngine`` 在命中前缀后直接
        ``model.pos1.fill_(matched)`` 把窗口移过去，而那是设备张量，主机侧的记账
        看不到它。每个入口先读一次，外部写入因此立即生效。
        """
        self._pos = int(self.pos1)
        return self._pos

    def _forward(self, ids: torch.Tensor, past: int,
                 keep_all: bool = True) -> tuple[torch.Tensor, ModelStates]:
        """在 ``past`` 之后跑这段 token，返回 logits 与更新后的状态。

        掩码与位置都在这里现造：QSA 要的是 4 维 bool 因果掩码，GDN/PLE 要的是
        ``[B, S]`` 的 keep 标记，两者绝不能互换（前者按头轴广播，后者按通道轴）。
        ``position_ids`` 显式给成 ``arange(past, past + n)`` 的三行 mRoPE 表，与
        ``CausalLM.forward`` 在 ``position_ids=None`` 时自己推的那份逐元素相同，
        所以两条路的结果可以直接逐位比较。

        Args:
            ids: ``[1, n]`` 的 token id。
            past: 这些 token 的第一个的绝对位置，同时是 KV 槽位和掩码的列偏移。
            keep_all: 算整段 logits，还是只算最后一个位置。中间分块用 ``False``：
                它们的 logits 没有任何消费者，而一次 ``[n, vocab]`` 的 lm_head 在
                248k 词表上是百 MB 量级。**需要跟整段前向逐位比对的调用必须传
                ``True``**——``logits_to_keep=1`` 会让 lm_head 走一次 M=1 的 GEMM，
                末位 logits 与整段前向差在 fp32 的末位。

        Returns:
            ``(logits, states)``。``logits`` 是 ``[1, n 或 1, vocab]``，``states``
            是整份新状态（KV、线性递推、PLE、全位置表都在里面）。
        """
        n = int(ids.shape[1])
        device = ids.device
        mask = causal_mask(1, n, past).to(device)
        conv_mask = torch.ones(1, n, dtype=torch.bool, device=device)
        positions = torch.arange(past, past + n, dtype=torch.long,
                                 device=device).view(1, 1, -1).expand(3, 1, -1)
        return self._live()(input_ids=ids, attention_mask=mask,
                            conv_mask=conv_mask, position_ids=positions,
                            states=self._states, ple_tables=self._ple_tables,
                            logits_to_keep=0 if keep_all else 1)

    @torch.no_grad()
    def prefill(self, tokens: list[int], chunk: int = 2048, spec_row: int | None = None,
                mm=None) -> torch.Tensor:
        """处理一段 prompt，返回最后一个位置的 logits。

        分块推进：每块一次前向，状态在块之间接着走，所以分块预填充分块与一次跑完
        在语义上等价（数值上会有 GEMM M 不同带来的末位差）。位置从 ``pos1`` 开始，
        因此接着上一次调用停下的地方继续。

        Args:
            tokens: 整个 prompt，按顺序。
            chunk: 一次前向吃多少 token，也就是 GEMM 的 M。参考实现没有内核可调，
                这个数只影响激活形状；``>= len(tokens)`` 时整段一次跑完。
            spec_row: 批处理投机解码用哪一行的分页草稿器 KV。本 lane 没有 MTP 头
                也没有草稿器，``None`` 之外的取值一律拒绝。
            mm: 图像载荷。本 lane 只做文本，非 ``None`` 直接拒绝——把它当纯文本
                预填充会得到一堆没人看得懂的输出，而不是一个错误。

        Returns:
            最后一个 prompt token 的 ``[vocab]`` logits，是活缓冲区的视图，下一次
            前向会覆盖它。

        Raises:
            ValueError: prompt 为空，或者放不进当前位置与 ``max_seq`` 之间的区间。
            NotImplementedError: 收到 ``mm`` 或非 ``None`` 的 ``spec_row``。
        """
        if mm is not None:
            raise NotImplementedError(
                "本族这条路上没有视觉塔：带图 prompt 要走 enable_vision + "
                "image_embeds，而那属于多模态，不在 M1 范围内。")
        if spec_row is not None:
            raise NotImplementedError(
                "spec_row 是批处理投机解码的行号；本 lane 没有 MTP 头也没有草稿器 KV。")
        n = len(tokens)
        if n == 0:
            raise ValueError("prompt 为空")
        pos0 = self._sync_pos()
        if n > self.max_seq - pos0:
            raise ValueError(f"{n} 个 token 放不进 max_seq={self.max_seq}"
                             f"（当前位置 {pos0}）")
        step = max(int(chunk), 1)
        logits = None
        for c0 in range(0, n, step):
            c1 = min(c0 + step, n)
            ids = torch.tensor([tokens[c0:c1]], dtype=torch.long,
                               device=self.pos1.device)
            logits, self._states = self._forward(ids, self._pos, keep_all=(c1 == n))
            self._pos += c1 - c0
        self.pos1.fill_(self._pos)
        return logits[0, -1]

    @torch.no_grad()
    def logits(self, token: int) -> torch.Tensor:
        """让一个 token 走一遍 eager 前向，返回它的 logits。

        这是 ``CausalLM.forward`` 在 ``past=pos1`` 上的单 token 形式，也是
        ``engine_api`` 契约里那条「不做 capture 的单步」。本平台不许用图，所以
        它与 :meth:`decode_step` 是同一条路径。

        Args:
            token: 要追加的 token id；它写在 ``pos1`` 处，位置随后加一。

        Returns:
            该位置的 ``[vocab]`` logits，是活缓冲区的视图。

        Raises:
            ValueError: 位置已经到 ``max_seq``。
        """
        pos = self._sync_pos()
        if pos >= self.max_seq:
            raise ValueError(f"位置 {pos} 已经到 max_seq={self.max_seq} 之外")
        ids = torch.tensor([[int(token)]], dtype=torch.long, device=self.pos1.device)
        out, self._states = self._forward(ids, pos)
        self._pos = pos + 1
        self.pos1.fill_(self._pos)
        return out[0, -1]

    def decode_step(self, token: int) -> torch.Tensor:
        """推进一个 token。

        ``engine_api`` 里这个入口是「重放 capture 好的解码图」。本平台不用设备图，
        所以它和 :meth:`logits` 是同一次 eager 前向——两个入口产出的数字因此按定义
        相同，``model_runner._forward_token`` 走哪个分支都不影响结果。

        Args:
            token: 要追加的 token id。

        Returns:
            ``token`` 之后那个位置的 ``[vocab]`` logits。

        Raises:
            ValueError: 位置已经到 ``max_seq``。
        """
        return self.logits(token)

    def reset(self) -> None:
        """把状态与位置倒回空上下文。

        状态整份丢掉（``ModelStates`` 是 None），所以 KV、GDN 递推/卷积、PLE 的
        n-gram 上下文与短卷积窗口、全位置表会一起消失。只清其中一部分会让下一个
        请求接到上一个请求的尾巴上，而那种错误看起来像模型不确定。
        """
        self._states = None
        self._pos = 0
        self.pos1.zero_()

    def capture(self, warmup: int = 3) -> None:
        """空实现：这条路径上没有图可捕获。

        平台硬约束（``platform/scow/AGENTS.md`` 第 1 条）禁止设备图，所以
        ``caps.graph`` 在这颗卡上必须是假，引擎的 eager 分支根本不会调这里；它存在
        只是为了让 ``LLMEngine.build`` 那条按 ``graph_ready`` 判断的分支不至于
        ``AttributeError``。``warmup`` 因此也没有工作可做。
        """
        return None

    @property
    def graph_ready(self) -> bool:
        """恒为 ``False``：本族的解码走 eager，没有可回放的图。"""
        return False

    def enable_batching(self, capacity: int) -> None:
        """接受 ``capacity == 1``，其余一律拒绝。

        Args:
            capacity: 要分配的行数。单流引擎恒为 1（一行就是预填充那一行）。

        Raises:
            ValueError: ``capacity < 1``。
            NotImplementedError: ``capacity > 1``。批解码要逐行的 KV 块表、哨兵行
                和一次多行前向，本 lane 只有一行状态。
        """
        capacity = int(capacity)
        if capacity < 1:
            raise ValueError(f"批容量必须 >= 1，收到 {capacity}")
        if capacity > 1:
            raise NotImplementedError(
                f"qwen4exp 还没有批解码：要 {capacity} 行意味着逐行 KV 块表、哨兵行"
                "和一次多行前向，而这条路的 KV 仍是稠密的 torch.cat 缓存"
                "（qsa.AttentionCache）。先做分页池（M3），批处理再谈。")

    def enable_paging(self, num_blocks: int = 512, block_size: int = 64):
        """分页池的挂载点；本 lane 还没有池。

        Args:
            num_blocks: 池里的块数。
            block_size: 每块的 token 数。

        Raises:
            NotImplementedError: 无条件。QSA 的 KV 与 indexer 池现在按位置
                ``torch.cat`` 增长，没有 ``[blocks, block_size, ...]`` 的池可挂；
                伪造一个池会让所有注意力读错地方而形状合法。
        """
        raise NotImplementedError(
            "qwen4exp 还没有分页 KV 池（qsa.AttentionCache 仍是 dense）。下一步："
            "把 KV 与 indexer keys 改成 [blocks, block_size, ...] 池、按 block_table "
            "寻址，见 Q4EXP-SERVING-PLAN §3.2；在那之前前缀缓存与批处理都没有落点。")

    def set_rows(self, tokens, positions, slot_rows=None, row_indices=None) -> None:
        """批解码的铺行入口；本 lane 只有一行。

        Args:
            tokens: 逐行 token id。
            positions: 逐行绝对位置。
            slot_rows: 逐行块表。
            row_indices: 每个条目对应的解码行。

        Raises:
            NotImplementedError: 无条件；见 :meth:`enable_batching`。
        """
        raise NotImplementedError(
            "set_rows() 是批解码的铺行入口；本引擎只有第 0 行、没有块表，"
            "先做 enable_batching（M3 之后）。")

    def decode_batch_eager(self, rows: int, bucket: int, pad_rows=()) -> torch.Tensor:
        """多行 eager 批解码；本 lane 没有多行。

        Args:
            rows: 活行数。
            bucket: 本次跑的行数（桶宽）。
            pad_rows: 桶内部的空行。

        Raises:
            NotImplementedError: 无条件；见 :meth:`enable_batching`。
        """
        raise NotImplementedError(
            "decode_batch_eager() 要逐行的 logits 缓冲区和哨兵行；本引擎只有一行，"
            "先做 enable_batching（M3 之后）。")

    def capture_batch(self, buckets: tuple[int, ...] = (1, 2, 4, 8),
                      warmup: int = 3) -> None:
        """每桶一张批解码图；本平台不用图，本 lane 也没有多行。

        Args:
            buckets: 要捕获的桶宽。
            warmup: 捕获前的即时执行轮数。

        Raises:
            NotImplementedError: 无条件；见 :meth:`enable_batching`。它在这里而不是
                在 ``llm_engine.build`` 里失败，是因为 ``enable_batching`` 会先拒绝。
        """
        raise NotImplementedError(
            "capture_batch() 是批解码的图捕获入口；本平台禁止设备图，"
            "本引擎也只有一行，先做 enable_batching（M3 之后）。")

    def decode_batch(self, rows: int, bucket: int, pad_rows=()) -> torch.Tensor:
        """重放某张批解码桶图；本 lane 没有图也没有多行。

        Args:
            rows: 活行数。
            bucket: 要重放的桶宽。
            pad_rows: 桶内部的空行。

        Raises:
            NotImplementedError: 无条件；见 :meth:`enable_batching`。
        """
        raise NotImplementedError(
            "decode_batch() 要回放一张捕获好的桶图；本平台禁止设备图，"
            "本引擎也只有一行，先做 enable_batching（M3 之后）。")

    def prefill_group(self, seqs: list, snapshot_interval: int | None = None):
        """一次前向预填充多个 prompt；本 lane 只有一条序列。

        Args:
            seqs: 序列对象列表。
            snapshot_interval: 状态快照的间隔。

        Raises:
            NotImplementedError: 无条件。分组预填充需要逐行块表与可复用的状态快照，
                本 lane 两者都没有。
        """
        raise NotImplementedError(
            "分组预填充要逐行的块表（pool.assign_table）和块边界的循环状态快照；"
            "本 lane 既没有分页池也没有快照，先做 M3。")

    def snapshot_linear_states(self, row: int = 0) -> dict[int, tuple[torch.Tensor,
                                                                     torch.Tensor]]:
        """克隆当前状态下每个线性注意力层的 (递推状态, 卷积状态)。

        Args:
            row: 要打快照的行；单序列引擎只有第 0 行。

        Returns:
            ``{层下标: (recurrent_state, conv_state)}``，只含 GDN 层，克隆脱离
            计算图。模型里没有 GDN 层（全 QSA 配置）时是空字典。还没有跑过前向时
            也是空字典——空上下文没有可快照的东西。

        Raises:
            NotImplementedError: ``row != 0``，或者模型带 PLE 层。``engine_api``
                给的是两元组，装不下 ``ple_context`` / ``ple_conv_state``，而只拷
                一半会让前缀复用拿一个从没见过这些 n-gram 历史的 PLE 状态继续——
                形状全对、数值错。
        """
        self._require_row(row)
        states = self._states
        if states is None:
            return {}
        layers = self._live().model.layers
        if any(layer.ple is not None for layer in layers):
            raise NotImplementedError(
                "qwen4exp 的 PLE 状态（ple_context / ple_conv_state）装不进 "
                "engine_api.PagedModel 的 {层: (循环状态, conv 状态)} 两元组；"
                "只拷 GDN 那两份会静默丢掉 n-gram 历史。等状态快照扩成整份 "
                "ModelStates（M3）再谈前缀复用。")
        out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for index, layer in enumerate(layers):
            layer_state = states.layers[index]
            if layer_state.recurrent_state is None:
                continue
            out[index] = (layer_state.recurrent_state.clone(),
                          layer_state.conv_state.clone())
        return out

    def restore_linear_states(self, states, row: int = 0) -> None:
        """把一份快照装进当前状态里对应的线性注意力层。

        只还原 GDN 那两份递推状态：KV、indexer 池和全位置表由调用方负责，而这个
        方法存在的场合（前缀复用）在本 lane 还不可达——那条路上的 ``model.prefill``
        会先撞上 ``enable_paging`` 的拒绝。

        Args:
            states: :meth:`snapshot_linear_states` 的返回值，键是本模型的层下标。
            row: 要装入状态的行；单序列引擎只有第 0 行。

        Raises:
            NotImplementedError: ``row != 0``。
            RuntimeError: 还没有任何状态可以装入（没跑过前向）。
            KeyError: ``states`` 里出现了本模型没有的层下标——来自另一套架构的快照。
                宁可拒绝，也不要恢复一份残缺状态。
        """
        self._require_row(row)
        if self._states is None:
            raise RuntimeError(
                "restore_linear_states() 需要先有一次前向产出的 ModelStates："
                "空上下文没有任何层状态可以接收这份快照。")
        layers = self._states.layers
        for index, (recurrent, conv) in states.items():
            if index < 0 or index >= len(layers):
                raise KeyError(f"本模型没有第 {index} 层（共 {len(layers)} 层）；"
                               "这份快照来自另一套架构")
            if layers[index].recurrent_state is None:
                raise KeyError(f"第 {index} 层是 QSA 层，没有线性注意力状态；"
                               "这份快照来自另一套架构")
            layers[index].recurrent_state.copy_(recurrent)
            layers[index].conv_state.copy_(conv)

    def move_linear_states(self, src_row: int, dst_row: int) -> None:
        """把一行状态搬到另一行。

        Args:
            src_row: 源行。
            dst_row: 目标行。两者相等时是空操作，也就是单流的情形。

        Raises:
            NotImplementedError: 两行不同。搬移是批处理把暂存行的状态交给解码行
                用的手段，单序列引擎没有第二行。
        """
        if int(src_row) == int(dst_row):
            return
        raise NotImplementedError(
            f"move_linear_states({src_row}, {dst_row}): 本引擎只有第 0 行，"
            "搬移属于批处理路径（enable_batching）。")

    def zero_linear_states(self, row: int) -> None:
        """把 ``row`` 那一行交给新请求：整份状态丢掉。

        比契约字面上的「清零」更彻底——单序列引擎里这一行就是全部状态，丢掉它连同
        KV、indexer 池和全位置表一起回到空上下文，正是新请求需要的起点。只把递推
        状态清零而留着 KV，会让新请求读到上一个 prompt 的前缀。

        Args:
            row: 要交出去的行；单序列引擎只有第 0 行。

        Raises:
            NotImplementedError: ``row != 0``。
        """
        self._require_row(row)
        self._states = None

    def enable_spec_batch(self, K: int, buckets: tuple[int, ...] = (1, 2, 4, 8),
                          **kwargs) -> None:
        """批友好的投机解码开关；本族这条路上没有 MTP 头。

        Args:
            K: 每个投机步的草稿数。
            buckets: 要为哪些批宽度捕获校验图。
            **kwargs: ``block_size`` / ``max_drafter_len`` / ``sparse_topk`` /
                ``verify_mode``；本入口不接投机，收下来只为把调用点原样表达出来。

        Raises:
            NotImplementedError: 无条件。它需要批处理与 MTP 头，而两者本 lane 都没有；
                ``model_runner.enable_spec`` 会先撞上 ``enable_batching`` 的拒绝。
        """
        raise NotImplementedError(
            "enable_spec_batch() 要批处理、分页池和 MTP 草稿头，本 lane 一个都没有；"
            "先看 enable_batching（M3 之后），投机另归 spec 那条路。")

    def capture_verify(self, c: int = 4, warmup: int = 3) -> None:
        """验证图宽度 ``c`` 的捕获入口；本族没有可捕获的验证图。

        Args:
            c: 验证宽度（草稿数加锚 token）。
            warmup: 捕获前的即时执行轮数。

        Raises:
            NotImplementedError: 无条件。验证步原本就是一次图重放，没有图就没有
                这一步；投机解码要走 MTP，那是另一条路。
        """
        raise NotImplementedError(
            "qwen4exp 的投机验证需要 MTP 头 + 一张捕获好的验证图；本平台不用设备图，"
            "所以这条入口不可用（见 Q4EXP-SERVING-PLAN：M1 只要求单序列）。")

    def spec_step(self, logits_prev: torch.Tensor, K: int):
        """单流投机步；本族这条路上没有 MTP 头。

        Args:
            logits_prev: 主模型对最后一个已提交 token 的 logits。
            K: 草稿数。

        Raises:
            NotImplementedError: 无条件。
        """
        raise NotImplementedError(
            "本引擎没有 MTP 草稿头，也没有可重放的验证图；投机解码不可用。")

    def spec_generate(self, prompt: list[int], max_new: int, K: int = 3,
                      eos: set[int] | None = None) -> list[int]:
        """贪心投机解码；本族这条路上不可用。

        ``CausalLM.spec_generate`` 是参考实现自己的单请求入口，返回
        ``(tokens, stats)``，形状与 ``engine_api.Model`` 的 ``list[int]`` 不同，直接
        转发会让调用方拿到一个元组当 token 列表用。要接它得先有 MTP 头。

        Args:
            prompt: prompt token。
            max_new: 生成上限。
            K: 每步草稿数。
            eos: 停止 token。

        Raises:
            NotImplementedError: 无条件。
        """
        raise NotImplementedError(
            "本族这条路上没有 MTP 头，spec_generate 不可用；参考实现的 "
            "CausalLM.spec_generate 返回 (tokens, stats)，与 Model 契约的 list[int] "
            "不同形，不做透明转发。")

    @property
    def spec_stats(self) -> dict:
        """投机统计。没有投机路径，因此是空字典。

        返回空字典而不是抛异常：日志与 ``ping`` 会无条件读这个属性，一条「没有统计」
        不该让健康检查失败。
        """
        return {}

    def close(self) -> None:
        """丢掉状态和模型的引用，让 reload 能重建。

        先丢 :class:`ModelStates` 再丢模型：那份状态里持有 KV、indexer 池和全位置
        表，加起来是这个模型占用的大头之一。
        """
        self._states = None
        self.model = None


__all__ = ["Qwen4Engine", "load_weights"]
