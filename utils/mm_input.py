"""服务端、模型和池共享的多模态（图像）输入管线。

一个带图请求由 :class:`MMInput` 描述：视觉塔产出的图像嵌入（按 LLM 的
``<|image_pad|>`` token 在 prompt 里出现的顺序排好）、整个 prompt 的三维 (T,H,W)
mRoPE 位置，以及把 token 下标映射到 mRoPE 位置的 ``rope_delta``。

Module: ``coffin/utils/mm_input.py``。夹在视觉塔和文本模型之间：
``models/qwen35/vision.py`` 产出嵌入，``models/qwen35/model.py`` 在预填充时消费
``MMInput``，``cache/radix_cache.py`` 读 ``digest`` 来给块定键，``serving/`` 为
每个带图请求构造一个。这里只 import ``torch``，所以模型和缓存层都可以依赖它。

为什么光有 prompt 的 token id 不够：纯文本的 mRoPE 会退化成普通一维 RoPE，引擎
拿 token 下标去查一维 cos/sin 表就行，完全不用管。prompt 里一旦有图像，这件事
在两处同时不成立。

* 图像的 token *不是*线性排布的：三个轴一起推进，图像块结束于
  ``current_pos + max(h, w) // spatial_merge``，所以图像之后 token 下标和 mRoPE
  位置相差一个常数。这个常数就是 :attr:`MMInput.rope_delta`。
* 图像内部三个轴各不相同（T 不变，H/W 在合并后的网格上跑），单查一张一维表
  算不出正确的 cos/sin。

``rope_delta`` 让解码仍然可以是一张 capture 下来的图：文本 token 三个轴都前进
一，位置就是 ``token_index + rope_delta``，现成的一维表照用。只有真正含图像
token 的预填充分块才需要显式的三维位置。

缓存安全：两个 ``input_ids`` 相同的 prompt 可以嵌入*不同*的图像（占位符段从
构造上就是一样的）。:attr:`MMInput.digest` 是内容哈希，基数树拿它给块键加盐，
缓存下来的 KV 前缀因此只会被同一份图像字节复用。digest 为空表示「纯文本」。

Constraints:
    * ``embeds`` 通过 :attr:`MMInput.spans` 按 prompt 的绝对位置索引，绝不用
      图像内偏移。分块预填充可能从图像中间开始，用局部偏移会灌进错误的行。
    * ``positions`` 携带的是*整个* prompt 的 ``[3, prompt_len]``，因为一个分块
      的位置无法从该分块自己的 token 推出来。
    * ``positions`` 的第 0/1/2 行分别是 T/H/W 轴；``ops/mrope.py`` 按步长 3
      以 ``[11,11,10]`` 的方式交错，所以某一行转置或换位只会转错图像 token——
      这是一个纯文本测试抓不到的数值偏差。
    * 块 id 由 token id 和 digest 共同决定：任一项变化都会让共享该块的所有
      prompt 的缓存 KV 失效。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch

#: Qwen3.5 词表里的 ``<|image_pad|>``（对应 config 的 ``image_token_id``）。
IMAGE_TOKEN_ID = 248056
#: ``<|vision_start|>`` / ``<|vision_end|>``：对注意力和位置来说它们就是普通
#: 文本 token，只有模板把它们当分隔符用。
VISION_START_ID = 248053
VISION_END_ID = 248054


def image_digest(chunks: list[bytes]) -> str:
    """一个请求里所有图像载荷的稳定内容哈希。

    刻意哈希*编码后的图像字节*而不是像素：这是客户端发来的原样内容，不需要
    解码，而且同一张图的两份不同 PNG 会被当作不同媒体——对 KV 缓存来说这是
    保守的选择。

    Args:
        chunks: 收到的图像载荷，按 prompt 顺序排列。

    Returns:
        对载荷长度与字节算出的十六进制 SHA-256；没有图像时为 ``""``（此时
        ``MMInput`` 表示纯文本）。每个长度都先于其字节参与哈希，两个载荷因此
        无法拼成同一个哈希。
    """
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(len(chunk).to_bytes(8, "little"))
        h.update(chunk)
    return h.hexdigest()


@dataclass
class MMInput:
    """模型需要知道的、关于一个请求里图像的全部信息。

    纯文本请求在调用点上通常用 ``mm=None`` 表示，而不是一个空实例，所以下面
    的字段描述的是带图的情况。``embeds`` 和 ``positions`` 在显存里，由需要它们
    的那个预填充分块读一次；这里没有任何东西跨请求缓存。
    """

    #: ``[total_image_tokens, hidden]`` 的 bf16 张量，按 prompt 顺序。
    embeds: torch.Tensor
    #: ``[num_images, 3]`` 的 ``(t, h, w)`` patch 网格，按 prompt 顺序。
    grid_thw: torch.Tensor
    #: 整个 prompt 的 ``[3, prompt_len]`` mRoPE 位置（T、H、W 三行）。
    positions: torch.Tensor
    #: ``max(positions) + 1 - prompt_len``；纯文本 prompt 为 0。
    rope_delta: int = 0
    #: 图像载荷的内容哈希（纯文本时为 ``""``）。
    digest: str = ""
    #: 每张图像在 prompt 里的绝对 ``[start, end)`` token 区间，按 prompt 顺序。
    spans: list[tuple[int, int]] = field(default_factory=list)

    def __len__(self) -> int:
        """持有的嵌入行数，供分块把自己的切片夹到这个范围内。

        Returns:
            行数；这个请求没有解出任何图像时为 0。
        """
        return int(self.embeds.shape[0]) if self.embeds is not None else 0

    @property
    def num_images(self) -> int:
        """图像数量，从网格推出而不是单独记一份。

        Returns:
            ``(t, h, w)`` 网格的个数，也就是 ``spans`` 的条目数；纯文本请求
            为 0。
        """
        return int(self.grid_thw.shape[0]) if self.grid_thw is not None else 0

    @property
    def has_image(self) -> bool:
        """是否有 prompt token 的嵌入来自视觉塔。

        Returns:
            请求至少带一张图时为 ``True``。
        """
        return self.num_images > 0

    def slice_embeds(self, start: int, end: int) -> torch.Tensor:
        """prompt 位置 ``[start, end)``（绝对）对应的嵌入。

        Args:
            start: 分块的第一个 prompt 位置，含。
            end: 分块最后一个 prompt 位置之后一个位置。

        Returns:
            按 prompt 顺序排列的行；分块横跨两张图时会把它们拼起来。不含图像
            token 的分块返回空的 ``[0, hidden]`` 视图而不是 ``None``，调用方
            因此总能直接索引。
        """
        out = []
        for (s, e) in self.spans:
            a, b = max(s, start), min(e, end)
            if a < b:
                out.append(self.embeds[a - s:b - s])
        if not out:
            return self.embeds[:0]
        return out[0] if len(out) == 1 else torch.cat(out, dim=0)

    def slice_positions(self, start: int, end: int) -> torch.Tensor:
        """prompt 位置 ``[start, end)`` 对应的 mRoPE 位置 ``[3, end-start]``。

        Args:
            start: 分块的第一个 prompt 位置，含。
            end: 分块最后一个 prompt 位置之后一个位置。

        Returns:
            该切片的三个轴行；调用方必须先建好完整的 ``[3, prompt_len]``
            张量，因为一个切片无法从它自己的 token 重建出来。
        """
        return self.positions[:, start:end]


def find_image_spans(input_ids, image_token_id: int = IMAGE_TOKEN_ID
                     ) -> list[tuple[int, int]]:
    """``input_ids`` 里 ``image_token_id`` 的绝对 ``[start, end)`` 连续段。

    对话模板每个合并后的图像 patch 发一个占位符，所以一段就是一张图；这里算
    出来的边界正是 :class:`MMInput` 和嵌入 scatter 寻址用的，边界错一个，
    后面每张图都会整体错开一个 patch。

    Args:
        input_ids: prompt 的 token id 序列。
        image_token_id: 当作占位符的 id；可覆盖，方便别的 checkpoint 词表复用。

    Returns:
        每段一个 ``(start, end)``，按 prompt 顺序；prompt 里没有占位符 token
        时为空，也就是纯文本的情况。
    """
    spans: list[tuple[int, int]] = []
    start = -1
    for i, t in enumerate(input_ids):
        if t == image_token_id:
            if start < 0:
                start = i
        elif start >= 0:
            spans.append((start, i))
            start = -1
    if start >= 0:
        spans.append((start, len(input_ids)))
    return spans


def mrope_positions(spans: list[tuple[int, int]], grids, prompt_len: int,
                    spatial_merge: int = 2, device=None) -> tuple[torch.Tensor, int]:
    """带图像区间的 prompt 的三维 mRoPE 位置，按 prompt 顺序。

    对 batch=1、无 padding 的情况复刻 ``Qwen3_5Model.get_rope_index``：文本段让
    三个轴取同一个值（游标每 token 前进一）；图像段让 T 停在游标处、H/W 在
    *合并后*的 ``(h, w)`` 网格上跑，随后游标前进 ``max(h, w) // spatial_merge``。
    ``rope_delta`` 则是 ``max(positions) + 1 - prompt_len``。

    Args:
        spans: :func:`find_image_spans` 给出的图像区间，按 prompt 顺序。
        grids: 图像处理器给出的 ``(t, h, w)`` patch 网格，顺序与 ``spans``
            一致；张量或列表都可以。
        prompt_len: prompt 的 token 总数，不含生成的 token。
        spatial_merge: 视觉塔的 patch 合并倍数；网格除以它得到 LLM 侧的网格。
        device: 返回的位置张量放哪。``None`` 表示用当前选中后端的设备：在
            这里写死会点名某一家的技术栈，而知道请求在哪个设备上的正是调用方。

    Returns:
        ``(positions, rope_delta)``：``[3, prompt_len]`` 的 int64 位置，以及一个
        标量偏移，使解码位置为 ``token_index + rope_delta``。纯文本 prompt 的
        三行完全相同且 ``rope_delta == 0``。

    Raises:
        ValueError: 占位符段长度与合并后的网格对不上。这是模板/处理器不匹配，
            不是坏请求，所以拦在这里，而不是产出一份默默错位的位置。
    """
    import numpy as np

    # ``cur_tok`` 是 prompt 下标（也是 ``pos`` 的列号）；``cur_pos`` 是下一个
    # token 该拿到的 mRoPE 位置。图像之后两者并不相同：它每个轴消耗
    # ``max(h, w) // merge`` 个位置，却只占 ``h*w/merge^2`` 个 token 槽，所以
    # 从图像开始两者恰好差 ``rope_delta``。把 token 下标当成位置（本函数最早
    # 的写法）会让 ``rope_delta`` 静默变成 0，图像之后每个 token 都转到错误的
    # 角度上。
    pos = np.empty((3, prompt_len), dtype=np.int64)
    cur_tok = 0
    cur_pos = 0
    for gi, (start, end) in enumerate(spans):
        if start > cur_tok:                                  # 文本段
            n = start - cur_tok
            pos[:, cur_tok:start] = np.arange(n, dtype=np.int64) + cur_pos
            cur_tok, cur_pos = start, cur_pos + n
        grid = grids[gi].tolist() if hasattr(grids[gi], "tolist") else grids[gi]
        _t, gh, gw = (int(x) for x in grid)
        llm_h, llm_w = gh // spatial_merge, gw // spatial_merge
        n_img = end - start
        if n_img != llm_h * llm_w * max(_t, 1):
            raise ValueError(
                f"image {gi}: {n_img} placeholder tokens but grid {grid} implies "
                f"{llm_h * llm_w * max(_t, 1)}")
        t_row = np.full((llm_h * llm_w,), cur_pos, dtype=np.int64)
        h_row = np.repeat(np.arange(llm_h, dtype=np.int64), llm_w) + cur_pos
        w_row = np.tile(np.arange(llm_w, dtype=np.int64), llm_h) + cur_pos
        pos[0, start:end] = t_row
        pos[1, start:end] = h_row
        pos[2, start:end] = w_row
        cur_tok = end
        cur_pos = int(max(t_row.max(), h_row.max(), w_row.max())) + 1
    if cur_tok < prompt_len:                                 # 尾部文本
        n = prompt_len - cur_tok
        pos[:, cur_tok:] = np.arange(n, dtype=np.int64) + cur_pos
    phi = int(pos.max()) + 1 - prompt_len
    if device is None:
        import backends

        device = backends.device
    return torch.from_numpy(pos).to(device), phi
