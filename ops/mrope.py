"""三维 mRoPE：给带图像 prompt 用的旋转表。

纯文本 mRoPE 退化成普通一维 RoPE，所以本引擎的每条解码路径都用 token 下标
去查 ``[max_seq, rotary_dim]`` 的一维表（``ops/rope.py``）。带图像的 prompt
从两方面打破这个前提：

* 图像那些 token 排在二维 (H, W) 的合并网格里，于是三个位置轴 (T, H, W)
  *在图像内部*各不相同——一次查表表达不了；
* 图像之后，图像块结束于 ``current_pos + max(h, w) / merge``，而实际只消耗了
  ``h*w/merge**2`` 个 token，所以此后 mRoPE 位置是 ``token_index + rope_delta``，
  ``rope_delta`` 是一个恒定的负数（:attr:`mm_input.MMInput.rope_delta`）。

所以这里只有两件事：

* :func:`mrope_cos_sin`——交错的三轴合成（也就是 checkpoint 里
  ``mrope_interleaved=true`` 的含义，也是 ``Qwen3_5TextRotaryEmbedding``
  唯一实现的那种布局）。一半频率来自 T，然后 H 和 W 交替插入，于是
  ``i % 3 == 0 -> T``、``1 -> H``、``2 -> W``。``cos`` 最终仍是
  ``cat(half, half)``——和一维表*相同的*布局，所以 ``rotate_half`` 把 ``i``
  与 ``i + rotary_dim/2`` 配对，纯文本 prompt 得到的行与
  ``ops.rope.make_rope_tables`` 逐位相同；
* :func:`scatter_image_embeds`——纯张量拼接，把视觉塔的嵌入按*绝对* prompt
  下标写进一个预填充分块的 token 嵌入里。

位置本身（``[3, prompt_len]`` 加 ``rope_delta``）来自集成面
:func:`mm_input.mrope_positions`；``tests/test_mrope.py`` 拿它和参考实现的
``get_rope_index`` 逐元素比对，本模块因此可以只做「给位置、出角度」这一层。

解码永远不经过这里：预填充之后三个轴同步前进，``pos + rope_delta`` 直接查
一维表，capture 过的图也保持 capture 状态。

Module: ``coffin/ops``。依赖 ``ops/rope`` 的共用频率轴；当某个预填充分块
携带图像嵌入时被 ``models/qwen35/model.py`` 调用。

Constraints:
    * 步长 3 的交错和 11/11/10 的分段由 checkpoint 的 ``rope_parameters``
      固定；用别的方式重算只会改变图像 token 的角度，数值测试很容易漏掉。
    * ``cos``/``sin`` 必须保持 ``cat(half, half)`` 布局，``rotate_half`` 才会
      配到和一维路径相同的频率。
    * ``scatter_image_embeds`` 就地改写 ``x``：缓冲区归调用方所有，改成返回
      副本会破坏调用方的别名假设。
"""

from __future__ import annotations

import torch

#: Qwen3.5 的 ``rope_parameters.mrope_section``：11/11/10，合计
#: ``rotary_dim/2 = 32``（H 是 11、W 是 10，因为 T 拿走三的倍数，即 32 个
#: 频率里的 11 个）。
MROPE_SECTION = (11, 11, 10)


def mrope_cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor,
                  mrope_section: tuple[int, int, int] = MROPE_SECTION
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """把 ``[3, C]`` 的位置轴合成为图像 prompt 的逐 token cos/sin。

    合成方式和参考实现一致：T 填满所有频率，然后 H 覆写
    ``slice(1, 3*section_h, 3)``、W 覆写 ``slice(2, 3*section_w, 3)``——步长 3
    的交错，频率 ``i`` 取自轴 ``i % 3``（T、H、W…），H 能到下标 31、W 到 29。
    把 32 个合成角度复制成 64，部分旋转的 ``rotate_half`` 配对才与一维表
    完全一致。

    Args:
        positions: 位置轴，``[3, C]``，行依次是 T、H、W；也接受单个位置三元组
            ``[3]``。
        inv_freq: 来自 :func:`ops.rope.make_inv_freq` 的 ``[half]`` fp32 频率轴。
            传共用轴才能让纯文本 prompt 与一维表路径逐位相同。
        mrope_section: 每根轴占的频率数，合计为 ``half``。

    Returns:
        ``(cos, sin)``，各自 ``[C, 2*half]``，采用复制半区的布局。
    """
    pos = positions
    if pos.dim() == 1:                      # 单个位置三元组
        pos = pos.reshape(3, 1)
    freqs = pos.to(torch.float32)[..., None] * inv_freq.to(pos.device)
    # freqs: [3, C, half]；T 是底，H/W 覆写各自交错的位置。
    out = freqs[0].clone()
    for dim, offset in ((1, 1), (2, 2)):
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        out[..., idx] = freqs[dim][..., idx]
    c, s = out.cos(), out.sin()
    return torch.cat([c, c], -1), torch.cat([s, s], -1)


def scatter_image_embeds(x: torch.Tensor, embeds: torch.Tensor,
                         spans: list[tuple[int, int]], start: int, end: int) -> torch.Tensor:
    """用视觉塔的嵌入覆写某个预填充分块的图像行。

    每个 span 与分块的重叠部分才会被拷贝，而写进 ``embeds`` 的偏移是 span 在
    拼接结果里的自身位置——于是在图像中间开始或结束的分块正好拿到属于它的
    那几行，分块预填充和命中缓存前缀的边界因此不用做任何特殊处理。

    Args:
        x: 分块的 token 嵌入，``[C, hidden]``，覆盖绝对 prompt 位置
            ``[start, end)``。就地改写：这块刚 gather 出来的缓冲区归调用方
            所有，拷贝一份只是多此一举。
        embeds: 所有图像的嵌入按 prompt 顺序拼接。
        spans: 这些图像的绝对 ``[s, e)`` token 区间，同样按 prompt 顺序，
            这样写进 ``embeds`` 的游标偏移才对得上。
        start: 分块的第一个绝对 prompt 位置（含）。
        end: 分块最后一个绝对 prompt 位置的下一个。

    Returns:
        ``x`` 本身，便于链式调用。分块里没有图像时原样返回，那是纯文本的
        快路径。
    """
    if embeds is None or not spans:
        return x
    base = 0
    for s, e in spans:
        a, b = max(s, start), min(e, end)
        if a < b:
            x[a - start:b - start].copy_(
                embeds[base + (a - s):base + (b - s)].to(x.dtype))
        base += e - s
    return x
