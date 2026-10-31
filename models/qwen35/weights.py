"""Qwen3.5 的 checkpoint 加载：把文本、MTP 和视觉张量搬到 GPU 上。

文本模型和 MTP 的张量按连续张量读取；checkpoint 带视觉塔、且调用方没有显式
关掉时，同一趟里把它也读进来（``config.json: vision_config``），因为常驻字典
正是 ``Qwen35.enable_vision`` 之后用来建塔的东西。MTP 头保留，投机解码要用。

视觉张量采用和文本张量不同的内存策略：

* 它们保留 ``model.visual.`` 前缀（``Qwen35`` 会把它们从字典里筛出去放进
  ``vision_sd``），而且绝不走文本的 int8 路径——视觉塔是计算受限的，量化它
  只有在 0.62 GiB 的 fp16 权重放不下时才有意义；
* 那种情况就是 ``vision_quant``，用同一个 :class:`~int8_weights.QWeight` 存成
  int8 加逐行 scale（常驻 0.31 GiB）。塔一次只反量化一个 GEMM，所以临时量是
  一个 fp16 矩阵，而不是再多一份塔的拷贝。

Module: ``coffin/models/qwen35``。依赖 ``backends``（决定张量落在哪个设备、
投影用什么 dtype）、``ops/dtypes`` 和 ``int8_weights``，以及同级的
``vision``；由 ``engine_api.load_weights`` 调用，返回的字典直接交给
``Qwen35``。

Constraints:
    * 只有二维矩阵走 ``is_quantizable``：归一化权重、``A_log``、``dt_bias``
      和 ``conv1d.weight`` 都被 bf16 激活内核消费，必须保持 bf16，否则那些
      内核会在错误的精度下静默运行。
    * 在加载过程中量化，而不是加载完再量化：fp16 张量一量化完就丢掉，于是峰值
      内存是两种形式中较大的那个而不是两者之和——9 GB 的 checkpoint 能塞进
      16 GB 卡靠的就是这一点。
    * 键按前缀字符串匹配（``model.language_model.``、``mtp.``、
      ``model.visual.``）；它们来自 checkpoint 布局，在这里改个名字不会触发形状
      检查，而是直接让加载坏掉。
    * ``vision=None`` 表示「自动」：checkpoint 声明了视觉塔才加载；在没有视觉
      张量的 checkpoint 上显式传 ``True`` 是错误，不是得到一个空塔。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from backends import device as _device
from int8_weights import QWeight, is_quantizable
from models.qwen35.vision import VISUAL_PREFIX, is_vision_quantizable
from ops.dtypes import D, WF

def load_weights(model_dir: str | Path, prefix: str = "model.language_model.",
                 quant: bool = False, weight_dtype: torch.dtype = WF,
                 vision: bool | None = None,
                 vision_quant: bool = False) -> dict:
    """把文本模型、MTP 和视觉张量从磁盘读上 GPU。

    这是慢的那一段（9 GB 约 1 分钟）；改完代码想重建模型的调用方，可以把返回的
    字典留在显存里，再经 ``Qwen35(..., weights=...)`` 传回去，省掉重新加载。

    开了 ``quant`` 时，文本的二维矩阵以 :class:`int8_weights.QWeight` 而不是
    fp16 的形式到达：每步解码要读的字节数减半，常驻占用也减半。
    ``vision_quant`` 对视觉塔的 GEMM 权重独立地做同样的事。

    Args:
        model_dir: 放着 ``config.json`` 和 ``*.safetensors`` 分片的 checkpoint
            目录。
        prefix: 文本模型在 checkpoint 里的参数名前缀。
        quant: 把文本的二维投影矩阵存成 int8 ``QWeight``。
        weight_dtype: 投影和视觉张量用的 dtype；默认 ``caps.weight_dtype``，
            在本引擎调优的那块 DCU 上是 fp16，因为那颗芯片有 packed fp16 计算。
        vision: 是否加载视觉塔。``None``（自动）表示 ``config.json`` 声明了
            ``vision_config`` 才加载。
        vision_quant: 把视觉塔的 GEMM 权重存成 int8 加逐行 scale。只有当放不下的
            正是 fp16 的塔时才值得。

    Returns:
        按 checkpoint 参数名索引的 state dict；文本/MTP 的键已去掉 ``prefix``，
        视觉的键仍带着 ``VISUAL_PREFIX``，好让调用方把它们拆出去。

    Raises:
        RuntimeError: checkpoint 声明了 ``vision_config`` 却找不到任何
            ``VISUAL_PREFIX`` 张量。宁可报出这份不匹配的 checkpoint，也不要启动
            一个会在第一张图上失败的引擎。
    """
    from safetensors import safe_open

    model_dir = Path(model_dir)
    with (model_dir / "config.json").open() as f:
        cfg = json.load(f)
    take_vision = bool(cfg.get("vision_config")) if vision is None else bool(vision)

    # ``lm_head.`` 无条件列出：在绑定权重的 checkpoint（4B）上这个张量不存在，
    # 前缀自然什么都匹配不到；而在不绑定的（27B）上，这里是唯一能拿到它的地方——
    # 它没有 ``prefix`` 可挂。漏掉它不会报错，头会悄悄退回用嵌入，那是错误答案而
    # 不是异常，所以两种情况只能按实际存在的张量来分辨。
    prefixes = (prefix, "mtp.", "lm_head.")
    sd: dict[str, torch.Tensor] = {}
    n_vision = 0
    for p in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(p), "pt") as f:
            for k in f.keys():
                if k.startswith(prefixes):
                    t = f.get_tensor(k).to(_device)
                    if not is_quantizable(k, t):
                        sd[k] = t.to(D).contiguous()
                        continue
                    t = t.to(weight_dtype).contiguous()
                    sd[k] = QWeight.quantize(t) if quant else t
                elif take_vision and k.startswith(VISUAL_PREFIX):
                    t = f.get_tensor(k).to(_device, dtype=weight_dtype).contiguous()
                    if vision_quant and is_vision_quantizable(k, t):
                        sd[k] = QWeight.quantize(t)
                    else:
                        sd[k] = t
                    n_vision += 1
    if take_vision and n_vision == 0:
        raise RuntimeError(
            f"config.json declares a vision_config but no {VISUAL_PREFIX}* tensor "
            f"was found under {model_dir}")
    return sd
