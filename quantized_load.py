"""从离线预量化目录直接读 int8 权重，全程不实例化 bf16。

Module: ``coffin/quantized_load.py``。导入图的叶子之一：只 import ``torch``、
``int8_weights`` 和 ``backends``（要问一句磁盘布局与后端是否对得上）。由
``models/qwen35/weights.py`` 在认出预量化目录之后整趟调用，所以
``engine_api.load_weights`` 一行都不用改，指到预量化目录就换了权重来源。

盘上的格式由 ``coffin/tools/quant_disk.py`` 写：

* 每个源分片对应一个 ``quant-000NN-of-000MM.safetensors``；可量化投影存成
  ``int8 [K, N]`` 连续，另一个 ``<key>#scale`` 键存 ``fp32 [N]``；
* ``quant_index.json`` 记录每个张量的 ``kind``/``shape``/``K``/``N``、它所在的分片
  和 scale 的键名，外加源 checkpoint 的 ``config.json`` 摘要。

那份索引不是装饰。没有它，读回来的形状违反一个常数倍的张量不会报错，而是在某一层
的 ``matmul`` 上错位；有了它，加载器能在装配之前把 K/N、dtype 和 scale 长度全部对
一遍，报出的是「哪个张量与索引不符」而不是「第 17 层形状不匹配」。

Constraints:
    * ``[K, N]`` 落到 :class:`int8_weights.QWeight` 时不复制、不转置。``QWeight.wq``
      是本文件的 ``[K, N]`` 缓冲区的一个转置视图，``backends/corex`` 的
      ``pack.to_kn`` 再取一次转置就回到原缓冲区本身，``.contiguous()`` 于是是空操作。
      这条路径的全部收益就在这里：读文件、装箱、完事。
    * 因此本格式只对 ``caps.weight_layout == "kn"`` 的后端有意义。声明 ``"nk"`` 的
      后端要把权重按行主序交给自己的内核，而它的内核按扁平指针索引连续内存，一份
      非连续的视图会读到错位的字节且不报错——所以那种后端在这里被拒绝，而不是被
      喂一个看起来能用、算出来是垃圾的张量。
    * 不读源 checkpoint 的任何权重分片。预量化目录自带 ``config.json``，少读 4.6 GB。
    * 非投影张量（归一化权重、``A_log``、``dt_bias``、``conv1d.weight``）在本文件里
      原样透传：它们是被 bf16 激活内核消费的，改 dtype 会让那些内核在错误的精度下
      静默运行。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from int8_weights import QWeight

#: 索引文件名。加载器靠它认出一个目录是预量化的，所以这个名字不能在任何
#: checkpoint 里出现。
INDEX_NAME = "quant_index.json"

#: 索引格式标识。布局一改就递增：读旧格式的加载器会明确报错，而不是按新布局错读
#: 字节——错读出来的 int8 是一堆看似合理的数字。
FORMAT = "coffin-int8-kn-v1"

#: 文本默认前缀，与 ``models/qwen35/weights.py`` 的那一组一致。
DEFAULT_PREFIX = "model.language_model."

#: 视觉张量的前缀。它们在预量化目录里只可能是稠密的。
VISUAL_PREFIX = "model.visual."

#: 索引里唯一一个允许 ``QWeight.wq`` 非连续的布局要求。见模块 docstring。
_KN_LAYOUT = "kn"


def is_quantized_dir(model_dir: str | Path) -> bool:
    """报告一个目录是不是本工具写出的预量化目录。

    Args:
        model_dir: 待判定的目录。

    Returns:
        目录里有 ``quant_index.json`` 时为 ``True``。只判这一个文件：它是加载器唯一
        需要的入口，其余分片名都记在里面，所以别的判据都与它冗余。
    """
    return (Path(model_dir) / INDEX_NAME).is_file()


def read_index(model_dir: str | Path) -> dict:
    """读出并校验预量化目录的索引。

    Args:
        model_dir: 预量化目录。

    Returns:
        解析后的索引字典，含 ``tensors`` 与 ``files``。

    Raises:
        FileNotFoundError: 目录里没有索引。
        ValueError: 索引的 ``format`` 不是本加载器认识的那个。按未知格式往下读只会
            得到形状对不上或数值不对的权重，不如在这里停住。
    """
    import json

    path = Path(model_dir) / INDEX_NAME
    index = json.loads(path.read_text(encoding="utf-8"))
    if index.get("format") != FORMAT:
        raise ValueError(
            f"{path}: format {index.get('format')!r} is not {FORMAT!r}; "
            f"re-run coffin/tools/quant_disk.py against the checkpoint")
    return index


def summary(index: dict) -> dict:
    """把索引里适合打日志的几项摘出来。

    Args:
        index: :func:`read_index` 的返回值。

    Returns:
        一个扁平字典：源路径、dtype、量化张量数、int8 与 scale 的字节数。启动横幅
        打它，于是「这次加载的是哪份权重、多大」不必再去翻索引文件。
    """
    totals = index.get("totals") or {}
    kinds = index.get("kinds") or {}
    return {
        "source": index.get("source"),
        "weight_dtype": index.get("weight_dtype"),
        "vision": bool(index.get("vision")),
        "q8_tensors": int(kinds.get("int8", 0)),
        "int8_bytes": int(totals.get("int8_bytes", 0)),
        "scale_bytes": int(totals.get("scale_bytes", 0)),
        "dense_bytes": int(totals.get("dense_bytes", 0)),
    }


def _require_kn_backend() -> str:
    """确认当前后端要的就是 ``[K, N]``，否则拒绝加载这种格式。

    Returns:
        后端的 ``weight_layout``，供报错信息使用。

    Raises:
        RuntimeError: 后端声明 ``weight_layout`` 不是 ``"kn"``。它自己的内核会按连续
            内存索引一份非连续的视图，读到的是错位的字节：结果是错的，而且不报错。
    """
    from backends import caps as _caps

    layout = _caps().weight_layout
    if layout != _KN_LAYOUT:
        raise RuntimeError(
            f"pre-quantised weights are stored as int8 [K, N] with one fp32 scale "
            f"per output column, which backend {_caps().name!r} cannot consume: it "
            f"declares weight_layout={layout!r}. Load the original checkpoint instead.")
    return layout


def _device_arg(device: torch.device | str) -> str:
    """把设备对象转成 ``safetensors`` 认识的设备串。

    Args:
        device: 目标设备。

    Returns:
        设备串，例如卡 ``0`` 上那个设备的完整写法。张量直接从文件读进目标
        设备，省掉一次主机内存往返；按 CPU 中转再拷贝，在 4.4 GB 上是几百毫秒的
        差别。
    """
    return str(torch.device(device))


def _check_q8(key: str, wt: torch.Tensor, scale: torch.Tensor, meta: dict) -> QWeight:
    """校验一组 int8 张量和它的索引，再包成 :class:`QWeight`。

    索引里的 ``K``/``N`` 与张量形状必须对得上，而且必须是同一次写入的产物：只对上一
    半，反量化出来的是一份尺寸正确、内容错位的权重。

    Args:
        key: checkpoint 参数名，只用于报错。
        wt: 盘上的 ``[K, N]`` int8 张量。
        scale: 同一个文件里的 ``[N]`` fp32 scale。
        meta: 索引里该张量的条目。

    Returns:
        ``wq`` 是 ``wt`` 的转置视图的 ``QWeight``；没有任何数据被复制或转置。

    Raises:
        ValueError: 形状、dtype 或 scale 长度与索引不符，或者 scale 不是 fp32。
            这些都会拿错元素算出看着合理的数，所以必须在装配前拒绝。
    """
    if wt.dim() != 2 or wt.dtype != torch.int8:
        raise ValueError(f"{key}: expected a 2-D int8 weight, got "
                         f"{tuple(wt.shape)} {wt.dtype}")
    k, n = int(wt.shape[0]), int(wt.shape[1])
    if [k, n] != list(meta.get("shape") or []):
        raise ValueError(f"{key}: index says shape {meta.get('shape')}, "
                         f"shard holds {[k, n]}")
    if scale.dim() != 1 or scale.dtype != torch.float32 or int(scale.numel()) != n:
        raise ValueError(f"{key}: scale must be fp32 [{n}], got "
                         f"{tuple(scale.shape)} {scale.dtype}")
    return QWeight.from_kn(wt, scale)


def load(model_dir: str | Path, *, device: torch.device, quant: bool = True,
         prefix: str = DEFAULT_PREFIX, weight_dtype: torch.dtype | None = None,
         vision: bool | None = None, vision_quant: bool = False) -> dict:
    """把预量化目录里的张量读成 state dict。

    Args:
        model_dir: ``coffin/tools/quant_disk.py`` 写出的目录。
        device: 张量落在哪个设备上；直接从文件读进去，不经主机内存。
        quant: 投影存成 int8 :class:`QWeight`。``False`` 会把它们反量化回
            ``weight_dtype``，那是给「只想换个权重来源」的调用方留的路，代价是完整
            的稠密权重常驻。
        prefix: 文本模型参数名前缀。``mtp.`` 与 ``lm_head.`` 无条件一并读入，理由与
            ``models/qwen35/weights.py`` 相同：绑定权重的 checkpoint 上没有
            ``lm_head``，不绑定的那一个只有它没有前缀。
        weight_dtype: 期望的量化前 dtype。``None`` 沿用索引里记的那个。给的与索引不同
            时报错——int8 码是照着某个 dtype 舍入出来的，换一个 dtype 就得重新量化，
            而静默沿用旧码会得到一个既不是 A 也不是 B 的权重。
        vision: 是否读视觉塔。``None``（自动）表示索引里写了视觉张量才读。索引声明
            没写、调用方又显式要，是错误，因为这里没有能凭空变出来的塔。
        vision_quant: 把塔的 GEMM 权重按 int8 存。本格式不量化视觉塔，所以这个组合
            被拒绝，而不是悄悄返回一份稠密的塔。

    Returns:
        按 checkpoint 参数名索引的 state dict。``quant`` 为真时投影是
        :class:`~int8_weights.QWeight`，``.wq`` 是盘上 ``[K, N]`` 缓冲区的转置视图。

    Raises:
        RuntimeError: 当前后端不是 ``[K, N]`` 布局；见 :func:`_require_kn_backend`。
            checkpoint 声明了视觉塔、而调用方显式要求读它时也在这里失败。
        ValueError: 索引格式不认识、源配置摘要对不上，或者某个张量与索引不符。
        FileNotFoundError: 索引里点名的分片不在目录里。
    """
    from safetensors import safe_open

    model_dir = Path(model_dir)
    index = read_index(model_dir)
    _require_kn_backend()

    stored_dtype = index.get("weight_dtype")
    if weight_dtype is None:
        weight_dtype = _dtype_from_name(stored_dtype)
    elif _name_of(weight_dtype) != stored_dtype:
        raise ValueError(
            f"{model_dir}: weights were quantised from {stored_dtype}, but this load "
            f"asked for {_name_of(weight_dtype)}. Re-run coffin/tools/quant_disk.py "
            f"with --weight-dtype {_name_of(weight_dtype)} instead: the int8 codes "
            f"cannot be reused across storage dtypes.")
    _check_source_config(model_dir, index)

    take_vision = bool(index.get("vision")) if vision is None else bool(vision)
    if take_vision and not index.get("vision"):
        raise RuntimeError(
            f"{model_dir}: this pre-quantised directory was written without the vision "
            f"tower (coffin/tools/quant_disk.py --include-vision would have kept it)")
    if take_vision and vision_quant:
        raise ValueError(
            f"{model_dir}: the pre-quantised format stores the vision tower dense; "
            f"--vision-quant has nothing to quantise here")

    prefixes = (prefix, "mtp.", "lm_head.")
    entries: dict[str, dict] = index["tensors"]
    wanted = {key: meta for key, meta in entries.items()
              if key.startswith(prefixes)
              or (take_vision and key.startswith(VISUAL_PREFIX))}
    if not wanted:
        raise ValueError(
            f"{model_dir}: index holds no tensor under prefixes {prefixes}")

    device_arg = _device_arg(device)
    sd: dict[str, Any] = {}
    for shard in sorted({meta["shard"] for meta in wanted.values()}):
        path = model_dir / shard
        if not path.is_file():
            raise FileNotFoundError(f"{model_dir}: index names {shard}, which is missing")
        with safe_open(str(path), framework="pt", device=device_arg) as handle:
            available = set(handle.keys())
            for key, meta in wanted.items():
                if meta["shard"] != shard:
                    continue
                if key not in available:
                    raise ValueError(f"{path}: index names {key}, shard does not hold it")
                if meta["kind"] == "q8":
                    scale_key = meta["scale_key"]
                    if scale_key not in available:
                        raise ValueError(
                            f"{path}: index names {scale_key}, shard does not hold it; "
                            f"the int8 weight without its column scale dequantises to "
                            f"a different model")
                    wt = handle.get_tensor(key)
                    scale = handle.get_tensor(scale_key)
                    sd[key] = _check_q8(key, wt, scale, meta) if quant else (
                        _check_q8(key, wt, scale, meta).dequant(weight_dtype))
                else:
                    sd[key] = handle.get_tensor(key)
    return sd


def _name_of(dtype: torch.dtype) -> str:
    """返回写进索引的 dtype 名字。

    Args:
        dtype: torch dtype。

    Returns:
        ``"bfloat16"`` 这类名字，与索引里的字符串可比。
    """
    return str(dtype).split(".")[-1]


def _dtype_from_name(name: str | None) -> torch.dtype:
    """把索引里的 dtype 名字解析回 torch dtype。

    Args:
        name: ``"bfloat16"`` / ``"float16"`` / ``"float32"``。

    Returns:
        对应的 torch dtype。

    Raises:
        ValueError: 名字不认识。猜一个 dtype 会让反量化在错误的精度上做，而这种
            错误只在精度报告里才看得出来。
    """
    table = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}
    if name not in table:
        raise ValueError(f"quant index records dtype {name!r}, which this loader does "
                         f"not know how to dequantise to")
    return table[name]


def _check_source_config(model_dir: Path, index: dict) -> None:
    """核对目录里的 ``config.json`` 是不是索引对应的那一份。

    Args:
        model_dir: 预量化目录。
        index: 读出的索引。

    Raises:
        ValueError: 两份 ``config.json`` 的摘要不同。用另一份几何形状的配置去配这批
            权重，坏的不是加载而是每一层的形状断言，报出来的位置离原因很远。
    """
    import hashlib

    expected = index.get("source_config_sha256") or ""
    config = model_dir / "config.json"
    if not expected or not config.is_file():
        return
    actual = hashlib.sha256(config.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"{model_dir}/config.json does not match the pre-quantised weights "
            f"(sha256 {actual[:12]} != {expected[:12]}); the directory was assembled "
            f"from two different checkpoints")
