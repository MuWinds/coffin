"""把 checkpoint 的投影离线预量化成 int8，落盘时就已经是内核要的形状。

Module: ``coffin/tools``。命令行入口，只 import ``torch``、``safetensors`` 和叶子模块
``int8_weights``。它写出的目录由 ``coffin/quantized_load.py`` 读回去，
``models/qwen35/weights.py`` 认出这个目录之后把整趟加载交给那条路径，所以
``engine_api.load_weights`` 一行都不用改。

要它存在的理由是三笔每次启动都要重付的开销：

* ``load_weights`` 现在在加载过程中量化，于是每次启动都要把 9.3 GB 的 bf16 读上卡
  再压成 4.4 GB。权重是常量，这一步只该做一次；
* 预填充路径的 ``linear_prefill`` 每次调用都要把 int8 反量化回 bf16（本卡实测
  +173 ms/请求），因为内核不接受 int8 操作数；
* 反量化之后的 bf16 拷贝要另占一份显存（27B 上是 +25.9 GiB），那是 hoisted-dequant
  在 32 GiB 卡上被判死的唯一原因。

盘上的布局就是 ``backends/corex/pack.py`` 的 ``KnQWeight`` 要的形状：``int8 [K, N]``
连续，加每输出列一个 fp32 ``[N]`` scale。存成 checkpoint 的 ``[N, K]`` 会让每次加载
都多一次转置，而转置正是本文件想省掉的那一步。

Constraints:
    * 量化结果必须逐位等于 ``QWeight.quantize`` 在同样输入上的结果，否则「预量化」和
      「运行时量化」是两个不同的模型。所以这里不重写公式，逐块调
      ``QWeight.quantize``，并且和运行时一样在 fp32 里算、在设备上算。
    * 必须分块。``int8_weights._QUANT_CHUNK_ROWS`` 的理由在这里原样成立：不分块时
      27B 的 ``lm_head`` 要一份 4.7 GiB 的 fp32 临时量，4B 上同样会把卡吃满。
      分块之后峰值是一个块的 fp32 临时量加一份 int8 目标，与矩阵高度无关。
    * 只读源分片、只写新文件。checkpoint 本身一个字节都不动。
    * 非投影张量按 ``is_quantizable`` 判定后原样（bf16）写出。归一化权重、``A_log``、
      ``dt_bias``、``conv1d.weight`` 由 bf16 激活内核消费，量化它们换不来带宽，只会
      让那些内核在错误精度下静默运行。
    * scale 和 int8 一起放在 safetensors 里，不放 JSON：``float`` 的十进制往返不是
      逐位的，scale 差一个 ulp 会挪动整列权重，而「预量化 vs 运行时量化逐位一致」
      正是这条路径的验收口径。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import torch

# 本文件在 ``coffin/tools/`` 下，脚本启动时 ``sys.path[0]`` 是工具目录而不是引擎根，
# 所以先把引擎根插进去；下面两条 import 必须留在这几行之后，否则 ``int8_weights``
# 导入不到。
_ENGINE = Path(__file__).resolve().parents[1]
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from int8_weights import _QUANT_CHUNK_ROWS, QWeight, is_quantizable  # noqa: E402

#: 索引文件名。加载器靠它认出一个目录是预量化的，所以这个名字必须不可能出现在
#: 任何 checkpoint 里。
INDEX_NAME = "quant_index.json"

#: 索引格式版本。布局一改就递增：读旧格式的加载器会明确报错，而不是按新布局错读
#: 字节——错读出来的 int8 是一堆看似合理的数字，不会自己报错。
FORMAT = "coffin-int8-kn-v1"

#: 必须保持稠密的二维矩阵后缀（Qwen3.5-MoE 的两处门）。
#:
#: 路由矩阵决定每个 token 选哪 8 个专家，int8 的噪声比 4-bit 小得多，但它仍然是
#: 「选谁」的依据，翻转一次就是整层错；共享专家门是逐 token 一个标量，量化它省不
#: 下带宽。两者都只有几十 KB，保持 bf16 的代价可以忽略。
DENSE_SUFFIXES = ("mlp.gate.weight", "mlp.shared_expert_gate.weight")

#: 每个源分片对应一个输出分片，输出名按输入顺序编号。
_SHARD_TEMPLATE = "quant-{i:05d}-of-{total:05d}.safetensors"

#: 视觉张量的前缀。P0 只做文本，所以默认不写它们（见 ``--include-vision``）。
VISUAL_PREFIX = "model.visual."

#: 默认的文本参数名前缀，与 ``models/qwen35/weights.py`` 用的那一组一致。
#: ``lm_head.`` 无条件列出：绑定权重的 checkpoint 上没有这个张量，前缀自然匹配不到。
DEFAULT_PREFIXES = ("model.language_model.", "mtp.", "lm_head.")

#: 非投影张量写到盘上的 dtype。必须与 ``weights.py`` 里的 ``t.to(D)`` 一致，
#: 否则两条路交出的 dense 张量 dtype 不同，消费它们的 bf16 内核会在错误精度下运行。
ACT_DTYPE = torch.bfloat16

#: ``--weight-dtype`` 接受的字符串。默认 bf16：本 checkpoint 与
#: ``backends/corex`` 声明的 ``caps.weight_dtype`` 都是 bf16，转换因此是空操作，
#: 盘上的 int8 与运行时量化出来的逐位相同。
_WEIGHT_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def _dtype_name(dtype: torch.dtype) -> str:
    """返回写进索引的 dtype 名字。

    Args:
        dtype: tensor 的 dtype。

    Returns:
        ``"bfloat16"`` 这类 torch 自己的名字；索引里存字符串而不是枚举，好让加载器
        能对着 ``caps()`` 的值比较。
    """
    return str(dtype).split(".")[-1]


def _pick_device(request: str) -> torch.device:
    """解析 ``--device``，并拒绝在一个没有加速卡的主机上量化。

    设备命名空间从 ``backends.caps().device`` 推出来，不写厂商拼写：本仓库的纯度门禁
    禁止后端包之外出现设备 API 的字面量，而且同一份工具要能在没有加速卡的开发机上跑
    （那时后端是纯 torch 的参考实现，设备就是 ``cpu``）。

    Args:
        request: ``"auto"``、``"cpu"`` 或任何 torch 设备串。

    Returns:
        量化实际落到的设备。``auto`` 选中后端有加速卡就用它。

    Raises:
        SystemExit: 指定了加速卡却看不见它。静默退回 CPU 会让几百 MB 的 fp32 中间量
            挤进主机内存，而且除法的舍入路径与运行时那条路不同，逐位一致就此失效。
    """
    from backends import caps as _caps
    from backends import device as _device

    if request == "auto":
        return torch.device("cpu") if _caps().device == "cpu" else _device
    device = torch.device(request)
    if device.type == "cpu":
        return device
    runtime = getattr(torch, _caps().device, None)
    probe = getattr(runtime, "is_available", None) if runtime is not None else None
    if probe is None or not probe():
        raise SystemExit(f"--device {request}: no accelerator visible from torch")
    return device


def quantize_projection(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """把 ``[N, K]`` 投影量化成 ``[K, N]`` int8 加 ``[N]`` fp32 列 scale。

    公式不在本文件里：每个行块原样交给 :meth:`int8_weights.QWeight.quantize`，所以
    「预量化」与「运行时量化」用的是同一段代码、同一串舍入。外层再按输出行分块，是
    为了把临时量限制在一块上；每行的 absmax 只读自己那一行，分块因此不改变结果。

    Args:
        w: 设备上的 ``[N, K]`` 权重，dtype 已经与运行时那条路的 ``weight_dtype`` 一致。

    Returns:
        ``(int8 [K, N] 连续, fp32 [N])``。两者都归调用方所有，``QWeight`` /
        ``KnQWeight`` 要求它们成对出现，拆开任何一个都会让反量化整体偏移。

    Raises:
        ValueError: ``w`` 不是二维。一维归一化权重走到这里说明 ``is_quantizable``
            的判定被别人绕过了，转置它只会在很远的地方报形状错误。
    """
    if w.dim() != 2:
        raise ValueError(f"projection must be 2-D, got {tuple(w.shape)}")
    n, k = int(w.shape[0]), int(w.shape[1])
    out = torch.empty((k, n), dtype=torch.int8, device=w.device)
    scale = torch.empty(n, dtype=torch.float32, device=w.device)
    for start in range(0, n, _QUANT_CHUNK_ROWS):
        stop = min(n, start + _QUANT_CHUNK_ROWS)
        q = QWeight.quantize(w[start:stop])
        # 跨列写而不是先攒一份 ``[N, K]`` 再整体转置：后者要同时活着两份 int8
        # （27B 的 lm_head 是 2 x 1.27 GB），而这里每个块的写是 K 段连续字节。
        out[:, start:stop] = q.wq.t()
        scale[start:stop] = q.scale
    return out, scale


def _config_digest(ckpt: Path) -> str:
    """算 checkpoint ``config.json`` 的 SHA-256。

    加载器会拿它核对预量化目录里的配置是不是同一份：把 int8 权重配上另一个
    几何形状的 ``config.json``，坏的不是报错而是每一层的形状断言，报出来的位置
    离真正的原因很远。

    Args:
        ckpt: checkpoint 目录。

    Returns:
        十六进制摘要；目录里没有 ``config.json`` 时返回 ``""``（调用方据此跳过核对）。
    """
    path = ckpt / "config.json"
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_sidecar_files(ckpt: Path, out: Path) -> list[str]:
    """把 checkpoint 里除权重分片之外的小文件复制过去。

    ``config.json`` 是 ``engine_api`` 做架构分派要读的那一份，``tokenizer*`` 是
    ``runtime_service`` 起服务时要的那一组。少了它们，预量化目录就只能被当成一个
    手工指定路径的权重仓库，不能直接当 checkpoint 目录用。

    权重分片一律不复制，``model.safetensors.index.json`` 尤其不能带过去：它会引导
    读它的人去打开一批不存在的分片。

    Args:
        ckpt: 源 checkpoint 目录。
        out: 预量化目录。

    Returns:
        复制过的文件名。
    """
    copied: list[str] = []
    for path in sorted(ckpt.iterdir()):
        if not path.is_file():
            continue
        if path.name.endswith((".safetensors", ".bin", ".pt", ".pth", ".safetensors.index.json")):
            continue
        if path.name == INDEX_NAME:
            continue
        shutil.copyfile(path, out / path.name)
        copied.append(path.name)
    return copied


#: 逐专家张量的名字。Qwen3.5-MoE 的 checkpoint 把它拆成 ``experts.{e}.{proj}.weight``，
#: 一层 256 专家 x 3 个投影 = 768 个二维张量，40 层共 30720 个。
#:
#: 为什么必须在产物里融合：运行时加载是按张量循环的，而单张量的固定开销（本机实测
#: int8 通路约 3 ms）乘上 30720 就是 90 s 起。Qwen3.8-Flash-Next 的 125B 能两分钟加载完，
#: 靠的正是它的 checkpoint 本来就是 ``[E, 2I, H]`` / ``[E, H, I]`` 两个三维栈。
_EXPERT_KEY = re.compile(
    r"^(?P<mlp>.+\.mlp)\.experts\.(?P<e>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$")

#: 融合后的两个栈在 state dict / 索引里的键名后缀。
GATE_UP_STACK = "experts.gate_up_stack"
DOWN_STACK = "experts.down_stack"


def _scan_experts(shards: list, prefixes: tuple, include_vision: bool
                  ) -> tuple[dict, dict, set]:
    """第一趟只读键：把逐专家张量按层归组，并记下每层归哪个输出分片。

    张量一律不读：这一趟的代价只与键的数量有关（31666 个键约几十毫秒），而真正的读盘
    与量化留到第二趟，按分片顺序做，峰值内存因此只有一层。

    Args:
        shards: 源分片，按名字排序。
        prefixes: 要收进来的参数名前缀。
        include_vision: 视觉张量是否一并收进来。

    Returns:
        ``(groups, owner, expert_keys)``：``groups[mlp][proj][e] = (shard, key)``；
        ``owner[mlp] = shard`` 是这层非专家张量所在的分片（融合栈就写到那里）；
        ``expert_keys`` 是要从逐张量通路里排除掉的键。
    """
    from safetensors import safe_open

    groups: dict = {}
    owner: dict = {}
    expert_keys: set = set()
    for source in shards:
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(VISUAL_PREFIX):
                    if not include_vision:
                        continue
                elif not key.startswith(prefixes):
                    continue
                match = _EXPERT_KEY.match(key)
                if match is None:
                    layer = _owner_of(key)
                    if layer:
                        owner.setdefault(layer, source)
                    continue
                mlp = match.group("mlp")
                groups.setdefault(mlp, {"gate_proj": {}, "up_proj": {}, "down_proj": {}})
                groups[mlp][match.group("proj")][int(match.group("e"))] = (source, key)
                expert_keys.add(key)
    return groups, owner, expert_keys


def _owner_of(key: str) -> str:
    """从任意非专家键取出它所属层的 ``mlp`` 前缀之外的稳定标识。

    Args:
        key: checkpoint 参数名。

    Returns:
        ``model.language_model.layers.N`` 这样的层前缀；匹配不到时返回空串。
    """
    match = re.match(r"^(.*\.layers\.\d+)\.", key)
    return match.group(1) if match else ""

def _fused_stacks(group: dict, read, weight_dtype: torch.dtype,
                  device: torch.device) -> tuple:
    """把一层 768 个逐专家投影量化并融合成两个三维栈。

    Args:
        group: ``{"gate_proj": {e: (shard, key)}, "up_proj": ..., "down_proj": ...}``。
        read: ``pair -> Tensor``，把 ``(shard, key)`` 读成 CPU bf16 张量。
        weight_dtype: 量化前把权重窄到的 dtype。
        device: 量化执行设备。

    Returns:
        ``(gate_up_wq, gate_up_scale, down_wq, down_scale)``：前两个是 ``[E, 2I, H]``
        int8 与 ``[E, 2I]`` fp32，后两个是 ``[E, H, I]`` 与 ``[E, H]``。gate 在前、
        up 在后——与参考实现 ``gate_up_proj[e].chunk(2, -1)`` 的顺序一致，反过来
        不会报错，只会让模型把门和升混起来算。
    """
    experts = len(group["gate_proj"])
    if len(group["up_proj"]) != experts or len(group["down_proj"]) != experts:
        raise SystemExit(
            f"expert count mismatch: gate={experts} up={len(group['up_proj'])} "
            f"down={len(group['down_proj'])}")
    gu_parts, gu_scales, dn_parts, dn_scales = [], [], [], []
    for e in range(experts):
        gq, gs = quantize_projection(
            read(group["gate_proj"][e]).to(device).to(weight_dtype).contiguous())
        uq, us = quantize_projection(
            read(group["up_proj"][e]).to(device).to(weight_dtype).contiguous())
        dq, ds = quantize_projection(
            read(group["down_proj"][e]).to(device).to(weight_dtype).contiguous())
        # 按 ``kn`` 布局拼：``quantize_projection`` 交出来的已经是 ``[K, N]``，gate 与 up
        # 沿 N 轴接起来就是 ``[K, 2I]``，转置一次都不用做。
        gu_parts.append(torch.cat([gq, uq], dim=1))
        gu_scales.append(torch.cat([gs, us], dim=0))
        dn_parts.append(dq)
        dn_scales.append(ds)
    return (torch.stack(gu_parts).contiguous(), torch.stack(gu_scales).contiguous(),
            torch.stack(dn_parts).contiguous(), torch.stack(dn_scales).contiguous())


def _emit_stack(payload: dict, tensors: dict, per_file: dict, kinds: Counter,
                mlp: str, suffix: str, wq: torch.Tensor, scale: torch.Tensor,
                shard_name: str) -> None:
    """把一个融合栈写进当前分片的 payload 与索引。

    Args:
        payload: 当前分片待写的张量字典。
        tensors: 索引里的张量表。
        per_file: 当前分片的字节统计。
        kinds: 全局的类型计数。
        mlp: 这一层的 ``mlp`` 前缀。
        suffix: :data:`GATE_UP_STACK` 或 :data:`DOWN_STACK`。
        wq: ``[E, N, K]`` int8 栈。
        scale: ``[E, N]`` fp32。
        shard_name: 当前输出分片名。
    """
    key = f"{mlp}.{suffix}"
    payload[key] = wq.cpu()
    payload[key + "#scale"] = scale.cpu()
    tensors[key] = {
        "kind": "q8-moe",
        "experts": int(wq.shape[0]),
        "n": int(wq.shape[2]),
        "k": int(wq.shape[1]),
        "shape": list(wq.shape),
        "scale_shape": list(scale.shape),
        "scale_key": key + "#scale",
        "shard": shard_name,
    }
    kinds["int8-moe"] += 1
    per_file["tensors"] += 1
    per_file["int8_bytes"] += wq.numel()
    per_file["scale_bytes"] += scale.numel() * 4

def convert(ckpt: Path, out: Path, *, weight_dtype: torch.dtype,
            device: torch.device, prefixes: tuple[str, ...], include_vision: bool,
            copy_sidecars: bool) -> dict:
    """把一个 checkpoint 目录整体转成预量化目录。

    Args:
        ckpt: 源 checkpoint，只读。
        out: 输出目录，会被创建；同名的旧输出分片会先删掉，免得两次转换的分片混在
            一个索引里。
        weight_dtype: 量化前把权重窄到的 dtype。必须与运行时加载器用的那个一致，
            否则盘上的 int8 与运行时量化出来的不是同一组字节。
        device: 量化落到的设备。
        prefixes: 要收进来的参数名前缀。
        include_vision: 把 ``model.visual.*`` 按 bf16 原样穿过去。视觉塔不在 P0 的
            范围里，但一个下游只想换权重、又不想改 ``vision=`` 参数的调用方需要它。
        copy_sidecars: 复制 ``config.json`` 与分词器文件，让输出目录能直接当
            checkpoint 目录用。

    Returns:
        写进 ``quant_index.json`` 的那个字典。

    Raises:
        SystemExit: 目录里没有 safetensors 分片，或者一个张量都没匹配上。空索引
            读回去是一个零权重模型，比直接失败更难查。
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    shards = sorted(ckpt.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no *.safetensors shard under {ckpt}")

    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("quant-*.safetensors"):
        stale.unlink()

    tensors: dict[str, dict] = {}
    files: dict[str, dict] = {}
    kinds: Counter[str] = Counter()
    totals = {"int8_bytes": 0, "scale_bytes": 0, "dense_bytes": 0}
    skipped_vision = 0

    # 第一趟只读键：把逐专家张量按层归组。第二趟按分片顺序读盘、量化、融合，峰值
    # 内存因此只有一层（gate/up/down 各 256 份 bf16 加融合后的 int8 栈约 2.4 GB）。
    groups, owner, expert_keys = _scan_experts(shards, prefixes, include_vision)
    #: 跨分片读专家时要同时打开多个源分片；它们都是内存映射，打开本身不花钱。
    extra: dict = {}

    def read(pair):
        """把 ``(shard, key)`` 读成 CPU 张量。

        Args:
            pair: ``(源分片, 参数名)``。

        Returns:
            该张量的 CPU 副本（safetensors 自己管缓存）。
        """
        origin, key = pair
        handle = extra.get(origin)
        if handle is None:
            handle = safe_open(str(origin), framework="pt", device="cpu")
            handle.__enter__()
            extra[origin] = handle
        return handle.get_tensor(key)

    for index, source in enumerate(shards, 1):
        name = _SHARD_TEMPLATE.format(i=index, total=len(shards))
        payload: dict[str, torch.Tensor] = {}
        per_file = {"tensors": 0, "int8_bytes": 0, "scale_bytes": 0, "dense_bytes": 0}
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                # 逐专家张量留到本分片末尾由 ``_fused_stacks`` 一次性融合，不走这里。
                if key in expert_keys:
                    continue
                if key.startswith(VISUAL_PREFIX):
                    if not include_vision:
                        skipped_vision += 1
                        continue
                elif not key.startswith(prefixes):
                    continue

                tensor = handle.get_tensor(key).to(device)
                # 视觉张量、非投影张量、以及路由器/共享专家门原样穿过去。它们不参与
                # W8A8，也没有内核能消费 int8 形式，量化只会白丢精度。
                if (key.startswith(VISUAL_PREFIX) or key.endswith(DENSE_SUFFIXES)
                        or not is_quantizable(key, tensor)):
                    dense = tensor.to(ACT_DTYPE).contiguous().cpu()
                    payload[key] = dense
                    kinds[_dtype_name(dense.dtype)] += 1
                    tensors[key] = {"kind": "dense", "shape": list(dense.shape),
                                    "dtype": _dtype_name(dense.dtype),
                                    "shard": name}
                    per_file["tensors"] += 1
                    per_file["dense_bytes"] += dense.numel() * dense.element_size()
                    del tensor, dense
                    continue

                packed = tensor.to(weight_dtype).contiguous()
                wt, scale = quantize_projection(packed)
                # 盘上只留 int8 和 scale，bf16 在这里就丢掉，峰值因此与源分片大小
                # 无关，只与当前这一个张量的 bf16 形式有关。
                del packed, tensor
                payload[key] = wt.cpu()
                # scale 的键带后缀并在索引里点名，加载器不从权重名推导它：推导规则
                # 一旦和写入端不一致，读到的是别的张量或干脆读不到。
                payload[key + "#scale"] = scale.cpu()
                kinds["int8"] += 1
                tensors[key] = {
                    "kind": "q8",
                    "k": int(wt.shape[0]),
                    "n": int(wt.shape[1]),
                    "shape": [int(wt.shape[0]), int(wt.shape[1])],
                    "scale_key": key + "#scale",
                    "shard": name,
                }
                per_file["tensors"] += 1
                per_file["int8_bytes"] += wt.numel()
                per_file["scale_bytes"] += scale.numel() * 4
                del wt, scale

            # 这一层属于哪个分片由它的非专家张量决定；融合栈写到同一个分片，
            # 加载侧因此仍能按分片顺序一次走完。
            for mlp, group in groups.items():
                if owner.get(mlp[:-4]) != source:
                    continue
                wq_gu, sc_gu, wq_dn, sc_dn = _fused_stacks(group, read, weight_dtype, device)
                _emit_stack(payload, tensors, per_file, kinds, mlp, GATE_UP_STACK,
                            wq_gu, sc_gu, name)
                _emit_stack(payload, tensors, per_file, kinds, mlp, DOWN_STACK,
                            wq_dn, sc_dn, name)
                del wq_gu, sc_gu, wq_dn, sc_dn

        save_file(payload, str(out / name), metadata={"format": FORMAT})
        for field in ("int8_bytes", "scale_bytes", "dense_bytes"):
            totals[field] += per_file[field]
        files[name] = per_file
        print(f"  [{index}/{len(shards)}] {source.name} -> {name}  "
              f"{per_file['tensors']} tensors", flush=True)
        del payload

    for handle in extra.values():
        handle.__exit__(None, None, None)

    if not tensors:
        raise SystemExit(
            f"no tensor under {ckpt} matched prefixes {prefixes}; refusing to write "
            f"an index that would load a zero-weight model")

    report = {
        "format": FORMAT,
        "generated_by": "coffin/tools/quant_disk.py",
        "source": str(ckpt.resolve()),
        "source_config_sha256": _config_digest(ckpt),
        "weight_dtype": _dtype_name(weight_dtype),
        "act_dtype": _dtype_name(ACT_DTYPE),
        "scheme": ("per-output-column symmetric absmax: s = max|w|/127 then "
                   "round(w/s) clamped to [-127, 127]; s floored at 1e-8"),
        "chunk_rows": _QUANT_CHUNK_ROWS,
        "layout": "int8 [K, N] contiguous + fp32 [N]",
        "prefixes": list(prefixes),
        "vision": bool(include_vision),
        "skipped_vision_tensors": skipped_vision,
        "files": files,
        "kinds": dict(sorted(kinds.items())),
        "totals": totals,
        "tensors": tensors,
    }
    if copy_sidecars:
        report["sidecars"] = _copy_sidecar_files(ckpt, out)
    (out / INDEX_NAME).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def _dir_bytes(path: Path) -> int:
    """目录里所有普通文件的总字节数，用来报磁盘占用的实测值。

    Args:
        path: 要统计的目录。

    Returns:
        字节数。符号链接按它的目标大小算，因为 ``du`` 读数也是那个意思。
    """
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main(argv: list[str] | None = None) -> int:
    """命令行入口：checkpoint 目录进，预量化目录出。

    Args:
        argv: 参数列表；``None`` 取 ``sys.argv[1:]``。

    Returns:
        ``0`` 转换完成，``2`` 参数的取值不认识。

    Raises:
        SystemExit: 源目录不存在，或者一个张量都没匹配上（见 :func:`convert`）。
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="源 checkpoint 目录（只读）")
    parser.add_argument("--out", required=True, help="预量化目录，会被创建")
    parser.add_argument("--weight-dtype", default="bf16", choices=sorted(_WEIGHT_DTYPES),
                        help="量化前窄到的 dtype，必须与运行时加载器的 weight_dtype 一致")
    parser.add_argument("--device", default="auto", help="量化设备：auto / cpu / cuda:N")
    parser.add_argument("--prefix", action="append", default=None,
                        help="要收进来的参数名前缀，可重复；默认文本+mtp+lm_head")
    parser.add_argument("--include-vision", action="store_true",
                        help="把 model.visual.* 按 bf16 原样带过去（默认不带）")
    parser.add_argument("--no-sidecars", action="store_true",
                        help="不复制 config.json 与分词器文件")
    args = parser.parse_args(argv)

    ckpt = Path(args.checkpoint)
    if not ckpt.is_dir():
        raise SystemExit(f"not a checkpoint directory: {ckpt}")
    out = Path(args.out)
    if out.resolve() == ckpt.resolve():
        raise SystemExit("--out must not be the checkpoint itself; it is opened read-only")
    weight_dtype = _WEIGHT_DTYPES[args.weight_dtype]
    device = _pick_device(args.device)
    prefixes = tuple(args.prefix) if args.prefix else DEFAULT_PREFIXES

    print(f"  source : {ckpt}", flush=True)
    print(f"  out    : {out}", flush=True)
    print(f"  device : {device}  weight_dtype={_dtype_name(weight_dtype)}", flush=True)
    source_bytes = _dir_bytes(ckpt)
    started = time.time()
    report = convert(ckpt, out, weight_dtype=weight_dtype, device=device,
                     prefixes=prefixes, include_vision=args.include_vision,
                     copy_sidecars=not args.no_sidecars)
    elapsed = time.time() - started

    written = _dir_bytes(out)
    summary = {
        "source_bytes": source_bytes,
        "written_bytes": written,
        "ratio": round(written / source_bytes, 4) if source_bytes else None,
        "seconds": round(elapsed, 1),
        "q8_tensors": report["kinds"].get("int8", 0),
        "dense_tensors": sum(v for k, v in report["kinds"].items() if k != "int8"),
        "int8_bytes": report["totals"]["int8_bytes"],
        "scale_bytes": report["totals"]["scale_bytes"],
        "device": str(device),
        "weight_dtype": _dtype_name(weight_dtype),
    }
    print(f"  wrote {written / 1e9:.2f} GB in {elapsed:.1f}s "
          f"({summary['ratio']}x of the checkpoint)", flush=True)
    print("QUANT_DISK_JSON_BEGIN", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("QUANT_DISK_JSON_END", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
