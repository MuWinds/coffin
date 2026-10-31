"""把投影权重离线量化为分组对称 W4 格式，并保留 A8 运行时的扩展空间。

Module: ``coffin/tools``。输出是独立格式；当前运行时加载器尚不消费它。量化只
改变二维文本投影，模型配置、分词器、MTP 和可选视觉张量仍随目录保留。

Constraints:
    * 每组沿 K 轴量化，默认 128 个权重共用一个 fp32 scale；两个有符号 4-bit
      值按低 nibble 对应偶数 K、高 nibble 对应奇数 K 打包。
    * 输出张量布局为 ``uint8 [ceil(K/2), N]``，与 CoreX ``[K,N]`` 投影布局同向；
      索引同时记录原始 K/N、分组数和 scale 配对，避免加载端猜测字节含义。
    * 源 checkpoint 只读；目标目录必须为空，避免覆盖现存量化结果。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

_ENGINE = Path(__file__).resolve().parents[1]
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from tools import quant_disk

FORMAT = "coffin-w4a8-kn-v1"
INDEX_NAME = "quant_index.json"
_SHARD_TEMPLATE = "quant-w4a8-{i:05d}-of-{total:05d}.safetensors"
_NIBBLE_OFFSET = 8
_QUANT_CHUNK_ROWS = 4096


def quantize_projection(w: torch.Tensor, group_size: int = 128
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """把 ``[N,K]`` 矩阵打包为 ``uint8 [ceil(K/2),N]`` 和 ``fp32 [G,N]``。

    Args:
        w: 已转换到目标量化 dtype 的二维权重。
        group_size: 沿 K 轴每组共享一个 scale 的元素数。

    Returns:
        打包权重及逐输出通道分组的 fp32 scale。权重按 CoreX 所需的 K-major
        布局连续存放。

    Raises:
        ValueError: 输入不是二维矩阵，或分组大小不是正数。
    """
    if w.dim() != 2:
        raise ValueError(f"projection must be 2-D, got {tuple(w.shape)}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    n, k = map(int, w.shape)
    groups = (k + group_size - 1) // group_size
    padded_k = groups * group_size
    padded_even_k = (padded_k + 1) // 2 * 2
    packed = torch.empty((padded_even_k // 2, n), dtype=torch.uint8, device=w.device)
    scales = torch.empty((groups, n), dtype=torch.float32, device=w.device)

    for start in range(0, n, _QUANT_CHUNK_ROWS):
        stop = min(n, start + _QUANT_CHUNK_ROWS)
        rows = w[start:stop].float()
        if padded_k != k:
            rows = torch.nn.functional.pad(rows, (0, padded_k - k))
        grouped = rows.reshape(stop - start, groups, group_size)
        scale = grouped.abs().amax(dim=-1).clamp_min(1e-8) / 7.0
        signed = (grouped / scale[..., None]).round().clamp_(-8, 7).to(torch.int8)
        codes = (signed.to(torch.int16) + _NIBBLE_OFFSET).reshape(stop - start, padded_k)
        if padded_even_k != padded_k:
            codes = torch.nn.functional.pad(codes, (0, padded_even_k - padded_k),
                                            value=_NIBBLE_OFFSET)
        pairs = codes.reshape(stop - start, padded_even_k // 2, 2)
        row_packed = (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)
        packed[:, start:stop] = row_packed.permute(1, 0)
        scales[:, start:stop] = scale.permute(1, 0)

    return packed.contiguous(), scales.contiguous()


def dequantize_projection(packed: torch.Tensor, scales: torch.Tensor, *,
                          k: int, group_size: int = 128) -> torch.Tensor:
    """解包并还原一个 W4 投影，作为 CPU 参考和格式校验入口。

    Args:
        packed: ``uint8 [ceil(K/2),N]`` 打包权重。
        scales: ``fp32 [ceil(K/group_size),N]`` 分组 scale。
        k: 原始 K，末组与奇数 K 的填充元素会被丢弃。
        group_size: 量化时使用的 K 分组大小。

    Returns:
        ``[N,K]`` fp32 近似权重。

    Raises:
        ValueError: 打包形状、dtype 或 scale 形状不符合格式。
    """
    if packed.dim() != 2 or packed.dtype != torch.uint8:
        raise ValueError("packed weight must be a 2-D uint8 tensor")
    if scales.dim() != 2 or scales.dtype != torch.float32:
        raise ValueError("scales must be a 2-D fp32 tensor")
    if k <= 0 or group_size <= 0:
        raise ValueError("K and group_size must be positive")
    n = int(packed.shape[1])
    groups = (k + group_size - 1) // group_size
    if tuple(scales.shape) != (groups, n):
        raise ValueError(f"scales must have shape {(groups, n)}, got {tuple(scales.shape)}")
    if packed.shape[0] * 2 < k:
        raise ValueError("packed weight is shorter than K")

    bytes_nk = packed.t().to(torch.int16)
    codes = torch.empty((n, packed.shape[0] * 2), dtype=torch.int16,
                        device=packed.device)
    codes[:, 0::2] = bytes_nk & 0xF
    codes[:, 1::2] = (bytes_nk >> 4) & 0xF
    signed = (codes - _NIBBLE_OFFSET).to(torch.float32)[:, :k]
    group_ids = torch.arange(k, device=packed.device) // group_size
    return signed * scales.t().index_select(1, group_ids)


def _validate_shards(shards: list[Path], prefixes: tuple[str, ...],
                     include_vision: bool) -> None:
    """在写出首个 shard 前拒绝重复 checkpoint 键和 sidecar 命名冲突。

    Args:
        shards: 按写出顺序处理的 safetensors 源分片。
        prefixes: 需要纳入转换的文本参数名前缀。
        include_vision: 视觉张量是否作为稠密 sidecar 保留。

    Raises:
        SystemExit: 多个源分片包含同名参数，或量化 scale 键与源参数冲突。
    """
    from safetensors import safe_open

    seen: set[str] = set()
    selected: list[str] = []
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in seen:
                    raise SystemExit(f"duplicate checkpoint tensor key: {key}")
                seen.add(key)
                if key.startswith(quant_disk.VISUAL_PREFIX):
                    if include_vision:
                        selected.append(key)
                elif key.startswith(prefixes):
                    selected.append(key)

    collisions = [key + "#scale" for key in selected if key + "#scale" in seen]
    if collisions:
        raise SystemExit(f"quantization scale key collides with source tensor: {collisions[0]}")


def convert(checkpoint: Path, out: Path, *, weight_dtype: torch.dtype,
            device: torch.device, prefixes: tuple[str, ...], include_vision: bool,
            copy_sidecars: bool, group_size: int = 128) -> dict:
    """逐 shard 转换 checkpoint，写入 W4 权重和可校验索引。

    Args:
        checkpoint: 只读源 checkpoint。
        out: 必须为空或不存在的输出目录。
        weight_dtype: 量化前权重 dtype。
        device: 分块量化执行设备。
        prefixes: 要量化的文本参数名前缀。
        include_vision: 是否将视觉张量按 bf16 原样保留。
        copy_sidecars: 是否复制模型配置和分词器文件。
        group_size: K 轴分组大小。

    Returns:
        包含各分片、张量形状及字节统计的索引字典。

    Raises:
        SystemExit: 无 safetensors、没有匹配权重，或输出目录非空。
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    shards = sorted(checkpoint.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no *.safetensors shard under {checkpoint}")
    if checkpoint.resolve() == out.resolve():
        raise SystemExit("output directory must not be the checkpoint itself")
    if group_size <= 0:
        raise SystemExit("group_size must be positive")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output directory must be empty: {out}")
    _validate_shards(shards, prefixes, include_vision)
    out.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, dict] = {}
    files: dict[str, dict] = {}
    kinds: Counter[str] = Counter()
    totals = {"w4_bytes": 0, "scale_bytes": 0, "dense_bytes": 0}
    skipped_vision = 0

    for index, source in enumerate(shards, 1):
        name = _SHARD_TEMPLATE.format(i=index, total=len(shards))
        payload: dict[str, torch.Tensor] = {}
        stats = {"tensors": 0, "w4_bytes": 0, "scale_bytes": 0, "dense_bytes": 0}
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                visual = key.startswith(quant_disk.VISUAL_PREFIX)
                if visual:
                    if not include_vision:
                        skipped_vision += 1
                        continue
                elif not key.startswith(prefixes):
                    continue

                tensor = handle.get_tensor(key).to(device)
                if visual or not quant_disk.is_quantizable(key, tensor):
                    dense = tensor.to(quant_disk.ACT_DTYPE).contiguous().cpu()
                    payload[key] = dense
                    tensors[key] = {"kind": "dense", "shape": list(dense.shape),
                                    "dtype": quant_disk._dtype_name(dense.dtype),
                                    "shard": name}
                    stats["tensors"] += 1
                    stats["dense_bytes"] += dense.numel() * dense.element_size()
                    kinds["dense"] += 1
                    del tensor, dense
                    continue

                dense = tensor.to(weight_dtype).contiguous()
                packed, scales = quantize_projection(dense, group_size)
                n, k = map(int, dense.shape)
                del tensor, dense
                payload[key] = packed.cpu()
                scale_key = key + "#scale"
                payload[scale_key] = scales.cpu()
                tensors[key] = {
                    "kind": "q4", "shape": [n, k], "n": n, "k": k,
                    "group_size": group_size, "groups": int(scales.shape[0]),
                    "packed_shape": list(packed.shape), "scale_shape": list(scales.shape),
                    "scale_key": scale_key, "shard": name,
                }
                stats["tensors"] += 1
                stats["w4_bytes"] += packed.numel()
                stats["scale_bytes"] += scales.numel() * 4
                kinds["q4"] += 1
                del packed, scales

        save_file(payload, str(out / name), metadata={"format": FORMAT})
        for field in ("w4_bytes", "scale_bytes", "dense_bytes"):
            totals[field] += stats[field]
        files[name] = stats
        print(f"  [{index}/{len(shards)}] {source.name} -> {name}  "
              f"{stats['tensors']} tensors", flush=True)

    if not tensors:
        raise SystemExit(f"no tensor under {checkpoint} matched prefixes {prefixes}")

    report = {
        "format": FORMAT,
        "generated_by": "coffin/tools/quant_disk_w4a8.py",
        "source": str(checkpoint.resolve()),
        "source_config_sha256": quant_disk._config_digest(checkpoint),
        "weight_dtype": quant_disk._dtype_name(weight_dtype),
        "activation_format": "int8 per token, runtime kernel required",
        "scheme": "symmetric per-output-channel K-group absmax; q=round(w/(maxabs/7)), "
                  "clamped to [-8,7], nibble code=q+8",
        "group_size": group_size,
        "layout": "uint8 [ceil(K/2), N], low nibble even K; fp32 [ceil(K/group_size), N]",
        "prefixes": list(prefixes),
        "vision": bool(include_vision),
        "skipped_vision_tensors": skipped_vision,
        "files": files,
        "kinds": dict(sorted(kinds.items())),
        "totals": totals,
        "tensors": tensors,
    }
    if copy_sidecars:
        report["sidecars"] = quant_disk._copy_sidecar_files(checkpoint, out)
    (out / INDEX_NAME).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    """执行命令行转换并打印压缩统计。"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="源 checkpoint 目录（只读）")
    parser.add_argument("--out", required=True, help="空的 W4 输出目录")
    parser.add_argument("--weight-dtype", default="bf16",
                        choices=sorted(quant_disk._WEIGHT_DTYPES))
    parser.add_argument("--device", default="auto", help="量化设备：auto / cpu / cuda:N")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--prefix", action="append", default=None,
                        help="要量化的参数名前缀，可重复")
    parser.add_argument("--include-vision", action="store_true",
                        help="将 model.visual.* 按 bf16 原样保留")
    parser.add_argument("--no-sidecars", action="store_true",
                        help="不复制 config.json 与分词器文件")
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_dir():
        raise SystemExit(f"not a checkpoint directory: {checkpoint}")
    out = Path(args.out)
    if out.resolve() == checkpoint.resolve():
        raise SystemExit("--out must not be the checkpoint itself")
    if args.group_size <= 0:
        raise SystemExit("--group-size must be positive")

    weight_dtype = quant_disk._WEIGHT_DTYPES[args.weight_dtype]
    device = quant_disk._pick_device(args.device)
    prefixes = tuple(args.prefix) if args.prefix else quant_disk.DEFAULT_PREFIXES
    print(f"  source : {checkpoint}", flush=True)
    print(f"  out    : {out}", flush=True)
    print(f"  device : {device}  group_size={args.group_size}", flush=True)
    started = time.time()
    report = convert(checkpoint, out, weight_dtype=weight_dtype, device=device,
                     prefixes=prefixes, include_vision=args.include_vision,
                     copy_sidecars=not args.no_sidecars, group_size=args.group_size)
    elapsed = time.time() - started
    print(json.dumps({"seconds": round(elapsed, 1), "format": FORMAT,
                      "kinds": report["kinds"], "totals": report["totals"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
