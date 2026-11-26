"""不依赖 torch 读 checkpoint：``python3 -m cxrt.store`` / ``cxrt.store.load``。

Module: ``coffin/cxrt``。要跑起来就必须能读权重，而仓库里的 checkpoint 是
``torch.save`` 的产物（``Qwen3.5-4B-int8-nk``、``occamy-1.0-W8A8-nk``）。这里按它
真实的容器格式解，不引入 torch：

- 新格式是一个 zip：``<name>/data.pkl`` + ``<name>/data/<key>`` + ``version`` +
  ``byteorder``；
- ``data.pkl`` 是 pickle，里面引用 ``torch._utils._rebuild_tensor_v2``、
  ``torch.Size`` 与各种 ``*Storage`` 类型。做法是给这些 global 名注册本地替身，
  让 unpickler 把它们解析到本模块的函数/类上；
- 真正的字节在 zip 的 ``data/<key>`` 里，``persistent_id`` 给出 ``(storage_type,
  key, location, numel)``，据此把原始缓冲读成 numpy。

读出来的是 :class:`cxrt.tt.Tensor`（``bfloat16`` 存储按 7 位尾数还原成 fp32），
所以下游可以直接当 ``state_dict`` 灌进 ``nn.Module``。
"""

from __future__ import annotations

import io
import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from cxrt.tt import Tensor, bfloat16, dtype as tt_dtype, from_numpy

#: torch 的存储类名到 numpy 类型的映射；``BFloat16Storage`` 单独处理（uint16 + 移位）。
_STORAGE_TYPES: dict[str, Any] = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "ShortStorage": np.int16,
    "IntStorage": np.int32,
    "LongStorage": np.int64,
    "BoolStorage": np.bool_,
}


class _StorageType:
    """pickle 里的存储类替身：只用来标记 dtype 与读字节。

    Attributes:
        name: 如 ``"FloatStorage"``。
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:                              # pragma: no cover - 展示
        return f"<storage {self.name}>"


class _Size(tuple):
    """``torch.Size`` 的替身：就是个 tuple。"""

    def __new__(cls, values: Any = ()) -> "_Size":          # pragma: no cover - pickle
        return super().__new__(cls, values)


def _bf16_from_bits(raw: np.ndarray) -> np.ndarray:
    """把 bf16 的 uint16 存储还原成 fp32。

    Args:
        raw: ``uint16`` 数组。

    Returns:
        ``float32`` 数组，值按 bf16 精度。
    """
    bits = np.asarray(raw, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32).astype(np.float32, copy=False)


class _Loader(pickle.Unpickler):
    """把 torch 的 global 名接到本地替身上的 unpickler。

    Attributes:
        archive: 打开着的 zip。
        prefix: zip 内的目录前缀（``data.pkl`` 所在的那层）。
        cache: 已读过的 storage，按 key 缓存（多个张量可能共享一块存储）。
    """

    def __init__(self, archive: zipfile.ZipFile, prefix: str) -> None:
        super().__init__(io.BytesIO(archive.read(prefix + "data.pkl")))
        self.archive = archive
        self.prefix = prefix
        self.cache: dict[str, np.ndarray] = {}

    def find_class(self, module: str, name: str) -> Any:
        """解析 pickle 里的 global 名。

        Args:
            module: 模块名。
            name: 属性名。

        Returns:
            本地替身。

        Raises:
            pickle.UnpicklingError: 出现未支持的类型（宁可报错也不猜）。
        """
        if module.startswith("torch"):
            if name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if name == "_rebuild_tensor":
                return _rebuild_tensor_v2
            if name in {"_rebuild_parameter", "_rebuild_parameter_with_state"}:
                return _rebuild_parameter
            if name == "Size":
                return _Size
            if name in _STORAGE_TYPES or name == "BFloat16Storage":
                return _StorageType(name)
            if name in {"Tensor", "Parameter"}:
                return Tensor
        if module in {"collections", "builtins", "numpy", "numpy.core.multiarray",
                      "numpy._core.multiarray"}:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"unsupported global {module}.{name}")

    def _storage(self, storage: Any, key: str, numel: int) -> np.ndarray:
        """读出一个 storage 的真实内容。

        Args:
            storage: :class:`_StorageType`。
            key: zip 内的文件名。
            numel: 元素个数。

        Returns:
            numpy 数组。
        """
        got = self.cache.get(key)
        if got is not None:
            return got
        raw = self.archive.read(self.prefix + "data/" + key)
        if storage.name == "BFloat16Storage":
            arr = _bf16_from_bits(np.frombuffer(raw, dtype=np.uint16, count=numel))
        else:
            arr = np.frombuffer(raw, dtype=_STORAGE_TYPES[storage.name], count=numel)
        self.cache[key] = arr
        return arr


# persistent_load 需要同时拿到 storage 类与 key、numel，所以把「解析 storage」拆成
# 一个显式的两步：先把 pid 原样交给这里，再由它决定用哪个缓存。


def _resolve_storage(loader: _Loader, pid: tuple) -> np.ndarray:
    """把 ``('storage', type, key, numel)`` 解析成数组。

    Args:
        loader: 当前 :class:`_Loader`。
        pid: persistent id。

    Returns:
        numpy 数组。
    """
    if len(pid) == 4:
        _kind, storage, key, numel = pid
    else:
        _kind, storage, key = pid[0], pid[1], pid[2]
        numel = pid[3] if len(pid) > 3 else 0
    return loader._storage(storage, str(key), int(numel))       # noqa: SLF001


def _rebuild_tensor_v2(storage: Any, storage_offset: int, size: Any, stride: Any,
                       requires_grad: bool = False, backward_hooks: Any = None,
                       metadata: Any = None) -> Tensor:
    """``_rebuild_tensor_v2`` 的替身：按 offset + size + stride 取出视图并拷贝。

    Args:
        storage: 已被 :func:`_resolve_storage` 换过的 numpy 数组，或 pid tuple。
        storage_offset: 元素偏移。
        size: 形状。
        stride: 元素步长。
        requires_grad: 忽略。
        backward_hooks: 忽略。
        metadata: 忽略。

    Returns:
        :class:`cxrt.tt.Tensor`。
    """
    arr = storage
    if isinstance(arr, np.ndarray):
        itemsize = arr.dtype.itemsize
        base = arr[int(storage_offset):]
        shape = tuple(int(v) for v in size)
        strides = tuple(int(v) * itemsize for v in stride)
        view = np.lib.stride_tricks.as_strided(base, shape=shape, strides=strides,
                                              writeable=False)
        out = np.array(view, copy=True)
        return from_numpy(out)
    raise pickle.UnpicklingError("storage was not resolved before rebuild")


def _rebuild_parameter(data: Any, requires_grad: bool = True, backward_hooks: Any = None,
                      state: Any = None) -> Tensor:
    """``_rebuild_parameter`` 的替身：本项目没有梯度，按普通张量给出。

    Args:
        data: 张量。
        requires_grad: 忽略。
        backward_hooks: 忽略。
        state: 忽略。

    Returns:
        :class:`cxrt.tt.Tensor`。
    """
    return data


class _StoreLoader(_Loader):
    """:class:`_Loader` 的收尾版：persistent id 直接解析成数组。

    之所以在子类里覆盖：``persistent_load`` 拿不到 ``self`` 之外的上下文，而
    ``_rebuild_tensor_v2`` 需要的是真实数组，所以两条路都在这里接上。
    """

    def persistent_load(self, pid: Any) -> Any:
        """返回 ``('storage', ...)`` 解析后的数组。

        Args:
            pid: persistent id。

        Returns:
            numpy 数组。
        """
        if not (isinstance(pid, tuple) and pid and pid[0] == "storage"):
            raise pickle.UnpicklingError(f"unsupported persistent id {pid!r}")
        return _resolve_storage(self, pid)


def _find_prefix(archive: zipfile.ZipFile) -> str:
    """找出 ``data.pkl`` 所在的目录前缀。

    Args:
        archive: 打开的 zip。

    Returns:
        前缀，形如 ``"archive/"`` 或 ``""``。
    """
    for name in archive.namelist():
        if name.endswith("data.pkl"):
            return name[: -len("data.pkl")]
    raise ValueError("no data.pkl in archive")


def load(path: Any) -> Any:
    """读 ``torch.save`` 产物**或** safetensors，不依赖 torch。

    Args:
        path: 文件路径（``str`` / ``Path``）。

    Returns:
        pickle 出来的对象（通常是 ``dict``，键 -> :class:`Tensor`），或 safetensors
        的 ``{张量名: Tensor}``。

    Raises:
        ValueError: 两种容器都不匹配。
    """
    fp = Path(path)
    if fp.suffix == ".safetensors":
        from cxrt import safetensors as _st

        return _st.load(fp)
    if not zipfile.is_zipfile(fp):
        # 兜底：缺后缀但其实是 safetensors（头是 8 字节长度 + JSON）。
        from cxrt import safetensors as _st

        if _st.is_safetensors(fp):
            return _st.load(fp)
        raise ValueError(f"{fp} is neither a torch zip archive nor safetensors")
    with zipfile.ZipFile(fp) as archive:
        prefix = _find_prefix(archive)
        loader = _StoreLoader(archive, prefix)
        version = b""
        if prefix + "version" in archive.namelist():
            version = archive.read(prefix + "version")
        obj = loader.load()
    # version 只用于记录，不影响解析：fail-safe 地把它挂在对象上便于排查。
    if isinstance(obj, dict):
        obj.setdefault("__version__", version.decode("ascii", "ignore") or "unknown")
    return obj


def save(obj: Any, path: Any) -> None:
    """写我们自己的存档（``.npz``），不追求 torch 兼容。

    Args:
        obj: ``dict[str, Tensor]`` 或 ``Tensor``。
        path: 目标路径。
    """
    fp = Path(path)
    payload: dict[str, np.ndarray] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            payload[key] = value.data if isinstance(value, Tensor) else np.asarray(value)
    else:
        payload["value"] = obj.data if isinstance(obj, Tensor) else np.asarray(obj)
    np.savez(fp, **payload)


def state_dict_of(path: Any) -> dict[str, Tensor]:
    """把 checkpoint 归成一个纯权重表（去掉 ``__version__`` 之类的元数据键）。

    Args:
        path: checkpoint 路径。

    Returns:
        ``{"layer.weight": Tensor}``。
    """
    raw = load(path)
    if isinstance(raw, dict):
        return {k: v for k, v in raw.items() if isinstance(v, Tensor)}
    if isinstance(raw, Tensor):
        return {"value": raw}
    raise ValueError(f"unsupported checkpoint payload {type(raw)}")


__all__ = ["load", "save", "state_dict_of"]
