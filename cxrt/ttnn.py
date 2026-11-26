"""``cxrt.tt`` 的 ``nn`` 半边：Module / Parameter / Linear / Embedding / 函数面。

Module: ``coffin/cxrt``。推理链只用到前向，所以这里的 ``Module`` 不做反向、不做
autograd，只负责三件真实需要的事：

1. ``__setattr__`` 把子模块与参数挂上（``state_dict`` 与 ``to()`` 要遍历它们）；
2. ``__call__`` 转发到 ``forward``；
3. ``state_dict`` / ``load_state_dict`` 用**扁平点号键**，和 checkpoint 的键名一致。

``functional`` 是一个命名空间对象（不是子模块），因为模型里写的是
``F = torch.nn.functional`` 再 ``F.pad(...)``；只要属性访问拿得到函数就够了。
"""

from __future__ import annotations

import types
from typing import Any, Iterable, Iterator

import numpy as np

from cxrt.tt import (
    Tensor, bfloat16, cat, float32, int64, matmul, ones, softmax, zeros,
)


class Parameter(Tensor):
    """参数张量。语义上「可训练」，这里只是标记。"""

    def __init__(self, data: Any, requires_grad: bool = True) -> None:
        super().__init__(data)
        self.requires_grad = requires_grad

    def detach(self) -> Tensor:
        """返回不带参数标记的视图。"""
        return Tensor(self.data, self.dtype, self.device)


class Module:
    """前向用的模块基类。

    Attributes:
        training: 训练/推理开关。推理链恒为假，但模型会读它。
    """

    training = False

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_non_persistent", set())

    # 挂载 ---------------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Tensor):
            self._buffers.setdefault(name, value)
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        for table in ("_parameters", "_buffers", "_modules"):
            store = self.__dict__.get(table)
            if store and name in store:
                return store[name]
        raise AttributeError(name)

    def register_buffer(self, name: str, value: Tensor | None,
                        persistent: bool = True) -> None:
        """登记缓冲区（不参与梯度，但要随 ``to()`` 迁移）。"""
        if value is not None:
            self._buffers[name] = value
            object.__setattr__(self, name, value)
        if not persistent:
            self._non_persistent.add(name)

    def register_parameter(self, name: str, value: Parameter | None) -> None:
        """登记参数。"""
        if value is not None:
            self._parameters[name] = value
            object.__setattr__(self, name, value)

    def add_module(self, name: str, module: Module | None) -> None:
        """登记子模块。"""
        setattr(self, name, module)

    # 前向 ---------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:   # pragma: no cover - 抽象
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    # 遍历 ---------------------------------------------------------------

    def children(self) -> Iterator["Module"]:
        """直接子模块。"""
        yield from self._modules.values()

    def named_children(self) -> Iterator[tuple[str, "Module"]]:
        """带名字的直接子模块。"""
        yield from self._modules.items()

    def modules(self) -> Iterator["Module"]:
        """自身与所有后代。"""
        yield self
        for child in self._modules.values():
            yield from child.modules()

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, "Module"]]:
        """带点号前缀的全部模块。"""
        yield prefix, self
        for name, child in self._modules.items():
            sub = f"{prefix}.{name}" if prefix else name
            yield from child.named_modules(sub)

    def parameters(self) -> Iterator[Parameter]:
        """全部参数。"""
        for value in self._parameters.values():
            yield value
        for child in self._modules.values():
            yield from child.parameters()

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        """带点号键的全部参数。"""
        for name, value in self._parameters.items():
            key = f"{prefix}.{name}" if prefix else name
            yield key, value
        for name, child in self._modules.items():
            sub = f"{prefix}.{name}" if prefix else name
            yield from child.named_parameters(sub)

    def buffers(self) -> Iterator[Tensor]:
        """全部缓冲区。"""
        yield from self._buffers.values()

    def state_dict(self) -> dict[str, Tensor]:
        """扁平点号键的权重表，与 checkpoint 的命名一致。

        Returns:
            ``{"layers.0.attn.q_proj.weight": Tensor, ...}``。
        """
        out: dict[str, Tensor] = {}
        for key, value in self.named_parameters():
            out[key] = value
        for key, value in self.named_buffers():
            if key not in self._non_persistent:
                out[key] = value
        return out

    def named_buffers(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        """带点号键的全部缓冲区。"""
        for name, value in self._buffers.items():
            key = f"{prefix}.{name}" if prefix else name
            yield key, value
        for name, child in self._modules.items():
            sub = f"{prefix}.{name}" if prefix else name
            yield from child.named_buffers(sub)

    def load_state_dict(self, state: dict[str, Any], strict: bool = True) -> Any:
        """按点号键灌权重。

        Args:
            state: ``{"layer.weight": Tensor}``。
            strict: 缺键/多键是否报错。

        Returns:
            与 torch 同形的结果对象：``incompatible_keys`` 两个列表。
        """
        want = self.state_dict()
        missing = [k for k in want if k not in state]
        unexpected = [k for k in state if k not in want]
        if strict and (missing or unexpected):
            raise KeyError(f"missing={missing[:4]} unexpected={unexpected[:4]}")
        for key, value in state.items():
            target = want.get(key)
            if target is None:
                continue
            src = value.data if isinstance(value, Tensor) else np.asarray(value)
            target.data[...] = src.reshape(target.shape)
        return types.SimpleNamespace(missing_keys=missing, unexpected_keys=unexpected)

    # 模式开关（都是空实现） ---------------------------------------------

    def to(self, *args: Any, **kwargs: Any) -> "Module":
        """递归迁 dtype / 设备。"""
        dtype_ = None
        for item in args:
            if hasattr(item, "name") and hasattr(item, "np_dtype"):
                dtype_ = item
        dtype_ = kwargs.get("dtype", dtype_)
        for name, value in list(self._parameters.items()) + list(self._buffers.items()):
            migrated = value.to(dtype_) if dtype_ is not None else value
            self._parameters[name] = migrated if name in self._parameters else None
            self._buffers[name] = migrated
            object.__setattr__(self, name, migrated)
        for child in self._modules.values():
            child.to(*args, **kwargs)
        return self

    def float(self) -> "Module":
        """递归转 fp32。"""
        return self.to(float32)

    def half(self) -> "Module":
        """递归转 fp16。"""
        return self

    def bfloat16(self) -> "Module":
        """递归转 bf16。"""
        return self.to(bfloat16)

    def cuda(self, index: int | None = None) -> "Module":
        """空实现，保留接口。"""
        return self

    def cpu(self) -> "Module":
        """空实现，保留接口。"""
        return self

    def eval(self) -> "Module":
        """切推理模式。"""
        for module in self.modules():
            module.training = False
        return self

    def train(self, mode: bool = True) -> "Module":
        """切训练模式。"""
        for module in self.modules():
            module.training = mode
        return self

    def requires_grad_(self, flag: bool = True) -> "Module":
        """空实现。"""
        return self

    def apply(self, fn: Any) -> "Module":
        """对每个子模块调用 ``fn``。"""
        for module in self.modules():
            fn(module)
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        """空实现。"""
        return None

    def extra_repr(self) -> str:
        """附加描述。"""
        return ""


class Identity(Module):
    """恒等模块。"""

    def forward(self, x: Any) -> Any:
        return x


class ModuleList(Module):
    """模块容器，支持下标、迭代与 ``append``。"""

    def __init__(self, modules: Iterable[Module] | None = None) -> None:
        super().__init__()
        self._index: list[Module] = []
        for module in modules or ():
            self.append(module)

    def append(self, module: Module) -> "ModuleList":
        """追加一个模块。"""
        self._modules[str(len(self._index))] = module
        self._index.append(module)
        return self

    def extend(self, modules: Iterable[Module]) -> "ModuleList":
        """批量追加。"""
        for module in modules:
            self.append(module)
        return self

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._index)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return ModuleList(self._index[index])
        return self._index[index]

    def __setitem__(self, index: int, module: Module) -> None:
        self._index[index] = module
        self._modules[str(index)] = module


class Linear(Module):
    """``y = x @ W^T + b``，权重按 torch 的 ``[out, in]`` 布局。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device: Any = None, dtype: Any = None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(zeros((out_features, in_features), dtype=dtype or float32))
        self.bias = Parameter(zeros((out_features,), dtype=dtype or float32)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        """前向。"""
        out = matmul(x, self.weight.transpose(-1, -2))
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:                            # pragma: no cover - 展示
        return f"in_features={self.in_features}, out_features={self.out_features}"


class Embedding(Module):
    """查表。"""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Any = None,
                 device: Any = None, dtype: Any = None) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = Parameter(zeros((num_embeddings, embedding_dim), dtype=dtype or float32))

    def forward(self, index: Tensor) -> Tensor:
        """按 id 取行。"""
        idx = index.data.astype(np.int64)
        return Tensor(self.weight.data[idx], self.weight.dtype, self.weight.device)


class LayerNorm(Module):
    """RMSNorm 之外的规范化层；推理链里只在个别分支用到。"""

    def __init__(self, normalized_shape: int | tuple[int, ...], eps: float = 1e-5,
                 elementwise_affine: bool = True) -> None:
        super().__init__()
        shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.normalized_shape = shape
        self.eps = eps
        if elementwise_affine:
            self.weight = Parameter(ones(shape))
            self.bias = Parameter(zeros(shape))

    def forward(self, x: Tensor) -> Tensor:
        """前向：减均值、除标准差。"""
        axes = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        mean = x.mean(dim=axes, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=axes, keepdim=True)
        out = (x - mean) / (var + self.eps).sqrt()
        if "weight" in self._parameters:
            out = out * self.weight + self.bias
        return out


# ---------------------------------------------------------------------------
# functional
# ---------------------------------------------------------------------------


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """``x @ W^T + b``。"""
    out = matmul(x, weight.transpose(-1, -2))
    return out if bias is None else out + bias


def silu(x: Tensor) -> Tensor:
    """``x * sigmoid(x)``。"""
    return x * (1.0 / (1.0 + (-x).exp()))


def gelu(x: Tensor, approximate: str = "none") -> Tensor:
    """GELU（tanh 近似与精确式都实现）。"""
    v = x.data.astype(np.float32)
    if approximate == "tanh":
        out = 0.5 * v * (1.0 + np.tanh(math_sqrt_2_over_pi * (v + 0.044715 * v ** 3)))
    else:
        out = 0.5 * v * (1.0 + np.vectorize(_erf)(v / np.sqrt(2.0)))
    return Tensor(out, x.dtype, x.device)


#: GELU tanh 近似里的常数，单独提出来免得在表达式里写一长串。
math_sqrt_2_over_pi = float(np.sqrt(2.0 / np.pi))


def _erf(value: np.ndarray) -> np.ndarray:
    """误差函数（numpy 没有 ``erf``，用 ``math.erf``）。"""
    import math

    return np.array([math.erf(float(v)) for v in np.asarray(value).reshape(-1)],
                    dtype=np.float32).reshape(np.shape(value))


def pad(x: Tensor, pad_spec: tuple[int, ...], mode: str = "constant",
        value: float = 0.0) -> Tensor:
    """按 torch 的 ``(last_dim_left, last_dim_right, ...)`` 顺序补边。"""
    if len(pad_spec) % 2:
        raise ValueError(f"pad spec must have even length, got {pad_spec}")
    widths = [(int(pad_spec[i]), int(pad_spec[i + 1])) for i in range(0, len(pad_spec), 2)]
    np_mode = {"constant": "constant", "reflect": "reflect", "replicate": "edge"}[mode]
    pad_width = [(0, 0)] * x.ndim
    for depth, (left, right) in enumerate(widths):
        pad_width[x.ndim - 1 - depth] = (left, right)
    out = np.pad(x.data, pad_width, mode=np_mode, constant_values=value)
    return Tensor(out, x.dtype, x.device)


def softmax_(x: Tensor, dim: int = -1) -> Tensor:
    """softmax 的函数形式。"""
    return softmax(x, dim)


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """零中心 RMSNorm：``x / sqrt(mean(x**2) + eps) * w``。

    均方与 rsqrt 在 fp32 里算，增益乘在激活 dtype 上——与内核的顺序一致，所以
    它既能当参考也能当 oracle。``F.rms_norm`` 是 torch 2.4+ 的入口，参考后端
    为了避免版本差异本来手写，这里补上同名函数是为了别的调用方也能用。
    """
    v = x.data.astype(np.float32)
    mean_sq = np.mean(v * v, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(mean_sq + float(eps))
    out = (v * inv).astype(np.float32) * weight.data.astype(np.float32)
    return Tensor(out, x.dtype, x.device)


def softplus(x: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    """``log(1 + exp(beta * x)) / beta``，超过 ``threshold`` 直接线性。"""
    v = x.data.astype(np.float32) * beta
    out = np.where(v > threshold, v, np.log1p(np.exp(np.minimum(v, threshold))))
    return Tensor((out / beta).astype(np.float32), x.dtype, x.device)


def sigmoid_(x: Tensor) -> Tensor:
    """logistic 的函数形式。"""
    return 1.0 / (1.0 + (-x).exp())


functional = types.SimpleNamespace(
    linear=linear,
    silu=silu,
    gelu=gelu,
    pad=pad,
    softmax=softmax_,
    sigmoid=sigmoid_,
    rms_norm=rms_norm,
    softplus=softplus,
)

__all__ = [
    "Embedding", "Identity", "LayerNorm", "Linear", "Module", "ModuleList", "Parameter",
    "functional", "gelu", "linear", "pad", "rms_norm", "silu", "softplus",
]
