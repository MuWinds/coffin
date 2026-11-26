"""RMSNorm 三兄弟的 corex 原生实现：形状校验在 Python，计算在 shim ``.so``。

Module: ``coffin/cxrt``。只 import ``rt`` 内的模块与 numpy，
**不 import torch**。它取代 ``backends/corex/__init__.py`` 里的 ``rms_norm`` /
``add_rms_norm`` / ``rms_norm_gated`` 三个适配器。

校验条件与 ``kernels/norm.cu`` 里的 ``TORCH_CHECK`` 逐条对应。这很重要：内核索引时
不做边界检查，一条漏掉的校验会变成「读穿缓冲区」而不是「报参数错误」，而读穿之后的
症状是几层之外的 NaN。

Constraints:
    * 目的地归调用方所有；不传 ``out`` 时才分配。捕获/重放的路径要求地址稳定，所以
      这条规则必须由调用点决定，而不是由本层偷偷分配。
    * ``add_rms_norm`` 的 ``x`` 原地累加。和必须活过归一化（它是下一层的残差输入），
      所以 ``out`` 不许与 ``x`` 共享存储；``out`` 可以与 ``add`` 共享（稠密间距时）。
    * 间距 ``stride`` 的单位是元素，且必须不窄于行宽。窄于行宽会让一行写到另一行
      尚未读取的数据上，症状像数值漂移。
"""

from __future__ import annotations

from cxrt.array import Array
from cxrt.driver import Cudart
from cxrt.dtypes import BF16, DType, F32, I8
from cxrt.errors import ShapeError
from cxrt.memory import Allocator
from cxrt.shim import NormShim

#: 本层接受两种激活 dtype。bf16 走位容器，fp32 是回退路径。
_SUPPORTED: tuple[DType, ...] = (BF16, F32)


def _check_activation(x: Array, what: str) -> None:
    """校验一块激活的形状与 dtype，条件同 ``norm.cu`` 的包装层。

    Args:
        x: 待校验的数组。
        what: 错误消息里的角色名（``"activation"`` / ``"addend"`` / ``"gate"``）。

    Raises:
        ShapeError: 不是二维、最内层步长不是 1，或 dtype 不在支持范围内。
    """
    if x.ndim != 2:
        raise ShapeError(f"{what} must be 2-D [M, N], got {x.describe()}")
    # 不是「连续」：带间距的情形恰恰是更宽缓冲区的一列切片，它永远不连续。
    if x.strides[-1] != 1:
        raise ShapeError(f"{what} must have unit innermost stride, got {x.describe()}")
    if x.dtype not in _SUPPORTED:
        raise ShapeError(f"{what} must be bf16 or fp32, got {x.dtype.name}")


def _pitch(x: Array, stride: int) -> int:
    """解析行间距。

    Args:
        x: 二维激活。
        stride: 调用方给的间距，``0`` 表示取行宽。

    Returns:
        实际行间距，单位元素。

    Raises:
        ShapeError: 间距窄于行宽（行会互相覆写）。
    """
    pitch = int(stride) if stride else x.shape[1]
    if pitch < x.shape[1]:
        raise ShapeError(f"row pitch {pitch} is narrower than the row {x.shape[1]}")
    return pitch


def _check_gain(w: Array, x: Array) -> None:
    """校验增益向量。

    Args:
        w: ``[N]`` 增益。
        x: 同形状的激活，用来对 dtype 与宽度。

    Raises:
        ShapeError: 元素数不是 ``N``，或 dtype 与激活不一致。
    """
    if w.numel != x.shape[1]:
        raise ShapeError(f"gain must have N={x.shape[1]} elements, got {w.describe()}")
    if w.dtype != x.dtype:
        raise ShapeError(f"gain dtype {w.dtype.name} != activation {x.dtype.name}")


class RMSNormOps:
    """三个归一化算子的调用面。

    Attributes:
        shim: 已装载的 C ABI 面。
        rt: 运行时句柄。
        alloc: 分配器。
    """

    def __init__(self, shim: NormShim, rt: Cudart, alloc: Allocator) -> None:
        """绑定运行时。

        Args:
            shim: ``rt.shim.load_norm_shim`` 的产物。
            rt: 运行时句柄。
            alloc: 分配器，用来在 ``out=None`` 时分配目的地。
        """
        self.shim = shim
        self.cudart = rt
        self.alloc = alloc

    def _dest(self, out: Array | None, x: Array, m: int, n: int) -> Array:
        """决定目的地：用调用方的，或分配一块稠密的。

        Args:
            out: 调用方给的目的地或 ``None``。
            x: 激活，用来对 dtype。
            m: 行数。
            n: 列数。

        Returns:
            目的地数组。

        Raises:
            ShapeError: 给的目的地不是稠密 ``[M, N]`` 或 dtype 不一致。
        """
        if out is None:
            return Array.empty(self.cudart, self.alloc, (m, n), x.dtype)
        if not out.contiguous or out.shape != (m, n):
            raise ShapeError(f"destination must be contiguous [{m}, {n}], got {out.describe()}")
        if out.dtype != x.dtype:
            raise ShapeError(f"destination dtype {out.dtype.name} != activation {x.dtype.name}")
        return out

    def rms_norm(self, x: Array, w: Array, stride: int = 0, eps: float = 1e-6,
                 out: Array | None = None, stream: int | None = None) -> Array:
        """``y = x * rsqrt(mean(x^2) + eps) * w``，每行一个 block。

        Args:
            x: ``[M, N]`` 激活，可以带行间距。
            w: ``[N]`` 增益，已经带着 checkpoint 的 ``1 + w`` 偏移。
            stride: 行间距，单位元素；``0`` 表示 ``N``。
            eps: 方差下限。
            out: ``[M, N]`` 稠密目的地；``None`` 时分配。
            stream: 流句柄；``None`` 表示默认流。

        Returns:
            ``out``，便于调用链串接。

        Raises:
            ShapeError: 见各校验函数。
        """
        _check_activation(x, "activation")
        _check_gain(w, x)
        m, n = x.shape
        pitch = _pitch(x, stride)
        dest = self._dest(out, x, m, n)
        self.shim.rms_norm(x.ptr, w.ptr, dest.ptr, m, n, pitch, float(eps),
                           x.dtype.code, stream or 0)
        return dest

    def add_rms_norm(self, x: Array, add: Array, w: Array, stride: int = 0,
                     eps: float = 1e-6, out: Array | None = None,
                     stream: int | None = None) -> Array:
        """``x += add``（原地），``out = rms(x) * w``，一次启动。

        Args:
            x: ``[M, N]`` 残差流，**原地累加**。
            add: ``[M, N]`` 待累加的混合器输出，间距同 ``x``。
            w: ``[N]`` 增益。
            stride: ``x``/``add`` 的行间距，单位元素；``0`` 表示 ``N``。
            eps: 方差下限。
            out: ``[M, N]`` 稠密目的地；``None`` 时分配。可以与 ``add`` 重叠。
            stream: 流句柄。

        Returns:
            ``out``。

        Raises:
            ShapeError: 形状、dtype、间距或别名关系与内核矛盾。
        """
        _check_activation(x, "activation")
        _check_activation(add, "addend")
        if add.shape != x.shape:
            raise ShapeError(f"addend must match the activation: {add.describe()} vs {x.describe()}")
        if add.dtype != x.dtype:
            raise ShapeError(f"addend dtype {add.dtype.name} != activation {x.dtype.name}")
        _check_gain(w, x)
        m, n = x.shape
        pitch = _pitch(x, stride)
        dest = self._dest(out, x, m, n)
        # ``out`` 可以与 ``add`` 重叠：加数在第一趟就被完全消费，早于任何对 ``out`` 的
        # 写入。只允许稠密间距，因为 ``add`` 用 stride 寻址而 ``out`` 用 N，同一块
        # 分配上两种间距会让一行的写入落到另一行尚未读取的加数上。
        if dest.same_storage(add):
            if pitch != n:
                raise ShapeError("destination may alias the addend only at a dense row pitch")
            if not add.contiguous:
                raise ShapeError(f"aliased addend must be contiguous, got {add.describe()}")
        elif dest.same_storage(x):
            raise ShapeError("destination must not alias the residual: the sum stays live")
        self.shim.add_rms_norm(add.ptr, x.ptr, w.ptr, dest.ptr, m, n, pitch, float(eps),
                               x.dtype.code, stream or 0)
        return dest

    def rms_norm_gated(self, x: Array, w: Array, gate: Array, eps: float = 1e-6,
                       out: Array | None = None, stream: int | None = None) -> Array:
        """``y = rms(x) * w * silu(gate)``，按最后一维归一化。

        Args:
            x: ``[M, N]`` 激活，必须稠密。
            w: ``[N]`` 按头的增益。
            gate: ``[M, N]`` 门，稠密，dtype 同 ``x``。
            eps: 方差下限。
            out: ``[M, N]`` 目的地；``None`` 时分配。
            stream: 流句柄。

        Returns:
            ``out``。

        Raises:
            ShapeError: 不连续、形状不一致或 dtype 不受支持。
        """
        _check_activation(x, "activation")
        if not x.contiguous:
            raise ShapeError(f"gated activation must be contiguous, got {x.describe()}")
        _check_activation(gate, "gate")
        if not gate.contiguous:
            raise ShapeError(f"gate must be contiguous, got {gate.describe()}")
        if gate.shape != x.shape:
            raise ShapeError(f"gate shape must match the activation: {gate.describe()} vs {x.describe()}")
        if gate.dtype != x.dtype:
            raise ShapeError(f"gate dtype {gate.dtype.name} != activation {x.dtype.name}")
        _check_gain(w, x)
        m, n = x.shape
        dest = self._dest(out, x, m, n)
        self.shim.rms_norm_gated(x.ptr, w.ptr, gate.ptr, dest.ptr, m, n, float(eps),
                                 x.dtype.code, stream or 0)
        return dest

    def _quant_dests(self, out_q: Array | None, out_s: Array | None, x: Array) -> tuple[Array, Array]:
        """决定 int8 目的地与每行 scale 目的地。

        Args:
            out_q: 调用方给的 ``[M, N]`` int8 目的地或 ``None``。
            out_s: 调用方给的 ``[M]`` fp32 scale 目的地或 ``None``。
            x: 激活，用来取行列数。

        Returns:
            ``(out_q, out_s)``。

        Raises:
            ShapeError: 形状或 dtype 与内核约定不一致。
        """
        m, n = x.shape
        if out_q is None:
            out_q = Array.empty(self.cudart, self.alloc, (m, n), I8)
        elif not out_q.contiguous or out_q.shape != (m, n) or out_q.dtype != I8:
            raise ShapeError(
                f"int8 destination must be contiguous [{m}, {n}] int8, got {out_q.describe()}")
        if out_s is None:
            out_s = Array.empty(self.cudart, self.alloc, (m,), F32)
        elif out_s.numel != m or out_s.dtype != F32:
            raise ShapeError(
                f"scale destination must be fp32 with {m} values, got {out_s.describe()}")
        return out_q, out_s

    def rms_norm_i8(self, x: Array, w: Array, stride: int = 0, eps: float = 1e-6,
                    out_q: Array | None = None, out_s: Array | None = None,
                    threads: int = 0, stream: int | None = None) -> tuple[Array, Array]:
        """``q, s = quantize(rms(x) * w)``，一次启动。预填充 W8A8 的激活入口。

        与「``rms_norm`` 写 bf16 再接 ``quant_act_i8``」逐位相同：中间那排归一化值
        舍入成激活 dtype 之后才进 absmax，量化仍走 round 后夹到 ``[-127, 127]``。

        Args:
            x: ``[M, N]`` 激活，可以带行间距。
            w: ``[N]`` 增益，已经带着 checkpoint 的 ``1 + w`` 偏移。
            stride: 行间距，单位元素；``0`` 表示 ``N``。
            eps: 方差下限。
            out_q: ``[M, N]`` int8 稠密目的地；``None`` 时分配。
            out_s: ``[M]`` fp32 每行 scale 目的地；``None`` 时分配。
            threads: 每块线程数；``0`` 表示由内核按 ``M`` 分档（``i8_fused_threads``）。
            stream: 流句柄。

        Returns:
            ``(out_q, out_s)``。

        Raises:
            ShapeError: 形状、dtype 或间距与内核矛盾。
            RtError: LDS 装不下这一行（``N`` 过大时要改用 ``rms_norm + quant_act_i8``）。
        """
        _check_activation(x, "activation")
        _check_gain(w, x)
        m, n = x.shape
        pitch = _pitch(x, stride)
        q, s = self._quant_dests(out_q, out_s, x)
        self.shim.rms_norm_i8(x.ptr, w.ptr, q.ptr, s.ptr, m, n, pitch, float(eps),
                              int(threads), x.dtype.code, stream or 0)
        return q, s

    def add_rms_norm_i8(self, x: Array, add: Array, w: Array, stride: int = 0,
                        eps: float = 1e-6, out_q: Array | None = None,
                        out_s: Array | None = None, threads: int = 0,
                        stream: int | None = None) -> tuple[Array, Array]:
        """``x += add``（原地），``q, s = quantize(rms(x) * w)``，一次启动。

        解码路径的 ``add_residual`` 走这条；``x`` 的累加结果要活过归一化，所以它是
        下一层的残差输入，不能被就地量化覆盖。

        Args:
            x: ``[M, N]`` 残差流，**原地累加**。
            add: ``[M, N]`` 待累加的混合器输出，间距同 ``x``。
            w: ``[N]`` 增益。
            stride: ``x``/``add`` 的行间距；``0`` 表示 ``N``。
            eps: 方差下限。
            out_q: ``[M, N]`` int8 目的地；``None`` 时分配。
            out_s: ``[M]`` fp32 scale 目的地；``None`` 时分配。
            threads: 每块线程数；``0`` 表示按 ``M`` 分档。
            stream: 流句柄。

        Returns:
            ``(out_q, out_s)``。

        Raises:
            ShapeError: 形状、dtype 或间距与内核矛盾。
            RtError: LDS 装不下这一行。
        """
        _check_activation(x, "activation")
        _check_activation(add, "addend")
        if add.shape != x.shape:
            raise ShapeError(
                f"addend must match the activation: {add.describe()} vs {x.describe()}")
        if add.dtype != x.dtype:
            raise ShapeError(f"addend dtype {add.dtype.name} != activation {x.dtype.name}")
        _check_gain(w, x)
        m, n = x.shape
        pitch = _pitch(x, stride)
        q, s = self._quant_dests(out_q, out_s, x)
        self.shim.add_rms_norm_i8(add.ptr, x.ptr, w.ptr, q.ptr, s.ptr, m, n, pitch,
                                  float(eps), int(threads), x.dtype.code, stream or 0)
        return q, s
