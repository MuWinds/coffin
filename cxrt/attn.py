"""注意力族三个小算子的 C ABI 面：``silu_mul`` / ``silu_mul_quant`` / ``rope_apply``。

Module: ``coffin/cxrt``。只 import ``ctypes`` 与 ``cxrt.*``，**不 import torch**。

为什么只有这三个：它们是这一族里唯一不需要 KV 池布局的算子——输入要么是「一片带行间距
的二维视图」，要么是「一组稠密头」。``attn_paged*`` / ``kv_store`` / ``attn_sparse``
要连块表一起定契约，等布局层立起来再说。

Constraints:
    * 行间距一律**显式**传参，和 C ABI 一致。投影输出的两半是同一个更宽缓冲区的列视图，
      所以「连续」在这里不是合法假设；反过来，从张量对象上推导间距需要读各家族的
      stride 语义，那是 ``tensors.py`` 层的职责，本层不猜。
    * ``out`` 允许与 ``up`` 共享存储（调用点把投影缓冲区交回来就不必分配）。除此外的
      重叠是调用方的 bug，不做检测。
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from cxrt import tensors as T
from cxrt.dtypes import BY_CODE, F32, I8
from cxrt.errors import ShapeError
from cxrt.shim import Shim


class AttnShim:
    """三个出口的薄转发层。"""

    def __init__(self, shim: Shim) -> None:
        """绑定出口。

        Args:
            shim: 已装载的共享库。
        """
        p = ctypes.c_void_p
        i = ctypes.c_int
        ll = ctypes.c_longlong
        self.silu_mul = shim.export(
            "coffin_silu_mul", [p, p, p, ll, ll, ll, ll, ll, i, p])
        self.silu_mul_quant = shim.export(
            "coffin_silu_mul_quant", [p, p, p, p, p, ll, ll, ll, ll, ll, i, p])
        # 门控 delta：``record`` 为空指针就是「不记录」那条（三个入口共用）。
        self.delta = shim.export(
            "coffin_delta", [p, p, p, p, p, p, p, p, i, i, i, i, i, i, p])
        self.delta_commit = shim.export(
            "coffin_delta_commit", [p, p, p, i, i, i, i, p])
        # 线性注意力的头前处理（prologue / batch / 分块预填充）。
        self.delta_prologue = shim.export(
            "coffin_delta_prologue",
            [p] * 12 + [ctypes.c_double, i, i, i, i, p])
        self.delta_prologue_batch = shim.export(
            "coffin_delta_prologue_batch",
            [p] * 12 + [ctypes.c_double, i, i, i, i, i, p])
        self.delta_prep_seq = shim.export(
            "coffin_delta_prep_seq",
            [p] * 10 + [ctypes.c_double, i, i, i, i, i, i, i, i, i, p])
        self.hyper_norm = shim.export(
            "coffin_hyper_norm", [p, p, p, i, i, i, ctypes.c_double, i, p])
        self.kv_store = shim.export(
            "coffin_kv_store", [p, p, p, p, p, p, p, p, p, i, i, i, i, p])
        # 可失败：paged 池的形状不合时返回非 0，调用方回退逐张量写回。
        self.kv_pool_store = shim.export_code(
            "coffin_kv_pool_store",
            [p, p, p, p, p, p, p, p, p, i, i, i, i, i, i, p])
        self.rope_apply = shim.export(
            "coffin_rope_apply", [p, p, p, p, p, p, ll, ll, ll, ll, ll, p])
        self.kv_quant = shim.export(
            "coffin_kv_quant", [p, p, p, ll, ll, ll, i, p])
        # 解码路径的融合链：``record`` 为空指针就是「不记录」那条，``*_code`` 是目的地与
        # 门投影各自的 dtype 码。几何不合时返回非 0，调用方必须退回三算子链。
        self.delta_normed = shim.export_code(
            "coffin_delta_normed",
            [p] * 11 + [ctypes.c_double] + [i] * 6 + [p, i, i, p])
        self.delta_normed_seq = shim.export_code(
            "coffin_delta_normed_seq",
            [p] * 11 + [ctypes.c_double] + [i] * 7 + [ll] * 5 + [p, i, i, p])
        # 注意力切片 + 合并（四个公开入口共用一个出口）。几何不合回非 0。
        self.attn_slice = shim.export_code(
            "coffin_attn_slice",
            [p] * 13 + [ctypes.c_double] + [i] * 12 + [p])
        # QSA 的两条：稀疏注意力（只读选中位置）与块 key（gather+池化+norm+RoPE）。
        self.attn_sparse = shim.export(
            "coffin_attn_sparse", [p] * 7 + [ctypes.c_double] + [i] * 6 + [p])
        self.qsa_block_keys = shim.export(
            "coffin_qsa_block_keys",
            [p] * 8 + [i] * 7 + [ctypes.c_double] + [p])
        self.path = shim.path


def load_attn_shim(lib_path: str | Path) -> AttnShim:
    """装载注意力 shim。

    Args:
        lib_path: ``cxrt.build`` 产出的 ``.so`` 路径。

    Returns:
        绑好三个出口的 :class:`AttnShim`。
    """
    return AttnShim(Shim(lib_path))


class AttnOps:
    """三个小算子的可调用面：校验形状、分配目的地、调 ABI。

    Attributes:
        shim: :class:`AttnShim`。
    """

    def __init__(self, shim: AttnShim) -> None:
        self.shim = shim

    def silu_mul(self, gate: T.TensorLike, up: T.TensorLike, out: T.TensorLike | None = None,
                 *, g_pitch: int = 0, u_pitch: int = 0, o_pitch: int = 0,
                 stream: int | None = None) -> T.TensorLike:
        """``out = silu(gate) * up``。

        Args:
            gate: 门那一半；最后一维是宽度，行间距由 ``g_pitch`` 给。
            up: 另一半，形状与 dtype 同 ``gate``。
            out: 目的地；``None`` 表示按 ``gate`` 的形状与 dtype 分配。
            g_pitch: ``gate`` 的行间距；``0`` 表示等于宽度。
            u_pitch: ``up`` 的行间距；``0`` 表示等于宽度。
            o_pitch: ``out`` 的行间距；``0`` 表示等于宽度。
            stream: 透传的流句柄。

        Returns:
            ``out``。

        Raises:
            ShapeError: 形状、dtype 或间距对不上。
        """
        width, rows = self._triplet(gate, up)
        if out is None:
            out = T.new_empty(gate, T.shape_of(gate), T.code_of(gate))
        self._check_out(out, gate)
        gp, up_, op = (self._pitch(g_pitch or width, width, "gate"),
                       self._pitch(u_pitch or width, width, "up"),
                       self._pitch(o_pitch or width, width, "out"))
        self.shim.silu_mul(T.ptr_of(gate), T.ptr_of(up), T.ptr_of(out), int(rows), int(width),
                           gp, up_, op, T.code_of(gate), _stream(stream))
        return out

    def silu_mul_quant(self, gate: T.TensorLike, up: T.TensorLike,
                       out: T.TensorLike | None = None, out_q: T.TensorLike | None = None,
                       out_s: T.TensorLike | None = None, *, g_pitch: int = 0,
                       u_pitch: int = 0, o_pitch: int = 0,
                       stream: int | None = None) -> tuple[T.TensorLike, T.TensorLike, T.TensorLike]:
        """``out = silu(gate) * up``，并顺手量化成 ``(out_q, out_s)``。

        Args:
            gate: 门那一半。
            up: 另一半。
            out: bf16/fp32 落盘目的地；``None`` 时分配。
            out_q: ``[rows, width]`` 稠密 int8；``None`` 时分配。
            out_s: ``[rows]`` fp32 每行 scale；``None`` 时分配。
            g_pitch: ``gate`` 行间距；``0`` 表示宽度。
            u_pitch: ``up`` 行间距；``0`` 表示宽度。
            o_pitch: ``out`` 行间距；``0`` 表示宽度。
            stream: 流句柄。

        Returns:
            ``(out, out_q, out_s)``。

        Raises:
            ShapeError: 形状、dtype 或目的地布局对不上。
        """
        width, rows = self._triplet(gate, up)
        code = T.code_of(gate)
        if out is None:
            out = T.new_empty(gate, T.shape_of(gate), code)
        self._check_out(out, gate)
        if out_q is None:
            out_q = T.new_empty(gate, (rows, width), I8.code)
        elif T.shape_of(out_q) != (rows, width) or T.code_of(out_q) != I8.code:
            raise ShapeError(
                f"out_q must be dense [{rows}, {width}] int8, got {T.shape_of(out_q)} "
                f"{BY_CODE[T.code_of(out_q)].name}")
        if out_s is None:
            out_s = T.new_empty(gate, (rows,), F32.code)
        elif T.numel_of(out_s) != rows or T.code_of(out_s) != F32.code:
            raise ShapeError(f"out_s must be fp32 with {rows} values")
        gp, up_, op = (self._pitch(g_pitch or width, width, "gate"),
                       self._pitch(u_pitch or width, width, "up"),
                       self._pitch(o_pitch or width, width, "out"))
        self.shim.silu_mul_quant(T.ptr_of(gate), T.ptr_of(up), T.ptr_of(out),
                                 T.ptr_of(out_q), T.ptr_of(out_s), int(rows), int(width),
                                 gp, up_, op, code, _stream(stream))
        return out, out_q, out_s

    def rope_apply(self, q: T.TensorLike, k: T.TensorLike, cos: T.TensorLike,
                   sin: T.TensorLike, q_out: T.TensorLike | None = None,
                   k_out: T.TensorLike | None = None, rotary_dim: int = 0,
                   stream: int | None = None) -> tuple[T.TensorLike, T.TensorLike]:
        """旋转每个 q/k 头最前面的 ``rotary_dim`` 个通道。

        Args:
            q: ``[M, nh, hd]`` 稠密 bf16。
            k: ``[M, nkv, hd]`` 稠密 bf16。
            cos: ``[M, rotary_dim]`` fp32。
            sin: 同 ``cos``。
            q_out: ``[M, nh, hd]`` 目的地；``None`` 时分配。
            k_out: ``[M, nkv, hd]`` 目的地；``None`` 时分配。
            rotary_dim: 要旋转的前导通道数；``0`` 表示取 ``cos`` 的列数。
            stream: 流句柄。

        Returns:
            ``(q_out, k_out)``。

        Raises:
            ShapeError: 形状、dtype 或 ``rotary_dim`` 与内核矛盾。
        """
        qshape = T.shape_of(q)
        kshape = T.shape_of(k)
        if len(qshape) != 3 or len(kshape) != 3:
            raise ShapeError(f"q/k must be [M, heads, hd], got {qshape} and {kshape}")
        if T.code_of(q) != T.BF16C or T.code_of(k) != T.BF16C:
            raise ShapeError("rope_apply only takes bf16 q/k")
        m, nh, hd = qshape
        if kshape[0] != m or kshape[2] != hd:
            raise ShapeError(f"q{tuple(qshape)} and k{tuple(kshape)} disagree about M or hd")
        cshape = T.shape_of(cos)
        rot = int(rotary_dim) or (cshape[1] if len(cshape) == 2 else 0)
        if len(cshape) != 2 or cshape[0] != m or cshape[1] != rot:
            raise ShapeError(f"cos must be [{m}, {rot}] fp32, got {cshape}")
        if T.shape_of(sin) != cshape:
            raise ShapeError("sin must have the same shape as cos")
        if T.code_of(cos) != F32.code or T.code_of(sin) != F32.code:
            raise ShapeError("cos/sin must be fp32 (narrowing them at the table would lose "
                             "angle precision at long context)")
        if rot <= 0 or rot > hd or rot % 2:
            raise ShapeError(f"rotary_dim must be even and no wider than hd={hd}, got {rot}")
        if q_out is None:
            q_out = T.new_empty(q, (m, nh, hd), T.BF16C)
        elif T.shape_of(q_out) != (m, nh, hd):
            raise ShapeError(f"q_out must be [{m}, {nh}, {hd}], got {T.shape_of(q_out)}")
        if k_out is None:
            k_out = T.new_empty(k, (m, kshape[1], hd), T.BF16C)
        elif T.shape_of(k_out) != (m, kshape[1], hd):
            raise ShapeError(f"k_out must be [{m}, {kshape[1]}, {hd}], got {T.shape_of(k_out)}")
        self.shim.rope_apply(T.ptr_of(q), T.ptr_of(k), T.ptr_of(cos), T.ptr_of(sin),
                             T.ptr_of(q_out), T.ptr_of(k_out), int(m), int(nh),
                             int(kshape[1]), int(hd), rot, _stream(stream))
        return q_out, k_out

    # 校验

    def kv_quant(self, x: T.TensorLike, out_q: T.TensorLike | None = None,
                 out_s: T.TensorLike | None = None, *, pitch: int = 0,
                 stream: int | None = None) -> tuple[T.TensorLike, T.TensorLike]:
        """对称 int8 量化：每行一个 fp32 scale，一次启动。

        这是 int8 KV 缓存的**写方**，读方是 ``attn_slice``。算术是契约（``amax`` →
        clamp 到 ``1e-5`` → 先除 127 再除 scale → 就近取偶 → 夹到 ``[-127, 127]``），
        换顺序不会让形状检查失败，只会让注意力输出漂移，所以验收口径是逐位一致。

        Args:
            x: ``[M, N]`` 激活，bf16 或 fp32；最后一维为宽度。
            out_q: ``[M, N]`` 稠密 int8；``None`` 时分配。
            out_s: ``[M]`` fp32 每行 scale；``None`` 时分配。
            pitch: ``x`` 的行间距，单位元素；``0`` 表示宽度。
            stream: 流句柄。

        Returns:
            ``(out_q, out_s)``。

        Raises:
            ShapeError: 形状、dtype 或间距与内核矛盾。
        """
        width, rows = self._triplet(x, x)
        code = T.code_of(x)
        if out_q is None:
            out_q = T.new_empty(x, (rows, width), I8.code)
        elif T.shape_of(out_q) != (rows, width) or T.code_of(out_q) != I8.code:
            raise ShapeError(
                f"out_q must be dense [{rows}, {width}] int8, got {T.shape_of(out_q)}")
        if out_s is None:
            out_s = T.new_empty(x, (rows,), F32.code)
        elif T.numel_of(out_s) != rows or T.code_of(out_s) != F32.code:
            raise ShapeError(f"out_s must be fp32 with {rows} values")
        self.shim.kv_quant(T.ptr_of(x), T.ptr_of(out_q), T.ptr_of(out_s), int(rows),
                           int(width), self._pitch(pitch or width, width, "x"), code,
                           _stream(stream))
        return out_q, out_s

    # 校验

    def _triplet(self, gate: T.TensorLike, up: T.TensorLike) -> tuple[int, int]:
        """检查两个操作数，返回 ``(宽度, 行数)``。

        Args:
            gate: 门那一半。
            up: 另一半。

        Returns:
            ``(width, rows)``：一维操作数算一行。

        Raises:
            ShapeError: 形状、dtype 不一致，或不是 bf16/fp32。
        """
        gshape = T.shape_of(gate)
        ushape = T.shape_of(up)
        if not gshape or ushape != gshape:
            raise ShapeError(f"gate/up must share a shape, got {gshape} and {ushape}")
        code = T.code_of(gate)
        if code not in (T.BF16C, T.F32):
            raise ShapeError(f"operands must be bf16 or fp32, got {BY_CODE[code].name}")
        if T.code_of(up) != code:
            raise ShapeError("gate and up must share a dtype")
        width = gshape[-1]
        rows = 1 if len(gshape) == 1 else T.numel_of(gate) // width
        return width, rows

    def _pitch(self, pitch: int, width: int, what: str) -> int:
        """检查行间距。

        Args:
            pitch: 行间距，单位元素。
            width: 行宽。
            what: 错误消息里的角色名。

        Returns:
            行间距。

        Raises:
            ShapeError: 间距窄于行宽（一行会写到另一行尚未读取的数据上）。
        """
        if pitch < width:
            raise ShapeError(f"{what} row pitch {pitch} is narrower than the row {width}")
        return int(pitch)

    def _check_out(self, out: T.TensorLike, gate: T.TensorLike) -> None:
        """检查目的地形状与 dtype。

        Args:
            out: 目的地。
            gate: 参照张量。

        Raises:
            ShapeError: 形状或 dtype 对不上。
        """
        if T.shape_of(out) != T.shape_of(gate):
            raise ShapeError(f"out{T.shape_of(out)} must match gate{T.shape_of(gate)}")
        if T.code_of(out) != T.code_of(gate):
            raise ShapeError("out must share the operands' dtype")


def _stream(stream: int | None) -> Any:
    """流句柄。

    Args:
        stream: 句柄号或 ``None``。

    Returns:
        ``c_void_p`` 可接受的值。
    """
    return None if stream is None else int(stream)
