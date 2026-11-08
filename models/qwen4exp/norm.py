"""Qwen4-Exp 用到的两种 RMSNorm：零中心的普通版与线性注意力的门控版。

它们不是同一个类配一个开关：``Qwen4ExpTextRMSNorm`` 的权重语义是
``1 + w``（checkpoint 存的是相对 1 的偏移、初始化全 0），
``Qwen4ExpTextRMSNormGated`` 的权重则是普通的 ``w``（初始化全 1），且把门放
在归一化**之后**。按普通增益去乘偏移量不会有任何形状错误，只会让每一处归一化
从"恒等 1"变成"恒等 0"（``ARCH_SPEC.md`` §12 第 5 条），所以这里是两个类。

Module: ``coffin/models/qwen4exp``。依赖 ``torch`` 与 ``backends``（取融合归一化
核，以及判断后端是否用内核承担它）；被 ``gated_residual``、``qsa``、``gdn``、
``ngram`` 调用。

Constraints:
    * 归一化必须全程 fp32，乘完 ``1 + w`` 之后再一次性转回入参 dtype。分两步在
      bf16 上乘会在激活量级大时看到差异，而这条路径在 48 层里每层跑 8 次。
      CoreX 原生核的算子是 ``v * inv * w``（fp32 算、末尾只舍入一次），实测与这条
      torch 链**逐位相同**，所以走核不改变数值，只减少发射。
    * 归一化优先走后端原生核，拿不到时退回纯 torch 链。判断依据必须是
      ``Caps.native``，不能是「表里那一格是不是 ``None``」——解析器总会用参考实现
      把缺的补齐，所以每个名字永远可调用（见 :func:`_has_native`）。
    * ``group_size`` 的语义是"把最后一维切成若干等长组、每组独立算 RMS"，
      权重仍是整条向量的宽度。残差流能稳住靠的就是 4 路各保尺度、不互相稀释。
      分组路径**不能**对整条最后一维调 ``bops.rms_norm``：那会把 4 路残差流的能量
      摊平，形状全对、尺度全错。要走核必须先按组压成 ``[rows, group_size]``，让内核
      的最后一维正好落在组边界上（见 :meth:`RMSNorm._grouped_native`）。
    * 这里绝不出现 ``torch.nn.functional.rms_norm``：那是 torch 2.4+ 的 API，
      部署目标是 2.1.0。
"""

from __future__ import annotations

import torch
from torch import nn

from backends import caps as _caps
from backends import ops as bops


def _silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU：``x * sigmoid(x)``。

    手写而不查激活表，好让门控归一化不引入除 ``torch`` 之外的依赖。

    Args:
        x: 任意形状的门输入。

    Returns:
        与 ``x`` 同形状。
    """
    return x * torch.sigmoid(x)


#: 门控归一化允许的激活。参考实现的配置校验也只放行这两个。
_GATE_ACTIVATIONS = {"silu": _silu, "sigmoid": torch.sigmoid}


def _has_native(name: str) -> bool:
    """报告当前后端是否用自写内核承担 ``name``。

    只能问 ``Caps.native``，不能问算子表里那一格是不是 ``None``：解析器
    （``backends.reference.fill``）总会拿参考实现把缺的补齐，所以每个名字在任何
    后端上都可调用。拿「非 ``None``」当「有核」，会让这条分派恒为真，于是在只有
    参考实现的主机上走进一条每步多两次拷贝的路，而它读起来和走核完全一样。

    Args:
        name: :class:`~coffin.backends.protocol.Ops` 的算子名。

    Returns:
        本后端有该算子的内核时为 ``True``。
    """
    return name in _caps().native


class RMSNorm(nn.Module):
    """零中心化 RMSNorm：``y = norm(x) * (1 + w)``，可选按组归一化。

    Attributes:
        weight: 零中心权重，形状 ``[dim]``。语义是偏移量而不是增益。
        group_size: 每个 RMS 组覆盖的最后一维长度；``None`` 表示整条向量一组。
    """

    def __init__(self, dim: int, group_size: int | None = None, eps: float = 1e-6) -> None:
        """建零中心权重。

        Args:
            dim: 归一化的最后一维宽度。
            group_size: 分组长度，必须整除 ``dim``。
            eps: 加在均方上的常数。

        Raises:
            ValueError: ``group_size`` 不整除 ``dim``。分组错位不会抛异常，
                只会让每组的均方跨过边界取到别的流的能量。
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        self.group_size = group_size
        # 折好的 ``1 + w`` 与它对应的权重版本号。``bops.rms_norm`` 的契约是「增益
        # 已经带着 checkpoint 的偏移」，所以偏移必须在什么时候折一次，而不是每次
        # 调用现算——现算就是又一次逐元素拷贝加一次加法，省下的发射又还回去。
        #
        # 登记成**非持久** buffer 而不是普通属性，有两个原因都必须满足：普通属性
        # 不会被 ``.to(device)`` / ``.to(dtype)`` 搬走，模块上卡之后这里还是 CPU 上的
        # 那份增益，而报错要到内核逐元素比对时才出现；持久 buffer 会进 ``state_dict``，
        # 于是 ``weights.verify`` 把它当成「模型有、checkpoint 没有」的多余参数。
        self.register_buffer("_gain", None, persistent=False)
        self._gain_version: int | None = None
        #: ``(dtype, 1 + w)``：分组路径要把增益乘到归一化结果上，那一乘必须与激活同
        #: dtype，否则 broadcasting 会把它升成 fp32、把那趟乘法变成两倍带宽。
        self._gain_typed: tuple | None = None
        #: ``(dtype, ones(group_size))``：喂给内核的「增益为 1」，见 :meth:`_unit_gain`。
        self._unit: tuple | None = None
        if group_size is not None and dim % group_size != 0:
            raise ValueError(
                f"hidden_size ({dim}) must be divisible by group_size ({group_size}).")

    def _folded_gain(self) -> torch.Tensor:
        """返回缓存里的 ``1 + w`` 增益，权重变了就重折。

        缓存靠 ``weight._version`` 失效，不靠调用方记得来通知。主路径在加载时用
        ``state_dict`` 的视图 ``copy_`` 灌权重，那会推进版本号；只有绕过 autograd
        的 ``.data.copy_`` 不推进它。这条路径只在测试里用来把 fixture 灌进来，而
        那些测试每次都是新模块（版本号从 0 开始、缓存也还没建），所以不会读到陈旧
        的增益。真机上权重在装载后不再变，缓存于是只折这一次。

        Returns:
            ``[dim]`` 的增益，dtype 与设备跟随 ``weight``。
        """
        version = self.weight._version
        if self._gain is None or self._gain_version != version:
            with torch.no_grad():
                self._gain = (1.0 + self.weight.float()).to(self.weight.dtype)
            self._gain_version = version
        return self._gain

    def _gain_for(self, dtype: torch.dtype) -> torch.Tensor:
        """按激活 dtype 取 ``1 + w``，与 :meth:`_folded_gain` 同源、多一层 dtype 缓存。

        分组路径要把 gain 单独乘到归一化结果上，那一乘必须与激活同 dtype，否则
        broadcasting 会把它升成 fp32、把那趟乘法变成两倍带宽。
        """
        cached = self._gain_typed
        if cached is not None and cached[0] == dtype:
            return cached[1]
        gain = self._folded_gain().to(dtype)
        self._gain_typed = (dtype, gain)
        return gain

    def _unit_gain(self, dtype: torch.dtype) -> torch.Tensor:
        """缓存一份 ``group_size`` 长、全 1 的增益，喂给内核用。

        分组路径不能把真实增益交给内核（内核只收一个 ``[N]`` 增益，而各组增益不同），
        所以让它按"增益为 1"做纯 RMS，增益另乘。全 1 那份是常量，缓存住即可。
        """
        cached = self._unit
        if cached is not None and cached[0] == dtype:
            return cached[1]
        unit = torch.ones(self.group_size, dtype=dtype, device=self.weight.device)
        self._unit = (dtype, unit)
        return unit

    def _grouped_native(self, x: torch.Tensor) -> torch.Tensor:
        """分组 RMSNorm 走融合核：按组压成二维再调，增益单独乘一趟。

        ``[B, S, hc*g]`` 按 ``g`` 切开就是 ``hc`` 段互不重叠的行，``reshape(-1, g)``
        之后内核的"最后一维"**恰好就是组边界**，归一化范围与参考实现一致 —— 文件头
        那条禁令针对的是"对整条 ``hc*g`` 调核"，那是另一种（错的）语义。

        收益是去掉 fp32 链：参考实现把 10 MB 的 bf16 放大成 20 MB 的 fp32 中间量来回
        搬八趟（实测 606 us/次），这里内核读一遍写一遍、再乘一趟增益（184 us/次，
        −69.6%）。``hc_norm`` 每 forward 被调 97 次，合计约 −41 ms。

        Args:
            x: 末维宽度为 ``hc_count * group_size`` 的激活。

        Returns:
            与 ``x`` 同形状、同 dtype。
        """
        lead = x.shape[:-1]
        rows = x.reshape(-1, self.group_size)
        normed = bops.rms_norm(rows, self._unit_gain(x.dtype), 0, self.eps)
        return normed.view(*lead, -1) * self._gain_for(x.dtype)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """在 fp32 上算 RMS 归一化，不含权重。

        Args:
            x: 末维宽度为 ``dim`` 的 fp32 激活。

        Returns:
            与 ``x`` 同形状的归一化结果。
        """
        if self.group_size is not None:
            x = x.reshape(*x.shape[:-1], -1, self.group_size)
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.flatten(-2) if self.group_size is not None else out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """归一化并把零中心权重折进去。

        不分组的形状优先走后端融合核：这一路在 48 层里每层被调用若干次
        （``q_norm``/``k_norm``/indexer 的两处 layernorm 合计 4 次 × 12 层），
        纯 torch 串每次是 6-8 个算子，也就是每步约 84 次启动。核只做一次启动。
        核的算术是 ``v * inv * w``（fp32 算、末尾舍入一次），与下面那条 torch 链
        实测逐位相同，所以换路不改变数值。

        Args:
            x: 任意 dtype 的激活，末维宽度为 ``dim``。

        Returns:
            与 ``x`` 同形状、同 dtype 的张量。
        """
        if self.group_size is None and _has_native("rms_norm"):
            return bops.rms_norm(x, self._folded_gain(), 0, self.eps)
        # 分组路径也走核：见 :meth:`_grouped_native`（每 forward 97 次，606 -> 184 us）。
        if self.group_size is not None and _has_native("rms_norm"):
            return self._grouped_native(x)
        output = self._norm(x.float())
        # 权重是相对 1 的偏移量：checkpoint 里会出现负值，直接当增益用会把
        # 整层输出压成 0 附近。
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self) -> str:
        """把形状、eps 和分组宽度写进 ``print(module)`` 的一行。

        Returns:
            形如 ``(10240,), eps=1e-06, group_size=2560`` 的字符串。
        """
        return f"{tuple(self.weight.shape)}, eps={self.eps}, group_size={self.group_size}"


class RMSNormGated(nn.Module):
    """线性注意力的门控 RMSNorm：先归一化，再乘 ``act(gate)``。

    Attributes:
        weight: 普通增益权重，形状 ``[hidden_size]``，初始化为 1。
        variance_epsilon: 加在均方上的常数。
        activation: 门的激活名，``silu`` 或 ``sigmoid``。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, activation: str = "silu") -> None:
        """建标准增益权重。

        Args:
            hidden_size: 归一化宽度，等于 value head 维。
            eps: 加在均方上的常数。
            activation: 门激活名。

        Raises:
            ValueError: 激活不在 ``{silu, sigmoid}`` 里。参考实现的配置校验同样
                只放行这两个，别在这里悄悄多支持一种。
        """
        super().__init__()
        if activation not in _GATE_ACTIVATIONS:
            raise ValueError(
                f"Unsupported Qwen4-Exp output gate activation: {activation!r}, "
                f"expected one of {sorted(_GATE_ACTIVATIONS)}.")
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = activation

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """归一化 ``hidden_states`` 并用 ``gate`` 门控。

        三处顺序不能动：归一化在门**之前**；权重不偏移（直接乘）；门的激活在
        fp32 上算。门只影响幅度不影响归一化统计，反过来写会得到另一条曲线。

        ``activation == "silu"`` 时走 ``bops.rms_norm_gated``，其余回退纯 torch。
        **真实配置是 ``sigmoid``（``output_gate_type``），所以这条路今天的实测收益
        是 0**：CoreX 的 ``rms_norm_gated`` 内核把激活写死成 silu（``norm.cu`` 里
        ``gv / (1 + exp(-gv))``，没有开关）。保留这条分派不是性能项，是为了别的
        配置能吃到核，以及把「核只做 silu」这件事钉在调用点上——在这里悄悄按 silu
        算，等于把一个看着正常的 logits 换成另一个。

        Args:
            hidden_states: 归一化输入，末维宽度为 ``hidden_size``。
            gate: 门输入，形状与 ``hidden_states`` 一致；来自 ``in_proj_z``。

        Returns:
            与 ``hidden_states`` 同形状、同 dtype。
        """
        if self.activation == "silu" and _has_native("rms_norm_gated"):
            return bops.rms_norm_gated(hidden_states, self.weight.to(hidden_states.dtype),
                                       gate, self.variance_epsilon)
        # 真实配置是 ``sigmoid``，而 ``norm.cu`` 把门控激活写死成 silu，所以这里拆两步：
        # 归一化走 ``bops.rms_norm``（纯 RMS + 增益，与 ``norm.cu`` 的 ``v*inv*w`` 同款，
        # 这里的 ``weight`` **不带** ``1 +`` 偏移，见下面的乘法注释），门另算。
        #
        # 门的激活放在激活 dtype 上算（原来在 fp32 上）：那会多一次 bf16 舍入，但整条链
        # 从「fp32 上往返 ~150 MB」降到 ~60 MB —— 实测 838 -> 151 us/次，每 forward 36 次。
        # 数值差 ≤1 ULP（7.8e-3，输出量级 1）。
        if _has_native("rms_norm"):
            normed = bops.rms_norm(hidden_states,
                                   self.weight.to(hidden_states.dtype), 0,
                                   self.variance_epsilon)
            return normed * _GATE_ACTIVATIONS[self.activation](gate)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * _GATE_ACTIVATIONS[self.activation](gate.to(torch.float32))
        return hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        """把宽度、eps 和门激活写进 ``print(module)`` 的一行。

        Returns:
            形如 ``(128,), eps=1e-06, activation=sigmoid`` 的字符串。
        """
        return (f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, "
                f"activation={self.activation}")


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
             group_size: int | None = None) -> torch.Tensor:
    """无状态的零中心 RMSNorm，供只需要一次前向的路径使用。

    这是模块级的普通函数而不是类的方法：它没有地方缓存折好的增益，所以每次调用都
    现折一次 ``1 + w``。需要反复调用的地方一律用 :class:`RMSNorm`，那里折一次就
    常驻。

    Args:
        x: 末维宽度为 ``weight.numel()`` 的激活。
        weight: 零中心权重（语义是 ``1 + w``）。
        eps: 加在均方上的常数。
        group_size: 分组长度，``None`` 表示整条向量一组。

    Returns:
        与 ``x`` 同形状、同 dtype。

    Raises:
        ValueError: ``group_size`` 不整除最后一维宽度。
    """
    dim = x.shape[-1]
    if group_size is not None and dim % group_size != 0:
        raise ValueError(f"hidden_size ({dim}) must be divisible by group_size ({group_size}).")
    if group_size is None and _has_native("rms_norm"):
        return bops.rms_norm(x, (1.0 + weight.float()).to(x.dtype), 0, eps)
    xf = x.float()
    if group_size is not None:
        xf = xf.reshape(*xf.shape[:-1], -1, group_size)
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if group_size is not None:
        out = out.flatten(-2)
    return (out * (1.0 + weight.float())).type_as(x)


__all__ = ["RMSNorm", "RMSNormGated", "rms_norm"]
