"""分块门控 delta 规则：预填充用的线性注意力递推。

所有只依赖分块内部值的部分——衰减矩阵、UT 系统、两次三角求解——是对
``[H * nchunks]`` 的一次批处理算子；只有状态扫描是串行的，而它对状态是仿射
的，所以每个分块只需一次 ``baddbmm``。与递推形式的相对误差约 1e-6
（``diag/check_chunked.py``）。

Module: ``coffin/ops``。纯 ``torch``；预填充时被
``models/qwen35/linear_attn.py`` 使用，绝不许反过来 import ``models/``。

Constraints:
    * 整个内核存在的意义就是保持带宽受限。把批处理阶段拆回逐分块的 Python
      循环，每层每分块要多出约 25 个受 launch 限制的内核，在 n=2048 时实测
      占预填充的 15%——不重跑探针就别「简化」这个结构。
    * 摊平后的批次是头优先的，所以分块 ``i`` 位于批下标
      ``i, nc+i, 2*nc+i, ...``；状态扫描里的跨步子切片依赖这个布局。
    * ``out_state`` 就地写入：调用方跨分块持有循环状态张量，前缀复用会给它
      打快照。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ops.dtypes import NEG_INF

from backends import ops as bops


#: 严格上三角掩码，每个分块长度一份，只建一次而不是每个分块建一次。衰减矩阵
#: 是分块内累积和的差，上三角必须在 exp 之前干掉；每层每个分块都分配并填充这份
#: 掩码纯属额外开销。
_UPPER_MASK: dict[tuple[int, int | None], torch.Tensor] = {}


def _upper_mask(n: int, device: torch.device) -> torch.Tensor:
    """返回缓存的、位于 ``device`` 上的严格上三角 ``[n, n]`` bool 掩码。

    Args:
        n: 分块长度；按长度和设备索引各缓存一份，因为衰减矩阵每个分块都要重建，
            掩码却永远不变。
        device: 掩码必须所在的设备。

    Returns:
        共享的 ``[n, n]`` bool 张量，对角线以上为 ``True``。请当作只读：每层
        每个分块拿到的是同一个对象。
    """
    key = (n, device.index)
    m = _UPPER_MASK.get(key)
    if m is None:
        m = torch.ones(n, n, dtype=torch.bool, device=device).triu(1)
        _UPPER_MASK[key] = m
    return m


def chunked_delta(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
                  beta: torch.Tensor, state: torch.Tensor, out_state: torch.Tensor,
                  chunk: int = 64) -> torch.Tensor:
    """门控 delta 规则的分块形式，供多 token 的预填充使用。

    输出是 token 优先的——``[C, H, dv]``，也就是各层存放激活的布局——所以
    调用方不需要转置拷贝。序列填充在这里吸收掉：用 ``g = beta = 0`` 补齐到
    整数个分块，填充行对状态没有任何贡献（``v_beta = k_beta = 0``，解出来的
    值也是零），同时 ``cum_decay`` 不受影响。

    和参考实现 ``torch_chunk_gated_delta_rule`` 一样分两个阶段：

    1. 所有只依赖分块*内部*值的部分——衰减矩阵、UT 系统、两次三角求解、衰减后
       的 q/k——对全部分块一次性算完，是对 ``[H * nchunks]`` 的一次批处理算子；
    2. 对分块做串行扫描来传递循环状态。这一步对状态是仿射的（见下文），所以
       每分块一次批处理矩阵乘，而输出是对全部分块的四次批处理矩阵乘。

    Args:
        q: 查询，``[H, C, d]``，已做 L2 归一化并缩放。
        k: 键，``[H, C, d]``，已做 L2 归一化。
        v: 值，``[H, C, dv]``。
        g: 逐 token 的对数衰减，``[H, C]``。
        beta: 逐 token 的写入门，``[H, C]``。
        state: 从更早分块带进来的循环状态，``[H, dk, dv]``。
        out_state: 更新后状态的目标，``[H, dk, dv]``。就地写入，调用方的状态
            张量得以保留，前缀复用要打快照的就是它。
        chunk: 每个分块的 token 数。决定批处理系统的形状；默认值是探针实测时
            用的长度。

    Returns:
        输出，``[C, H, dv]``；加了填充时会被截回真实 token 数。

    Raises:
        RuntimeError: ``state``/``out_state`` 在头轴或 head_dim 轴上和
            ``q``/``v`` 对不上，由矩阵乘暴露出来。调用方必须按喂进来的同一批头
            来定状态的尺寸。
    """
    H, C, dk = q.shape
    dv = v.shape[-1]
    nc = (C + chunk - 1) // chunk
    if nc * chunk != C:
        pad = nc * chunk - C
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        g = F.pad(g, (0, pad))
        beta = F.pad(beta, (0, pad))
    L = chunk

    # reshape 上的 ``.contiguous()`` 不是装饰。调用方把激活存成 [C, H, d] 并传
    # 转置视图过来，而拆分转置视图的分块维*确实*可以不拷贝就表达——于是 reshape
    # 返回一个带 stride 的四维视图。在它上面做逐元素运算走不到向量化路径，慢约
    # 5x（实测：单个内核 392 us 而不是 79 us，C=2048 时每层 26.1 对 13.8 ms）。
    # 拷贝本身是一次分块遍历（每张量约 7 us），只在需要时才发生。
    qc = q.reshape(H, nc, L, dk).contiguous()
    kc = k.reshape(H, nc, L, dk).contiguous()
    vc = v.reshape(H, nc, L, dv).contiguous()
    cum = g.reshape(H, nc, L).contiguous().cumsum(-1)                 # [H, nc, L]
    pd = (cum.unsqueeze(-1) - cum.unsqueeze(-2)) \
        .masked_fill(_upper_mask(L, q.device), NEG_INF).exp()         # [H, nc, L, L]

    bc = beta.reshape(H, nc, L).contiguous()
    kb = kc * bc.unsqueeze(-1)
    vb = vc * bc.unsqueeze(-1)
    ut = (kb @ kc.transpose(-1, -2)) * pd
    ica = (qc @ kc.transpose(-1, -2)) * pd
    ecum = cum.exp()                                                  # 下面复用两次
    dkb = kb * ecum.unsqueeze(-1)

    # 两次求解共用 ``ut``，而三角求解的成本大致与右端列数成正比，所以求一次逆再
    # 用两次批处理矩阵乘施加，胜过两次各 128 列的 TRSM。数学相同：两种做法都是
    # A^-1 B，差异只在 fp32 舍入。
    #
    # 求解本身走算子表，因为三角求解可不可用是平台事实：CoreX 上每条
    # ``torch.linalg`` 求解路径都会死在 ``dlopen``（它按一个集群上不存在的路径去
    # 找 cuSOLVER），而 ROCm 一次调用就答了。把调用留在这里意味着模型代码里要写
    # `if platform`；放进后端则两个平台跑同一个文件，一字不改。
    bn = H * nc
    ut2 = ut.reshape(bn, L, L)
    eye = torch.eye(L, dtype=ut.dtype, device=ut.device).expand(bn, L, L)
    ut_inv = bops.triangular_solve(ut2, eye)
    nv = torch.bmm(ut_inv, vb.reshape(bn, L, dv)).reshape(H, nc, L, dv)
    kcd = torch.bmm(ut_inv, dkb.reshape(bn, L, dk)).reshape(H, nc, L, dk)

    q2 = qc * ecum.unsqueeze(-1)
    k2 = kc * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    cd = cum[..., -1].exp()                                           # [H, nc]

    # 阶段 2 对状态是仿射的，不只是线性。把 ``v_new_i = nv_i - kcd_i @ st_i`` 代入
    # 状态更新得到
    #     st_{i+1} = (cd_i * I - k2_i^T kcd_i) st_i + k2_i^T nv_i  =  A_i st_i + B_i
    # 于是串行部分每分块只要*一次* baddbmm，而不是三次小 baddbmm 加一次广播乘；
    # 并且——因为此时已知进入每个分块的状态——输出退化成对 [H * nchunks] 的四次
    # 批处理矩阵乘，而非每分块五个算子。在 n=2048 的单层上是 4.3 ms 对逐分块
    # 形式的 9.7 ms（diag/probe_delta_scan_struct.py），FLOPs 约 1.8x，launch 数
    # 只有四分之一——在一颗小内核要 8-15 us 才能启动的芯片上，这个取舍是对的。
    #
    # ``bn`` 在上面（阶段 1）已定义；``A``/``B`` 按头优先摊平，所以分块 ``i``
    # 位于批下标 ``i, nc+i, 2*nc+i, ...``——下面的跨步子切片正是批处理 GEMM 想要
    # 的形态，扫描因此不需要 gather，分块之间也不需要拷贝。
    k2t = k2.transpose(-1, -2).reshape(bn, dk, L)
    A = torch.bmm(k2t, kcd.reshape(bn, L, dk)).neg_()                  # [bn, dk, dk]
    A.diagonal(dim1=-2, dim2=-1).add_(cd.reshape(bn, 1))               # + cd_i * I
    B = torch.bmm(k2t, nv.reshape(bn, L, dv))                          # [bn, dk, dv]

    sts = torch.empty(H, nc, dk, dv, dtype=v.dtype, device=v.device)
    st = state
    for i in range(nc):
        sts[:, i] = st
        st = torch.baddbmm(B[i::nc], A[i::nc], st)
    out_state.copy_(st)

    v_new = nv - torch.bmm(kcd.reshape(bn, L, dk), sts.reshape(bn, dk, dv)) \
        .reshape(H, nc, L, dv)
    out = torch.baddbmm(torch.bmm(q2.reshape(bn, L, dk), sts.reshape(bn, dk, dv)),
                        ica.reshape(bn, L, L), v_new.reshape(bn, L, dv)) \
        .reshape(H, nc, L, dv)
    # 再把分块填充丢掉；调用方只要真实 token。permute 加 reshape 是本函数里唯一
    # 一次重排拷贝，而 [C, H, dv] 的结果让调用方可以免费 reshape 成 [C, Hv, dv]
    # （*视图*，因为 [C, H, dv] 是连续的）。注意 [H, nc, L, dv] 不能直接 reshape
    # 成 [nc*L, H, dv]：那是连续内存上的合法视图，会悄悄把轴重排。
    return out.permute(1, 2, 0, 3).reshape(nc * L, H, dv)[:C]
