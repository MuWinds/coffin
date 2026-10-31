# 把残差相加折进 RMSNorm

> 2026-10-28，`gpu-node-05`，Iluvatar BI-V100，`COFFIN_BACKEND=corex`。
> `../../../PERF_COREX.md` 的 Workstream D。**部分落地：补丁 A 已进，补丁 B 未进。**

## 已落地的部分

`kernels/norm.cu` 里的 `add_rms_norm(x, add, w, out, stride, eps)`、`ops/norm.py`
里的 `RMSNorm.add_residual(x, add, out)` 入口，以及一个调用点：
`blocks.Layer.__call__`（补丁 A）。这消除了这一层两次残差相加里的**第一次**——
每个解码步 32 次启动。

**第二次**相加仍然是单独的 `aten::add`。它*没有*被折叠，下面一节说的就是原因。
在「完成」这个工作流之前先读它。

## 容易弄错的地方：两次相加并不对应同一个 norm

```python
h = self.mix(self.in_ln(x), *ctx)      # 对 OLD 残差的归一化
x = x + h                              # 相加 #1
return x + self.mlp(self.post_ln(x))   # 相加 #2
```

- **相加 #1**（`x = x + h`）由 `post_ln` 消费，就在同一方法的三行之后。它在局部
  折叠，落地的是这个：

  ```python
  normed = self.post_ln.add_residual(x, h, h)
  return x + self.mlp(normed)
  ```

- **相加 #2**（`x + mlp(...)`）由**下一层的 `in_ln`** 消费——或者在最后一层，由
  `lm_head` 投影之前的 `Qwen35.norm` 消费。在 `Layer.__call__` *内部*，它之后
  没有 norm。

### 把相加 #2 折进 `post_ln` 会坏在哪

不是结果变慢，而是结果**错**。`post_ln` 在相加 #2 运行*之前*、对另一个操作数
已经施加到 `x` 上了。把相加 #2 折回 `post_ln`，归一化的就是和 `x + mlp(...)`，
而这一层需要的是和本身；然后下一层的 `in_ln` 又去归一化一个已经归一化过的值。
残差流悄悄不再是残差流：它仍然有限、仍然看着合理，这就是这件事需要说清楚而不是
靠测试发现的原因。在参考后端上实测：相等是 `False`。

每层只允许存在一个残差和。`add_rms_norm` 累加进 `x`，归一化写进*另一个*缓冲区，
正是为了让这个和活过归一化：`x` 是下一层往里加的那个缓冲区。`out` 可以与 `add`
重叠（和消费掉混合器输出之后它就死了，无需分配的写法由此而来），但永远不能与
`x` 重叠。内核包装层用 `is_alias_of` 拒绝 `x`-`out` 重叠，视图和同一个张量都能
抓住。

## 补丁 B：需要做什么

把每层的相加 #2 折进*后一个* norm，这意味着在循环迭代之间携带归一化后的流，而不
是重算它。在 `models/qwen35/model.py` 的 `_forward` 里（也只在那里——`prefill` /
`verify` / `batch` 各自带自己的残差链，不在 23.5 ms 的解码路径上）：

```python
normed = self.layers[0].in_ln(x)
for i, layer in enumerate(self.layers):
    normed = layer.forward_fused(x, normed, ctx)   # 相加 #1 进 post_ln，返回它
    m = layer.mlp(normed)                          # ... 相加 #2 进下一个 in_ln
    nxt = self.layers[i + 1].in_ln if i + 1 < len(self.layers) else self.norm
    normed = nxt.add_residual(x, m, m)             # x += m，normed 写进 m
```

大致是：`Layer` 多一个接收已归一化输入的方法，`_forward` 多一个携带缓冲区和循环
下标。在 4 层上验证过与已发布的写法逐位相同——归一化链和残差流都是 `True`。

**为什么还是推迟了。** 它要重构 `_forward`，而 `_forward` 正是 `capture()` 记录的
那个函数。在那里改动缓冲区拓扑会改变被捕获的图所寻址的内容，而这个驱动上的捕获
曾经卡死过卡（`backends/corex/__init__.py`，`_GRAPH_VERIFIED`）。按策略
`caps.graph` 是 `False`，所以今天服务的是 eager 路径，但这个改动仍无法在不冒丢卡
风险的情况下在卡上测。23.5 ms 的步里约 0.5 ms 不值得冒这个险，补丁 B 只有在同时
做出重新审视捕获的明确决定时才应捡起来。

## 数值：折算是逐位相同，而不只是接近

两个细节使它成立，两者都不是可选项——它们都在 `add_rms_norm_kernel` 里：

1. 在它的平方进入均方之前，和先窄化成激活 dtype。分开的写法把和存进 `x`（一次
   bf16 舍入），第二个内核再把那个舍入后的值读回来，所以一趟里累加未舍入的 fp32
   和会算出不同的方差。增益乘法留在第 2 趟也是同样的理由：折进第 1 趟会改变两次
   舍入的先后顺序。
2. 增益在第 2 趟从 `x` 读回的值上施加，和 `rms_norm_kernel` 完全一样。

在主机侧对着以同样方式驱动的已发布 `rms_norm_kernel` 校验过：两种 dtype、稠密行
和带间距的行、宽度 1…2561、M 1…8，以及补丁 A 实际用到的那种情形——`out` 与
`add` 重叠，全部逐字节相同。零加数逐字节重现 `rms_norm`，所以融合路径和它替代的
那个 norm 是同一个函数。

## 别名规则，按内核实际执行的那样

| 关系 | 允许 | 为什么 |
| --- | --- | --- |
| `out` 与 `add` 重叠 | 允许，**仅限稠密间距** | `add` 在第 1 趟、任何 `out` 写入之前就被消费；一个线程只碰它已经读过的下标 |
| `out` 与 `x` 重叠 | **永远不允许** | 残差和必须活到下一层 |
| `add` 与 `x` 重叠 | 不检查，也不使用 | 会让第 1 趟读到一行被部分更新的行 |

稠密间距这条限制存在的原因是：`add` 用 `stride` 寻址，而 `out` 用 `N` 寻址；
同一块分配上两种间距会让一行的写入落到另一行尚未读取的加数上。

因为 `out` 可以与 `add` 重叠，内核的 `add` 和 `y` 指针故意**不是** `__restrict__`
——`__restrict__` 会授权编译器把 `add` 的一次加载保留到对 `y` 的存储之外，那是一种
单通道的主机侧检查看不到的误编译。`x` 和 `w` 保持 restrict。

解码调用路径 `add_residual(x, h, h)` 用一维 `[hidden]` 操作数走稠密分支：
`_rows(h)` 是一个与 `h` 共享存储的 `[1, 2560]` 视图，它的行间距是 2560，所以
`pitch == N`，放宽的分支适用。`add_residual` 不会因为这一点抛异常。

## 复现这些检查

用到的主机侧 harness 在 `.tmp_cusyn/` 里，已随工作流一起删除；如果要重新审视，值得
重建两样东西，都不需要卡：

- 把 `norm.cu` 当主机 C++ 编译（`.tmp_cusyn/rewrite.py` 加 `cuda_stub.h`），用
  `blockDim.x = 1` 调 `add_rms_norm_kernel`，让共享树归约正确退化成单通道，然后和
  跑在预先相加的和上的 `rms_norm_kernel` 比较。对残差（逐元素）来说
  `blockDim.x = 64/128` 是成立的，但对 norm 不成立。
- `COFFIN_BACKEND=reference python` 跑一条层形状的链，两种写法都跑；用
  `torch.equal` 比较，不要用 `allclose`。这才是能抓住局部折叠加相 #2 的检查，因为
  它保持有限，只是不一样。
