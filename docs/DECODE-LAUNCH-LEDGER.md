# Decode 发射账本与融合路径（occamy-1.0 W8A8 / EP4 / 单流）

本文记录 **2026-11-14** 在 gpu-node-17 `sandbox4` 上实测的 decode 发射构成、常驻网格栅栏的
可用性结论，以及把发射压到极值的两条路线与各自的确切障碍。文中的数字全部来自当次
`ixprof`/`pdprof` 采集，不是估算；标「待核」的条目是尚未逐个读源码确认的几何参数。

> **记账口径已被后来的布局收敛取代（2026-11-26）**：文中出现的 `gemv_kn*`
> （`gemv_kn` / `gemv_kn_i8` / `gemv_kn_i8_split` / `gemv_kn_split` …）、
> `moe_grouped_i8_kernel` 的 kn 主体核、以及 `pack.to_kn` / `as_kn` / `nk_weight` 这些
> 名字**在仓库里已经不存在**——`[K, N]` 的内核、`[E, K, N]` 的专家栈内核与所有 kn 转换
> 全部删除，发射账本现在是 nk 原缓冲的 `gemv_nk_i8(_m)` / `moe_grouped_i8_nk_*`。这一节
> 及以下数字按当时的那份代码读，不要拿它当现在的发射数。

复现口径：`EXT=coffin_corex_ops_v54_moelk`、`model=occamy-1.0-W8A8-nkfull`、
`COFFIN_PARALLEL=ep`、`COFFIN_EP_COLLECTIVE=p2p`、`--num-blocks 512`、`--max-num-seqs 1`、
`COFFIN_MOE_I8_NK=0`。探针 `work/p1.py`，日志 `sandbox4/logs/701_p1s.out`。

---

## 1. 当前状态

| 指标 | 当前实测值 |
|---|---|
| 启动次数 / decode 步 | **1061** |
| 步时 | **26.79 ms**（37.33 tok/s） |
| 每层发射 | **26.5** |
| 设备时间合计 | 33764 us |
| prefill（1024 token） | 108.7 ms |
| 生成结果 | `[11751, 11, 264, 3177, 34756, 364, 1141, 25438]`（正常） |
| native ops | 45/45（含本次新增的 `kv_pool_store`） |

日志：`sandbox4/logs/711_p1n.out`。队列脚本 `queue/711_p1n.sh`（本地 `.scratch-perf/p1m.sh`）。

### 1.1 已经落地的融合（五类，共 −230 次/步）

前四类用的是**同一个机制**：分片核写完之后不另起一趟归约，而是用「每列块一个到达计数 +
`__threadfence` + 末块就地收尾 + 计数器自复位」把两趟压成一趟。**零自旋**，所以块数不受
驻留约束，也不会像全网格屏障那样静默死等；计数器按 (设备, 块数) 常驻缓存（`torch::zeros`
只在首次建），每次由末块自复位，所以不额外付一次 memset 发射。

| 融合 | 做法 | 省的发射/步 | 落点 |
|---|---|---|---|
| int8 split-K 归约 | `gemv_kn_i8_split` 末尾就地归约，删掉 `gemv_reduce_i8` | **−80** | `gemv.cu` |
| bf16 split-K 归约 | `gemv_kn_split` 同法，删掉 `gemv_reduce`（router 用） | **−40** | `gemv.cu` |
| 分组投影归约 | `moe_grouped_i8` 末尾就地归约，删掉 `moe_grouped_i8_reduce` | **−40** | `moe_grouped_i8.cu` |
| 分组投影行归并 | `moe_grouped_i8_merge_tokens` 末尾就地归并，删掉 `merge_rows` | **−40** | `moe_grouped_i8.cu` |
| paged 池写槽 | `kv_pool_store` 一次启动替掉 `full_like` + 4×`index_put_` | **−30** | `attn.cu` + `cache/paged_cache.py` |

第 4 项的到达阈值必须用**有效行数**（`offsets[E]` × 片数）而不是整行数：EP4 上一个 token 的
8 个槽位里只有约 1/4 落在本地，其余行在核内整块早退、不计数，用整行数当阈值会让收尾永远攒
不齐，`out` 留着上一层的值——这是融合版第一次跑出乱码的原因，记在这里免得重犯。

第 5 项顺带打通了一条**新增契约算子的完整链路**（`kv_pool_store`）：`backends/protocol.py`
的 `Ops` 加一个字段 → `backends/reference/__init__.py` 加一个纯 torch 参考实现并登记 →
`backends/corex/__init__.py` 加同名适配器（`native_ops` 按同名约定自动收进 `_NATIVE`）→
`.cu` 里写 `COFFIN_BIND`。**`Ops.NAMES` 由字段派生、`native_ops` 只认契约名**，所以漏掉
protocol 那一步会表现为「适配器写了但永远跑参考实现」，且日志只说 native 数没变——踩过一次。
以后要把任意阶段合成一个新算子，走的就是这四步。

算术核对：**1061 ≈ 40 层 × 26.5 次/层**。EP4 的每个 rank 持全部 40 层，pp2 只出现在 prefill
侧，所以 decode 的 1061 次里没有任何一次是「另一段流水」的。这是后面所有结论的基础。

---

## 2. 逐核台账（按 kernel 归类的一步发射与设备时间）

| kernel | calls | us | 占比 |
|---|---:|---:|---:|
| `p2p_combine_kernel` | 40 | 6763.4 | 16.2% |
| `gemv_nk_i8_kernel` | 81 | 5820.5 | 13.9% |
| `moe_grouped_i8_kernel` | 80 | 3341.9 | 8.0% |
| `delta_kernel` | 30 | 2104.4 | 5.0% |
| `gemv_kn_i8_split_kernel` | 80 | 1926.6 | 4.6% |
| `aten::copy_` | 123 | 1563.4 | 3.7% |
| `gemv_kn_split_kernel` | 40 | 1478.5 | 3.5% |
| `moe_router_t<16>` | 40 | 1323.0 | 3.2% |
| `add_rms_norm_kernel` | 79 | 1228.4 | 2.9% |
| `gemv_reduce_i8_kernel` | 80 | 1143.2 | 2.7% |
| `moe_dispatch_kernel` | 40 | 1062.9 | 2.5% |
| `aten::_index_put_impl_` | 40 | 966.1 | 2.3% |
| `elementwise_kernel<512,1>` | 71 | 886.0 | 2.1% |
| `moe_grouped_i8_merge_rows_kernel` | 40 | 757.1 | 1.8% |
| `shared_gate_scale_kernel` | 40 | 755.8 | 1.8% |
| `gather_quant_i8_kernel` | 40 | 690.8 | 1.6% |
| `silu_mul_quant_kernel` | 40 | 650.9 | 1.6% |
| `moe_grouped_i8_reduce_kernel` | 40 | 615.2 | 1.5% |

表内合计 33.08 ms / 41.87 ms（逐核时间含重叠，步时 29.06 ms，比值只用于排序）。

三条读数：

1. **`p2p_combine` 单次 169 us**，历史上同一形状是 19.6–22.1 us。它现在吃掉整步 23% 的设备
   时间，而语义只是「4 个 rank 把各自专家部分求和」。这是当前最大的单点异动，先修它比再融
   合 10 个小核更值。
2. 权重流量按 1.08 GB/步 计，对应有效带宽约 37 GB/s（峰值的 ~6.7%）。**当前不是带宽受限，
   是发射数与依赖链受限**，所以融合的方向是对的。
3. `aten::copy_`(123) + `elementwise_kernel<512,1>`(71) + `_index_put_impl_`(40) = **234 次
   /步（18%）** 根本不是算子，是 Python 侧簿记（KV 槽写入的 4 次 `index_copy_` 占其中 40，
   其余是视图拷贝与 `.add_`）。这一段不需要写内核就能去掉一部分。

---

### 2.1 当前构成（1061 次/步的设备时间前 18 名）

| kernel | calls/step | us | 占设备时间 |
|---|---:|---:|---:|
| `gemv_nk_i8_kernel` | 81 | 5664.9 | 16.8% |
| `moe_grouped_i8_kernel` | 80 | 3582.8 | 10.6% |
| `p2p_combine_kernel` | 40 | 3231.5 | 9.6% |
| `gemv_kn_i8_split_kernel` | 80 | 2349.9 | 7.0% |
| `delta_kernel` | 30 | 2083.6 | 6.2% |
| `gemv_kn_split_kernel` | 40 | 1615.6 | 4.8% |
| `aten::copy_` | 123 | 1521.2 | 4.5% |
| `moe_router_t<16>` | 40 | 1366.4 | 4.0% |
| `add_rms_norm_kernel` | 79 | 1226.6 | 3.6% |
| `moe_dispatch_kernel` | 40 | 1069.5 | 3.2% |
| `elementwise_kernel<512,1>` | 71 | 850.4 | 2.5% |
| `shared_gate_scale_kernel` | 40 | 744.4 | 2.2% |
| `gather_quant_i8_kernel` | 40 | 692.1 | 2.0% |
| `silu_mul_quant_kernel` | 40 | 656.1 | 1.9% |
| `aten::add` | 51 | 621.9 | 1.8% |
| `prologue_kernel` | 30 | 542.0 | 1.6% |
| `attn_slice_kernel` | 10 | 504.5 | 1.5% |
| `modern elementwise_kernel` | 41 | 491.0 | 1.5% |

要点：**发射的大头已经变成 torch 侧簿记**（123 + 71 + 51 + 41 = 286 次、27%），单核设备时间
的大头是 `gemv_nk_i8`（81 次）与 `p2p_combine`（40 次 3.2 ms）。这两列一起说明下一步该往哪
走：先把簿记折进核（发射题），再修 `p2p_combine` 的单次耗时（带宽题）。

---

## 3. 每层的 18 个 kernel 与发射构成

`Layer.step` = in_ln → mixer → post_ln → mlp。30 层 linear-attn、10 层 full-attn。

| 阶段 | linear-attn 层 | full-attn 层 |
|---|---|---|
| in_ln (`add_rms_norm`+quant) | 1 | 1 |
| mixer 投影 | `qkvzb_a` 1 | `qkv` 1 |
| mixer 其余 | `conv1d_silu` 1、`delta_prologue` 1、`delta` 1、门控 `norm` 1、`out_proj` 2（split+reduce） | `rope_apply` 1、`kv_quant` 1、`index_copy_`×4、`attn_slice`(+`attn_combine`)、`o_proj` 2 |
| post_ln | 1 | 1 |
| mlp（`moe_decode_block` 单入口） | 11 | 11 |

`moe_decode_block` 内部的 11 次发射，顺序固定：

```
gemv_kn(router bf16) → moe_router → gemv_kn_i8(shared gate_up) → silu_mul
→ gemv_kn_i8(shared down) → shared_gate_scale → moe_dispatch
→ gather_quant_i8 → moe_grouped_i8(up) → silu_mul_quant
→ moe_grouped_i8_merge_tokens(+add_out) [+ 分片时各带一个 reduce/merge 尾巴]
```

EP 语义约束：本函数只返回**本 rank 专家**的部分，共享专家必须在 40 次 `p2p_combine` 之后
再加，所以「把共享专家折进 merge 核」这条路不存在。

---

## 4. 「20 次」的两种口径

| 口径 | 现状 | 20 的含义 | 可行性 |
|---|---|---|---|
| **每步 20 次** | 1061 | 40 层 × 0.5 次/层 | 只有把整栈（或至少两层一步）收进常驻网格 megakernel 才可能；单层 1 发射的下限是 **40**，每 rank 20 层（真 pp2 decode）的下限是 20，但 `parallel.py` 的 tp/pp/ep 互斥，pp2+ep4 只能是 PD 分离，decode 侧仍是 40 层 |
| **每层 20 次** | ~32.3 | 每层砍掉 ~12 次 | 纯逐阶段融合即可达成（见 §7 路线 A 的前半），不需要 megakernel |

这两个口径对应的工作量差一个数量级，必须选一个再动手。下面两条路线分别对应它们。

---

## 5. 常驻网格栅栏：已验证可用

`backends/corex/kernels/mega.cu`（新增，已编译并在卡上跑通）：

```cuda
__device__ void pinned_barrier(unsigned* counter, unsigned* gen, unsigned nblocks);
```

两段式（计数 + 代次），`atomicAdd` → 非末块 `volatile` 轮询代次 → 末块 `__threadfence()`
后复位计数并 `atomicAdd(gen, 1)`。冒烟结果：

| blocks | threads | us/栅栏 | 代次断言 |
|---:|---:|---:|---|
| 16 | 512 | **1.24** | 全块一致 |
| 16 | 1024 | 1.25 | 全块一致 |
| 1 | 512 | 0.95 | 全块一致 |

三次均正常返回，无挂死。`DEV` 行确认 **Iluvatar BI-V100，sm=16，max_threads_per_multiprocessor
=8192，warp=64 lanes**。

要点：16 块 × 1024 线程 = 16384 < 16 × 8192，**整卡能被 16 个块铺满且全部驻留**，这是常驻
网格不发散的前提；`pinned_barrier` 不保证驻留，所以只允许在「网格数 ≤ 16 × 每 SM 块数」时
使用，且必须像 `moe_dispatch.cu` 注释里说的那样**由主机侧断言**而不是靠运气。

---

## 6. 路线 B 的真正障碍：块几何不可统一（已逐条核实）

单一 `blockDim` 无法同时满足下列内核的硬编码宽度：

| kernel | block | grid | smem | 块内 `__syncthreads` | 对 blockDim 的假设 |
|---|---|---|---|---|---|
| `gemv_nk_i8_kernel` | `warps*32`（8→256；浅 K 时 `wcol=4`） | `(cols, tokens, span)` | `K` 字节 | 无 | `n0=(blockIdx.x*(blockDim.x>>5)+grp)*W` — **宽度写进列映射** |
| `moe_grouped_i8_kernel`(kn) | `I8_COLS=128` | `(cols, tokens, span)` | `K` 字节 | **有**（:192） | 一列一线程，需恰好 128 |
| `moe_grouped_i8_nk_kernel` | `warps*32` | 同上 | `K` 字节 | **有**（:285） | 同上 |
| `moe_dispatch_kernel` | 单块 | 1 | `(2*experts+1)` int | **4 次** | 前缀和需全局同步 |
| `gather_quant_i8_kernel` | `QUANT_THREADS` | `rows` | `K*sizeof(XT)` | **有**（树形归约） | `smax[blockDim.x]` + `blockDim.x>>1` 循环 |
| `add_rms_norm_i8_kernel` | `rms_threads(N)`（2048→1024） | `rows` | `threads*4+N*2` ≤48 KB | **有** | 共享树归约（warp 是 64 lanes，不能用 shuffle 假设 32） |
| `silu_mul` / `silu_mul_quant` | 256 | `(ceil(w/256), rows)` | 0 | 无 | `blockIdx.x*256+tid` |
| `shared_gate_scale_kernel` | 单块 | `rows` | 静态 `tree[256]` | **有** | 树宽 256 |
| `rope_apply` | `hd`(256) | `(hd/threads, M, 2)` | 0 | 无 | 无 |
| `kv_quant` | `kv_quant_threads(N)` | `rows` | 待核 | 待核 | 待核 |

结论：**带 `__syncthreads` 的 body 不能用 `if (tid<vw)` 保护**（部分线程执行屏障是未定义行为），
而 `gemv_nk` / `moe_grouped_nk` 把「每块几个 warp」写进了列映射。所以「物理 blockDim=256 +
每 body 接一个 vw 参数」只对无屏障的 body 成立；对那 5 个有屏障的，必须要么让 body 在物理
宽度下运行（意味着把它的算术改成 256 宽），要么把它内部的并行宽度真正参数化。

这是路线 B 的第一件工作，也是唯一真正的架构障碍——**不是**栅栏，栅栏已经证明能用。

---

## 7. 两条路线

### 路线 A：逐阶段融合（不需要栅栏）

按「同层相邻阶段合并」做，纯主机侧 + 少量 kernel 内部改造：

| # | 动作 | 省的发射 | 状态 | 风险 |
|---|---|---|---|---|
| A1 | `gemv_kn_i8_split` + `gemv_reduce_i8` → 末块收尾（每列组计数，末块做归约，自复位） | −2/层 = **−80** | **已上线** | 低（无自旋，只有 `__threadfence`+`atomicAdd`） |
| A1b | 同一机制用到 bf16 `gemv_kn_split` + `gemv_reduce`（router） | −1/层 = **−40** | **已上线** | 低 |
| A2 | `moe_grouped_i8` + `moe_grouped_i8_reduce` → 同上 | −1/层 = **−40** | **已上线** | 低 |
| A3 | `moe_grouped_i8_merge_tokens` + `merge_rows` → 同上（阈值用 `offsets[E]`×片数） | −1/层 = **−40** | **已上线** | 低 |
| A4 | router 簇：`moe_router` + `moe_dispatch` + `gather_quant` → 1（需一道栅栏，G=max(rows,1)≤16） | −2/层 = **−80** | 待做 | 中（栅栏首次落到真实算子） |
| A5 | shared 支路：`gate_up` + `silu` + `down` + `gate_scale` → 1（需一道栅栏） | −3/层 = **−120** | 待做 | 中 |
| A6 | `post_ln`（`add_rms_norm`，40 次/步）折进 MoE 的第一道栅栏前 | −1/层 = **−40** | 待做 | 低 |
| A7 | KV 槽：`PagedCache.write_slot` 的 4×`index_put_` + `full_like` → 1 个 `kv_pool_store` | −3 × 10 层 = **−30** | **已上线**（另 2 次/层来自预填充的行表写回，走 `write_range`，未动） | 低 |
| A8 | Python 侧簿记（123 `copy_` + 71 elementwise + 51 `add`）按热点逐个折进相邻核 | 最多 **−200** | 待做（需先定位来源） | 中 |
| — | `delta_normed`（prologue+delta+norm 三合一，逐位一致） | −2 × 30 层 = **−60** | **未采用** | 每层步时会变慢，开着它等于用吞吐换发射数 |

A1/A1b/A2/A3/A7 已上线：当前 **1061**。再做完 A4/A5/A6：**1061 → 861（每层 21.5）**，再加
A8 才可能到 700 上下（每层 ~17.5）。**每步 20 只能靠路线 B**；「每层 20」用路线 A 即可达成。

### 路线 B：常驻网格 megakernel

前提是 §6 的几何统一，之后：

| 粒度 | 发射/步 | 需要的栅栏数/步 | 说明 |
|---|---:|---:|---|
| 1 发射/层，网格 G=40 | **40** | 40 × ~8 道 = 320 道 ≈ 0.40 ms | 每层一次发射，跨层仍是 40 次 |
| 整栈 40 层 1 发射，G=16 | **1** | ~320 道 ≈ 0.40 ms | 权重每层 27 MB，40 层 1.08 GB 走一遍；16 块只有 16 个 SM 的带宽 |
| 整栈 40 层 2 发射，G=32 | **2** | 同上 | 每 SM 2 块 |

G 的选择是「驻留保证 vs 带宽」的交换：G=16 时每个 SM 一块，跨 SM 的权重流只有 16 路并发，
按 §2 的 37 GB/s 现状，带宽大概率成为新瓶颈——**megakernel 的下一步一定是 1.08 GB/步的
带宽题，而不是发射题**。

工作量（按 §6 的 5 个屏障 body 折算）：**1–2 周**。顺序：
`add_rms_norm` → `silu/silu_quant` → `gather` → `dispatch` → 两个 GEMV → 两个分组 GEMM →
mixer 的 `conv1d/delta/rope/attn/kv_quant`，每接一个都要在 `mega.cu` 里加一个「整层 driver」
和一个 `mega_layer_*` host 入口，并保持 `_NATIVE` 的同名约定。

---

## 8. 已否决（勿重试）

| 手段 | 否决理由 |
|---|---|
| CUDA graph | `AGENTS.md §1` 禁用；实测 `PING graph=false` |
| `cooperative launch` / 不受约束的全网格屏障 | `moe_dispatch.cu` 头注释判定为静默死等（除非 grid 恰等于 SM 数）；本仓已改为单块 + 共享前缀和 |
| tp/pp/ep 组合 | `parallel.py` 三者互斥，pp+ep 不存在；pp2+ep4 只能是 PD 分离 |
| 批引擎做诊断 | `--max-num-seqs 8` 时 `profile`/`bench` 直接报错（"diagnostic ops need the single-stream engine"） |
| `delta_normed` | 逐位一致但每层慢 5.3–13.1 us（见 A 表末行） |
| 投机解码（批模式） | 0.74x |
| int4 | 净收益 ~1.4x，且 KV 读放大 ×6 才是前置问题 |
| `ixprof -m all` 采 decode 逐核时间 | 逐核 replay 会给四卡会合注入停顿，`807_ixp1` 挂死在 `p2p_combine`；要设备时间只能用不改变执行顺序的手段（见 §10.2） |

---

## 9. 结论

1. 每层的阶段几乎全是顺序依赖，可并行合并的极少——这正是「末块收尾」这类办法性价比最高的
   原因：它不改变并行度，只删掉依赖链上多余的那一次发射。已经用它在四组「分片 + 归约」上
   压成单趟，加上 paged 池写槽，一共 **−230 次/步**。
2. 当前 **1061 次/步、26.79 ms（37.33 tok/s）**，每层 26.5 次，输出正常。
3. 剩下的构成里最大的一块不再是张量核，而是 **torch 侧簿记**：`aten::copy_` 123、
   `elementwise_kernel<512,1>` 71、`aten::add` 51、`modern elementwise` 41，合计 286 次
   （27%）。它们全是逐层的点式小动作，都能用同一条四步链路（protocol → reference →
   corex 适配器 → `.cu`）折进相邻核。
4. 「每步 20 次」只有常驻网格这一条路，且受 §6 的块几何约束；栅栏（`mega.cu`）已经验证可用
   （16 块 × 512 线程，1.24 us/道，代次一致），剩下的是把 5 个带屏障的 body 在 256 宽下重写
   （1–2 周）。
5. 「每层 20 次」可以用路线 A 剩余的 A4–A6 + A8 达成，不需要 megakernel，也不需要牺牲吞吐。
6. `p2p_combine` 仍是最大的单核设备时间（40 次 3231 us，9.6%）；历史水位 19.6–22.1 us/次
   对应约 0.8 ms，这里还压着约 2.4 ms 没兑现。

---

## 10. 主机侧预算：一次 decode 步的 25 ms 花在哪（2026-11-16 实测）

§1–§9 都是按**设备时间**排的账。那一轮之后 `bench` 多了一个 `host_ms_per_step` 字段
（不 sync 之前的主机墙钟），于是同一段解码循环可以量出「主机把发射喂进去要多久」：

| 指标 | 值 | 口径 |
|---|---|---|
| 步时 | **26.42 ms** | `bench n=12`，每步一次 sync |
| 主机下发 | **25.33 ms** | 同上，sync 之前读钟 |
| 两者之差 | 1.09 ms | 设备在最后一批发射之后还要跑的时间 |
| 发射次数 | **1060** | `cudaLaunchKernel` 计数 |
| 平均每次发射 | **23.9 µs** | 25.33 ms / 1060 |

日志 `sandbox4/logs/806_p3a.out`。这个「23.9 µs/次」是把整步主机墙钟摊到 native 调用数上，
**它并不等于一次发射的成本**——§10.4 的微基准量出净发射只要 6 µs，剩下的 19 ms 在调用之间
的 Python 簿记里。下表的 cProfile 只是「CPU 耗在哪些调用里」。


### 10.1 逐调用归因（`cprof` op，6 步，49627 次函数调用）

| 调用 | 次数/步 | 单次 | 备注 |
|---|---:|---:|---|
| `moe_decode_block`（native） | 40 | **96–137 µs** | 29 个 `torch::Tensor` 入参；pybind 逐参转换 + 内部 11 次发射 |
| `ep.all_reduce_sum` | 40 | **54–62 µs** | 13 个入参，其中 8 个张量入参是每步不变的常量 |
| `add_residual`（含 native） | 40 | **~105 µs** | 一条 2048 元素的逐元素加，纯主机簿记占 ~21 µs，其余在 native 调用里 |
| `gemv_nk_i8`（native） | 81 | 12 µs | 真正的矩阵向量乘 |
| `add_rms_norm`（native） | 79 | 8.4 µs | |
| `full_attn.__call__` | 10 | **~430 µs** | 每层 8 次 native + 约 15 次视图/包装调用 |
| `reshape` / `to` / `as_strided` / `view` | 558 / 111 / 42 / 33 | 2–9 µs | 合计 **3.5 ms/步** |

### 10.2 设备侧的逐核表这次不可用（两个独立原因）

1. 框架自带的 `torch.profiler` 在 `806` 上给出 `total_us=35844`、其中内核行合计
   **33014 µs**，比整步 26.42 ms 还大；同时 `gemv_nk_i8` 的 81 次被记成 70 µs/次
   （这个核真正要读的权重不到 1 MB）。两项一起说明这张卡上报的内核时间戳含排队等待，
   **不允许把它们直接当设备时间用**。
2. `ixprof -m all` 在这台卡上是**逐核 replay**（日志里逐条 `Replaying kernel … (done)`）：
   它把每个内核单独重跑一遍来计时。EP4 的 `p2p_combine` 是四卡会合，replay 注入的停顿
   会让另外三个 rank 在会合点上永远等下去。作业 `807_ixp1` 因此挂住、被手动掐掉，
   **这条路径对本仓的 decode 不可用**（见 §8）。

### 10.3 本轮改了什么（两处保留、一处撤回）

| 改动 | 位置 | 效果 |
|---|---|---|
| 贪心回填改在设备内完成（argmax 与写回各一个小内核，整步零 D2H） | `model.decode_step(token=None)` / `greedy_logits` / `engin/prefix_engine.decode` | `Memcpy DtoH` 1 次/步 → **0**；服务路径每步少两次同步 |
| `add_residual` / `RMSNorm.__call__` / `all_reduce_sum` 的视图与校验按缓冲身份缓存 | `ops/norm.py`、`models/qwen35/ep.py` | 40×3 次重复的形状/视图计算变成命中 |
| EP decode 尾部 `out = out + shared` → `out.add_(shared)` | `models/qwen35/moe.py` | **已撤回**：省 0.16 ms，但会改写驱动输出缓冲（见 §10.7） |

口径：改前 `37.55 tok/s`（host 25.52 ms），改后 `37.86 tok/s`（host 25.33 ms），
`ms_per_step` 26.62 → 26.42；清掉死代码后再测一次 `38.06 tok/s`（host 25.21 ms，26.28 ms/步），
日志 `sandbox4/logs/808_p3b.out`。**但这 0.5 tok/s 在后续两次同代码测量里没有复现**
（§10.7 的 37.4–38.1 区间），所以下面 §10.4 算的「本该省多少」和实测对不上是预料之中的。

### 10.4 单次发射的真实成本（微基准 `810_lb2`）

「24.8 µs/次」是把 25.2 ms 摊到 1060 次 native 调用上，混进了 Python 簿记。同一台卡、单进程、
不加载模型的微基准（`sandbox4/work/lbench.py`，日志 `logs/810_lb2.out`）给的是**净发射成本**：

| 调用 | 入参数 | 主机 µs/次 | 含排空 µs/次 |
|---|---:|---:|---:|
| `add_rms_norm`（3 个张量 + 3 个标量） | 6 | **6.09** | 8.60 |
| 同上，行数 4 | 6 | 6.84 | 9.68 |
| `aten::copy_` | 2 | 2.51 | 5.71 |
| `aten::add` | 2 | 5.45 | 6.74 |
| `torch.empty`（`[1,2048]` bf16） | — | 3.34 | 3.36 |

三条结论：

1. **本卡的净发射下限是 2.5–6 µs**（与 §7 引用文档里的 5.75 µs 一致），设备在最后一批之后
   只需再排空 2.5 µs。**没有任何队列反压的迹象**——26 ms 的步时不是设备在拖着主机。
2. 每个 **张量入参** 的 pybind 转换约 **2–3 µs**。这就是 `moe_decode_block`（29 个张量入参）
   在模型里要 96–137 µs、而 3 个张量的 `add_rms_norm` 只要 6 µs 的原因；也是 `all_reduce_sum`
   （8 个张量入参、其中 6 个是每步不变的常量）要 54–62 µs 的原因。
3. 1060 次 native 调用按净成本算只值 **6.4 ms**，而实测主机墙钟是 **25.2 ms**。差额约 19 ms
   不在 native 发射里，而在**调用之间的 Python 簿记**：`cprof` 一步里可见 8271 次 Python 调用
   （`reshape` 558、`to` 111、`as_strided` 42、`view` 33、`index_select` 32、适配器的 dtype/形状
   分派、`_rows`/`_pitch_rows` 这类小工具），按 2–3 µs/次正好凑成那 19 ms。

### 10.5 为什么这一轮的三处缓存只值 0.19 ms

它们删掉的是这 8271 次里的约 700 次（40 次 `_rows`×2 + 40 次 `_check_payload`/切片 + 40 次
分配），按 2–3 µs 算本该值 1.5–2 ms，实测 0.19 ms——因为缓存本身也要做 key 比较
（`(id(x), id(add), id(out))` 元组构造 + `getattr`），而且原先那几处「四次 `_rows`」的成本
被高估了。**结论：单点缓存是错的杠杆**；要拿那 19 ms，必须把整条调用链压成少数几次调用
（§10.6 的 H1/H2 与「每层一个 driver」），而不是给每个小工具加缓存。

### 10.6 60 tok/s 的算术

`60 tok/s` ⇒ **≤16.67 ms/步**，当前 **26.28 ms**，要砍 **9.6 ms**。按 §10.4 的分解，25.2 ms
主机里约 6.4 ms 是 native 发射下限、约 19 ms 是调用之间的 Python 簿记：

| # | 动作 | 省的调用 | 预估 | 状态 |
|---|---|---|---:|---|
| H1 | `moe_decode_block` 改「prepare 句柄」：29 个常量入参在装载时装配成一份句柄，每步只传 3 个张量 | 40 次 × ~80 µs 的入参转换 | 预估 −3.2 ms / **实测 −0.1 ms**（§10.8：入参边际成本只有 0.2–0.3 µs/个，这条预估按旧模型算高了 30 倍） | **已上线**（`v55_moelk`，`817_h1c`） |
| H2 | `all_reduce_sum` 折进 H1 的驱动（槽指针、flag、rank、轮次都是常量），顺带把 bf16 cast 与 `add shared` 一起做 | 40 次调用 + 40 次 cast | **−2.4 ms** | 待做（同上） |
| H3 | 每层一个 driver：把 `linear_attn`/`full_attn` 每层的 8–10 次 native + 约 15 次视图收进一次调用 | 30×~25 + 10×~40 次 Python 调用 | **−4.0 ms** | 待做（同上，工作量最大） |
| H4 | 适配器（`bops.linear`/`gemv`/`norm`）的 dtype 与形状分派按 `(w, act)` 缓存 | 81 + 120 次 × ~4 µs | **−0.8 ms** | 待做 |
| — | A4/A5/A6（§7 的八道栅栏融合） | −240 次发射 ≈ **−0.6 ms**（发射下限口径） | 与 H1–H3 叠加才把发射数带到 700 附近 | 待做 |

H1–H3 合计约 **−9.6 ms** ⇒ 16.7 ms ⇒ 约 60 tok/s，全部落在**主机侧调用结构**上，而且
H1/H2/H3 都要动 C++ 驱动（重编译）。这也是本轮最重要的结论：**这台卡的设备侧不是瓶颈
（§10.4 第 1 条），60 tok/s 是纯主机侧的结构题**，不需要重写内核、不需要 megakernel。

### 10.7 本轮落地与实测

见 §10.3 的表。三次独立测量（同一份代码，节点干净程度不同）：

| 日志 | 步时 | 主机 | tok/s | 备注 |
|---|---:|---:|---:|---|
| `808_p3b` | 26.28 ms | 25.21 ms | **38.06** | 节点干净 |
| `812_gen32` | 26.72 ms | 25.63 ms | 37.42 | 有残留 replay 进程占卡 |
| `813_fin` | 26.64 ms | 25.57 ms | **37.54** | 撤回 `add_` 之后 |

所以本轮的净变化是 **37.33（11-14）→ 37.4–38.1**，量级 **+1%~+2%**，落在 `bench` 自身的
±0.6 tok/s 抖动里。**这一轮的价值不在这个百分比**，而在 §10.4 那次微基准校准出来的成本
模型：它把「25 ms 花在哪」从猜测变成可复算的账（6.4 ms 净发射 + 19 ms 调用间簿记），
并因此把 60 tok/s 的路径改写成了 H1–H3 三件 C++ 事（§10.6）。

撤回的一处：EP decode 尾部原先改成 `out.add_(shared)`（省一次分配，0.16 ms/步）。它在
`812` 的 32 token 生成里对应的是一段尾部重复
（`"…France France in the upper Seine-S左 bank Seine ParisParis Paris"`）；撤回后
`813` 的同一条 prompt 是
`" Paris, a city that has a rich history and culture. Paris is located in the north
north-central France, on the River Seine. It is the the capital"`，32 个 token 全程通顺。
`out` 可能是驱动自己输出缓冲的视图，就地相加会把它改写掉，而那块缓冲下一层还要当输入
读——**为 0.16 ms 拿这个风险不值得**，保持 `out = out + shared`。

测法上的注意事项：`807_ixp1` 被判超时之后 **ixprof 的 replay 进程还活着**（一直占着一张卡
和 31561 端口，`811` 就是被这个端口冲突挡住的），`812` 就是在它旁边跑的。同一份代码在
「旁边有别人的进程占卡」时会慢 2–4%：**报数前要确认节点干净**，否则会把噪声当回归。
清理脚本 `work/cleanup_ixp.sh`。

### 10.8 H1 落地与它把 §10.4 的成本模型打掉了（2026-11-16 23:20 实测）

H1 已经上线：`moe_block.cu` 拆成 `MoeDecodeArgs` + `moe_decode_impl` + 30 参包装，新增
`moe_decode_prepare(26 个张量, top_k, slices, with_shared) -> int64` 与
`moe_decode_run(handle, x) -> out`（句柄表按进程 static map + mutex，**句柄持有张量本体**
防悬垂）；Python 侧装载时装配一次、每步只传 `(handle, x)`。**`.cu` 重编译只要 45.3 s**
（`work/build55.sh`，`814_b55` 日志 `BUILD_S 45.3`），三个入口 `HAS ... True`——这让「再改
一次内核」在本轮时间预算里是可行的，是这次的额外收获。

**两次测量把「零收益」分成两种解释**：

| 日志 | 句柄路径 | 步时 | 主机 | tok/s |
|---|---|---:|---:|---:|
| `815_h1` | **没进去**（静默降级回 30 参） | 26.68 ms | 25.61 ms | 37.49 |
| `817_h1c` | 进去了（标记行 `[moe] handle path on, handle=1..40`） | 26.62 ms | 25.53 ms | 37.57 |

静默降级的原因值得写下来，因为它是一个**通用陷阱**：`models/qwen35/moe.py` 里
`from backends import ops as bops`，`bops` 不是 `backends.corex` 包，而是**算子表实例**；
表只由 `backends/protocol.py` 里 `Ops` 的 dataclass 字段构成（`native_ops(globals())` 按
`Ops.NAMES` 收同名可调用对象）。所以新入口除了在 corex 模块里写同名 wrapper，**还必须在
`Ops` 里加字段**，否则 `getattr(bops, name, None)` 恒为 `None`、路径悄悄退回老形态，而
日志上一片正常。已在 `Ops` 里加 `moe_decode_prepare` / `moe_decode_run` 两个字段。

**H1 实测只值 0.08–0.17 ms，不是 §10.6 预估的 −3.2 ms。** 这把 §10.4 的第 2 条结论修正为：

* §10.4 说的「每个张量入参 2–3 µs」是把 `add_rms_norm` 的 **6.09 µs 总成本**摊到了入参
  上。真正的分解是「固定开销（Python wrapper + pybind 分派 + 发射）约 5 µs + **每个张量
  入参的边际成本约 0.2–0.3 µs**」。29 个入参的边际合计约 5–8 µs，去掉 27 个也就省这
  5–8 µs，40 层乘起来 0.2–0.3 ms。
* 因此 19 ms 的「调用间簿记」**不是入参编组**，而是**解释器层面的操作条数**：8271 次调用
  ×约 3 µs（`reshape`/`view`/`as_strided`/`to` 这类也要真建 TensorImpl）。结论改写为：
  **只有把 Python 层的「操作条数」拿掉才值钱，把参数搬进句柄不值钱。**
* 由此 §10.6 的 H2/H3/H4 预估全部要按「省掉的调用次数」重估，不能按「省掉的入参」算；
  H3 的 −4.0 ms 只有在它真的把每层那 8–10 次 native **调用**合并成一次时才成立。H1 保留
  （已上线、数值逐位相同、结构上给 H2/H3 留了落点），但**它本身到不了 60 tok/s**。

测法补一条：服务端 stdout 走 `occ5/logs/${TAG}_d0.log`，`TAG` 不改就会读到上一次运行的同名
文件（`815` 的第一版就踩了：在 `sandbox4/logs/815_h1.out` 里只看到上一轮 `807` 的 replay
尾巴）。所以 `[moe] handle` 这类服务端标记行要 `grep -a` 那个文件，`probe` 的 `BENCH`/`GEN32`
才在 `sandbox4/logs/<queue名>.out`。

---

## 11. 「每层一次 C++ 调用」：前段挣 0.6–1.0 ms，p2p 段把输出打坏（2026-11-17）

按 §10.8 的结论（**拿掉 Python 操作条数才值钱**），这一轮把一层 MoE 的四段
（`add_residual` → 整块 `moe_decode_run` → `all_reduce_sum` → `+ shared`）压成
`moe_layer_prepare` + `moe_layer_run` 一次调用。

### 11.1 实测：一层省三次往返，只值 0.5 ms

| TAG | ms/步 | 主机 | tok/s | 输出 |
|---|---:|---:|---:|---|
| `001_chk`（改动前） | 25.64 | 24.59 | 39.0 | 正常 |
| `010_f8` / `011_f9` | 25.05 / 25.03 | 23.96 | 39.92 / 39.95 | **乱码** |
| `016_fe` | 25.16 | 24.14 | 39.75 | 乱码 |
| `018_fg` / `021_fj` | 25.75 / 25.11 | 24.64 / 24.07 | 38.83 / 39.82 | 乱码 |
| `022_fk`（只融合前段） | 25.72 | 25.01 | **38.88** | **正常** |

一层省掉三次 Python→C 往返（40 层共 120 次）只换到 **0.5–1.0 ms/步**，与 §10.8 修正后的
成本模型一致（每次往返约 4–8 µs）。**这条路线到不了 60 tok/s**：再省 9 ms 要每步再少掉
约 1000 次 Python 调用，而一步总共只有 1061 次发射。与「设备侧启动开销为主、host 侧几乎
无关」的判断方向一致。

### 11.2 融合路径的输出为什么坏：逐段自检

`moe.py::_checked_step`（`COFFIN_FUSE_CHECK=1` 门控、每进程一次，**跑完返回参考链结果**
所以那一跑的生成质量不受影响）先备份 `x_res` → 跑融合 → 把 `fused` / `_ln_buf` /
`p2p._out[:1]` **各 clone 一份**（不 clone 就会被参考链覆写，差恒为 0）→ 恢复 `x_res` →
跑参考链 → 打各段差。

三次读数（`010_f8` / `021_fj` / `023_fl`，同一层第一步）：

| 段 | 差（max/mean） | 判读 |
|---|---|---|
| `normed` | **0.000e+00 / 0.000e+00** | 归一化逐位相同（gain、`eps=1e-6`、`stride=0` 都对） |
| `sum32` | 1.039e-01 / 3.478e-03 | fp32 累加位与参考链**不是同一套数** |
| `out` | 1.045e-01 / 3.479e-03 | 最终输出同 |

### 11.3 C++ 侧一次性标量 dump：p2p 与最终核**都真的跑了**

`moe_layer_run` 里 `COFFIN_FUSE_DUMP=1` 打四个标量（`023_fl`）：

```
[fuse] rank=0 round=41 grid=1x256 part_dims=[1, 2048] red32_dims=[1, 2048]
       |part|=0.00311737 |red32|=0.00518389 max|red32-part|=0.0167313
       |shared|=0.00151875 |out|=0.00558522 max|out-part|=0.103897
```

* `max|red32-part|`=1.67e-2 ≫ 0 ⇒ **p2p 写了**（`red32` 不是 `part`）；
* `max|out-part|`=0.104 ⇒ **最终核也写了**；
* `|shared|`=1.519e-3 与 Python 侧 `|ref_shared|` 完全一致；
* 但 `|red32|`=5.18e-3 而参考链 `|ref_sum|`=8.26e-3 —— **两套不同的数据**。

`021_fj` 与 `023_fl` 两次的四个数**逐位相同**，说明这是**系统性错位**，不是竞态。

### 11.4 单点切除：坏在 C++ 的 p2p 段？——**这个结论已被 §11.8 推翻，别引用**

> 保留原文只为记录「错在哪一步的推理」，**不要**据此动手。

`moe_layer_run` 加了 `with_p2p` 入参（`0` 时只做前段并早退）。`COFFIN_FUSE_HYBRID=1`
（当时做成了默认）跑 `022_fk`：

```
GEN32 text = " Paris, which is located in the north of the country ... <think> ..."
```

**输出正常**，25.72 ms / 38.88 tok/s。当时的判读是：

1. C++ 前段（`add_rms_norm` + `moe_decode_impl`）**是对的**，且它本身就值 0.6–1.0 ms；
2. 坏掉的只有 **C++ 里直接调 `p2p_combine` + `moe_reduce_add_kernel` 这一环**。

第 1 条是对的（§11.8 仍然成立）；**第 2 条是错的**——它在 11.8 的已知答案探针下被证伪，
真正的差异是「返回常驻缓冲 vs 返回新张量」这种生命周期问题，不是 p2p 本身。

已排除的四个解释：`has_p2p` 没置上（`moe_block.cu:420` 明文 `a.has_p2p = true`，且 dump 里
两者都写了）；形状不对（`part_dims`/`red32_dims` 就是 `[1, 2048]`，槽区偏移 `rank*total` 由
`rows*hidden` 推出，与 Python 同）；`normed` 错（严格为 0）；载荷不同（同一 `ln_normed`）。

剩下最可疑的两条，留给下一轮（都不值得为 0.5 ms 现在动）：

* **rank 号来源**：`a->p2p_rank` 取的是 `int(p2p.rank)`，而 Python 的 `all_reduce_sum` 用
  `self._rank`。两者若不同号，载荷会写进错的槽段——结果确定性地错、且不随运行变化，
  正好符合 11.3 的「两次逐位相同」。
* **`round` 口径**：融合路径用 Python 的 `p2p._round + 1` 并回写，Python 自己的
  `all_reduce_sum` 是「先加后用」。两者在同一条序列上应当一致，但融合路径把「占用一轮」
  和「发布一轮」拆到了两处，值得用一次 `_round` 打点确认。

### 11.5 本轮落地（默认形态）

`models/qwen35/moe.py::layer_step` 默认走**只融合前段**：`moe_layer_run(..., with_p2p=0)`
拿回本 rank 部分和，然后 `self._reduce(...) + self._block_b[-2]` 用参考链算求和与加共享。
`COFFIN_FUSE_FULL=1` 可切回全融合（供继续查 p2p 段）。这不是回退：融合入口、句柄装配、
一维/二维形状跟随全部保留，被切掉的只是实测「值 0.5 ms 但会打坏输出」的那一环。

### 11.6 四个测法陷阱（都在这轮踩过）

1. **节点 Python 是 3.10，本地是 3.12**：f-string 表达式里换行在 3.12 上因 PEP 701 合法，
   在 3.10 上是 `SyntaxError`，而本地 `ast.parse(src)` 默认按 3.12 解析**拦不住**。
   必须 `ast.parse(src, feature_version=(3, 10))`；症状是队列脚本 `rc=70` + 10 s 结束
   （`017_ff`）。
2. **`std::printf("%e", float)` 在这套工具链下把 4 字节 float 当 double 读**：打出来是
   `4.9e-315` 这种 double 次正规数（恰好等于「float 位型落在 8 字节槽低半部」）。
   格式一律走 `c10::str`（iostream 转，不吃变参）。
3. **别直接读 `data_ptr<float>()` 打头几个元素**：临时张量析构后是悬空指针，四个 rank
   会打出完全相同的一串尾数——看起来"四个人算出同一个值"，其实只是同一块被回收的内存。
4. **逐层 `.item()` 会同步**：dump 放在每层每次调用里，bench 从 25 ms/步 掉到 38.27 ms/步
   （`020_fi`，26.13 tok/s）。诊断打点必须一次性（`static bool dumped`）。

### 11.7 对 60 tok/s 的含义

`60 tok/s` ⇒ ≤16.67 ms/步，当前 **25.7 ms**（输出正常那一档），要砍 **9 ms**。§10.6 的
H1–H3 里，H1（句柄）已被 §10.8 证伪（−0.1 ms），H2（把 p2p 折进驱动）就是本轮做的那件事
（−0.5 ms 实测，且当前实现还坏）。**剩下能兑现的只有 H3「每层一次 driver」**——把
`linear_attn`/`full_attn` 每层的 8–10 次 native 与约 15 次视图收进一次调用（预估 −4 ms），
叠加 A4–A6（−0.6 ms）后仍在 21 ms 附近，**到不了 16.67 ms**。

要走完最后 4–5 ms，只剩两条：

* **常驻网格 megakernel**（§9.4 那条「每步 20 次发射」的路，栅栏已验证 1.24 µs/道，
  要把 5 个屏障 body 在 256 宽下重写，**1–2 周**）；
* 用户提的「kernel 在跑时先把下个 kernel 备好」——在本仓对应的就是 CUDA Graph，
  而 graph 是 `AGENTS.md §1` 明令禁用、且运行时 `PING graph=false`（§8），
  **唯一等价形态仍是常驻网格**。

顺带一条没兑现的账：`p2p_combine` 仍是最大单核设备时间（40 次 3231 µs，9.6%），
历史水位 19.6–22.1 µs/次对应约 0.8 ms，**这里压着约 2.4 ms**（§9.6）。它与本轮无关
（本轮动的是主机侧调用结构），但它是唯一一块「已知有 2.4 ms、且不需要重写整个执行模型」
的设备侧余量。

### 11.8 继续深挖：p2p 是对的，自检是坏的，真凶是缓冲生命周期（同日 13:40）

用户要求「别不划算，继续找出来」，于是把 11.4 的结论整个推翻重做。三个新事实：

**(a) 自检里的参考链是假的。** 把融合与参考两条链的头 8 个元素逐位摆出来（`025_fn`）：
`drv` 与 `fu` **八个元素完全相同**。`drv` 是参考链 `_decode_block` 的输出、`fu` 是融合
输出，两者逐位相同只能说明 **`_decode_block` 根本没重算**，它返回的就是融合自己写的那块
缓冲。所以 §11.2/§11.3 里所有的 `sum32`/`out` 差值、以及「p2p 的跨 rank 求和没有生效」
这个结论，**全部作废**——它们比的是融合输出自己。`|ref_part|` 恰好等于 `|out|`（5.585e-3）
就是这条的旁证。

**(b) p2p 完全正确。** 四张卡的 dump 摊在一起（`023_fl`）：

| rank | `|part|` | `|red32|` | `|out|` |
|---|---|---|---|
| 0 | 3.117e-3 | 5.18389e-3 | 5.58522e-3 |
| 1 | 1.064e-3 | 5.18389e-3 | 5.58522e-3 |
| 2 | 0.747e-3 | 5.18389e-3 | 5.58522e-3 |
| 3 | 2.380e-3 | 5.18389e-3 | 5.58522e-3 |

`part` 各不相同（各卡拥有不同专家）、`red32`/`out` **四卡逐位相同**——这正是 all-reduce
正确的特征（各 rank 必须算出同一个全局和）。再上**已知答案探针**（`COFFIN_FUSE_PROBE=1`，
把载荷填成 `rank+1`，正确结果必然全 10，`026_fo`）：

```
[probe] rank=0/1/2/3 round=41 red32[0:4]= 10   （四张卡都是 10）
```

会合、槽区偏移、fp32 累加、四卡一致性，全部正确。**`p2p_combine` 与 `moe_reduce_add_kernel`
这一环没有问题**，不要再往这里投钱。

**(c) 真凶：驱动把自己那块常驻缓冲当结果交出去。** `blocks.step` 把本层输出当 `carry`
交给下一层，而下一层 `_add_then_norm(self.in_ln, x, carry, carry)` 会**把这个张量当输出
缓冲就地覆写**；驱动交出的 `_block_b[-1]` 又是同一层句柄的 `a->out`（`moe_decode_impl` 写
的那块）。混合路径（只融合前段）每次返回新张量，所以从来不出问题。加
`COFFIN_FUSE_CLONE` 交回副本后（`027_fp`）：

```
BENCH 25.20 ms / 24.18 host / 39.68 tok/s
GEN32 " Paris, a city city a."      ← 干净文本
```

**但同一份代码复跑一次（`028_fq`）给的是 `" Paris,,'/'''ss\n\n\"\""`。** 贪心解码本该
确定，两次不同 ⇒ 全融合路径**还剩一个竞态没关掉**，所以目前默认仍走「只融合前段」
（`022_fk` 32 个 token 通顺、25.72 ms / 38.88 tok/s，`024_fm` 25.70 ms / 38.91）。

**这个竞态的最可能位置（下一轮直接改这里）**：槽区在 decode 上是**同一块 8 KB 每轮复用**，
而 `flag` 协议只保证「四卡都写过本轮」（`while (mine[r] < round)`），**不保证「四卡都读完
本轮」**。快的那张卡打完本轮的和就往前走，下一轮的载荷写进同一块区域，慢的那张卡刚看到
flag 放行、正在读——读到的是「自己这轮的前半 + 别人下轮的后半」的混合物。
**修法**：按轮次奇偶双缓冲槽区——写/读都加 `(round & 1) * world * total` 的偏移，
槽区大小翻倍（decode 这一档只有 16 KB，代价为零）。这条同时能解释「探针那一轮是 10、
别的轮次偶尔错」：探针只看第 41 轮，双缓冲没做之前这种错误是概率性的。

复现用的落盘件：队列 `019`–`028`，服务端日志 `occ5/logs/fh..fq_d0..d3.log`。
探针与 dump 全部由环境变量门控、都是一次性（`COFFIN_FUSE_PROBE` / `COFFIN_FUSE_DUMP`），
不带环境变量时零开销、零输出。

---

## 12. 设备时间轴：25.9 ms/步 = 1077 个核 × 24 µs，主机侧已被证伪（2026-11-17 14:20）

§10.8/§11 的主机账本一直建立在 cprof 的 tottime 上。这一轮先用驱动内建的 `[moeT]` 分段
打点把那笔账**推翻**，再用 `ixprof --print-gpu-trace`（非 replay）拿到真正的设备时间轴，
结论与方向都换了。

### 12.1 主机只花 1.4–2.1 µs/层：`moe_decode_run` 的 145 µs 是 profiler 假账

`COFFIN_MOE_TMOE=1`（`032_moet`，5 步 × 40 次调用的平均）：

```
[moeT] calls=40 router=11.30 shared=20.30..30.34 cast=1.40 dispatch=2.09
        gather=1.70 gu=4.63 silu=5.55 merge=5.92 addsh=0.03 total=53.4..84.0
```

`total` 是**一步 40 层合计**的主机时间，即 **1.3–2.1 µs/层**；而 §11 引用的 cprof
给的是 **129–145 µs/次**。两者差两个量级，说明 cprof 对这类 pybind builtin 的每次记账
本身就要 ~0.1 ms——它量的是 profiler，不是驱动。

同一份 `BENCH` 里 `ms_per_step=25.95`、`host_ms_per_step=25.27`：主机时间**等于**步时，
只可能是**被发射队列背压顶住**（队列一旦满，每次 `cudaLaunchKernel` 都要等设备推进）。
所以「主机侧 Python 做的活」总量约 5 ms/步，**主机优化路线的上限就是零点几毫秒**，
这与 §11.1 实测「省 120 次往返只值 0.5 ms」完全一致，也解释了为什么 §10.6 的 H1/H2 都兑现不了。

### 12.2 非 replay 的 ixprof 时间轴在 EP4 上可用

`ixprof --print-gpu-trace` 只取内核起止时间戳、不逐核 replay，因此**不会给四卡会合注入
停顿**（§10.2 挂死的那条是 `-m` 的 replay 模式）。做法：四卡照常起服务，只给 rank0 挂它，
采两步 decode。代价是每核约 +5 µs 插桩（1077 核 ≈ +5.4 ms：traced 净忙 31.1 ms 对真实
25.9 ms，这个 21% 的膨胀在下面的表里逐核减掉）。

落盘：`occ5/logs/tr1_d0.log`（11433 行）；解析脚本 `.scratch-perf/trace.py`（切段/报账）、
`trace2.py`（单次最长核与网格）、`trace3.py`（按 >1 ms 空隙切段，分出 prefill 与 decode）、
`trace5.py`（把 torch 逐元素核按网格归组）。**注意**：`solve_kernel`（101 µs × 30）与
`*_seq` 系列全部来自 prefill 段，decode 段里没有它们——第一次不切段就报账时它们把结论
带偏过一次。

### 12.3 逐核账（decode 5 步平均；`真实` = traced − 5 µs）

| 内核 | 网格 | 次/步 | traced µs | 真实 µs | 字节/次 | 折合 |
|---|---|---:|---:|---:|---|---|
| `gemv_nk_i8<4,1,1>` | (3088,1,1) | 30 | 80.4 | 75.4 | 25.3 MB | 335 GB/s |
| `gemv_nk_i8<4,1,1>` | (62080,1,1) | 1.2 | 1118 | 1113 | 508 MB | **456 GB/s**（lm_head） |
| `gemv_nk_i8<4,1,1>` | (2304,1,1) | 10 | 67.8 | 62.8 | 18.9 MB | 300 GB/s |
| `gemv_nk_i8<4,1,1>` | (512,1,1) | 40 | 37.2 | 32.2 | 4.2 MB | 130 GB/s |
| `moe_grouped_i8` | (16,8,4) | 40 | 44.2 | 39.2 | ~6 MB | ~150 GB/s |
| `moe_grouped_i8` | (8,8,8) | 40 | 56.3 | 51.3 | ~6 MB | ~117 GB/s |
| `gemv_kn_i8_split<4>` | (8,32,1) | 40 | 33.6 | 28.6 | 8.4 MB | 293 GB/s |
| `gemv_kn_i8_split<4>` | (8,8,1) | 40 | 25.1 | 20.1 | 2 MB | 100 GB/s |
| `gemv_kn_split<4>` | (4,16,1) | 40 | 39.0 | 34.0 | 1 MB（路由） | **30 GB/s** |
| `moe_router_t<16>` | (1,1,1) | 40 | 36.0 | 31.0 | 0 | 单块纯延迟 |
| `moe_dispatch` | (1,1,1) | 40 | 26.5 | 21.5 | 0 | 单块纯延迟 |
| `shared_gate_scale` | (1,1,1) | 40 | 18.8 | 13.8 | 8 KB | 单块纯延迟 |
| `gather_quant` | (8,1,1) | 40 | 18.0 | 13.0 | 32 KB | — |
| `silu_mul_quant` | (1,1,1) | 40 | 16.5 | 11.5 | 16 KB | — |
| `silu_mul` | (1,1,1) | 40 | 12.3 | 7.3 | 16 KB | — |
| `moe_cast_idx` | (1,1,1) | 40 | 11.0 | 6.0 | 64 B | — |
| `add_rms_norm` | (1,1,1) | 79 | 15.8 | 10.8 | 4 KB | 地板 |
| `delta` | (32,1,1) | 30 | 69.5 | 64.5 | 2 MB | **32 GB/s** |
| `prologue` | (32,1,1) | 30 | 18.1 | 13.1 | — | — |
| `conv1d_silu` | (32,1,1) | 30 | 12.5 | 7.5 | — | — |
| `rms_norm_gated` | (32,1,1) | 30 | 15.0 | 10.0 | 16 KB | — |
| `attn_slice` / `attn_combine` | — | 10 / 10 | 41.9 / 20.9 | 36.9 / 15.9 | — | — |
| `p2p_combine` | — | 40 | 35.2 | 30.2 | 8 KB | 会合延迟 |
| torch 拷贝（**稳态**） | 4 种几何 | 82 | 12.1 或 4.5 | — | — | 合计 **1.17 ms/步** |
| torch 拷贝（**仅 warmup 首步**） | 4096/1024 两档 | 8（摊薄） | 124.5 / 57.3 | — | — | 见 §12.7，不是稳态成本 |

地板证据：`add_rms_norm` 只有 **1 个块、读 4 KB**，仍要 10.8 µs；`direct_copy` 4 KB 7.7 µs。
所以**单次发射在这张卡上就是 8–10 µs**，与搬多少字节无关。1077 × 24 µs ≈ 25.9 ms 正好是步时，
即 **步时 ≈ Σ 每核延迟**，不是带宽受限、也不是主机受限。

### 12.4 时间去哪了（按块归类，真实值）

| 块 | 次/层 | µs/层 | ms/步 | 备注 |
|---|---:|---:|---:|---|
| MoE（路由+共享+专家） | 12 | ~276 | **11.0（42%）** | 一层的**真实权重只有约 11 MB**，按 335 GB/s 应是 33 µs ⇒ **8 倍余量** |
| ↳ 其中 5 个小核（router/cast/dispatch/gather/gate/silu） | 6 | ~96 | 3.8 | 几乎不搬字节，纯发射+串行扫描 |
| 线性注意力层 ×30 | 6 | ~213 | 6.4 | `delta` 单独 1.9 ms、`qkvzb` 2.3 ms |
| 全注意力层 ×10 | 7 | ~175 | 1.75 | |
| `lm_head` | — | — | 1.1 | 456 GB/s，已接近这条路径的有效峰值 |
| `p2p_combine` | 40 | 30 | 1.2 | §9.6 说历史水位 19.6–22.1 µs，仍压着约 0.6 ms |
| 残差/归一化/torch 拷贝 | — | — | 1.2–1.5 | 40 次是 4 KB 的 `layer_out` 拷贝（每次 12 µs，§12.7）；另有 30+10 次 4096 元素的 `norm→out_row` |

### 12.5 被证伪的结构假设：双流重叠（实测无收益，已回退）

假设：共享专家支路（`sg` 投影 → SwiGLU → `sd` 投影 → 门标量，4 次发射，约 61 µs 设备时间）
与路由/分派/专家那串（约 135 µs）**没有数据依赖**，放第二条流上并行应能藏掉 61 µs/层。

改法：`moe_decode_impl` 里用 `getStreamFromPool` 的侧流 + 进出各一个 `cudaEvent`，
中间量逐个 `CUDACachingAllocator::recordStream`，`COFFIN_MOE_NOSIDE=1` 单点切除。

| TAG | ms/步 | tok/s | `[moeT]` 读数 |
|---|---:|---:|---|
| `032_moet`（改前） | 25.95 | 38.53 | `shared=20–30 µs`、`addsh=0.03 µs` |
| `037_side`（双流） | **26.21** | **38.15** | `shared=21 µs`、`addsh=137–244 µs` |

`addsh` 那一段就是主流等侧流的 event：**61 µs 的支路在并发下变成约 200 µs**（三倍互相拖慢），
净收益为零、还多两次 event 往返。结论：这张卡在「每核都很小」的 decode 负载上**已经被喂满**，
加流只是把同一份吞吐切开。**已回退**，`moe_block.cu` 回到单流形态（`038_final` 复核
25.9 ms 档、输出正常）。这条是负结果，但它砍掉了「靠并发藏延迟」这一整条思路——以后不必再试。

### 12.6 修正后的 60 tok/s 路线：每层 26.5 核 → 12 核

步时 ≈ Σ 每核延迟、地板 8–10 µs/核 ⇒ 60 tok/s（16.67 ms）在目前 24 µs/核的均值下要求
**≤700 次发射**，即每层 12 次。按块算（每核地板 8–10 µs，括号是发射数）：

| 改动 | 核/层 | Δ次/步 | Δms/步 |
|---|---:|---:|---:|
| MoE 12 → 4：route(top-k+cast+dispatch+gather) 合 1；共享 up + 专家 up 合 1 次发射（同一网格两个分支）；down 同理 | 12 → 4 | −320 | −3.3 |
| 线性注意力 6 → 3：`prologue`+`delta` 合；`norm_gated` 折进 `delta` 尾声（每头独立，块内即可） | 6 → 3 | −90 | −1.0 |
| 全注意力 7 → 4：`rope`+`kv_quant`+`kv_store` 合 1（只有 2 个 kv 头，单块够） | 7 → 4 | −30 | −0.3 |
| 残差与 torch 拷贝（归一化折进邻核、杀掉可复用的 `clone`） | — | −200 | −2.0 |
| **小计** | | **−640** | **−6.6 → 19.3 ms（52 tok/s）** |

再往下要 2.6 ms，只能动 MoE 内部的带宽：`moe_grouped` 现在约 150 GB/s（内部循环是
`wexp[k*N+col]` 的逐字节读 + smem 广播，K=2048 时每块只覆盖 64 列） ⇒ 300 GB/s 可再省
约 1.5 ms；`delta` 32 GB/s（2 MB/层）⇒ 若按头并行改成 128 块/层，可再省约 1 ms。
**所以 60 tok/s = 「每层 12 核」+「MoE/delta 带宽翻倍」，是 §9.4 那个 1–2 周量级的活，
但目标形态从常驻网格 megakernel 降级为「12 核/层」，不再需要重写执行模型。**

复现：队列 `030`（ixprof `--help` 核参）/`031_tr`（时间轴）/`032_moet`（主机分段）/
`037_side`（双流）/`038_final`（回退复核）；探针 `work/p5.py`（两步窗口）、`work/p4.py`
（bench+cprof+GEN32）；runner `work/tr_run.sh`、`work/h6_run.sh`、`work/build55.sh`。

### 12.7 拷贝溯源：那两大拷贝只在 warmup 首步出现一次，稳态根本没有

§12.3 里「torch 拷贝 3.28 ms/步」是**把一次性初始化摊进了每步**，这一节把它纠正掉。

**(a) 分布证据**（`trace8.py` 打每个拷贝的核序号与相邻间隔）：`grid(4096)` 与 `grid(1024)`
各 40 次、**成对紧邻**（间隔 1 个核），全部落在第 16–1125 号核之间，对与对之间隔 25 个核
（每 4 层有一次 34）——即「每层同一个位置、各一次」；第 1125 号之后的 4261 个核里**一次
都没有**。而这一段正好 5 步（`p5.py` 的 bench = 3 次预热 + 2 次计时），所以这 40×2 次
全部落在**预热第一步**：decode 起始的状态初始化。`grid(1024)` 那档 128×4×1024 =
**524,288 元素 = `32×128×128` 的 delta 状态**（与 `ops/delta.py:165-167` 的
`out_state[pr] = st` / `out_state.copy_(st)` 同形），`grid(4096)` 那档 = **2,097,152 =
2048×1024**（EP 槽区/`gu` 缓冲同形）。

**(b) 稳态的真实拷贝开销只有 1.17 ms/步**（`trace7.py` 按网格归组，只算每步都出现的档）：

| 档 | 次/步 | µs/次 | µs/步 | 身份（待核） |
|---|---:|---:|---:|---|
| `grid(4)×512thr` | 42 | 12.0 | **502** | 4 KB，40 层各一次 + 2 次 logits（`layer_out` 那类小拷贝，**每次 12 µs 全是发射地板**） |
| `grid(8)×512thr` | 30 | 12.7 | 382 | 4096 元素 = 线性层的 `norm → out_row`（`linear_attn.py:485`） |
| `grid(8)×128thr` | 10 | 16.3 | 163 | 全注意力层 |
| `fill grid(1)` | 11 | 11.0 | 121 | 小张量清零 |

**(c) Python 侧溯源器一条都没命中——大拷贝不是 Python 发的。**
`serving/cpy_trace.py`（`COFFIN_CPY_TRACE=1` 门控，按「算子+形状+调用点」去重、一次性、
不带环境变量时连 import 都不发生）把 `copy_`/`clone`/`contiguous`/`to`/`__setitem__`/
`index_copy_`/`cat` **全包上**后（`041_cpy`，25.91 ms / 38.60 tok/s，对计时无可测影响），
decode 期间**一条都没打印**；打印出来的 26 条全是加载期：`parallel.py:1000/1007/1010`
的权重搬运、`pack.py` 的 `to_kn`/`nk_weight` 重排、`ops/delta.py:165/167` 的
`(32,128,128)`。
结论：**这些拷贝要么来自 C++（`torch::Tensor` 上的 `clone`/`copy_` 不走 Python 分派），
要么来自自定义算子的 C++ 实现**。下一轮要定位只能读 C++ 或按 `COFFIN_FUSE_DUMP` 的方式
在 C++ 里打标量，**不要再包 Python**（这条弯路已经走过一次）。

### 12.8 本轮收尾状态

* 双流改动**已完整回退**（`moe_block.cu` 的 include 只剩 `CUDAStream.h`/`StringUtil.h`，
  共享支路恢复串行）。`038_final`（build55 + h6_run，TAG=`rv1`、PORT=31611）复核：
  `BUILD_S 36.2`（`BUILD_RC=1` 是 smoke 段 `m.add` 的历史残留，判编译成功看 `BUILD_S`）、
  `BENCH 25.73 ms / host 25.04 / 38.87 tok/s`、`GEN32` 文本通顺。即回退后回到 25.7 ms 档。
* `041_cpy`（带溯源器）：25.91 ms / 38.60 tok/s。
* 本轮**没有净性能收益**：唯一改动（双流）零收益且已撤回。产出是 §12.1–12.7 这四笔账
  （主机侧证伪、设备侧逐核表、双流证伪、拷贝账修正）与 §12.6 的路线表。

### 12.9 双流为什么反而更慢：这张卡不跨流并发，事件还会刷提交队列

§12.5 只说了"并发把 61 µs 拖成 200 µs"，是个现象。这一节给机制：一个不碰模型的
纯 torch 微基准（`work/micro.py`，单卡，队列 `042_micro`，`Iluvatar BI-V100 / 16 SM`）。

**1) 两条流上的核根本不并发——双流一律更慢。**

| 负载 | 单流串行 | 分两条流 | 差 |
|---|---:|---:|---:|
| 2 × 4 MiB 拷贝 | 98.6 µs | **129.2 µs** | −31% |
| 8 × 4 KiB 拷贝 | 101.2 µs | **135.9 µs** | −34% |
| 「支路模式」4 核侧流 + 4 核主流 + 进出事件 | 104.7 µs | **180.0 µs** | −72% |

第三行就是我在 `moe_decode_impl` 里加的那个形态，微基准里它**比单流慢 72%**，与模型上
`26.21` 对 `25.95` 的符号完全一致。原因不是挡住彼此，而是这套运行时**按全局提交序执行**：
换一条流发核要额外付一次提交/切换（约 30–35 µs），并发收益为零。⇒ **单流是这张卡上
唯一正确的形态**；任何多流/流水线重叠的写法都只会更慢，以后不必再试。

**2) 事件不是免费的：一次 record 1.3 µs、一次 wait 4.0 µs，但一整套往返 25.4 µs。**
`event.record` 与 `wait_event` 单看都很便宜（3 倍价差说明**每次跨流操作都会把提交队列
刷一遍**，这是它比单次 op 贵 10 倍的原因）。40 层 × 25.4 µs = **1.0 ms/步**的纯主机成本，
这就是 `[moeT] addsh` 从 0.03 µs 涨到 137–244 µs 的第一块。

**3) 剩下那 100 多 µs 是分配与背压。** `torch.empty(2048)` 实测 **20.7 µs/次**（侧流 20.3，
两者相同，所以不是跨流分配的锅）；汇合点那几个张量一分配就是 60 µs 起。而汇合
（`wait_event`）正好落在主机被发射队列反压的位置，主机在这里本来就排队，于是这段
被计成了 137–244 µs。

**4) 那么"更慢"到底慢在哪：** 可确定的净亏是每层一次事件往返（25 µs 主机）+ 每层一次
跨流提交切换（约 30 µs）≈ 55 µs/层 ≈ 2.2 ms/步；模型上只看到 +0.26 ms，是因为主机本来
被反压顶在 25 ms，多出来的主机活大部分被"本来也要等"吸收掉了。**剩下的 +0.26 ms 里
可能有一部分是抖动**（同一份代码历史上跑过 25.64–26.72），但方向是确定的：这套运行时
上双流只会更慢。

**顺带校准的两条地板**（warm，单卡空载）：4 KiB 拷贝 **12.6 µs**（= 8 次 101.2 µs），
4 MiB 拷贝 **49 µs**（≈171 GB/s 含读写）。前者与时间轴上 `grid(4)` 那 42 次 @12 µs 对上，
后者说明**小核的成本全是提交/启动，与搬多少字节关系不大**——这条是 §12.6 整张路线表的
前提，现在有了独立于 ixprof 的旁证。

### 12.10 双流在这套驱动上做不了：驱动侧文档 + 两组微基准的双证

§12.9 的机制是实测出来的，这一节把它钉到驱动侧：`corex-torch/corex/docs/05-DRIVER-CONSTRAINTS.md`
（对 `libixthunk.so` 的 40 个 `Thunk_*` 入口与 `iluvatar.ko` 字符串表/反汇编的逆向结论）
里有五条直接对上。

**驱动侧事实**

* `CONCURRENT_KERNELS = 1` 只是**报告值**；该文档 **R4** 明写「性能模型按单流算……
  任何『靠 stream 并行提速』的设想都要先测」。
* 提交是「一次 ioctl 带一整块 cmdbuf」，包链是固定四元组
  `POLL_MEM_REG + 0x1a + NDR_KICKOFF + EVENT_WRITE`；**每个 `NDR_KICKOFF` 后面必须跟
  自己的 `EVENT_WRITE`**，串行性由这些包保证（R6）。⇒ 想让两段计算重叠，就得在一条
  cmdbuf 里去掉这些排序包——文档 R1/§1 明写自拼 cmdbuf 已经把一张卡跑死过一次
  （`pkt_bd_len` 少算 4 字节），所以这条路不能走。
* 队列是 KMD 内部对象（`hEngine = (context_id << 32) | queue_id`），`AIP_QUEUE_CREATE`
  不读用户给的 ring 描述 ⇒ **流=队列，数量与所有权都在内核侧**。
* 每个队列只有 **4 个 user fence 槽**（`usr_fences[4]`，R3）⇒ 事件是稀缺资源。§12.5 那种
  每层挂 2 个事件的写法本身就在踩这条线。
* 该文档 §3 把「多条流在 wall-clock 上到底有没有并发收益」列为 **UNKNOWN**，规定测法是
  「两条流各跑一段带宽 kernel，比较总时间」——**就是下面这两组**，现在有答案了。

**实测（`work/micro.py` / `work/micro2.py`，单卡空载，队列 `042` / `044`）**

| 测项 | 单流 | 双流 | 差 |
|---|---:|---:|---:|
| 流句柄 | default=`0` | s1=`135729600` / s2=`135923488` | 三条真实且不同的队列，不是 torch 别名 |
| 2 × 4 MiB 拷贝 | 98.6 µs | **129.2 µs** | −31% |
| 8 × 4 KiB 拷贝 | 101.2 µs | **135.9 µs** | −34% |
| 2 × 64 MiB 拷贝（单核 557 µs） | 946.4 µs | **992.3 µs** | **−5%** |
| 支路模式（4 核侧流 + 4 核主流 + 进出事件） | 104.7 µs | **180.0 µs** | −72% |

第三行是关键：**把核放大到 557 µs 仍无一丁点重叠**，只剩一次队列切换的 46 µs——所以
「小核不值得并发」也排除了。配合 `s1 != s2 != 0` 的句柄，结论只能是**并发在这张卡上不存在**。

（另有一个「DMA 与计算是否并发」的测项，两次 compute 基数差了 3 倍——时钟没预热稳，
测不准，不作为证据。）

**判读与含义**

1. 「把共享支路藏到第二条流」这个设想**本身不成立**，不是实现得不好。`037_side` 的
   0.26 ms 净亏 = 每层事件往返 25 µs + 每层队列切换约 30 µs（§12.9），被主机背压吸收掉
   大半后剩下的可见部分。
2. **单次发射 12 µs 的地板是 CP 走包链的成本**（每次 kickoff 连带自己的 `EVENT_WRITE`
   与一次 `POLL_MEM_REG`），不是主机开销。⇒ 「减少发射次数」省的是**设备侧 CP 的时间**，
   而这正是驱动文档 R2 背书的唯一路径（「要减少发射次数，走的是驱动侧已验证的
   『一次提交 N 条命令』那条路」）。
3. 所以 §12.6 的「每层 26.5 → 12 核」不是若干可行方案之一，而是**这张卡上唯一正确的
   形态**；多流 / 流水线重叠 / 靠并发藏延迟，到此全部出局。

### 12.11 第一次落地：去掉 `moe_cast_idx`，一步 1076 → 1036 次发射

按 §12.6/§12.10 那条唯一路线动手，第一刀选的是**纯浪费**的那一发：

* 改动：`moe_dispatch` 的内核按入参 dtype 选模板（`int` / `int64_t` 各一份实例），
  于是 `moe_block.cu` 直接把 `moe_router` 写出来的 int64 `idx` 交过去，删掉
  ``a.idx.reshape({-1}).to(kInt)`` 那一发窄化（`moe_cast_idx`，11 µs/层）。
  Python 那条路仍传 int32，行为不变；`moe_cast_idx_kernel` 保留但标 `[[maybe_unused]]`
  当退路（理由见该函数注释）。
* 踩了一下：`a.idx` 是 `[tokens, top_k]` **二维**，第一版入参检查写成 `dim() == 1`，
  服务端起不来（`expert_idx must be contiguous CUDA int32 or int64 [slots]`，队列 045）。
  放宽到 `dim() <= 2` 后通过（队列 046）。
* 实测（`046_cast`，TAG=`cast1`，build 37 s）：

```
BENCH 25.45 ms / host 24.78 / 39.30 tok/s      ← 改前 038_final 25.73 / 25.04 / 38.87
GEN32 " Paris, a is the the largest city in France. The capital of the French government is Paris."
[moeT] cast=0.038 （原来 11.30）
```

净收益 **−0.28 ms / +0.43 tok/s**。发射数每步从 **1076 降到 1036**（−40），按 §12.9 的
12 µs 地板应值 −0.48 ms，实测 −0.28 落在历史抖动带（25.45–26.72）内，属于「确定去掉了
一发、收益量级符合」而不是「吃满理论值」。文本正常（贪心解码在这套 p2p 上本来就不是逐
token 可复现，`n_new` 每次不同，见 §11）。

### 12.12 第二刀的两种做法，以及一个必须先排掉的陷阱

下一个目标还是「每层 12 → 更少」。按 §12.3 的账，一层 MoE 里真正只吃发射地板的有一组
1-block 小核：`moe_router`(31 µs) / `moe_cast_idx`(已去掉) / `moe_dispatch`(21.5) /
`gather_quant`(13) / `silu_mul`(7.3) / `silu_mul_quant`(11.5) / `shared_gate_scale`(13.8)。

**(A) 两发 1-block 并一发（可做，收益 ~0.5 ms/步）。** `silu_mul_quant`（[512] 的
SwiGLU + amax 量化）与 `shared_gate_scale`（2048 点积 + [512] 缩放）都在同一时刻、都是
单 block、互不依赖（`sh` 在专家支路之前就算好了），合成一个核：phase A 算
`silu*up`+amax+量化，phase B 算门标量并缩放 `shared_hidden`，中间一次 `__syncthreads`、
复用同一块 `tree[256]`。省的是**一次发射地板**（12 µs/层 = 0.48 ms/步），两段工作本身
照做。

**(B) 路由两发并一发（收益 1.4 ms/步，但做法有讲究）。** 现在 `gemv_kn_split`（34 µs，
1 MB 权重只跑到 30 GB/s）+ `moe_router_t`（31 µs，单 block 256 选 8）= 65 µs/层。
**陷阱：不能用「单 block 的融合核」**——§12.3 已测出这张卡单 SM 的流式带宽只有约
**28 GB/s**（lm_head 456 GB/s ÷ 16 SM），一个 block 读 1 MB 就要 36 µs，比现在的 34 µs
还慢，再叠 top-k 只会更糟。
**正确形态**是本仓已经用熟的「分片核 + 到达计数 + 末块就地收尾」（§1.1 那套零自旋栅栏）：
gemv 的 4 个分片照旧多 block 并行，每个分片写完 `logits` 后 `__threadfence` + 命中计数，
**最后到的那一块直接跑 `moe_router_t` 的 top-k 主体**（它本来就是「读全局 logits 的
1-block 核」，语义天然契合），然后计数器自复位。这样 65 µs → 约 34 µs，
**−31 µs/层 = −1.2 ms/步**，而且不牺牲 gemv 的并行度。

**(C) 同一套栅栏还能顺手把 `gather_quant` 并进 up 投影**：`moe_grouped_i8` 的 up 段本来
就要按 `token_of_pos` 取行（`gather_quant` 只是「按位置表把行拷进连续缓冲并量化」），
把这段搬进 up 核的每个 block 头部去做，`gather` 那一发（13 µs）就消失了。

**顺序建议**：先 (A)（纯加法，风险最低，改两处 + 一个新核），再 (B)（要接栅栏），
再 (C)（动的是核心分组 GEMM）。每刀 0.5–1.2 ms，凑到 16.67 ms 需要把这一整组做完
（约 10–18 次这样的合并，对应 §12.6 的「1–2 周」）。

§12.12(A) 后来没做，原因记在这里：`silu_mul_quant`（专家支路，`moe_block.cu:269`）与
`shared_gate_scale`（共享支路，`:245`）中间隔着 dispatch、gather、up 分组投影，**不是
邻居**，合核等于把一个核挪动位置去凑另一个，收益还是只有一次发射地板。同一条理由下
`shared_gate_scale` 真正的邻居是它前面的 `sd` 投影（通用 int8 gemv），把门标量塞进通用
核的 epilogue 不值得。⇒ 「两发 1-block 并一发」只对**真邻居**成立：本层里真邻居只剩
`moe_router`↔`moe_dispatch`（可并，−21 µs/层）、`moe_dispatch`↔`gather_quant`（有依赖
但同块可接，−13 µs/层）。

### 12.13 第二刀：路由 gemv 的 split 从 8 抬到 32（−0.37 ms）

上一节说的「路由两发」暂不动，先试**不动内核**的那条：路由 gemv 慢不是带宽问题，是
**并行度不够**。

* `gemv_kn(a.router_wt, x, 0, 0, 0)` 在 `N=256, K=2048` 上走自动规则：
  `pick_threads(N=256)=64`（`N>=512` 才给 128，`N>=2048` 给 256），
  `blocks=(256+63)/64=4`，窄形状 `by_blocks=(64+4-1)/4=17`、`by_k=2048/GEMV_MIN_SLICE`，
  最终 `split=8` ⇒ **32 个块 / 每块 64 线程**。
* 这张卡 16 个 SM：32 个块 = 每 SM 2 个块，而 64 线程一块只有 **1 个 warp**（64 通道）
  ⇒ 每 SM 只有 2 个 warp 在飞。1 MB 权重跑 34 µs = **30 GB/s**，是纯内存延迟受限，不是
  带宽受限（同卡峰值 456 GB/s，§12.3 的 lm_head）。
* 改法：显式传 `split=32`（`moe_block.cu` 的 `kRouterSplit`），128 个块 ⇒ 每 SM 8 个 warp，
  是 4 倍的未完成访存条数。注释里写明了「只走驱动内解码路径，不碰 `bops.linear` 那条
  逐位一致契约」——这是唯一需要交代的越界。
* 实测（`047_rter`，build 36.2 s，代码 = §12.11 之后再改这一处）：

```
BENCH 25.08 ms / host 24.41 / 39.87 tok/s     ← 上一刀 25.45 / 24.78 / 39.30
[moeT] router=11.0~11.2（主机侧不变，收益全在设备侧）
```

净 **−0.37 ms / +0.57 tok/s**（累计 25.73 → 25.08，−2.5%）。这一刀**没有**新增发射，
只是把一个核的并行度补到它该有的样子 ⇒ 说明 §12.3 那张表里还有一批「块数 < 16 SM」的
小核，它们的 30–120 GB/s 可能都是同一个病因，值得逐个按「块数是否够一个 wave」复核。

* 附带记录一条**不是本次引入**的偶发：`047` 的 GEN32 空，rank2 报
  `shape '[5, 2048]' is invalid for input of size 2048`，rank0/1/3 因对端退出而
  `p2p combine rendezvous aborted`。把 `grep -l 'invalid for input of size'` 打到全部
  服务端日志上，命中 `fm_d1` / `moeT1_d3` / `tmoe_d2` 三个**改动之前**的 TAG ⇒ 生成探针
  这条路径本来就偶发出这个错，与 `split` 无关，别去追。

### 12.14 时间轴复核：`split=32` 确实吃到设备侧，但只有预测的 1/4

`049_tr9`（`tr_run.sh`，TAG=`tr9`，rank0 挂 `--print-gpu-trace`，`p5.py` 只 bench n=2），
新原始数据 `.scratch-perf/tr9_d0.log`（11233 行），按 `trace9.py`（>1 ms 全空白切段 +
按「内核名+网格」归组）：decode 段 **5186 核 / 5 步 = 1037 核/步**（1076 − 40 + 1 的账对上
了），**traced 合计 30.45 ms/步**，减 1037 × 5 µs 膨胀 ⇒ 真值 **25.26 ms/步**，与 BENCH
的 25.08/25.35 相符。

| 内核 | grid | 次/步 | traced µs | 真值 µs | 计入 ms/步 |
|---|---|---:|---:|---:|---:|
| `moe_grouped_i8`（down/merge） | (8,8,8) | 40 | 61.65 | 56.7 | **2.27** |
| `gemv_nk_i8`（qkvzb 四条） | (3088) | 30 | 80.38 | 75.4 | **2.26** |
| `delta_kernel` | (32) | 30 | 69.70 | 64.7 | **1.94** |
| `moe_grouped_i8`（up） | (16,8,4) | 40 | 44.71 | 39.7 | **1.59** |
| `gemv_nk_i8` | (512) | 40 | 37.49 | 32.5 | 1.30 |
| `moe_router_t<16,u16>` | (1) | 40 | 35.59 | 30.6 | 1.22 |
| `p2p_combine` | (1) | 40 | 35.05 | 30.1 | 1.20 |
| `gemv_kn_i8_split`（共享 sg） | (8,32) | 40 | 33.76 | 28.8 | 1.15 |
| `gemv_nk_i8`（lm_head） | (62080) | 1.2 | 1123 | 1118 | 1.34 |
| `moe_router` 的 gemv | **(4,32)** | 40 | **31.25** | 26.3 | 1.05 |
| `add_rms_norm` | (1)/1024thr | 79 | 15.93 | 10.9 | 0.86 |
| `moe_dispatch<long>` | (1) | 40 | 27.00 | 22.0 | 0.88 |
| `gemv_kn_i8_split`（共享 sd） | (8,8) | 40 | 24.92 | 19.9 | 0.80 |
| `gather_quant_i8` | (8) | 40 | 18.25 | 13.3 | 0.53 |
| `shared_gate_scale` | (1) | 40 | 18.61 | 13.6 | 0.54 |
| `silu_mul_quant` | (8) | 40 | 16.75 | 11.8 | 0.47 |
| `silu_mul` | (2) | 40 | 12.60 | 7.6 | 0.30 |

**路由 gemv：traced 39.0 → 31.25（真值 34 → 26.3），−7.8 µs/层 = −0.31 ms/步**——与 BENCH
的 −0.37 一致。除它以外，其余每个核都在 **±10% 抖动**内（`moe_grouped` down 56.3→61.7、
`moe_router_t` 36.0→35.6、`dispatch` 26.5→27.0、`p2p` 35.2→35.1），所以「一次改动的收益」
只能靠**这一个核自己的数**来认，不能靠 BENCH 的差值——这是复核 §12.11/§12.13 时学到的：
BENCH 的抖动带是 **±0.3 ms**（同一份代码连跑 25.08 / 25.35）。

**为什么只吃到 1/4：** 抬到 128 块后每块仍需串行 **128 次**「64 线程 × 128 B」的读取
（1 MB / 128 块 = 8 KB，每轮 128 B）。这条链是纯延迟，`unroll=4` 也摊不平。⇒ 要让路由
gemv 真正消失，只有 §12.12(B) 那条「栅栏 + 末块就地 top-k」：把 `moe_router_t` 的 30.6 µs
整个吃掉，而不是继续加块。**继续加 `split` 已经不值**（再翻一倍最多 −5 µs/层，还要多担
一份 256 块的分片开销）。

`grid(4096)`(124.68×8) 与 `grid(1024)`(57.67×8) 那两组大拷贝仍在，且**只落在预热步**：
5 步里各 8 次 ⇒ §12.7 的结论（稳态不存在、只有起始状态初始化那一次）继续成立，
它们的 1.46 ms/步**不是**稳态成本。

### 12.15 「GEN32 没出文本」的定位：prefill 路径上的既有竞态，与 split 无关

§12.13 那一版跑出三次空 GEN32，必须先排除「是不是 split=32 把生成弄坏了」。做法与结论：

1. **先把 split 撤回自动值（16，见下）复核**（`051_rv2`）：`BENCH 25.44 / 39.32`，
   `GEN32 n_new=32`「 Paris, a is also the country's largest city and a major
   globalcenter for finance, culture, and tourism.」正常。
2. **再带着 split=32 复现一次**（`052_diag1`）：`BENCH 25.78 / 38.79`，
   `GEN32 n_new=32`「 Paris, a city known for its art, fashion, gastronomy, and
   culture.\nA.\nB.\nC.\nD.」**同样出文本**。
   ⇒ 空结果**不是**这个改动导致的，是间歇的。
3. 错误文案本身指向哪：`shape '[5, 2048]' is invalid for input of size 2048` ——
   一个 2048 元素（1 行）的激活被当成 **5 行**（prefill 的 5 个 token）去 `reshape`。
   它落在 **prefill** 路径上，而路由 split 只影响 **decode**（tokens=1）的 256 个 logits
   求和次序，两条路径不相交。同一文案在**改动之前**的 `fm_d1` / `moeT1_d3` / `tmoe_d2`
   三个 TAG 里都出现过（`grep -l 'invalid for input of size' occ5/logs/*.log`）。
   ⇒ 定性：prefill 侧「拿到的激活行数与 token 数不一致」的既有竞态，命中率约
   每几次生成探针一次，落点随 run 在 rank 之间漂移（047 是 d2、048 是 d1）。
4. 顺手把诊断补上：`serving/runtime_parts.py` 的引擎失败分支原来只 `print` 一行消息，
   栈全丢（这正是本次只能靠「换个 split 再跑」去二分的原因）。现在加
   `traceback.print_exc()`。异常分支不在热路径，成本为零；**下次复发就能直接看到
   文件:行号**，不用再猜。

**split 的现状与定论**：`kRouterSplit = 32` 留着。四次实测 `25.08 / 25.09 / 25.35 / 25.78`
（中位 25.22，最好 25.08），对照 §12.14 的「路由 gemv 真值 34 → 26.3 µs = −0.31 ms/步」，
账对得上；`kRouterSplit = 0` 那次是 25.44。抖动带 ±0.3 ms 意味着单次跑分不出高下，
**认这笔收益靠的是那个核自己的时间轴**，不是 BENCH 排行。

### 12.16 第三刀：`delta_kernel` 拆链 + 拆块（69.70 → 55.99，−0.41 ms/步）

`delta_kernel` 原来是「一个线程独占一列 `j`，在寄存器里把 `dk` 走**两遍**」，靠「两遍都在
本线程内、完全不发生跨通道通信」换正确性。解码期 `rows = 1`，网格只有 `heads` = 32 个块
（`blocks_y = ceil(dv/128) = 1`），每块 128 线程。两 MB 状态走 69.70 us。

试了两档，**第一档几乎白做**：

| 版本 | 块 / 线程 | 链长 | traced µs |
|---|---|---:|---:|
| 原版 | 32 / 128 | `2*dk` = 256 | 69.70 |
| 只加路数 `WAY=4` | 32 / **512** | 64 | 59.88 |
| 再加块数（列切成 4 段） | **128** / 128 | 64 | **55.99** |

第一档说明**链长不是瓶颈**（链缩到 1/4 只买到 15%）；第二档把值轴切成 4 段、网格变成
(32, 4)，再拿 7%。两档合计 **−13.7 µs/层 × 30 = −0.41 ms/步**，`BENCH 24.83 / 40.27`
（`053_dlt` 那版 `GEN32 n_new=32`，文本正常；第二档对每列的四份部分和求和次序与第一档
**逐位相同**，所以 053 的文本样本覆盖当前这版）。

实现要点（都要留在文件里，因为它们是这张卡的事实）：
* `WAY` 路拆 `dk` 需要两个跨线程归约（记忆、输出各一次），共享里一块 `[WAY, dv]`
  部分和 + 两次 `__syncthreads()`；**越界列绝不能提前 return**，否则新增的屏障会
  等一个已经不在的线程（设备侧死锁，那条注释原本就写在文件里）。
* `red` 的步长用 `dv` 而不是 `ncol`，这样列切成 4 段后各块的共享区互不重叠，块内
  求和次序也保持不变。

**由此得到一条要写进路线图的经验律**：在这张卡上，给这类「每块带宽撑死几百 MB/s」的小核
**靠加并行度**（加线程或加块，4 倍量级）只能换 **约 20%**，两次独立验证——
路由 gemv（32 块 → 128 块，39.0 → 31.25，−20%）、`delta`（32 → 128 块，69.7 → 56.0，
−20%）。⇒ §12.6 里「MoE/delta 带宽翻倍」那 2.5 ms **不可能靠补块数拿到**，必须改
**访问模式本身**（把 `wexp[k*N+col]` 的逐字节读换成每线程 4 列的 32 位读），
或者把整段工作从设备上拿掉（合并发射）。这一条把下一阶段的预期收益从「乐观」拉回
「每刀 0.3–0.5 ms」。

### 12.17 第一大户 `moe_grouped_i8` 的病因：不是带宽，是指令发射

两个 `moe_grouped_i8` 合计 3.86 ms/步（§12.14）。§12.6 把它的低带宽（150/117 GB/s vs
峰值 456）记成「逐字节读」，这一节把账算清楚，结论与那条路线不同：

**内循环一笔一笔算**（`moe_grouped_i8.cu:216`，`[E,K,N]` 布局，权重是 ``kn``）：

```cpp
for (; k + 16 <= k1; k += 16) {                        // 每轮 4 个 dp4a
    const unsigned q0 = i8_dword(wexp, col, N, k);    // 4 次单字节读 + 3 次移位/或 = 7 条
    ...                                                // q0..q3 → 共 28 条 + 4 次 smem 读
    s0 = __dp4a((int)q0, a0, s0); ...                  // 4 条
}
```

一个 16-k 轮 ≈ 4 + 28 + 4 = **36 条指令，其中 28 条（78%）是在为 ``__dp4a`` 拼那个 32 位
字**。评估：``up`` 是 (16,8,4)=512 块 × 128 线程，每线程 ``K/slices/4`` = 128 轮
⇒ 单核约 2.3 万条/线程、全卡约 3 亿条指令量级，正好落在实测的 45 us 上。
⇒ **它是指令发射受限，不是带宽受限**。这解释了两件事：为什么它的有效带宽只有
150 GB/s（同卡 lm_head 能到 456），以及为什么 §12.16 那条「4 倍并行只换 20%」在它上面
同样会成立——加块数不减少指令。

**唯一的根治办法是让那个 32 位字来自一次加载**，也就是 ``nk`` 布局
（``[E,N,K]``，K 在列内连续）。查过两条路，各自的真实约束：

1. **`COFFIN_MOE_I8_NK=1` 现在开不了。** 它走 `moe_grouped_i8_nk_kernel`（warp 认领一列），
   而**那个内核没有末块收尾**（`moe_grouped_i8.cu:1203/1208/1223` 的启动里 `cnt`/`mcnt`
   全是 `nullptr`）。解码驱动的「一次发射完成 down 投影 + 跨槽位加权合并」依赖 `mcnt`
   那一支（§1.1 那套栅栏），所以开这个开关等于把 40 次/步的合并退回去，而且没有任何
   现成的 nk 版收尾可以接——nk 里一个 warp 独占一列，收尾要重写成 warp 级形态。
2. **内存不是障碍，这一条要纠正 §12.6 的旧说法。** `QWeightMoE.nk_stack()`
   （`coffin/int8_weights.py:216`）在 `layout == "nk"` 时**直接返回 checkpoint 那一份，
   零拷贝**；只有 `layout != "nk"` 才做 `transpose(...).contiguous()` 翻倍。
   我们的模型目录就是 `occamy-1.0-W8A8-nkfull`，而布局由读盘时有没有 `#nk` 键决定
   （`models/qwen35/parallel.py:971` `nk_native = read_key != key`）。
   ⇒ **已确认：`~/models/occamy-1.0-W8A8-nkfull/quant_index.json` 里有 80 个 `#nk` 键**
   （40 层 × 2 份专家栈，`grep -c '#nk'` = 80），也就是这个 checkpoint 本来就是 nk 原生，
   `nk_stack()` 会零拷贝直接交出那份存储。**缺的只是 nk 内核的收尾，不是内存。**

**折中方案（不动布局）**：`[E,K,N]` 下仍然能凑出 16 字节对齐的读——让一个线程认领
**4 个相邻列**，每个 k 读一个 ``int32``（4 列各 1 字节），4 个 k 拼出 4 个 dp4a 操作数需要
``__byte_perm`` 转置（每列 3 条）。算术上：16 字节的活从 32 条指令降到 20 条（1.6×），
但要先确认这张卡的编译器支持 ``__byte_perm``，而且列的映射一变，`I8_COLS` 那条
「128 列 = 一个整事务」的实测结论要重新量。**它是备选，不是首选。**

















## §12.18 第四处收益：nk 零拷贝布局 + 融合驱动的 nk 分支（24.83 → 24.28 / 41.19）

§12.17 结在「nk 路线缺的只是内核收尾」。这一节把它真正打开，并且顺手拆掉两个**不在
内核里**的闸门——它们才是让 nk 一直「开不起来」的真正原因。

### 12.18.1 三处闸门（按发现顺序）

1. **加载器**：`models/qwen35/parallel.py:950` 读盘时按 **`COFFIN_MOE_STACK=nk`** 挑
   `#nk` 那份键；只开 `COFFIN_MOE_I8_NK=1` 时读的还是 kn 那份，`layout` 停在 `"kn"`。
   shard 头实测（`quant-00001` 的 safetensors header，不经过 torch）：
   `gate_up_stack [256,2048,1024] I8` 与 `gate_up_stack#nk [256,1024,2048] I8` 并存，
   `down_stack [256,512,2048]` 与 `down_stack#nk [256,2048,512]` 并存 —— 两份各自的
   字节量相同，读哪份都是「一次读盘」，**零拷贝成立**。
2. **融合驱动写死 kn**：`moe_block.cu` 里两处 `/*nk=*/0`（gate_up 与 merge_tokens），
   以及 `inter = gu_w.size(2)/2`、`hidden = gu_w.size(1)` 这些按 kn 取的几何。
   nk 栈是 `[E,N,K]`，同两个表达式会算出 1024/1024（应为 512/2048），报的却是
   `shape '[1, 1024]' is invalid for input of size 2048` —— 落在 `moe_layer_run` 的
   `hidden`，也就是 `layer_step` 那条整段融合路径上。
3. **整块驱动自己退出**：`moe.py` 里一句 `if self.gate_up.layout != "kn": self._block_driver = None`。
   它的理由（nk 原生栈上 `wq` 是一次整栈转置 40 层 32 GB）在**有** `#nk` 键时不成立：
   那时该走 `nk_stack()` 而不是 `wq`。现在改成「按 layout 选栈，并把 nk 传给句柄」。

### 12.18.2 内核收尾移植的落点（与 §12.17 的计划不同的一点）

计划是搬进 `moe_grouped_i8_nk_kernel`，这一步做对了；但**第一次放错了入口**：decode
走的是 `moe_grouped_i8_merge`（`:1170`）而不是 `_merge_tokens`（`:959`），收尾加在后者
等于没加。两个入口都已接上（`mcnt` 只在 nk 分支开，kn 那一支不动，避免同一刀里改两件事）。

nk 版收尾与 kn 版逐字同口径（行升序、fp32、每行先乘两 scale 再乘路由权重），差别只有
「一列归一个 warp」，所以由 `lane == 0` 收它那 `WCOL` 列；`slot_of_pos == nullptr` 时
路由权重按行号取（`merge` 入口的契约），非空时按槽位取（`merge_tokens` 的契约）。
另有一处**必须改**的守卫：`colbase >= N` 原来直接 `return`，而 epilogue 里有
`__syncthreads()`，per-warp 提前退出会等一个不存在的线程（设备侧死锁、无 traceback），
已改成 `nc = 0` 夹住。

### 12.18.3 实测（`061_nk4` / `062_trnk2` / `064_nk6`）

BENCH：**24.28 / 41.19**、**24.39 / 40.99**（`GEN32 n_new=32`，文本正常，
" Paris, which is also the most populous city in the country.\n\n<think>…"）。
上一版（kn）是 24.83 / 40.27。**净 −0.45 ms / +0.7 tok/s。**

时间轴（`trnk2`，同一套 `>1 ms` 切段 + 内核名/网格归组）：

| 投影 | kn（`trB`） | nk（`trnk2`） | 差 |
|---|---|---|---|
| gate_up `(8,8,8)` → `(128,8,1)` | 61.11 us × 40 | **35.73 us × 40** | **−0.98 ms/步** |
| down `(16,8,4)` → `(64,8,2)` | 43.99 us × 40 | 56.15 us × 40 | +0.48 ms/步 |
| 合计 | 4.20 ms/步 | 3.68 ms/步 | **−0.52 ms** |

两条都要读准：

* **§12.17 的定性被证实**。gate_up 是 K=2048 的深栈，kn 那一支每 16 个 k 要 28 条指令
  拼 `__dp4a` 的 32 位字；nk 里这个字就是一次加载，**61.11 → 35.73（−42%）**。
  3.86 ms 的「两次分组投影」里，可摘的那一半确实在 gate_up 上。
* **down 反过来是亏的**（+12 us），而且**不是**因为收尾：down 的 K=512 在 nk 上只有
  一轮宽步，浅栈的并行度靠 `WCOL=4`（一个 warp 认领 4 列）撑着。试过抬到 8
  （`063_nk5`）：**24.28 → 25.34**，更差——块数从 64 掉到 32，每块的启动与共享区
  装载摊不薄，多出来的在飞请求也没换来带宽。已回 4，并在常量处留了这段账。

### 12.18.4 由此产生的下一批刀（按时间轴排序，不再按 BENCH）

1. **down 的 nk 内核（K=512，2.25 ms/步）**：唯一还比 kn 慢的核。方向不是加列并行
   （已证亏），而是让浅 K 的 **32 条通道各读 32 字节**（`VB=32`，一轮 1 KB），把
   「一列一轮」变成「两列一轮」，同时不改变「一条指令一个连续段」这条已被验证的规律。
   这需要 `I8_NK_WIDE_K` 之上再加一档 `VB`，属于改内核形状而不是调常量。
2. `gemv_nk_i8`(3088) 2.37 ms、`delta` 1.70、`gemv_nk_i8`(512) 1.57、
   `moe_router_t` 1.50、`p2p_combine` 1.41 —— 顺序与 `tr9` 一致，仍按「每刀 0.3–0.5 ms」
   估。
3. 距离 16.67 ms 还差 7.7 ms，等于把每层发射从 26 压到 12；单靠「省发射」拿不到，
   必须继续按 §12.17 的思路改内循环形状。

**方法论补充一条**：本节两次「开不起来」都不是内核的问题，而是**开关的下游没接上**
（加载器读哪份键、驱动写死 kn、Python 侧自查退出）。以后动任何一个「布局级」开关，
先在三个层面各问一次「这个值传到内核了吗」：读盘键 / 驱动常量 / Python 守卫。

### 12.19 收尾并行到底是哪一环：寄存器读数（不是寄存器）

**问题**：nk 收尾从「`lane == 0` 串行读 16 个 int32」改成「行摊到 lanes、warp 归约」
（`065_nk7` / `066_trnk3`）后，down 自己的时间轴 56.15 → **47.24 us（−16%）**，但整体
墙钟两次都是 +0.3 ms。当时的第一嫌疑是寄存器：收尾里多出来的 64 位索引与 5 级归约把
`numRegs` 顶过一档、主循环可驻留块数掉了。

**做法**：不再靠墙钟推，直接问驱动。`i8_nk_report<XT,VB,WCOL,KCOMP>(tag, threads)`
（`moe_grouped_i8.cu`，nk 内核定义之后）用 `cudaFuncGetAttributes` +
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` 打一行，`COFFIN_KATTR=1` 门控、
每进程每实例一次；探针挂在 4 个发射点（`moe_grouped_i8` 的 gu wide `wcol=1`、
gu shallow `wcol>1`；`moe_grouped_i8_merge_tokens` 的 down `wcol>1`、down `wcol=1`）。

**读数**（`070_nk11`，行并行收尾版本，rank0 的 `nk1_d0.log`）：

```
[kattr] nk down wcol>1 regs=39 smem=16 maxthr=4096 threads=256 occ=25 blk/sm
```

**结论：不是寄存器。** 39 个寄存器在 256 线程/块这一档离任何占用率台阶都很远；
每 SM 可驻留 **25 个块**（25 x 256 = 6400 线程，已经越过常见的 2048 线程/SM 档，
说明这个 SM 的线程上限本身更高）——寄存器与块数都不是约束。既然占用率不是瓶颈，「收尾快 9 us 却整体慢 0.3 ms」
就不能记在寄存器头上——它只能是 BENCH 的 ±0.3 ms 抖动。同版本两次读数
**24.56 / 24.56**，而串行版 `068_nk9` 是 **24.75**，行并行版并不更差。

**因此保留行并行收尾**（用户两次否决回退，这个读数支持那个判断）

- 收尾并行 + span=2（当前）：`065_nk7` 24.56、`070_nk11` **24.56 / 40.71**，down 47.24 us；
- 收尾串行 + span=2：`068_nk9` 24.75 / 40.41，down 56.15 us。

**顺手确认的另一件事（`067_nk8` 已否掉）**：down 取 1 片（`span = (K <= 32*vb) ? 1 : 2`
让 `per == STEPK == 512`、宽步循环进得去）算下来 `24.6 / 40.65`，但 **GEN32 文本退化**
（`"The 1900s"` 连发 5 次）。结构上说得通、数值上没对齐，已回 2 片并在两处
（`moe_grouped_i8_merge_tokens` 与 `moe_grouped_i8_merge`）的注释里留账。
→ 这条也说明**只看 BENCH 会漏掉文本坏掉**：tok/s 一点没掉，输出已经崩了。
以后每刀都要带 GEN32 文本看。

**方法论补两条**
1. 想判断「是不是占用率/寄存器」，用 `cudaFuncGetAttributes` +
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 直接读，一次两行，比任何墙钟推断都硬；
2. 探针的 printf 落在**推理服务进程**的 stdout（`occ5/logs/nk1_d0.log`），不在
   `sandbox_chk/logs/<TAG>.out` 里——找错地方会以为探针没生效。

### 12.20 密集投影 `gemv_nk_i8`：先否掉「展开度」，再动「每次请求的大小」

**起点（设备侧读数，`072_trnk4`，5 个 decode 步）**：同一实例 `gemv_nk_i8_kernel<uint16_t,4,4,1>`
按 grid 分三档，每发 90.05 / 36.92 / 24.74 us。最大那档 grid 是 `(3088,1,1)`，128 线程，
即每个 warp 认领一列、N = 3088 x 4 = 12352，K = 2048 ⇒ 权重 25.3 MB。
**25.3 MB / 90 us = 281 GB/s** —— 这在这张卡上是很低的一档。

**第一刀：抬展开度（`071_u8` / `073_trnk5`）**，`COFFIN_GEMV_NK_UNROLL=8`（新加的 env，
Python 侧透传给 `gemv_nk_i8_m` 的 `unroll`，内核里走 `U = 8` 的现成实例）：

| grid | U=4（trnk4） | U=8（trnk5） |
|---|---|---|
| `(3088,1,1)` | 90.35 us | **89.82 us** |
| `(512,1,1)`  | 36.91 us | 37.79 us |
| `(256,1,1)`  | 25.00 us | 25.71 us |

核自己几乎没动（两档还略差）。BENCH `071_u8` = **24.28 ms / 41.18**、GEN32 41.66、文本
正常——与 `061_nk4` 的 24.28 / 41.19 并列，所以这点差别在抖动带里，不能记成收益。
**结论：它不缺在飞请求**（U 从 512 B/warp 抬到 1 KB/warp 没有任何变化），env 已回 0。

**第二刀（`074_vb16`，改内核形状）**：4 字节步下每个 warp 一步只有 **128 字节**连续段，
DRAM 侧不成段，这才是 281 GB/s 的解释。于是把 nk 密集 GEMV 的每 lane 读**从 4 字节加宽
到 16 字节**（warp 一条 512 字节连续段，权重一次 `uint4`、激活一次两条 16 字节），
与 §12.18 里让 gate_up 掉 42% 的那一招同构。条件：`K % 512 == 0` 且激活基址 16 字节对齐，
否则逐字退回原来的 4 字节步（任何形状都仍然正确）。位置
：`gemv.cu` 的 `gemv_nk_i8_kernel` 主体（`wide16` 分支）+ `NkLoad::sixteen`。

**未完成**：`074_vb16` 的结果没读回来——scow 的接入网关（`wvpn.buu.edu.cn`）报
`SSLEOFError`，连续两次读日志都断在 `getAppInitialConfig`。作业本身已入队、在被远端执行，
下次能连上时第一件事就是读 `logs/074_vb16.out`（先看 `BUILD_S` 有没有过去，再看 BENCH/GEN32）。

**方法论**：这条链说明了「抬并行度」与「抬请求大小」是两个不同的旋钮——在本卡上前者对
小核只值 ~20%（§12.9）、对这里**几乎是 0**；真正决定带宽的是**一条指令运多少字节**。

### 12.21 宽步的根因：不是寄存器，也不是溢出，是「每 lane 请求宽度」本身

`074_vb16` 那一刀（nk 密集 GEMV 每 lane 从 4 字节抬到 16 字节）实测 **24.28 -> 38.79 ms、
tok/s 41.18 -> 25.78、文本坏成混语乱串**。按「别撤，抓出来具体原因」的要求，做了三步读数：

**第一步：换紧凑写法，排除「临时数组挤爆寄存器」。** 第一版是 `int8_t w16[16]` +
`XT x16[16]`（M = 4 时每线程 64 字节权重 + 64 个 XT）。改成权重只占 4 个 `unsigned`
（一次 `uint4` 读）+ 激活仍每 4 个 K 元素读 8 字节的紧凑版（`076_vb16b`）：
**24.28 -> 38.84 ms，几乎没变**。可见临时数组不是原因。

**第二步：直接读属性，排除溢出与占用率。** 新增 `nk_gemv_report<XT,U,M,W>`（与
`i8_nk_report` 同一个套路，`COFFIN_KATTR=1` 门控）打在 `launch_nk_i8_one` 里：

```
[kattr] gemv_nk_i8 regs=24 local=0 smem=0 threads=128 occ=64 blk/sm
[kattr] gemv_nk_i8 regs=49 local=0 smem=0 threads=128 occ=39 blk/sm
[kattr] gemv_nk_i8 regs=77 local=0 smem=0 threads=128 occ=25 blk/sm
```

`local=0` —— **没有任何溢出到本地内存**；occ 25~64 blk/sm，占用率也远不是约束。
所以「寄存器/溢出/占用率」这条解释链整条断掉。

**第三步：量那个核自己的时间轴（`077_trnk6`，与 `072_trnk4` 同口径）。**

| grid | 4 字节步（trnk4） | 16 字节步（trnk6） |
|---|---|---|
| `(3088,1,1)` | 90.35 us | **323.11 us（3.6x）** |
| `(512,1,1)`  | 36.91 us | 90.97 us |
| `(256,1,1)`  | 25.00 us | 37.33 us |

**结论（这就是具体原因）**：这张卡在**这个访问模式**上不奖励更宽的每 lane 请求。
每行权重是一条独立流（相邻 warp 的行相隔 K = 2048 字节），4 字节 x 32 lane 给出的
**128 字节段**能被 DRAM 交错得更满；抬到每 warp 一条 **512 字节段**，可交错的段数直接
砍掉 4 倍，于是同一个核慢了 3.6 倍。与 §12.18 里 MoE gate_up 的 −42% 并不矛盾：那边的
`[E,N,K]` 是**同一份专家栈里连续的多行**，加宽后仍是同一条流里的更大段；这里每 warp
本来是**一条独立流**，加宽等于把 4 条流并成 1 条。

**收口**：宽步实现保留、由 `COFFIN_GEMV_NK_WIDE=1` 门控、**默认关**；`078_wideoff` =
**24.36 ms / 41.05**、GEN32 41.49、文本正常。`nk_gemv_report` 长期保留。

**方法论第三条**：「一条指令运多少字节」不是单调越大越好——**先看这条流是不是和别的流
共享同一个连续区域**。共享（同一个栈里连续的行）→ 加宽有效；各自独立 → 加宽是自伤。
