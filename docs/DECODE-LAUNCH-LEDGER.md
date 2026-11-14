# Decode 发射账本与融合路径（occamy-1.0 W8A8 / EP4 / 单流）

本文记录 **2026-11-14** 在 gpu-node-17 `sandbox4` 上实测的 decode 发射构成、常驻网格栅栏的
可用性结论，以及把发射压到极值的两条路线与各自的确切障碍。文中的数字全部来自当次
`ixprof`/`pdprof` 采集，不是估算；标「待核」的条目是尚未逐个读源码确认的几何参数。

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
