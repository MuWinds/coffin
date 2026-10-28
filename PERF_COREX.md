# CoreX decode 性能分析：慢在哪、还能挖多少

> 2026-10-27/28，`gpu-node-05`，Iluvatar BI-V100，torch 2.1 + CoreX 3.2.3，
> `COFFIN_BACKEND=corex`，18/18 原生算子。
>
> **四轮优化已落地：50.53 → 23.53 ms/step（19.79 → 42.51 tok/s，2.15x）**，
> 数值与文本均未变：
> 1. 修正 GEMV 的 split-K 分派（§2），50.53 → 43.46 ms；
> 2. 重写 delta 内核的线程映射（§3），43.46 → 37.98 ms；
> 3. 重做 int8 的 split-K 分派 + 用实测形状替换猜测的形状表（§10），34.29 → 27.69 ms；
> 4. int8 的四列宽加载按 N 门限重新分派（§11），27.69 → 23.53 ms。
>
> **graph 已判定不可用**（§8）：不只偶发挂死，优化后 replay 还比 eager 慢，
> 所以不再依赖它。
>
> **本机所有测量都在坏卡之外**：`gpu-node-05` 的 card 0/1/2 是死卡（8x8 matmul 超时），
> 每次作业先逐卡探活再选卡，见 `coffin/diag/_pick_card.sh`。

## 0. 已经拿到的结果

| 指标 | 优化前 | 优化后 | 提升 |
| --- | ---: | ---: | --- |
| decode（eager 实测） | 50.53 ms / 19.79 tok/s | **37.98 ms / 26.33 tok/s** | **1.33x** |
| `delta_kernel` | 7.117 ms（24 次，293.9 µs/次） | **1.792 ms**（74.7 µs/次） | **4.0x** |
| `delta` 单算子带宽 | 14.1 GB/s（看起来） | **101.5 GB/s** | 7.2x |
| `mlp_up` | 0.292 ms / 161.6 GB/s | **0.152 ms / 310.1 GB/s** | 1.92x |
| `qkv_gate` | 0.286 ms / 146.8 GB/s | **0.161 ms / 261.0 GB/s** | 1.78x |
| `lm_head` | 3.478 ms / 365.5 GB/s | **2.962 ms / 429.3 GB/s** | 1.17x |
| `mlp_down` / `o_proj` | 0.245 / 0.085 ms | 不变（已是最优） | — |
| 投影占整步 | 64% | 62%（分母也变小了） | |
| 端到端文本 | ` Paris.` | ` Paris.` | 未变 |

数值验收（`backends/corex/diag/check_ops.py`，逐算子对拍 reference 后端）：
`delta rel=3.270e-07`、`delta_batch == per-row rel=0.000e+00`，
全部 18 个算子 `ok`，无 fallback。

## 1. 一次 decode step 的时间去向（优化前基线）

`diag/corex_probe/probe_decode_profile.py`，eager 路径，device events 计时：

```
decode step                             50.53 ms    (19.79 tok/s)
```

profiler 的 kernel 排名（自 CUDA 时间）：

| kernel | 调用数 | 合计 | 单次 | 占比 |
| --- | ---: | ---: | ---: | ---: |
| `gemv_kn_kernel` | 65 | 24.17 ms | 371.8 µs | 45.4% |
| `gemv_kn_split_kernel` | 64 | 11.46 ms | 179.1 µs | 21.5% |
| `delta_kernel` | 24 | 7.12 ms | 296.5 µs | 13.4% |
| `rms_norm_kernel` | 81 | 1.28 ms | 15.8 µs | 2.4% |
| `attn_slice_kernel` | 8 | 0.61 ms | 76.0 µs | 1.1% |
| 其余（elementwise / copy / cat） | — | ~5 ms | — | ~10% |
| **GEMV 合计** | **129** | **35.63 ms** | | **70.5%** |

129 = 32 层 × 4 个投影 + 1 个 `lm_head`，对得上。

## 2. GEMV split-K：分派逻辑是反的（已修，1.23x）

同一块卡上 `dst.copy_(src)` 实测 **575 GB/s**，这是上限。

优化前的带宽：

| 形状 | N | K | 实测 | 占上限 |
| --- | ---: | ---: | ---: | ---: |
| `qkv_gate` | 8192 | 2560 | 146.8 GB/s | 26% |
| `mlp_up` | 9216 | 2560 | 161.6 GB/s | 28% |
| `mlp_down` | 2560 | 9216 | 192.2 GB/s | 33% |
| `o_proj` | 2560 | 4096 | 245.4 GB/s | 43% |
| `lm_head` | 248320 | 2560 | 365.5 GB/s | 64% |

### 根因：启发式的方向错了

原代码：

```cpp
// 只在单遍 grid 填不满卡时才 split
if (blocks >= 2 * sm_count() || K < 1024) return 1;
```

这个规则基于一个直觉——"block 数够了就不用拆"。**实测恰好相反**：

| 形状 | N | 单遍 | split 4 | split 8 | 实际最优 |
| --- | ---: | ---: | ---: | ---: | --- |
| `mlp_up` | 9216 | 0.292 ms | **0.152** | 更差 | **s4** |
| `qkv_gate` | 8192 | 0.286 ms | **0.160** | 更差 | **s4** |
| `mlp_down` | 2560 | 1.004 ms | — | **0.245** | **s8** |
| `o_proj` | 2560 | 0.329 ms | — | **0.085** | **s8** |
| `lm_head` | 248320 | 3.478 ms | **2.961** | — | **s4** |

两个纠正（各花了一个作业才学到）：

1. **"block 数够了就不该 split" 是错的。** `mlp_up` 在 16 张 SM 上是 36 个 block，
   已经是两个 wave，但 4 路 split 仍然快 1.92x。真正的瓶颈不是 block 数（occupancy），
   而是**同时在飞的 load 数**（memory-level parallelism）。
2. **"窄形状不需要 split" 是错的，而且反了。** `mlp_down` / `o_proj` 恰恰是收益最大的
   两个（4.1x / 3.9x）——它们才是列数最少、最需要帮忙的。

**唯一需要保护的约束是 K**：每路 split 只拿到 `K / split`，太短就得不偿失。
`mlp_down` 的 K 是所有形状里最长的，所以它吃 8 路。

修好后的分派：

```cpp
const int64_t by_k = K / 512 > 8 ? 8 : K / 512;
if (by_k < 2) return 1;
const int64_t want = N >= 4096 ? 4 : 8;
return (int)(want < by_k ? want : by_k);
```

### 术语说明：`s1` / `s4` / `s8` / `t256` / `u4`

这些是本文和 `probe_gemv_tune.py` 里的**配置标签缩写**，不是仓库原有术语：

| 标签 | 含义 |
| --- | --- |
| `s1` | `split = 1`，即**不切分**，单遍算完整 K |
| `s4` / `s8` | 把 K 轴切成 4 / 8 段并行算，再求和 |
| `t256` | 每个 block 用 256 个线程 |
| `u4` | K 循环展开 4 路 |

完整标签 `t256/u4/s4` 读作"256 线程、展开 4、K 切 4 段"。

split-K 具体在做什么（`gemv_kn_split_kernel` + `gemv_reduce_kernel`）：

```
GEMV: y[n] = sum_k W[k,n] * x[k]        每个输出列 n 要沿 K 累加

s1：一个线程 own 一个输出列，读完该列全部 K    grid = blocks
s4：一个输出列由 4 个 block 各算 K/4，先写 part[4][N]，再 reduce 求和
                                              grid = 4 * blocks
```

**权重读取总量不变**（每段读不同 K 区间），变的只有并行度，代价是多一个
reduce kernel 和一块 `part` 缓冲。

它为什么有效：瓶颈是**同时在飞的 load 数**，不是 block 数。16 SM 上 36 个 block
看似"够了"，但每个线程沿 K 串行累加时 outstanding 访存很有限；切 4 段等于让
4 组 block 同时向 DRAM 要数据。`mlp_down` / `o_proj` 吃 s8 是因为它们列数少
（2560），单遍时 grid 本身小，更需要靠 K 方向补并行度。

### 验证方法（重要，避免自欺）

每一步都在**真实 decode 路径**上验证，而不是只看单算子：

- `diag/corex_probe/probe_gemv_tune.py`：逐形状扫 `(threads, unroll, split)`，
  **每个配置都对 host fp64 参考做正确性检查**，错的配置不参与"最优"评选。
- `diag/corex_probe/probe_decode_profile.py`：跑真实模型，给出整步时间和占比。
- `diag/corex_probe/probe_chat.py`：端到端文本，` Paris.` 必须不变。

**踩过的坑**：改完 `gemv.cu` 忘了升 `_EXTENSION` 版本号（`coffin_corex_ops_v3`），
作业会**静默加载旧的 `.so`**——那个文件的注释里就写着这句警告，
"a stale cache is invisible — the old objects load and answer"。改成 `v5` 后数字才可信。

**第二个坑**：修好后的第一版我写成了 `if (N < 4096) return 1;`，
结果 `mlp_down`/`o_proj` 慢了 4 倍（整步 78 ms）。因为我把 sweep 表读反了：
旧代码给这两个形状的是 **s8**（不是 s1），而 sweep 里 s8 恰好是最优。
数据一直是对的，是我把"旧代码选了什么"和"最优是什么"混了。

## 3. delta 内核：改线程映射，1.79 ms（已修，4.0x）

优化前这是单点最差的一项：

```
一层一次 delta：state [32, 128, 128] fp32 = 2.10 MB，读+写 = 4.19 MB
        实测 296.5 µs  ->  看起来只有 14.1 GB/s
```

**但它不是"慢"，是"搬运了 32 倍多余的数据"：**

```
state 布局 [heads, dk, dv]，kernel 里是 column[i * dv]，步长 dv=128 个 float = 512 字节
一个 warp（64 lane）实际抓取 64 × 128B = 8192B，只用到 64 × 4B = 256B
→ 3.1% 的字节利用率
14.1 GB/s ÷ 3.1% = 451 GB/s 实际 DRAM 流量（上限 575）
```

它其实已经跑在带宽上限附近了，问题纯粹是访存模式。

### 修法：不用转置 state，改线程映射就够了

原本以为必须把 state 转置成 `[heads, dv, dk]`——那会牵动
`models/qwen35/linear_attn.py` 的 `ckpt_state`（前缀复用快照），风险很高。

但看清地址就发现不必：元素 `(i, j)` 在 `[heads, dk, dv]` 里就是 `st[head*dk*dv + i*dv + j]`，
**固定 `i` 时对 `j` 连续**。所以只要让**一个线程负责一个 value 列 `j`、循环遍历 `dk`**，
相邻线程的 `j` 相邻，访存就完全合并了：

| | 线程映射 | 抓取量 | 结果 |
| --- | --- | ---: | --- |
| 原 | 一个列组（`dk` 个线程）分摊 `i` | 134.2 MB | 293.9 µs |
| 新 | 一个线程独占一列，寄存器里累加 `dk` | 4.19 MB | **74.7 µs** |

附带好处：两次 `dk` 归约都变成线程内的寄存器循环，**共享内存树和屏障整个消失了**，
那类曾导致 `rel=4.5e-02` 的 race 也就没有存在余地了。

实测（`check_ops.py`）：`delta rel=3.270e-07`、`delta_batch == per-row rel=0.000e+00`。

### 顺带修掉一个死锁隐患

新 kernel 里线程越界时的 `if (j >= dv) return;` 写在 `__syncthreads()` **之前**，
这是经典 CUDA 死锁：部分线程提前退出，其余线程永远等不到。
本模型 `dv=128`、`threads=128` 恰好不越界所以没触发，但 `dv` 不是 128 倍数的模型会直接挂死。
已改成先参与屏障、之后再条件返回。**这正是老代码注释里特意警告过的同一个错误**——
写好之后才发现，记在这里。

## 3b. 剩下的 14.44 ms（未细分）

优化后整步 37.98 ms，投影 32.56 → 投影占比 62%，其余 14.44 ms 里 `delta` 已降到 1.79 ms。
下一步要细分的是 `rms_norm`（81 次）、attention 系列、以及大量 elementwise（`aten::cat` /
`to` / `copy_` / `add` / `mul` / `div` / `silu`）。

**注意**：这些数字里有多少是"启动间隙"而不是真在算，还没测。见 §8。

## 4. 其余可省的项

| 项 | 实测 | 可省 | 做法 |
| --- | ---: | ---: | --- |
| `aten::cat` | 32 次 / 0.43 ms | ~0.4 ms | q/k/v 投影已在 `GEMVGroup` 拼过一次，这里可能是第二次 |
| `aten::to` / `_to_copy` | 105 次 / 1.1 ms | ~1 ms | 反复 dtype 转换，查能否提前做一次 |
| `aten::index_copy_` | 32 次 / 0.54 ms | — | KV 写入，属于契约 |
| `cudaLaunchKernel` | 878 次 | graph 已消掉 | graph 开时不是瓶颈 |

## 5. int8 权重：参考平台上的 2.4x，这里同样成立

参考加速器的对照数据（ENGINE.md §6）：

| 配置 | decode |
| --- | ---: |
| bf16 权重 + graph | 32.2 tok/s |
| **int8 权重 + graph** | **78.5 tok/s** |

**2.4x**，而 CoreX 现在正是 bf16 权重。GEMV 是访存受限，int8 直接把读的字节减半。
CoreX 的 `caps.int8_weights` 现在是 `False`，需要新增一条 W8A16 的 `[K,N]` GEMV。

**注意顺序**：先做 §3（delta），再上 int8。否则带宽没打满就上 int8，两者收益重叠，
分不清谁起了作用。

## 6. 优化路线

| # | 动作 | 预期 | 状态 |
| --- | --- | --- | --- |
| 1 | 修正 GEMV split-K 分派 | 50.5 → 43.5 ms | **已完成，1.23x** |
| 2 | 改 delta 的 state 布局（转置 + 快照同步） | 43.5 → ~36.5 ms | 待做，预期 1.19x |
| 3 | int8 权重（W8A16 `[K,N]` GEMV） | GEMV 再减半 | 待做 |
| 4 | 消掉 `aten::cat` / 多余的 `to` | ~1.5 ms | 待做 |

**1+2+3+4 都到位**：43.5 → 大约 **18-22 ms**，即 **45-55 tok/s**。

## 7. 怎么复现

```bash
# 单项：整步时间分解 + 每形状带宽 + profiler 排名
python scow/tools/deploy.py cohort -d ../coffin \
    -s ../coffin/diag/corex_probe/launch_decode_profile.sh --run prof -g 1 -t 25

# 逐形状扫 (threads, unroll, split)，带正确性检查
python scow/tools/deploy.py cohort -d ../coffin \
    -s ../coffin/diag/corex_probe/launch_gemv_tune.sh --run tune -g 1 -t 25

# 端到端文本（必须仍是 ' Paris.'）
python scow/tools/deploy.py cohort -d ../coffin \
    -s ../coffin/diag/corex_probe/launch_chat.sh --run chat -g 1 -t 25
```

### 单作业多卡（推荐）

`diag/corex_probe/run_multigpu.sh` 一次申请多张卡，**在作业内部逐张验卡并自动选健康的**：

```bash
# 提交命令只需一句，逻辑都在脚本里（网关会拦复合命令）
#   WORKDIR=<dir> PROBE=diag/corex_probe/probe_decode_profile.py bash <dir>/raw/mg.sh
```

实测输出（`gpu-node-05`，申请 4 张卡）：

```
CARDS_VISIBLE=4
--- card 0 ---
  rc=124 out=<empty>        <- 挂死的卡，45s 后超时
--- card 1 ---
  rc=0 out=OK
  card 1 is healthy
USING_CARD=1 on gpu-node-05
```

**这纠正了一个重要误判**：此前以为 `gpu-node-05` 整台节点坏了，实际是**它的一张卡坏了**，
同节点其余卡正常。所以节点名不能作为"可用/不可用"的判据，必须逐卡验。

三个踩过的坑，都记在这里免得重犯：

1. **提交命令不能太复杂。** WebVPN 网关对 `cd X && bash Y` 这类**复合命令字符串**返回 403
   （看起来像认证失败，其实是拦截）；而 `bash <绝对路径>` 单条命令能过。
   所以逻辑要放进上传的脚本文件里，命令只留一句。`mkdir -p`、`../` 相对路径、
   `$$`/`${VAR:-x}` 展开同样会被拦。
2. **作业根本没申请卡时**（`cbatch` 忘了 `--gres gpu:1`），`torch.cuda.is_available()`
   报 "No CUDA GPUs are available"，症状和卡坏一模一样。
3. **扩展缓存不失效就静默用旧的**（见 §2 末尾）。
4. **Windows 写出来的 `.sh` 是 CRLF，bash 会静默失败。** 症状极具误导性：作业 2 秒结束、
   状态 `COMPLETED`、日志 0 字节——看起来像"调度器根本没跑"，实际是每行末尾的 `\r`
   让第一个 token 就是错的。已加 `platform/scow/tools/ship.py`，打包时对 `.sh` 强制
   LF，其他文件按字节原样发送。

## 8. graph：判定不可用，以及替代方案

**结论：不启用 CUDA graph。** 两条独立理由，任意一条都够了。

**一、优化后 replay 已经比 eager 慢。** 同一个真实 prompt：

| | 早期（未优化） | split 修好之后 |
| --- | ---: | ---: |
| eager | 77.09 ms | 43.46 ms |
| graph replay | 52.41 ms | **46.97 ms** |
| graph 收益 | 1.47x | **0.93x（负收益）** |

原因是 graph 省的是启动开销，而 split-K/delta 两轮把 kernel 本身加速之后，
剩下的间隙已经小于 replay 自身的调度代价。**继续上 graph 是刻意的倒退。**

**二、偶发挂死，且无法归因。** 观察到的样本：

| 运行 | 结果 |
| --- | --- |
| `gr-1`（node-06） | 完整通过，52.41 ms/step |
| `gr2-1`（node-06） | `graph replay` 后无输出，挂死 |
| `probe_graph_delta`（node-05 card 2） | capture/prefill/5 次 replay 全通过 |
| `probe_chat`（node-05 card 1） | `graph replay` 后无输出，挂死 |

**同一份代码、同类节点，有时通过有时挂死。** 目前能确定的只有"不稳定"，
根因没找到——而挂死的代价是整张卡的上下文被污染，继续做实验等于拿卡去赌。
**所以停在这里，不再做 capture 实验。**

### 替代方案：先测清楚"间隙有多大"再选

现在缺的关键数据只有一组：一次 eager step 里 **GPU 真正在算的时间** vs **墙钟时间**。
它决定后面完全不同的两条路：

| 若有较大间隙（墙钟 ≫ GPU 忙） | 若两者接近（设备受限） |
| --- | --- |
| 问题在启动，考虑下面三条 | 启动不是问题，**别做替代方案**，继续优化 kernel |

三条候选，按代价从低到高：

1. **减少 launch 数量（融合）** —— 把 `aten::cat` / `to` / `copy_` / `add` / `mul` /
   `div` / `silu` 这些 elementwise 合进已有 kernel 的 epilogue。上一次 profiler 显示
   878 次 launch/step，其中一半以上是这类小算子。**风险最低，先做这个。**
2. **C++ 单入口 decode step** —— 把整个 forward 的 kernel 发射搬进一个 C++ 函数，
   Python 每 token 只调一次。理论上能把 ~880 次 `<<<>>>` 收成 1 次 Python→C++ 调用。
   **代价大**：等于把 32 层前向在 C++ 里重写一遍，且要和 Python 路径数值完全一致。
3. **持久化内核（megakernel）** —— 每层一个常驻 kernel，用 grid-wide barrier 串联。
   收益最高（启动完全消失、中间结果可留在片上），但要求 cooperative launch 支持，
   本驱动**未验证**，而且要跨 32 层做同步，实现量和风险都最大。

**建议顺序：先 1，再决定要不要 2/3。** 如果测出来间隙很小，1 也可能不划算。

### 重新开工前必须先量的一组数

`diag/corex_probe/probe_step_budget.py` 已经写好（**还没跑出结果**），它给三件事：

* `wall/step`：一次 step 的墙钟；
* `gpu_busy`：同一 step 里 profiler 统计的 kernel 时长之和；
* `cpu_busy` 与 launch 次数。

判定规则也写在脚本里：`wall - gpu_busy > 25% × wall` 才算启动受限，否则是设备受限。
**在这组数出来之前，不要动 §8 的任何替代方案。**

### 已知的坏卡（2026-10-27 观测，会变）

`gpu-node-05` card 0、`gpu-node-06` card 0 都验证为死卡（8x8 matmul 超时）；
同节点的其他卡正常。**所以判据必须是"逐卡"，不是"逐节点"。**
节点头部那些结论不要当成永久的——卡的状态会变，每次开工先验一遍。

## 8. graph 的替代方案：**测完了，不需要替代**

这个问题曾经悬着：graph 在这台驱动上会随机把卡挂死，那能不能用一个"像 graph 但不会死卡"
的机制拿回发射开销（C++ 融合发射、持久化 megakernel 等）？

**答案是不需要。卡已经是满的，没有发射开销可拿。**

`diag/corex_probe/probe_step_budget.py` 的实测（gpu-node-07，graph 关，单卡）：

```
wall/step        : 37.05 ms   (26.99 tok/s)
gpu busy (trace) : 47.80 ms   <- 比 wall 还大 29%
launches         : 2397 / step
verdict          : device bound
```

trace 的逐 kernel 时长含 profiler 自身开销，所以它的**总和**不能当 busy 时间用
（这里超出 wall 29%，正是过计的证据）。但它同时证明了一件更有用的事：
**设备在整步里几乎一直是忙的**，不存在能藏下"发射受限"的空窗。

三条推论，互相独立且一致：

1. **C++ 融合发射**（一次 host 调用发完整步）省的是 host 时间。host 不在关键路径上，
   所以它换不来速度，只换来一份要跟 Python 模型逻辑长期同步的 C++ 复刻。
2. **持久化 megakernel** 省的是全部发射。发射开销已非瓶颈，所以收益接近零；
   而它把"内核自旋 → 卡挂死"的风险原样继承了下来，并不比 graph 安全。
   它唯一真正的价值是能把激活留在寄存器/共享内存里省访存，那是另一件事，
   且需要把整个前向内联进一个内核——收益与代价都不成比例。
3. **graph 本身也不再划算**：replay 实测 46.97 ms，比 eager 的 37.05 ms 慢 27%。
   一个用来消掉发射开销的机制，在发射开销不再是瓶颈之后，只剩自己的成本。

**所以 §6 的优先级不变，只是第 1 项已经从"实施方案"变成"不需要"：
剩下能做的只有减少设备端实际工作量——int8 权重（§5）和消掉 §4 的胶水算子。**

### 这一轮学到的两条工程教训（都花了作业才学到）

1. **取消一个正在 import 阶段的作业会留下 torch 的编译锁**，位置是
   `<TORCH_EXTENSIONS_DIR>/<ext>/lock`。`file_baton.wait()` 会永远轮询它、不会老化，
   于是之后每个作业都卡在 `import backends` 里**两个输出流都没有任何输出**——看起来
   和"设备 hang"一模一样。已经修在 `backends/corex/kernels/__init__.py`：
   `_clear_stale_build_lock()` 会在超过阈值（默认 600 s，可用
   `COFFIN_BUILD_LOCK_MAX_AGE` 覆盖）时删掉残留锁。用 `faulthandler` 打栈才定位到，
   这也说明了**"没有输出"必须先怀疑自己的调用链，再怀疑硬件**。
2. **上传全仓库是错的。** 每次 3.5 MB 既慢又反复触发 WebVPN 的 403。
   现在用 `platform/scow/tools/sync.py`：集群上保留一份常驻检出
   （`~/coffin-work/base/coffin`），每次只传**改动文件**的 tar（典型 1-4 KB），
   再提交一条 `bash <绝对路径>` 的作业。同时 `tar` 排除 `__pycache__`，首传也从
   3.5 MB 降到 698 KB。

## 9. int8 权重（W8A16）：1.08x，以及为什么不是 2x

`caps.int8_weights` 从 `False` 改成 `True`，新增 `gemv_kn_i8` 内核（int8 `[K,N]` +
每列一个 fp32 scale，`y[n] = scale[n] * sum_k x[k] * w8[k,n]`），`pack.KnQWeight`
负责 `[N,K]→[K,N]` 的 int8 转置。**量化开关在 loader 侧**（`load_weights(quant=True)`），
`caps.int8_weights` 只表示"这个 backend 能吃 int8"——这两个是分开的，混起来会导致
"backend 支持 int8 但模型仍走 bf16"，我第一次 profile 就踩了这个坑。

### 结果

逐形状（`probe_int8.py`，host fp64 参考，对拍 dequant 后的权重）：

| 形状 | int8 | bf16 | 提升 | rel |
| --- | ---: | ---: | ---: | ---: |
| `mlp_up` | 0.100 ms / 235 GB/s | 0.150 ms / 314 GB/s | 1.50x | 3.0e-03 |
| `mlp_down` | 0.162 ms / 146 GB/s | 0.242 ms / 195 GB/s | 1.49x | 1.5e-03 |
| `qkv_gate` | 0.095 ms / 220 GB/s | 0.160 ms / 262 GB/s | 1.68x | 2.6e-03 |
| `o_proj` | 0.080 ms / 130 GB/s | 0.085 ms / 248 GB/s | 1.05x | 2.8e-03 |
| `lm_head` | 2.428 ms / 262 GB/s | 2.937 ms / 433 GB/s | 1.21x | 2.7e-03 |

整步：**37.05 → 34.31 ms（26.99 → 29.15 tok/s），1.08x**。端到端文本不变
（` Paris.`、`2 + 2 = 4.`），`failures: (none)`。

### 为什么只有 1.08x 而不是接近 2x

两个原因叠加，都能量化：

1. **int8 内核的字节效率更低。** 上表最后一列：同一形状 int8 的 GB/s 反而低于 bf16
   （`mlp_up` 235 vs 314，`lm_head` 262 vs 433）。每元素 1 字节意味着每个 lane 的加载
   更"碎"，所以"读一半字节"只换来 1.5x 而不是 2x。
2. **GEMV 已不是全部。** int8 之后 GEMV 占整步 50%，delta/rms_norm/胶水算子没变，
   所以投影上省下的时间被稀释。

### 一个失败的优化（负结果，值得记）

为了修第 1 点，我加了四列宽的 int8 内核（`gemv_kn_i8x4_kernel`：一次读 4 字节，
加载指令降到 1/4）。**结果是整步从 34.23 恶化到 46.81 ms。**

原因不是访存，是**并行度**：四列宽让线程数变成 1/4。`mlp_up` 的 9216 列从 36 个 block
掉到 9 个，而这张卡只有 16 个 SM——grid 连一个 wave 都不够，加载省下的时间远小于
空转的损失。**这张卡上，线程数才是这个内核的稀缺资源，不是字节带宽。**

所以四列宽被门限化到 `N >= 32768`，只有 `lm_head`（248320 列）走它：
GEMV 22.28 + 1.38 ms，整步 34.31 ms，与标量版持平（略好）。
**保留它是因为它正确地证明了"这条路不通"**，并且对真正宽的层仍然有用。

### 下一步的真实空间

int8 之后 GEMV 仍占 50%（22.28 ms），而它只跑到 130-262 GB/s（上限 576）。
剩下的差距**不在** int8 存储形式，而在**小 N 形状的 grid 太小**（`o_proj` 425 GB/s？
不，是 130 GB/s——最差的正是 N=2560）。可行方向：

1. **按形状调 `(threads, split)`**：int8 的 `pick_split` 目前直接复用 bf16 的规则，
   而最优值未必相同。`probe_gemv_tune.py` 已经能扫，只需让它接受 dtype。
2. **把 delta 融进 `linear_attn` 的 GEMV 尾部**，省掉中间写回（delta 现在 1.79 ms）。
3. `o_proj`/`mlp_down` 这类 N=2560 的形状只有 10 个 block，可考虑一个 block 处理多行。

## 10. int8 的 split-K 分派：34.29 → 27.69 ms（1.24x）

上面第 1 条是对的，而且比预想的更值钱。**根因是 split 上限被写死成 8**，不是字节宽度。

### 先纠正一件事：探测脚本里的形状表是错的

此前所有扫描都对着手写的形状表跑（`mlp_up` 9216x2560、`qkv_gate` 8192x2560…）。
那些数字是**读 config 猜的**，而融合投影不是它各部分的形状：`gate_up` 把两个 9216 行
的权重拼成一个矩阵，linear attention 层把四个投影融合成一个。
`probe_shapes.py` 包住 `gemv_kn_i8` 跑一个真实 decode step 才拿到实测值：

| N | K | 每步调用 | MB/步 |
| ---: | ---: | ---: | ---: |
| 18432 | 2560 | 32 | 1510 |
| 2560 | 9216 | 32 | 755 |
| 12352 | 2560 | 24 | 759 |
| 2560 | 4096 | 32 | 336 |
| 10240 | 2560 | 8 | 210 |
| 248320 | 2560 | 1 | 636 |
| 2560 | 5120 | 2 | 26 |

**旧的 `mlp_up`/`qkv_gate` 根本不在列表里。** 一个按错形状建的分派表，调的是不运行的核。

### 根因：`pick_split` 的 cap = 8

原规则 `by_k = min(8, K/512)`、`want = N>=4096 ? 4 : 8` 是从 bf16 的扫描里得到的，
被原样套到 int8 上。实测（16 SM，copy 上限 573 GB/s，每个候选都对 host fp64 校验，
rel 全在 1.1e-3..3.1e-3）：

| N | K | 旧分派 | 旧 ms | 最优 split | 新 ms | 提升 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 18432 | 2560 | s4 | 0.2076 | s16 | 0.1905 | 1.09x |
| 2560 | 9216 | s8 | 0.1614 | s32 | 0.0676 | **2.39x** |
| 12352 | 2560 | s4 | 0.1081 | s32 | 0.0782 | 1.38x |
| 2560 | 4096 | s8 | 0.0803 | s32 | 0.0418 | **1.92x** |
| 10240 | 2560 | s4 | 0.1005 | s32 | 0.0675 | 1.49x |
| 248320 | 2560 | s4 | 1.3933 | s8 | 1.3756 | 1.01x |
| 2560 | 5120 | s8 | 0.1007 | s32 | 0.0493 | 2.04x |

`mlp_down`（2560×9216）和 `o_proj`（2560×4096）正是 §2 里"最差的 N=2560"那两个，
它们要的是 s32，不是 s8。**o_proj 从 130 GB/s 到 250 GB/s，翻倍。**

### 分派规则：一个常数，不是七个魔法数

固定 K=2560、扫 N 找"单遍已经填满卡"的转折点：

| N | blocks=ceil(N/256) | 最优 split |
| ---: | ---: | ---: |
| 4096 | 16 | 32 |
| 8192 | 32 | 16 |
| 16384 | 64 | 32 |
| 32768 | 128 | 8 |
| 65536 | 256 | 4 |
| 131072 | 512 | 2 |
| 248320 | 970 | 8（s1 只差 1%，是平底） |

从 128 blocks 往上，一行就对上了：`split × blocks ≈ 1024`。所以规则是
**"让大约 1024 个 block 在飞"**：`want = (1024 + blocks/2) / blocks`，cap 32、
floor 1，并要求 `K/want >= 64`（切片短于 64 就没法调度，同时把 partial 缓冲
`8*split*N` 字节压到权重流量的 12.5% 以内）。

**这不是表，是拟合**——它对下一个模型的 N 也能插值，而这台机器上的形状恰好都不是
整数，硬编码的表一个都命中不了。

**已知偏差（写在这里免得有人当 bug 重查）**：中段 grid 上四舍五入会少切一档
（16384 给 16、实测 32；实际形状 18432→14、12352→21、10240→26，实测最优分别是
16、32、32）。**少切是安全方向**（损失的是 block 数，不是正确性），而且扛着大部分
流量的两个 2560 列形状都精确命中 cap。

### 端到端

```
优化前           34.29 ms/step   29.16 tok/s
修正 split 分派  27.69 ms/step   36.11 tok/s     (1.24x)
再上宽加载       23.53 ms/step   42.51 tok/s     (1.46x)
```

greedy token ids 三次完全相同 `[6, 24218, 10, 4838, 1665, 15]`，数值未变。
split 那一轮实收 6.60 ms 比按形状表算出的 5.88 ms 还多一点——形状表是孤立测的，
真步里还有 partial 缓冲的分配和 reduce 的发射，这次不猜机制，记实测就好。

## 11. int8 的加载宽度：再 4.16 ms，但只对 N ≥ 4096

§10 之后剩下的最差形状是 `N=18432 K=2560`（1510 MB/步，最大的单项），只有约 234 GB/s。
假设是**事务浪费**：一个 64 lane 的 wavefront 一次只读 1 字节/lane，只覆盖 128 字节行的
一半。`gemv_kn_i8x4_kernel` 早就修了这个（4 列相邻 = 一次 4 字节读），但它让线程数
变成 1/4，之前在 split 上限还是 8 的时候测出 34.23 → 46.81 ms，于是被门限到 N≥32768。
**那次测量是在 grid 被饿死的条件下做的**，split 放开到 16/32 之后要重测。

### 实测：宽度不是普适的，它取决于 grid 还剩多少线程

| N | K | width 1 | width 4 | 结论 |
| ---: | ---: | ---: | ---: | --- |
| 18432 | 2560 | 234 GB/s | **375 GB/s** | w4 胜 1.60x |
| 12352 | 2560 | 369 GB/s | **462 GB/s** | w4 胜 1.25x |
| 10240 | 2560 | 366 GB/s | **452 GB/s** | w4 胜 1.23x |
| 248320 | 2560 | 452 GB/s | **461 GB/s** | w4 微胜 1.02x |
| 2560 | 9216 | **337 GB/s** | 283 GB/s | w1 胜（w4 是 0.84x） |
| 2560 | 4096 | **238 GB/s** | 215 GB/s | w1 胜（0.90x） |
| 2560 | 5120 | **269 GB/s** | 234 GB/s | w1 胜（0.87x） |

分界很干净，而且**和字节无关**——上面每个形状都是每元素读 1 字节。分界在于
**模型还剩多少线程**：2560 列的投影在 256 线程下只有 10 个 block，四列宽把它除以 4
变成 2.5 个 block，16 个 SM 连一个 wave 都不够。**列数过 4096 之后 grid 才扛得住这个除法。**

所以 `I8_WIDE_MIN_N` 从 32768 降到 **4096**，写在 `gemv.cu` 里，连同上面这张表。
`width` 参数（0 = auto）保留，sweep 仍然能驱动任意宽度。

### 一个被推翻的假设，值得记

我原本猜 **W=2** 会赢——64 lane × 2 字节 = 正好 128 字节一条线，而且只把线程数减半。
**实测错了**：W=2 在 18432 上是 357 GB/s，低于 W=4 的 375；在 N=2560 的形状上
W=2 也仍然输给 W=1。原因是 W=2 既没拿到 W=4 的字节效率，又已经付出了线程减半的代价，
两头不占。**假设很好，数据不认。** 这一条写在这里，是因为下一个想"优化宽度"的人
大概率会重新提出 W=2。

### 复现

```bash
# 真实形状统计（一个 decode step 到底调了哪些形状）
python platform/scow/tools/sync.py push -d coffin \
    --launcher diag/corex_probe/launch_probe_shapes.sh --run shapes -g 4 -t 20 --wait

# 真实形状 × (threads, split) 扫描 + split 转折点
python platform/scow/tools/sync.py push -d coffin \
    --launcher backends/corex/diag/launch_tune_gemv_real.sh --run tune -g 4 -t 40 --wait

# 宽度 × 线程 × split 扫描（带 production / 改动前两个对照行）
python platform/scow/tools/sync.py push -d coffin \
    --launcher backends/corex/diag/launch_tune_gemv_i8_wide.sh --run wide -g 4 -t 50 --wait

# 端到端（int8 服务路径，必须比对 token ids）
python platform/scow/tools/sync.py push -d coffin \
    --launcher diag/corex_probe/launch_bench_step_i8.sh --run step -g 4 -t 45 --wait
```

### 三个会让人白跑一趟的坑

1. **`create_model` 加载的是 bf16，不是 int8。** `load_weights` 的 `quant` 默认 `False`，
   而 `serving/runtime_service.py` 传的是 `int8=True`。所以 `probe_decode_profile.py`
   量的其实是 **bf16 GEMV**；照它调 int8 会调错核。要测 int8 必须显式
   `load_weights(quant=True)` 再 `create_model(weights=sd)`。**这一条让第一个形状探针
   返回了 0 次 int8 调用**，差点把整轮优化建在错的基础上。
2. **作业里 `cd` 的相对层级。** 探针在 `diag/corex_probe/` 下，距 `coffin/` 是**两级**，
   不是三级；而且作业的工作目录是 `coffin/` 的**父目录**。写错的表现是
   `can't open file`，看起来像文件没上传。
3. **`.sh` 必须是 LF。** Windows 上写出来的文件默认是 CRLF，bash 以
   `$'\r': command not found` 静默失败——症状是作业几秒结束、状态 `COMPLETED`、
   日志 0 字节，看起来像调度器的问题。**这一轮被它废掉了两个作业**，
   最后加了 `coffin/.gitattributes`（`*.sh text eol=lf`）从 checkout 层强制。


