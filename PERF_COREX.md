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
| 历史 graph 验收（node-05 card 2） | capture/prefill/5 次 replay 全通过；当前按驱动安全规则不再复跑 |
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

> **2026-10-28 更正 —— 上面这个"device bound"的结论是错的，见 §12.1。**
>
> 错在把 trace 的逐 kernel 时长**求和**当成设备忙时间。那个和不可加：profiler 的逐 kernel
> 时长**按 kernel 计**开销，所以过计量随**发射数**增长，而不是随工作量增长。一个
> 2407 次发射、求和比墙钟多 45% 的步，恰恰是"大部分时间花在启动工作"的样子。
>
> 直接测量（`probe_launch_floor.py`）给出的是 **5.75 us 的发射地板**：一个什么都做的
> 1 元素 kernel 也要这么多。2407 × 5.75 us = **13.8 ms = 59% 的墙钟**。
> 按 §12 把胶水算子融合掉之后实测 **23.53 → 20.26 ms（1.16x）**，发射数 2407 → 1399。
>
> **教训**：profiler 的时长可以作为**排序**依据，不能作为**占比**依据——它的每一项都被
> 计入了一次它自己带来的开销。要谈占比就单独测一个地板，或者用 events 计时。
> 不过 §12.3 又说回来：地板是**发射间隙**的地板，不是**工作**的地板，
> 所以"能少发一次就少发一次"也不成立，只有"工作也一起消失"才成立。

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

## 12. 胶水算子融合：23.53 → 20.26 ms（1.16x），以及两个被实测否掉的融合

> 2026-10-28，`gpu-node-05` / `gpu-node-06`，card 3 / card 1，int8 权重，eager。
> 起点就是 §0 的 **23.527 ms / 42.51 tok/s**，终点 **20.263 ms / 49.35 tok/s**。
> greedy ids 全程 `[6, 24218, 10, 4838, 1665, 15]` 未变。

### 12.1 先量"一次发射值多少钱"，这决定了后面所有取舍

`diag/corex_probe/probe_launch_floor.py`（作业 `floor2`，同一张卡）：

```
add_ bf16 x1      5.75 us        <- 发射地板：kernel 什么都不做也要这么多
add_ bf16 x1024   5.75 us
add_ bf16 x65536  8.33 us
add_ bf16 x1048576 13.47 us
```

一步 decode 有 **2407 个 profiler 行 / 946 个真 kernel 发射**，墙钟 23.53 ms，
即 **9.77 us/发射**。地板 5.75 us × 2407 ≈ **13.8 ms = 59% 的墙钟**。

**这条推翻了 §8 的判断。** §8 当时用 profiler 逐 kernel 时长求和得到 34 ms（> 墙钟 23 ms），
据此写下"device bound，去掉发射没用"。那个和**不可加**：profiler 的逐 kernel 时长本身
带开销，**按 kernel 计**，所以它的膨胀量随**发射数**增长而不是随工作量增长。
一个 2407 次发射、求和比墙钟多 45% 的步，恰恰是"设备大部分时间在启动工作"的样子。
**地板是直接测出来的，求和是间接推出来的；这里以直接测量为准。**

### 12.2 落地了什么

| # | 改动 | 省掉的发射 | 状态 |
| --- | --- | ---: | --- |
| B1 | `kv_quant`：k/v int8 量化七连（`abs/amax/clamp_min/div/round/clamp_/to`）→ 1 核 | ~100 → 16 | **落地** |
| B3 | `rope_apply`：`apply_rope` 的 `to/mul/cat/mul/add/cat` 串 → 1 核 | ~96 → 8 | **落地** |
| B2 | `silu_mul`：`F.silu(gate) * up` 两次发射 → 1 核，写回 `up` 自己的缓冲 | 64 → 32 | **落地** |
| D | `add_rms_norm`：残差 `x += h` 折进紧随其后的 RMSNorm | 32 | **落地** |
| C | `delta_normed`：prologue + delta + 门控 norm 三合一 | −48 | **实测更慢，已回退** |
| A | 把 split-K reduce 融进 GEMV split 核 | −128 | **实测更慢，已回退** |

逐算子实测（`probe_glue_ab.py`，**比的是比特相等而不是容差**）：

| 核 | 每次发射 | 原链 | 每步省 |
| --- | ---: | ---: | ---: |
| `kv_quant` | 8.3 us | 68.2 us | 0.96 ms（16 次） |
| `silu_mul` | 8.6 us | 17.1 us | 0.27 ms（32 次） |
| `rope_apply` | 11.1 us | 136.4 us | 1.02 ms（8 次） |

三者的核内时间合计只占 0.885 ms，**其余 ~1.77 ms 全是发射地板的消失**——12.1 的论点成立。

### 12.3 两个融合**实测更慢**，都回退了，这是本节最该看的

**A（split-K reduce 融进 GEMV）**：`gemvreduce2`，同一作业内三种模式互为对照：

| reduce 形态 | 发射/步 | 步时间 | 相对 |
| --- | ---: | ---: | ---: |
| 原两段式 | 1231 | **20.411 ms** | — |
| last-block 融合（1 段） | **975** | 21.299 ms | **+0.888 ms 更慢** |
| split-in-block（1 段、无 partial 缓冲） | — | 33.324 ms | +12.9 ms 更慢 |

发射确实少了 128 个，步反而更慢。**省下的时间没消失，只是搬了家：**

| kernel | 原 | 融合 | 差 |
| --- | ---: | ---: | ---: |
| `gemv_kn_i8x4_split` | 6.445 ms | 8.095 ms | **+1.650** |
| `gemv_kn_i8_split` | 4.750 ms | 5.380 ms | **+0.630** |
| `gemv_reduce_i8` | 2.076 ms | — | −2.076 |

两个结构性代价，都不可调：①每个 split block 上一次 `__threadfence()`（4096 次）是
device 域 flush，比它替掉的普通 store 贵；②**丢掉了发射流水线**——独立 reduce 的 block
可以在 partial grid 收尾时就开始跑，融合版必须等整个 tile 到齐。
split-in-block 更差的原因正是当初预判的：`cols = threads/split`，split 16–32 时
`cols` 只有 8–16，64 lane 的 wavefront 横跨 4–8 个 k 行，事务效率崩掉。

**C（delta 三合一）**：`probe_delta_ab.py`，同进程交替 A/B，5 轮，抖动 <0.5%：

```
chain (3 发射): 86.26 us/call
fused (1 发射): 91.57 us/call      fused - chain = +5.32 us/call
输出比特相等: True     state 比特相等: True
```

真实步里更糟：`delta_normed` 2.781 ms / 24 层 = **115.9 us/层**，而三段式是
**15.1 + 73.7 + 14.0 = 102.8 us/层**（`prologue` 0.362 ms、`delta` 1.768 ms、
`rms_norm_gated` 0.336 ms，均 24 次调用，同节点同卡 meas1），
**+13.1 us/层 × 24 = +0.315 ms/步**；实测 20.285 → 20.615 ms（+0.330），方向与量级一致。

**三个口径互相不一致，而不一致本身有信息**，记在这里因为事后从树里推不出来：

| 口径 | 每层 | 折算每步 |
| --- | ---: | ---: |
| 同进程 A/B（最可信的单点测量） | +5.3 us | +0.128 ms |
| 真实步逐层拆分 | +13.1 us | +0.315 ms |
| 真实步直接实测 | — | **+0.330 ms** |

A/B 少预测了约 **2.5 倍**，说明融合的代价**有一部分在核体之外**——占用率或调度，
不只是多出来的算术。这和 12.1 那条（profiler 逐 kernel 时长不可加）是同一根因的两面：
**孤立的单算子计时只说明核体，说明不了它在真步里挤掉了谁。**
所以 102.8 → 115.9 不能当预算表读，它只解释了一部分回归。

原因是几何冲突：`delta_kernel` 之所以快，是因为**一个线程独占一个 value 列、把 `dk` 走完
寄存器循环、全程没有跨线程通信**；而 prologue 和门控 norm 各需要一个 block 归约。
合成一个核必须为三者取一个几何，那个折中比省下的两次发射更贵。

**两个负面结论的共同教训**：12.1 的 5.75 us 地板是**"发射间隙"的地板，不是"工作"的地板**。
**去掉一次发射只有在工作也跟着消失时才划算**——把工作从 A 核搬到 B 核，发射省了、时间没省。

### 12.4 三个会静默改掉 token 的舍入陷阱

都不能用容差检查发现，写在这里因为下一个做融合的人一定会再撞：

1. **`silu_mul`**：`F.silu` **返回 bf16**，所以链上是"激活先窄化、再乘"，**两次舍入**。
   只做一次 fp32 舍入在 18432 个元素里差 ~5000 个。
2. **`rope_apply`**：`apply_rope` 先 `cos.to(q.dtype)`（fp32 表直接算差 296/1024），
   而且**逐运算**舍入（mul、mul、add 各一次）。合成一个 fp32 表达式只舍入一次，差 435/个。
3. **`kv_quant`**：必须是 fp32 **除法**（不是乘倒数）、**ties-to-even**。
   乘倒数在末位不同，会让写 cache 的核和读 cache 的核**对同一个 scale 的理解不一致**。

三个陷阱现在都写进了核里的注释，并且由探针把"错误写法"的差异数打印出来，
所以后来的人不会把它们当冗余删掉。

### 12.5 一个基础设施级的坑：适配器写完不等于算子会跑

`_NATIVE` 是手工维护的 `名字 -> 适配器` 字典，**字典里没有的项会被 `reference.fill`
用纯 torch 实现补上**。于是"适配器 + bindings + 头声明 + protocol 字段 + reference 实现"
可以全部落地、全套测试全绿，**而算子在卡上从未真正跑过**——用它的测量结果会是
**一个干净的零**，和"这个融合没用"完全无法区分。本轮真的踩了：
`kv_quant` / `silu_mul` / `rope_apply` / `add_rms_norm` / `delta_normed` 全都写完但没登记。

更隐蔽的第二层：**`m.def` 忘了写**。`delta_normed` 有声明、有定义、有 `_NATIVE` 条目，
卡上仍然 `AttributeError`——pybind 注册是另一件事，而且只有 import 扩展时才暴露。
`bindings.cpp` 那个 \_NATIVE 都查不出来。

现在有两道本地门禁（`tests/test_backends.py::NativeRegistrationTest`），
且**验证过它们真的会红**：把 `m.def("delta_normed", ...)` 删掉后，
`test_every_native_op_is_bound_in_the_extension` 立刻报 `['delta_normed']`。

### 12.6 现在的账

| 指标 | 前 | 后 | |
| --- | ---: | ---: | --- |
| decode 步 | 23.527 ms | **20.263 ms** | **1.16x** |
| 吞吐 | 42.51 tok/s | **49.35 tok/s** | |
| profiler 行/步 | 2407 | **1399** | |
| 原生算子 | 18/18 | **23/23** | |
| greedy ids | — | 未变 | |

复现：

```bash
# 发射地板（无模型，最便宜的一条）
python platform/scow/tools/sync.py push -d coffin \
    --launcher diag/corex_probe/launch_floor.sh --run floor -g 4 -t 20 --wait

# 整步：发射清单 + 逐核时间 + greedy ids
python platform/scow/tools/sync.py push -d coffin \
    --launcher diag/corex_probe/launch_measure.sh --run meas -g 4 -t 25 --wait

# 融合 C 的同进程 A/B（回退与否的证据）
python platform/scow/tools/sync.py push -d coffin \
    --launcher diag/corex_probe/launch_delta_ab.sh --run dab -g 4 -t 20 --wait
```

**还剩下的**：`gemv_kn_i8x4_split` + `gemv_kn_i8_split` 合计 11.17 ms（55% 的步），
是纯权重读，只有 int8→int4 或更大 M 的路子能动；`delta_kernel` 1.85 ms；
`gemv_reduce_i8` 2.07 ms（A 已证明融不掉，但它是 8% 的步，值得从"减少 split"方向再看一次）。

## 13. prefill：第一次实测（2026-10-28，gpu-node-05 card 3）

此前**没有任何可用的 prefill 数字**：`PERF_COREX.md` 全部是 decode，而唯一条目
是 `COREX_PORT.md` §6b 的 `prefill 1056.5 ms (5 tokens)`。那个数字不能用，原因不是
它走了错的分支（`Qwen35.prefill` 只在 `n < 4` 才回退到解码路径，5 token 走的是批量路径），
而是**它是进程里的第一次调用**：首次调用要付 kernel 懒加载、内存池增长、cuBLAS 句柄
建一次的全部代价。同一个 prompt 在 warm 之后是毫秒级。**冷启动的第一次 prefill
不是吞吐量**，这也是本节所有数字都先 warmup 一遍再计时的原因。

### 13.0 结论先说

全部是 **paged KV**（服务路径），n=2048，单卡，eager：

| 配置 | ms | tok/s |
| --- | ---: | ---: |
| **int8 权重，当前服务默认（`mtp_prefill=True`）** | **1142.7** | **1792** |
| int8 权重，`mtp_prefill=False` | 920.9 | 2224 |
| bf16 权重，`mtp_prefill=False` | 747.9 | **2738** |
| bf16 权重，`mtp_prefill=True` | 962.5 | 2128 |

**开箱即用的服务默认是 int8 + `mtp_prefill=True` = 1792 tok/s**；把 drafter 的废功
关掉（§13.2a，一行 Python）就是 2224，换 bf16 权重是 **2739**。

同一份 bf16 模型上，drafter prefill 开关的差值就是它自己的成本：**962.5 − 747.7
= 214.8 ms**，与 int8 上的 1142.7 − 920.9 = **221.8 ms** 相差 3%。

对照 gfx906（`ENGINE.md` §6，2048 token）：**1729 tok/s**。

**所以 CoreX 的 prefill 不比 DCU 慢**，bf16 下还快 1.58x。原因可以直接读出来：
这张卡的 bf16 tensor-core GEMM 实测 **45–51 TFLOP/s**（§13.3），而 DCU 的 rocBLAS
在 fp16 上只有 13–20 TFLOP/s。decode 上 CoreX 落后 3.7x 是另一件事（访存受限），
prefill 是算力受限，两者的排序本来就不同。

### 13.1 打满程度：45%

2048 token 的前向 FLOPs ≈ `2 x 2048 x 4.0e9` ≈ **16.4 TFLOP**（4B 参数，不含 attention）。
bf16 最优配置 747.9 ms → **21.9 TFLOP/s 有效算力**，对单 GEMM 峰值 45–51 TFLOP/s
是 **约 45%**。也就是说"全做对"应该有 2x 左右的空间，但**空间不在投影上**：

profiler（n=2048, chunk=2048, int8），**按叶子 kernel 聚合**（不是按 `aten::*` —— 后者
有嵌套，`aten::to` → `aten::_to_copy` → `copy_` 三层会重复计数，加起来超过 100%，
第一版表格就是这么错的）：

| 叶子 kernel | ms | 占 928.5 ms |
| --- | ---: | ---: |
| TCU GEMM 三个变体（`gemm_bf16_bf16_tcu_25` 256.8 / `tcuh` 62.7 / `gemm_stride_tcuh2` 46.7） | 366.2 | **39%** |
| elementwise 三个变体（`elementwise_kernel<512,1>` × 2 + `<128,4>`） | ≥198.2 | **≥21%** |
| 其余（delta 扫描的 `bmm`/`baddbmm` kernel、int8 的 dequant `copy`、`solve`、`conv1d_silu_seq`） | ≤364.1 | ≤39% |

`aten::` 层面的对照值（同一份 profile 的 `CUDA total`，彼此不嵌套）：`aten::mm` 319.5、
`aten::copy_` 207.5、`aten::mul` 86.4、`aten::bmm` 85.9、`aten::baddbmm` 46.9。

**两个结论**：

1. **真正的 GEMM 只占 39%**（这份 profile 的 drafter prefill 已关，所以全是主模型），所以优化顺序是先砍废功和胶水，
   而不是去调 GEMM —— 它已经在跑 45–51 TFLOP/s 了。
2. **elementwise ≥21%**：`masked_fill` + `amax` + 两次 rescale 的 online softmax，
   每层每 tile 各来一遍，加上 dequant 的 `mul`。这是纯带宽/发射浪费，不是必要的算术。

### 13.2 三个实测出来的、可以直接拿的收益

#### (a) MTP prefill 是**纯废功**，而且它的慢是 M=2047 造成的：7.46x

**两层，两个独立的问题。**

**第一层：它本来就不该跑。** `Qwen35.mtp_prefill` 默认 `True`，只有
`enable_batching(capacity>1)` 会把它设成 `False`。但 `caps.graph == 0` 时
`LLMEngine.build()` **不允许开投机解码**（spec 每一步都是重放按桶捕获的图，
没有 eager 替身），`Server` 会把 `spec_k` 清零 —— **drafter 的 prefill KV 写完之后
没有任何人读**。实测这一层值 1.29x：

| 配置 | `mtp_prefill=True` | `mtp_prefill=False` | 收益 |
| --- | ---: | ---: | ---: |
| bf16, paged | 962.5 ms / 2128 tok/s | **747.7 / 2739** | **1.29x** |
| int8, paged | 1142.7 / 1792 | **920.9 / 2224** | 1.24x |

**第二层：它慢得离谱，因为 GEMM 的 M 是 2047。** `MTP.prefill` 被
`Qwen35.prefill` 用 `m1 - c0` 行调用，而 `m1 = min(c1, n-1)` —— 最后一个 token
没有"下一个 token"，所以 drafter 的 GEMM **永远比主模型少一行**。这一行把它踢出
tensor-core 路径。逐 kernel 的 A/B（bf16, n=2048, 同一进程）：

```
mtp_prefill=True  : GemmEx_kernel::gemmh_new<...>   194.1 ms   <- 非 TCU 回退
mtp_prefill=False : （该 kernel 完全不存在）

drafter 独占的 GEMM 形状（每个 1 次调用）：
  [2047,2560] x [2560,18432]   75.7 ms   (M=2048 时 32 次共 137.2 ms，即 4.29 ms/次)
  [2047,2560] x [2560,10240]   42.3 ms
  [2047,9216] x [9216,2560]    38.1 ms
  [2047,5120] x [5120,2560]    21.1 ms
  [2047,4096] x [4096,2560]    16.9 ms
```

**同一个 `[2048,2560]x[2560,18432]` 是 4.29 ms，`[2047,...]` 是 75.7 ms —— 17.6x。**

##### M 的对齐悬崖（扫了 M=1024..4096）

三个投影形状上**完全一致**：

| M | `mlp.gate_up` ms / TFLOP/s | `mlp.down` | `mix.out_proj` |
| ---: | ---: | ---: | ---: |
| 2040 / 2044 / 2046 | 3.71 / 3.72 / 3.72 ms, **51.9–52.0 TFLOP/s** | 2.67, 36.0–36.1 | 0.80, ~54 |
| **2047** | **75.50 ms, 2.6** | **37.90, 2.5** | **16.67, 2.6** |
| 2048 | 4.28, 45.1 | 2.16, 44.8 | 0.89, 48.2 |
| **2049** | **83.74, 2.3** | **43.89, 2.2** | **18.47, 2.3** |
| 2050 / 2052 / 2056 / 2064 | 3.97–3.98, ~49 | 2.74–2.76, ~35 | 0.93, ~46 |
| 1535 | 56.74, 2.6 | 29.15, 2.5 | 12.78, 2.5 |
| 1536 | 3.21, 45.2 | 1.73, 41.9 | 0.68, 47.3 |
| 3071 / 3072 | 113.47 → 6.38 | 57.59 → 3.45 | 24.86 → 1.33 |
| 4095 / 4096 | 151.11 → 8.52 | 76.19 → 4.31 | 33.05 → 1.72 |

**判据是 M 的奇偶性，不是 2 的幂、也不是 256 的倍数。** 17 个扫描点全部一致：

| | M | 结果 |
| --- | --- | --- |
| **奇数** | 1535, 2047, 2049, 3071, 4095 | **全部 2.2–2.6 TFLOP/s（慢 17x）** |
| 偶数 | 1024, 1536, 2040, 2044, 2046, 2048, 2050, 2052, 2056, 2064, 3072, 4096 | 全部 35–54 TFLOP/s |

5 个奇数点全慢、12 个偶数点全快，三个投影形状无一例外。所以

> **奇数 M 落进非 tensor-core 的 `GemmEx_kernel::gemmh_new` 回退，偶数 M 走 TCU。**

2049 这个点很关键：它是 8 的倍数加一，却和 2047 一样慢 —— 如果判据是 8 或 256 的
倍数，2049 和 2050 应该有同样的行为，但 2050 是快的。**只有奇偶性能同时解释
2047 和 2049。** 至于为什么是奇数触发（很可能是一行 wavefront / 64 lane 的对齐），
这版没继续挖，因为不需要 —— 修法只依赖"补齐到偶数"这个可观测事实。

于是修法很清楚：**把 drafter 的 GEMM 行数补齐到偶数**（补一行到 2048）。
实测收益：

```
MTP.prefill 直接调用，n=2047 vs n=2048：
  rows=2047   215.1 ms
  rows=2048    28.8 ms   -> 7.46x
```

**7.46x 的差距只来自补了一行。** 补的那一行是真实计算（不是 mask 掉的），
所以数值不受影响；而且 `MTP.prefill` 已经按 `n = min(...)` 截断，多出来的行
不会进入 KV 的有效窗口。

**这一条两个层次加起来**：`mtp_prefill=False`（1.29x，纯 Python）+ 对齐修好
（drafter 真正需要时 7.46x；在 `graph=False` 下它本来就该关，所以这是给
"将来这个后端开图"留的）。

#### (b) int8 权重在 prefill 上反而**更慢**，代价是每次请求固定 ~173 ms

`linear_prefill` 对 int8 权重每次都调 `pack.KnQWeight.dequant_kn()`，
把整个权重物化成 bf16：4.4 GB 读、8.8 GB 写。

这个开销**与 M 无关**（权重摊完就是常数），实测非常干净：

| n | int8 ms | bf16 ms | 差 |
| ---: | ---: | ---: | ---: |
| 128 | 255.0 | 81.5 | **+173.5** |
| 512 | 376.4 | 201.1 | +175.3 |
| 2048 | 920.9 | 747.9 | +173.0 |

固定 ~173 ms/请求。所以短 prompt 上它更致命：n=128 是 **502 vs 1572 tok/s（3.1x）**，
n=2048 只有 1.23x。

**把 dequant 缓存住能拿回多少，直接量了**（把 128 个投影的 `w` 换成已 dequant 的
`[K,N]` bf16，其他一字不动）：

```
int8, 每次调用 dequant     909.9 ms   2251 tok/s
int8, 只 dequant 一次      738.5 ms   2773 tok/s   (1.23x)
额外常驻 bf16              6.65 GiB
```

**1.23x 的代价是 6.65 GiB**，而 int8 权重本来省的就是这块（约 4.4 GB）。
所以这条不是"免费的优化"：它把 decode 省的显存还给 prefill。真正的选择是

- 只服务 prefill 为主的工作负载 → 直接上 bf16 权重，别用 int8（747.9 ms，最快且最省事）；
- 两者都要 → dequant 结果要**按需缓存 + 可驱逐**（不是全量常驻），或者在
  int8 GEMV 里做融合 dequant 的 prefill 核（后者是写 kernel，见 §13.4）。

#### (c) `chunk` 在 2048 上封顶，但 2048 之后没法从这个 sweep 判断

| chunk | bf16 ms | int8 ms |
| ---: | ---: | ---: |
| 512 | 851.1 | 1560.7 |
| 1024 | 775.9 | 1125.5 |
| **2048** | **747.9** | **920.9** |
| 4096 | 747.9 | 921.0 |

4096 和 2048 一模一样，因为 prompt 本身只有 2048 token —— **一个 chunk**。
所以"2048 是不是最优点"这个问题在这组数据里**没有被回答**，只能说不小于。
要判断得上 8k/16k 的 prompt。DCU 上当年测到 2048 封顶是因为 rocBLAS 在 M>2048 不再涨，
这张卡的 GEMM 没有那个特征，值得单独测一次。

### 13.3 逐 shape 的 `linear_prefill` 带宽与算力

模型自己的权重，M=2048（int8 走 dequant 路径，bf16 直接 GEMM）：

| 投影 | N | K | bf16 ms | bf16 TFLOP/s | int8 ms | int8 TFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp.gate_up` | 18432 | 2560 | 4.27 | 45.3 | 6.56 | 29.4 |
| `mlp.down` | 2560 | 9216 | 2.16 | 44.7 | 3.35 | 28.8 |
| `mix.in_proj`（linear 层融合） | 12352 | 2560 | 2.53 | 51.2 | 4.14 | 31.3 |
| `mix.out_proj` | 2560 | 4096 | 0.88 | 48.7 | 1.45 | 29.5 |
| `mix.in_proj`（full 层融合） | 10240 | 2560 | 2.40 | 44.7 | 3.69 | 29.1 |

bf16 的 45–51 TFLOP/s 就是这张卡的 prefill 天花板，int8 路径的 29–31 是
**dequant 拖的**（同样的数学，多了一遍 4.4 GB 读 + 8.8 GB 写）。

### 13.4 stage 分解：投影 15%，delta 扫描 + MLP 是主项

`Layer.prefill` 的四个 hook 点（warmup 的 event 清掉之后再计时，否则桶加起来会是
墙钟的 197% —— 第一版就是这么错的）：

| 段 | ms | 占 909.1 ms |
| --- | ---: | ---: |
| `mlp` | 335.0 | 36.9% |
| `mix_state_linear`（24 层 delta 扫描 + conv + gated norm） | 294.9 | 32.4% |
| `mix_state_full`（8 层因果分块 attention） | 134.9 | 14.8% |
| `mix_proj_linear` | 99.8 | 11.0% |
| `mix_proj_full` | 29.6 | 3.3% |
| 残差（两个 RMSNorm、add、lm_head） | 14.9 | 1.6% |
| **accounted** | **894.2** | **98.4%** |

两项加起来 63%，投影 15%。**优化重心和 decode 完全相反**。

### 13.5 delta 扫描的 chunk：64 已经是最优，且上限是 LDS 不是算法

`chunked_delta` 的 `chunk` 默认 64，而 `Layer.prefill` **不把 model 的 `chunk`
传给 `LinearAttn.prefill`**，所以它永远是 64。2048 token 于是每层 32 个顺序段，
24 层 = **768 次 `baddbmm`**，profiler 里 59.7 ms。

想放大 chunk 减少段数，实测：

| delta chunk | ms | vs 64 |
| ---: | ---: | ---: |
| 64 | 908.2 | 1.00x |
| 128 | 997.4 | **0.91x** |
| 256 | 启动失败 | `L=256 needs 65536 bytes of shared memory, over the 49152 byte limit` |

两点都值得记：

1. **128 反而慢 9%。** 段数减半，但 chunk 内的 `[H, L, L]` decay 矩阵和三角系统
   是 O(L²) 增长（`pd` / `ut` / `ica` 各 `[32, nc, L, L]`），在 L=128 时已经吃掉
   掉段数省下的 launch 时间。**所以 64 不是"没调"，是调对了。**
2. **256 撞的是 `triangular_solve` 的 48 KB LDS 上限**，不是算法限制。要往上走
   必须重写那个核（分块把 `L x L` 流式读入，而不是整个驻留）—— 这是写 kernel 的活。

### 13.5b 已落地：两条改动，真机验收 PASS（2026-10-28，作业 3925）

两条都改完并上了卡，`diag/corex_probe/probe_prefill_fixes.py` 的三项检查全过：

```
dequant vs old spelling: 128 projections, max |diff| = 0.00e+00  BIT-IDENTICAL
greedy ids: [6, 24218, 10, 4838, 1665, 15]  baseline [same]  UNCHANGED
mtp_prefill=True  (old default)     1052.0 ms    1947 tok/s
mtp_prefill=False (shipped)          834.6 ms    2454 tok/s   1.26x
```

**1. `pack.KnQWeight.dequant_kn` 改成 cast-then-scale**（`backends/corex/pack.py`）

原写法 `(wt.float() * scale).to(bf16)` 会物化一个 `[K, N]` fp32 临时量，是它立刻要
窄化成的结果的两倍大；128 个投影上读 3.3 GiB、写 13.3 GiB 才产出 6.7 GiB。

改成先 cast 再原地乘：

```python
out = self.wt.to(dtype)
out.mul_(self.scale.view(1, -1))
```

**关键点是 `scale` 不窄化。** `mul_` 的 dtype 取自目标张量，所以 `bf16.mul_(fp32)`
以 fp32 计算、只舍入一次 —— 和旧写法的单次舍入完全相同。实测：

| 写法 | ms | 精确 | 备注 |
| --- | ---: | --- | --- |
| 旧：`(wt.float()*scale).to(bf16)` | 174.8 | ✅ | 基准 |
| **新：cast + fp32 `mul_`** | **102.6** | **✅ 逐位相同** | **已落地，1.70x** |
| 变体：`scale.to(bf16)` 后乘 | 93.0 | ❌ 每元素偏 1 ULP | **不采用** |
| 行分块（R=8..256） | 300–2782 | ✅ | 全部更慢，废弃 |

那个 1 ULP 的变体虽然再快 10%，但权重会变；**权重喂 32 层，不值得为单个算子 10% 的
差异去动数值**。测试 `tests/test_backends.py::CorexPackTest` 用 `torch.equal` 把它钉死。

**2. `graph=False` 时关掉 MTP prefill**（`engine/llm_engine.py` 的 `build()`）

关在图关闭的分支里，不是 `Qwen35.__init__`：

```python
if self.model.mtp is not None:
    self.model.mtp_prefill = False
```

**为什么不在 model 里**：`spec_generate` 是一条可以直连的公共 API，它走
`capture_verify` 的 eager 替身，`_verify_graph` 非空，所以**确实会读** drafter KV。
放在 `build()` 这个"已知 spec 被拒"的地方才安全 —— 我第一版写在 `__init__` 里，
那会让直连 `spec_generate` 静默错。

验收 1052.0 → 834.6 ms（**1.26x**），token ids 未变。

**两条合起来对服务默认（int8 + old 默认）的效果**：1142.7 → **834.6 ms**，
1792 → **2454 tok/s（1.37x）**，纯 Python 改动，无需写 kernel。

### 13.5c 扫参：`PREFILL_ATTN_KB` 1024 → 4096（长 prompt 再 4.9%），tile 与 chunk 不值得动

`diag/corex_probe/probe_prefill_sweep.py`，12 个 (tile, kb) 组合 × 3 个 prompt 长度，
每个组合一个独立子进程（两个常量都是 `tuning.py` 在 import 期读环境变量，同进程改不了）。
**整套扫了两遍**（作业 `psweep2` / `psweep3`），36 个格子最大差异 **1.41%**。

#### kb（key block）：单调、可复现、有效

n=8192，每档相对 kb=512：

| tile | kb=512 | kb=1024 | kb=2048 | kb=4096 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 4213.3 | +10.4% | +12.2% | — |
| **512** | 4037.4 | +5.1% | +10.5% | **+12.7%** |
| 1024 | 4106.5 | +7.5% | +10.5% | +12.1% |
| 2048 | 4213.2 | +7.6% | +11.5% | — |

两次独立运行的 kb 效应（2048 vs 512，n=8192）：tile 256 是 12.2/11.3%、512 是 8.9/8.9%、
1024 是 10.5/10.5%、2048 是 11.5/10.9% —— **每个 tile 宽度上方向一致、幅度一致**。
2048→4096 在 tile=512/1024 上分别是 3677.4→3591.0（+2.4%）和 3675.6→3609.7（+1.8%），
单次任务里量到，没有第二遍，所以只当"没到顶"的证据，不当精确值。

**所以默认值从 1024 改成 4096**（`models/qwen35/tuning.py`）。为什么敢改：

- 方向在 4 个 tile 宽度上全一致，且两次扫参复现；
- 显存是可控的：tile=512、kb=4096 时一层活着的 score block 是 **128 MB**
  （`[nkv=4, rep=4, 512, 4096]` fp32），对 32 GB 卡 + 4.5 GB int8 权重不算事；
  accumulator 才随上下文增长，block 不随；
- 引擎默认 tile 就是 512，所以实际生效的是 128 MB 那一格。

**n=2048 没有收益**（720.0 → 720.2 ms）：一个 prompt 只有一个 tile，block 循环几乎不跑。
**收益全在长 prompt 上**，这也解释了为什么 DCU 当年调出 1024 —— 它测的场景/tile 不同。

#### tile：1.5–2.1%，在噪声以上但不足以作为依据

kb=2048、n=8192 时四个 tile 的排名两次都不稳定：

| 运行 | 256 | 512 | 1024 | 2048 | 最优 | 极差 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 第一遍 | 3697.2 | 3677.4 | **3675.8** | 3729.2 | 1024 | 1.5% |
| 第二遍 | 3737.0 | 3677.3 | **3675.7** | 3754.2 | 1024 | 2.1% |

1024 两次都最优，但**它和 512 只差 0.05%**（3675.8 vs 3677.4）—— 那不是效应，是巧合。
tile=2048 明显最慢（+1.5%~+2.1%，两次一致，可解释：少了一个 tile 就多吃上三角的活）。
**结论：512 保持不动。** 换成 1024 的唯一理由是"两次都正好最低"，而这个差值比扫参
自身的 1.41% 复现误差还小。

#### chunk：超过 prompt 长度之外没有收益

| n | chunk | chunks | ms | tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 2048 | 1 | 745.1 | 2749 |
| 4096 | 2048 | 2 | 1694.1 | 2418 |
| 4096 | **4096** | 1 | 1674.8 | **2446** |
| 8192 | 2048 | 4 | 4213.2 | 1944 |
| 8192 | **4096** | 2 | 4169.9 | **1965** |

（上表是 `psweep2` 那一遍；`psweep3` 的重跑给 4213.2 / 4169.9，逐格一致。）

chunk 4096 比 2048 只快 **1.0–1.3%**，而且这是**两个因素合在一起**的结果（dequant 次数
减半 + GEMM 的 M 变大），单看每个因素都不到 1%。**默认 chunk=2048 保持不动**：
1% 换两倍的 attention 暂时显存（`[nkv, rep, C, KB]`，C 翻倍）不划算，而且大 chunk 更容易
踩到 §13.6b 那个奇数 M 的悬崖。

#### 复现

```bash
# 12 个 (tile, kb) 组合，每个一个子进程，跑在 3 个 prompt 长度上
python platform/scow/tools/sync.py push -d coffin \
    --launcher diag/corex_probe/launch_prefill_sweep.sh --run psweep -g 4 -t 55 --wait

# 落地后的端到端验收（含 kb=4096 的 ids 复核）
python platform/scow/tools/sync.py push -d coffin \
    --launcher diag/corex_probe/launch_prefill_fixes.sh --run pfix -g 4 -t 30 --wait
```

#### 落地后的真机验收（作业 3931）

```
PREFILL_ATTN_TILE=512 PREFILL_ATTN_KB=4096
n=2048      825.7 ms   2480 tok/s
n=8192     4021.5 ms   2037 tok/s
greedy ids: [6, 24218, 10, 4838, 1665, 15]  baseline [same]  UNCHANGED
verdict: PASS
```

`kb` 变了会改 online-softmax 的累加顺序，所以 **ids 必须重新验**，不是假设不变 —— 复核通过。

### 13.6 该做什么，按"实测收益 ÷ 代价"排序

| # | 动作 | 实测/预期收益 | 代价 | 状态 |
| --- | --- | --- | --- | --- |
| 1 | `caps.graph == 0` 时把 `mtp_prefill` 设 `False` | **1.26x** 实测（1052.0→834.6 ms） | 一行 Python | **已落地，§13.5b** |
| 2 | `MTP.prefill` 的 GEMM 行数补齐到偶数 | **7.46x**（215.1→28.8 ms） | 补一行 + 一处上取整 | **可直接做** |
| 3 | GEMM 的 M 统一按偶数上取整（所有 prefill 分块） | 未知（M=2047/2049/3071 各慢 17x） | 同 #2 的通用化 | 需先确认主路径会不会出奇数 M |
| 4 | int8 权重的 prefill dequant 缓存 | 1.23x（909.9→738.5 ms） | **不需要了** | **不采用，见 §13.5b** |
| 5 | `PREFILL_ATTN_TILE` / `PREFILL_ATTN_KB` 在这张卡上重扫 | kb 1024→4096：**长 prompt +4.9%**；tile 不动 | 已扫，见 §13.5c | **已落地** |
| 6 | chunk > 2048 | 只值 1.0–1.3%（§13.5c） | 已扫 | **不改，不值得** |
| 7 | delta 扫描的 elementwise / `[H,nc,L,L]` 融合核 | 该段 294.9 ms 的一部分 | 写 kernel，最大工程量 | 最后做 |

**#4 为什么不做了**：它要 6.65 GiB 常驻，而 int8 权重省的正是 4.4 GB；而 §13.5b 的
一行改动已经把 dequant 从 174.8 降到 102.6 ms（1.70x），**剩下的 102.6 ms 再拿
6.65 GiB 去换不值得**。真想要 hoisted 的 738.5 ms，直接上 bf16 权重（747.9 ms）即可，
差 1% 但逻辑全免。

**MTP 的两层收益要分开看**：#1 是"别算它"，#2 是"算就得算对"。在当前 `graph=False`
的配置下 #1 已经把整件事解决掉了；#2 是给将来这个后端开图（`_GRAPH_VERIFIED`）之后
准备的 —— 那时 drafter 会真的被用到，而它的 M 仍然是 n−1。

### 13.6b 一个必须记住的通用教训：GEMM 的 M 决定走不走 tensor-core

这次最值钱的发现不是某个百分比，而是一条**形状契约**：

> 在这张卡上，`[M,K] @ [K,N]` 的 M 是奇数 → 走 `GemmEx_kernel::gemmh_new`
> → **慢 17x**。M 是偶数 → 走 `Gemm_bf16_bf16_tcu` → 35–54 TFLOP/s。

所以任何"每层/每请求算一个行数"的地方都可能踩到它，而且**症状不是报错，是慢**：
profiler 里只是多了几行 `GemmEx`，总时间悄悄翻倍。已确认会踩的：`MTP.prefill`
（M = n−1）。**没确认、但机制上同样危险**的：chunked prefill 的最后一个 chunk
（`n % chunk` 是奇数时）、`prefill_group` 的 `C_total`、verify 的 `B*c`。

自查办法：profiler 里出现 `GemmEx_kernel::gemmh_new` 就是踩了，TCU 路径的
kernel 名里带 `Gemm_tcu_bi_kernel`。这条比任何一次调度参数都更值得先查。

### 13.7 复现

```bash
# headline + chunk sweep + MTP A/B + profiler（int8 与 bf16 各一遍）
python platform/scow/tools/sync.py push -d coffin \
    --launcher diag/corex_probe/launch_bench_prefill.sh --run prefill -g 4 -t 40 --wait

# stage 分解 / delta chunk / 逐 shape / dequant 缓存实验
python platform/scow/tools/sync.py push -d coffin \
    --launcher diag/corex_probe/launch_prefill_attr.sh --run pattr -g 4 -t 40 --wait

# paged 路径的 headline + 两组 profiler
python platform/scow/tools/sync.py push -d coffin \
    --launcher diag/corex_probe/launch_prefill_detail.sh --run pdetail -g 4 -t 45 --wait

# MTP drafter 的逐 kernel 代价（A/B 两份 record_shapes profile）
python platform/scow/tools/sync.py push -d coffin \
    --launcher diag/corex_probe/launch_prefill_mtp.sh --run pmtp -g 4 -t 40 --wait

# GEMM 的 M 对齐悬崖（17 个 M 点 × 3 个形状 + drafter 补行实测）
python platform/scow/tools/sync.py push -d coffin \
    --launcher diag/corex_probe/launch_gemm_align.sh --run galign -g 4 -t 40 --wait
```

三个 launcher 都先 `pick_card` 验卡（`gpu-node-05` 的 card 0/1/2 是死卡）。

### 13.8 测量口径（免得下一个人重复踩）

1. **必须 paged。** 服务路径是 `PrefixCacheEngine` + `PagedKVCache`。逐项同条件对照，
   paged 稳定慢 **1.1–1.4%**（就是 `PagedKVCache.gather` 物化 dense view 的成本）：

   | 条件 | dense cache | paged | 差 |
   | --- | ---: | ---: | ---: |
   | bf16, mtp off | 737.7 | 747.9 | +1.4% |
   | int8, mtp off | 908.9 | 920.9 | +1.3% |
   | int8, mtp on | 1130.7 | 1142.7 | +1.1% |

   差值小，但**混着比会得到完全错误的结论**：本节的表格第一版就把 dense 的
   `mtp on`（952.5）和 paged 的 `mtp off`（747.9）放进了一张表，读起来像"paged 快 27%"，
   实际那 27% 全是 `mtp_prefill` 开关的差别。横竖两种口径都要标清楚。
2. **`create_model` 默认加载 bf16。** 测 int8 必须 `load_weights(quant=True)` 再
   `create_model(weights=sd)`，否则你测的是 bf16 路径（这条在 PERF_COREX §11 已经栽过一次）。
3. **stage hook 必须在 warmup 之后清桶**，否则各段之和会是墙钟的 197%。
4. **`chunk=4096` 在 n=2048 上等价于 2048**（一个 chunk），不是"没收益"。
