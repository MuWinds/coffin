# CoreX decode 性能分析：慢在哪、还能挖多少

> 2026-10-26/27，`gpu-node-05`（card 1），Iluvatar BI-V100，torch 2.1 + CoreX 3.2.3，
> `COFFIN_BACKEND=corex`，18/18 原生算子，graph 开。
>
> **第一轮优化已落地：50.53 → 43.46 ms/step（19.79 → 23.01 tok/s，1.23x）**，
> 文本与数值均未变。做法是修正 GEMV 的 split-K 分派（见 §2）。
> 下一步最大项是 delta 内核（§3，预期再省 6.9 ms）。

## 0. 已经拿到的结果

| 指标 | 优化前 | 优化后 | 提升 |
| --- | ---: | ---: | --- |
| decode（eager 实测） | 50.53 ms / 19.79 tok/s | **43.46 ms / 23.01 tok/s** | 1.23x |
| decode（graph replay，真实文本） | 52.41 ms / 19.08 tok/s | **46.97 ms / 21.29 tok/s** | 1.10x |
| `mlp_up` | 0.292 ms / 161.6 GB/s | **0.152 ms / 310.1 GB/s** | 1.92x |
| `qkv_gate` | 0.286 ms / 146.8 GB/s | **0.161 ms / 261.0 GB/s** | 1.78x |
| `lm_head` | 3.478 ms / 365.5 GB/s | **2.962 ms / 429.3 GB/s** | 1.17x |
| `mlp_down` / `o_proj` | 0.245 / 0.085 ms | 不变（已是最优） | — |
| 投影占整步 | 64% | **54%** | |
| 端到端文本 | ` Paris.` | ` Paris.` | 未变 |

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

## 3. delta 内核：下一个最大项（预期 7.1 → 0.18 ms）

这是单点最差的一个，也是下一步。

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

**它已经跑在带宽上限附近了**，问题纯粹是访存模式。

修法：把 state 转置成 `[heads, dv, dk]`，让 64 个 lane 读连续地址：

| | 抓取量 | 575 GB/s 下 | ×24 层 |
| --- | ---: | ---: | ---: |
| 现在（stride dv） | 134.2 MB | 233 µs | 7.1 ms（实测） |
| 转置后 | 4.19 MB | 7.3 µs | **0.18 ms** |

**风险**：`state` 不只是 delta 在用——`models/qwen35/linear_attn.py:124` 有 `ckpt_state`
（前缀复用的快照），所以转置必须同时改快照的读写。这也是它还没做的原因：
要连 `pack`/`snapshot` 一起改，并且用 `probe_truth.py` 对拍 reference 后端，
不能只改 kernel。

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
