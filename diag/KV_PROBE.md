# Kernel-level probes — int4 GEMV / attention / CPU attention

Synthetic tensors only, no model weights. 2026-10-22, pod `cgrp-v87dg651`
(1× DCU Z200SM_80, 16 GB, cgroup 7 cores / 60 GiB, DTK 26.04).  GPU idle except where noted.

Scripts: [`probe_gemv_int4.py`](probe_gemv_int4.py) + [`gemv_int4_ext.cu`](gemv_int4_ext.cu),
[`probe_attn_scale.py`](probe_attn_scale.py), [`cpu_attn.c`](cpu_attn.c).
Raw logs on the pod: `/tmp/probe4b.log` (int4 GEMV, fixed), `/tmp/attn5.log`, `/tmp/cpu_attn`.

---

## 1. int4 weight GEMV (the decode floor)

`y[N] = W4[N,K] @ x` , W4 = packed nibbles, group-128 fp16 scale, bf16 activation, one warp per row.
Compared against the engine's int8 GEMV (`hip_ops.gemv_w8a16`) on the same shapes.

| N × K | int8 GB/s (best R,U) | **int4 GB/s (best U)** | int4 ms | ratio |
| --- | --- | --- | --- | --- |
| 17408 × 5120 (gate/up) | 561 | **363** (U=4) | 0.127 | 0.65 |
| 34816 × 5120 (gate+up) | 613 | **430** (U=2) | 0.214 | 0.70 |
| 61440 × 5120 (qkv+z 组合) | 639 | **451** (U=2) | 0.357 | 0.71 |
| 248320 × 5120 (lm_head) | 668 | **485** (U=2) | 1.345 | 0.73 |
| 5120 × 17408 (down_proj) | 613 | **401** (U=2) | 0.107 | 0.65 |

* 数学对拍：修正后的 kernel `max_abs_err` 0.003–0.014（bf16 输出量级），**带宽与正确性都可用**。
  （第一版 kernel 把 `-8` 校正做了两次，`max_abs_err` 32–114；这是 kernel bug，不是量化误差，
  修法：`wv4 = nibble` 且减 `XS`，或 `nibble-8` 且不减。带宽数字不受影响。）
* **结论：int4 GEMV 是 ALU 受限，不是带宽受限**——每 4 B 权重要 8 次移位/掩码/int→float + 8 次 FMA。
  有效带宽只有 int8 的 0.65–0.73×，但每个权重只占一半字节，净收益 ≈1.4×。
  按模型权重混合平均 ~**430 GB/s** ⇒ 12.9 GiB 权重读一遍 **≈32 ms/token**（= 30 tok/s 地板）。
* HBM 纯流式读 685 GB/s，说明离带宽屋顶还有 1.6×：把激活按 128 组量化成 int8、用 gfx906 的
  `v_dot4_i32_i8`（`__builtin_amdgcn_sdot4`）一次吃 4 个 MAC，目标 ~600 GB/s。
  注意 gfx906 **没有** packed bf16，只有 packed fp16。

## 2. 单层 fused attention 随上下文长度（现有 decode 核，int8 KV）

27B full-attention 几何：nh=24, nkv=4, hd=256, rep=6；int8 KV（K+V 各一份）+ fp32 per-(head,token) scale。
每行是一个 layer 的一次调用；`split` 是 key 维切分数（现有 split-K 模板）。

| S | 最优 split | ms/层 | 等效 KV GB/s（按单次读算） | 16 层 ms/token | 折算 tok/s（只算注意力） |
| --- | --- | --- | --- | --- | --- |
| 8k | 32 | 0.51 | 33.6 | 8.1 | 123 |
| 16k | 64 | 0.95 | 35.8 | 15.2 | 66 |
| 32k | 64 | 1.86 | 36.7 | 29.7 | 34 |
| 64k | 64 | 3.66 | 37.3 | 58.5 | 17 |
| 128k | 64 | 6.95 | 39.2 | 111 | 9.0 |
| **256k** | **64** | **13.50** | **40.4** | **216** | **4.6** |

* 固定开销（pos=0）只有 0.023–0.024 ms，split 从 8 提到 64 已经把并行度吃满。
* **致命点：KV 读放大 ×6**（一个 block 管一个 q head，rep=6 的 6 个 q head 各读一遍同一个 kv head）。
  256k 时真实 KV 流量 = 6 × 0.537 GB × 16 = **51.5 GB/token**，216 ms ⇒ 真实 ~238 GB/s。
* 两个必改项：(a) **GQA-aware**：一个 block 管 1 个 kv head 的 6 个 q head → 流量 /6 → ~36 ms；
  (b) **KV int4** + 向量化 int8 载入（现在逐字节读）→ 再 /2 → 目标 **8–15 ms @256k**。
  4B 几何（nh=16/rep=4）同条件为 9.5 ms/层 @256k（57.5 GB/s 等效），说明放大倍数差就是主因。
* 对 decode 的直接影响：不重写时 32k 上下文光注意力就 29.7 ms、64k 58.5 ms、256k 216 ms，
  对应天花板 34 / 17 / 4.6 tok/s（**还没算权重**）。所以 27B 的注意力核重写不是优化项，是前置条件。

## 3. CPU 侧 cold attention（7 核，原型）

`cpu_attn.c`：GQA decode attention，int8/int4 KV，72 个 (kv head × split) OpenMP 任务，
`omp_get_max_threads()` = 7，output online-softmax 合并；单层、S=262144。

| mode | ms/层 | GB/s | GFLOP/s | 16 层 ms/token | tok/s |
| --- | --- | --- | --- | --- | --- |
| read8（朴素 32B 步进求和） | 28.2 | 9.5 | — | — | — |
| int8 attention | **166.8** | 3.3 | 38.6 | 2669 | **0.37** |
| int4 attention | 487.1 | 0.6 | 13.2 | 7794 | 0.13 |

* 结论：**朴素 CPU attention 比 PCIe 流式（5.3 tok/s）慢一个数量级**，不能替代 GPU 冷路径。
  原因是 (a) 原型逐元素标量点积；(b) int4 解包在 CPU 上更贵。
* 上限估算：7 核 DRAM 读 35.5 GB/s（`HW_PROBE.md`），int8 点积若做到 ~500 GFLOP/s，
  256k 的 103 GFLOP + 2.26 GiB 冷读 ≈ 200–250 ms/token（~4–5 tok/s），**最多与 PCIe 打平**。
  KTransformers 的 CPU 优势来自 AMX + MoE 稀疏，这两个前提我们都没有。
  → 列为可选（§6-E），不进主路径。

## 4. host-mapped 零拷贝读 + pinned copy（已完成）

`gemv_int4_ext.cu::host_read_bw`（`hipHostMalloc` + `hipHostGetDevicePointer` + 流式读核 +
`hipMemcpyAsync` 对照），512 MB pinned 缓冲，5 次平均，GPU 空闲：

| 项 | GB/s |
| --- | --- |
| 设备端 kernel 直接读 host-mapped pinned 内存（zero-copy） | **12.41** |
| pinned H2D `hipMemcpyAsync` | **12.90** |
| pinned D2H | **14.10** |
| 对照：HBM 内纯流式读（uint4，1 GiB） | **685.4** |

与 `HW_PROBE.md` 的 13.1 / 14.1 GB/s 一致。结论：**“kernel 直接读 host KV”不比“先 H2D 到暂存区”快**
（12.4 vs 12.9，噪声内），所以冷 KV 走显式分块 H2D 即可，不需要 zero-copy 特殊路径；
按 13 GB/s 算，256k 的全量冷 KV：int8 8.6 GB → 660 ms/token，int4 4.3 GB → 330 ms/token。
48 µs / 4 KB 的拷贝延迟意味着**必须按 MB 级连续块搬**（4 KB 粒度只能到 ~84 MB/s）。

## 5. CPU 侧 cold attention 全窗口扫描（7 核）

同一 `cpu_attn.c`，`OMP_NUM_THREADS=7`，split=8（int8 路径是 AVX2 向量化点积，int4 是标量解包，
后者只是负面上界；读上限实测朴素 32 B 步进只有 9.2 GB/s，`HW_PROBE.md` 的 torch 7 线程读为 35.5 GB/s）：

| S | int8 ms/层 | int8 GB/s | int8 GFLOPS | 16 层 tok/s | int4 ms/层 | int4 GB/s |
| --- | --- | --- | --- | --- | --- | --- |
| 32k | 40.5 | 1.7 | 19.9 | 1.54 | 80.0 | 0.4 |
| 64k | 63.5 | 2.1 | 25.4 | 0.98 | 132.7 | 0.5 |
| 128k | 97.5 | 2.8 | 33.0 | 0.64 | 234.6 | 0.6 |
| 256k | 166.0 | 3.3 | 38.8 | 0.38 | 438.9 | 0.6 |

单层 166 ms 是**计算受限**（38.8 GFLOP/s），不是内存受限；即使把点积优化到 ~500 GFLOP/s，
16 层也要 200–250 ms/token（4–5 tok/s），和 PCIe 直搬打平，仍然进不了主路径。

## 6. 复现（容器内）

```sh
cd /root/private_data/engine
export MKL_THREADING_LAYER=GNU PYTHONPATH=$PWD TORCH_EXTENSIONS_DIR=/root/private_data/torch_ext
python3 diag/probe_gemv_int4.py      # int4/int8 GEMV + HBM 读上限 + 零拷贝（本机跑 ~60 s）
python3 diag/probe_attn_scale.py     # 注意力窗口伸缩（2 几何 × 6 窗口 × 4 split，~90 s）
gcc -O3 -mavx2 -mfma -fopenmp -o /tmp/cpu_attn diag/cpu_attn.c -lm
OMP_NUM_THREADS=7 /tmp/cpu_attn int8 262144 8
```

原始日志（pod）：`/tmp/probe4b.log`（修正后的 int4 GEMV）、`/tmp/attn_full.log`（注意力全窗口）、
`/tmp/dbg4b.log`（int4 对拍）。**条件**：GPU `DCU% = 0` 时启动/结束；host/PCIe 数字可能受并行
IO 压测影响，GPU 数字无争用。int4 GEMV 第一版 kernel 有 bug（`-8` 校正做了两次），
带宽数字不受影响，正确性以修正版为准（`max_abs_err` 0.003–0.014）。

## 7. 由这些常量得到的单卡/双卡判断

- **单卡，稠密 256k，冷 KV 全读**：每 token 4.3 GB(int4) / 13 GB/s ≈ 330 ms → **~3 tok/s**；
  int8 KV 减半。CPU 直算更差（166 ms/层 → 2.7 s/token）。硬盘只能做前缀持久化（NFS 0.56 GB/s，
  读 4.3 GB 要 8 s/token），不能进 decode 循环。
- **单卡，只留“高频” KV（sink + 最近窗口 + 重击块）在显存**：注意力回到显存内，
  decode 回到权重受限（~20–30 tok/s），但 256k 冷段等于不看，长程召回无保证。
- **2 卡 TP（32 GB）**：int4 权重 6.7 GB/卡 + int8 KV 4.3 GB/卡（256k 全量）都在 HBM；
  每卡每 token ~11 GB，按 500 GB/s 有效带宽 ~22 ms → 40+ tok/s 上限、现实 25–35 tok/s。
  allreduce 每 token 仅 ~1.3 MB（13 GB/s 下 ~0.1 ms），可忽略；主要工作量在分片 kernel/权重加载。
  2 卡时注意力核同理要先做 GQA-aware：现有核 8 层 × 2.15 GB/卡 × 6（q head 重复读）= 12.9 GB
  @238 GB/s ≈ 54 ms → 18 tok/s；GQA-aware 后 2.15 GB → ~9 ms，不再限制。
  **这是唯一“全量 256k + 可用速度”的方案。**

