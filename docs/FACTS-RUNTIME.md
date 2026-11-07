# FACTS-RUNTIME —— 引擎跑出来的数

> 本文件只记**运行期实测事实**（性能、天花板、负结果、坑），每条带口径；
> 不含时间线、不含方案与计划。口径写错比数字错更坑，所以每条都写清"在哪张卡、
> 什么权重、单流还是批"。

## 1. 引擎性能（Qwen3.5-4B，单流，海光 DCU Z200SM_80 16 GB，int8 权重）

| 项 | 值 | 口径 |
| --- | --- | --- |
| decode | **78.5 tok/s（9.8x）** | 单请求贪心 decode 稳态；首版纯 PyTorch 是 7.98 tok/s，倍数按这两点算 |
| prefill | **1729 tok/s（75x）** | 批量 prefill（fp16 权重、causal 分块、chunked gated delta）；首版逐 token 复用是 23 tok/s |
| 上下文 | **256k** | 4k/16k/64k 的 decode 一步 15.2/21.6/45.7 ms（改前 25.6/63.5/215.8） |
| 长上下文显存 | **峰值 12.4 GiB 与长度无关** | 流式 `ATTN_KEY_BLOCK=1024` + online softmax；改前 16k 就 OOM |
| 前缀复用 | **12.2x** | 两轮对话命中 3456 token：4.15 s → 0.34 s；paged 只比 dense 慢 2.8% |
| 前缀复用（单次提示词） | **15.4x** | 541 token 提示词 2137.8 → 138.9 ms |

## 2. 硬件天花板（gfx906 级 DCU，同一次会话实测）

- **copy ≈700 GB/s** —— 后续所有带宽优化拿它当上限（HBM 读 630 / 写 678）。
- **hipBLAS GEMV（M=1）55–296 GB/s**，同尺寸不同 N 能差 2×，选核不可预测；
  自写 warp-per-row GEMV 做到 540–680 GB/s。
- **fp16 13–21 TFLOP/s vs bf16 2.3–7.7 TFLOP/s**（GEMM）—— 这卡有 packed fp16、
  没有 packed bf16，所以视觉塔与 prefill 权重一律走 fp16。
- 互连与存储：PCIe pinned H2D/D2H **13.1 / 14.1 GB/s**、4 KB 拷贝延迟 **47.6 µs**；
  CPU DRAM 读 1/7 线程 **20.3 / 35.5 GB/s**；NFS 顺序 **0.56 GB/s**、随机 4k p99 **14.9 ms**；
  CPU/GPU 重叠效率 **88.6%**。这三条合起来否掉了"把 KV 放到 CPU/硬盘参与 decode"的路。

## 3. 负结果（做过、量过、不划算，别再试）

- **批模式下投机解码 0.74x**：B=8、K=1、接受率 85.2%，96.14 ms/step vs 贪心 38.43 ms/step，
  显存 +1.25 GB。原因是 verify 要核 c 个候选 token，每个都过一遍主模型权重 —— 用 c 倍
  访存换 1/c 倍收益。**保留 opt-in 开关，默认关。**
- **"一次前向 verify" 更差（0.27x）**：M=16 掉进 prefill 的 dequant+hipBLAS 路径，
  有 ~250–290 ms 固定成本。
- **int4 净收益 ≈1.4x，不是 2x**：int4 GEMV 是 **ALU 受限不是带宽受限** —— 有效带宽只有
  int8 的 0.65–0.73×，但每个权重只占一半字节 → 净收益 ≈1.4×；按权重混合平均 ~430 GB/s
  ⇒ 32 ms/token（≈30 tok/s）就是地板。
- **CPU 注意力 / PCIe / 硬盘都不能进 decode 循环**：256k 全量精确注意力走哪条都只有
  3–5 tok/s；CPU 注意力在 7 核上比 PCIe 还慢一个数量级。
- **现有注意力核 KV 读放大 ×6**（一个 block 管一个 q head，rep=6 时 6 个 q head 各读一遍
  同一个 kv head）→ 256k 真实流量 51.5 GB/token、216 ms/step。**GQA-aware 改写是前置条件，
  不是优化项。**

## 4. 27B + 256k 的规模账

- 27B int4 group128 权重 ~14.3 GB，单卡可用 15.5 GiB，扣常驻后**热 KV 只剩 1.4–1.7 GiB**
  （≈45–55k token int8），而 **256k 的 KV 是 8.66 GB（int8），永远进不了显存**。
- 所以"只放高频 KV"必须落成**注意力稀疏选择**（页级 top-k + 频次热区 + sinks/最近窗口），
  不是单纯的存储分层。
- **结论：单卡 27B + 256k 不可行，必须 2 卡 TP。**

## 5. 坑清单（写错就静默出错，或直接挂）

**数值/布局**

- `RMSNormGated` 要沿**最后一维**归一（`[32,128]`），不是压成 1D。
- `cos/sin` 是 fp32，和 bf16 相乘会把 q 提升成 fp32。
- delta rule 里 k/q 外积**广播方向写反**（`[:,:,None]` vs `[:,None,:]`）。
- 同时留 `w` 和转置 `wt` 会**双倍显存直接 OOM**。
- `q.reshape(H, nc, L, dk)` 对**转置视图**返回 strided view → 同一个核 **392 µs vs 79 µs**
  （整条 delta 26.1 vs 13.8 ms/层）。

**运行期**

- `inference_mode` 会把状态张量变成 inference tensor，之后在模式外 `.zero_()` 报错 → 用 `no_grad`。
- 内核必须跑在 `at::cuda::getCurrentCUDAStream()`，否则**捕获不到** → graph replay 后输出 NaN。
- 注意力循环上界必须用 `nvalid = pos + 1`，且 `pos` 要是 **device 指针**，否则图捕获后位置被烤死。
- 归约缓冲要加 barrier（两个 `block_sum` 共用 `red` 会 race）。

**调用契约**

- `enable_paging()` 之后、`capture()`/`prefill()` 之前**必须 `assign_table`**，否则 `pos2slot`
  全零、所有位置映射到同一个 slot —— 现象像"模型不确定"，实为调用错误。
- `TORCH_EXTENSIONS_DIR` **必须按部署目录隔离**：共享目录时每个新目录都触发 9 文件 HIP 重编译
  并抢同一个构建锁，多人并发即死等（症状：进程 `kfd_wait_on_events`）。

**测量与工程**

- **跨进程比较 4% 级别的改动会被 ±3% 的抖动淹没**，要改结构就得同进程交替 A/B。
- 容器里 torch 只有 `libmkl_gnu_thread.so` → 跑任何脚本前必须 `MKL_THREADING_LAYER=GNU`。
- Git Bash 会把 `/root/...` 转成 Windows 路径 → 推送要加 `MSYS_NO_PATHCONV=1`。
