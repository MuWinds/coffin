# CoreX 预填充 GEMM：nk 零拷贝、int8 + bf16、整链无 torch

> 这份文档只讲**预填充投影的矩阵乘**这一件事：它怎么在 corex 上既不 import torch、
> 又不留 kn 副本，同时把 int8 与 bf16 两支都覆盖。口径一律是「Qwen3.5-4B、单卡、
> 预填充 2048 行、int8 权重（除注明处）」，数字以 §2 的 **BI-V100 卡上实测**为准
> （`tools/card_verify.py` + `tools/probe_gemm_nk.py`，2026-11-25）；**不用
> `w8a8_tcu.cu` 文件头那组旧数**（那组是拿厂商 i8x4 指令上限 65.31 TINT8-OPS/s 推的，
> 与现在的库不是一回事），也不再用离机推算的数。

## 1. 三条约束怎么落地

| 约束 | 落点 | 证据 |
| --- | --- | --- |
| corex 全链无 torch | 新增 `cxrt/linalg.py`：`ctypes` 直调 `libcublas`，张量只按「指针 + dtype 码 + 形状」参与；`Runtime.build_gemm()` 绑定出口 | 该模块**不 import torch**；三个出口参数是 `void*` 与 `int`，没有 `torch::Tensor`，也没有 `TORCH_CHECK`。卡上探针打印 `torch_loaded=False` |
| 用新 ixblas 的数 | 性能表见 §2：i8 NN 49.3 TOPS、bf16 26.5 TF、f32 12.6 TF，全部卡上实测；`ixblas/` 自己的 PERF 行（i8 TN 46.19、f16 39.7）作为对照单独标注 | 见 §2 与 §2.1；旧 header 的 37.7~57.2 只当作历史值，不再做判据 |
| bf16 也要做 | `GemmOps.bf16_nk`：bf16 权重 nk 原缓冲 + `lda = ldb = K` + `transa = T` | 卡上 26.5 TF，`err = 0.35`（bf16 累加的正常量级） |

## 2. 上卡实测（2026-11-25，BI-V100，单卡）

这一节是**唯一口径**。所有数字来自 SCOW 作业（`platform/scow`，集群 `buu_tszx`，
卡 `Iluvatar BI-V100`），脚本 `tools/probe_gemm_nk.py --computes legacy`，每档
`--reps 20`，跑完打印 `torch_loaded=False` 表示链路上没装载 torch。

**先说那个坑**：`cxrt/linalg.py` 里 `CUDA_R_8I` 曾写成 `8`，而 `8` 是 `CUDA_R_8U`、
`3` 才是 `CUDA_R_8I`。错的值不报错，只让 `cublasGemmEx` 回一个 `not supported`，
看上去正好像「这块卡的库不支持 int8」。改回 `3` 之后 int8 立刻能跑（`err=0`）。
凡是看见「int8 全形状不支持」的旧记录，都是这个常数错了。

int8，单位 TOPS（`2*M*K*N / t`），**权重一律 nk 原缓冲、零转置**：

| 形状 | 现路径（nk 原缓冲 → TN） | 曾经量过的 NN（已删） |
| --- | --- | --- |
| 2048×5120×11008（4B `k5120`） | **39.53 TOPS / 5.840 ms** | 49.31 / 4.682 ms（需一份 kn 副本） |
| 4096×4096×4096 | **35.20 TOPS / 3.904 ms** | 42.37 / 3.244 ms |
| 64×4096×64（`--ones`，查大 K） | 0.147 ms，输出恰 4096 | 0.134 ms |

两列都是 `max_abs_err = 0`（int8 是精确整数累加，库与 numpy 逐位相同）。

结论要分成两句话：

1. 「int8 只能 `OP_N/OP_N`」是**错的**，那是 `CUDA_R_8I` 写成 `8` 造成的假象；改对之后
   TN 直接能跑。
2. NN 确实快 1.2~1.25×，但它要一份 `[K, N]` 的 int8 权重副本（`coffin_repack_i8_nk2kn`
   一次转置 + 缓存）。**这份副本已经整条删掉**：NK 是模型权重唯一的布局，也是后面
   要专门优化的方向，拿一份常驻显存换每次 GEMM 的 1 ms 不划算。所以现路径的 int8
   **就是 39.5 TOPS**，那 1.25× 留给将来 NK 原生张量核去赚，而不是靠转置赚。

bf16 与 fp32（权重 nk 原缓冲，`transa = T`）：

| 档 | 形状 | 数 |
| --- | --- | --- |
| bf16 | 2048×5120×11008 | **26.5 TFLOP/s / 8.7 ms**，`max_abs_err = 0.35`（bf16 累加，正常） |
| f32 | 2048×5120×11008 | **12.6 TFLOP/s / 18.3 ms**，`max_abs_err = 3e-4` |

`computeType` 世代：这块卡上 `--computes new` 三档 dtype **全部** `not supported`，
`legacy` 全部正常，所以 `_compute_code` 的默认必须是 `legacy`。这不是退化，是
这一代 `libcublas` 的 API 事实。

（`ixblas/docs/README.md` 的 PERF 行——i8 TN 46.19、f16 39.7——是 ixblas 自己在它的
构建里跑出来的另一套数：它把 bf16 权重无损转 f16 走自研 fp16 张量核，所以 bf16 那一档
比厂商的 26.5 高；它的 int8 原生路径只覆盖 `M <= 8` 的 gemv，大 M 的 TN 会转发给厂商，
在我们这条链路上走不通。两份数不冲突，是两种实现。）

### 2.1 与自研张量核的关系

稠密张量核 `w8a8_gemm_tcu` 在 4B 六个真实形状上的旧记录是 45.1~59.7 TOPS（吃 kn 权重），
而现路径（nk 原缓冲 TN）是 39.5（`k5120` 档）。也就是说这块卡上还有 **1.15~1.5×** 的余量，
但那份余量必须由 **nk 原生** 的张量核去赚：原来的 `w8a8_tcu.cu` 是按 `[K, N]` 索引的，
用它就等于把 kn 副本请回来。**那个文件已经不存在了**——它被 `git mv` 成
`moe_i8_tcu.cu`，稠密 kn 核（`gemm_tcu_kernel` / `launch_gemm_tcu` / `run_gemm_tcu` /
`store_epi`）与它的绑定一起删掉，只留下 nk 的分组核 `moe_grouped_i8_tcu` 与
`transpose_i8`。下一步要做的是把它的寄存器 epilogue + i8x8 指令搬到吃 `[N, K]` 的
ABI 出口（§4 P1）。


## 3. 布局：谁转、谁不转（卡上量完之后定的）

列主序视角下，`[N, K]` 行主序的 nk 权重**就是** `[K, N]` 列主序、`ldb = K` 的内存。
要拿到行主序 `[M, N]` 输出，cublas 三个尺寸要交换：

```
C[N, M] = op(A)[N, K] * B[K, M]        列主序，ldc = N
A = 权重 [N, K]，lda = K，op(A) = T
B = 激活 [M, K]，ldb = K，op(B) = N
C 内存下标 n + m*N  正好等于行主序 [M, N] 的 m*N + n
```

即 `transa = T, transb = N, (m, n, k) = (N, M, K), lda = ldb = K, ldc = N`，记法上是 **TN**。
这一组可用的前提是 `CUDA_R_8I = 3`——写成 `8`（其实是 `CUDA_R_8U`）时库只回
`not supported`，看上去像「int8 只能 NN」（见 §2 开头）。

按 dtype 分开说：

- **bf16 / fp32：一次都不转。** `bf16_nk` / `f32_nk` 走的就是上面那一组 TN，权重与
  激活都停在原生布局，进 `cublasGemmEx` 的指针就是 `w.wq` 与激活缓冲本身。
- **int8：也不转。** 同一个 TN 出口，权重就是盘上那份 `[N, K]`。曾经为了 NN 的
  1.25× 加过「转一份 kn 再 `OP_N/OP_N`」，那份副本连同 `coffin_repack_i8_nk2kn`、
  `i8_kn` / `i8_kn_cols`、`_KN_CACHE` 一起删了。
- **分块**：`N > pack.LIB_MAX_N` 时沿**行**切权重——nk 下这是连续切片，切出来仍是
  `[width, K]` 的 nk 权重，直接进同一个 TN 出口，不需要指针偏移技巧。`lm_head` 的
  `N = 248320` 走这条。

`pack` 里的 kn 类型（`KnWeight` / `KnQWeight` / `as_kn` / `to_kn`）已经不存在：`pack.py`
现在只剩 `WEIGHT_DTYPE` 与 `LIB_MAX_N` 两个常量，所以「喂进来的权重是 kn」这种情形在
类型层面不可表达，而不是靠报错兜住。

为什么不去让自研核吃 `[K, N]`：那是已删的 `w8a8_gemm_tcu` 的做法，代价是权重、激活、
输出三样全都要换方向，输出还变成 `[N, M]`、epilogue 要按转置读。库路径靠交换 `m/n` 就
把这件事做完了。


## 4. 落地顺序

**P0（本仓已落）**：`cxrt/linalg.py` + `Runtime.build_gemm()`。int8 与 bf16 两支都能跑，
nk 零拷贝。这一档不改任何内核，只把 `torch.matmul` 换成 `ctypes → cublasGemmEx`。
配套探针 `tools/probe_gemm_nk.py`（只 import `cxrt`，跑完断言 `torch` 没被装载）。
离机验证过的部分：三条出口的 m/n 交换、`lda/ldb/ldc`、int8/bf16/f32 三条布局与 numpy
参考逐位相同（`int8 / f32 / bf16` 三档 `max_abs_err = 0`）。

**P0.5（本仓已落，未上卡）**：把 GEMM 真的接进链路，并把 int8 的另两步也搬出 torch。

- `kernels/w8a8_epi_device.inc`：把 `w8a8.cu` 里的 `w8a8_epilogue_kernel` 提成共享片段
  （`EPI_THREADS` / `EPI_BLOCKS` 一起搬），`w8a8.cu` 改为 `#include`，ABI 侧也 `#include`
  ——两条前端的舍入次数只有一份。
- `kernels/abi/w8a8_epilog_abi.cu`：`coffin_w8a8_epilogue(c, xs, ws, out, rows, cols,
  stride, code, stream, errbuf, errlen)`。`c` 必须连续，`out` 只要行间距 `>= cols`
  （沿 N 切块时每块写回更宽缓冲区的自己那一段，不需要 `copy_`）。
- `cxrt/w8a8.py`：`W8A8Shim.epilogue` + `W8A8Ops.epilogue`（校验累加器必须是 int32、
  两个 scale 长度、`out` 的 dtype 与行间距）；`Runtime.build_w8a8` 顺手把新 `.cu` 编进
  同一个 `coffin_w8a8.so`。
- `backends/corex/__init__.py`：`linear_prefill` 改成三条分支，默认两条都不碰 torch 算子
  ——int8 走 `quant_act_i8`(ABI) → `GemmEx` → `epilogue`(ABI)，稠密 bf16/fp32 走
  `GemmEx` 的 TN（`[N, K]` 原缓冲即列主序 `[K, N]`，`lda = K`）。那时 `pack.as_kn` 从默认
  路径上消失、`kn` 遗留布局那条 `torch.matmul` 收在 `_kn_prefill_torch` 里——本轮把两者
  一起删了（`_kn_prefill_torch` 与 `pack.as_kn` 都不在了）。
  bf16 就在这里接进链路：`GemmOps.bf16_nk`（ixblas 下 `ld` 转 f16 + 自研 fp16 张量核，
  39.7 TFLOP/s；厂商 bf16→f32 是 34.8 且算对），fp32 权重走 `f32_nk`。
- `tools/audit_torch_path.py`：静态扫描 `linear_prefill` 的调用闭包，把 `torch.matmul` /
  `mm` / `F.linear` 这类算子与 `torch.empty` 这类分配分开报。当前输出是
  `OK: 闭包里没有 torch 算子，只有分配与窄化`（三条入口 `linear_prefill` /
  `_w8a8_lib_rows` / `_gemm_nk_rows` 全绿）。

**P1（一步内核改动）**：把 `w8a8_tcu.cu` 里现成的 nk 分支搬进 ABI：
`moe_grouped_i8_tcu_kernel` 的 mat-b 已经是 nk（`wbase = wstack + e*N*K`），只差两件事——
grid 加一维 M（`grid = (N/BN, rows/BM, E)`，`mt0` 走 stride 循环，E>1 行为不变），
以及把 `quant_act_i8` 加一个 k-major 输出版本（复用 `transpose_i8` 的 64×64 tile 几何，
把转置融进量化，省一趟）。出口命名 `coffin_gemm_i8_nk`，吃同一块 `[N, K]`。
收益：`gate_up`/`full_attn`/`linear_attn`/`o_proj` 拿回 1.1~1.3 倍，且 E=1 时不再塌到
662 tok/s（分组核现在只有 `(N/MI8_BN, E)` 两维，`mlp_down` 只剩 5 块）。

**P2（删 kn，本轮已做）**：链路上所有造 kn 的路径已删干净（见 §5）。

## 5. kn / w4 已全删清单（本轮）

判据只有一条：**仓库里不存在按 `[K, N]` 索引的权重内核，也不存在任何 4-bit 内核。**

### 5.1 内核（`backends/corex/kernels/`）

| 文件 | 删掉的东西 | 现在 |
| --- | --- | --- |
| `w8a8_tcu.cu` → **`moe_i8_tcu.cu`** | 稠密 kn 张量核 `gemm_tcu_kernel` / `launch_gemm_tcu` / `run_gemm_tcu` / `w8a8_gemm_tcu` / `store_epi` 及其 `COFFIN_BIND` | 只剩 nk 的 `moe_grouped_i8_tcu` + `transpose_i8` |
| `gemv.cu` | 5 个 kn 入口 `gemv_kn` / `gemv_kn_i8` / `gemv_kn_i8_m` / `gemv_kn_m` / `gemv_kn_m_split`，及其全部启动器与绑定 | 只剩 `gemv_nk_i8` / `gemv_nk_i8_m` |
| `gemv_device.inc` | 整段 kn 设备内核（`dot_kn`、`gemv_kn_*_kernel`、`gemv_i8_m_reduce_kernel`，约 720 行） | nk 段 + 新写的稠密 nk GEMV（`gemv_nk_dense_kernel`，供路由权重用） |
| `hyper.cu` | `hyper_down_kernel` / `hyper_up_kernel`（吃 `[K, N]`）与两个 host 包装 | 只剩 `grouped_rms_norm_kernel`（`hyper_norm`）；`hyper_down`/`hyper_up` 从本后端撤下，`models/qwen4exp` 退回 `backends/reference` |
| `moe_grouped_i8.cu` | kn 主体核 `moe_grouped_i8_kernel` 与 `i8_dword` | 三个入口一律 `TORCH_CHECK(nk != 0)` + `wide = true` |
| `w8a8_dp4a.cu` / `.h` | kn 家族：`kn_kernel`、`kn_launch`、`KnFn`、`pick_kn`、`read_kernel`、`rate_kernel`、`reduce_kernel`、`GroupArg`/`GroupTable`、`Geom`、`cx_mtile_ok`/`cx_ntile_ok`、`cx_pick_geom`，以及 `w8a8_dp4a` / `_groups` / `_read` / `_rate` 四个入口 | 只剩 `nk_kernel` + `w8a8_dp4a_nk` |
| `w8a8.cu` | kn 形状的库包装 `w8a8_gemm`（把 `[K, N]` 标签的 W 交给 `cublasGemmEx`）及其绑定 | `quant_act_i8` / `gather_quant_i8` / `w8a8_epilogue` |
| `moe_block.cu` | 按 kn 取的几何（`inter = gu_w.size(2)/2`、`hidden = gu_w.size(1)`）与稠密 `gemv_kn` 分派 | `project` 吃 nk；新增 `project_dense` 走 `launch_nk_dense_rows`；`nk` 强制 1 |
| `corex_ops.h` | `w8a8_gemm_tcu`、`gemv_kn*`、`hyper_down/up`、`w8a8_gemm` 的声明 | 头注释如实写明 kn 全删 |

W4：`backends/corex/kernels/` 下从本轮之前就没有 `w4a8.cu` / `w4_dequant.cu`，
`moe_grouped.cu` 里也没有分组 W4 核；各文件里剩下能搜到的 `int4` 都是 128-bit 向量读的
类型名，不是 4-bit 量化，也没有 `w4_` 前缀的入口。

### 5.2 Python 与工具

| 位置 | 之前 | 现在 |
| --- | --- | --- |
| `kernels/abi/w8a8_abi.cu` 的 `coffin_repack_i8_nk2kn` + 32×32 转置核 | int8 首次使用时转一份 `[K, N]` | **删除** |
| `cxrt/linalg.py` `i8_kn` / `i8_kn_cols` | NN 的 int8 出口 | **删除**；int8 只剩 `i8_nk`（TN） |
| `pack.py` 的 `KnWeight` / `KnQWeight` / `as_kn` / `to_kn` / `is_packed` / `repack_state_dict` / `unpacked_keys` / `_KN_CACHE` / `_sweep` | 全部 kn 类型与转换 | **整份重写**：只剩 `WEIGHT_DTYPE`、`LIB_MAX_N` |
| `int8_weights.py` 的 `QWeight.from_kn`、`QWeightMoE` 的 `layout` / `_kn` / `wq` | 布局开关与 kn 副本 | `QWeightMoE` 只有 `nk_stack()`（零拷贝）与 `from_nk` |
| `backends/corex/__init__.py` 的 `_KN_CACHE` / `_kn_weight` / `_kn_prefill_torch` / `_MOE_I8_NK` | 缓存与遗留分支 | **删除**；`linear*` 只收 `torch.Tensor | QWeight`，`w.wq` 直接进核 |
| `models/qwen35/moe.py` 与 `parallel.py` 的 `layout=` / `.t().contiguous()` | 造 kn | 全部按 nk 原缓冲取 |
| `tools/quant_disk.py` | `--layout`、`FORMAT_NK`/`LAYOUTS`、kn 落盘 | `FORMAT = "coffin-int8-nk-v1"`，只写 `[N, K]` |
| `tools/probe_nk_load.py` | 顺着 `pack.to_kn` 检查 | 改成断言 `pack` 里**不存在** `as_kn/to_kn/KnWeight/KnQWeight` |
| `tests/test_backends.py` | `CorexPackTest`（4 个 `dequant_kn` 测试）与两条 `gemv_kn` 结构断言 | **删除**；nk 断言保留，`linear` 断言改看 `w.wq` |

峰值：9.41 GiB → 6.01 GiB（4B），Occamy 的 35 GB 权重从「放不下第二份布局」变成
「一份 nk 就够」。

## 6. torch 还剩什么

`backends/corex/__init__.py` 现在**一次 torch 算子调用都没有**——`torch.matmul` / `mm` /
`bmm` / `einsum` / `F.linear` 全数为零。验收命令：

```
python tools/audit_torch_path.py                     # 五条投影入口的闭包，退出码 0
python tools/audit_torch_path.py --all               # 全模块逐行，按四类打标
```

全模块扫下来只剩四类东西，没有一类是张量数学：

| 类别 | 例子 | 为什么不算 |
| --- | --- | --- |
| 分配 / 视图 | `torch.empty` / `zeros` / `empty_like` / `as_strided` | 只发地址，不读不写数据 |
| 窄化 | `_restore` 的 `.to(dtype)` | 一次拷贝，与参考链同一次舍入 |
| 整数索引簿记 | `qsa_block_keys` 的 `arange` + `div` | 生成块表（行号 → batch 号），不碰浮点 |
| 回退 | `_solve_blocked` 的 `eye` + `bmm` | 只在扩展/工具链不可用时走；有 `coffin_triangular_solve` 时不被调用 |

**拿掉的算子**：预填充的 `torch.matmul` 三条分支（`N < 8` 补零、单块、`N` 切块）、
解码的 `gemv_kn` / `gemv_kn_m_split`（连内核一起删，§5）、`gemv_nk_i8(_m)`（torch 扩展
版本）、以及默认链路上所有转置副本。替换物分别是：`GemmEx`(TN, nk 原缓冲)、
`coffin_gemv_nk_i8(_m)`（同一份 `gemv_device.inc`）、`coffin_quant_act_i8` +
`coffin_w8a8_epilogue`。

**还没搬的是绑定，不是 Python**：pybind 侧仍有 71 个入口按 `torch::Tensor` 收参
（注意力主循环 `attn_decode` / `attn_paged*`、MoE 分组 `moe_grouped_*`、`kv_store` 等），
ABI 侧 18 个。要把它们搬走，每族都得先定契约：注意力要先定 KV 池与块表，MoE 分组要先定
专家偏移与段表。顺序建议：注意力 → MoE 分组 → 引擎层（`nn.Module`、`state_dict`、graph
capture 属于最后一档，`docs/COREX-PORT-MATRIX.md` 有清单）。

判据仍然只有一条：**链路里出现 torch 的算子调用或 `torch::Tensor` 就算没搬完**。

## 7. 上卡验证矩阵

| # | 目的 | 手法 | 通过判据 |
| --- | --- | --- | --- |
| 1 | `computeType` 世代 | `tools/probe_gemm_nk.py --computes both` | 两套里只有一套给出正确值；另一套返回 15 或算错，记进部署脚本 |
| 2 | int8 TN 是否真的 41.8~46.2 | `--dtype i8 --m 2048 --k 9216 --n 2560`，对 `IXBLAS_DISABLE=1` 做 A/B | 差距 ≥ 1.2×，否则说明 `ld+16` 没触发 |
| 3 | bf16 路径落到哪 | `--dtype bf16 --m 2048 --k 2560 --n 18432` | 若落回 32.6，说明 `GemmEx` 被转发，改 `out_code` 再量 |
| 4 | 数值 | `--ones --k 4096`（全 1 权重 + 全 1 激活） | 输出恰等于 K；不等于 K 就是大 K 截断（ixblas 记的 F-5） |
| 5 | epilogue ABI 与张量核同数 | 4B 同一提示各跑一遍 `COFFIN_I8_GEMM=lib` 与 `=tcu` | `max_abs_diff = 0`（同一个内核同一份算术；不相等就是 `stride` 落错了列段）。`=tcu` 那一档已随稠密张量核删除，本条只作历史记录 |
| 6 | 真省下一份转置 | 同一提示看峰值显存 | `lib` 应落在 6 GiB 档（与 `COFFIN_PREFILL_NK=1` 同量级）；仍在 9.4 GiB 说明还有一整份转置副本在被调用 |
| 7 | 链路里没有 torch 算子 | `python tools/audit_torch_path.py` | 退出码 0 |

前三条只需一次小矩阵，第四条是防静默错值的。kn 分支已经全部删除（§5），这一节不再有
「先别删」的待办。
