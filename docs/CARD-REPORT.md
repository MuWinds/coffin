# 卡上验收报告

## 1. 环境

- python: `3.10.16` (/data/software/Anaconda3/2024.10-1/envs/py310/bin/python)
- import cxrt+torch: ok — `cxrt /data/home/2024240384035/coffin-work/base/coffin-dv/cxrt/__init__.py
torch 2.1.0`
- import cxrt 之后 torch loaded: `torch loaded: False`
- `ixsmi`: `GPU  0: Iluvatar BI-V100 (UUID: GPU-61301513-7b38-5e5b-8aaa-eb5fee021bf3)`

## 2. 内核编译

| 组件 | 结果 | 耗时 | 尾部 |
| --- | --- | --- | --- |
| w8a8 | ok | 0.3s | `built` |
| gemv | ok | 0.3s | `built` |
| gemm | ok | 0.3s | `built` |

## 3. GEMM 探针

> 只有 ``legacy`` 这一档进判定：上卡实测厂商 ``libcublas`` 对新表的 ``computeType`` 一律 ``not supported``（三档 dtype 都一样），所以默认档已经切到旧表；``new`` 那一行只作记录，不算失败。

| 档位 | computes | 结果 | 尾部 |
| --- | --- | --- | --- |
| i8 | new | 记录（非判定） (1s) | `    ops.i8_nk(ax, aw, ao, stream=stream) /   File "/data/home/2024240384035/coffin-work/base/coffin-dv/cxrt/linalg.py", line 304, in i8_nk /     self.cublas.gemm_ex(m=n, n=m, k=k, /   File "/data/home/2024240384035/coffin-work/base/coffin-dv/cxrt/linalg.py", line 243, in gemm_ex /     raise RtError(f"cublasGemmEx({m}x{n}x{k}, ld={lda}/{ldb}/{ldc}) " / cxrt.errors.RtError: cublasGemmEx(11008x2048x5120, ld=5120/5120/11008) -> not supported` |
| i8 | legacy | ok (1s) | `[legacy] i8 2048x5120x11008  39.49 TFLOP/s  5.846 ms  max_abs_err=0 / numeric=OK scale=9218 worst=0 torch_loaded=False` |
| i8-4096 | new | 记录（非判定） (1s) | `    ops.i8_nk(ax, aw, ao, stream=stream) /   File "/data/home/2024240384035/coffin-work/base/coffin-dv/cxrt/linalg.py", line 304, in i8_nk /     self.cublas.gemm_ex(m=n, n=m, k=k, /   File "/data/home/2024240384035/coffin-work/base/coffin-dv/cxrt/linalg.py", line 243, in gemm_ex /     raise RtError(f"cublasGemmEx({m}x{n}x{k}, ld={lda}/{ldb}/{ldc}) " / cxrt.errors.RtError: cublasGemmEx(4096x4096x4096, ld=4096/4096/4096) -> not supported` |
| i8-4096 | legacy | ok (1s) | `[legacy] i8 4096x4096x4096  35.15 TFLOP/s  3.910 ms  max_abs_err=0 / numeric=OK scale=8339 worst=0 torch_loaded=False` |
| bf16 | new | 记录（非判定） (3s) | `    ops.bf16_nk(ax, aw, ao, stream=stream, out_code=rt.BF16.code) /   File "/data/home/2024240384035/coffin-work/base/coffin-dv/cxrt/linalg.py", line 343, in bf16_nk /     self.cublas.gemm_ex(m=n, n=m, k=k, /   File "/data/home/2024240384035/coffin-work/base/coffin-dv/cxrt/linalg.py", line 243, in gemm_ex /     raise RtError(f"cublasGemmEx({m}x{n}x{k}, ld={lda}/{ldb}/{ldc}) " / cxrt.errors.RtError: cublasGemmEx(11008x2048x5120, ld=5120/5120/11008) -> not supported` |
| bf16 | legacy | ok (3s) | `[legacy] bf16 2048x5120x11008  26.35 TFLOP/s  8.760 ms  max_abs_err=0.3528 / numeric=OK scale=103.2 worst=0.3528 torch_loaded=False` |
| f32 | new | 记录（非判定） (2s) | `    ops.f32_nk(ax, aw, ao, stream=stream) /   File "/data/home/2024240384035/coffin-work/base/coffin-dv/cxrt/linalg.py", line 377, in f32_nk /     self.cublas.gemm_ex(m=n, n=m, k=k, /   File "/data/home/2024240384035/coffin-work/base/coffin-dv/cxrt/linalg.py", line 243, in gemm_ex /     raise RtError(f"cublasGemmEx({m}x{n}x{k}, ld={lda}/{ldb}/{ldc}) " / cxrt.errors.RtError: cublasGemmEx(11008x2048x5120, ld=5120/5120/11008) -> not supported` |
| f32 | legacy | ok (3s) | `[legacy] f32 2048x5120x11008  12.63 TFLOP/s  18.277 ms  max_abs_err=0.0003357 / numeric=OK scale=103.2 worst=0.0003357 torch_loaded=False` |
| i8-ones-k4096 | new | 记录（非判定） (0s) | `    ops.i8_nk(ax, aw, ao, stream=stream) /   File "/data/home/2024240384035/coffin-work/base/coffin-dv/cxrt/linalg.py", line 304, in i8_nk /     self.cublas.gemm_ex(m=n, n=m, k=k, /   File "/data/home/2024240384035/coffin-work/base/coffin-dv/cxrt/linalg.py", line 243, in gemm_ex /     raise RtError(f"cublasGemmEx({m}x{n}x{k}, ld={lda}/{ldb}/{ldc}) " / cxrt.errors.RtError: cublasGemmEx(64x64x4096, ld=4096/4096/64) -> not supported` |
| i8-ones-k4096 | legacy | ok (0s) | `[legacy] i8 64x4096x64  0.24 TFLOP/s  0.140 ms  max_abs_err=0 / numeric=OK scale=4096 worst=0 torch_loaded=False` |

## 4. 静态审计

- 默认入口: 退出码 0 — `OK: 闭包里没有 torch 算子，只有分配与窄化`
- --all: 退出码 0 — `_solve_blocked（回退：扩展不可用时才走）: torch.empty_like@2524, torch.eye@2525, torch.bmm@2540, torch.bmm@2539, torch.bmm@2532, torch.bmm@2539`

## 4b. 无 torch 门槛

- 运行时断言: 退出码 0（0s） — `file: /data/home/2024240384035/coffin-work/base/coffin-dv/cxrt/torchshim.py / tensor: <class 'cxrt.tt.Tensor'>`

> 进程里没有真 torch：``sys.modules['torch']`` 是 ``cxrt.torchshim``，``torch.Tensor`` 是 ``cxrt.tt.Tensor``。

## 5. 离机单测

| 测试 | 结果 | 尾部 |
| --- | --- | --- |
| test_cxrt_w8a8.py | ok | `Ran 18 tests in 0.001s / OK` |
| test_cxrt_gemv.py | ok | `Ran 14 tests in 0.001s / OK` |
| test_cxrt_tensors.py | ok | `Ran 20 tests in 0.001s / OK` |
| test_corex_rt.py | ok | `Ran 40 tests in 0.009s / OK` |
| test_protocol_torchfree.py | ok | `Ran 6 tests in 0.005s / OK` |
| test_cxrt_attn.py | ok | `Ran 28 tests in 0.002s / OK` |
| test_cxrt_solve.py | ok | `Ran 12 tests in 0.006s / OK` |

## 6. 模型端到端（可选）

> int8 现在只有一条路（nk 原缓冲的 TN），`COFFIN_I8_GEMM` 这个切换开关已经删了，所以这里跑两遍同一个命令，只看「两次是否一致」——它检查的是确定性，不是 A/B。

| 轮次 | 退出码 | 耗时 | 尾部 |
| --- | --- | --- | --- |
| 1 | 0 | 5s | `B. False / Answer: / A / prefill 5 tok in 199ms (25 tok/s); decode 16 tok in 0.27s (60.15 tok/s)  (graph=False)` |
| 2 | 0 | 5s | `B. False / Answer: / A / prefill 5 tok in 198ms (25 tok/s); decode 16 tok in 0.26s (61.70 tok/s)  (graph=False)` |

## 7. 结论

**PASS**（编译/审计/无 torch/单测全过）

## 附录 A：编译（kn / w4 全删之后）

`card_verify` 的编译档只覆盖它自己 build 的三块（`w8a8` / `gemv` / `gemm`）。**全量**
是另一趟：`run_chat.py` 一启动就会把 `kernels/_SOURCES` 里全部 22 个源文件编进
`coffin_corex_ops_v55`，作业 5922–5926 都是这一趟，`error:` 计数为 0。删 kernels 时
暴露出来、已修的编译问题有两处：

| 症状 | 根因 | 修法 |
| --- | --- | --- |
| `no matching function for call to 'nk_kernel'` ×18（`w8a8_dp4a.cu`） | 用花括号配对删 kn 家族时，把 `nk_kernel` / `nk_launch` 之间的 `template<>` 头与几段 kn 文档留成了孤儿 | 整份重写该文件：只留 `cx_store_scaled`、`nk_kernel`、`nk_launch(_mode)`、`out_mode_of`、`w8a8_dp4a_nk` 与它的绑定 |
| `no template named 'CxScalar'` ×16（`gemv_device.inc`，从 `moe_block.cu` 进来） | `moe_block.cu` 新 `#include` 了这个 `.inc` 用它的稠密 nk GEMV，但没带 `corex_scalar.h` | `gemv_device.inc` 自己 `#include "corex_scalar.h"` |

## 附录 B：端到端，真 torch 路径（Qwen3.5-4B-int8-nk，单卡，`--no-graph`）

```
loading weights ...
loaded Qwen3.5-4B-int8-nk (qwen3_5) in 1.7s
prompt tokens: 5
output:
 Paris.
A. True
B. False
Answer:
A
prefill 5 tok in 4577ms (1 tok/s); decode 16 tok in 0.26s (61.24 tok/s)  (graph=False)
```

这一栏和上面 §6 是同一件事的两个读数：§6 是 `card_verify` 自己跑的两轮（`199ms` /
`198ms` prefill、`60.15` / `61.70` tok/s），这里是手工那一趟。

## 附录 C：同一条命令连跑会分叉（既有缺陷，不是本轮引入）

作业 5927 里同一条命令跑三遍，`2/3` 给出上面那段输出，`1/3` 在第二个 token 之后走飞
（` Paris. hamro that that that.n`）。作业 5928 把融合的层驱动关掉
（`COFFIN_LAYER_DRIVER=0`）再跑四遍，仍然 `1/4` 在最后一个 token 分叉
（`Answer: 8.`）。所以：

* **不是本轮引入的**：关掉本轮重写的 `moe_block.cu` 那条路径之后它还在。
* 分叉点落在近等概率处（`Answer:` 之后 `A` 还是 `8.`），而槽位顺序由
  `moe_dispatch.cu` 的 `atomicAdd(&offsets[e+1], 1)` 决定——那个文件本轮没有改，
  它上面自己的注释就写着「到达顺序决定 `token_of_pos`」。top_k 两路结果的 fp32 加法
  顺序因此每次不同，末位差一个 ulp 就足够在近等概率处翻转。
* 结论：这是**既有**的确定性缺陷；要修它得让归并按固定顺序做，那是另一件事。

## 附录 D：同一个进程，没有真 torch（2026-11-26 复验）

§4b 是 `card_verify` 的判定项，只断言「`sys.modules['torch']` 是壳」。下面这几次是把
**整条解码链**真的跑在壳上的实测：

| 作业 | 命令 | 结果 |
| --- | --- | --- |
| 5929 | 身份探针 | `verify: True` / `torch` = `cxrt.torchshim` / `torch.Tensor` = `cxrt.tt.Tensor` / `torch.cuda.is_available(): False` |
| 5930 | `python -m cxrt.run_qwen35`（全量 4B，稠密折表） | `rc=0`，441 个张量折表 46.0 s、建模型 23.2 s、prefill 5 tok 2.5 s、decode 8 tok 2.5 s，输出 `' Paris.\nA. True\nB'` |
| 5932 | `COFFIN_NO_TORCH=1 python run_chat.py --model ... `（稠密） | `rc=0`，`loaded ... in 105.2s`，输出 ` Paris.`，decode 3.10 tok/s |
| 5932 | 同上 `--int8` | 载入 0.8 s、出 prompt token 后停住：壳下 `torch.cuda.is_available()=False`，corex 后端 `usable()` 为假，回落到 `backends/reference`（numpy 上跑 4B int8，极慢）。**不是崩溃，是回退** |

也就是：**「进程里没有真 torch 就能出 token」这句话现在有作业号撑着**，而 int8 的卡上
快路径还卡在 `cxrt.tt` 仍是 numpy 后端这件事上（见 `docs/COREX-NO-TORCH.md`）。
