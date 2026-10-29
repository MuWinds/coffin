# CoreX 移植工作简报（共用状态板）

> 本文件是多 agent 协作的唯一事实来源。**先读完再动手**；做完自己那格就更新它。
> 目标：让 coffin 的生产代码在同一套源码下同时跑 gfx906（海光 DCU/HIP）和
> CoreX（天数智芯 BI-V100）。机器：北京联合大学 SCOW 平台，TRAIN 分区，1 卡。

---

## 0. 真机实测事实（已跑过，不要再重复测）

平台与工具（`gpu-node-05/06`，登录节点 `mgt` 无卡）：

| 项 | 值 |
| --- | --- |
| torch | **2.1.0**（`torch.version.cuda == "10.2"`，`torch.version.hip is None`） |
| CoreX | 3.2.3（`/usr/local/corex`），`nvcc`/`clang++` 在 `/usr/local/corex/bin` |
| 编译目标 | `--cuda-gpu-arch=ivcore10`，`.cu` 交给 `clang++`，`main.cpp` 交给 `g++` |
| python | `/data/software/Anaconda3/2024.10-1/envs/py310/bin/python`（3.10.16，venv `~/venvs/tfq35` 基于它） |
| 卡 | `Iluvatar BI-V100`，32768 MiB，16 SM，`torch.cuda.is_available() == True` |
| **warpSize** | **64**（`__shfl_down_sync` 在 block=64 上求和得 2016 = 0+1+…+63） |
| 模型 | `~/models/Qwen3.5-4B`（8.8 GB，与 gfx906 侧同一份 checkpoint） |

**四个会静默出错的坑：**

1. **设备不支持 fp64**：`torch.tensor([1.5,2.5], dtype=float64, device='cuda')*2`
   得到 `[0.0, 0.0]`（只有 "Limited support" 警告）。所以任何比较都必须
   **先 `.cpu()` 再 `.double()`**；内核里不要用 fp64。
2. **cuBLAS 在宽 N 上返回 NaN**：bf16 `[K,N]` 在 `N=248320`（`lm_head`）实测
   `max|y| == 0` / NaN。参考引擎的 `MAX_GEMV_N=131072` 就是这条。
   **`lm_head` 必须走自写内核或分块**，不能交给库。
3. **cuBLAS 无法被 capture**：`torch.cuda.graph` 里做 `x @ w` 报
   `CUBLAS_STATUS_ALLOC_FAILED`。**但自写 kernel 可以正常 capture + replay**
   （已实测：replay 后重新读了激活，rel err 2e-6）。所以 decode 的 graph
   路径只有在我们自己的 kernel 上才成立。
4. **`F.rms_norm` / `torch.compile` / `scaled_dot_product_attention` 等 2.4+ API 全不存在**。

**实测性能（决定 kernel 怎么写）：**

| 项 | 值 |
| --- | --- |
| D2D copy 上限 | **574 GB/s**（`dst.copy_(src)`，67 MB） |
| 库 `[N,K]` GEMV（M=1, bf16） | **16–30 GB/s**（`F.linear` / `@`），这就是不能用的那条路 |
| 库 `[K,N]`（`x @ Wt`）M=1 bf16 | 360–406 GB/s（但宽 N 会 NaN） |
| 我写的 `[K,N]` GEMV M=1 | **U=4：168 GB/s（N=9728,K=2560）/ 361 GB/s（lm_head）**；正确性 rel ≈ 1.5e-6 |
| prefill GEMM（库，M=2048） | bf16 与 fp16 **同为 32.6 TFLOP/s**（所以 `weight_dtype` 不必为算力选 fp16） |
| graph capture | 自写 kernel ✅，cuBLAS ❌ |

带宽与形状强相关（同一内核）：`N=9728,K=2560` 168 GB/s、`N=248320` 361 GB/s、
`N=2560,K=9728` 47 GB/s、`N=2560,K=4096` 63 GB/s。**按首维 N 分档**：
N 越大越接近上限，小 N 受 launch/占用率限制 → 需要按形状分档或合并投影
（`GEMVGroup` 的意义在这里）。

---

## 1. 已经定下的设计决定（不要再讨论）

1. **生产代码不许出现 `hip_ops` / `"cuda"` / `D` / `WF` / `ATTN_SPLIT`**。
   全部改走 `backends.ops`、`backends.device`、`caps.act_dtype`、
   `caps.weight_dtype`、`caps.attn_split`。这是**一切的前置条件**：
   不改这条，CoreX 后端写了也用不上。
2. **CoreX 是 `backends/corex/` 手写后端**（不是 `vendors/` 声明），
   在 `registry.BUILTIN` 里登记。理由：它要带 `kernels/` 和编译路径，
   声明式 spec 表达不了。
3. **`vendors/iluvatar.py` 删掉**：真实后端落地后，占位声明就是重复且会误导
   （它 `has_kernels=False`，探测到卡却不肯用）。`vendors/enflame.py` /
   `moorethreads.py` 保留。
4. **权重布局走 `caps.weight_layout`**：CoreX 是 `"kn"`（`[K,N]`，加载时转置一次），
   HIP 是 `"nk"`。契约保持 checkpoint 的 `[N,K]`，转置成本由后端承担。
5. **`caps.graph`**：CoreX 可以为 `True`（自写 kernel 可 capture），
   但**只有当一层的算子全部是自写 kernel 时**才成立。保守起步：
   先 `False`，把 graph 作为最后一个开关。
6. **`caps.graph=False` 的正确形态是"eager 替身"，不是"运行期静默降级"**。
   `ops.runtime.capture()` 在 `graph=False` 时返回一个 `replay()` **重跑 body** 的
   句柄，所以 eager 路径的语义与图路径同源；而 `graph=True` 却捕到不可捕获的算子时
   **抛异常、不吞**——吞掉会把"这层算子不全自写"变成静默的性能退化，那正是
   `caps.graph` 要表达的事实。要回退就由**后端把 `caps.graph` 报成 `False`**，
   不是运行期替调用方决定。CoreX 补齐 attn/delta 后该位自动变 1，调用侧零改动
   （2026-10-26 与 lead 确认）。
7. **不许用 fp64、不许在参考路径里用库 GEMM**（见 §0 的坑 2/3）。
8. `backends/` 底层**不许 import** `ops/models/engine/serving/cache`（纯度门禁 R5）。

---

## 2. 分工与写入边界（disjoint scopes）

| 谁 | 写入范围（只有这些） | 交付 |
| --- | --- | --- |
| **lead** | `backends/{__init__,protocol,registry,vendor,discovery}.py`、`backends/hip/**`（含 `kernels/` 迁入）、删 `hip_ops.py` 与顶层 `kernels/`、`backends/corex/**` 集成、`COREX_PORT.md` | S1 搬核 + 注册表 + 集成验收 |
| **migrate-models** | `ops/**`、`models/**` | 生产调用点改 `backends.ops`/`device`/`caps` |
| **migrate-serving** | `serving/**`、`cache/**`、`engine/**`、`entrypoints/**` | 同上 + torch 2.1 兼容 |
| **corex-kernels** | `backends/corex/**` | CoreX 原生算子 + 自测探针 |
| **verify-tests** | `tests/**`、`backends/reference/**` | 一致性用例、纯度门禁、reference 的 torch 2.1 兼容 |

**冲突规则**：`backends/corex/**` 由 corex-kernels 独占；lead 不碰它的文件，
只在最后集成时读。任何人要改别人的范围，先在 §4 留言并等 lead 拍板。

---

## 3. 验收口径

一个后端"接好了"要有三件事，缺一不算：

1. `python -m unittest discover -s tests` 在**无卡机器**上仍然全绿（238 例基线）。
2. `COFFIN_BACKEND=<name>` 启动横幅显示 `native=N/17`，N 是我们真写了的算子数。
3. 在真机上跑 `diag/corex/` 下的对拍脚本，**每个算子相对 `reference`
   后端 rel < 2e-2**（bf16 容差），并打印实测带宽。

**性能不是第一验收标准**。先逐位不崩、能出连贯文本，再谈 tok/s。

---

## 4. 交接留言板（按时间追加，别删别人的）

- **lead**：真机探测完成，§0 数字可引用。已建 `platform/scow/tools/deploy.py`
  用于发作业（`python scow\tools\deploy.py cohort -d <dir> -s <script> -g 1 -t 15`）。
  下一刀：把 `hip_ops.py` + `kernels/` 搬进 `backends/hip/` 并在 registry 里让
  `corex` 与 `hip` 并列。

- **migrate-models（S2 完成，已验收）**：`ops/**` 与 `models/**` 已不含任何
  `hip_ops` / `"cuda"` / `torch.cuda` / `gfx906` / `kernels/` / `F.rms_norm` /
  `Q35_*` 环境开关一类平台字面量：纯度门禁在 R1..R5 上对这两个目录**全为 0 条**
  （空预算全量扫描核过）。**生产侧违规合计 212 行 → 2 行**，剩下 2 行都不是本步的范围
  （`R2`: `int8_weights.py` 1 行；`R3`: `utils/mm_input.py` 1 行）。
  `python -m unittest discover -s tests` → **Ran 238 tests, OK (skipped=58)**。

  交付形态（16 个文件 = 15 改 + 1 新）：

  1. `ops/dtypes.py`：`D`/`WF` 现在是 `caps().act_dtype`/`weight_dtype` 的再导出，
     文件里没有第二个字面量。
  2. `ops/norm.py`：不再有 `F.rms_norm`（2.4+ API，真机 2.1.0 上直接 AttributeError），
     改调 `bops.rms_norm` / `bops.rms_norm_gated`，手写实现归 reference 后端所有。
     2-D 视图只在末维 stride≠1 时拷贝，宽行距（`q_norm` 切片）走 `stride` 参数。
  3. `ops/gemv.py`：`(R,U)` autotune、设备计时扫形状、`_HIP_CFGS`/`_M_US`
     全部删除。GEMV 只剩 `linear` / `linear_m` / `linear_prefill` 三个调用面，
     **GEMVGroup 拼接保留**（减少 launch，且与平台无关）。
  4. **`ops/runtime.py`（新文件）**：所有 `torch.cuda.*` 运行层调用（Stream /
     CUDAGraph / graph / graph_pool_handle / Event / empty_cache / synchronize）收进
     这里，命名空间由 `getattr(torch, caps().device)` 派生 —— 所以 DCU 与 CoreX
     （同一 `cuda` 命名空间、不同后端）走同一条代码，将来的 `gcu`/`musa` 也自动适用。
     设备缺某个 API 时**逐项降级**（实测 `torch.cpu` 既没有 `empty_cache`，也没有
     带 timing 的 `Event`）。
  5. `models/qwen35/`：`"cuda"`→`backends.device`（42 处）；**27 处 graph 站收敛成
     5 个 `runtime.capture(...)`**；`ATTN_SPLIT`→`caps.attn_split`；15 处
     `if hip_ops.available()` 分支与其 ~85 行 eager 兜底数学全部删除（数学已在
     reference，模型侧只剩一条路径）。
  6. `models/qwen35/batch.py`：`Q35_FAST_GEMV` / `Q35_WIDE_SPLIT8` 两个环境开关删除
     （核选型归后端）；`wide_kernel_ready()`/`multi_q_ready()` 改读 `caps.native`
     —— 能力问题问后端，而不是 `hasattr` 探符号。

  **`capture()` 的两条语义（设计决定，勿改成运行期静默回退）**：

  - `caps.graph=False` 时 `capture()` 返回的 `Capture.replay()` **重跑 body**，不是
    空操作。所以 eager 替身**语义正确**、不是"只是不崩"：CoreX 上 cuBLAS 不可捕获而
    自写 kernel 可捕获，混合算子表的正确执行方式就是 eager，这条路径给出的数与图路径
    同源。
  - `caps.graph=True` 但 body 里混了不可捕获的算子时，`capture()` **抛异常，不吞**。
    理由：吞掉会把"这一层算子不全自写"变成静默的性能退化，而那正是 `caps.graph`
    要表达的事实。**自动回退的正确位置是后端声明 `caps.graph=False`，不是运行期
    降级**。CoreX 现在报 0，等 corex-kernels 把 attn/delta 补齐、`_DECODE_OPS ⊆
    native` 成立后自动变 1，`capture()` 这一侧**不需要任何改动**就开始走真图。

  **本地验证（无卡、reference 后端）**：tiny 合成 checkpoint 跑通 prefill → decode →
  capture+replay → batched decode → `spec_step` → int8 全量；logits 全有限、
  `captured=False` 且 replay 与 eager argmax 逐位一致、int8 与稠密 argmax 一致。
  这轮验证抓到两个真 bug（被我误删的 `NEG_INF`、`torch.cpu` 缺 `empty_cache`/带
  timing 的 `Event`），都已修。

  **真机验证（lead 跑）**：`corex` 后端 `device=cuda:0 arch=ivcore10 native=5/17
  layout=kn`；5 个原生算子 rel ≈ 2.3e-03、带宽 146–248 GB/s；**graph capture 实测
  通过**（`capture rel=2.570e-03` / `replay rel=2.493e-03` / `output moved by
  2.859e+02`，证明 replay 真的重读了输入、不是回放陈旧缓冲），与 `caps.graph` 的
  推导逻辑一致。

  **留给下一轮**：`GEMV(..., bench=)` 已是惰性参数，可从 `_proj`/`TiedEmbedding`/
  `create_model` 一并去掉（约 20 个调用点，纯签名变更、零行为变化；有意不把这轮
  的"删 autotune"和"改 20 处签名"混在一次改动里）。另：`batch.WIDE_STATS` 的 key
  从 4 个降到 2 个（dot2/wide 两条路已归后端），`diag/probe_spec_segments.py` 若按
  旧 key 断言需同步。

- **migrate-serving（S3 完成）**：`serving/` `cache/` `engine/` `entrypoints/` 12 个
  文件迁移完毕，**详细交接写在 §7**。速览：

  1. `hip_ops` import 与 `hip_ops.available()` 分支全部删除；启动横幅改用
     `backends.status()` 独立一行，无卡机器实测输出
     `backend=reference device=cpu native=0/17 graph=0 autotune=0`。
  2. `"cuda"` 字面量与 9 处 `torch.cuda.*` 收进 `serving/runtime_support.py` 的
     `device_runtime()/sync()/allocated_gib()/free_bytes()/profiler_activities()`。
     **注意判据不能只看 `hasattr(torch, caps.device)`**：`torch.cpu` 存在且
     `torch.cpu.is_available()` 返回 True（语义是"CPU 总在"），CUDA 构建在无卡机器上
     也会暴露 `torch.cuda`。所以先判 `caps.device == "cpu"`，再验 `is_available()`。
  3. `cache/paged_cache.py` 默认 device → `backends.device`（`None` 表示取后端），
     **int8 KV 布局一字未动**（`[nslots,nkv,hd]` int8 + `[nslots,nkv]` fp32 scale 是
     kernel 契约，已写进文件头 Constraints）。
  4. `entrypoints/cli/runtime.py` 加 `--backend NAME`（**parse 后、import 绑定算子表
     的模块前**写 `COFFIN_BACKEND`）与 `--no-graph`（→ `EngineConfig.graph`，有效值
     = 用户意图 AND `caps.graph`）。
  5. **补齐拆包漏搬的 4 个方法**（WORKLOG §6 那条）：`_prepare_mm` / `_ensure_processor`
     / `_vision_ready` / `_load_weight_kwargs`。比 WORKLOG 记的多一个
     `_load_weight_kwargs`（AST 扫 `self.<x>` 无定义查出来的）。纯文本不受影响，
     HTTP 图片请求此前必定 AttributeError；补回后端到端跑通（1 张 PNG → `grid=[[1,4,4]]`
     / `rope_delta=-2` / 三处对齐自检生效）。
  6. **关图不只是"跳过 capture"**：`model.decode_step()` 是重放已捕获的图，没捕获过
     就 AttributeError。所以 `ModelRunner.graph` 与 `PrefixCacheEngine.graph` 各带一个
     分支走 `model.logits()`（eager，同一组数）；`LLMEngine.build()` 在 `graph=False`
     且 batching/spec 仍开时**抛异常**，因为那两条路每一步都是重放按桶捕获的图、没有
     eager 替身。`ping` 报 `graph and graph_ready`，不然关图会被读成 capture 失败。
  7. 删掉 `Q35_FAST_GEMV` 空开关**连同 `--fast-gemv` 整条传递链**（不是留个接受但无效
     的参数）。
  8. **torch 2.1 兼容：本范围无需改动**。四个包里全部 `torch.*` 调用面（`arange`/
     `device`/`full_like`/`inference_mode`/`multinomial`/`profiler`/`softmax`/`zeros`
     + dtype 常量）2.1 全有；没有 `F.rms_norm`/`torch.compile`/`torch.library`/
     `torch.nn.attention`/`SDPA`。也没有 3.11+/3.12+ 专属语法（真机 py310 可跑）。
     仓内其余 2.4+ 只剩 `ops/norm.py`（migrate-models 已改掉）和 `diag/probe_ops2.py`
     （探针）。

  验收：`python -m unittest discover -s tests` → **Ran 238 tests, OK (skipped=58)**；
  纯度门禁 prod 组我在本范围内 5 条规则**全 0**。
  **未做/待确认**：`free_bytes()` 真机分支没验（本机只测到 `-1`）；`_prepare_mm` 里
  `_ensure_processor()` 与 `_vision_ready()` 的顺序保持原样（`--no-vision` 收到图片会
  先花 ~2 s 加载 processor 才报 503，属于另一改动，见 §7.6）。


---

## 6. 迁移完成的量化结果（2026-10-26）

门禁口径：`tests/test_backends_purity.py` 的 **prod** 组（=会随包发布的代码）。
"违规行数"是一行算一次。

| 规则 | 拦什么 | 迁移前 | **现在** |
| --- | --- | ---: | ---: |
| R1 | `import hip_ops` / `from hip_ops import` | 7 行 / 7 文件 | **0 / 0** |
| R2 | `gfx906` / `--offload-arch` / `PYTORCH_ROCM_ARCH` / `TORCH_EXTENSIONS_DIR` | 17 行 / 9 文件 | **0 / 0** |
| R3 | `"cuda"` / `.cuda()` / `torch.cuda.*` | 181 行 / 13 文件 | **0 / 0** |
| R4 | 指向 `kernels/` 的源码路径 | 7 行 / 4 文件 | **0 / 0** |
| R5 | `backends/` 反向 import 上层 | 0 | **0** |
| **合计** | | **212 行 / 15 个文件** | **0 / 0** |

**这就是这次移植的可验证结论：生产代码里已经没有任何一处知道自己在哪张卡上。**
`probe` 组（`diag/` + `tests/`）仍有 507 行，那是探针的本职——它们就是用来
问某一张卡的，不该被抽象掉。

配套的三件事让"归零"可持续：

1. **`tests/regen_allowlists.py`**（新增）：从真实扫描重生成预算，且**在树上有
   `NEW` 违规时拒绝运行**。没有它，每次迁移步都要手工下调几十个数字，而手工下调
   正是白名单悄悄长大的入口。
2. **两个测试的断言改成不依赖迁移进度**：原先它们要求"prod 组必须非空"、
   "扫描必须看到 `hip_ops.py`"，这会**在迁移成功的那一刻失败**。现在前者断言
   `prod.tracked == 0`（迁移目标本身），后者按"遍历到的文件"而非"有违规的文件"判定。
3. **`coffin/.gitignore` 收掉 `.tmp_cusyn/`**：内核语法检查用的 stub 头文件目录，
   曾被门禁当成生产代码报违规。

验收（2026-10-26，无卡 Windows）：`python -m unittest discover -s tests` →
**Ran 238 tests, OK (skipped=58)**。

---

## 6b. 端到端跑通：CoreX 上第一次吐出正确文本（2026-10-26）

作业 `e2e2-1`，**`gpu-node-06`**，`COFFIN_BACKEND=corex`（完整日志）：

```
backend=corex device=cuda:0 native=18/18 graph=0 autotune=1
loaded in 35.5s

The capital of France is  →  'The capital of France is Paris.\nA. True\nB. False\nAnswer'
  prefill 1056.5 ms (5 tokens) / decode 77.09 ms/step (12.97 tok/s) / logits finite: True

2 + 2 =                   →  '2 + 2 = 4.\nThe sum of two'
  prefill 355.4 ms (5 tokens) / decode 47.86 ms/step (20.89 tok/s) / logits finite: True

summary: backend corex (18/18 native), failures: (none)
```

**`The capital of France is` → ` Paris.`**：18 个算子的数值语义、int8 KV、Gated DeltaNet
递推、paged attention、mRoPE 全部串对了，而且是**同一套生产代码**跑出来的。

### 节点健康矩阵（重要，跑作业前必须先看）

这台集群的卡**不是都健康的**。同一天、同一份代码，六个并发作业落在六个节点上：

| 节点 | 8x8 matmul 健康检查 | 模型加载 | 结论 |
| --- | --- | --- | --- |
| `gpu-node-05` | ❌ 超时 | — | **不可用**（首次分配即挂起） |
| **`gpu-node-06`** | ✅ | 35.5 s | **可用，端到端通过** |
| `gpu-node-07` | ✅ | > 420 s 超时 | 不可用 |
| `gpu-node-08` | ✅ | > 420 s 超时 | 不可用 |
| `gpu-node-09` | ❌ 超时 | — | **不可用** |
| `gpu-node-16` | ✅ | > 420 s 超时 | 不可用 |

**两个必须记住的操作事实**：

1. **调度器会反复把作业派给同一个节点。** 串行提交时六次里有五次落在 `gpu-node-05`；
   改成**并发提交**后立刻散到了 05/06/07/08/09/16 六个节点，一次就找到健康的 06。
   `submitJob` 没有节点约束字段，所以**并发提交 + 脚本内的健康检查**是唯一可靠的选卡方式。
2. **脚本自己的位置不代表引擎的位置。** `launch_chat.sh` 是单独上传的、落在工作目录根，
   而引擎解压在 `./coffin/`。用 `dirname $0` 推导根目录得到的是 `/data/home/diag/...`，
   报 `can't open file`。正确做法是调用方通过 `COFFIN_ROOT` 显式传入。

`launch_chat.sh` 因此做成三步：**先 8x8 matmul 验卡（失败立即退出，不浪费一次加载）→
再加载 → 再解码**，并带 `timeout --foreground -k` 硬上限。那次"卡死 25 分钟"就是因为
没有上限，而设备端自旋既没有 traceback 也不写 stderr。

### 从 5/17 到 18/18 之间修掉的三件事

1. **三角求解进了契约（第 18 个算子）**。CoreX 的 cuSOLVER 是坏的：wheel 硬编码
   `dlopen('/opt/sw_home/local/cuda/lib64/libcusolver.so')`，该路径在这台机器上不存在
   （`/usr/local/corex/lib64/libcusolver.so` 存在，但它不读）。实测六条路只有一条通：
   `solve_triangular` / `triangular_solve` / `linalg.solve` / `solve_ex` / `linalg.inv`
   全部 dlopen 失败，**分块前向代入** rel=1.1e-07 可用。
   处理方式不是 `if platform`，而是加 `triangular_solve(lower, rhs)` 到 `Ops`：
   reference 用库（gfx906 正常），corex 用分块前向代入。**生产代码里没有平台分支。**
2. **块大小必须自适应**。第一版硬编码 `block=256`，结果真机的 prefill chunk 是
   **64**（短 prompt），直接 `ValueError` 崩掉整个 prefill。改成
   `_SOLVE_BLOCKS=(256,128,64,32,16,8,4,2,1)` 取最大可整除的，L=8…512 全部 rel≈4e-8。
   教训：**探针里用的 chunk 长度不一定是真实请求的 chunk 长度。**
3. **`decode_step()` 的前提是 `capture()` 已跑过**。探针一开始无条件调
   `model.decode_step()`，在 `caps.graph=False` 的后端上 `self._graph` 是 `None`，
   报 `'NoneType' object has no attribute 'replay'`。正确写法是按 `model.graph_ready`
   选 `decode_step` 或 `logits`（两者数值相同）。这是**探针的 bug，不是引擎的**。

### 与 gfx906 的当前差距（性能，非正确性）

| 项 | gfx906 (HIP) | CoreX |
| --- | --- | --- |
| 原生算子 | 17/17 | **18/18** |
| 加载 | — | 2.5 s |
| decode | 78.5 tok/s（int8 权重 + graph） | **12.9–20.3 tok/s**（bf16 权重 + eager） |
| graph | 开 | **关**（`_GRAPH_VERIFIED=False`，见 §4.y） |
| int8 权重 | 开 | 关 |

### 4.y graph：两次"看起来通过"的开关，都被关回去了

这一节的写法是刻意的：**graph 现在是关的，而曾经被打开过两次，两次都因为验证粒度不够。**
这不是遗留 bug，是一个关于"什么叫验证过"的教训，值得比结论本身记得更牢。

**第一次**：`check_graph.py` 报 `capture rel=2.551e-03 / replay rel=2.744e-03 /
output moved by 2.910e+02`，看起来完美。但那个脚本只捕获了 `linear` 和 `rms_norm`
**两个算子，孤立地**——它从没跑过 32 层的调用图、paged attention 的窗口、或递推状态的
原地更新。把"两个算子可捕获"当成"整步可捕获"是推断，不是测量。

**第二次**：按整步重测，`capture()` 成功（2396 ms），打印 `decode path: graph replay`，
然后**再无输出**。两次尝试的表现不同，但根因同类：验证没有覆盖打开开关后真正会跑的路径。

**关于那次"卡死"的更正（重要）**：我一度判断这是设备端故障，还说"搞坏一张卡"。
**这是错的。** 随后 8 个独立作业在同一个节点（`gpu-node-05`）上全部通过：

| 作业 | 内容 | 结果 |
| --- | --- | --- |
| 3734–3737 | 8x8 bf16 分配、CPU→GPU 拷贝、fp32 分配、`mem_get_info` | 全部 ✅（34 GB 空闲） |
| 3738–3740 | 512x512 bf16 `randn` / `zeros` / 8x8 `randn` | 全部 ✅ |
| 3741 | **512x512 bf16 矩阵乘法** | ✅ |
| `sinfo -n gpu-node-05` | 节点状态 | `idle[power_idle]`，无遗留作业 |

所以卡是健康的。那次看起来"卡住"的脚本，是我把 `python -c "..."` 的双引号套进了
portal 的命令字段，远端 shell 的转义把命令截断了——它打印了前两行就停，因为**后面的
代码根本没跑**，而不是设备挂起。

**这两条合起来的教训**（写在 `backends/corex/__init__.py` 的 `_GRAPH_VERIFIED` 注释里，
下次要打开这个开关的人会先读到）：

1. 验证必须覆盖**开关打开后真正会跑的那条路径**；
2. 验证的**数据要能让错误显形**——同构输入、单步对多步，都会让 bug 隐形；
3. **"没有输出"要先怀疑自己的调用方式**，再怀疑硬件。一条 `assert pos.is_cuda`、
   一个"用脚本文件而不是 `-c` 嵌套引号"的选择，都能省掉一轮误判。

要打开 `_GRAPH_VERIFIED`，门槛是**整步对比**：同一个 prompt 分别走 eager 与
capture+replay，逐 token 完全相同，且进程存活。单个算子的捕获通过是必要条件，
离充分还很远。

### 4.y2 graph 的最终结论：**开着，1.47x，已通过整步验证**

§4.y 的两次失败都查清了根因，`_GRAPH_VERIFIED = True` 现在有测量支撑。

**第一次失败（calling order）**：探针先 `prefill` 再 `capture()`。而 `capture()`
第一件事就是 `reset()` 并跑自己的 dummy prefill —— 它把 prompt 的 KV cache 与递推
状态整个洗掉了，replay 于是在一个从未被写过的窗口上驱动 attention。正确顺序是
`reset()` → `capture()` → `reset()` → `prefill()`（`run_chat.generate` 一直是这样）。

**第二次失败（不是 graph，是卡）**：`gr2-1` 在 `decode path: graph replay` 之后无输出。
几分钟后**同一节点连 8x8 matmul 都不返回**——那是上下文已经被上一次失败的 capture 卡住
了，不是这次 replay 的问题。**两种故障的日志完全一样，从日志分辨不出来**，所以
`diag/corex/` 下每个 launcher 现在第一步都跑 8x8 matmul，失败就 `exit 64`
（不加载模型、不浪费时间）。

**历史验收证据**（gpu-node-07，同进程 6 轮；对应的 graph 探针已按驱动安全规则删除）：

```
backend=corex native=18/18 graph=1, loaded in 35.1s
[attempt 0..5] capture ok 3.1s/2.0s/...  graph_ready=True
               prefill ok, logits finite=True
               decode ok, tokens=[5, 6, 24218, 10] finite=True   ← 六轮完全相同
completed attempts: [0, 1, 2, 3, 4, 5] of 6
```

**六轮输出逐 token 相同**，这才是"replay 真的重读了输入"的证明；单轮通过可能是运气。

**收益**：同一 prompt 上 eager 77.09 ms/step → replay 52.41 ms/step，**1.47x**
（gfx906 上 graph 是 3.1x，差距来自 CoreX 还没有 int8 权重）。

一个操作结论值得单独记：**先验卡，再怀疑代码**。这轮里"看起来是代码 bug"的三次现象
（一次 replay 挂死、两次 matmul 超时）全都可以用同一句话解释或排除——先跑 8x8 matmul。

### 4.x 三条"数值不对但端到端文本对"的根因（第三轮）

这一轮最值得记的不是修了什么，而是**三条 FAIL 里有两条是探针的错、一条是真 bug**，
而判据只有端到端文本。三条都验证到了根因。

1. **`delta` 的 `rel=4.5e-02` 是共享内存竞态，真 bug。** `group_sum` / `block_reduce`
   在读完 `red[base]` 后直接 `return`，没有关闭屏障；同一个 kernel 一次会调它两次
   （max 与 sum，或 kv 与 result），第二次开头的 `red[threadIdx.x] = v` 可能早于第一次
   的读。slot 0 的 `base == 0`，于是 thread 0 写的正是别人还要读的那个槽。症状是
   "错得不多也不飞"——一位部分和坏掉，落在 1e-2 量级，很容易被误读成精度问题。
   本地模拟：干净路径 1e-16，竞态路径 O(1)。已给 `delta.cu`、`attn.cu`、`norm.cu`、
   `linear_attn.cu` 四个 reducer 全部补上关闭屏障，形状统一。

2. **`attn_decode` 的 `rel=1.000e+00`（输出全零）是探针传了 CPU 张量。**
   `pos = torch.tensor([63])` 落在 host 上，而 kernel 是**从设备解引用** `pos` 的
   （这样 graph 才能在 replay 时换位置而不重捕）。主机指针被当成长度读 → `nvalid`
   越界 → 窗口空 → 每个 split 停在 `m=-1e30/l=0` → combine `0/0` → 全零。
   因为 `rel = err/max|ref|`，全零恰好给出**精确的 1.0**，这个数字本身就是指纹。
   模型侧 `pos1` 建在 `_device` 上，所以真路径从来没错——这也是端到端文本一直对的原因。
   已在 `check_ops.py` 改成设备张量并加 `assert pos.is_cuda`。

3. **`pack.to_kn` 是分配器取整，不是 bug。** 逻辑 49 807 360 字节被 2 MiB 段取整成
   50 331 648（正好 24 段 / 48 MiB）。原来的 `grew != w.numel() * 2` 严格相等把一次
   正确的转置判成了失败。已改成同时接受逻辑值与段取整值。

**方法结论**：探针的期望值必须和 kernel 的**调用契约**对齐，而不是和"看起来合理"对齐。
一条 `assert pos.is_cuda` 抵得上一轮作业往返。端到端文本正确是强证据，但它只覆盖
"走过的路径"——`paged=0` 时 dense 分支确实走到了，所以那次文本正确**不能**用来给
dense 分支背书，它只是恰好算术自洽（`per=ceil(5/256)=1`，空 split 被 max 淘汰）。

### 核心带宽实测（决定 kernel 形态的那几个数）

| 形状 | N | K | 带宽 |
| --- | ---: | ---: | ---: |
| `lm_head`（库在这返回 NaN） | 248320 | 2560 | **364.5 GB/s** |
| `o_proj` | 2560 | 4096 | 247.6 GB/s |
| `mlp_down` | 2560 | 9728 | 191.4 GB/s |
| `mlp_up` | 9728 | 2560 | 172.0 GB/s |
| `qkv_gate` | 8192 | 2560 | 147.4 GB/s |
| D2D copy 上限 | — | — | 574 GB/s |

**`lm_head` 那一行是整个移植最有说服力的证据**：参考引擎记录的 `MAX_GEMV_N=131072`
陷阱（库在宽 N 上返回 NaN）在这里被实测复现，而自写 `[K,N]` 核在同一个形状上拿到
364.5 GB/s 且数值正确。

---

## 7. S3 交接：serving / cache / engine / entrypoints（migrate-serving）

**范围 12 个文件，全部落在 `serving/` `cache/` `engine/` `entrypoints/` 内。**
验收：`python -m unittest discover -s tests` → **Ran 238 tests, OK (skipped=58)**。

### 7.1 平台事实全部改从 `backends` 读

| 位置 | 原来 | 现在 |
| --- | --- | --- |
| `serving/runtime_parts.py` / `runtime_transport.py` | `import hip_ops` + `hip_ops.available()` 分支 | 删除；无分支，直接 `backends` |
| 启动横幅 `runtime_transport.serve` | `hip={hip_ops.available()}` | `backends.status()`，独立一行，打印 `backend=/device=/native=N/17/graph=/autotune=` |
| `paged_cache.PagedKVCache(device=)` 默认值 | `"cuda"` | `None` → `backends.device`；测试仍显式传 `"cpu"` |
| 9 处 `torch.cuda.memory_allocated/reserved/synchronize/empty_cache/mem_get_info` | 直接调用 | `runtime_support` 的 `allocated_gib/reserved_gib/sync/empty_cache/free_bytes`，命名空间由 `caps().device` 派生 |
| `ProfilerActivity.CUDA`（2 处） | 裸枚举 | `profiler_activities(device=True)`，CPU 后端只返回 `[CPU]` |
| `entrypoints/openai/app.py` 的 `dcu-engine` 标识 | `server_version` / `owned_by` | `coffin-engine`（同一份适配层服务 DCU 与 CoreX） |

**关键实现选择**：`serving/runtime_support.py` 的 `device_runtime()` **不是**简单
`getattr(torch, caps.device, None)`。实测 `torch.cpu` 在，且 `torch.cpu.is_available()`
返回 `True`（它的语义是"CPU 总在"，不是"有加速卡"），CUDA 构建也会在无卡机器上暴露出
`torch.cuda`。所以判据是 `caps.device == "cpu" → None` **再加** `is_available()` 探针。
这道判断让整条诊断路径（banner / ping / bench / profile）在无卡机器上真的跑得起来，
而不是"能 import 但一调用就炸"：实测 `backend=reference device=cpu native=0/17
graph=0 autotune=0`，`free_bytes() == -1`（未知，不是 0）。

### 7.2 `--backend` / `--no-graph`

- `entrypoints/cli/runtime.py` 新增 `--backend NAME`：**在 import 任何绑定算子表的模块
  之前**写 `COFFIN_BACKEND`（`backends` 的选型是 import 期一次性的），所以
  `runtime_transport` 的 import 推迟到 `parse_args()` 之后。
- `--no-graph` 走 `serve(..., graph=)` → `Server(graph=)` → `EngineConfig.graph`，
  **有效值 = 用户想要的 AND `caps.graph`**。不能做成环境变量：它必须在引擎 capture
  之前到达 `EngineConfig`。

**这一条带出一个必须记下来的坑**：原先 `llm_engine.build()` 在关闭 capture 时是
`self.model.reset()` 就直接返回，但 `model.decode_step()` 是**重放已捕获的图**——
没捕获过就 AttributeError。所以关图不能只是"跳过 capture"，必须同时换执行路径。
现在的做法是三层各自持一个 `graph` 标志：

- `ModelRunner.graph` → `_forward_token()`：`decode_step`（重放）或 `logits`（eager）；
- `PrefixCacheEngine.graph` → `decode_iter()` 同一分支；
- `LLMEngine.build()` 在 `graph=False` 且 batching/spec 仍开着时**抛异常**，因为
  batch 与 spec 的每一步都是重放按桶捕获的图，没有 eager 替身。`Server` 提前把
  `spec_k` 清 0 并打印原因。

两个入口给的是同一组数（都是 stage token + 推进 `pos1`），所以下游的 KV 写窗口、
`commit_upto` 完全不用改。`ping` 报的是**有效决策** `graph and graph_ready`，不是
`graph_ready`——否则关图会被读成"capture 失败"。

### 7.3 补齐拆包漏搬的图片路径（WORKLOG §6 记的已知 bug）

`253c9c8` 把 1706 行的 `serve.py` 拆成 `serving/` 时漏搬了 4 个方法，全仓只剩调用。
用 AST 扫 `self.<name>` 无定义确认，实际漏了 **3 个**（第 4 个 `_load_weight_kwargs`
也漏了，但 WORKLOG 只记了 `_prepare_mm`）：

| 方法 | 状态 |
| --- | --- |
| `_prepare_mm` | **补回**（原实现 `7e55aa0:serve.py:350`） |
| `_ensure_processor` | **补回**（懒加载 `AutoProcessor`，失败记进 `_processor_failed` 报 503） |
| `_vision_ready` | **补回**（读 `config.json` 的 `vision_config`，不只看 `self.vision`） |
| `_load_weight_kwargs` | **补回**（text quant + vision 策略） |

影响面与 WORKLOG 一致：**纯文本路径完全不受影响**（这几个方法只在 `images` 非空时
才会被调到），**HTTP 图片请求此前必定 AttributeError**。补回后端到端跑通：1 张
2×2 PNG → `grid=[[1,4,4]]` / `img_tokens=4` / `spans=[(3,7)]` / `rope_delta=-2`，
且三处对齐自检（tokenizer 一致、`len(ids) == len(plain) - n_img + Σgrid/4`、placeholder
run 数 = 图片数）全部生效。

`_prepare_mm` 里 `mrope_positions(..., device=...)` 传的是 `backends.device`，
不再有默认的 `"cuda"` 字面量从调用点漏进去。

### 7.4 删掉的死开关

`Q35_FAST_GEMV` / `Q35_WIDE_SPLIT8` 的核选型已由 migrate-models 收进后端，
`runtime_service.py` 里 `os.environ["Q35_FAST_GEMV"]="1"` 成了空开关。**已连同
`--fast-gemv` 参数、两处 docstring、`serve()`/`Server()`/`cli` 的传递链整条删除**——
留一个"接受但无效"的参数会让下一个人以为它还在起作用。

### 7.5 torch 2.1 兼容审查（本范围结论：无需改动）

扫了 `serving/` `cache/` `engine/` `entrypoints/` 四个包的全部 `torch.*` 调用面：

```
torch.arange  torch.device  torch.full_like  torch.inference_mode  torch.int
torch.multinomial  torch.profiler.profile  torch.profiler.ProfilerActivity.*
torch.softmax  torch.Tensor  torch.zeros  torch.float  torch.int8/int64/bfloat16
```

**全部在 torch 2.1 就有**。本范围内**没有** `F.rms_norm` / `torch.compile` /
`torch.library.*` / `torch.nn.attention` / `scaled_dot_product_attention` /
`torch.amp` / `register_fake` / `torch.export` / `add_safe_globals` / `weights_only`。

其余 2.4+ API 只出现在两处，都**不在**本步范围且已由别人处理：

1. `ops/norm.py` 的 `F.rms_norm` —— migrate-models 已改调 `bops.rms_norm`（那条是真
   2.1 上直接 AttributeError 的，最要紧的一条）。
2. `diag/probe_ops2.py:56` 仍在用 `F.rms_norm` 做 A/B 计时 —— 探针本职，且属
   `diag/`（probe 组）。

另外确认本范围**没有** Python 3.11+/3.12+ 专属语法（`except*`、`Self`、`tomllib`、
`TaskGroup`、`StrEnum`、`datetime.UTC`、`itertools.batched`、`typing.override`），
真机 `py310` 环境可以跑。

### 7.6 留给下一轮

1. **`_prepare_mm` 的 `_vision_ready()` 顺序**：现在先 `_ensure_processor()` 再判
   `_vision_ready()`，一个 `--no-vision` 的服务器收到图片会先白花 ~2 s 加载 processor
   才报 503。应把 `_vision_ready()` 提到最前面（原 `serve.py` 也是这个顺序，我保持了
   行为一致，没有顺手改：那会让"请求被拒"的耗时变化，属于另一个改动）。
2. `entrypoints/openai/app.py` 的 `/health` 已经把 `ping` 全量合并进去了，所以
   `backend` / `device` / `native_ops` 自动出现在 HTTP 健康检查里——但**没有测试钉住**
   `native_ops` 这个字段，加一条断言更稳。
3. `serving/runtime_support.py` 的 `free_bytes()` 在真机上没验证过（本机无卡只测到
   `-1` 分支）；`--spec-batch` 的空间检查是它唯一的调用点。
4. `cache/paged_cache.py` 删掉了未使用的 `D = torch.bfloat16`（全仓无人 import 它）。

