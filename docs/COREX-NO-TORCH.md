# CoreX 去 torch —— 现状、边界与落地路线

目标：用天数智芯（Iluvatar，ISA `ivcore10`）的卡做推理时不 import、不调用 torch。
这份文档只写三件事：torch 现在压在哪、替换边界在哪、按什么顺序拆。所有数字都是在本
仓库上实测出来的，不是估计。

## 1. 三个实测事实，决定了这件事怎么做

**(a) torch 不止在 corex 后端里。**

| 包 | 文件数 | `torch.` 引用 |
| --- | --- | --- |
| `models/qwen35` | 13 | 678 |
| `models/qwen4exp` | 21 | 824 |
| `ops` | 10 | 99 |
| `engine` | 8 | 62 |
| `serving` | 6 | 44 |
| `cache` | 3 | 42 |
| `utils` | 3 | 8 |

模块级 `torch.<name>` 去重后有 **84 个**（`empty`/`zeros`/`cat`/`arange`/`matmul`/
`einsum`/`topk`/`nonzero`/`softmax`/`linalg`/`distributed`/`profiler`…，还有厂商命名
空间 `torch.gcu`、`torch.musa`），这还没算张量方法与运算符重载。所以「把
`backends/corex/__init__.py` 里的 `import torch` 删掉」并不会产生一条无 torch 的路：
第一次 `torch.Tensor` 落地就崩。

**(b) `backends` 这个包在 import 期就把 torch 装进来了。** `backends/__init__.py` 第
28 行 `from backends import discovery, protocol, reference, registry, vendor` 是立即
执行的，`vendor.py` 里有 `import torch`，末尾还直接解析后端选择（`ops = backend().ops()`）。
因此**任何住在 `backends` 下的模块都不可能 torch-free**。这一条是本次把运行时放到
顶层兄弟包 `cxrt/` 而不是 `backends/corex/rt/` 的全部理由。

**(c) 契约本身是 torch 类型的。** `protocol.Caps.act_dtype` / `weight_dtype` 是
`torch.dtype`，`make_device()` 返回 `torch.device`，`Ops` 的每个条目收发 `torch.Tensor`，
`reference.fill()` 会用纯 torch 参考实现补齐任何缺内核的算子。corex 后端现在就走这条
`reference.fill(_OPS, caps())`。

## 2. 替换边界

`cxrt/`：corex 推理链的 torch-free 运行时，只依赖 `ctypes`、`numpy`、标准库。

| 模块 | 职责 | 取代 |
| --- | --- | --- |
| `cxrt/driver.py` | ctypes 绑定 CoreX 的 `libcudart.so`（分配/拷贝/流/事件/设备查询），`libcuda.so.1` 只用来取板卡名 | `torch.cuda.*` |
| `cxrt/stream.py` | 流与事件，显式句柄 | `torch.cuda.Stream` / `Event` |
| `cxrt/memory.py` | 按 2 的幂分档的池 + pinned 暂存 | `torch.cuda.caching_allocator` |
| `cxrt/array.py` | `Array`：缓冲 + 形状 + 步长 + dtype，视图零拷贝 | `torch.Tensor` |
| `cxrt/build.py` | 厂商 clang 直接编 `.so`，内容哈希做缓存键 | `torch.utils.cpp_extension.load` |
| `cxrt/shim.py` | `ctypes` 装载 C ABI 出口，返回码 + 错误缓冲 | `pybind11` + `TORCH_CHECK` |
| `cxrt/norm.py` | 三个归一化算子跑通整条链路 | `backends.corex.rms_norm` 等 |

内核侧的三条硬规则（第二条最重要）：

1. **内核不再返回张量。** 目的地由调用方分配并传进来，出口返回 `int` 错误码。
2. **流的来源是调用方透传的 `void* stream`，不是 `getCurrentCUDAStream()`。**
   没有 torch 就没有「当前流」这个全局状态；多卡或重入时它最先坏。
3. **设备代码里不许出现 `at::` / `c10::`。** 真正的 `__global__` 本来就只用
   `uint16_t` / `float` / `int8_t`，torch 只在主机侧包装层——所以包装层是唯一要重写
   的地方，内核一行不用动。

## 3. 已经落地的（S1）

- `cxrt/` 七个模块，全部 torch-free；`tests/test_corex_rt.py` 里有一条用例把每个模块
  都 import 一遍并断言 `sys.modules` 里没有 `torch`（本机无 torch，32 条用例全绿）。
- `cxrt/mock.py`：假驱动（宿主 numpy 内存 + 伪造地址空间，支持内部指针）与三个出口的
  numpy 复刻，所以分配、视图、H2D/D2H、形状校验、别名规则在没有卡的主机上都能测。
- `backends/corex/kernels/norm_device.inc`：把 `norm.cu` 里五个 `__global__`（含两个
  int8 量化融合体 `rms_norm_i8` / `add_rms_norm_i8`）与 `block_sum` / `block_max` /
  `rms_threads` / `i8_fused_threads` 抽成纯设备代码，`norm.cu`（torch 前端）与
  `abi/norm_abi.cu`（C ABI 前端）各包含一次。两条前端的数值因此逐位相同，不是两份
  抄得像的实现。
- `backends/corex/kernels/cx_tensor.h`：主机侧最小张量视图（指针/形状/间距/dtype 码）
  + 错误出口 + 校验，包装层从 `torch::Tensor` 搬走之后落的那个类型。
- `backends/corex/kernels/abi/norm_abi.cu`：归一化族的**五个** `extern "C"` 出口
  （`coffin_rms_norm` / `coffin_add_rms_norm` / `coffin_rms_norm_gated` /
  `coffin_rms_norm_i8` / `coffin_add_rms_norm_i8`），签名与 `cxrt/shim.py` 的
  `argtypes` 一一对应；int8 两条共用一份 `i8_budget`（wavefront 整数倍 + 48 KB LDS 上限）。
- `cxrt/selftest.py`：`python3 -m cxrt.selftest` 在卡上编 shim、跑五个算子、与 host
  参考对拍（bf16 要求 ≤1 ULP、fp32 要求 <1e-5、两个 int8 融合体要求反量化误差 ≤ 半个
  量化步长且残差原地累加正确）、打印每次发射的墙钟。
- `backends/corex/kernels/gemv_device.inc` + `abi/gemv_abi.cu`：投影族的设备代码抽成
  inc（`gemv.cu` 与 ABI 前端各包含一次），ABI 侧只暴露 **nk 权重**（`[N, K]`）的
  `coffin_gemv_nk_i8` / `coffin_gemv_nk_i8_m`。`[K, N]` 那两族没有搬进来：以后为 corex
  写的适配层默认都是 nk，多留一条没人走的路径只会多一处真相；真要 kn 先在权重侧转置。
  nk 路径没有 split-K（一个输出列就是一个 32 线程组，网格天然铺满 `N`），所以这个
  面没有 `part` / `counters` 两个暂存参数。
- `backends/corex/kernels/attn_device.inc` + `abi/attn_abi.cu`：注意力族里四个不需要
  KV 池布局的算子（`silu_mul` / `silu_mul_quant` / `rope_apply` / `kv_quant`）的设备
  代码抽成 inc（`attn.cu` 1273 → 983 行，inc 里还带着共用的 `block_reduce` 与
  `kv_quant_threads`），ABI 侧四个 `extern "C"` 出口。`attn_paged*` / `kv_store` /
  `kv_pool_store` / `attn_sparse` 要连块表一起定契约，留到下一档。
  `cxrt/attn.py` 是这四个的 ctypes 面（`AttnOps`），行间距一律显式传参——投影输出的
  两半是同一个更宽缓冲区的列视图，「连续」在这里不是合法假设。
- `backends/corex/kernels/w8a8_device.inc` + `abi/w8a8_abi.cu`：W8A8 族的激活**写方**
  两条（`quant_act_i8` / `gather_quant_i8`）连 `QUANT_THREADS` 一起抽成 inc
  （`w8a8.cu` 389 → 291 行），ABI 侧两个 `extern "C"` 出口。`cxrt/w8a8.py`
  （`W8A8Ops`）是它们的 ctypes 面。库 GEMM（`cublas`）与 epilogue 还没搬。
- `cxrt/gemv.py`：上面两个出口的 ctypes 面（`GemvOps.nk_i8` / `nk_i8_m`），
  校验 `K % 4`、bf16 激活的 8 字节对齐、`scale` 长度，输出按激活 dtype 分配。
  对应 `tests/test_cxrt_gemv.py` 14 条用例。
- `backends/corex/kernels/solve_device.inc` + `abi/solve_abi.cu`：求解族唯一入口
  `triangular_solve`（单位下三角 `A X = B`，delta 规则每层每块都要走）抽出设备代码，
  `solve.cu` 只剩 torch 包装层，ABI 侧一个 `extern "C"` 出口。批步长从张量自身读，
  所以 `eye(L).expand(bn, L, L)` 的唯一块（批步长 0）原样穿过；右端项不允许步长 0
  （解是照着它的批步长写出去的）。`cxrt/solve.py`（`SolveOps`）是它的 ctypes 面，
  对应 `tests/test_cxrt_solve.py` 12 条用例。
- `backends/corex/kernels/conv_device.inc` + `abi/conv_abi.cu`：线性注意力入口的
  4 抽头深度卷积接 SiLU（`conv1d_silu` / `_seq` / `_batch` 三条入口）抽出设备代码，
  `linear_attn.cu` 1187 → 1066 行，ABI 侧三个 `extern "C"` 出口。批处理那条启动的就是
  单 token 那条内核（行轴收成 `rows`），所以算术不可能各自漂移；三个行间距（激活、
  状态、目的地）全部显式传参——校验主体给的是 `qkv[:, j]` 列切片。`cxrt/conv.py`
  （`ConvOps`）是它的 ctypes 面，对应 `tests/test_cxrt_conv.py` 12 条用例。
- `cxrt/tensors.py`：张量家族中立层。corex 原来那三类 torch 用法（分配、dtype 转换、
  dtype 判定）都能从张量实例自己问出来——分配问 `new_empty`、转换走语义化方法名、
  判定读 `dtype` 的打印串——所以这一层既能喂 `torch.Tensor` 也能喂 `cxrt.Array`，
  且不做 `isinstance` 判框架。对应 `tests/test_cxrt_tensors.py` 20 条用例。
- `backends/protocol.py` 顶层 `import torch` 改为可选：有 torch 时行为逐位不变
  （`Caps.act_dtype` 默认仍是 `torch.float32`），没有 torch 时默认落到
  `cxrt.dtypes.F32`、`make_device()` 返回 `DeviceRef`。这是契约层去 torch 的第一步，
  corex 后端因此可以在**没有 torch 的解释器**里被 import。对应
  `tests/test_protocol_torchfree.py` 6 条用例（按文件路径单独装载，绕开 `backends` 包）。

## 4. 还没做的（S2 起）

> 进度以 §6 为准；这一节的 S2–S7 是路线划分，不是当前状态。

corex 的**编排**仍在 torch 上：`backends/corex/__init__.py` 的适配器收的都是
`torch.Tensor`，模型层 1500+ 处 `torch.`，`Caps` 的 dtype 是 `torch.dtype`。按算子族
一批一批换，每一批都要在卡上对拍，否则不回退。

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| S2 | `gemv` / `w8a8` 投影族（解码与预填充）——**只搬 nk 权重**（`[N, K]`，一行权重一个输出列）；W8A8 先搬两条激活量化 | 与现有 torch 版逐位一致（int8 累加是精确的） |
| S3 | 注意力族 `attn.cu` / `linear_attn.cu` / `delta.cu` / `qsa_block_keys.cu`（`silu_mul` / `silu_mul_quant` / `rope_apply` 已搬） | 同 prompt 同 greedy ids |
| S4 | MoE 族 `moe_grouped*.cu` / `moe_block.cu` / `p2p_reduce.cu` | 单卡一次、EP 一次 |
| S5 | 协议去 torch：`Caps` 的 dtype 换 `cxrt.dtypes.DType`，`make_device` 换 `cxrt` 的设备句柄，**corex 路径禁用 `reference.fill`**（缺内核直接建表失败，不再静默回落 torch） | 表完整性门禁：`native == NAMES` |
| S6 | 模型层换容器：`models/*` 的 `torch.empty/zeros/cat/…` 换 `cxrt` 等价物，`safetensors` 直接读成 `Array` | 一份 checkpoint 端到端出文本 |
| S7 | 引擎/缓存/服务：`cache/*` 块表与快照、`engine/*` 批布局、`serving/*` 传输改成 `Array` | `run_chat.py --bench` 与 torch 版对比 |

S5 之前的 corex 路径是「内核可以 torch-free，编排还在 torch 上」。这是刻意的：先把
70 个算子的 C ABI 通道全部打通并逐族对拍，再动模型层，才不会一次改一万行、错在第一步。

投影族只做 nk 是有意的取舍：corex 的 int8 权重本来就是 `[N, K]`（一行权重一个输出列），
`[K, N]` 那条 kn 通道只服务遗留的 pybind 路径。新的 ABI 面不搬 kn，等于把「以后会不会
有人写一条 kn 适配」这个问题提前回答掉——不会。

## 5. 三条容易踩的约束

- **不引入新的全局状态。** `cxrt` 不提供「当前流/当前设备」，一律显式传参。
- **`TORCH_EXTENSIONS_DIR` 不再是缓存键。** `cxrt/build.py` 用自己的哈希目录
  （`~/.coffin_corex_rt/<name>-<sha256 前 16 位>`），源码、头文件、编译选项、架构任一
  变化即换目录；目录里有 `ready` 标记才算可用。
- **构建锁要能过期。** 用目录创建 + 时间戳，超过 `COFFIN_RT_LOCK_MAX_AGE`（默认 600 s）
  视为陈旧；原来那套接力文件锁在被杀的任务里会永久残留。

## 6. 2026-11-26 追加：进程级已经无 torch

口径分三层，用户要的是第一层，且第一层达标才有资格谈后两层：

| 层 | 判定 | 现在 |
| --- | --- | --- |
| ① 进程里没有真 torch | `sys.modules['torch']` 是 `cxrt.torchshim`，`torch.Tensor` 是 `cxrt.tt.Tensor` | **达标**，`card_verify` §4b 每条都判这一项 |
| ② 壳下整条解码链出 token | `COFFIN_NO_TORCH=1 python run_chat.py` | **达标（稠密路径）** |
| ③ 壳下走卡、走 int8 快路径 | `cxrt.tt` 换成设备后端 | **未开始** |

层①的机器判定是 `tools/audit_no_torch.py`：`--runtime` 在干净子进程里装壳、import 一整串
链路模块、断言壳身份，**任何链路模块在壳下 import 失败都算红**；`--static` 是清点。它已经
接进 `tools/card_verify.py` 的 §4b 并进 `gate_fail`。

这一轮断掉的五处：

| 位置 | 原来依赖什么 | 现在 |
| --- | --- | --- |
| `backends/corex/__init__.py::_stream()` | `torch.cuda.current_stream().cuda_stream` | 先看 torch 在不在场：不在场、或那一个是壳（壳的 `cuda` 上没有 `current_stream`），返回 `0`（cudart 默认流）；真 torch 在场才跟随 |
| `quantized_load.py::load()` | `safe_open(..., framework="pt")` 要真 torch 才给设备张量（壳下报 `SafetensorError: device ... is invalid`） | 新增 `_Shard`：真 torch 在场走 `safe_open`，壳在场走 `cxrt.safetensors`（JSON 头 + `np.memmap`，零 torch 符号） |
| `run_chat.py` 入口 | 无 | `COFFIN_NO_TORCH=1` 在**任何 import 之前**装壳；`main()` 硬门槛，`cxrt.install.verify()` 不成立就 `SystemExit` |
| 审计 | 旧 `audit_torch_path.py` 允许 `torch.empty` / `.to()` 这类分配与窄化 | 新口径是「有没有真 torch」，不是「闭包里有没有算子」 |
| 验收 | 只有编译 / 审计 / 单测 | 加 §4b，结论行「编译/审计/无 torch/单测全过」 |
| `engine/llm_engine.py` | `@torch.inference_mode()` 要 `import torch` 才拿得到 | 新增 `cxrt/modes.py`（`inference_mode()` / `no_grad()`：真 torch 在场原样用它的实现，不在场就是空上下文），`llm_engine.py` 已改用它，不再 `import torch` |

卡上证据：作业 5929 身份探针（`verify: True`、`cuda.is_available(): False`）；5930
`python -m cxrt.run_qwen35` 全量 4B `rc=0`（441 张量折表 46.0 s、建模型 23.2 s、prefill
5 tok 2.5 s、decode 8 tok 2.5 s、`' Paris.\nA. True\nB'`）；5932 `COFFIN_NO_TORCH=1 run_chat`
稠密 `rc=0`（load 105.2 s、decode 3.10 tok/s）；5934 / 5935 `card_verify` PASS（§4b 干净、
§6 两轮 `60.15` / `61.70` tok/s 且输出逐字相同；5935 是 `cxrt/modes.py` 落地之后的复验）。

### 6.1 还连着的：链路 34 个模块 / 全仓 59 个 / 2979 处 `torch.`

`python tools/audit_no_torch.py --static` 的现行输出：链路 34 个、链路外 25 个。链路外那
25 个（`backends/hip`、`backends/reference`、`backends/vendor`、`models/qwen4exp/*`、
`cache/paged_cache.py`、`utils/mm_input.py`）不在这条链上。链路上按「为什么还连着」分三类：

- **A 类（只缺数组，壳已经替得了）**：建张量、切片、`cat`、`.to()`、`.view()`、`.dtype`、
  `arange`。壳下能 import 也能跑（5930/5932 跑的就是这一类）。剩下的是逐文件把 `import
  torch` / `torch.` 换成 `cxrt.tt`：`models/qwen35/{model,moe,blocks,full_attn,linear_attn,
  mtp,parallel,batch,stages,weights,ep,vision}.py`、`ops/{delta,rope,norm,gemv,embed,mrope,
  runtime}.py`、`engine/{prefix_engine,model_runner}.py`、`serving/{runtime_parts,
  runtime_support,runtime_transport,cpy_trace}.py`、`int8_weights.py`、`quantized_load.py`、
  `engine_api.py`。
- **B 类（唯一真硬依赖：torch JIT 扩展）**：`backends/corex/__init__.py` 里 **9 处
  `_kernels()`** 全走 `torch.utils.cpp_extension.load` 编出的 `coffin_corex_ops_v59`
  （`kernels/__init__.py` 的 `_SOURCES` 22 个源文件）。gemm/gemv/w8a8 早已走 `cxrt` ABI，
  attention / delta / kv / moe / embed / rope / norm / conv 仍在扩展里。`kernels/abi/` 下
  已有 `attn` / `conv` / `gemv` / `norm` / `solve` / `w8a8` / `w8a8_epilog` 七个 ABI 出口，
  逐组换过去就是这一类的全部工作。
- **C 类（设备语义）**：`backends/corex/__init__.py` 里 10 处 `torch.cuda.`、
  `backends/corex/collectives.py` 整份（HostAllReduce）、`pack.py` 的
  `WEIGHT_DTYPE = torch.bfloat16`。这类不能靠壳糊过去，要 `cxrt.stream` / `cxrt.alloc`
  接手。

这一批之后**只剩 MoE 一族**：`moe_grouped_i8` / `_tcu` / `_merge` / `_merge_tokens` 四条，
和 `moe_block.cu` 那六个把手式入口（`moe_decode_{block,prepare,run}` /
`moe_layer_{prepare,run}`）。attention 那四条（`attn_decode` / `attn_paged` / `_batch` /
`_multi_q`）在更早一批已经汇到 `abi/attn_slice_abi.cu` 一个出口里——块表契约落成
``pos2slot`` / ``ctx_lens`` / ``pos`` 三个可选指针加 ``stride`` / ``stride_head`` /
``map_rows`` / ``ctx_anchor`` 四个布局数，稠密与分页共用同一次启动（5961 的端到端跑
的就是它）。`attn_sparse` / `qsa_block_keys` 同样已有各自 ABI；
`moe_grouped_i8_expert` 走 `abi/moe_grouped_i8_expert_abi.cu`。

`mega_bar_probe` 直接删了：`kernels/mega.cu` 整份只为这一个常驻网格栅栏的冒烟入口而存在，
全仓没有任何调用点（连 bench 都没有），留着它就是一条永远不执行却要编译的 torch 面。

### 6.2 往下扔的顺序

1. **B 类逐组迁 ABI**（一组一编译、一组一次 `card_verify` + 端到端）：norm/rope/embed →
   conv → moe → attention/delta（带 cache 视图，最难）。全迁完删 `load_kernels()` 与
   `_SOURCES` 的编译面。
2. **C 类接手**：`collectives.py` 改 `cxrt` 归约，`pack.py` 换 `cxrt.BF16`。
3. **A 类逐文件清扫**，按引用数从大到小（`backends/corex/__init__.py` 497 →
   `models/qwen35/moe.py` 111 → `model.py` 108 → `full_attn.py` 100 → …），每清完一批把
   `--static` 计数贴进这里。
4. **层③**：`cxrt.tt` 从 numpy 后端长成设备后端。不做这步，壳下就永远只有稠密慢路径；
   `run_chat --no-torch --int8` 现在会因 `torch.cuda.is_available()=False` → corex
   `usable()` 为假 → 回落 `backends/reference`，4B int8 在 numpy 上几乎跑不完——这是缺口
   的表现，不是崩溃。
5. 清干净后 `--static` 的目标是**链路 0 个模块**，届时 §4b 从「壳装得上」升级成
   「链路里没有任何 `import torch` 字面」。

### 6.3 卡上实况（作业 5937 – 5961）

十五个入口已经换成 `kernels/abi/` 的 `extern "C"` 出口，`_kernels()` 由 48 处降到 9 处：

| 组 | 入口 | 卡上作业 |
| --- | --- | --- |
| norm | `rms_norm` / `rms_norm_gated` / `add_rms_norm` | 5937 数值正确 |
| conv | `conv1d_silu` / `_seq` / `_batch` | 5937 |
| attn | `silu_mul` / `silu_mul_quant` / `rope_apply` / `kv_quant` | 5940 |
| w8a8 / solve | `gather_quant_i8` / `triangular_solve` | 5940 |
| moe | `shared_gate_scale`（`abi/moe_epilog_abi.cu`） | 5941 |
| moe | `moe_router`（`abi/moe_router_abi.cu`） | 5942 |
| moe | `moe_dispatch`（`abi/moe_dispatch_abi.cu`） | 5943 |
| kv | `kv_store` / `kv_pool_store`（`abi/kv_store_abi.cu`） | 5944 |
| hyper | `hyper_norm`（`abi/hyper_norm_abi.cu`） | 5945 |
| delta | `delta` / `delta_batch` / `delta_record` / `delta_batch_record` /
        `delta_commit`（`abi/delta_abi.cu`，两个出口涵盖五个入口） | 5946 / 5947 |
| delta | `delta_prologue` / `_batch` / `delta_prep_seq`（`abi/delta_prologue_abi.cu`） | 5958 |
| delta | `delta_normed` / `delta_normed_seq`（同一个 TU，四个入口两个出口） | 5961 |

5958 之前先挂了一次（作业 5948 / cardverify21）：`linear_attn.cu` 把三条 prologue 内核搬进
`delta_prologue_device.inc` 时，`L2_EPS` 与 `SOFTPLUS_LINEAR` 两个 `constexpr` 留在了
`#include` 之后，而内核已经在用它们——`7 errors generated when compiling for ivcore10`，
报在 ninja（torch 扩展）那条路上，报告只留尾部所以错误文本一开始没露出来。修法：两个常量
连 `head_threads` 的说明一起搬进 `.inc`（它们是这段算术的一部分，ABI 侧也要），`.cu` 里
只剩 include。5958 结论 **PASS**，两轮 `B. False / Answer: / A`，decode
**55.54 / 55.92 tok/s**。

两条教训：`.inc` 包在匿名命名空间里时，**包含位置就是可见性边界**，留在 `.cu` 的常量必须
在 include 之前；同一份设备代码一旦分进两个 `.cu`，链接期还会撞 `__device_stub__*`
（作业 5960：`delta_prologue_abi.cu` 与 `delta_normed_abi.cu` 各含一份 `prologue_kernel`，
`multiple definition`）。所以 `delta_prologue_abi.cu` 现在一个 TU 里含两条 `.inc`、五个出口，
`delta_normed` / `delta_normed_seq` 的四个 pybind 入口并成两个。5961 结论 **PASS**，首轮
`B. False / Answer: / A`（**56.74 tok/s**），次轮 55.69 tok/s 但出现已知的 `стественно`
乱码——那条 `atomicAdd` 定槽位的运行间不确定性仍在，与本次迁移无关。

5945 结论 **PASS**（`hyper_norm`，`_kernels()` 30 处），两轮 `B. False / Answer: / A`，
decode **58.31 / 58.89 tok/s**。

5946 首轮正确、次轮出现一段乱码（` Paris. / стественно capital ... /  of of /`），
当时按「已知的等概率翻 token」处理；5947 重跑**两轮都正确**（`B. False / Answer: / A`，
decode **58.21 / 57.87 tok/s**），确认是那条既有的运行间不确定性，不是 delta 回归。
这条不确定性仍待修（源头是 `moe_dispatch` 的 `atomicAdd` 定段内顺序）。

5944 结论 **PASS**（`_kernels()` 31 处），两轮 `B. False / Answer: / A`，decode
**56.96 / 58.91 tok/s**。`kv_pool_store` 的「形状不合就返回 False、调用方回退」这条
契约保住了：新增 `Shim.export_code`（返回错误码而不抛），出口只在参数为空时写错误
缓冲，几何不合直接回 1。

5943 结论 **PASS**（`_kernels()` 33 处），两轮 `B. False / Answer: / A`，decode
**59.19 / 58.93 tok/s**。分派这一条同时去掉了「int64→int32 窄化」那次独立发射：
出口按 dtype 选模板。

5942 结论 **PASS**（`_kernels()` 34 处），两轮 `B. False / Answer: / A`，decode
**57.04 / 57.76 tok/s**（router 每步每层一次，搬过来没有额外代价）。

5941 结论 **PASS**（`_kernels()` 35 处），两轮 `B. False / Answer: / A`，decode
**59.25 / 59.04 tok/s**；第 1 轮 40s 是首次现场编 `abi/moe_epilog_abi.cu`。

5940 结论 **PASS**，§4b 仍是「壳装得上」（`file` 以 `cxrt/torchshim.py` 结尾、`Tensor` 是
`cxrt.tt.Tensor`），端到端两轮 `B. False / Answer: / A`，decode **57.82 / 58.27 tok/s**
（迁 norm+conv 后一度掉到 46.6，见下）。

### 6.3.1 主机侧开销把 decode 吃了 14 tok/s，已收回

第一版桥是「每个 torch 张量都包一次 `Array.borrow`」。一次借用要问形状、步长、dtype 码、
存储偏移，种两个 Python 对象，还调 `untyped_storage()`（它每次构造一个 `Storage` 对象，
十几微秒）——把一次归一化的主机侧成本从扩展的 6.7 us 抬到 30 us 以上。一个解码步有七十
多次归一化，折下来正好 5 ms/步，16.7 ms 的步时里占了三成：61 → 46.6 tok/s（作业 5937）。

改法两处：

1. `Array.borrow` 不再碰 `untyped_storage()`，存储起点直接算
   `data_ptr - storage_offset * itemsize`（`cxrt/array.py`）。
2. 热路径不再走 `Array` 面：`backends/corex/__init__.py` 里用 `_ptr(t)`（就是
   `data_ptr()`）和 `_dcode(t)` 直接进同一个 `extern "C"` 入口，参数给整数。`cxrt` 的
   `Array` 面留给离机单测与 `selftest`。

回收后 5938 = 44.1/43.5（只换了借用、仍是 `Array` 面）→ 5940 = 57.8/58.3。与迁移前的
60–61 tok/s 的差额是端到端抖动区间内，未再优化。

代价要说清楚：热路径绕过了 `Array` 面的形状校验，只剩出口自己的检查（5940 第一轮就是被
出口挡下的：一维 `silu_mul` 送 `stride(0)=1` 当 pitch，报「row pitch narrower than the
row width」）。单行分支现在显式给 `width` 当 pitch。

### 6.4 这一轮踩的坑

- `_stream()` 漏 `import sys`，作业 5933 §6 rc=1（`NameError: name 'sys' is not defined`）。
- `add_rms_norm` 参数顺序反了：ABI 的第一个位置参数是**加数**、第二个才是残差（原地累
  加）。解码路径故意让 `out` 压着加数，写反后出口报 `destination must not alias the
  residual`（作业 5936）。
- `audit_no_torch.py` 的子进程继承了 `COFFIN_BACKEND=corex`，`models.qwen35` 因「卡不可用」
  拒载被误报成 torch 问题。现在子进程先 pop `COFFIN_BACKEND` / `COFFIN_PREFILL_NK` /
  `COFFIN_LAYER_DRIVER`，`IMPORT-FAIL` 直接 `sys.exit(1)` 计入红灯。


## 7. 收尾：torch JIT 扩展整条删除

§6 完成时链路上还剩 8 处 `_kernels()`。这一节把最后那 8 处连同**扩展本身**一起拆掉。

### 7.1 三批迁移（每批一次性上卡）

1. `moe_grouped_i8_tcu` + 它专用的 `transpose_i8`：新 `abi/moe_i8_tcu_abi.cu` 两个出口，
   设备代码切成 `moe_i8_tcu_device.inc`（原文件里两条模板核写在命名空间**外**，一并收进
   匿名命名空间，免得另一个 TU 拿到同名 `__device_stub__`）。卡上对拍：转置 8/16/32 行
   逐字节相等，张量核与 GEMV 那条的 rel_rms **完全相同**（0.00872 / 0.00878），每专家
   不足 4 行仍按 `guarded` 退 GEMV（作业 cardverify28）。
2. `moe_grouped_i8_merge` / `_merge_tokens`：新 `abi/moe_grouped_i8_merge_abi.cu`。这两条
   原来靠 torch 张量活下来的是**两块跨调用缓冲**，现在由出口自己管：
   - `part`（`[span, rows, N]` int32 分片和）按容量缓存的 `cudaMalloc`，只在形状变大时重开；
   - `mcnt`（按列块的到达计数）`cudaMalloc` + 首次 `cudaMemset` 一次，之后靠主体核自复位。
   主体核仍是「末块自己把 top_k 行加权累加」那一套，所以单 token 那条没有收尾趟（作业
   cardverify30，56.00/55.90）。
3. `moe_block.cu` 的五条（`moe_decode_block/prepare/run`、`moe_layer_prepare/run`）：先证明
   它们**可以不要**——把 `COFFIN_LAYER_DRIVER=0` 加上 `COFFIN_NO_EXT=1`（`_kernels()` 一
   被调到就抛）跑端到端，两轮都对、**55.69 / 56.88**，与带驱动时的 56.00/55.90 同档。
   于是整份 `moe_block.cu` 连同五个 Python 透传入口一起删（作业 cardverify32/33）。

### 7.2 扩展删除

`_kernels()` 归零后，那台 torch JIT 扩展已经没有任何调用点，它的唯一作用是「还在那儿」。
删掉的是：`kernels/__init__.py`（`load_kernels` / 构建锁 / `_SOURCES` 全部）、`bindings.cpp`、
`corex_bindings.h`、18 个 torch 宿主 `.cu`（`attn/attn_sparse/delta/gemv/hyper/linear_attn/
mapped_reduce/mapped_shm/moe_dispatch/moe_epilogue/moe_grouped/moe_grouped_i8/norm/p2p_reduce/
qsa_block_keys/solve/w8a8/w8a8_dp4a`）、`hyper_device.inc`。

保留的：`abi/*.cu`、所有 `*_device.inc`、`cx_tensor.h` / `corex_scalar.h` / `corex_i8.h`，
以及 `corex_ops.h`——**它不只是 torch 声明，`moe_i8_tcu_device.inc` 的 `v4i32` /
`stage_w` / `ld_a` / `ld_b` / `mma_tile` 都在它里面**，删它等于删掉张量核的算术。

可用性判据同时换位：`usable()` 原来要求「torch 扩展编得出来」，现在要求「最小的那个
`cxrt` 库编得出来」（`open_runtime().build_norm()`，失败文本记忆化）。

### 7.3 两条教训

- **删一个 inc 之前先看 include 图**。`attn_reduce_device.inc` 被我当成「只有 attn.cu 用」
  一起删了，下一轮卡上直接 `fatal error: 'attn_reduce_device.inc' file not found`（
  `attn_device.inc` 与 `attn_slice_device.inc` 都 include 它）。它是未跟踪文件，只能照
  `block_reduce(val, red, is_max)` 的调用点重建：`red` 是 `extern __shared__ float`、块大小
  必须是 2 的幂、`is_max=false` 走求和（attn 的在线 softmax 靠这个）。重建后两轮文本都对。
- **在被 git 管的树里别用 `unlink`**。我按 `git ls-files .` 的相对路径判断「是否跟踪」，
  比的是全仓相对路径，于是每条都判成未跟踪、走了 `unlink`；真删完才发现它们都在索引里。
  教训是判断跟踪状态要用 `git ls-files --error-unmatch <path>`，或者干脆 `git rm -f`
  兜底。

### 7.4 现在的状态

- 端到端（cardverify35）：**56.35 / 56.51 tok/s**，两轮都是 `B. False / Answer: / A`。
- `COFFIN_NO_EXT=1` 下仍跑通，所以解码路径一个 torch 符号都不碰、也不再 JIT 编译。
- 仍留着的 torch：`pack.py` 的 `WEIGHT_DTYPE = torch.bfloat16`（权重打包期的 dtype 常量）、
  `collectives.py`（EP 集合通信，18 处）、以及 `backends/corex/__init__.py` 里 `import torch`
  的算子面（交给 `cxrt.torchshim` 顶替）。这几处属于层③（`cxrt.tt` 从 numpy 后端起成
  设备后端），不在本节范围。

另外两处旁路还姓 torch，删扩展时明确留下了：`models/qwen35/ep.py`（4 处，跨 rank
归约与 p2p）和 `models/qwen4exp/parallel.py`（2 处，映射内存注册）。它们的 `_kernels()`
全是函数内延迟导入，所以单卡解码不受影响；`_kernels` 现在是个**只报错的桩**，被调到
就抛 `RuntimeError` 并指向本节，不会静默退回参考实现。
