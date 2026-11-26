# CoreX 无 torch 跑两个 w8a8 模型：缺口矩阵

目标（用户原话）：**一直做，直到 Qwen3.5 4B 与 Occamy 1.0（均为 w8a8）都能跑起来**。
本文件把「都能跑起来」拆成可勾选的条目，并把当前证据写清楚——哪些已经有 C ABI 出口，
哪些还只有 torch 路径。数字全部由 `backends/protocol.py` 的 `Ops.NAMES` 与仓库源码的
AST 扫描得出，不是估计。

## 1. 两个目标如何启动

| 模型 | 架构族 | checkpoint | 启动 |
| --- | --- | --- | --- |
| Qwen3.5-4B | `models/qwen35`（混合：全注意力 + 线性注意力 + 稠密 MLP） | `Qwen3.5-4B-int8-nk`，`--int8` 认它 | `python3 run_chat.py --model $HOME/models/Qwen3.5-4B-int8-nk --int8` |
| Occamy 1.0 | `models/qwen4exp`（MoE + hyper 连接 + delta 网络） | `occamy-1.0-W8A8-nk`（`tools/quant_disk.py` 产出） | `bash tools/launch_occamy_w8a8.sh` |

验收不是「import 不报错」，而是：同一条 prompt、同 `--max-new`、greedy 解码，输出与
torch 版**逐 token 相同**（在近等概率处允许翻转 argmax，此时以逐张量 rel_rms 与逐算子
对拍为准，见 §11），且 `sys.modules` 里没有 `torch`。

## 2. 契约面：51 个算子，两个模型实际用到 33 个

模型与 `ops/` 包装层经 `from backends import ops as bops` 调用后端，所以缺口要按这个并集
算，而不是按 78 个 pybind 入口算。

**已有 C ABI 出口：7 个族、17 个 `extern "C"`**（都在无 torch 进程里可调用，且已在卡上
编译过，见 §5）

| 族 | 出口 | 设备代码 | Python 面 |
| --- | --- | --- | --- |
| norm | `coffin_rms_norm`、`_add_`、`_gated`、`rms_norm_i8`、`add_rms_norm_i8` | `norm_device.inc` | `cxrt/norm.py` |
| gemv | `coffin_gemv_nk_i8`、`_m` | `gemv_device.inc` | `cxrt/gemv.py` |
| attn | `coffin_silu_mul`、`silu_mul_quant`、`rope_apply`、`kv_quant` | `attn_device.inc` | `cxrt/attn.py` |
| w8a8 | `coffin_quant_act_i8`、`gather_quant_i8` | `w8a8_device.inc` | `cxrt/w8a8.py` |
| solve | `coffin_triangular_solve` | `solve_device.inc` | `cxrt/solve.py` |
| conv | `coffin_conv1d_silu`、`_seq`、`_batch` | `conv_device.inc` | `cxrt/conv.py` |
| hyper | （`hyper_device.inc` 已写，出口未接） | `hyper_device.inc` | — |


**还缺 23 个**，按「一次能搬完一族」分组：

| 组 | 算子 | 设备代码所在 | 为什么这一档难 |
| --- | --- | --- | --- |
| A 投影 | `linear`、`linear_m`、`linear_m_verify`、`linear_prefill` | `gemv.cu`、`w8a8.cu`、`w8a8_tcu.cu`、`w8a8_dp4a.cu` | 走库 GEMM（`cublas`）与 DP4A/TCU 两条硬件路径；契约要覆盖 A/B/C 三块缓冲与 `LIB_MAX_N` 切块，不能只搬一个内核 |
| B KV 布局 | `attn_decode`、`attn_paged`、`attn_paged_batch`、`attn_paged_multi_q` | `attn.cu` | 要连块表（`pos2slot`/`stride_slot`/`stride_head`）与 int8 KV 布局一起定契约，`attn_slice`/`attn_combine` 也得跟着 |
| C qwen35 线性注意力 | `conv1d_silu`、`conv1d_silu_batch`、`conv1d_silu_seq`、`delta`、`delta_record`、`delta_batch`、`delta_batch_record`、`delta_commit`、`delta_prologue`、`delta_prologue_batch`、`delta_normed_seq`、`delta_prep_seq` | `linear_attn.cu`、`delta.cu` | 12 个入口，结构是「准备 → 记录 → 提交」三段状态机，契约里带状态缓冲（record/commit 成对） |
| D Occamy MoE | `moe_router`、`moe_grouped`、`moe_grouped_i8`、`moe_grouped_i8_tcu` | `moe_grouped.cu`、`moe_grouped_i8.cu`、`moe_block.cu` | 分组偏移由路由结果给出，契约要收 `group_offsets`；预填充走库 GEMM |
| E hyper | `hyper_norm`、`hyper_down`、`hyper_up` | `hyper.cu` | 三个入口，独立文件，最接近已搬的那一档 |
| F solve | ~~`triangular_solve`~~ **已搬**（`solve_device.inc` + `abi/solve_abi.cu` + `cxrt/solve.py`） | `solve.cu` | 只读严格下三角、对角线按 1、批步长可为 0——三条都是 delta 的语义，不是简化 |

计数：A 4、B 4、C 9、D 3、E 3 = 23（F 的 1 个、C 的 3 个已搬）。

## 3. 非算子层的缺口（搬完 27 个也不算能跑）

1. `backends/corex/__init__.py`：148 行非注解 `torch.*`（分配 50、dtype 常量与比较 103、
   4 处库 GEMM 前后的 `torch.matmul`、若干视图与设备查询）。它是**适配层**，27 个 ABI 入口
   搬完之后，这里才是把它们串起来的地方。
2. `models/qwen35`（13 文件 / 678 处 `torch.`，其中 714 处 `torch.Tensor` 注解）与
   `models/qwen4exp`（21 文件 / 824 处）：容器（`torch.empty/zeros/cat/arange`）、
   `dtype=D`（来自 `ops/dtypes.py` 的 `D`/`WF`）、`.to()`/`.view()` 全是 torch 语义。
   这是最大的一块，也是唯一必须**在卡上边跑边改**的一块。
3. `engine`（8 文件 / 62）、`serving`（6 / 44）、`cache`（3 / 42）：批布局、KV 块表、
   传输与快照。
4. `kernels/__init__.py` 用 `torch.utils.cpp_extension` 编译，`bindings.cpp` 收
   `torch::Tensor`：只要还在装载 pybind 模块，进程就 still 有 torch。

## 4. 顺序（每一步都能独立验收）

1. **F + E + C 的逐元素部分**（`triangular_solve`、`hyper_*`、`conv1d_silu*`）：单入口、
   无布局，搬完立刻能在卡上用 `python3 -m cxrt.selftest` 那套对拍。
2. **A 投影**：先把 `linear_prefill` 的库 GEMM 直接接 `cublas`（不经 torch 的
   `torch.matmul`），再把 `linear`/`linear_m` 的 nk GEMV 路径接上（nk 已完成 2 个入口）。
3. **B KV 布局**：定义块表契约（`slot`/`stride_slot`/`stride_head` 三个整数 + 一张
   int32 映射），再搬 `attn_paged*`。
4. **D MoE**：分组偏移契约。
5. **适配层 + 模型层**：`backends/corex` 换成 `cxrt` 装配，`models/*` 的容器换成
   `cxrt.Array`。这一段必须在卡上做，且要按「一层一层对拍 logits」推进。

## 5. 现在的证据状态

**主机侧（无卡、无 torch）**：`cxrt` 十三个模块 + 七个 ABI 面的本地用例
（`python -m unittest discover -s tests -p "test_cxrt_*.py"`，98 条）全绿；
`import cxrt` 后 `sys.modules` 无 `torch`。

**卡上（真集群，作业 5789–5794，`platform/scow` 的 `qjob.py` 投递）**：

- 七个族的 `.so` 全部用厂商 `clang++ -x cuda --cuda-path=/usr/local/corex --cuda-gpu-arch=ivcore10`
  现编成功：`coffin_norm-244b717a…`、`coffin_solve-1fd61744…`、`coffin_conv-ac5c0282…`、
  `coffin_gemv-020a7994…`、`coffin_attn-75a71707…`、`coffin_w8a8-f4d33113…`（缓存在
  `~/.coffin_corex_rt/`，按源码哈希寻址，第二次起零编译）。
- 数值对拍 **8/8**（`python3 -m cxrt.selftest --dtype bf16`，rc=0，设备
  `Iluvatar BI-V100`，落在 `gpu-node-09` 的物理 0 号卡；卡健康是会话级事实，脚本每次
  逐卡实探）：

  | 用例 | 口径 | 实测 |
  | --- | --- | --- |
  | `rms_norm` | bf16 ≤1 ULP | 1 |
  | `add_rms_norm` | bf16 输出/残差 ≤1 ULP | 0 / 0 |
  | `rms_norm_gated` | bf16 ≤2 ULP（`__expf`） | 1 |
  | `rms_norm_i8` | 反量化 ≤1 步、code ≤1 | 2.7e-02 / 1 |
  | `add_rms_norm_i8` | 同上 + 残差 | 2.5e-02 / ok |
  | `triangular_solve` | fp32 相对 4e-6·scale | 9.2e-05（tol 1.7e-03）|
  | `triangular_solve` | bf16 | 0（逐位相同）|
  | `conv1d_silu` ×3 | 相对 2e-5 | 8e-08 / 8e-08 / 1e-07，窗口逐位相同 |

- **踩到并已修的四个真问题**（主机侧不可能发现，都是「一上卡就错」那一类）：
  1. `Shim.export` 自己追加 `err`/`errlen`，而 `solve.py` / `conv.py` 的 `argtypes` 又
     把这两项写了一遍 → `ctypes` 报 `takes at least 14 arguments (12 given)`；
  2. selftest 的卡上参考用 fp64 累加，比内核「更准」，`L=16` 的对拍差 6e-5 被误判成
     内核错——参考必须按内核的逐步 fp32 落盘重写；
  3. bf16 参考若不先把输入窄化成设备真正读到的那一份，差的是「参考输入不同」而不是
     内核误差；改成 `_narrow` 之后五条归一化的 ULP 差从 2/48024/3 降到 1/0/1；
  4. 早先 `add_rms_norm` 报 48042 ULP 的根因就是第 3 条，不是内核。

**因此「两个 w8a8 模型能跑」当前仍不成立**，缺的是第 2 节 23 个算子与第 3 节的适配层、
模型层。

## 6. 模型层这一块的规模（下一步的起点）

`tools/scan_torch_api.py` 的 AST 盘点（`models` / `engine` / `cache` / `serving` / `ops`）：

- `torch.*` 调用面 **1895 处**，分布 **56 个文件**；其中 `torch.Tensor` 809 处（绝大多数是
  注解）、`float32` 109、`zeros` 104、`empty` 82、`long` 69、`cat` 58、`arange` 48。
- 也就是说容器与 dtype 常量占了七成，真正的算子调用（`matmul`/`bmm`/`softmax`/`where`/
  `nonzero`/`cumsum`…）不到两百处，其余走 `backends.ops`。

因此模型层的最短路径是**造一个 torch 形状的薄壳**（`cxrt/tt.py`：`Tensor` + dtype 常量 +
`zeros/empty/cat/arange/stack/split/where/matmul` 这十来个入口 + `nn.Module`/`Parameter`/
`Linear`/`Embedding` 的空壳），然后各文件只改 `import` 一行、内部逻辑先不动。壳里的
`Tensor` 用 `Array` 存设备内存、元素级运算走 numpy、重算子回落到 `cxrt` 的 ABI——先求
「跑得通且能逐层对拍 logits」，再谈性能。

## 7. 进展（本轮，卡上实测）

**已通的四段**：

1. **七个 ABI 族全部编译并数值对拍**（见 §5，作业 5789–5794，8/8）。
2. **真实 checkpoint 能读，且不需要 torch**：落盘是 safetensors 而不是 `torch.save` 的
   zip，`cxrt/safetensors.py` 按 8 字节头长度 + JSON 头 + `data_offsets` 解析，用
   `np.memmap` 直接映射（不把 2.5 GB 读进内存）。作业 5796 的实测：
   - `Qwen3.5-4B-int8-nk`：2 个分片，分别 174 / 524 个张量；`dtypes={int8, float32}`，
     权重 `int8 [N, K]`（`embed_tokens` 是 `(248320, 2560)`）+ 每输出行的
     `weight#scale` fp32；norm/conv 是 bf16。
   - `occamy-1.0`（bf16 源）：14 个分片，每片约 4000–5000 个张量；MoE 专家在源里是
     每专家 `[512, 2048]` / `[2048, 512]` 的 bf16。盘上的 kn 产物（`-W8A8`、
     `-W8A8-moe`、`-W4A8-g128`）**已全部删除**，要用
     `tools/quant_disk.py --layout nk` 从源重新产出 `occamy-1.0-W8A8-nk`。
3. **两个模型包在无 torch 进程里能 import**：`cxrt/install.py` 把 `cxrt.torchshim` 装成
   `sys.modules['torch']`（含 `torch.nn` / `torch.nn.functional` / `torch.distributed` 等子模块
   名），`import models.qwen35` 与 `import models.qwen4exp` 都通过，`verify()` 为真、
   `sys.modules` 里没有真的 torch。真的 torch 在场时 install 会拒绝覆盖。
4. **本地用例 132 条全绿**（`test_cxrt_*.py`）：tt 20 + store 5 + safetensors 5 +
   install 4 + 原有 98。

**下一道墙（已实测定位）**：`engine_api.create_model(..., quant=True)` 走
`quantized_load.load`，两处都还没换（**注：这一节记的是当时的 kn 门，§10 之后这道门
已经拆了**——加载器只认 nk，下面两条留作历史）：

- `quantized_load._require_kn_backend()` 读 `backends.caps().weight_layout`，要求 `"kn"`
  （即内核按 `[K, N]` 连续索引）；`reference` 后端声明 `"nk"`，因此拒绝加载 int8 权重。
  真正能过的后端只有**还没写的 cxrt 适配层**。
- `quantized_load.load` 里 `from safetensors import safe_open` 后按 torch 张量校验 dtype：
  这条路要改成 `cxrt.store` / `cxrt.safetensors` 出 `Tensor`，否则 dtype 校验对不上。

所以剩下的活按顺序是：① 写 `cxrt` 后端适配（声明 `weight_layout="kn"`，把 17 个 ABI
出口映射成 `Ops`，缺的 23 个先用 numpy 兜住并标 `TODO`）；② 把 `quantized_load` 的
读盘换成 `cxrt.store`；③ 逐层对拍 logits（先 qwen35 的一层，再整塔）。

## 8. Qwen3.5-4B 已端到端出 token（无 torch，卡上实测）

绕开 `quantized_load` 的那道 kn 门：`cxrt/dense_weights.py` 把 int8 权重连同
`#scale` 折成 bf16 稠密表（`[K, N]` → `[N, K]`，因为模型的 `Linear`/`GEMV` 一律按行主序
`[N, K]` 建），再走模型自己的 `weights=` 入口。这条路不碰 ABI 适配层，也不需要那 23
个缺口算子——它验的是「模型代码能不能在这套壳上跑完一次前向」。

作业 5816（`q35dense17`）实测，`gpu-node-07`，`--cores 32`：

```
[run] dense tensors=441 in 68.8s
[run] built in 22.7s layers=32 hidden=2560
[run] prefill 5 tok in 4.8s
[run] decode 8 tok in 4.5s
[text] ' Paris. ham hamlet of the same'
[run] rc=0
```

同一条路跑到 24 token（`q35long`）：

```
[text] ' Paris.\n\nThe capital of France手电 is Beijing.\n \nThe capital of France is Paris.\n\n'
```

结论：**权重能读、模型能建、prefill 与 decode 都出 token、tokenizer 能解码**，第一个
token 就是对的（`The capital of France is` → ` Paris`）。中间那个 `手电` 说明数值还不是
逐位一致——int8 反量化 + bf16 舍入 + 参考后端的 fp32 累加三者叠加，量级上是对的，
峰值上会抖。要把它逼到逐位一致，得换回 int8 内核（也就是 §2 的 33 个算子），这条路
只负责证明「能跑」。

**上卡才暴露、已修的 11 处壳缺口**（每条都是作业跑出来的，不是猜的）：

| 症状 | 根因 | 修法 |
|---|---|---|
| `could not broadcast (2560,4096) into (2560,8192)` | checkpoint 存 `[K,N]`，模型按 `[N,K]` 建 | `dequantize(transpose=True)` |
| `unsupported operand ** : int ** Tensor` | 缺 `__rpow__` | 补 `__pow__`/`__rpow__`/`__rtruediv__` |
| `Tensor has no attribute stride` | 缺 `stride()` | 按**元素**报（`bytes / data.dtype.itemsize`） |
| `torch.promote_types` 缺失 | 参考后端选计算 dtype | 补推进阶表（bf16+fp32→fp32） |
| `.t()` / `unfold` / `repeat_interleave` 缺失 | conv/delta 的三出口 | 逐个补，`unfold` 用 `sliding_window_view` |
| `masked_fill` 布尔索引形状不符 | 掩码 `[L,L]` 要盖 `[H,nc,L,L]` | 改 `np.where` 广播 |
| `output array is read-only` | `np.diagonal` 是只读拷贝，torch 是视图 | `as_strided` 造**可写**对角视图 |
| `add_residual needs a dense destination; got strides (5120,2)` | bf16 用 fp32 存，步长按逻辑 itemsize 除错了 | 按 `data.dtype.itemsize` 报 |
| `got strides (1,)` 仍不稠密 | `Tensor(...)` 构造器会按 dtype 重排→视图变拷贝 | 新增 `_wrap()`，元数据操作不再拷贝 |
| `Ops has no attribute kv_store` | 参考后端从没实现单槽位融合写回 | 补 `_kv_store` + 协议字段 |
| `torch.eye` 缺失 | 分块 delta 的单位阵 | 补 `eye`/`linspace`/`baddbmm`/`maximum` |

**内存是硬约束**：`--cores 8` 跑稠密路会在 prefill 里被 OOM 杀掉（rc 137），换
`--cores 32` 才过。4B 的稠密表就是 8 GiB 量级（int8 4.03 GiB 折成 fp32 存放的 bf16
再加临时量），作业的内存配额看起来随核数走。

## 9. Occamy 1.0 W8A8：还没跑起来，卡点已定位到具体一行

`models/qwen4exp/engine.py` 的装载入口对量化产物是**显式拒绝**的（作业
`occamprobe` 在卡上实测，不是读代码推的）：

```
[probe] quant=True refused: NotImplementedError: qwen4exp 的 int8 装载还没接：
        量化产物由 proj.set_quantized 供货 … 真机部署请先用 --no-quant 跑通，再等量化产物接上。
[probe] bf16 lane: AttributeError: module 'cxrt.torchshim' has no attribute 'UntypedStorage'
        ← models/qwen4exp/checkpoint.py:89  with safe_open(str(path), "pt")
```

第二条说明连「按 bf16 建个索引」这一步都还过不去：真 `safetensors` 包在 import 期就要
`torch.UntypedStorage` / `frombuffer` 这一层，本壳没造。所以 Occamy 缺的不是一个参数，
是两段：**量化装载（kn 适配）** 与 **safetensors↔Tensor 的桥**。


而 `--no-quant` 那条路要的是 bf16 源（参考 checkpoint 360 GB：专家 246 GB、n-gram 表
102 GB），盘上根本没有——只有 `occamy-1.0-W8A8`（35 GB int8，nk 布局）、
`-W8A8-moe`。

想用「Qwen 那条稠密捷径」也不行，两个量级上的理由：

1. 专家是 256 个 × `moe_intermediate_size=512` × `hidden=2048` × 2 个矩阵 × 40 层 ≈
   **21.5 B 参数**，折成 fp32 存放的 bf16 是 86 GB 起，还得叠上 248 K 词表嵌入与
   注意力；作业配额撑不住，且 `weights.plan` 要的是**全局 3D 融合专家张量**，
   盘上是每专家 `[512, 2048]`，得先在源侧拼。
2. 更关键的是 `weights.load_into` 期望的是「流式源」而不是 state dict，量化张量的
   对应参数是 `QWeight`/`QWeightMoE`，由 `proj.set_quantized` 供货——那正是 §2 里
   还没搬的 23 个算子与 kn 适配层要接的东西。

所以 Occamy 的下一步是确定的，不是猜的：① 写 `cxrt` 适配后端（`weight_layout="nk"`，
17 个 ABI 出口 → `Ops`，缺的先用 numpy 兜）；② 给 `qwen4exp` 接一个
`TensorSource`，把 14 个 int8 分片按 `#scale` 现折现给，并让 `plan` 认得
「每专家 `[512,2048]` → 每层 3D 融合」这一层映射；③ 再把 `quantized_load` 换成
`cxrt.store`。这三步里 ① 是 Qwen 与 Occamy 共用的，做完两边同时解开。

## 10. 布局收敛到 nk：盘上 `[N, K]`，加载器不再假定 kn

上一版把「kn 门」绕过去才跑通 4B，现在这道门本身被拆了。盘上只剩一种布局：`int8
[N, K]` 连续（一行输出通道一段连续字节、K 轴在内）+ `fp32 [N]` 每行 scale。

改的是读取端，不是内核端：

| 位置 | 之前 | 现在 |
|---|---|---|
| `tools/quant_disk.py` | 写 `[K, N]`，MoE 栈另存一份 `#nk` 副本 | `--layout` 默认 `nk`，主键就是 `[N, K]`/`[E, N, K]` |
| `quantized_load.py` | `_require_kn_backend()` 挡掉非 kn 后端，`QWeight.from_kn` 造转置视图 | 只认 `weight_layout == "nk"`，`QWeight(wt, scale)` 直接包盘上缓冲 |
| `cxrt/dense_weights.py` | 按列乘 scale 再 `dense.T` | 按行乘 scale，**不转置**（模型本来就按 `[N, K]` 建） |
| `backends/corex/pack.py` | `to_kn` 只收 `[K, N]` | 当时是 `KnQWeight(..., nk=)` 交进现成的 nk 缓冲；**后来整份 `pack.py` 收成只有 `WEIGHT_DTYPE` 与 `LIB_MAX_N` 两个常量**，kn 类型与转换一个不剩（见 `docs/COREX-GEMM-NK.md` §5） |

后端不再向加载器要 `[K, N]`：库 GEMM 用 `transa = T` 直接吃 `[N, K]`，int8 解码的内核
本来就要 nk。所以加载器与内核之间**没有任何转置**，也没有任何一份副本。

卡上实测（`gpu-node-07`）：

- `quant4bnk`：4B bf16 → int8 nk，9.34 GB → 4.36 GB（0.466），43.4 s，257 个 q8 +
  184 个稠密张量；`tools/check_nk_disk.py` 逐张量比源 bf16，相对 RMS 1.0–1.7%（int8
  步长量级），`errors=[]`。
- `nkload`：`quantized_load.load` → `QWeight` → `pack.to_kn`。四个张量的 `wq` 形状与源
  同形（不是转置）、`nk_weight()` 原对象返回（零拷贝）、`kn.t() == nk` 逐位相等。
- `q35nki8`：corex 后端 + `--int8` 直读 nk 目录，1.7 s 装完，prefill 71 tok/s、
  decode 64.5 tok/s，输出 `' Paris.\nA. True\nB. False\nAnswer:\nA'`。
- `q35nkref`：同一 prompt、bf16 源、reference 后端 —— 那次跑**逐 token 相同**，所以 nk
  这条读取链与未量化的基线一致，上面那个 token 序列不是量化噪声。⚠️ 这条结论只对那次
  观测成立：后来用 int8 目录跑 reference 后端（`nkref1/nkref2`）在第二个 token 就分叉，
  而那个 prompt 恰好落在近等概率区（见 §11 第 3 条）。也就是说这条路**不构成**量化
  无损的证据，逐张量 rel_rms 才是。

仍待办的一条：§8 的 `cxrt` 稠密路（它现在也读 nk）在同一 prompt 上给出的是
`' Paris.\n\nThe capital of France is Paris...'`，与基线在第二个 token 就分叉。那条路只
用于验证「模型代码无 torch 能跑」，分叉得单独查（最可能是 bf16 稠密表在
`int8×scale` 之后又被舍入一次）。




## 11. 4-bit 全链路删除与 4B 的性能现状

`w4a8` 这一族从契约、模型、内核、工具、文档里一起删掉了：盘上不再有 4-bit 产物，
加载器不再有 W4 分派，`Ops` 表里 `moe_grouped_gemm` 这个 W4 专用的大 M 入口整条消失
（预填充改走 `moe_grouped_i8_tcu`，nk 原生）。删掉的东西：

| 层 | 删掉的内容 |
|---|---|
| 契约 | `Caps` 里对 packed W4 的分支、`Ops.moe_grouped_gemm`、`weight_layout` 只剩 `"nk"` |
| 模型 | `models/qwen35/moe.py` 的 `W4A8MoEWeight` 融合分支与 `_grouped`、`models/qwen35/parallel.py` 的 W4 索引路径与 `RowParallelProjection` 的 W4 分支、`models/qwen4exp/*` 全部 W4 引用（改为 int8 `QWeight`/`QWeightMoE`） |
| 内核 | `w4a8.cu`、`w4_dequant.cu`、`moe_grouped.cu` 的分组 W4 内核、`w8a8_tcu.cu` 的 `w4_gemm_tcu` + `w4a8_gemm_tcu` + 分组 W4 GEMM、`gemv.cu` 的 `gemv_kn_w4`、`corex_ops.h` 里对应声明 |
| 工具 | `w4a8_weights.py`、`quant_disk_w4a8.py`、`quant_disk_qwen4.py`、四个 `probe_q4_*`、两个 W4 启动脚本 |

顺带修掉的一条**隐性能损失**：`linear` / `_gemv_rows` / `linear_prefill` 过去一律先转
一份 `[K, N]`，对 nk 原生的 `QWeight` 来说这一步会为每个权重转置出完整的副本（4B 是
4.3 GB），随后又把原缓冲要回来——那份副本除了吃显存什么都不做。现在三条入口都直接把
`w.wq`（`[N, K]`）交给内核或库的 TN 出口，`linear_prefill` 也不再为 `w8a8_gemm_tcu`
转置任何东西——那个稠密张量核本身已经删掉（`w8a8_tcu.cu` 被 `git mv` 成
`moe_i8_tcu.cu`，只剩 nk 的分组核）。

### 4B（`Qwen3.5-4B-int8-nk`，corex，单卡，`gpu-node-07/08`）

`tools/bench_qwen35_nk.sh` 的扫描（`graph=False`，每档之后解码 32 token）：

| 上下文 | 预填充 tok/s | 解码 tok/s |
|---|---|---|
| 512 | 2469 | 62.5 |
| 2048 | 2822 | 57.7 |
| 4096 | 2639 | 55.1 |

加载 1.69 s，峰值显存 **9.41 GiB**（`torch.cuda.max_memory_allocated`，含最重的一次
预填充）。128 那一档是 159 tok/s / 803 ms，多出来的是首次调用那份 kn 副本与冷缓存，
不是稳态，不参与比较。作为量级对照，reference 后端同一目录是预填充 2 tok/s、解码
0.54 tok/s（那个后端按行逐个循环，没有内核）。

### 同一版里另外确认的五条（都不是布局问题）

1. `rms_norm` 现在是**精确的**：`tools/probe_corex_norm.py` 在 M = 1/2/8/32 四档上
   rel_rms 全 0。图捕获挂在 `shape '[32, 2560]' is invalid for input of size 2560`
   这件事根因不在 `norm.cu`，而在 `ops/norm.py` 的缓存键：它只按 `id(x)` 判断「同一块
   激活」，可预填充各层拿到的是同一块 `[M, hidden]` 缓冲的不同视图，`id` 相同而行数
   不同，于是解码步缓存的「一行」`_rows` 被拿去喂 `M = 32` 的调用。改成
   `(id(x), x.shape, x.stride())` 之后捕获就不再崩（见第 4 条）。
2. 两支投影内核各自对拍都对：`tools/probe_corex_gemv.py`（512×2560）里
   bf16 kn 是 0.0、int8 nk 是 0.24–0.89%（per-row absmax 的量化噪声）。而整模在
   bf16 稠密下输出仍是 `The The The…`——投影两支都对，说明差在别处（attention /
   delta / rope 那一族还没改完），不是投影也不是布局。
3. 这个 5 token 的 prompt（`The capital of France is`）本身落在近等概率区，别用它当
   正确性标准：corex int8（eager）与 reference 后端的稠密 bf16 前两个 token 是一致的
   （` Paris.` / `A.unner`），但图路径给出 `A. True/ and the the is the capital?`，
   而更早一次 int8 跑的是 `A. True`。也就是说**同一个模型换一条执行路径就能换一套
   token**，谁也证不了谁。能比的只有 §10 那套逐张量 rel_rms（1.0–1.7%）与逐算子的
   `probe_corex_*` 对拍；把「输出和某次运行逐 token 相同」当验收条件，在这个 prompt
   上必然失败，跟布局无关。
4. `graph=True` 修完第 1 条之后捕获成功、解码 63.70 tok/s，不再崩。但它给出的 token
   仍与 eager 不同（`A. True/ and the the is the capital?` 对 `A.unner`），所以图
   路径**还没有资格上线**：崩溃修掉了，数值差没查。量产跑法仍是 `--no-graph`。
5. **分组 int8 张量核有个真实缺陷（已挡住）**：`tools/probe_corex_moe_i8.py`
   （E=8、N=512、K=2048、nk 零拷贝）量到 `moe_grouped_i8_tcu` 在每专家 1/2/3 行时
   rel_rms 分别是 0.86/0.72/0.86，同一个栈走 GEMV 结构是 0.008；每专家 4 行起才回到
   0.0087。原因是共享内存里 64 行 tile 末端没有零填充，短段的垃圾行会被 mma 算进
   有效行——症状是「输出看着像话但数值不对」。包装层现在对 `tokens < 4*experts`
   抛 `ValueError`（`_make_grouped` 捕获后退回 GEMV），probe 里短段显示 `"guarded"`。
   正常批量（`COFFIN_MOE_GEMM_MIN_ROWS` 默认 128 行/专家）踩不到，但 Occamy 那种
   `top_k=8` 的稀疏解码如果按专家切段就会出现。

### 预填充的两条路：省显存还是省时间

`linear_prefill` 对 int8 nk 权重有两条路径：

| 路径 | 权重侧动作 | 4B 预填充 512 行 | 峰值显存 |
|---|---|---|---|
| 库 TN（`GemmEx`，`transa = T`，**默认**） | 零拷贝，`[N, K]` 原缓冲 | 207 ms / 2469 tok/s | 9.41 GiB |
| `moe_grouped_i8_tcu` 的 `E = 1` 视图（`COFFIN_PREFILL_NK=1`） | 零拷贝，nk 原生 | 774 ms / 662 tok/s | 6.01 GiB |

两条都**不转置**：上面第一行的 9.41 GiB 是 `gpu-node-07` 上量到的峰值，其中最大的一块
是那次 2048 行预填充本身的工作区，不是权重副本——那条 `kn 副本 + w8a8_gemm_tcu` 的
9.41 GiB 旧记录已经随稠密张量核删除而不再成立。分组内核按专家切并行度，`E = 1` 只剩
一维可用，所以慢 3.7 倍；它换来的是库里那一趟 workspace。Occamy 的 35 GB 权重在
31.7 GiB 的卡上放不下那一档工作区，到那一步才走后者——这条开关是为它留的，不是为了 4B。
