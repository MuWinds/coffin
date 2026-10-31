# 多平台算子架构（coffin）

> 状态：**结构方案待评审**（2026-10-25）。范围：把 `coffin/` 从"单平台（海光 DCU / gfx906）引擎"
> 改成"一套模型语义 + 多套平台算子"。
> 目标平台：**国产加速卡**——天数智芯（Iluvatar）与燧原（Enflame）优先，摩尔线程（Moore Threads）次之；
> 海光 DCU 保持现有性能不变。机器与数据后续逐步补，本期目标是**把布局定死**：
> 加一个平台 = 加一个文件，别的什么都不用改。
> 本文只讲**结构与契约**；性能数字来自 `ENGINE.md` / `VISION.md`，不在这里重复。

---

## 0. 结论（先看这 10 条）

1. **现在的阻塞点不是"没有抽象层"，而是三处平台假设直接写进了模型代码**：`import hip_ops`（生产侧
   7 个 `.py` 文件、75 处属性引用）、`"cuda"` / `torch.cuda.*`（生产侧 13 个文件、181 行）、
   `ops/dtypes.py` 里 `D=bf16 / WF=fp16`（这是 gfx906"只有 packed fp16"的硬件结论，不是模型结论）。
   三者都必须变成后端的 **capability**。
2. **抽象粒度选语义算子，不是 `hip_ops` 的签名**。`gemv(w, x, R, U)` 里的 `R`/`U` 是 HIP 核的 tiling 参数，
   `attn(...)` 里的 `part_acc/part_ml` 是 split-K 的 online-softmax 缓冲区——这些是**后端的私事**。
   跨平台契约按语义定：`linear(w, x)` / `rms_norm(x, g)` / `delta(...)`。
3. **7 个投影入口收敛成 3 个算子**（`linear` / `linear_m` / `linear_prefill`），
   内核选型与 autotune 全部下沉到后端。`gemv_w8a16_wide_dbg` 这类 A/B 调试入口留在 HIP 后端内部，不进契约。
   int8 权重不是第四个算子：`w` 是 `QWeight` 还是普通张量，由后端内部分派（`int8_weights` 是叶子模块，后端可以 import）。
4. **算子表对调用方是全量的（total）**：`backends.ops` 对 17 个算子名**永远返回可调用对象**，后端没有原生核就
   返回 reference 的纯 torch 实现。于是 `models/` 里 15 处 `if hip_ops.available(): ... else: <eager>` 判断
   连同约 200 行分散的兜底代码**全部删除**，模型代码只剩一条路径。这是本次重构最大的代码量削减，也是
   "新平台不需要改模型"的前提。`Ops` 里允许 `None`，但那只在后端内部用作"我没有原生核"的信号，
   由注册表在组装时用 reference 填平，**不会到达调用方**。
5. **`reference` 后端就是规格（spec）**。每个算子先有纯 torch 参考实现，HIP 后端与未来平台都要和它对齐；
   一致性用例 `tests/conformance.py` 按后端参数化跑，**新平台通过一致性用例才算接入完成**，
   性能是第二步。
6. **能力位（caps）替代 `available()` 分支**。字段以落地的 `backends/protocol.py` 为准（14 项）：
   `name`、`device`、`device_index`、`device_name`、`arch`、`native`（原生算子名集合）、
   `act_dtype`、`weight_dtype`、`graph`、`autotune`、`attn_split`、`batch_attn`、`int8_weights`、`recurrent_state`。
7. **`ops/dtypes.py` 的 `D`/`WF` 迁到 caps，`models/qwen35/tuning.py` 的 `ATTN_SPLIT` 迁到后端**。
   后者现在注释就写着"必须等于 `kernels/attn.cu` 里的 `ATTN_SPLIT`"——这是后端 ABI，放在模型族里是错位。
8. **目录结构**：`coffin/backends/{protocol,vendor,discovery,registry}.py` + `backends/{reference,hip}/` + `backends/vendors/`；厂商平台只登记在 `vendors/` 下，一个文件一个平台。
   `hip_ops.py` 与 `kernels/` **整体搬进 `backends/hip/`**，顶层同名模块**删除**（用户已定：不留兼容 shim）。
   HIP 专有探针搬进 `backends/hip/diag/`，与模型/服务探针分开。
9. **一次性全量改名，但分 6 步落地，每步树都是绿的**（S0 骨架 → S1 搬核 → S2 设备/dtype → S3 删回退分支
   → S4 探针归位 → S5 收紧门禁 + CPU 冒烟）。门禁用 `tests/test_backends_purity.py`（纯扫描），
   违规清单从"当前全量"收敛到空，迁移进度可量化。
10. **目标是国产加速卡**：天数智芯（`cuda` 命名空间 + 设备名识别）、燧原（`gcu`）、摩尔线程（`musa`）。
    三家都已登记为占位后端（`has_kernels=False`），机器到位后按 §7 的顺序逐个接。
    加第四家仍然只是"加一个文件"——临实测过：放进 `vendors/ascend.py` 后 `available_names()` 立刻多一项。

---

## 1. 现状：三处平台假设

| # | 位置 | 现状 | 为什么挡路 |
| --- | --- | --- | --- |
| P1 | `coffin/hip_ops.py` + `coffin/kernels/*.cu`（25 个导出符号） | 唯一的设备算子入口，全局模块名 `hip_ops` | `ops/`、`models/qwen35/`、`serving/` 直接 `import hip_ops`（生产侧 7 个文件、75 处属性引用），换平台要改模型代码 |
| P2 | `device="cuda"` / `.to("cuda")` / `torch.cuda.*` | 生产侧 13 个文件 181 行（`model.py` 82、`linear_attn.py` 29、`mtp.py` 25、`full_attn.py` 15、`ops/gemv.py` 10、`serving/runtime_parts.py` 10，其余 7 个文件各 1–2 行） | 在 torch 里 `cuda` 是**设备类型名**，而且多个国产卡都复用它（天数智芯即是）。写成字面量等于写死了一个平台 |
| P3 | `ops/dtypes.py`：`D=bf16`、`WF=fp16` | 注释写明依据是"gfx906 有 packed fp16、没有 packed bf16" | 这是硬件的结论，却被当成模型的常量；换了卡 `WF` 可能该是 bf16 |
| P4 | `models/qwen35/tuning.py:ATTN_SPLIT = 8` | 注释："必须等于 `kernels/attn.cu` 的 `ATTN_SPLIT`（这一侧拥有 partial 缓冲）" | 后端 ABI 泄漏到模型族；`part_acc/part_ml` 的形状由它决定 |
| P5 | `ops/gemv.py` 用 `torch.cuda.Event` 在加载时实测挑 `(R, U)` | 6 组候选、每形状一次 sweep | 平台探测 + 调参策略写在模型侧的投影类里，新平台要重写一份 |
| P6 | `models/qwen35/model.py` 27 处 `CUDAGraph`/`graph_pool_handle` | decode 全静态形状 + 一次捕获 | 不能捕获的平台必须能退化成 eager，而不是崩 |
| P7 | `tests/support.py:has_hip()` + `diag/` 93 处 `hip_ops` | 探针直接调核 | 探针也要按后端分层，否则"支持多平台"立刻变成"维护两份 diag" |

补充事实（扫描所得，不是估计）：

- 生产侧 `import hip_ops` 的真正 `.py` 只有 **7 个**：`ops/gemv.py`、`ops/norm.py`、
  `models/qwen35/{batch,full_attn,linear_attn}.py`、`serving/{runtime_parts,runtime_transport}.py`。
- `diag/` 有 **35 个文件 / 360 行**设备假设、**9 个文件** `hip_ops` import；`tests/` 另有 7 个文件 77 行。
  门禁把两者合成 `probe` 组（435 行 / 42 文件），因为它们同样不是产品代码。
- `engine_api._RELOAD_ORDER` 现在 33 项；后端模块必须登记进去，否则 `reload()` 按设计直接抛错。
- 数字口径：门禁统计的是**违规行数**（一行只算一次），不是正则命中次数；逐文件清单见
  `.tmp_dsh/platforms/INVENTORY.md`。

---

## 2. 目标结构

**核心约定：加一个平台 = 加一个文件。** 不需要改注册表、不需要改测试清单、不需要写粘合代码。
`backends/vendors/<名字>.py` 放一个 `SPEC` 就是全部；文件名即后端名（`COFFIN_BACKEND=<名字>`）。

```
coffin/
├─ backends/                    新增：平台层（唯一允许出现硬件事实的地方）
│  ├─ __init__.py               对外唯一出口：ops（算子表）、backend()、caps、device、reload()
│  ├─ protocol.py               Ops（17 算子表）/ Caps（能力位）/ Backend 协议
│  ├─ vendor.py                 VendorSpec / Detection / SpecBackend / detect——声明式后端的全部机制
│  ├─ discovery.py              扫描 vendors/，import 每个 SPEC，包装成 backend（无中心清单）
│  ├─ registry.py               COFFIN_BACKEND 选择 + 按 claim 强度排序 + 回退
│  ├─ reference/                纯 torch 后端 = 语义规格；任何平台都能兜底
│  ├─ hip/                      手写后端：gfx906（现有实现整体搬入，含 kernels/、tuning.py、diag/）
│  └─ vendors/                  ← 加平台只动这里
│     ├─ __init__.py            目录约定（这里写明厂商清单与申报规则）
│     ├─ iluvatar.py            天数智芯：cuda 命名空间 + 设备名识别，[K,N] 权重
│     ├─ enflame.py             燧原：gcu 命名空间，int32 索引
│     └─ moorethreads.py        摩尔线程：musa 命名空间，MTT 设备名
├─ ops/                         保留：模型级算子组合，只调 backends.ops
├─ models/qwen35/               保留：模型族，不再出现 hip_ops / "cuda" / dtypes 常量
├─ engine/, cache/, serving/, entrypoints/, utils/    保留
└─ layers/                      删除（已无人 import 的兼容壳，与"不留 shim"冲突）
```

为什么 `hip` 是手写的、其它是声明的：`hip` 有真实内核、核内 tiling、autotune 和构建路径，
这些不是一个声明能表达的东西；其余平台在拿到机器之前只有"事实与期望"，正好是声明能表达的。

依赖方向（单向，门禁强制）：

```
backends/reference ← backends ← ops ← models/* ← engine_api ← serving/ ← entrypoints/
```

- `backends/` **不许** import `ops`、`models`、`engine`、`serving`、`cache`。
- `ops/` 以下**不许**出现 `hip_ops`、`gfx906`、`--offload-arch`、`PYTORCH_ROCM_ARCH`、
  `TORCH_EXTENSIONS_DIR`、`"cuda"` 字面量、`kernels/` 路径。
- `diag/` 只留后端无关的探针；后端专有探针进 `backends/hip/diag/`。

### 2.1 两种探测路径，一个机制

调研三个目标平台时发现了一个必须吸收的差异，这也是 `detect()` 返回**claim 强度**而不是布尔值的原因：

| 平台 | torch 命名空间 | 怎么认出它 |
| --- | --- | --- |
| 海光 DCU | `cuda` | ROCm 构建标记（手写 `backends/hip`） |
| 天数智芯 | `cuda` | **设备名**——命名空间与海光共用，名字是唯一区分依据 |
| 燧原 | `gcu` | 命名空间本身就是厂商独占 |
| 摩尔线程 | `musa` | 命名空间本身就是厂商独占 |

- **独占命名空间**（`gcu`/`musa`）：`caps.device` 一验即知，不需要名字规则。
- **共用命名空间**（`cuda`）：`torch.cuda.is_available()` 对海光、天数、NVIDIA 都返回 True，
  所以**不能用它当平台判据**；只能靠设备名。天数智芯的厂商适配仓库自己写的是
  `torch.cuda.get_device_name() == "Iluvatar BI-V100"`，且 ComfyUI 实机日志是
  `Device: cuda:0 Iluvatar BI-V150`。

于是 `detect()` 给出 `CLAIM_NONE / CLAIM_NAMESPACE / CLAIM_NAME` 三档，注册表取**最强 claim**。
天数卡上 ROCm 后端与天数后端都可能说"我看到一块 CUDA 设备"，天数因为名字命中而胜出。

**这条设计的好处**：它不依赖一个尚未定论的事实——天数智芯的 runtime 是否也设
`torch.version.hip`（有报告说它抛 `No HIP GPUs are available`，那串字只存在于 HIP 派生构建里，
意味着 `version.hip` 可能非空）。无论答案如何，按 claim 强度排序都得到正确结果。

### 2.2 加一个平台要做什么

```python
# coffin/backends/vendors/acme.py —— 全部内容，别的文件一个字都不用改
from backends.vendor import Detection, VendorSpec

SPEC = VendorSpec(
    name="acme",                                   # 名字 = 文件名 = COFFIN_BACKEND 值
    detection=Detection(namespace="cuda", name_markers=("acme",)),
    weight_layout="kn",                            # 只有量过才写，否则留默认
    has_kernels=False,                             # 有真核之前不声明可用
)
```

然后 `python -m unittest discover -s tests -p test_backends.py` 自动把新厂商纳入测试
（有专门用例枚举 `discovery.modules()`，不写死名单）。实测：临时放进一个 `ascend.py`，
`available_names()` 立刻从 5 项变 6 项，删掉又变回去。

拿到机器后接第一个算子的路径：把 `has_kernels=True`、写 `build_ops()` 返回**部分**表
（比如只有 `linear`），其余 16 个算子由 reference 自动补齐——模型一行不改就能跑。

---

## 3. 算子契约

### 3.1 为什么不是 `hip_ops` 的签名

| 现状签名 | 泄漏了什么 | 契约里改成 |
| --- | --- | --- |
| `gemv(w, x, R, U)` | `R` 行/线程块、`U` K 展开：HIP 核的 tiling | `linear(w, x)` |
| `gemv_m(w, x, U)` | 只对 M ∈ {2,4,8} 有核 | `linear_m(w, x)`（后端决定 M 分档） |
| `gemv_w8a16_wide*(w, scale, x, rows, mw)` | wide-M 专用核 + A/B 调试旋钮 | 同上 `linear_m`；`_dbg` 留在后端内部 |
| `attn(..., part_acc, part_ml)` | split-K 的 online-softmax 缓冲 | `attn_decode(...)`，缓冲由后端按 `caps.attn_split` 自持或由调用方按 caps 尺寸分配 |
| `available()` / `last_error()` | 布尔式的"有没有核" | `caps.native` / `backend().error` |

原则：**契约参数只能是模型语义里存在的量**（形状、dtype、量化格式、位置、因果长度）；
凡是"为了这块卡怎么切线程/怎么分块"的量，都留在后端。

### 3.2 算子表（17 个）

| 组 | 算子 | 语义 | 现有来源 |
| --- | --- | --- | --- |
| 投影 | `linear(w, x)` | 单/少 token 权重读：`x [K]` 或 `[C,K]` → `[N]` / `[C,N]`；`w` 可为 `QWeight` | `gemv` / `gemv_w8a16` |
| | `linear_m(w, x)` | 小 M 批投影，一次权重流出 M 行 | `gemv_m` / `gemv_w8a16_m` / `gemv_w8a16_wide*` |
| | `linear_prefill(w, x)` | prefill 分块 GEMM（大 M） | `F.linear` + `dequant`（现状） |
| 归一化 | `rms_norm(x, g, stride, eps)` | 零中心 RMSNorm，支持行跨步 | `rms_norm` |
| | `rms_norm_gated(x, w, gate, eps)` | 标准 RMSNorm × silu(gate) | `rms_norm_gated` |
| 注意力 | `attn_decode(q, gate, cache, pos, ...)` | 单 token 全注意力（dense 缓存） | `attn` |
| | `attn_paged(q, gate, pool, pos2slot, pos, ...)` | 分页池上的单 token 注意力 | `attn_paged` |
| | `attn_paged_batch(...)` | B 行一次 launch，按行 `context_lens` | `attn_paged_batch` |
| | `attn_paged_multi_q(...)` | verify：c 个 query 共享一次历史 KV | `attn_paged_multi_q` |
| DeltaNet | `conv1d_silu(w, x, state, out)` | 因果 depthwise conv4 + SiLU（单 token，改 state） | `conv1d_silu` |
| | `conv1d_silu_seq(w, x, state)` | 同上，整块 prefill | `conv1d_silu_seq` |
| | `conv1d_silu_batch(...)` | B 行 conv+silu | `conv1d_silu_batch` |
| | `delta_prologue(...)` | q/k 归一 + 头扩展 + beta/g（单 token） | `delta_prologue` |
| | `delta_prologue_batch(...)` | B 行头准备 | `delta_prologue_batch` |
| | `delta_prep_seq(...)` | prefill 头准备，输出 `[H,T,d]` | `delta_prep_seq` |
| | `delta(...)` | gated delta 递推一步（改 state） | `delta` |
| | `delta_batch(...)` | B 行递推一步 | `delta_batch` |

以上 17 个名字与 `backends/protocol.py:_OP_NAMES` 逐字一致（S0 已落地并有测试钉住）。

**不进契约的**：`gemv_w8a16_wide_dbg`、`gemv_w8a16_wide_dot2_dbg`（A/B 调试）、`_load()`、`_error`。

**已知重复，本期不动**：`attn_decode`（dense 缓存）与 `attn_paged` 只差寻址方式；dense 路径是单流遗留，
真正的合并方式是"永远走分页池，不启用 paging 时给一张单位槽位表"。这会把 4 个注意力算子降到 3 个，
但要改 `full_attn` 的缓冲与 `capture`，放到 S5 之后单独一轮，避免和平台迁移混在一起。

### 3.3 全量表格 + reference 兜底

```python
# backends/protocol.py（S0 已落地，这里是阅读用的摘要）
@dataclass(frozen=True, slots=True)
class Caps:
    name: str                    # "hip" / "iluvatar" / "enflame" / ...
    device: str                  # "cuda" / "gcu" / "musa" / "cpu"
    device_index: int = 0        # 与 device 拼成 torch.device
    device_name: str = ""        # 板卡名，只给日志/横幅
    arch: str = ""               # "gfx906" 等，编译旗标从它派生，不写字面量
    native: frozenset[str] = ...  # 有原生核的算子名
    act_dtype: torch.dtype       # 原 ops/dtypes.py 的 D
    weight_dtype: torch.dtype    # 原 ops/dtypes.py 的 WF
    graph: bool                  # 能否捕获 decode 图
    autotune: bool               # 能否按形状实测选核
    attn_split: int              # attention split-K 宽度（后端 ABI）
    batch_attn: bool             # 有批量 attention 核
    int8_weights: bool           # 有 int8 权重原生路径
    recurrent_state: bool        # 快照/恢复是否由后端负责（hybrid 模型）
```

`Backend.ops` **对每个算子名都返回可调用对象**：有原生核返回原生；没有就返回 `reference` 的实现。
于是模型侧只有一条路径：

```python
# 现在（models/qwen35/full_attn.py）
if self.pool is not None and hip_ops.available():
    out = hip_ops.attn_paged(q.contiguous(), gate.contiguous(), kp, vp, kps, vps,
                             self.pool.pos2slot[self.prefill_row], self.attn_out1,
                             pos1, self.max_seq, self.rep, self.scale,
                             self.part_acc1, self.part_ml1)
elif hip_ops.available():
    out = hip_ops.attn(...)          # dense 缓存
else:
    ... 12 行 eager 回退 ...

# 之后
out = bk.ops.attn_paged(q.contiguous(), gate.contiguous(), self.pool.view(self.prefill_row),
                        self.attn_out1, pos1, self.max_seq, self.rep, self.scale)
```

**这条规则的价值**（数字来自 AST 扫描，不是估计）：生产侧一共 **15 处** `if hip_ops.available()` /
`hasattr(hip_ops, ...)` 判断，分布在 5 个文件：`ops/norm.py` 2、`ops/gemv.py` 4、
`models/qwen35/{batch,full_attn,linear_attn}.py` 2/3/4。它们带出的纯 torch 兜底实现约 200 行，其中
**161 行已经是 `ops/delta.py` 的 `chunked_delta`**（纯 torch，本来就是 reference 形态），剩下的是
`full_attn` 的 eager 注意力（~14 行）、`linear_attn` 的 eager mixer（~71 行）、`gemv` 的
`deq/lin/mv/wt` 四种兜底形式（~40 行）与 `norm` 的两个 `F.rms_norm` 调用。

这 15 处判断与它们的兜底代码全部删除，数学**搬进 `reference`**：从此只在一个地方维护、只被一致性用例验证，
而不是散落在模型文件里各写一遍。删掉之后，"这块卡有没有原生核"不再影响 `models/` 的任何一行。

---

## 4. 选择、回退与可观测性

选择顺序：

1. `COFFIN_BACKEND` 环境变量精确指定（`hip` / `iluvatar` / `enflame` / `moorethreads` / `reference`）——排障与 A/B 用；
2. 自动探测：**按 claim 强度取最强**——设备名命中厂商标记（`CLAIM_NAME`）优于仅命名空间命中（`CLAIM_NAMESPACE`），见 §2.1；
3. 兜底 `reference`（纯 torch，永远可用，CPU 也能跑）。

必须写进启动日志的一行（现在 `serving/runtime_transport.py` 打印 `hip=...`）：
`backend=hip device=cuda:0 native=17/17 graph=1 autotune=1`。
`serving` 的 `ping`/`status` 返回里 `hip` 字段改成 `backend`（含 name/native/graph），并保留 `hip_err` 等价物
`backend_error`。

**`importlib.reload(hip_ops)` 改成 `backends.reload()`**，且 `engine_api._RELOAD_ORDER` 增补
`backends.protocol` / `backends.registry` / `backends.<name>` / `backends`（顺序：protocol → registry →
具体后端 → reference → 包）。

---

## 5. 迁移步骤（每步结束都必须绿）

| 步 | 内容 | 绿的标准 |
| --- | --- | --- |
| **S0** | 落平台层骨架（protocol + vendor + discovery + registry + reference + hip 包装 + `vendors/` 三家声明）与门禁 `tests/test_backends_purity.py`（含当前违规白名单） | `test_structure`、`test_ops`、`test_backends`、`test_backends_purity` 全绿；HIP 路径行为不变 |
| **S1** | `hip_ops.py` → `backends/hip/__init__.py`，`kernels/` → `backends/hip/kernels/`；7 个生产 `.py` 改调 `backends.ops`；删顶层 `hip_ops.py` | 同 S0；`import hip_ops` 在 `backends/` 外为 0 |
| **S2** | `device="cuda"` → `backend().caps.device`；`ops/dtypes.py` 的 `D`/`WF` → `caps.act_dtype`/`caps.weight_dtype`（`D`/`WF` 保留为 `caps` 的再导出，避免 200 处改名） | 生产侧 `"cuda"` 字面量为 0（`diag/` 除外）；`test_quant`/`test_attention` 数值不变 |
| **S3** | 删 `models/` 里的 `hip_ops.available()` 分支，eager 数学搬进 `reference`；`ATTN_SPLIT` 迁到 `backends/hip/tuning.py`，由 `caps.attn_split` 驱动缓冲尺寸 | 全量 `unittest`（逐模块）绿；真实请求输出与迁移前逐位一致 |
| **S4** | HIP 专有探针 → `backends/hip/diag/`；`tools/push_*.py` 的远端路径同步；`engine_api._RELOAD_ORDER` 补齐 | 探针可跑；`check_style.py --style` 无误 |
| **S5** | 白名单收紧到空；加 `COFFIN_BACKEND=reference` 的 CPU 冒烟（无卡机器跑 `run_chat --no-graph` 与一致性用例） | 门禁 0 违规；CPU 冒烟绿 |
| **S6** | 某一厂商落地：`linear`/`rms_norm`/`attn_paged`/`delta` 四个热点 + 其余 reference；文档与验收 | 一致性用例通过；该卡上真实请求出文本 |

S1 之前所有工作都不改生产代码，所以"方案评审 → S0/S1"可以立刻开工，风险最低。

### 5.1 S0 落地后暴露的三个"必须在 S3 补上、否则性能/语义悄悄变差"的点

1. **按形状实测选核的 autotune 还没搬进 HIP 后端**：`backends/hip` 的 `_TUNING` 目前是空的，所有形状都取
   静态默认 `(R, U) = (1, 2)`。`ops/gemv.py::_pick` 里那套 `torch.cuda.Event` sweep 仍然在原地生效，
   所以**现在还不能删调用点**——S3 先把 sweep 迁进 `backends/hip`（由 `caps.autotune` 控制），再删旧路径。
   否则 decode 会悄悄掉几个百分点，而这种退化不会让任何测试变红。
2. **`Q35_FAST_GEMV` / `Q35_SPLIT8` 两个环境开关还没有后端归宿**：它们现在在 `models/qwen35/batch.py`，
   属于核选型而不是模型语义，S3 随 autotune 一起迁进 HIP 后端。
3. **`caps.attn_split` 已存在但无人读**：`hip` 报 8、其余后端报 1；`part_acc`/`part_ml` 的尺寸仍是
   `models/qwen35/tuning.py:ATTN_SPLIT` 写死的 8。S3 必须改成从 caps 取，否则换平台时 split-K 缓冲会把
   reference（split=1）写越界。

另外两处已知的语义待核对（S3 前必须与核对齐，不能只信 reference）：

- `attn_paged_multi_q` 的 `block_mask` 语义：reference 按 `block_mask[b]` 展开 `mask_block` 后按 query 截断/补齐，
  这是**从契约读出来的解释**，尚未与 `kernels/batch.cu` 对拍。若核按块边界向上取整，需要改 reference。
- `serving/` 的显存统计（`memory_allocated` / `mem_get_info`）没有胶囊：S2 需要一个后端可选钩子
  `backends.<name>.mem_report()`，否则这两个调用点只能留 `"cuda"` 分支。

---

## 6. 正确性门禁（这是整个重构的验收物）

| 层 | 用例 | 判据 |
| --- | --- | --- |
| 算子一致性 | `tests/conformance.py`，**按后端参数化**：每个算子在每个已注册后端上与 `reference` 对比 | 逐位或 ≤1e-6（fp32 参考）/ 容差表内（fp16 核） |
| 结构纯度 | `tests/test_backends_purity.py`：扫描 `backends/` 外的硬件字面量、跨层 import | 违规 0；白名单只减不增 |
| 回退完整性 | 每个后端对 17 个算子都必须给出实现（`None` 即失败） | `test_backends` 断言全量 |
| 模型回归 | 现有 `test_{structure,ops,attention,quant,mtp,paging,prefix_batch,batch_decode,engine,spec,vision,mrope,radix_mm,openai*}` | 全绿；文本请求逐位一致；decode ≥78 tok/s（gfx906） |
| 新平台接入 | 任一厂商上跑 conformance + 一次真实 chat | 出文本；`native` 数目与日志一致 |

一致性用例是**新平台的规格书**：以后接第三家，先让它通过 `conformance`，再谈性能。

---

## 7. 厂商接入清单（拿到机器后的顺序）

三家平台的布局都已登记在 `backends/vendors/`，全部 `has_kernels=False`。下面是把一家从占位变成
可用后端的顺序——三家通用，因为机制是同一个。

### 7.1 通用顺序

| 步 | 做什么 | 验收 |
| --- | --- | --- |
| 1 | 跑探针，把 `[待实测]` 的格填掉（命名空间、`get_device_name`、dtype、扩展路径、graph） | 事实表无空白 |
| 2 | 改 `SPEC`：`has_kernels=True`，写 `build_ops()` 但**先只返回一个算子** | `native=1/17` 出现在启动横幅 |
| 3 | 一致性用例：该算子 vs `reference` | 容差内 |
| 4 | 一次真实 chat（`COFFIN_BACKEND=<vendor> run_chat.py`） | 出文本 |
| 5 | 按热点顺序补算子：`linear` → `rms_norm` → `attn_paged` → `delta` → 其余 | `native` 逐项上升，每步都跑一致性用例 |
| 6 | 性能调优（autotune、graph、int8 权重） | TTFT / tok/s 达标 |

**第 2 步是关键**：一个算子就够跑通，因为其余 16 个由 reference 补。不要等"全实现完"再接。

### 7.2 各平台已知硬事实（决定内核怎么写）

| 平台 | 必须按这个写内核 | 出处 |
| --- | --- | --- |
| 天数智芯 | 权重用 `[K, N]` + `x @ Wt`（`weight_layout="kn"`）。`[N, K]` 行主序在本卡只有 19–30 GB/s，占 decode 时延 92%；`[K, N]` 有 362–410 GB/s | `reference/biv100_llm_engine/core/linear.py` |
| 天数智芯 | `N > 131072` 的 row-major-B GEMV **返回 NaN**（131072 好、196608 坏）。`lm_head` 正好超宽，必须分块投影 | 同上 `MAX_GEMV_N` |
| 天数智芯 | int8 KV 是 per-(token, head)：`round(x/s).clamp(-127,127)` + 每 (token, head) 一个 scale | `reference/biv100_llm_engine/cache/kv_cache.py` |
| 天数智芯 | `torch.cuda.Event` 可用，按形状 autotune 可以直接搬 | 同上 `diagnostics/perf_probe.py` |
| 燧原 | 索引必须 `int32`（`index_dtype=torch.int32`）：torch_gcu 无 I64，静默降到 32 位并告警；且 64 位张量的 `data_ptr()` 文档说"可能有误" | torch_gcu 用户指南 |
| 燧原 | `torch.cuda.*` 一律换成 `torch.gcu.*`；命名空间为 `gcu` | 同上 |
| 摩尔线程 | 命名空间为 `musa`，且 `torch.cuda` **故意不重定向**（`is_available()` 返回 False），所以平台探测是干净的 | torchada 文档 |
| 摩尔线程 | 设备名是 `MTT S4000` 这类，用 `MTT` 前缀识别 | `chitu` 的 `MOORE_DEVICE_PATTERNS` |

### 7.3 两个跨平台的设计约束（已写进 Caps）

- **`device_name` 不是日志字段**：在共用 `cuda` 命名空间里它是唯一区分依据。`Caps.device_name`
  的 docstring 已经把这点写死，防止后来者把探测改回 `torch.cuda.is_available()`。
- **`recurrent_state` 不是"有没有原生快照核"**：循环状态的快照/恢复是引擎侧
  `clone()`/`copy_()`，没有内核参与（`reference/biv100_llm_engine/cache/kv_cache.py` 就是这么做的）。
  新平台如果因为"我没写快照核"报 `False`，会白白丢掉前缀复用这个最大的 TTFT 收益。

---

## 8. 明确不做

- **不做运行时热切换后端**：一个进程一个后端，`COFFIN_BACKEND` 在导入期决定。
- **不做 entry-point 插件**：厂商平台是仓库内 `vendors/` 下的一等文件，用目录扫描发现即可。
  这不是"不做自动发现"——目录扫描已经做到了"加文件即生效"；只是不引入 Python entry-point
  那套安装期注册，因为它会引入一个当前不存在的部署步骤。
- **不留兼容 shim**：顶层 `hip_ops` 直接删除，`layers/`（已无人 import）删除。用户已拍板。
- **本期不合并 dense/paged 两套注意力**（见 §3.2 末尾），单独一轮。
- **本期不做视频 / 新平台的视觉塔优化**：`vision.py` 走 reference 就能跑，性能后置。

---

## 9. 待评审的决定（需要你点头的 6 条）

| # | 决定 | 备选与代价 |
| --- | --- | --- |
| Q1 | `D`/`WF` 从 `ops/dtypes.py` 搬进 `caps`，但在 `ops/dtypes.py` **保留同名再导出** | 备选：全量改名成 `caps.act_dtype`（200+ 处 import 要改）。再导出是"能力位是唯一来源、导入点不变"的折中，代价是模块名与内容略不符名 |
| Q2 | `ATTN_SPLIT` 迁到 `backends/hip/tuning.py`，由 `caps.attn_split` 驱动 `part_acc`/`part_ml` 的尺寸 | 备选：留在 `models/qwen35/tuning.py`。那等于承认"模型族知道这块卡的 split-K 宽度"，换平台必错 |
| Q3 | dense/paged 两套注意力本期**不合并** | 合并能减 1 个算子、少 2 个缓冲，但要动 `capture()` 与 `full_attn` 的缓冲布局，和平台迁移混在一起会掩盖回归 |
| Q4 | 删除 `layers/__init__.py`（全仓库无人 import，是对 `ops` 的兼容壳） | 保留则与"不留 shim"矛盾；删除无风险（grep 确认 0 引用） |
| Q5 | 各厂商第一期 `caps.graph=False`，decode 走 eager | 代价：decode 明显慢于 gfx906 的 graph 路径（gfx906 上 graph 是 3.1x）。收益：模型层零改动即可出文本；graph 支持单独一轮 |
| Q6 | S5 加"无卡 CPU 冒烟"（`COFFIN_BACKEND=reference` 跑 `run_chat --no-graph`） | 需要一条 `reference` 后端能跑的极小 checkpoint 或合成权重；好处是本地/CI 无需 DCU 就能验证结构改动 |

---

## 10. S0 已落地（2026-10-25）

方案里标为"骨架"的部分已经写完并在**无卡 Windows CPU 环境**跑通，可以直接评审代码。

| 交付物 | 内容 | 验证 |
| --- | --- | --- |
| `coffin/backends/protocol.py` | `Ops`（17 个可调用槽位，`from_mapping`/`as_dict`/`implemented`/`missing`）、`Caps`（14 项）、`Backend` 协议、`make_device` | `test_backends.py` |
| `coffin/backends/{vendor,discovery,registry}.py` | 声明机制（VendorSpec/Detection/SpecBackend）、目录发现、按 claim 强度选择 | `test_backends.py` |
| `coffin/backends/__init__.py` | 对外出口：`ops`、`device`、`caps()`、`backend()`、`reload()`、`status()` | 同上 |
| `coffin/backends/reference/` | 17 个算子全部纯 torch 实现（spec）；`build(caps)` / `fill(partial, caps)` 供原生后端补齐 | 48 个用例，含 17 个算子对拍 |
| `coffin/backends/hip/` | 现有 `hip_ops` 的契约适配：8 个投影入口 → `linear`/`linear_m`/`linear_prefill`；`hip_ops` **惰性 import**（CPU 机器上不会去设 ROCm 环境变量） | 需 DCU 才能跑原生路径 |
| `coffin/backends/vendors/` | 三家厂商声明：天数智芯（cuda+名字）、燧原（gcu+int32）、摩尔线程（musa）；全部 `has_kernels=False`，表由 reference 填满 | `test_backends.py` |
| `coffin/tests/test_backends.py` | 48 例：表/能力/注册表、reference 全量性、17 个算子 Oracle、依赖方向、子进程环境隔离 | `Ran 48 ... OK (skipped=1)` |
| `coffin/tests/test_backends_purity.py` | 5 条规则 + 分组预算 + `--report` / `--emit-allowlist` 两个入口 | `Ran 8 ... OK` |
| `.tmp_dsh/platforms/INVENTORY.md` | 逐文件迁移清单（与门禁同一次扫描生成） | 360 行 |

整体状态：`python -m unittest discover -s tests` → **233 例通过 / 58 skip**（无卡机器，skip 的都是 GPU 用例）；
`python ../diag/check_style.py --style backends tests/test_backends*.py` → **0 error / 0 warning**。
`git -C coffin status --short` 只有 `backends/`、两个测试文件、`PLATFORMS.md`、`.gitignore` 一行新增。

**S0 没做的事**（都是 S1–S3 的范围，见我上文的 5.1 缺口清单）：
- 生产调用点一个都没改（`hip_ops` 仍在 `ops/`、`models/`、`serving/` 里），所以现有行为与性能逐位不变；
- `hip_ops.py` / `kernels/` 没搬，`engine_api._RELOAD_ORDER` 没加后端模块；
- HIP 的按形状 autotune 与两个环境开关没迁，`caps.attn_split` 还没人读；
- `layers/` 没删。

---

## 11. 待你确认的下一步

1. **评审 `backends/` 的契约**（`protocol.py` 的 17 个名字 + 14 个 caps 字段）。这是后面所有工作的地基，
   改名字的成本随迁移步数上升。
2. **确认 S1 范围**：把 `hip_ops.py` + `kernels/` 搬进 `backends/hip/`，7 个生产文件改调 `backends.ops`，
   删顶层 `hip_ops`。这一步在 DCU 上必须跑一次全量回归（`test_ops`/`test_attention`/`test_quant`/`run_chat --bench`），
   我这边只能验到 CPU 结构与 reference 数值。
3. **确认"一个算子就能接"是否可接受**：拿到机器后先只接 `linear`，其余 16 个算子走 reference，先出文本再提速。

---

## 12. 现状扫描附录

数字口径以门禁 `tests/test_backends_purity.py` 为准（**违规行数**，一行只算一次；`prod` = 产品代码，
`probe` = `diag/` + `tests/`）。逐文件的机械替换清单在 `.tmp_dsh/platforms/INVENTORY.md`，
那里与门禁的 `ALLOWLISTS` 由同一次扫描生成；本节只保留结论。

门禁自带两个读法（S0 已实现）：

```bash
cd coffin
python tests/test_backends_purity.py --report            # 一屏看进度：prod 剩余才是迁移进度
python tests/test_backends_purity.py --emit-allowlist    # 树必须是绿的，否则拒绝生成
```

**`--emit-allowlist` 在红树上会直接拒绝**（S0 加固）：它把"现在扫到什么"写成新预算，如果在红树上跑，
等于把当前违规固化成新基线——这是白名单唯一能"长大"的路径。所以用法固定为：
先保证绿 → 改代码 → `--report` 读 `NEW`/`OVER`/`LOOSE`/`STALE` → 再 `--emit-allowlist` →
`git diff` 里 `ALLOWLISTS` 只允许数字变小或条目消失。

### 12.1 门禁基线（S0，当前树，门禁全绿）

```
R1_hip_ops_import: [probe: tracked=9 budget=9 files=3]  [prod: tracked=7 budget=7 files=7]  hits=28 exempt=12
R2_platform_literal: [probe: tracked=26 budget=26 files=6]  [prod: tracked=17 budget=17 files=9]  hits=58 exempt=15
R3_cuda_literal: [probe: tracked=435 budget=435 files=42]  [prod: tracked=181 budget=181 files=13]  hits=642 exempt=26
R4_kernels_path: [probe: tracked=2 budget=2 files=2]  [prod: tracked=7 budget=7 files=4]  hits=20 exempt=11
R5_backend_dependency: [prod: tracked=0 budget=0 files=0]  hits=0 exempt=0
```

| 规则 | 拦什么 | prod | probe | 终点 |
| --- | --- | ---: | ---: | --- |
| R1 | `import hip_ops` / `from hip_ops import`，外加 `import_module("hip_ops")` 与 `sys.modules["hip_ops"]` | 7 行 / 7 文件 | 9 行 / 3 文件 | prod 归零；`diag/` 到 S4 |
| R2 | `gfx906` / `--offload-arch` / `PYTORCH_ROCM_ARCH` / `TORCH_EXTENSIONS_DIR` | 17 行 / 9 文件 | 26 行 / 6 文件 | 归零 |
| R3 | `"cuda"` / `.cuda()` / `torch.cuda.*` | 181 行 / 13 文件 | 435 行 / 42 文件 | prod 归零 |
| R4 | `kernels/` 源码路径 | 7 行 / 4 文件 | 2 行 / 2 文件 | 归零 |
| R5 | `backends/` 反向 import `ops/models/engine/serving/cache`（绝对与相对两种写法） | 0 | — | 永远 0 |

合计 **prod 212 行 / 15 个文件**、probe 472 行 / 43 个文件。R3 的 181 行按用途拆：约 90% 是
`device="cuda"` 分配（→ `backends.device`），11 行是 autotune 计时（→ 后端内部），
17 行图捕获 + 8 行显存统计（→ `caps.graph` 守卫与后端可选 `mem_report()`）。

门禁的四种失败前缀就是"只减不增"的机械保证：`NEW`（新违规）、`OVER`（比预算多）、
`LOOSE`（比预算少，必须同一次提交里调低）、`STALE`（条目已失效，必须删）。

### 12.2 生产侧 `hip_ops` 属性引用 → 算子

prod 侧 75 处属性引用、25 个符号，归并到 17 个算子名：

| `hip_ops` 符号 | prod 调用点 | `backends.ops` | 说明 |
| --- | ---: | --- | --- |
| `available()` | 18 | `caps.native` | 不是算子；分支要删 |
| `last_error()` | 1 | `backend().error` | |
| `gemv` / `gemv_w8a16` | 2 / 2 | `linear` | 单行权重读；`R`/`U` 不再出现 |
| `gemv_m` / `gemv_w8a16_m` | 3 / 3 | `linear_m` | 小 M；M 分档由后端定 |
| `gemv_w8a16_wide*`（`getattr` 探测 5 处） | 0 直接调用 | `linear_m` | `_dbg` 版本留 HIP 后端内部 |
| `rms_norm` / `rms_norm_gated` | 1 / 1 | 同名 | |
| `attn` | 2 | `attn_decode` | dense 缓存路径 |
| `attn_paged` / `attn_paged_batch` / `attn_paged_multi_q` | 3 / 2 / 1 | 同名 | |
| `conv1d_silu` / `_seq` / `_batch` | 3 / 1 / 2 | 同名 | |
| `delta` / `delta_batch` | 4 / 2 | 同名 | |
| `delta_prologue` / `delta_prologue_batch` | 3 / 2 | 同名 | |
| `delta_prep_seq` | 1 | 同名 | |

涉及生产侧 **7 个 `.py` 文件**（`ops/{gemv,norm}.py`、`models/qwen35/{batch,full_attn,linear_attn}.py`、
`serving/{runtime_parts,runtime_transport}.py`），替换形态与工作量见 INVENTORY §8。

### 12.3 迁移时必须写进契约、否则会静默出错的 6 条

1. **int8 KV 布局是契约，不是后端自由**：`[nslots, nkv, hd]` int8 + `[nslots, nkv]` fp32 scale，
   `round(x/s)` clamp 到 `[-127, 127]`。后端必须接受这个布局，因为模型侧的 `paged_cache` 写的就是它。
2. **`caps.graph=False` 的语义是"允许 eager"**，不是"允许在默认流上假装捕获"——后者会让捕获出的图
   静默丢内核（`hip_ops` 的约束里记过这个坑）。同时要写明没有 `graph_pool_handle` 等价物时显存翻倍的代价。
3. **`caps.autotune=False` 是"后端给固定选路"**，不是"运行时测一组候选"；并且 `bench=False` 的构造路径
   必须保留，无卡测试依赖它跳过 sweep。
4. **`D`/`WF` 再导出只是省改名**：语义来源必须唯一是 `caps`，`ops/dtypes.py` 里不能再有第二个字面量。
5. **不是所有 `diag/` 探针都能平台无关化**：`probe_hw.py` / `probe_longctx.py` 测的就是这张卡，
   正确去向是 `backends/hip/diag/`，不要硬改成后端无关。
6. **`TORCH_EXTENSIONS_DIR` 必须按后端隔离**：共享目录会让并发布署抢同一个构建锁（已踩过，见 VISION §11.5）。

