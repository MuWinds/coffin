# AGENTS.md — coffin 注释与提交规范

本仓库的注释契约。改动任何注释之前先读完本文件。
`coffin/` 是独立 git 仓库，所以本文件就是它的顶层规则。
适用范围：`coffin/` 下所有 `.py`、`.cu`、`.c`、`.h`、`.cpp`、`.md`。

---

## 0. 目录约定
## 布局

| 路径 | 职责 |
| --- | --- |
| `engine/` | 请求状态、调度、执行、块所有权和 `LLMEngine`；按运行时职责拆分，避免入口脚本承载引擎状态 |
| `cache/` | 分页 KV 与 radix 前缀缓存；模型无关的存储与复用 |
| `utils/` | 多模态 prompt 元数据和 tool-call 解析等叶子辅助模块 |
| `layers/` | 稳定的层/算子导出面；底层实现暂时由 `ops/` 和 `models/` 保持兼容 |
| `engine_api.py` | 服务侧唯一入口：`Model` / `PagedModel` 协议、`create_model()`（按 `config.json` 的 `model_type` / `architectures` 分发）、`load_weights()`、热重载 `reload()` |
| `ops/` | 与模型族无关的设备算子：`dtypes.py`、`gemv.py`、`embed.py`、`norm.py`、`rope.py`、`delta.py` |
| `models/qwen35/` | Qwen3.5-4B 一族：`model.py`、`blocks.py`、`batch.py`（批投影）、`linear_attn.py`、`full_attn.py`、`mtp.py`、`weights.py`、`tuning.py` |
| `hip_ops.py` + `kernels/` | HIP 内核的 host 薄壳与 `.cu` 源码（`batch.cu` 是批核） |
| `serving/` | 常驻引擎服务：`runtime_service.py`、运行时 mixin、socket 传输；不承载 HTTP 协议和 CLI 解析器 |
| `entrypoints/openai/` | OpenAI HTTP/SSE 适配：`app.py`、请求规范化、流式 tool-call、引擎 socket 客户端 |
| `entrypoints/cli/` | 命令行入口：`runtime.py` 启动/控制引擎，`openai.py` 启动 HTTP 适配器 |
| `utils/tool_call.py` / `agent.py` / `run_chat.py` | tool-call 解析、工具循环、单机入口 |
| `diag/` | 归因 / 带宽探针与 benchmark（profile、AB、数值对比、`bench_batch.py` 批吞吐） |
| `tests/` | 引擎自身要素的功能回归（unittest）：包结构与模型族注册、分块预填与抢占调度、基数前缀与分页块生命周期、OpenAI 协议、后端选择与后端纯度门禁 |

## 依赖方向

```
ops / layers  ←  models/*  ←  engine_api  ←  serving/
cache / utils ←  engine/*  ←  serving/ ← entrypoints/cli/
                                      ↖ entrypoints/openai/
```

- `ops/` 不许 import `models/`，模型族可以 import `ops/`。
- 同一族内的模块互相 import；跨族只通过 `engine_api`。
- 只有 `engine_api.py` 知道“有哪些模型族”。
- `engine/sequence.py` / `engine/config.py` 不 import 其它引擎模块；
  `engine/llm_engine.py` 是唯一把 runner、scheduler、块管理器装到一起的地方。

## 批处理的行布局

静态批执行时模型有 `capacity = max_bucket + 1` 行：

| 行 | 用途 |
| --- | --- |
| `0 .. capacity-2` | decode 行；一个序列在 admit 时分到一行，运行期间固定，结束释放 |
| `capacity-1` | prefill 暂存行（`prefill_row`）。prefill 写这里，再把递推态搬到该序列的 decode 行 |

一行空洞（中间的序列先结束）不搬状态：每步按“最高活跃行”选桶，`decode_batch`
用哨兵行把空洞补齐（`pad_rows`），所以状态永远在原来的行里。

模块内部一律用绝对 import（`from ops.gemv import GEMV`），因为入口脚本把
`engine/` 目录本身放进 `sys.path`，`ops` / `models` 是和 `hip_ops` 平级的顶层包。

## 加一个模型族

1. 新建 `models/<family>/`，实现 `engine_api.Model`：`reset` / `prefill` /
   `decode_step` / `logits` / `capture` / `graph_ready` / `capture_verify` /
   `spec_generate` / `spec_stats` / `close`，以及 `max_seq`、`pos1`、`verify_c`、
   `mtp` 四个属性；
2. 层里带递推态的（像这里的线性注意力），再加 `PagedModel` 的
   `enable_paging` / `snapshot_linear_states` / `restore_linear_states`；
3. 在 `engine_api.py` 末尾 `register((...), ModelClass, load_weights)` 一行。

模型族接入不需要修改 `serving/` 或 `entrypoints/`；运行时从
`python -m entrypoints.cli.runtime --serve` 启动，OpenAI 适配器从
`python -m entrypoints.cli.openai` 启动。

## 部署默认值

| 变量 / 参数 | 作用 |
| --- | --- |
| `ENGINE_MODEL` / `--model` | checkpoint 目录；架构由它的 `config.json` 决定，启动日志会打印 `模型名 (架构族) at 路径` |
| `ENGINE_SOCK` / `--sock` | 引擎 socket，默认 `/tmp/qwen35.sock`；跑第二个模型时换一个 |
| `--served-model-name` | OpenAI 适配器的 `/v1/models` id；默认取引擎报告的目录名 |

## 改完怎么验

```sh
# 容器里：整套引擎回归。不读权重、不建图、不要求加速卡，十几秒
cd /root/private_data/engine
MKL_THREADING_LAYER=GNU PYTHONPATH=$PWD python3 -m unittest discover -s tests -v

# 再跑一次真实请求
python3 run_chat.py --bench -n 32
```

`tests/` 只守引擎自身、与具体模型和硬件都无关的要素。接入的模型和硬件会一直
增加，模型级数值、算子级等价、量化格式、视觉塔这类回归按 `AGENTS.md` 第 10 条
写在临时目录里跑，不回到这个目录——写在这里的用例会随着模型和硬件的替换一起过时。

推文件：`python platform/scow/tools/sync.py push -d coffin --root-name coffin`
（增量 tar，只传改动的文件；远端常驻检出是 `~/coffin-work/base/coffin`）。
不传权重、不传 `diag/`。原来那个 `coffin/tools/push_engine.py` 已经不在仓库里。

## 多模态（图片）

设计见仓库根的 [`docs/VISION.md`](../docs/VISION.md) §2。落地位置：

| 文件 | 职责 |
| --- | --- |
| `utils/mm_input.py` | `MMInput`（图片 embeds / grid / 3-D mRoPE positions / `rope_delta` / digest / spans）、`mrope_positions`、`find_image_spans`、`image_digest` |
| `models/qwen35/vision.py` | `VisionTower`：patch_embed / pos_embed 重采样 / axial RoPE / 分块非因果注意力 / 24 block / merger；fp16 或 int8 |
| `ops/mrope.py` | `mrope_cos_sin`（stride-3 交错，`[11,11,10]`）、`scatter_image_embeds`（绝对下标注入） |
| `models/qwen35/model.py` | `enable_vision` / `image_embeds` / `prefill(..., mm=)` / `prefill_group` 读 `seq.mm` / `rope_delta` |
| `cache/radix_cache.py` | block 级媒体签名（图片指纹），纯文本 block 零开销 |
| `serving/runtime_service.py` / `entrypoints/openai/app.py` | `image_url` data URL → PIL → `Qwen3VLProcessor` → `MMInput` |

新增开关：`--no-vision`、`--vision-int8`、`--image-max-pixels`（默认 1 MP）、
`--image-min-pixels`、`--max-images`（默认 4）、`--max-image-bytes`。

两条必须遵守的契约（写错会表现为“模型不确定”或“卡死”）：

1. **`enable_paging()` 之后、`capture()`/`prefill()` 之前必须 `assign_table`**
   （用 `PrefixCacheEngine` 就自动满足）。否则 `pos2slot` 全零，所有位置映射到同一 slot。
2. **`TORCH_EXTENSIONS_DIR` 按部署目录隔离**（如 `/root/private_data/torch_ext_canon`）。
   共享目录时每个新目录都会触发 HIP 全量重编译并抢同一个构建锁，多人并发即死等。

引擎侧与模型无关的那一半：`test_radix_mm.py`（带媒体签名的基数键）与
`test_openai_mm.py`（OpenAI 图片入口）。视觉塔本身的数值回归按 `AGENTS.md`
第 10 条写在临时目录，跟着模型一起换。


## 1. 语言：一律中文

- 所有注释、docstring、文件头、文档正文一律用**中文**书写。
- 不保留成段的英文散文。英文只允许出现在下面这些「原样保留」的内容里。
- 原样保留（不翻译、不改写）：
  - 标识符、函数名、类名、模块路径
  - 张量形状 `[B, S, H]`、数值、单位、实测数字
  - 文件路径、URL、命令行开关、环境变量名
  - 协议字段名（如 OpenAI wire format 的 `tool_calls`）
  - 结构标记：`Args:` / `Returns:` / `Raises:` / `Module:` / `Constraints:`
    / `@file` / `@brief` / `@param` / `@return` / `@throws` / `#:`
  - 工具指令：`# noqa: E402`、`# type: ignore[...]`、`# pragma: no cover`
- 工具指令要在上面单独一行用中文解释它为什么在那里，指令本身绝不改写。
- 代码块、行内反引号、引号里的样例数据可以保留原语言（例如一段英文
  prompt）。门禁按「散文」判定，样例不算散文。

---

## 2. 写「为什么」，不写「是什么」

自检标准：**删掉这条注释，读者会不会因此改错代码？**
如果不会，这条注释就是噪音，删掉。

见到就删：

```python
i += 1                  # i 加一
for x in xs:            # 遍历列表
self.n = 0              # 初始化 n
return out              # 返回结果
import torch            # 导入 torch
if not xs:              # 判断是否为空
```

该保留，因为代码本身回答不了：

```python
# 旧版 API 的分页是从 1 开始的；这个 +1 让两处调用点的语义对齐。
offset = page * size + 1

# gfx906 只有 64 KB LDS，超过 48 KB 的动态分配需要额外声明属性，
# 所以关键窗口用流式读取而不是整块暂存。
for blk in window:
    ...

# 131072 是这张卡上仍然正确的行主序-B GEMV 最宽尺寸；
# 196608 实测会返回 NaN。
if n > 131072:
    ...

# 把缓存实体化成 bf16 要多两次 [nkv, S, hd] 拷贝。保持 int8、在内核里
# 反量化，用几条 ALU 指令换掉 1 GB+ 显存。
```

一条注释值得占一行，至少要回答下列之一：

1. 为什么必须这样写，不这样写会坏在哪（NaN、显存翻倍、缓存别名、协议不匹配）？
2. 哪个外部契约迫使这样写（旧版 API、OpenAI 报文格式、checkpoint 布局）？
3. 哪个硬件限制决定了这个形状（LDS、带宽、wavefront、INT8 行主序权重）？
4. 这个魔数从哪来，实测过什么？
5. 放弃了什么（取舍是什么），为什么可以接受？

---

## 3. 标准 docstring 格式

| 语言 | 风格 | 小节标记（保持英文） |
| --- | --- | --- |
| Python | **Google Style** | `Args:` / `Returns:` / `Raises:` |
| CUDA / C / C++ | **Doxygen** | `@brief` / `@param` / `@return` / `@throws` |

- 不用 Sphinx 的 `:param:` / `:returns:` / `:raises:`（`reference/` 里随源码
  引入的第三方包保持它自己的风格，那不属于本仓库）。
- 不用 NumPy 的 `Parameters` / `----------` 写法。
- 小节顺序固定：`Args:` → `Returns:` → `Raises:`。某一段为空就整段省略。
- 首行是一句话摘要，以句号结尾。空一行。然后才是细节和小节。
- 常量和 dataclass 字段仍然用 `#:` 前缀。

```python
def match_prefix(self, tokens: list[int], require_states: bool = False) -> tuple[int, int]:
    """在基数树里找出 ``tokens`` 最长的已缓存前缀。

    只返回按块对齐的长度：KV 以整块为粒度共享，半块的命中会在块表里留下
    一个不完整的块，而写时复制无法为它维护引用计数。

    Args:
        tokens: 完整请求 token（prompt 加已生成部分），不能为空。
        require_states: 要求命中点处有线性注意力状态快照。混合模型上只有
            KV 命中是错误的，那会把共享的 key 配上一个从没见过它们的状态，
            所以命中会被截断到有快照的最深边界。

    Returns:
        ``(prefix_len, node_id)``。``prefix_len == 0`` 表示没有命中，
        此时 ``node_id`` 为 ``-1``。

    Raises:
        ValueError: ``tokens`` 为空。空请求应该在调用方就拒绝，这里返回
            一个零长度命中只会把 bug 藏起来。
    """
```

```cpp
// @brief 对 INT8 KV 做单 token 全注意力，单次遍历，online softmax。
//
// @param q        查询向量，[heads, head_dim]，已经乘过 1/sqrt(d)。
// @param k_cache  分页池的 INT8 K 视图，通过 slot map 寻址。
// @param out      输出缓冲区，[heads, head_dim]，连续存放。
// @return 无。错误通过 host 侧的 hipGetLastError() 暴露。
// @throws 无；HIP 以错误码上报，非法参数由 host 侧断言拦下。
```

详细程度：

| 目标 | 最低要求 |
| --- | --- |
| 有副作用、会抛异常或参数不直观的公开函数 | 四段齐全 |
| 简单的判定函数或 getter | 一句话摘要，说清它存在的理由 |
| 私有辅助函数 | 目的加关键约束 |
| 测试用例 | 不写 `Args` / `Returns`。说明它守的是什么不变量、为什么重要 |

`Returns:` 描述语义和边界值，不描述类型——类型在签名里。
`Raises:` 说明触发条件以及调用方该怎么处理；只抛不解释的不算文档。

---

## 4. 文件头

每个源文件开头必须讲清三件事。

### Python

```python
"""一句话职责：这个文件为什么存在。

Module: ``coffin/<包名>``。它在依赖图里的位置——谁调用它，它允许调用谁。

Constraints:
    * 覆盖整个文件的一条硬规则，附上原因（硬件、精度、不变量、导入约定）。
    * 如果还有第二条就继续写。
"""
```

### CUDA / C / C++

```cpp
// @file attn.cu
// @brief 对 INT8 KV 缓存做融合的单 token 全注意力。
//
// Module: coffin/kernels。由 hip_ops.py 作为 torch 扩展加载。
//
// Constraints:
//   * 每个 CU 的 LDS 预算是 64 KB；关键窗口流式读取，绝不整块暂存。
//   * 每个查询头一个 block；不存在跨 block 同步。
```

- **职责**是这个文件解决的问题，不是它的函数清单。
- **Module** 写清这里适用的依赖方向（例如 `ops/` 不许 import `models/`）。
- **Constraints** 是覆盖整个文件的规则，不是单个函数的备注。典型的有：
  入口脚本把 `coffin/` 加进 `sys.path`，所以用绝对导入；权重是 INT8 行主序，
  新内核必须接受 `[K, N]`；线性注意力的循环状态必须和 KV 一起打快照，否则
  前缀复用会算错；不做加锁，所有变更由 `llm_engine` 单线程驱动。

---

## 5. 内联注释与块注释

1. 不直观的算法、业务黑魔法和魔数必须有注释点名取舍。宁可写实测数字，
   也不要写形容词：`3.1 us/token at 64k, ~24 GB/s of KV traffic against a
   ~600 GB/s roof` 比「这样更快」有用得多。
2. 标记后一个空格：`# 正文`、`// 正文`。行尾注释前两个空格：
   `x = 1  # 原因`。
3. 语句级注释与它上面的代码之间空一行，免得读起来像上一句的补充说明。
4. 块注释紧贴它解释的代码，描述这段代码达成什么，而不是逐行翻译。
5. 小节分隔只用朴素的 `# 小节名`。**禁止 ASCII 分隔线**——不许 `# ----`、
   `# ====`、`# ****`。已有的 `# -- 子系统 ---` 按这条规则删除。
6. 不留注释掉的死代码。如果某条分支因为硬件限制被关掉，说明为什么保留它、
   什么条件下能重新打开。

---

## 6. 文档（`.md`）规则

- 正文一律中文。标题、正文、表格、列表、说明文字都翻译。
- 原样保留：代码块里的代码、命令、路径、URL、字段名、实测数字、表格里的
  标识符。
- 标题层级、锚点、链接目标不变。内部链接指向的文件名不改。
- 不删内容、不合并章节、不改变结论。这是翻译与改写，不是重新创作。
- 术语按 `coffin/GLOSSARY.md` 统一，同一概念全书只用一种译法。

---

## 7. 重构纪律

- **只改注释和散文。** 任何代码 token、字符串字面量、数字、标识符、import
  顺序、语句都不许变。结果必须语义完全等价。
- 不要一次性重写整个大文件。按函数规模分批改，每批之后跑一次等价性检查。
- 删噪音，不要凑篇幅。宁可少而准，也不要为覆盖率补废话。
- 避免生成式中文的腔调：「需要注意的是」「值得注意的是」「该函数负责」
  「这里我们」「综上所述」「起到了……的作用」「简单来说」。陈述事实，
  然后结束。
- 不要为了清掉一条警告给一行的小函数硬补四段式——那正是第 7 条禁止的
  凑覆盖率。

---

## 8. 检查

改完文件之后：

```bash
cd coffin

# 每个文件仍然能编译
python -m py_compile $(find . -name '*.py' -not -path './__pycache__/*')
```

只改了注释时，按 §7 的纪律要求「代码 token 一字不动」，但**目前没有自动门禁**：
原先做 AST 等价、剥注释 token 流等价和风格检查的 `diag/check_style.py` 已随
`diag/` 目录一起删除，仓库里没有替代品。所以这一段靠 `git diff` 人工确认——
改动只允许落在注释与 docstring 上，`+`/`-` 行里出现代码 token 就是改错了。

要恢复自动门禁就得先把那个检查器写回来；在那之前，别在文档里把它当成能跑的命令。

单文件检查清单：

1. 文件头写清了职责、模块位置、约束。
2. 公开函数有 Google/Doxygen 小节，`Args:` 覆盖每个参数，`Raises:` 说明触发条件。
3. 每一条留下的注释都能通过第 2 节的删除测试。
4. 没有复述代码的注释，没有英文散文，没有 ASCII 分隔线。
5. 标记后有空格的排版，语句级注释前有空行。

---

## 9. 提交规范

本仓库所有提交信息一律用中文。

### 格式

```
<类型>: <一句话说清做了什么>

<可选正文：为什么这么做、放弃了什么、实测数字>
```

- 首行不超过 72 个字符，结尾不加句号。
- 首行之后空一行，再写正文。
- 正文每行不超过 80 个字符。
- 类型标记保留英文小写，后面跟中文冒号加空格。

### 类型

| 类型 | 用途 |
| --- | --- |
| `feat` | 新增能力 |
| `fix` | 修 bug |
| `perf` | 只改性能，行为不变 |
| `refactor` | 只改结构，行为不变 |
| `docs` | 只改文档和注释 |
| `test` | 只改测试 |
| `chore` | 构建、依赖、杂务 |
| `repo` | 仓库结构调整（移动、拆分） |

范围可以加在类型后面：`corex: 修 split-K 派发` 里的 `corex` 是范围。
范围本身保持英文（它是目录名或平台名），冒号后面的描述用中文。

### 示例

```
docs: 把引擎注释统一改写为中文

术语按 GLOSSARY.md 统一，结构标记（Args:/@brief 等）保持英文，
代码 token 一字未动，diff 里只有注释行。
```

```
perf: 融合 decode 步骤的胶水算子，23.53 -> 20.26 ms（1.16x）
```

### 禁止

- 英文提交信息。
- `update`、`fix bug`、`修改`、`.` 这类没有信息的首行。
- 把多个无关改动塞进一个提交。
- 在首行使用句号或感叹号。

---

## 10. 测试规范
本仓库做的一切测试都只跟之前的已有数据去比较，只测试因修改产生的运算结果正确性/性能数据部分，不要写 A/B 测试
面向新的硬件、新的模型、新的算子的测试一律只写在临时目录，因为对接的模型和硬件越来越多，没法做到全放在目录下测正确性。
`tests/` 只保留引擎自身、与具体模型和硬件无关的要素（OpenAI 协议、基数前缀、调度、注册表、后端抽象）。模型级数值、算子等价、量化格式、视觉塔这类回归不要提交进这个目录。
请当前任务一切完备后再去做 style 风格检查测试，不要中途检查。