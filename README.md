# coffin/ — 推理引擎本体

这里只记**目录约定**，方便以后接第二个模型。设计取舍与契约写在代码里：
每个模块的文件头讲清职责与硬约束，`AGENTS.md` 规定注释与 docstring 的格式。

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
