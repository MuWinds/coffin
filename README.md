# coffin/ — 推理引擎本体

设计取舍见 `NOTES.md`（逐项优化流水账），契约见 `ARCH_BATCH.md`。
这里只记**目录约定**，方便以后接第二个模型。

> 说明：本目录是从仓库根目录整体搬进来的引擎代码（`git mv`，历史保留）。
> 旧文档 `ENGINE.md` / `NOTES.md` / `ARCH_BATCH.md` 早前被误放到仓库根目录后丢失，
> 需要时可按 `coffin/README.md` 的布局表重建。

## 布局

| 路径 | 职责 |
| --- | --- |
| `config.py` | `EngineConfig`：服务/引擎的全部开关（窗口、块大小、批容量、快照预算） |
| `sequence.py` | `Sequence`（一个请求的状态机 + block table + 批内行号）、`SamplingParams` |
| `scheduler.py` | 连续批调度：每步最多 admit 一个 prefill，同时解码所有 running；超时退休 |
| `block_manager.py` | 块池 + 引用计数（可选 radix 树）：`ensure_blocks` / `release` / `assign` |
| `model_runner.py` | 一步怎么执行：单流 `decode_step` 或静态批 `decode`（行 / 桶 / 采样） |
| `llm_engine.py` | `LLMEngine`：把 model + runner + scheduler + block manager 装起来，一步一步驱动 |
| `engine_api.py` | 服务侧唯一入口：`Model` / `PagedModel` 协议、`create_model()`（按 `config.json` 的 `model_type` / `architectures` 分发）、`load_weights()`、热重载 `reload()` |
| `ops/` | 与模型族无关的设备算子：`dtypes.py`、`gemv.py`、`embed.py`、`norm.py`、`rope.py`、`delta.py` |
| `models/qwen35/` | Qwen3.5-4B 一族：`model.py`、`blocks.py`、`batch.py`（批投影）、`linear_attn.py`、`full_attn.py`、`mtp.py`、`weights.py`、`tuning.py` |
| `qwen35_engine.py` | 旧扁平导入的兼容壳，`diag/` 的探针还在用；新代码别用 |
| `hip_ops.py` + `kernels/` | HIP 内核的 host 薄壳与 `.cu` 源码（`batch.cu` 是批核） |
| `paged_cache.py` / `radix_cache.py` / `prefix_engine.py` | 分页 KV 与 radix 前缀复用（模型无关） |
| `serve.py` / `openai_server.py` | 常驻服务（engine 线程 + 每连接一个线程）与 OpenAI 兼容外壳 |
| `tool_call.py` / `agent.py` / `run_chat.py` | tool-call 解析、工具循环、单机入口 |
| `diag/` | 归因 / 带宽探针与 benchmark（profile、AB、数值对比、`bench_batch.py` 批吞吐） |
| `tests/` | 功能回归（unittest）：结构、算子、attention、量化、投机、MTP、分页、批 decode、引擎、OpenAI 协议 |

## 依赖方向

```
ops  ←  models/*  ←  engine_api  ←  serve.py / run_chat.py / prefix_engine.py
                 ↖  config / sequence / scheduler / block_manager / model_runner
                    ↖  llm_engine  ←  serve.py
```

- `ops/` 不许 import `models/`，模型族可以 import `ops/`。
- 同一族内的模块互相 import；跨族只通过 `engine_api`。
- 只有 `engine_api.py` 知道“有哪些模型族”。
- `sequence.py` / `config.py` 不 import 其它引擎模块；`llm_engine.py` 是唯一把
  runner、scheduler、block manager 装到一起的地方。

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
2. 层里带递推态的（像这里的 linear attention），再加 `PagedModel` 的
   `enable_paging` / `snapshot_linear_states` / `restore_linear_states`；
3. 在 `engine_api.py` 末尾 `register((...), ModelClass, load_weights)` 一行。

`serve.py`、`prefix_engine.py`、`openai_server.py` 都不用动。

## 部署默认值

| 变量 / 参数 | 作用 |
| --- | --- |
| `ENGINE_MODEL` / `--model` | checkpoint 目录；架构由它的 `config.json` 决定，启动日志会打印 `模型名 (架构族) at 路径` |
| `ENGINE_SOCK` / `--sock` | 引擎 socket，默认 `/tmp/qwen35.sock`（旧路径保持兼容）；跑第二个模型时换一个 |
| `--served-model-name` | `openai_server.py` 的 `/v1/models` id；默认取引擎报告的目录名 |

## 改完怎么验

```sh
# 容器里：结构用例，不读权重、不建图，几秒钟
cd /root/private_data/engine
MKL_THREADING_LAYER=GNU PYTHONPATH=$PWD \
    python3 -m unittest discover -s tests -p test_structure.py -v

# 整套功能回归（会读权重、建图，约 3~5 分钟）
MKL_THREADING_LAYER=GNU PYTHONPATH=$PWD python3 -m unittest discover -s tests -v

# 再跑一次真实请求
python3 run_chat.py --bench -n 32
```

推文件：`python tools/push_engine.py --verify`（整个 `engine/` 的代码，不含
`diag/` 与权重；要连 diag 一起推就加 `--with-diag`）。
