# coffin — 推理引擎

面向国产加速卡的大模型推理引擎，分成两层：一个常驻的引擎进程持有模型和 KV，通过 unix socket 应答请求；外侧是命令行客户端与 OpenAI 兼容的 HTTP 适配器，两者都不持有模型。

已接入两家硬件：天垓100（Iluvatar CoreX，BI-V100，ISA `ivcore10`）和海光 DCU（gfx906 级，Z100 / Z200 同族）。没有加速卡的主机也能起来，但是后端会解析到纯 torch 的参考实现， 测试套件就跑在这条路径上。

目前已在天垓 100 实现 Qwen3.5 到 Qwen4 新模型的适配，实现了单卡 prefill 2000+ tps，decode 60+ tps，在海光 DCU 完成 Qwen3.5-Qwen3.8 新模型的适配。

## 快速测试与气动

```sh
cd coffin
export PYTHONPATH=$PWD MKL_THREADING_LAYER=GNU TORCH_EXTENSIONS_DIR=$HOME/torch_ext_coffin

# 1) 先确认这张卡还能出文本（单进程、无服务端、无分页缓存）
python3 run_chat.py --raw "The capital of France is" -n 32

# 2) 起常驻引擎，前台跑，Ctrl-C 或 stop 子命令结束
python3 -m entrypoints.cli.runtime --serve --model $HOME/models/Qwen3.5-4B

# 3) 另开一个终端，走同一条 socket 问它
python3 -m entrypoints.cli.runtime ping
python3 -m entrypoints.cli.runtime chat --user "用一句话介绍你自己" -n 64
```

三个环境变量不能省：`PYTHONPATH` 指向 `coffin/` 本身（`backends`、`serving`、
`entrypoints` 都是这里的顶层包）；`MKL_THREADING_LAYER=GNU` 是因为容器里的 torch 只带
`libmkl_gnu_thread.so`；`TORCH_EXTENSIONS_DIR` 是内核 JIT 缓存目录，**每个部署一个**。

## 1. 前置条件

### 依赖

| 项 | 说明 |
| --- | --- |
| Python | 3.10（集群上的 `py310` 环境） |
| `torch` | 跟卡走：海光用 ROCm 构建，天垓用 CoreX 构建，纯 CPU 用任意构建 |
| `tokenizers` | 读 checkpoint 的 `tokenizer.json`，`run_chat.py` 直接用它 |
| `safetensors` | 读权重 |
| `transformers` | 只服务进程用：`AutoTokenizer` / `AutoProcessor` 加载分词器与图像处理器 |
| `Pillow` / `numpy` | 只有多模态请求用（解码图片、算 mRoPE 位置） |

### 环境变量

| 变量 | 作用 |
| --- | --- |
| `PYTHONPATH` | 指向 `coffin/`；不设的话 `python3 -m entrypoints.cli.runtime` 只在 `coffin/` 目录里才找得到包 |
| `MKL_THREADING_LAYER=GNU` | 容器里 torch 只有 GNU 版 MKL 线程层，不设时任何脚本都在起来之后报符号错 |
| `TORCH_EXTENSIONS_DIR` | 内核 JIT 缓存。共享一个目录时，每个新目录都触发全量重编译并抢同一把构建锁，多人并发即死等 |
| `ENGINE_MODEL` | checkpoint 目录，`--model` 不写时取它 |
| `ENGINE_SOCK` | 引擎 socket，默认 `/tmp/qwen35.sock`；跑第二个模型必须换一个 |
| `COFFIN_BACKEND` | 钉住后端（`hip` / `corex` / `reference`），不设则自动探测 |
| `COFFIN_PARALLEL` | 多卡拓扑：`tp`（默认）/ `pp` / `ep` |
| `COFFIN_COREX_EXT` | 覆盖 CoreX 扩展名后缀，同一棵树里多人改内核时各自拥有独立构建目录 |


## 2. 单机测试：`run_chat.py`

改完内核之后第一个要跑的检查。它不启服务、不调度、不分页，只回答「这次构建还能不能出
文本、有多快」。

```sh
python3 run_chat.py --raw "The capital of France is" -n 32
python3 run_chat.py --chat "introduce yourself" --system "Keep it short" -n 64
python3 run_chat.py --bench -n 128                               # 固定 prompt，报 tok/s
python3 run_chat.py --raw "..." -n 32 --no-graph                 # 不重放设备图，跑即时路径
python3 run_chat.py --model $HOME/models/Qwen3.5-4B-int8-nk --int8 --raw "..." -n 32
```

三点要记住：

1. 解码是贪心的，同一个 prompt 每次都可复现；`--bench` 的数字因此能和早先的运行比。
2. 预量化目录（带 `quant_index.json` 的那种）**必须加 `--int8`**，否则加载器会把它当普通
   checkpoint 走反量化路径，稠密权重在卡上就是 OOM 而不是一句「用错了目录」。
3. `--max-seq` 只影响这一次冒烟（默认 4096），部署时的窗口由服务端的 `--max-seq` 决定。

## 3. 起常驻引擎

引擎进程独占模型和 socket，所有客户端都只是连上去。

```sh
python3 -m entrypoints.cli.runtime --serve \
    --model $HOME/models/Qwen3.5-4B \
    --sock /tmp/qwen35.sock \
    --max-seq 32768
```

起来之后日志里有四行，运维时只看这四行（字段值随部署而变）：

```
[serve] model Qwen3.5-4B (qwen3_5) at /home/x/models/Qwen3.5-4B
[serve] backend=corex device=cuda:0 native=N/M graph=1 autotune=1
[serve] weights 12.3s + build 4.1s  eos=[...]  int8=True  max_seq=32768  prefix_cache=paged (bs=64, snapshots=8)  batch=off  blocks=...  spec_k=0  max_seconds=none  vision=fp16 (...)  gpu_after_load=...
[serve] listening on /tmp/qwen35.sock
```

- 第一行给出 checkpoint 与架构族：架构从 `config.json` 读，不是从开关读，所以换模型只换
  `--model`。
- 第二行是后端自报的状态。`native` 是已生效的原生算子数与契约算子总数；其余走参考实现。
  出现你意料之外的 `backend=` 时，先怀疑卡和构建，不要怀疑模型。
- `gpu_after_load=` 是权重占用的显存，用来判断下一次 OOM 是谁吃掉的。

首次启动比之后慢：内核是懒编译的 `.cu`，编出来落在 `TORCH_EXTENSIONS_DIR`。换了缓存目录
或改了内核源码就得再付一次。

### 客户端命令

`sock` 之后跟子命令，不写就是 `ping`。

```sh
python3 -m entrypoints.cli.runtime ping
python3 -m entrypoints.cli.runtime generate --prompt "The capital of France is" -n 32
python3 -m entrypoints.cli.runtime chat --think --user "What is 2+2? Number only." -n 96
python3 -m entrypoints.cli.runtime chat --tools tools_demo.json --user "天气和 12*8"
python3 -m entrypoints.cli.runtime bench -n 128                    # 稳态 decode tok/s
python3 -m entrypoints.cli.runtime profile -n 64                   # 逐内核归因
python3 -m entrypoints.cli.runtime ops -n 64                       # 逐算子归因
python3 -m entrypoints.cli.runtime spec --spec-k 2 -n 64           # 投机解码对拍
python3 -m entrypoints.cli.runtime reload                          # 热重载，权重不重读
python3 -m entrypoints.cli.runtime reset
python3 -m entrypoints.cli.runtime stop
```

`profile` / `ops` / `bench` / `spec` 只在单流引擎上可用（`--max-num-seqs` 为 1），批引擎上它们直接报错。客户端调用建议自己套 `timeout`：引擎卡在集合通信上时，没有超时的调用会占住队尾。


## 4. OpenAI 兼容 HTTP

适配器是独立进程，把 HTTP/SSE 转成引擎 socket 调用；模型和 KV 都不在它这边。

```sh
python3 -m entrypoints.cli.openai \
    --sock /tmp/qwen35.sock \
    --host 0.0.0.0 --port 18080 \
    --served-model-name qwen35
```

它启动时先轮询 `ping` 拿引擎信息（模型 id、路径、有没有视觉能力），连续三次失败就退出
——带着空配置起 HTTP 服务只会把每个请求变成 500。可用的端点只有三个：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health`、`/healthz` | 转发 `ping`，引擎不通时回 503 |
| GET | `/v1/models` | 直接用缓存状态回答，不碰引擎 |
| POST | `/v1/chat/completions` | 非流式与 SSE 流式，带 `tool_calls` |

```sh
curl -s http://127.0.0.1:18080/health | python3 -m json.tool

curl -s http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen35","messages":[{"role":"user","content":"用一句话介绍你自己"}]}'
```

图片走 OpenAI 的 `image_url` data URL，服务端解码成 PIL 再喂视觉塔。像素上限是显存和延迟
的唯一真开关（默认 1 MP 约 1024 图像 token），见第 6 节的图像一组。

思考与工具调用跟 checkpoint 的对话模板走，不跟模型名走：qwen3.5（4B）和
occamy 1.0（MoE）共用 Qwen3.5 那一族的线格式，模板来源却不同——前者写在
`tokenizer_config.json`，后者是同目录的 `chat_template.jinja`，加载时统一补成
`tokenizer.chat_template`。

| 行为 | 契约 |
| --- | --- |
| 推理段 | 走 `reasoning_content`，与 `content` 同时流出；`usage.reasoning_tokens` 报落在推理段里的 token 数 |
| 工具调用 | `<tool_call>` 转成结构化 `tool_calls`，`finish_reason` 为 `tool_calls`；推理段里写出来的调用不算 |
| 参数类型 | 只认请求 `tools` 里声明的那一个：`{"type": "string"}` 的 `0755` 保持字符串，`integer` 才转数字 |
| 流式一致性 | 把 `function.arguments` 的片段拼起来，与非流式返回的字符串逐字相同 |

`--served-model-name` 不参与这些判断：同一个引擎既能服务 4B，也能服务 occamy 1.0，
换模型不需要改外壳。

## 5. 多卡

多卡的约定是环境变量，不是新入口：`RANK` / `WORLD_SIZE` / `MASTER_ADDR` /
`MASTER_PORT` 由 torch 的 `env://` 初始化读，`COFFIN_PARALLEL` 决定切分方式。

| `COFFIN_PARALLEL` | 切什么 | 生成期能不能并发多条请求 |
| --- | --- | --- |
| `tp`（默认） | 张量并行，每层跨卡归约 | 不能：每步都要 gloo 归约，集合顺序会被打断 |
| `pp` | 按层切分，段间走 UDP | 能 |
| `ep` | 专家并行，每卡持全部层、只切专家 | 能（跨卡归约已搬到设备 P2P） |

现成脚本在 `tools/`，做的是同一件事：挑健康卡、错峰启动各 rank、等 socket、跑一组请求、收工。

```sh
OCCAMY_WORLD=2 bash tools/launch_occamy_ep.sh     # 先验两卡
OCCAMY_WORLD=4 bash tools/launch_occamy_ep.sh
bash tools/launch_occamy_tp4.sh
bash tools/launch_occamy_pp.sh
```

`ep` 多两个前提：共享区必须在任何进程启动前建好（`COFFIN_EP_SHM`），因为 `EpShmReduce`
只映射、不 ftruncate；各 rank 要错峰 4 秒启动，否则一起抢构建锁，后面的 rank 撞上连接超时。
两条都写在 `tools/launch_occamy_ep.sh` 里。

`tp` / `pp` / `ep` 三者互斥；`pp`+`ep` 的形态只能是 PD 分离，不在这个入口里。

## 6. 开关怎么选

运行时开关集中在 `entrypoints/cli/runtime.py`，按用途分四组。

模型与显存

| 开关 | 默认 | 什么时候动它 |
| --- | --- | --- |
| `--model` | `$ENGINE_MODEL` | 换 checkpoint；架构由它的 `config.json` 决定 |
| `--max-seq` | 262144 | KV 池按窗口一次性分配，256k 大约 4.4 GB 显存（8 层 int8 全注意力） |
| `--fp16` | 关（即 int8 权重） | 多花约 4 GiB、decode 更慢，换约 0.3% 更好的困惑度 |
| `--num-blocks` | 0（按窗口推算） | 窗口拉长时显式调小；池是预分配的 |
| `--no-vision` | 关（有视觉塔就加载） | 省 0.62 GiB；之后图片请求回 503 |

前缀缓存与批

| 开关 | 默认 | 说明 |
| --- | --- | --- |
| `--no-prefix-cache` | 开 | 关掉分页基数缓存；agent 反复重发自己的对话时这是最贵的一项 |
| `--block-size` | 64 | KV 块粒度 |
| `--snapshots` / `--snapshot-interval` | 8 / 2048 | 循环态快照预算与间隔，快照点就是后续请求能复用的接点 |
| `--max-num-seqs` | 1 | 大于 1 才有批解码步；请求仍排队，调度器最多准入这么多 |
| `--prefill-quantum` | 2048 | 一个请求每步最多预填充多少 token，防止长 prompt 堵住短 prompt |

投机解码

| 开关 | 说明 |
| --- | --- |
| `--spec-k` | 每个投机步的草稿数，0 为关 |
| `--spec-batch` | 批引擎里也投机，约 0.79x 一个普通批 decode 步（输出一致） |
| `--spec-min-ctx` | 序列缓存够长才在批引擎里投机 |
| `--spec-sparse-k` | 近似的批验证，只留每行 top-k 历史块；**输出不再保证一致** |

图像

| 开关 | 默认 | 说明 |
| --- | --- | --- |
| `--image-max-pixels` | 1048576 | 超了就在处理器里降采样 |
| `--image-min-pixels` | 65536 | 放大后的最小尺寸，一张图至少 64 token |
| `--max-images` | 4 | 单请求图片数（四张 1 MP 已近 4k 图像 token） |
| `--max-image-bytes` | 8 MiB | 单图解码字节上限 |
| `--vision-int8` | 关 | 视觉塔二维权重压到 int8，按 GEMM 逐个解量化 |

后端与图

| 开关 | 说明 |
| --- | --- |
| `--backend` | 只在带 `--serve` 的同进程里生效 |
| `--no-graph` | 不重放捕获图、跑即时 decode。图回放不可信的后端只能这么选；声明不支持图的后端自动即时 |

OpenAI 适配器侧的开关与它对应：`--default-max-tokens`（默认 8192）、
`--max-decode-seconds`（默认 180 秒）、`--enable-thinking`、`--timeout`、`--max-images`、
`--max-image-bytes`、`--max-image-total-bytes`。引擎是单流且没有取消通道，这两个默认值
就是给「客户端不读了却不挂断」兜底的。

## 7. 量化产物

权重可以是原始 checkpoint（加载时量化），也可以是离线量化好的目录：后者省掉每次启动的量化开销，也让大模型不必在显存里先铺一份稠密副本，推荐使用 int8 量化进行推理部署。

```sh
python3 tools/quant_disk.py      --checkpoint $HOME/models/occamy-1.0 --out $HOME/models/occamy-1.0-W8A8-nk --include-vision
python3 tools/quant_disk.py      --checkpoint $HOME/models/Qwen3.5-4B --out $HOME/models/Qwen3.5-4B-int8-nk
```

产物目录里有一份 `quant_index.json`（格式 `coffin-int8-nk-v1`，权重布局 `nk`：盘上
`int8 [N, K]` + 每行一个 fp32 scale），加载时用 `--int8` 认它。`tools/launch_occamy_w8a8.sh`
是完整链路示例：挑健康卡 → 量化到磁盘 → 回读校验每个源张量都在产物里 → 打印产物大小。校验那一
步不要跳，漏张量的表现是权重看着齐了但输出不对。

## 10. 排障

| 症状 | 先看什么 |
| --- | --- |
| 任何脚本在 import 之后报符号错 | `MKL_THREADING_LAYER=GNU` 设了吗 |
| 启动卡在编译、进程停在 `kfd_wait_on_events` | 有没有两个部署共享同一个 `TORCH_EXTENSIONS_DIR` |
| 输出像是「模型不确定」 | `enable_paging()` 之后、`prefill()` 之前有没有 `assign_table` |
| OOM 但窗口不长 | 启动横幅里的 `gpu_after_load=` 和 KV 池谁占的 |
| HTTP 全 500 | 引擎 socket 通不通：`python3 -m entrypoints.cli.runtime --sock <path> ping` |
| 慢十倍 | 启动横幅第二行是不是 `backend=reference` |
| 权重加载后立刻 OOM | 预量化目录漏了 `--int8`，走了反量化路径 |
