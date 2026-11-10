# occamy-1.0 四卡 W8A8 服务：落地记录

本文件在 `coffin/docs/occamy/` 下并纳入 git —— 早前放在仓库根的 `docs/` 里、既未跟踪也没进同步链，
2026-11-10 被某次清理连带删掉，整份丢失，所以迁到这里重建。

## 1. 结论

`Accio-Lab/occamy-1.0`（`Qwen3_5MoeForConditionalGeneration`，70.24 GB bf16）在 4 张 BI-V100（32 GiB/张，
PCIe Gen4 x16、无 NVLink）上以 **W8A8 int8 + 层切分管线（PP4）** 跑通，并能正常吐字：

| 指标 | 数值 |
| --- | --- |
| 权重加载 | **2.5 s**（起点 466-509 s） |
| decode（5/23/14/2001 tok 四档） | **26.41 / 25.51 / 25.52 / 23.29 tok/s** |
| prefill | 611 / 919 / 559 ms；2001 tok **2332 ms** |
| 每卡显存 | 9.23 GiB / 32 GiB |
| 原生算子 | `native=34/34`，`graph=0` |

## 2. 关键取舍

- **W4A8 停用，改 W8A8**：W4 的按组 scale 在这套内核上收益不抵复杂度；int8 的逐 (专家, 列) scale
  让 K 方向分片可以先把整数和相加、最后一次乘 scale（部分和精确到 int32）。
- **A8 激活量化**：逐行 absmax/127，对 fp32 基准 rel **1.9e-2**（decode 一直这个口径）。
- **PP4 是为「装得下」选的**，不是为延迟：PP 四段严格串行，等价于 40 层相加。

## 3. 三条会让四卡方案静默失败的坑

1. **扩展编译失败会静默回退 reference**：`backends.status()` 那行的 `hip: absent ... reference: usable`
   就是回退标志；现象是所有数字看起来正常但其实是纯 torch 在跑。新增内核必须递增
   `backends/corex/kernels/__init__.py` 的版本号（否则构建缓存复用旧 `.so`）。
2. **禁 CUDA graph、禁 cp.async**：这张卡会静默挂死（参考仓库 `patch_ops.sh` 记录 Triton 计算核永久挂死）。
3. **四 rank 同启动会撞扩展构建的 `lock` 文件**（`kernel build failed: FileNotFoundError .../lock`），
   其余 rank 干等导致 relay 超时 ⇒ **各 rank 之间 `sleep 4` 错峰**。

## 4. 启动耗时：不是 I/O，是张量数

466-509 s 的加载里没有 I/O 瓶颈，是逐张量构造与设备搬运的次数问题。改成按栈加载 + 预分配后降到 2.5 s。

## 5. 四卡落地：PP4

- 每卡 10 层，`layer_begin/end` 由 `parallel.layer_range()` 决定；首段留 `embed`，
  **每个 rank 都留 `norm`+`lm_head`**（head 复制，见 §8.8）。
- `_forward` 的管线尾巴：中间段把 `[1,hidden]` 交给下一段；**末段把 hidden 回给各段、各段自己算 logits**
  （实测比「末段算完广播 1 MB logits」快 22%）。
- relay 走 UDP（类型字节 + 收件箱 + `/dev/shm` 大块），单跳约 0.9 ms。

## 6. 服务路径上的四个坑

1. 加载器缺 `q8-moe` 分支 ⇒ 专家栈被当稠密权重加载（报张量形状不匹配）。
2. `SparseMoE._reduce` 在 PP 下不能做 gloo 归约（`Connection closed by peer`）⇒ 只在 `mode=="tp"` 时归约。
3. 探针两端的圈数必须一致，否则一方 send 完等 ack、另一方从没 post recv ⇒ 双端挂死。
4. UDP 多分块会失步：`_await_ack` 静默丢弃其它对端的报文 ⇒ 改成带类型字节的收件箱 + 非匹配报文回插队首。

## 7. W8A8 的内核与收益

- `moe_grouped_i8`（分组 int8 投影，`[E,K,N]` + 逐 (专家,列) scale）+ `moe_grouped_i8_merge`
  （decode 时把 top_k 行按路由权重加权累加成一行，省掉 `[top_k,H]` 激活与三次发射）。
- `moe_grouped_i8_merge` 单次 48.5 µs 是单入口最贵的之一；`moe_dispatch` 40.6 µs 次之。

## 8. 负结果（都带数字，别重做）

### 8.1 输出头收回末段：-18%

「末段算完广播 1 MB logits」= 21.48 tok/s，而「head 每段复制、各段自己算」= 26.26 tok/s。
教训：**不要往关键路径上加每段成本，除非确认它能塞进既有等待窗口**。

### 8.2 路由 top-8 softmax 与 span 规则：端到端 -2.6%（两次跌幅相同 = 环境漂移）

隔离测量都是正收益（`moe_router` 42.2→35.1 µs、gate_up 240→249 GB/s、down 107→112 GB/s），
但端到端两次都恰好 -2.6%——**两次跌幅相同本身就是漂移的指纹**。正确表述是「未证明」而非「负收益」。

### 8.3 `down_merge` 的 111 GB/s 是结构性上限

span 1/2/3 全在 111 附近、span=8 掉到 97 ⇒「96 vs 216 GB/s 还有 2× 空间」不成立。

### 8.4 投机解码：天花板 1.04x

单 token decode 39.41 ms；**M=4 校验轮 123.66 ms = 3.1×**（`Layer.verify` 的注意力与 GDN 递推**逐行串行**，
只省下权重读一遍那 22%；M=4 远低于 TCU 门槛 64 行）。一轮 = 3×9.07（草稿）+123.66 = 150.9 ms ⇒
α=1.0 也只有 **26.5 tok/s（1.04x）**。PLD 更要否掉：散文 mean_accept 0.03、代码 0.30。
α 实测是**双峰**：代码/SQL **8/8 全同**、散文 **1/8**。验证路径已做通（三模式 12 token 无漂移、
回滚正确），留待多行内核之后复用。

### 8.5 EP4：M=1 下 0.95x（gloo）/1.37x（ixCCL）

互连实测（4 卡 PCIe Gen4 x16）：all-to-all **27.4 GB/s**、all-gather 25.8、reduce-scatter 25.0、
**all-reduce 仅 14.0**；小消息延迟墙 **0.28-0.42 ms 且与载荷无关**（≤64 KiB 纯亏）。
PP4 单步 39.4 ms ⇒ TP4 直接死（80 次归约/步 × 40 层 ≈ 31 ms，实测 all_reduce 378 µs 与载荷无关）。
EP4 省 0.613 ms/层、但**每层一次**集合通信（dense 每卡复制，层间不能合并）要付 0.35-0.67 ms ⇒ 打平。
**EP4 在 M>=8 才成立**（每层专家工作量随之增长）。

### 8.6 每层 305 µs 的模型侧 glue

一层 876 µs（微探针）= 发射 128 + wrapper 142 + **模型侧 glue 305** + 设备 316。
glue 由 **223 次 torch 调用**构成，其中 **194 次是形状/步长簿记**（`__get__ 97、stride 22、reshape 21、dim 20`）。
只包 `bops` 的计时器看不见它——这是前两轮对不上的原因。
`linear_attn` 的 plan 缓存 + `norm._rows` 快路径合计只拿到 -2.9%（在 3.4% 噪声带内）。

### 8.7 输出缓冲池（arena）：-1.3%，已回退

`aten::empty` 每步 1056 次 / 1471 µs，占每段 CPU（5.06 ms）的 29%；而 PP 里每段 CPU 1:1 进 lap
且乘 4 段 ⇒ 理论上限 14%。实测 **-1.3%**，四档同向变差。

原因有两层：`aten::empty` 走 caching allocator 本身只有 **1.4 us/次**（1471/1056），
那 1471 µs 主要是 Python 调用开销而不是分配成本；而且**最热的点没进池**——decode 时
`moe_grouped_i8` 的行数是 `tokens * top_k = 8`（打包后的行），被 `shape[0] == 1` 的守卫挡在池外。

### 8.8 relay 载荷改 bf16 + pinned：+2.6%（保留）

每步只有 **2 次** relay。fp32 39.65 ms/步 → bf16 38.63 ms/步，**+2.6%**；贪心 24 token 的 ids
**逐位相同**，8 项测量全部同向。保留的依据是「逐位一致 + 结构性减半 + `COFFIN_PP_RELAY_F32=1`
可运行时回退」，而不是端到端过了止损线。

**顺带纠正一个把方向带偏的判断**：`relay` 里那个 5.014 ms 不是传输，是
`current_stream().synchronize()` 在**等本段 10 层内核排空**；流空了之后同一跳只剩 **0.13 ms**。
真正的 29 ms 是 `await`——**PP 的结构性气泡**，不是 relay 开销。所以「改 ack 省 6 个 RTT」
「链式扇出」那些推算全部作废（按 0.13 ms/跳算，最多省 0.3 ms）。

### 8.9 split-K 打开：64k 注意力 3.94x（本轮最大单笔）

`caps().attn_split` 从 1 改到 8，**内核源码一行没动**——`grid.y = split`、`part_acc/part_ml`、
`attn_combine_kernel` 早就实现好了。

| ctx | split=1 | split=8 | 加速 |
| --- | --- | --- | --- |
| 4k | 1.407 ms | 0.827 | 1.80x |
| 32k | 11.171 | 3.550 | 3.15x |
| 64k | 24.493 | 6.222 | **3.94x** |

根因值得记住：`caps()` 的 docstring 原本写「`attn_split=1`，因为注意力内核还不会按 K 切分」——
**把「能力记录里没开」当成了「内核没实现」，且从未实测**。代价是每层只有 `nh=16` 个 block，
在这张卡约 60 个 CU 上只用到四分之一——这才是 64k 下 22.9 ms 的真正原因，不是带宽也不是 ALU。

端到端：decode 26.41/25.51/25.52/23.29 → **27.92/26.71/27.02/28.82**（G4 +23.7%）。

### 8.10 「声明了但没注册」已经出现三次

`bindings.cpp` 里缺 `m.def` 会让算子看起来存在（头文件有声明、`.cu` 有定义）但模块里没有：
`attn_paged_batch`（批处理一跑就 `AttributeError`）、`attn_paged` 与 `attn_paged_multi_q`
（**服务会在注意力处直接崩**）。需要一个「头文件声明的算子必须在 bindings 里注册」的自检脚本。

## 9. 测量纪律

- **同一份二进制连跑的散布是 3.4%**（三连跑四档：26.21/26.24/25.35 等）。
  任何小于它的差异不成立。大于 10% 的结论都稳。
- **探针口径必须与服务一致**：`max_seq=4096` 的稠密 KV 量到 45.90 ms/步，而服务走 `--num-blocks 256`
  的分页路径（`attn_paged`）是 39.4 ms——**差 16%，会把 3.4% 噪声带彻底淹没**。
- `backends.status()` 返回字符串；判算子用 `backends.caps().native`（frozenset）。
- `bops` 是 frozen dataclass，**不能**逐入口 monkeypatch 计时；`torch.profiler`（kineto）在这套 torch 上不出输出。

## 10. 批处理曲线（下一步的主线）

每层 = 固定项 + 每行项，实测 M=1/2/4/8/16 → 1.99/2.87/3.43/4.75/6.96 ms；
拟合 `layer(M) = 2.05 + 0.315*M`（探针口径），in-lap 为 **0.670 ms 固定 + 0.315 ms/行**。

| batch | lap | 每 token | 聚合 | 倍数 |
| --- | --- | --- | --- | --- |
| 1 | 39 ms | 39.4 ms | 25.4 tok/s | — |
| 4 | 77 | 19.3 | **51.8** | **1.97x** |
| 8 | 127 | 15.9 | **62.8** | **2.38x** |
| 16 | 228 | 14.3 | 70.1 | 2.66x |

顺序：**批处理 → 削固定项 0.670 ms/层 → 再 EP4**（那时每层专家工作量大，通信才摊得开）。

## 11. 长上下文

- 只有 **10/40 层**是全注意力（3:1 线性注意力），`nkv=2, head_dim=256` ⇒ KV **1 KiB/层/token**。
- 256k 上下文：bf16 KV **2.56 GiB/序列**、int8 KV **1.28 GiB**（每卡 0.32-0.64 GiB）；
  GDN 递推状态 `[16,256,256]` × 30 层 ≈ **60 MiB/序列**。**显存能扛 48-60 个 256k 会话**，
  但 256k prefill 的 O(n²) 注意力是十几分钟量级 ⇒ **实际并发 1-2**。
- 注意力内核把 K 行改成 `int4` 向量加载后：4k 1.53×、**64k 1.39×**、2001-token 档端到端 **+6.1%**。
  原先的「GQA 8 次重读」不是瓶颈——真因是逐字节标量加载造成 **32× 读取浪费**（有效带宽只有 17 GB/s，
  而实际流量已约 540 GB/s）。分组共读的实现数值等价但慢 0.59-0.69×（共享内存放大 8 倍），已回退。

## 12. 复现

```bash
# 量化（一次，约 5 min）
bash ~/coffin-work/base/coffin/tools/launch_occamy_w8a8.sh

# 四卡起服务 + 四档冒烟（队列里投，约 9 min）
OCCAMY_WORLD=4 OCCAMY_PORT=30601 OCCAMY_SOCK=/tmp/occamy.sock \
OCCAMY_SEQ=4096 OCCAMY_BLOCKS=256 \
bash ~/coffin-work/base/coffin/tools/launch_occamy_pp.sh
```

客户端：
```bash
python -m entrypoints.cli.runtime --sock /tmp/occamy.sock generate --prompt "<prompt>" -n 32
python -m entrypoints.cli.runtime --sock /tmp/occamy.sock chat --user "..." -n 64
```

注意：**GPU 作业必须走队列**——直接 `scow run` 落在登录节点，没有 CUDA 库（`libcufft.so.10` 缺失）。
