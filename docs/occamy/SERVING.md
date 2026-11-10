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

### 8.10 低方差量具：判 0.5% 的改动

端到端 decode tok/s 的散布是 3.4%，判不出小改动。改用 PP4 单进程量具（`~/models/gauge_a.py`）：
四 rank 圈数严格一致、先短 prefill 填 KV、连跑 100 步 × 5 轮，报 `process_time`（CPU）与
`perf_counter`（wall）的**中位数 + IQR**。

基线（`53ba0f5`、`native=35/35`、`attn_split=8`）：

| rank | cpu_med | cpu_iqr | wall_med | wall_iqr |
| --- | --- | --- | --- | --- |
| 0 | 10.258 ms | 0.053 | 38.486 ms | 0.242 |
| 1 | 9.968 | 0.039 | 38.485 | 0.243 |
| 2 | 9.606 | 0.027 | 38.485 | 0.242 |
| 3 | 9.760 | 0.057 | 38.485 | 0.243 |

**IQR 0.5%**，比 tok/s 好一个量级。注意 `process_time` 是进程级 CPU（含 torch 线程），
与「每段 5.06 ms」的墙钟口径不同，只可作相对比较。

### 8.11 `bops` 包装只有 1.93 µs/次（同形 3000 次 × 5 轮）

```
rms_norm 直调 pybind = 9.61 µs/次
rms_norm 经 bops     = 11.53 µs/次
⇒ 包装开销 = 1.93 µs/次
```

所以「`bops` 包装 142 µs/层 = 14% 上限」这个估算是错的：真实是
**1.93 × 17 次/层 = 33 µs/层 = 1.3 ms/lap ≈ 3.4%**，而且可削的只有 Python 那一小段。

试过四处短路（`_pitch_rows` 已是 `[rows, width]` 时原样返回、三处 `gain.contiguous()`
条件化）：wall 38.486 → 38.753（**+0.7%，变差**）、四 rank 方向不一致，已回退。
**1.93 µs 主要在 pybind 参数编组，Python 侧压不动了。**

### 8.12 融合的价值公式

```
收益 ≈ 6-8 µs × 被消掉的 Python 层算子数 × 层数
```

`shared_gate_scale` 值 5-9%，是因为它消掉 **6 个** torch 算子（6 × ~6.5 µs × 10 层/段
≈ 0.39 ms/段）。按这个公式：
- 「两个内核合成一个」只消 1 个算子 ⇒ **<1%**，不值得做；
- 只有每层还残留 **≥6 个连续 Python 算子**的地方才值得动。

逐条查完的结果：GDN 层只剩 9 个算子（`qkvzb_a` 已一次投影出四份、conv+silu 已融合、
plan 已缓存）、层壳 4 个（两处残差相加都折进了 norm）、MoE 4 个。**唯一还留着的链**
是 `blocks.py:167` 的 `x + self.mlp(normed)`，折掉省 1 次 `aten::add`（4.66 µs）⇒
186 µs/lap ≈ **0.48%**，正好压在量具分辨率上，且要在最热的函数上改层契约——不建议做。

### 8.13 relay 载荷改 bf16 + pinned：+2.6%（保留）

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

### 8.11 KV 池改头主序：单层 1.93x，端到端 0（默认关）

单层合成探针（同一内核、同一份数据、只换池的 stride）：

| 池布局 | 4k 单层 |
| --- | --- |
| `[slots, nkv, hd]`（token 主序） | 0.806 ms |
| `[slots, 1, hd]` | 0.418 ms |
| 稠密 `attn_decode` 对照 | 0.410 ms |

于是把池改成头主序 `[nkv, slots, hd]`（`cache_row` 改成两个步长的无分支形式，池、写路径、
四个入口、参考后端全部跟着改）。**结果：4k/32k/64k 的层内收益没在服务里复现。**

| 档 | 头主序开 | 头主序关 |
| --- | --- | --- |
| G1 5 tok | 27.42 | 27.47 |
| G2 23 tok | 26.23 | 26.32 |
| G3 14 tok | 26.55 | 26.64 |
| G4 2001 tok | 28.3（prefill 2302.9 ms） | 28.4（prefill 2291.7 ms） |

文本逐字相同、`G1..G4=0`、显存不变，说明改动本身是对的、只是**没有收益**。原因判断：合
成探针的池很小且槽位密集，`nkv=1` 那一档顺带改变了访问窗口；服务里池有 16k 槽位、每个键
仍然是 256/512 B 的独立行，两种布局的流式行为没有差别。

**结论：默认关（`COFFIN_KV_HEAD_MAJOR=1` 可开），代码保留**。教训：单层的布局/带宽对比必
须用**服务同款池形状**做，否则测出来的是探针自己的形状效应。

### 8.15 MoE 块合成一次 C++ 驱动：+3.2~3.8%，逐位相同

一层 decode 的 MoE 块有 17 个 pybind 调用（其中 11 个在这个块里）。`moe_decode_block` 把它们压成
**1 次**（路由投影 → `moe_router` → 共享专家两次投影 → `shared_gate_scale` → `moe_dispatch` →
`quant_act_i8` → `moe_grouped_i8` → `silu_mul` → `quant_act_i8` → `moe_grouped_i8_merge` →
加共享专家），中间缓冲全部由调用方预分配复用。

**收益上限只有个位数的原因**：单个 `bops` 入口 11.53 us 里，1.93 us 是 Python 包装、约 2 us 是
pybind 编组，剩下约 7.5 us 是 `torch::empty` 分配输出 + 发射内核——**C++ 驱动照样要付这两笔**。
能省的只有约 4 us/次 x 17 次。

| 口径 | 改前 | 改后 |
| --- | --- | --- |
| 单层 MoE 块 | 0.416 ms | **0.383 ms（-7.8%）** |
| G2 / G3 / G4 | 26.26 / 26.57 / 28.33 tok/s | **27.27 / 27.58 / 29.24（+3.2~3.8%）** |
| 逐位对拍 | — | **maxdiff = 0.000e+00，rel = 0** |

**三个只在真调用时才暴露的内核问题**：

1. `moe_router` 写 int64 下标，而 `moe_dispatch` 只收 int32——Python 包装层自己做的那次转换，驱动里漏了；
2. `moe_grouped_i8_merge` 的逐行权重契约是 fp32，而 `rw` 跟着 logits 是 bf16；
3. `project` 对 `[1, H]` 走了多行内核，而 `GEMV.linear_m` 在 `M == 1` 时会**绕回一维入口**——
   两条路的累加方式不同，不照抄这个判据就不可能逐位相同。

**一个模型特性决定了驱动要有两条路由路径**：`mlp.gate.weight` 在 `DENSE_SUFFIXES` 里
（量化噪声会翻转专家选择），所以路由权重是 **bf16 稠密**的；驱动因此增了一条 bf16 的
`gemv_kn` 分支，用零长度的 `router_scale` 表示「走稠密」。

`COFFIN_LAYER_DRIVER=0` 可退回 Python 编排，用于同二进制 A/B。

### 8.16 批处理服务级验证：机制通、level=2 正确、level>=3 分叉

服务级（4 卡 PP4、`--max-num-seqs 8`、`--num-blocks 1024`、8 条不同 prompt、贪心、n=96/192）：

| level | 逐字一致 | 聚合 tok/s |
| --- | --- | --- |
| 1（基准） | — | 26.2 |
| **2** | **8/8** | 23.8-25.0 |
| 3 | 6/8 | 20.0 |
| 4 | 3-4/8 | 23.6-24.4 |
| 8 | 3-4/8 | 30.6-32.1 |

**并发确实发生了**：`[step]` 的 running 分布为 1 (3162 次) / 2 (63) / 3 (321) / 5 (63) / 6 (99) / 7 (30)。

**分叉位置**：level=3 在 token 72 与 74；level=4/8 在 38、72、74、76——不同请求的分叉位置不同、
且都在较后位置，不是首 token（首 token 就错的那个 bug 已由行步长修复解决）。

**聚合收益有限**：level=8 只有 30.6-32.1 tok/s（约 1.2x），level=3 甚至低于单流（20.0 vs 26.2）。
这与「每行成本接近一整份单流成本」的实测一致。

默认路径与单流均无回归（这是放行 `f6856a1` 的依据）：

```
默认路径（快路径开 + OCCAMY_SEQS=4）：28.02 / 26.90 / 27.18 / 30.12 tok/s、prefill 2310.0 ms、9.42 GiB
单流（OCCAMY_SEQS=1）：            28.38 / 27.24 / 27.53 / 29.28 tok/s、prefill 2302.3 ms、9.23 GiB
四档 G1..G4 = 0 0 0 0
```

下一步怀疑方向：批组成变化时的行重分配（`set_rows` / `assign_table`）——请求陆续结束会让行被复用，
而分叉只出现在较后位置且与 level 相关，符合「中途换了行」的形态。`COFFIN_KV_TRACE=1` 已能打印
块级重复占用（`ALIAS_BLOCK`）与槽位区间相交，可用于区分这两类损坏。

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

## 13. 批处理 serving 的最终裁决：不成立

两轮独立服务级验证（4 rank PP4、卡 4-7、`--max-num-seqs 4`、8 条不同 prompt、贪心、N=192）：

| level | 聚合 tok/s | 相对顺序基准 26.9 | 逐字一致 |
| --- | --- | --- | --- |
| 顺序（基准） | 26.9 | 1.00x | — |
| 2 | 25.67 | **0.95x** | **8/8 通过** |
| 4 | 24.01 | **0.89x** | 3/8 |
| 8 | 26.63 | **0.99x** | 3/8 |

**四个档位没有一个超过单流。** 并发在这张卡上只是时间切片，不是吞吐增益——与「边际 ≈22.9 ms/行，每多一行几乎付一整份单流成本」完全一致。
（注：该归因是从「聚合与单流相等」推断，未做 `_decode_batch` 逐段打点。）

### 修掉的两个真 bug（都有价值，留着）

1. **行步长**：`linear_attn.cu:393` 的 `prologue_batch_kernel` 按「行步长 = kdim」算地址，而调用方传
   `conv_out[:, :kdim].reshape(...)` —— **`reshape` 对非连续切片返回 view 不拷贝**，行步长仍是 `conv_dim`。
   于是 row 0 偏移恒为 0 恰好正确、**row >= 1 全读到别人的行**。线索是调用方只给 b/a 加了 `.contiguous()`。
   修法：q/k/v 补 `.contiguous()`。
2. **行表缓存不同步**：`_row_tables` 只在 `reset()` 里清，而 `_stage_sentinels` 直接改池表却没失效缓存；
   新序列拿到该行时若块表恰好等于旧值，`assign_table` 被跳过 ⇒ 行仍指向只有一块的暂存区 ⇒ 超出一个块后开始错。
   修法：`self._row_tables.pop(int(row), None)`。

**level >= 4 仍有第二处缓存/状态不同步**（分叉在较后位置、非首 token）。若将来重启这条线，从「哨兵行的 GDN 状态」
与「块表的第二份缓存」查起。

### 结论

`--max-num-seqs 1` 为默认。**多用户靠多实例，不靠实例内并发**。批处理代码保留（机制通、level 2 逐位正确），
默认关闭。

## 14. 工具链的四个坑（会让人得出错误结论）

1. **`platform/scow/tools/sync.py` 判「无变化」时静默不推送** —— 已发生过一次修复被漏推（远端 grep 为 0 而本地已提交）。
   可靠做法：本地 `tar` 单个改动文件 → `scow put` → 远端 `tar xzf`。
2. **远端 `~/coffin-work/base/coffin` 不是 git 仓库**，脚本里 `git log` 会 fatal。
3. **`scow run` 会静默吞 stdout**（部分日志内容读不出来）—— 结果一律远端 `base64 -w0` 再本地解码。
4. **`scow run` 落在登录节点、无 CUDA 库** —— 卡状态、进程检查、任何 GPU 作业都必须走队列。
   另：这台机 **`ps -u $(whoami)` 会报错**（用户名是纯数字 uid），用 `ps -eo pid,uid,args --no-headers | awk -v d=$(id -u) '$2==d && /entrypoint/ {print $1}'`。

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
