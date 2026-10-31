# GLOSSARY — coffin 中英术语对照

本仓库注释与文档统一使用下列译法。**同一个概念只允许一种译法**，不要在不同
文件里换词。表中没有的词，优先保留英文原词（它本身是标识符或平台名），或者
按下面的构词习惯新造一个，并把它补进本表。

结构标记（`Args:` / `Returns:` / `Raises:` / `Module:` / `Constraints:` /
`@brief` / `@param` / `@return` / `@throws` / `#:`）**一律保持英文**，只有标记
后面的正文翻译。

---

## 核心概念

| 英文 | 中文 | 说明 |
| --- | --- | --- |
| engine | 引擎 | `engine/` 目录、`LLMEngine` |
| model | 模型 | |
| model family | 模型族 | 例如 Qwen3.5 一族 |
| backend | 后端 | `backends/`，一个平台一套 |
| vendor | 厂商 | Iluvatar、Enflame、Moore Threads |
| platform | 平台 | |
| kernel | 内核 | GPU 上的 `.cu` 核函数 |
| operator / op | 算子 | |
| device | 设备 | 不译作「器件」 |
| host | 主机侧 | `host 侧` 也可，指 CPU 那一边 |
| runtime | 运行时 | |
| entrypoint | 入口 | |
| serving | 服务 | `serving/` |
| reference | 参考实现 | `backends/reference/` |

## 推理流程

| 英文 | 中文 | 说明 |
| --- | --- | --- |
| prefill | 预填充 | 不译作「预填」 |
| decode | 解码 | 逐 token 生成阶段 |
| batch / batched | 批 / 批处理 | |
| sequence | 序列 | |
| request | 请求 | |
| prompt | prompt | 保留英文，它是协议层概念 |
| token | token | 保留英文 |
| tokenizer | 分词器 | |
| logits | logits | 保留英文 |
| sampling | 采样 | |
| greedy | 贪心 | |
| speculation / spec | 投机 | 投机解码 |
| drafter | 草稿模型 | MTP 里的 `drafter` |
| suffix | 后缀 | |
| chunk / chunked | 分块 / 分块预填充 | |
| stream / streaming | 流式 | |
| prefix | 前缀 | |
| position | 位置 | |
| context length | 上下文长度 | |

## 缓存与调度

| 英文 | 中文 | 说明 |
| --- | --- | --- |
| cache | 缓存 | |
| paged cache | 分页缓存 | |
| radix cache / radix tree | 基数缓存 / 基数树 | |
| block | 块 | KV 块 |
| block table | 块表 | |
| block manager | 块管理器 | |
| slot / slot map | 槽位 / 槽位表 | |
| pool | 池 | 内存池、KV 池 |
| refcount | 引用计数 | |
| copy-on-write | 写时复制 | |
| prefix reuse | 前缀复用 | |
| eviction | 淘汰 | |
| scheduling | 调度 | |
| preemption | 抢占 | |
| admission | 准入 | |
| budget | 预算 | 显存预算、快照预算 |
| capacity | 容量 | |

## 模型结构

| 英文 | 中文 | 说明 |
| --- | --- | --- |
| layer | 层 | |
| weight | 权重 | |
| checkpoint | checkpoint | 保留英文 |
| attention | 注意力 | |
| full attention | 全注意力 | |
| linear attention | 线性注意力 | |
| recurrent state | 循环状态 | |
| state snapshot | 状态快照 | |
| KV cache | KV 缓存 | |
| vision tower | 视觉塔 | |
| image | 图像 | |
| media | 媒体 | 图像等多模态输入的统称 |
| pixel | 像素 | |
| projection | 投影 | 线性投影 |
| residual / residual add | 残差 / 残差相加 | |
| norm / RMSNorm | 归一化 / RMSNorm | RMSNorm 保留原名 |
| head | 头 | 注意力头 |
| head_dim | head_dim | 保留英文（标识符） |
| hidden size | 隐含维度 | |
| embedding | 嵌入 | |
| RoPE / mRoPE | RoPE / mRoPE | 保留英文 |
| quantization | 量化 | |
| dequantize | 反量化 | |
| int8 / int4 | int8 / int4 | 保留 |
| row-major | 行主序 | |
| contiguous | 连续存放 | |
| layout | 布局 | |

## 硬件与性能

| 英文 | 中文 | 说明 |
| --- | --- | --- |
| LDS | LDS | 保留 |
| shared memory | 共享内存 | |
| CU / SM | CU / SM | 保留 |
| wavefront | wavefront | 保留（AMD 术语） |
| warp | warp | 保留 |
| bandwidth | 带宽 | |
| latency | 延迟 | |
| throughput | 吞吐 | |
| launch overhead | 启动开销 | |
| occupancy | 占用率 | |
| tiling | 分块 | 与 chunk 的「分块」区分时用「切块」 |
| split-K | split-K | 保留 |
| K-unroll | K 方向展开因子 | |
| scratch | 中间缓冲区 | 共享内存语境写「共享暂存」 |
| row pitch | 行距 | `stride` 作为参数名保留英文 |
| eager path | 即时执行路径 | 与「图捕获」对举 |
| GEMV / GEMM | GEMV / GEMM | 保留 |
| offload | 卸载 | |
| VRAM | 显存 | |
| ABI | ABI | 保留 |
| driver | 驱动 | |

## 工程动词

| 英文 | 中文 |
| --- | --- |
| implement | 实现 |
| fall back to | 回退到 |
| dispatch to | 分派到 |
| materialise | 实体化 |
| snapshot | 打快照 / 快照 |
| reuse | 复用 |
| fail / break | 出错 / 坏掉 |
| verify | 校验 |
| guard | 保护 / 拦住 |
| raise | 抛出 |
| caller | 调用方 |
| contract | 契约 |
| invariant | 不变量 |
| trade-off | 取舍 |
| constraint | 约束 |
| responsibility | 职责 |
| module | 模块 |

---

| claim / claim strength | 认领 / 认领强度 | 后端声明支持某算子的动作 |
| eager path | 即时执行路径 | 与「图捕获」对举 |
| tiling | 切块 | 与 chunk 的「分块」冲突时用「切块」 |
| allowlist | 允许列表 | 不用「白名单」；该表现在只能朝空收敛 |
| partials | 部分和 | split-K 的 part_acc / part_ml 累加缓冲 |
| split | 分片 | split-K 的分片数 |
| staging row | 暂存行 | 预填充语境写「预填充暂存行」 |
| high-water mark | 高水位 | |
| decode row | 解码行 | |
| draft / drafting | 草稿 / 草拟 | drafter 用「草稿模型」 |
| verify | 校验 | 投机语境写「投机校验」 |
| tap / taps / tap sum | 抽头 / 抽头 / 抽头求和 | |
| packed fp16 | packed fp16 | 硬件打包乘加，无贴切中文 |
| bucket | bucket | 分桶参数名，保留英文 |
| staging row | 暂存行 | |
| sentinel row | 哨兵行 | |
| salt / salted | 盐 / 加盐 | |
| span | 跨度 | |
| oracle | 对照实现 | |
| stub | 桩 | 桩引擎 |
| tapping | 抽头 | |
| dense model | 稠密模型 | |
| fixture | fixture | 保留英文 |
| online softmax | online softmax | 保留英文 |
| teacher forcing | teacher forcing | 保留英文 |

## 构词习惯

- 英文名词后接中文动词，不加「的」凑字：写「KV 缓存被复用」，不写
  「KV 缓存被进行复用」。
- 中英文之间加空格：`INT8 权重`、`64 KB LDS`、`[B, S, H]`。
- 数字与单位保持原样：`23.53 -> 20.26 ms`、`1.16x`、`~24 GB/s`。
- 不用「进行」「予以」「加以」这类空动词。
- 中文标点用全角：，。；：（）「」；代码和标识符内的标点保持半角。
