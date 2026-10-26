# 能不能把集群的 torch 升上去 —— 结论：不能，卡在驱动

> 问的是"能不能给那台机器换个新 torch"。查完的答案是**不行**，而且不是权限问题、
> 也不是找不到安装包的问题：**这台卡的驱动（内核模块）太老，新 torch 的用户态库要的
> ABI 它没有**。换驱动是管理员在共享节点上做的事，一个作业不该碰，也碰不了。
>
> 全部结论来自只读探测（作业 3681 / 3686 / 3687 与四个节点的驱动查询），
> **没有安装任何东西、没有改 `/usr/local`、没有动内核模块**。

---

## 1. 事实

### 1.1 驱动版本被钉死在 3.2.1

```
$ ixsmi | head -3
| IX-ML: 3.2.3       Driver Version: 3.2.1       CUDA Version: 10.2 |
$ lsmod | grep iluvatar
iluvatar              446464  0
```

注意这一行的错位：**用户态工具链是 3.2.3（甚至 `/usr/local/corex` 指向 3.2.3），
但加载的驱动模块报 3.2.1**。节点之间不一致（`gpu-node-05` 装了两套 SDK，
`06/07/08` 只有 3.2.1），但**驱动版本全都是 3.2.1**。

### 1.2 新工具链的用户态库要一个驱动没导出的符号

```
$ LD_LIBRARY_PATH=<4.5 lib64>:/usr/local/corex/lib64 \
  python -c "import ctypes; ctypes.CDLL('./libnccl.so.2')"
OSError: ./libnccl.so.2: undefined symbol: ixdrvDeviceGetAttribute, version CUDA
```

`ixdrvDeviceGetAttribute` 是**驱动侧**的导出符号（`ixdrv*` 前缀 = Iluvatar driver）。
装了 3.2.3 用户态 SDK 的 `libcuda.so.1` 里 `nm -D` 数出来是 **0 个** `ixdrv` 符号——
它转发给驱动模块，而驱动模块没有这个入口。

同一个符号在 4.5 栈的另一个库上也炸：`libixToolsExt.so: undefined symbol: ixdrvInit`。

### 1.3 穷举矩阵：每一种组合都失败

| 解释器 | 库路径 | 结果 |
| --- | --- | --- |
| py310（能用的那个） | 宿主 3.2.x | **WORKS** — torch 2.1.0 / BI-V100 / matmul OK |
| py310 | 只有 4.5 | FAIL `libcuda.so.1` 找不到（4.5 目录里没有它） |
| py310 | 4.5 + 宿主 3.2.x | FAIL `libixToolsExt.so: undefined symbol: ixdrvInit` |
| py312 | 宿主 3.2.x | FAIL `libixattnbkd.so.1` 找不到（2.10 要的库 3.2.x 没有） |
| py312 | 只有 4.5 | FAIL `libcuda.so.1` 找不到 |
| py312 | 4.5 + 宿主 3.2.x | FAIL `libnccl.so.2: undefined symbol: ixdrvDeviceGetAttribute` |

**关键读法**：新 torch（2.10 + CoreX 4.5）和旧 torch（2.1 + CoreX 3.2.x）是两套
**不兼容的用户态栈**。把 4.5 的库塞进 3.2.1 驱动上跑，缺的是驱动符号；
用 3.2.x 的库配 2.10 的 torch，缺的是 3.2.x 根本没提供的库（`libixattnbkd.so.1`）。
**中间没有交集**：不存在"既能被 3.2.1 驱动喂饱、又够新到有 `F.rms_norm`"的 torch。

### 1.4 安装包这条路也断了一半

- 厂商 wheel 服务器上 `4.5.0` 的 URL **已经 404**（10-01 还能下，现在没了）；
  其他版本目录（3.2.1 / 3.2.3 / 4.0 / 4.2）连 key 都不存在。扫描了
  `10 个版本 × 2 个 python tag × 6 个 torch 版本`，**一个可下载的都没有**。
- 本地 `~/whl45/` 有一个 `torch-2.10.0+corex.4.5.0-cp312` 的 437 MB wheel
  （10-01 下载的存货），但它是 **cp312**，而且如 §1.3 所示跑不起来。

---

## 2. 结论

1. **不能升级**，除非有人把计算节点的 `iluvatar` 驱动从 3.2.1 升到 4.x。
   那是内核模块，要重启/重装，属于集群管理员，不属于用户作业。
2. **也不需要升级**。`F.rms_norm` 是唯一真正挡住 reference 后端的 API，
   已经改成手写实现（`backends/reference/__init__.py`，fp32 累加 + rsqrt）。
   coffin 自己写算子，本来就不该依赖 torch 的高层融合 API——这条依赖越少越好。
3. 如果**将来**管理员升级了驱动，`backends/corex` 的 `caps` 会自动跟着驱动走
   （`device_name` / `arch` 都是运行时读的），不需要改代码；届时可以考虑
   `weight_dtype` 与 graph 的重新评估。

---

## 3. 对移植工作的直接影响（已经照此执行）

- **必须支持 torch 2.1**：`F.rms_norm` / `torch.compile` / `torch.library.custom_op` /
  `torch.nn.attention` 一律不可用（真机探测确认后三个都缺）。
- **CoreX 后端自己写 rms_norm**，不指望 torch。
- **`weight_dtype` 选 bf16**：实测 prefill GEMM 在 bf16 与 fp16 上**同为 32.6 TFLOP/s**
  （与 gfx906 完全不同——那边 bf16 只有 fp16 的 1/7），所以没有理由为算力选 fp16。
- **graph 要按算子表判断**：cuBLAS 在这张卡上不能被 capture（`CUBLAS_STATUS_ALLOC_FAILED`），
  自写 kernel 可以。所以 `caps.graph` 不是"平台属性"而是"这一层算子是否全自写"。

---

## 4. 复现方式

```bash
# 只读survey（驱动/工具链/解释器/安装包）
python scow/tools/deploy.py cohort -d ../coffin/diag/corex_probe \
    -s ../coffin/diag/corex_probe/launch_upgrade.sh --run upg -g 1 -t 10

# 解释器 × 库路径矩阵（每个组合一个子进程，不安装任何东西）
python scow/tools/deploy.py cohort -d ../coffin/diag/corex_probe \
    -s ../coffin/diag/corex_probe/launch_upgrade_run.sh --run upgrun -g 1 -t 12

# 各节点驱动版本对照
python scow/tools/probe_driver_matrix.py
```
