"""四卡同机的进程间归约：共享内存实现，替掉 gloo。

实测（`diag/qwen4/probe_shm_allreduce.py`，作业 4410，卡 3/4/5/6）：

| 实现 | `[1, 2560]` fp32 单次 | 48 层折算 | decode 上限 |
| --- | --- | --- | --- |
| gloo `all_reduce` | **291–420 µs** | 14.0–20.2 ms | 50–72 tok/s |
| 共享内存 barrier | **32.9 µs** | 1.6 ms | 633 tok/s |

差 8.9 倍。原因是机制选错了：四个 rank 全在同一台机器上，走 TCP loopback 做同机
进程间通信要经过两次系统调用、两次 skb 拷贝和 Nagle 定时器，而共享内存是每个
rank 直接写对方可见的缓存行。

**这个结论改变并行拓扑的选择。** 原本按 232 µs/次 算，TP=4 每层归约一次就是
11 ms、每层两次是 22 ms，会逼着去选 EP 来躲通信；换成 33 µs 之后，每层归约一次
是 1.6 ms，**通信不再是决策变量**，拓扑可以纯粹按「哪个算得快」来挑。

## 两条路按行数分，不是按载荷大小分

四进程 EP4 实测（`tools/probe_q4_ep4_gen.py`，48 层，B=1）：gloo 侧一次归约
**0.71 ms**，48 层/步 = **34 ms/步**，占 187.72 ms 步时间的 18%；共享内存侧
`rows=1` 是 **32.9 µs/次**，48 层/步 = **1.6 ms**。所以 :class:`DecodeShmCombine`
把 decode（`rows=1`）交给共享内存板、prefill 继续走 gloo。**不能反过来按「小载荷」
分**：`rows=64` 时共享内存是 194 ms/次（Python 层 4 次全量拷贝），比 gloo 的
0.73 ms 慢 266 倍。

**0.71 ms 里有一半不是通信。** 那个探针在每次归约前多做一个 1 元素 all_reduce
（帧号校验，用来发现各 rank 的调用序列错开一格），而 gloo 的成本几乎全是固定
开销、与载荷无关，于是「载荷 + 校验」把单次成本抬了一倍。纯载荷约 0.35 ms，
48 层/步 17 ms。这两个数不能混用：和 187.72 ms 比要减 34 ms（那是探针付的），
和「引擎该付多少」比只能算 17 ms。

Module: ``coffin/models/qwen4exp``。模型族内的并行原语。依赖只有 ``torch`` 和
标准库——它不碰 ``backends``，因为这条路径只交换 CPU 上的 fp32，与平台无关；
``blocks.py`` / ``model.py`` 在构造时按 rank 拿到它，单卡时传 ``None``。

Constraints:
    * **每个 rank 只写自己的格子，绝不共享一个计数器。** 第一版用「四个 rank 都对
      一个 int64 读-改-写」当 barrier，实测直接死锁：读-改-写不是原子的，两个 rank
      会读到同一个值再都写回 ``cur+1``，计数器永远到不了 ``world * round``。
    * **绝不在这里 ftruncate 到 0。** 文件由调用方在任何进程启动之前建好清好；
      在这里截断会让四个进程互相拆台，报 ``mmap length is greater than file size``，
      而就算侥幸建好也会把别人写进去的数据清零。
    * 自旋有上限。一个 rank 挂了其余会永久空转，所以到点就抛 ``TimeoutError``，
      让外面的 ``timeout`` 收尸——绝不无限等。
    * 成本与 payload 的关系只在 rows<=8 时近似常数（32.9 → 65.5 µs），说明它仍是
      「次数受限」的；rows=64 时 Python 层 4 次全量 copy 吃掉 188 ms，那是调用方的
      开销不是通信的。**大 payload 不要直接拿这个类去 reduce 一个 prefill chunk。**
    * 没有序列号，所以各 rank 必须以相同顺序调用；顺序错了会把两轮的数据混在一起
      而不是报错。
"""

from __future__ import annotations

from collections.abc import Sequence

import ctypes
import mmap
import os
import struct
import time

import torch

#: 归约的宽度。qwen4_exp 的 ``hidden_size``，MoE 的 combine 归约的就是它。
HIDDEN = 2560

#: 诊断开关：``1`` 时把跨 rank 归约降级成恒等（**数值错误，只用于计时**）。
#:
#: 为什么要它：作业 416/417/418 量到「四卡 decode − 单卡 decode = 53.4 ms」，
#: 而 ``combine`` 的墙钟是 50.2 ms，两者几乎相等；但它的内部分量里
#: ``pre_us``（870 µs/层）是「等本层设备算完」——那段时间单卡也在算，不该算新增成本。
#: 分量和总量对不上，所以**在改架构之前必须先量出 combine 的真实端到端代价**：
#: 把它短路成 ``return x``，四卡 decode 的差分就是它真正的成本上限。
_COMBINE_NOOP = os.environ.get("COFFIN_COMBINE_NOOP", "").strip() == "1"


class ShmAllReduce:
    """若干 rank 通过一块 mmap 做全归约，语义同 ``dist.all_reduce(SUM)``。

    Args:
        path: mmap 文件路径。必须由调用方在**任何进程启动之前**创建并清空。
        rank: 本进程编号。
        world: 进程数。
        rows: 每次归约的行数，``[rows, 2560]`` fp32。

    Raises:
        ValueError: ``rows`` 或 ``world`` 非正。
    """

    HEADER = 64

    def __init__(self, path: str, rank: int, world: int, rows: int = 1) -> None:
        """打开共享板，为自己的 rank 建好写入口。

        各 rank 的槽位在构造时就算好偏移并缓存成 ``memoryview``：归约热路径上
        每轮要读 ``world`` 次，临时切片的开销在这个尺度上是可观的。

        Args:
            path: mmap 文件路径。调用方必须在起任何进程之前创建并清零。
            rank: 本进程编号，决定写哪一格、以及哪一格的到达计数归自己。
            world: 进程数。
            rows: 每次归约的行数，``[rows, 2560]`` fp32。

        Raises:
            ValueError: ``rows`` 或 ``world`` 非正。
        """
        self.rank = rank
        self.world = world
        self.rows = rows
        self.slot_elems = rows * HIDDEN
        self.slot_bytes = self.slot_elems * 4
        self.total = self.HEADER + world * self.slot_bytes
        self.round = 0
        self._fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
        # 只保证「至少这么大」，绝不缩到 0：清空由调用方在起进程之前做完，
        # 那是唯一没有并发者的时刻。
        if os.fstat(self._fd).st_size < self.total:
            os.ftruncate(self._fd, self.total)
        self._mm = mmap.mmap(self._fd, self.total)
        self._arrivals = memoryview(self._mm)[:world * 8]
        # 只读视图与只写视图分开，避免反复切片产生临时对象。
        self._slots = [memoryview(self._mm)[self.HEADER + i * self.slot_bytes:
                                              self.HEADER + (i + 1) * self.slot_bytes]
                       for i in range(world)]
        # 复用一个临时张量承接每个槽位，避免每次归约分配 world 次。
        self._stage = torch.empty(rows, HIDDEN, dtype=torch.float32)
        self._stage_view = memoryview(self._stage.numpy()).cast("B")

    def __call__(self, tensor: torch.Tensor, out: torch.Tensor,
                 *, spin_limit: float = 20.0) -> torch.Tensor:
        """把各 rank 的 ``tensor`` 求和写进所有 rank 的 ``out``。

        Args:
            tensor: 本 rank 的 fp32 CPU 张量，形状 ``[rows, 2560]``，连续。
            out: 结果写入的张量，形状同 ``tensor``，会被覆盖。
            spin_limit: 自旋等待上限（秒）。

        Returns:
            ``out``，便于链式调用。

        Raises:
            TimeoutError: 某个 rank 在 ``spin_limit`` 内没有到达。
            ValueError: 形状或 dtype 与构造时不一致。
        """
        if tensor.shape != (self.rows, HIDDEN) or tensor.dtype != torch.float32:
            raise ValueError(
                f"expected [{self.rows}, {HIDDEN}] fp32, got {tuple(tensor.shape)} "
                f"{tensor.dtype}")

        src = memoryview(tensor.contiguous().numpy()).cast("B")
        self._slots[self.rank][:len(src)] = src

        off = self.rank * 8
        cur = struct.unpack_from("<q", self._arrivals, off)[0]
        struct.pack_into("<q", self._arrivals, off, cur + 1)
        self.round += 1

        want = self.round * self.world
        deadline = time.perf_counter() + spin_limit
        while True:
            total = 0
            for i in range(self.world):
                total += struct.unpack_from("<q", self._arrivals, i * 8)[0]
            if total >= want:
                break
            if time.perf_counter() > deadline:
                raise TimeoutError(
                    f"rank {self.rank} timed out waiting at round {self.round}")

        out.zero_()
        for i in range(self.world):
            self._stage_view[:] = self._slots[i]
            out.add_(self._stage)
        return out

    def close(self) -> None:
        """释放映射和文件描述符。"""
        try:
            self._slots.clear()
            self._arrivals.release()
            self._mm.close()
        finally:
            os.close(self._fd)


def board_path(path: str, rows: int) -> str:
    """一块板用一个文件：``{path}-r{rows}``。

    到达计数在文件头、每 rank 8 字节，而每个实例的 ``round`` 都从 0 开始。同一个
    文件上放两块板（decode 的 ``rows=1`` 与 prefill 的 ``rows=128``）时，两边往同一
    组计数器上累加，某个 rank 的 ``want = round * world`` 会被另一块板提前满足，
    于是读到别人那一轮的数据——**不报错，只是数值错**。

    Args:
        path: 板文件前缀。
        rows: 这一块板每次归约的行数。

    Returns:
        该行数对应的板文件路径。**行数不同就是不同文件**，这条不能省。
    """
    return f"{path}-r{int(rows)}"


def prepare_shm_boards(path: str, ranks: int, rows: Sequence[int]) -> list[str]:
    """在任何 rank 打开板文件之前，由 rank 0 把它建好并清零。

    ``ShmAllReduce`` 自己不负责清空，理由见它的 ``Constraints``：四个并发
    ``ftruncate`` 会互相把对方刚建好的文件截到 0，然后报
    ``mmap length is greater than file size``。清空只允许发生在「没有并发者」的时刻，
    所以调用顺序必须是：rank 0 建好 → ``dist.barrier()`` → 其余 rank 才构造实例。

    Args:
        path: 板文件前缀，与传给 :class:`DecodeShmCombine` 的一致。
        ranks: 进程数，决定每块板要几个格子。
        rows: 需要哪几档行数的板。decode 只有 1 一档。

    Returns:
        建好的文件路径列表。

    Raises:
        ValueError: ``rows`` 里出现非正数，或 ``ranks`` 非正。
    """
    if ranks <= 0:
        raise ValueError("ranks must be positive")
    made: list[str] = []
    for row in sorted({int(r) for r in rows}):
        if row <= 0:
            raise ValueError(f"rows must be positive, got {row}")
        target = board_path(path, row)
        total = ShmAllReduce.HEADER + ranks * row * HIDDEN * 4
        fd = os.open(target, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            # 先截到 0 再截到 total：新文件的空洞读出来就是零，与「写满零」等价，
            # 但不用真的搬 40 KB 字节。这样「清空」和「定尺寸」是同一次系统调用序，
            # 后开的人只会看到已经够大的文件，不会再进 ``__init__`` 的 ftruncate。
            os.ftruncate(fd, 0)
            os.ftruncate(fd, total)
        finally:
            os.close(fd)
        made.append(target)
    return made


class DecodeShmCombine:
    """MoE 路由专家输出的跨 rank 归约：decode 走共享内存板，prefill 走 gloo。

    模型侧只要 ``combine(x) -> x_reduced``（语义 ``all_reduce(SUM)``），所以这个类对
    ``blocks.py`` / ``moe.py`` 是透明的。分路判据是行数，不是载荷大小：

    * ``rows <= shm_rows``（decode，``rows=1``）：32.9 µs/次，48 层/步 1.6 ms；
    * ``rows > shm_rows``（prefill，``tokens * top_k`` 从 80 起）：gloo 0.4-0.7 ms/次，
      **绝不能用共享内存**——``rows=64`` 时它是 194 ms/次，按行数线性恶化。

    一次 decode 调用只有一处同步：把激活从设备搬到锁页内存（``copy_`` 阻塞）。回传
    用锁页缓冲 + ``non_blocking=True``，下一次调用的那一次阻塞拷贝会把它等干净，
    所以复用同一个回传缓冲不会与未完成的 H2D 打架。

    Attributes:
        calls / calls_shm / calls_gloo: 三条计数分开记——换个判据时要知道各自付了
            多少次，而不是只看到总耗时对不上。
        millis / millis_shm / millis_gloo: 墙钟毫秒。**不能用 CUDA event 计时**：
            载荷在主机上时设备全程空闲，event 量到的是循环本身。
        tag: 当前正在归约的层号，由调用方在每层之前写进来。没有它，``trace`` 只是
            一串没有层号的数，分不清「第 0 层算错」与「第 3 层才算错」。
        trace: 前 ``trace_limit`` 次调用的 ``(tag, rows, 本 rank 输入和, 归约后和)``。
            输入和四个 rank 各不相同是**对的**（各持 1/world 的专家），归约后必须
            相同；两者能一眼分开「归约没生效」与「输入本身就不对」。
        timing: 前 ``timing_limit`` 次调用的
            ``(tag, rows, 走的是哪条路, 到达时刻, 等待微秒, 到达前微秒)``。

            **为什么把「到达」和「等待」拆开**：整个调用里有一段是等设备跑到这一层
            （阻塞 D2H），另一段才是等**最慢的那个 rank**。四卡 EP 的每次归约都在付
            「最慢 rank」的钱，而它随每层的本地命中数变化；把两段混成一个数，就无法
            回答「某次改动为什么把等待放大了」。

            到达时刻用 ``time.perf_counter()``：Linux 上它是系统级单调时钟，同节点
            不同进程之间可直接相减，因此四个 rank 的**到达时刻极差**是可测的。
    """

    def __init__(self, path: str, rank: int, world: int, dist,
                 dtype: torch.dtype = torch.bfloat16, shm_rows: int = 0,
                 trace_limit: int = 0, spin_limit: float = 20.0,
                 timing_limit: int = 0) -> None:
        """建共享板缓存与锁页中转缓冲，暂不打开任何板。

        板在第一次用到那一档 ``rows`` 时才构造：decode 只有 ``rows=1`` 一档，而
        prefill 永远走 gloo，于是 prefill 不会因为这里多建一块板而多一个文件。

        Args:
            path: 板文件前缀。必须已经由 :func:`prepare_shm_boards` 建好并清零。
            rank: 本进程编号。
            world: 进程数。
            dist: 已初始化的 ``torch.distributed`` 模块，``rows > shm_rows`` 时用它。
            dtype: 回传给模型的激活 dtype。
            shm_rows: 行数不超过它就走共享板。
            trace_limit: 记录前多少次调用的摘要；``0`` 表示不记录，引擎侧用 0，
                因为每次记录要多付一次主机侧求和。
            spin_limit: 共享板自旋上限（秒），到点抛 ``TimeoutError`` 而不是永久空转。
            timing_limit: 记录前多少次调用的「到达 / 等待」耗时；``0`` 表示不记录。
                探针量归约等待时打开，引擎侧保持 0（每次多两次 ``perf_counter``）。

        Raises:
            ValueError: ``world`` 或 ``shm_rows`` 非正。
        """
        # ``shm_rows == 0`` 是合法值，表示**永远不开共享板**（全部走 gloo）。
        # 它现在是 decode 的默认：作业 421/422 vs 423/416 两次对照给出
        # gloo 135.54/135.98 对共享板 142.51/141.15（均值 135.76 vs 141.83，**快 4.28%**）。
        # 共享板原本的 32.9 us/次是在「只计归约本身」的口径下测的；端到端它的阻塞 D2H
        # 会打断流水线，实测真实代价 37.98 ms（COFFIN_COMBINE_NOOP 对照），比 gloo 贵。
        if world <= 0 or shm_rows < 0:
            raise ValueError("world must be positive and shm_rows must be non-negative")
        self.path = path
        self.rank = int(rank)
        self.world = int(world)
        self.dist = dist
        self.dtype = dtype
        # env 覆盖：``COFFIN_COMBINE_FORCE_GLOO=1`` 把 ``shm_rows`` 压到 0，让**所有**行数
        # （含 decode 的 rows=1）都走 gloo 路径。用途是 A/B：模块头记着共享内存 32.9 us/次、
        # gloo 0.4-0.7 ms/次，但那是在「只计归约本身」的口径下测的；端到端（含阻塞 D2H
        # 打断流水线）实测 combine 值 37.98 ms。若 gloo 端到端更快，就直接换过去。
        forced = os.environ.get("COFFIN_COMBINE_FORCE_GLOO", "").strip() == "1"
        self.shm_rows = 0 if forced else int(shm_rows)

        # 设备侧归约：核直接读写一块被四个进程共享、且被 CUDA 映射过的锁页区，
        # 主机只轮询到达计数、**不调任何同步 API**。
        #
        # 为什么要它：主机侧交换每层阻塞一次 D2H 等本层算完，队列排空、流水线断裂。
        # 实测 combine 端到端代价 37.98 ms（48 层，COFFIN_COMBINE_NOOP 对照），
        # 换 gloo 后仍有 31.0 ms。这条路径把「等」的对象从设备算完换成计数到达，
        # 主机不再需要等设备。
        #
        # 只在 ``rows <= self.mapped_rows``（默认 1，即 decode）时启用：prefill 的
        # rows 从 80 起，映射区按最大行数开一次就够，但那条路本来就走 gloo 且够快。
        # 默认 **0 = 关闭**。作业 432 实测打开后四卡 decode 从 135.76 劣化到 963.35 ms
        # （7.1 倍）。**成因未逐项拆解**，最可能是 ``finish_kernel`` 从主机内存读四个槽：
        # 每层 4x2560 个 fp32 全走 PCIe 读，单次 cache line 往返是微秒级，48 层即吃掉全部收益。
        # 内核本身数值正确（MR_MAXERR 0.0），所以问题在**访存方向**而不是算法：
        # 该让核写进设备显存再做设备内归约，而不是让核去读主机内存。
        # 保留为 opt-in（COFFIN_COMBINE_MAPPED_ROWS=1），下一步按这个方向改。
        self.mapped_rows = int(os.environ.get("COFFIN_COMBINE_MAPPED_ROWS", "0"))
        self._mapped = None
        self._mapped_round = 0
        self._mapped_out = None
        self.trace_limit = int(trace_limit)
        self.spin_limit = float(spin_limit)
        self.timing_limit = int(timing_limit)
        self.tag = -1
        self.trace: list[tuple[int, int, float, float]] = []
        self.timing: list[tuple[int, int, int, float, float, float]] = []
        self.calls = 0
        self.calls_shm = 0
        self.calls_gloo = 0
        self.millis = 0.0
        self.millis_shm = 0.0
        self.millis_gloo = 0.0
        self._boards: dict[int, ShmAllReduce] = {}
        self._pairs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._host = torch.empty((0, HIDDEN), dtype=torch.float32, pin_memory=True)

    def _setup_mapped(self, rows: int):
        """建（必要时扩容）设备侧归约的共享映射区。

        区域布局：前 ``world * rows * HIDDEN`` 个 fp32 是各 rank 的槽，紧接
        ``world`` 个 int64 是到达计数。**这个布局必须与 ``mapped_reduce.cu`` 里的
        寻址一致**，改一边就得改另一边。

        为什么用文件 mmap 而不是 ``cudaHostAlloc``：后者分配的内存**无法跨进程映射**，
        四个进程必须看到同一份物理页，只能先 mmap 共享、再各自 ``cudaHostRegister``。

        Args:
            rows: 行数。只处理不超过 ``mapped_rows`` 的情形，更大的行数由调用方回落。

        Returns:
            ``(host_base, dev_base, total_bytes)``；``dev_base == 0`` 表示这张卡
            不支持映射锁页内存，调用方应回落到 gloo。
        """
        from backends.corex import _kernels
        total = self.world * rows * HIDDEN * 4 + self.world * 8
        total = (total + 4095) // 4096 * 4096
        path = f"{self.path}-mapped"
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            os.ftruncate(fd, total)
            mm = mmap.mmap(fd, total, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        # ``mmap`` 对象本身要留着：它被回收时页会失效，而设备指针还指着那里。
        host_base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        dev_base, detail = _kernels().mapped_register(host_base, total)
        self._mapped = {"mm": mm, "host": host_base, "dev": int(dev_base),
                        "rows": rows, "bytes": total, "detail": detail}
        return host_base, int(dev_base), total

    def _pinned(self, rows: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """取（必要时分配）这一档行数的锁页中转：``(stage, out, act)``。

        Args:
            rows: 行数。

        Returns:
            ``stage`` 承接本 rank 输入、``out`` 承接归约结果（都是 fp32），``act``
            是收窄回激活 dtype 的回传缓冲（锁页，才能让 H2D 真正异步）。
        """
        pair = self._pairs.get(rows)
        if pair is None:
            pair = (
                torch.empty((rows, HIDDEN), dtype=torch.float32, pin_memory=True),
                torch.empty((rows, HIDDEN), dtype=torch.float32, pin_memory=True),
                torch.empty((rows, HIDDEN), dtype=self.dtype, pin_memory=True),
            )
            self._pairs[rows] = pair
        return pair

    def _host_for(self, rows: int) -> torch.Tensor:
        """取（必要时扩容）gloo 路径的锁页中转。

        Args:
            rows: 行数。

        Returns:
            ``[rows, HIDDEN]`` 的 fp32 锁页张量，视图长度正好是 ``rows``。
        """
        if self._host.shape[0] < rows:
            self._host = torch.empty((rows, HIDDEN), dtype=torch.float32,
                                     pin_memory=True)
        return self._host[:rows]

    def _board(self, rows: int) -> ShmAllReduce:
        """取（必要时打开）这一档行数的共享板。

        Args:
            rows: 行数，必须不超过 ``shm_rows``。

        Returns:
            对应的 :class:`ShmAllReduce`。

        Raises:
            RuntimeError: 板文件不存在或尺寸不够。**在这里拦下而不是让
                ``ShmAllReduce`` 自己去 ftruncate**：那一步在四个 rank 同时发生时
                会互相截断，症状是 ``mmap length is greater than file size`` 或者
                更糟——四块板都建好但计数被清过，第一次归约就死等。
        """
        board = self._boards.get(rows)
        if board is not None:
            return board
        target = board_path(self.path, rows)
        total = ShmAllReduce.HEADER + self.world * rows * HIDDEN * 4
        size = os.path.getsize(target) if os.path.exists(target) else -1
        if size < total:
            raise RuntimeError(
                f"shared board {target} has {size} bytes, needs {total}: call "
                "prepare_shm_boards() on rank 0 and barrier before constructing")
        board = ShmAllReduce(target, self.rank, self.world, rows)
        self._boards[rows] = board
        return board

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """归约一次，返回同形状、同 dtype、同设备的张量。

        Args:
            x: ``[rows, hidden]`` 激活，设备上的张量。

        Returns:
            归约后的张量，语义同 ``all_reduce(SUM)``。

        Raises:
            ValueError: 形状与 ``hidden`` 不符（宽度不是 2560 时共享板按错误的
                步长寻址，数值会静默错位，所以在这里拦）。
            TimeoutError: 某个 rank 没在 ``spin_limit`` 内到达共享板。
        """
        if x.dim() != 2 or x.shape[-1] != HIDDEN:
            raise ValueError(f"combine expects [rows, {HIDDEN}], got {tuple(x.shape)}")
        if _COMBINE_NOOP:
            # 诊断：跳过跨 rank 归约与两次拷贝，只保留函数形状。数值是错的（少了三个 rank
            # 的贡献），**只用于量 combine 的端到端代价**，不能进任何功能验证。
            self.calls_shm += 1
            return x
        rows = int(x.shape[0])
        tracing = len(self.trace) < self.trace_limit
        timing = len(self.timing) < self.timing_limit
        started = time.perf_counter()
        # 设备侧归约优先：decode 的 rows=1 走这条。主机只轮询计数，不调同步 API。
        if rows <= self.mapped_rows and os.environ.get(
                "COFFIN_COMBINE_NO_MAPPED", "").strip() != "1":
            if self._mapped is None or self._mapped["rows"] < rows:
                self._setup_mapped(rows)
            if self._mapped["dev"] != 0:
                from backends.corex import _kernels
                kern = _kernels()
                if self._mapped_out is None or self._mapped_out.shape[0] < rows:
                    self._mapped_out = torch.empty((rows, HIDDEN), dtype=torch.float32,
                                                   device=x.device)
                self._mapped_round += 1
                kern.mapped_stage(x, self._mapped["dev"], self.world, self.rank,
                                  256, self._mapped_round)
                # 轮询到达计数：读的是普通共享内存，没有 CUDA 同步。
                counts = torch.frombuffer(self._mapped["mm"], dtype=torch.int64,
                                          count=self.world,
                                          offset=self.world * rows * HIDDEN * 4)
                deadline = time.perf_counter() + self.spin_limit
                while True:
                    if int(counts.min()) >= self._mapped_round:
                        break
                    if time.perf_counter() > deadline:
                        raise TimeoutError(
                            f"rank {self.rank} timed out at mapped round "
                            f"{self._mapped_round}")
                out_f = self._mapped_out[:rows]
                kern.mapped_finish(out_f, self._mapped["dev"], self.world, 4, 256)
                self.calls_shm += 1
                return out_f.to(self.dtype)

        if rows <= self.shm_rows:
            stage, out, act = self._pinned(rows)
            # 一次阻塞的 D2H 同时完成 dtype 收口：``copy_`` 允许跨 dtype。
            stage.copy_(x)
            local_sum = float(stage.sum()) if tracing else 0.0
            # 到达时刻＝本 rank 已经可以往板上写的时刻。它之后到返回之间的那一段才是
            # 「等最慢 rank」，之前那一段是「等本层设备算完」。
            arrive = time.perf_counter()
            self._board(rows)(stage, out, spin_limit=self.spin_limit)
            back = time.perf_counter()
            value = act.copy_(out).to(x.device, non_blocking=True)
            self.calls_shm += 1
            spent = (time.perf_counter() - started) * 1e3
            self.millis_shm += spent
            # 摘要和从**主机缓冲**取，不从回传的设备张量取：``float(value.sum())`` 会把
            # 刚排下去的 H2D 等干净，等于让被记录的那些调用多付一次设备同步，而它们
            # 正是拿来算单次微秒的样本。
            out_sum = float(out.sum()) if tracing else 0.0
            timed = (self.tag, rows, 0, arrive, (back - arrive) * 1e6,
                     (arrive - started) * 1e6) if timing else None
        else:
            host = self._host_for(rows)
            host.copy_(x)
            local_sum = float(host.sum()) if tracing else 0.0
            arrive = time.perf_counter()
            self.dist.all_reduce(host, op=self.dist.ReduceOp.SUM)
            back = time.perf_counter()
            value = host.to(x.dtype).to(x.device)
            self.calls_gloo += 1
            spent = (time.perf_counter() - started) * 1e3
            self.millis_gloo += spent
            out_sum = float(host.sum()) if tracing else 0.0
            timed = (self.tag, rows, 1, arrive, (back - arrive) * 1e6,
                     (arrive - started) * 1e6) if timing else None
        if tracing:
            self.trace.append((self.tag, rows, local_sum, out_sum))
        if timed is not None:
            self.timing.append(timed)
        self.calls += 1
        self.millis += spent
        return value

    def close(self) -> None:
        """释放所有共享板。

        不关会泄漏 mmap 与文件描述符；进程退出时操作系统会回收，但探针一个进程里
        可能跑好几档配置，攒着会先把文件描述符用光。
        """
        for board in self._boards.values():
            board.close()
        self._boards.clear()



