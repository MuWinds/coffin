"""同机多进程的全归约，两条路：主机暂存（:class:`EpShmReduce`）与设备侧（:class:`MappedReduce`）。

Module: ``coffin/models/qwen35``。EP 的通信底座：``moe.py`` 每个 MoE 层调一次
``combine``，除此之外没有调用方。除了 torch 与标准库，它只碰后端绑定里的
``mapped_*`` 那几个入口。

两条路的差别只有一个：**主机搬不搬数据**。

* :class:`EpShmReduce` 把载荷 D2H 进 mmap、主机累加、再 H2D。单次 30-100 µs，但它
  每次都等设备把队列排空，一步 decode 40 层就是 40 次全流同步，实测每层多付约
  445 µs（EP4 因此从 19.3 掉到比 PP4 的 29.6 tok/s 还慢）。
* :class:`MappedReduce` 让核直接读写同一块物理页（``/dev/shm`` 文件 +
  ``cudaHostRegister``），主机只轮询到达计数、求和由 ``mapped_finish`` 在设备上做完。
  这才是 EP 想要的东西；前提是这张卡支持映射锁页内存（``mapped_pinned_probe`` 验）。

Constraints:
    * 所有 rank 必须以**完全相同的顺序**调用同一个实例。协议里没有序号校验（省一次读），
      调用错开时不报错，只是把两轮的数据混在一起。
    * 一个板文件只给一个实例用。两个实例共用一个文件时，它们往同一组 arrival 格子上各写
      自己的 round，某一个 rank 的 ``want`` 会被另一块板提前满足，于是读到别人那一轮的数据
      ——不报错，只是数值错。载荷大小不同就开不同的文件。
    * 每个 rank 只写自己那一格计数，绝不读-改-写共享计数器：那组操作不是原子的，
      qwen4exp 的第一版就是四个 rank 读到同一个值再都写回 ``cur+1``，计数器永远到不了
      ``world * round``，实测直接死锁。
    * ``__init__`` 绝不 ftruncate。文件由启动器在**任何**进程启动之前建好：并发 ftruncate
      会把别人刚建好的文件截到 0，随后报文 ``mmap length is greater than file size``。
    * 自旋有上限，到点抛 ``TimeoutError``。一个 rank 挂了，其余的不该永久空转——那样外面
      的 ``timeout`` 连尸都收不到。
    * 槽里一律是 fp32。bf16 四个 rank 直接相加要丢两三位尾数，而这里省下的只是窄化那一次
      拷贝，不值得。
    * 可见性只保证 x86-64：先写载荷再写自己的到达计数，靠 TSO 的写序让对端「看到计数就
      一定看到载荷」。换到弱序架构（ARM）要在两次写之间补一条内存屏障。
"""

from __future__ import annotations

import ctypes
import gc
import mmap
import os
import struct
import time

import torch

#: 归约分段的累计秒表，只在 ``COFFIN_EP_TRACE=1`` 时记账、进程退出时打印一次。
#: 它回答的是「那 428 us/层到底花在 stage、同步、轮询还是 finish 上」——没有这个，
#: 只能靠改代码猜，而每改一次都要重编 20 多个内核。
_EP_TRACE = os.environ.get("COFFIN_EP_TRACE", "").strip() == "1"
_EP_STATS: dict[str, list[float]] = {}


def _ep_stat(key: str, dt: float) -> None:
    """记一次分段耗时；每 640 次（约一条 16 token 回复）打一次累计。

    不用 ``atexit`` 收尾：服务进程是被 kill 掉的，进程退出钩子根本不会跑。

    Args:
        key: 段名（``stage`` / ``sync`` / ``poll`` / ``finish`` / ``cast``）。
        dt: 该段耗时，秒。
    """
    row = _EP_STATS.get(key)
    if row is None:
        _EP_STATS[key] = [dt, 1]
    else:
        row[0] += dt
        row[1] += 1
    _EP_STATS.setdefault("_calls", [0.0, 0])[1] += 1
    calls = int(_EP_STATS["_calls"][1])
    if calls % 640 == 0:
        _ep_dump()


def _ep_dump() -> None:
    """打印分段累计与均值（累计值，比对相邻两行即可得到一段区间的平均）。"""
    for key, (total, n) in sorted(_EP_STATS.items()):
        if key == "_calls":
            continue
        print(f"[ep] {key}: n={n} total={total * 1e3:.1f}ms "
              f"avg={total / n * 1e6:.1f}us", flush=True)


if _EP_TRACE:
    # 开局打一行：没有它就看不出「统计没出来」是因为环境变量没进到 serve 进程，
    # 还是因为计数没到 640（上一次就吃了这个哑巴亏）。
    print("[ep] trace on", flush=True)


class EpShmReduce:
    """四进程共享内存全归约，``[1, 2560]`` 一次往返在几十微秒量级。"""

    def __init__(self, rank: int, world: int, path: str, payload_bytes: int) -> None:
        """打开已存在的共享板，并算好自己的槽位偏移。

        Args:
            rank: 本进程编号，决定写哪一格计数、哪一个槽。
            world: 进程数，也决定头部要几个 arrival 格子。
            path: 共享板文件路径。调用方必须在起任何进程之前建好并清零，长度至少
                ``(world + 1) * 8 + world * payload_bytes``。
            payload_bytes: 每个 rank 的载荷区容量，不是四个 rank 的合计。

        Raises:
            ValueError: ``world`` 非正、``rank`` 越界或 ``payload_bytes`` 非正。
            RuntimeError: 文件不存在，或比需要的短。这里绝不自己扩容：并发扩容正是
                上面那条死锁的同源问题。
        """
        if world <= 0:
            raise ValueError(f"world must be positive, got {world}")
        if not 0 <= rank < world:
            raise ValueError(f"rank {rank} out of range [0, {world})")
        if payload_bytes <= 0:
            raise ValueError(f"payload_bytes must be positive, got {payload_bytes}")

        self.rank = int(rank)
        self.world = int(world)
        self.path = path
        self.payload_bytes = int(payload_bytes)
        #: 头部 = world 个到达计数 + 一个 epoch，全部 int64。
        self.header_bytes = (self.world + 1) * 8
        self.total_bytes = self.header_bytes + self.world * self.payload_bytes
        #: 自旋上限（秒）。给得比一次 decode 步大三个数量级，只用来把死锁变成异常。
        self.spin_limit = 20.0
        self._round = 0
        self._calls = 0
        self._closed = False
        self._views: dict[tuple[int, int, int], tuple[torch.Tensor, ...]] = {}
        self._acc: torch.Tensor | None = None

        try:
            self._fd = os.open(path, os.O_RDWR)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"共享板不存在：{path}。启动器要先建好 {self.total_bytes} 字节") from exc
        size = os.fstat(self._fd).st_size
        if size < self.total_bytes:
            os.close(self._fd)
            raise RuntimeError(
                f"共享板太短：需要 {self.total_bytes} 字节"
                f"（头部 {self.header_bytes} + {self.world}×{self.payload_bytes}），"
                f"实际 {size} 字节")
        self._mm = mmap.mmap(self._fd, self.total_bytes)
        # 头部只取一次视图：自旋里每轮要按 rank 读 world 次，临时切片在这个尺度上是噪声。
        self._head = memoryview(self._mm)[:self.header_bytes]

    def _slot_views(self, slots: int, numel: int, index: int) -> tuple[torch.Tensor, ...]:
        """取本轮各 rank 槽位在本进程的零拷贝视图，按 rank 号排。

        Args:
            slots: 每个 rank 的载荷区切成几块。
            numel: 本次载荷的元素数（fp32）。
            index: 本轮用第几块，由调用次数对 ``slots`` 取模。

        Returns:
            长度为 ``world`` 的元组，第 ``r`` 项是 rank ``r`` 那一段 ``[numel]`` fp32。
        """
        key = (slots, numel, index)
        cached = self._views.get(key)
        if cached is not None:
            return cached
        slot_bytes = self.payload_bytes // slots
        first = self.header_bytes + index * slot_bytes
        views = tuple(
            torch.frombuffer(self._mm, dtype=torch.float32, count=numel,
                             offset=first + r * self.payload_bytes)
            for r in range(self.world))
        # 槽数被载荷撑开时缓存会无界增长，最多留八组：decode 只会用到一两组。
        if len(self._views) >= 8:
            self._views.pop(next(iter(self._views)))
        self._views[key] = views
        return views

    def _accumulator(self, numel: int) -> torch.Tensor:
        """取本进程的 fp32 累加缓冲，不够长就换一条更大的。

        Args:
            numel: 需要的元素数。

        Returns:
            ``[numel]`` 的 fp32 CPU 张量，复用上一次的分配。
        """
        acc = self._acc
        if acc is None or acc.numel() < numel:
            acc = torch.empty(numel, dtype=torch.float32)
            self._acc = acc
        return acc[:numel]

    def all_reduce_sum(self, x: torch.Tensor, slots: int | None = None) -> torch.Tensor:
        """把所有 rank 的 ``x`` 求和，返回同 dtype / shape / device 的新张量。

        Args:
            x: 本 rank 的贡献，浮点张量，CPU 或 CUDA 都行，非连续也可（会先拉平）。
            slots: 每个 rank 的载荷区切成几块轮转使用。省略时按容量推导，取 2 到 4 块：
                只有两块时上一轮的读和这一轮的写靠奇偶分开，更多块只是让不同步长的调用
                不撞同一段，再往上只浪费容量。

        Returns:
            求和结果，形状与 dtype 同 ``x``，device 也同 ``x``。是新分配的张量，调用方
            可以一直留着——下一次调用不会覆盖它。

        Raises:
            ValueError: ``x`` 不是浮点、``slots`` 非正、或载荷超出单槽容量。
            RuntimeError: 实例已 ``close()``。
            TimeoutError: 某个 rank 在 ``spin_limit`` 内没有到达。
        """
        if self._closed:
            raise RuntimeError("shared board already closed")
        if not x.is_floating_point():
            raise ValueError(f"all_reduce_sum needs floats, got {x.dtype}")

        numel = x.numel()
        if numel == 0:
            # 空张量也要走完两次 barrier，否则本 rank 和别人的调用序列错开一格。
            self.barrier()
            self.barrier()
            return x.clone()

        need = numel * 4
        if slots is None:
            slots = max(2, min(4, self.payload_bytes // need))
        slots = int(slots)
        if slots <= 0:
            raise ValueError(f"slots must be positive, got {slots}")
        slot_bytes = self.payload_bytes // slots
        if slot_bytes < need:
            raise ValueError(
                f"载荷 {need} 字节放不进 {slots} 个槽（每槽 {slot_bytes} 字节）："
                f"payload_bytes 至少要 {2 * need}，当前 {self.payload_bytes}")

        index = self._calls % slots
        self._calls += 1
        views = self._slot_views(slots, numel, index)
        # 写自己的槽。``copy_`` 顺便把 dtype（bf16→fp32）和 device（CUDA→CPU）一次转完，
        # 所以这里没有额外的一次中转。
        views[self.rank].copy_(x.reshape(-1))
        self.barrier()

        acc = self._accumulator(numel)
        # 按 rank 号顺序累加、只窄化一次：bf16 逐次累加在 2560 维上会偏。第一次用
        # ``out=`` 直接落地，省掉一次独立的拷贝调度——这里每次调度在 CPU 上是微秒级。
        if self.world == 1:
            acc.copy_(views[0])
        else:
            torch.add(views[0], views[1], out=acc)
            for slot in views[2:]:
                acc.add_(slot)
        # 第二次 barrier 不是冗余：没有它，跑得快的 rank 会在别人还没读完时写下一轮。
        self.barrier()

        out = torch.empty_like(x)
        out.copy_(acc.view(x.shape))
        return out

    def barrier(self) -> None:
        """让所有 rank 在这里对齐一次。

        Raises:
            TimeoutError: 某个 rank 在 ``spin_limit`` 内没有到达。
            RuntimeError: 实例已 ``close()``。
        """
        if self._closed:
            raise RuntimeError("shared board already closed")
        self._round += 1
        # 只写自己那一格：共享计数器上的读-改-写不是原子的，两个 rank 会互相覆盖。
        struct.pack_into("<q", self._head, self.rank * 8, self._round)
        want = self._round
        deadline = time.perf_counter() + self.spin_limit
        while True:
            behind = -1
            for i in range(self.world):
                if struct.unpack_from("<q", self._head, i * 8)[0] < want:
                    behind = i
                    break
            if behind < 0:
                break
            if time.perf_counter() > deadline:
                epoch = struct.unpack_from("<q", self._head, self.world * 8)[0]
                raise TimeoutError(
                    f"rank {self.rank} 在 round {want} 等 rank {behind} 没到；"
                    f"rank 0 停在 epoch {epoch}")
        if self.rank == 0:
            # 只给超时诊断看：它回答「队列是从第几轮开始没动的」，这是区分「别人卡住」
            # 和「整队根本没起来」的唯一线索。
            struct.pack_into("<q", self._head, self.world * 8, want)

    def close(self) -> None:
        """释放映射与文件描述符。可重复调用。"""
        if self._closed:
            return
        self._closed = True
        self._views.clear()
        self._acc = None
        # ``torch.frombuffer`` 的视图持有 mmap 的导出指针，引用计数掉下来才关得掉。
        # 关不掉就交给进程退出收尸，绝不在 close 里抛异常打断收尾。
        gc.collect()
        try:
            self._head.release()
        except (BufferError, ValueError):
            pass
        try:
            self._mm.close()
        except BufferError:
            # 关不掉就交给进程退出收尸：在这里抛只会打断收尾，而映射本来就是进程级资源。
            pass
        finally:
            os.close(self._fd)


class MappedReduce:
    """设备侧跨 rank 求和：核直接读写共享锁页区，主机只轮询到达计数。

    与 :class:`EpShmReduce` 的差别全在「主机等不等数据」：那条路要把载荷搬到主机、
    累加、再搬回去，每层一次全流同步（实测 445 µs/层 = 一步 decode 17.8 ms）；
    这条路主机只读 ``world`` 个 int64，乘加由 ``mapped_finish`` 在设备上完成。

    共享区的布局由本类与 ``mapped_reduce.cu`` 共同约定（改一边必须改另一边）：

    ``[buf0: world*rows*hidden fp32][counters0: world int64][buf1: ...][counters1: ...]``

    两块缓冲区按轮次交替，是为了让「跑得快的 rank 已经开始写下一层」不会覆盖
    「跑得慢的 rank 还在读的上一层」：每块缓冲区带自己的到达计数，互相不干扰。

    Constraints:
        * 共享区必须已存在且足够大；这里只映射，不扩容。
        * 载荷必须是 bf16 的二维连续 CUDA 张量——``mapped_stage`` 按位展开成 fp32，
          换成别的 dtype 会在内核入口被拒。
        * 每层一次往返，所以调用顺序必须各 rank 严格一致（与主机暂存那条同一条约束）。
    """

    def __init__(self, rank: int, world: int, path: str, rows: int, hidden: int,
                 spin: float = 20.0) -> None:
        """映射共享板，并把两半各自注册成设备可寻址的锁页内存。

        Args:
            rank: 本进程编号。
            world: 进程数。
            path: 共享板路径，启动器预先建好并清零。
            rows: 单次归约的最大行数（预填充分块的 token 数）。
            hidden: 隐藏宽度。
            spin: 轮询到达计数的上限秒数，只用来把死锁变成异常。

        Raises:
            ValueError: rank/world/rows/hidden 不合法。
            RuntimeError: 共享板不存在或太短；``mapped_register`` 拿不到设备指针。
        """
        if world <= 0 or not 0 <= rank < world:
            raise ValueError(f"rank {rank} out of range [0, {world})")
        if rows <= 0 or hidden <= 0:
            raise ValueError("rows and hidden must be positive")
        self.rank, self.world = int(rank), int(world)
        self.rows, self.hidden = int(rows), int(hidden)
        self.spin = float(spin)
        #: 等待策略。默认 ``host``：先 ``cudaDeviceSynchronize`` 再轮询。实测这是三条
        #: 里最快的一条（EP4 decode 20.3 tok/s 对设备侧自旋的 15.1、ixCCL 的 15.7、
        #: 纯忙等的不可用）——这个运行时上「设备侧自旋等对端写宿主内存」不进展，
        #: 而主机忙等又会被饿 20 ms，所以只剩「同步一次、然后几微秒轮询」这条路。
        #: ``device`` 保留给以后换驱动/换卡复测。
        self.host_wait = os.environ.get("COFFIN_EP_WAIT", "host").strip() != "device"
        #: 一块缓冲区里各 rank 槽位的字节数。
        self.slot_bytes = self.world * self.rows * self.hidden * 4
        #: 紧随其后的到达计数，每个 rank 一个 int64。
        self.counter_bytes = self.world * 8
        self.frame = self.slot_bytes + self.counter_bytes
        self.total_bytes = 2 * self.frame
        self._buf = 0
        self._rounds = [0, 0]

        try:
            self._fd = os.open(path, os.O_RDWR)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"mapped 共享板不存在：{path}。启动器要先建好 {self.total_bytes} 字节"
            ) from exc
        size = os.fstat(self._fd).st_size
        if size < self.total_bytes:
            os.close(self._fd)
            raise RuntimeError(
                f"mapped 共享板太短：需要 {self.total_bytes} 字节"
                f"（2 x ({self.world}x{self.rows}x{self.hidden} fp32 + 计数)），"
                f"实际 {size} 字节")
        self._mm = mmap.mmap(self._fd, self.total_bytes)

        from backends.corex import _kernels

        module = _kernels()
        base = ctypes.addressof(ctypes.c_char.from_buffer(self._mm))
        self._dev: list[int] = []
        for i in range(2):
            dev, detail = module.mapped_register(base + i * self.frame, self.frame)
            if dev <= 0:
                raise RuntimeError(
                    f"mapped_register 失败（第 {i} 半）：{detail}。这张卡不支持把"
                    f"mmap 区映射成设备可寻址的锁页内存时，EP 只能退回主机暂存")
            self._dev.append(int(dev))
        #: 设备侧求和结果，fp32；每步复用，避免每层一次分配。
        self._out = torch.empty((self.rows, self.hidden), dtype=torch.float32,
                                device="cuda")
        self._threads = 256
        self._blocks = max(1, min(512, (self.rows * self.hidden +
                                        self._threads * 4 - 1) // (self._threads * 4)))

    def _await(self, offset: int, want: int) -> None:
        """轮询本轮的到达计数，直到所有 rank 都到达 ``want``。

        轮询里必须定期让出主机。实测（``[1, 2048]``、4 rank）：纯忙等的版本每次要
        **20 ms** 才看到计数，而同一段代码先 ``cuda.synchronize()`` 再轮询就只要 2 µs
        ——设备写得很早，是 Python 忙等把下发/完成路径饿住了（四个进程在同一批核上
        死等，驱动线程拿不到时间推进）。每 64 轮 ``sleep(0)`` 一次就能让出，代价是
        每次等待多几微秒，换来的是 20 ms → 几十微秒。

        Args:
            offset: 本轮计数器在共享板里的字节偏移。
            want: 本轮的轮次值。

        Raises:
            TimeoutError: 超过 ``spin`` 秒还有 rank 没到。
        """
        deadline = time.perf_counter() + self.spin
        view = self._mm
        world = self.world
        spins = 0
        while True:
            ready = True
            for r in range(world):
                if struct.unpack_from("<q", view, offset + r * 8)[0] < want:
                    ready = False
                    break
            if ready:
                return
            spins += 1
            if not spins & 63:
                # ``sleep(0)`` 只让出时间片，不引入定时器开销；忙等版本没有这一行。
                time.sleep(0)
            if time.perf_counter() > deadline:
                raise TimeoutError(
                    f"rank {self.rank} 在轮次 {want} 等不到全部 rank 的到达计数；"
                    f"共享板 {self.fd_name}")

    def all_reduce_sum(self, x: torch.Tensor) -> torch.Tensor:
        """把各 rank 的 ``x`` 求和，返回同 dtype/shape/device 的新张量。

        Args:
            x: 本 rank 的贡献，bf16 ``[rows, hidden]``，连续 CUDA。

        Returns:
            求和结果，dtype 与 ``x`` 相同。

        Raises:
            ValueError: dtype/形状/宽度与初始化的约定不符。
            TimeoutError: 有 rank 没有到达本轮。
        """
        rows = int(x.shape[0])
        if x.dim() != 2 or int(x.shape[1]) != self.hidden:
            raise ValueError(
                f"mapped combine expects [rows, {self.hidden}], got "
                f"{tuple(x.shape)}")
        if rows > self.rows:
            raise ValueError(
                f"mapped combine payload {rows} rows exceeds the {self.rows} the "
                f"shared board was sized for; raise COFFIN_EP_ROWS")
        if x.dtype != torch.bfloat16:
            raise ValueError(f"mapped combine needs bf16, got {x.dtype}")
        from backends.corex import _kernels

        module = _kernels()
        buf = self._buf
        self._rounds[buf] += 1
        want = self._rounds[buf]
        dev = self._dev[buf]
        t0 = time.perf_counter()
        module.mapped_stage(x.contiguous(), dev, self.world, self.rank,
                            self._threads, want)
        t1 = time.perf_counter()
        # 到达计数紧跟在**本次载荷**的槽位之后（``mapped_reduce.cu`` 按
        # ``world * rows * hidden`` 算偏移，不是按共享板的容量），所以这里的偏移
        # 必须用实际的 ``rows`` 复算；按预分配容量算会永远等不到。
        off = buf * self.frame + self.world * rows * self.hidden * 4
        if self.host_wait:
            # 两个实测过的坑都在这里：纯忙等要被**饿 20 ms**（这个运行时把 kernel
            # 下发压在 flush 点上，主机不放手它就不放 kernel 出去）；所以先
            # ``cudaDeviceSynchronize`` 再轮询（那只剩 106 us）。代价是每层清空一次
            # 下发队列——这部分开销见 ``COFFIN_EP_TRACE`` 的分段秒表。
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            self._await(off, want)
            t3 = time.perf_counter()
            out = self._out[:rows]
            module.mapped_finish(out, dev, self.world, self._blocks, self._threads)
            t4 = time.perf_counter()
        else:
            # 设备侧等待：主机只下发 stage 与 finish。这条在这颗卡上**跑不起来**
            # （自旋读不到对端写，实测 15 tok/s），保留给换驱动/换卡复测。
            out = self._out[:rows]
            module.mapped_finish_wait(out, dev, self.world, self._blocks,
                                      self._threads, want)
            t2 = t3 = t4 = time.perf_counter()
        self._buf = 1 - buf
        result = out.to(x.dtype)
        if _EP_TRACE:
            t5 = time.perf_counter()
            _ep_stat("stage", t1 - t0)
            _ep_stat("sync", t2 - t1)
            _ep_stat("poll", t3 - t2)
            _ep_stat("finish", t4 - t3)
            _ep_stat("cast", t5 - t4)
        return result

    @property
    def fd_name(self) -> str:
        """共享板的文件名，只用于报错时说清是哪一块。"""
        try:
            return os.readlink(f"/proc/self/fd/{self._fd}")
        except OSError:
            return "<unknown>"

    def close(self) -> None:
        """解注册映射并释放 fd。可重复调用。"""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            from backends.corex import _kernels

            module = _kernels()
            base = ctypes.addressof(ctypes.c_char.from_buffer(self._mm))
            for i in range(2):
                module.mapped_release(base + i * self.frame)
        except Exception:                                     # noqa: BLE001
            # 解注册失败不该打断收尾：映射随进程退出一起消失。
            pass
        try:
            self._mm.close()
        except BufferError:
            pass
        finally:
            os.close(self._fd)


class P2pReduce:
    """四卡设备内存上的 all-to-all 归约，主机全程不参与。

    这是 EP 归约在这颗卡上唯一能跑满的那条路。宿主机映射内存那条要系统级栅栏，
    而 ``__threadfence_system()`` 在 clang-16/ivcore10 上会让 llc 崩（十行最小
    复现见 ``.tmp_stage/pd/ice/fence.cu``）；设备内存之间的跨卡写只要设备作用域
    ``__threadfence()``，它编得过。于是「写对端槽 → 置计数 → 自旋等齐 → 求和」
    可以压成一个核，一次发射，实测 ``[1, 2048]`` 一次归约 19.6-22.1 us，而主机侧
    同步那条即使在引擎里也要 927 us/层。

    布局：每个 rank 自己的显存里有一块 ``[world, rows, hidden]`` fp32 槽区和
    ``[world]`` int64 计数；各 rank 用 IPC 句柄互相映射，靠 :meth:`all_reduce_sum`
    里的那一个核完成交换。

    Constraints:
        * ``world`` 必须是 4：内核把四个指针用标量传，省掉一次设备指针数组的分配。
        * 载荷必须 bf16 且 ``rows <= rows_max``；fp32 累加在核内完成、只窄化一次。
        * 计数单调递增，同一个 :class:`P2pReduce` 实例不能跨请求复用轮次。

    Args:
        rank: 本进程编号。
        world: 进程数，只能是 4。
        rows: 共享槽区能容纳的最大行数（预填充分块大小决定）。
        hidden: 宽度。

    Raises:
        ValueError: ``world`` 不是 4，或 ``torch.distributed`` 没初始化（句柄换不出去）。
        RuntimeError: IPC 句柄或映射失败。
    """

    #: 内核支持的进程数。改成别的值要同时改 ``p2p_reduce.cu`` 的指针参数。
    WORLD = 4

    def __init__(self, rank: int, world: int, rows: int, hidden: int) -> None:
        if int(world) != self.WORLD:
            raise ValueError(f"p2p combine needs world={self.WORLD}, got {world}")
        import torch.distributed as dist

        if not dist.is_initialized():
            raise ValueError(
                "p2p combine exchanges IPC handles over torch.distributed, but no "
                "process group is initialized")
        from backends.corex import _kernels

        self.rank, self.world = int(rank), int(world)
        self.rows, self.hidden = int(rows), int(hidden)
        self._module = _kernels()
        self._round = 0
        total = self.rows * self.hidden
        self._slots = torch.zeros(self.world * total, dtype=torch.float32,
                                  device="cuda")
        self._flags = torch.zeros(self.world, dtype=torch.int64, device="cuda")
        self._out = torch.empty(self.rows, self.hidden, dtype=torch.float32,
                                device="cuda")
        self._slot_ptr = [0] * self.world
        self._flag_ptr = [0] * self.world
        self._slot_ptr[self.rank] = self._slots.data_ptr()
        self._flag_ptr[self.rank] = self._flags.data_ptr()
        slot_handles = self._exchange(self._module.p2p_handle(self._slots))
        flag_handles = self._exchange(self._module.p2p_handle(self._flags))
        for peer in range(self.world):
            if peer == self.rank:
                continue
            self._slot_ptr[peer] = self._module.p2p_open(slot_handles[peer])
            self._flag_ptr[peer] = self._module.p2p_open(flag_handles[peer])

    def _exchange(self, handle: torch.Tensor) -> list[torch.Tensor]:
        """把本 rank 的 IPC 句柄广播给所有人，收齐每个人的句柄。

        Args:
            handle: :meth:`p2p_handle <backends.corex._kernels.p2p_handle>` 产出的
                CPU uint8 张量。

        Returns:
            长度 ``world`` 的句柄张量列表，下标是 rank。

        Raises:
            RuntimeError: 收回来的句柄数量或大小不对（进程组配置错位）。
        """
        import torch.distributed as dist

        payload = [handle.clone() for _ in range(self.world)]
        dist.all_gather_object(payload, handle.clone())
        if len(payload) != self.world:
            raise RuntimeError(
                f"p2p handle exchange got {len(payload)} handles for "
                f"world={self.world}")
        return payload

    def _grid(self, total: int) -> tuple[int, int]:
        """按载荷大小挑网格：decode 一行也要用满 SM，预填充分块不超 64 块。

        Args:
            total: 元素数（``rows * hidden``）。

        Returns:
            ``(blocks, threads)``。
        """
        threads = 256
        blocks = max(1, min(64, (total + threads * 8 - 1) // (threads * 8)))
        return blocks, threads

    def all_reduce_sum(self, x: torch.Tensor) -> torch.Tensor:
        """把各 rank 的专家部分和加成本 rank 的完整输出。

        Args:
            x: 本 rank 的贡献，bf16 ``[rows, hidden]``，连续 CUDA。

        Returns:
            求和结果，dtype 与 ``x`` 相同。

        Raises:
            ValueError: dtype/形状超出约定，或行数超过槽区容量。
        """
        rows = int(x.shape[0])
        if x.dim() != 2 or int(x.shape[1]) != self.hidden:
            raise ValueError(
                f"p2p combine expects [rows, {self.hidden}], got {tuple(x.shape)}")
        if rows > self.rows:
            raise ValueError(
                f"p2p combine payload {rows} rows exceeds the {self.rows} the slot "
                f"area was sized for; raise COFFIN_EP_ROWS")
        if x.dtype != torch.bfloat16:
            raise ValueError(f"p2p combine needs bf16, got {x.dtype}")
        self._round += 1
        out = self._out[:rows]
        blocks, threads = self._grid(rows * self.hidden)
        self._module.p2p_combine(x.contiguous(), out, *self._slot_ptr,
                                 *self._flag_ptr, self.rank, self._round,
                                 blocks, threads)
        return out.to(x.dtype)

    def close(self) -> None:
        """释放槽区与计数。对端映射随进程退出一起消失。"""
        self._slots = None
        self._flags = None
        self._out = None
