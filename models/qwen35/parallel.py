"""按注意力头和 MLP 通道切分 Qwen3.5 的 W4 权重，并给出专家并行的归约件。

Module: ``coffin/models/qwen35``。权重加载器和模型组装调用此模块；算子通过
CoreX 整数投影完成。三种拓扑的跨 rank 数据面完全不同：``tp`` 走 Gloo 的 CPU
fp32 全归约（每层两次），``pp`` 走 UDP 点对点（每 token 几次），``ep`` 走同机
共享内存（每层一次，40 层/步 = 1.3 ms）。

Constraints:
    * 每个进程只暴露一张健康 GPU，使既有设备分配和 KV/循环状态自然局部化。
    * K 轴分片必须落在量化组边界；q/k/v 和卷积通道要分别切分后重新拼接。
    * 行并行默认共享激活 A8 scale；诊断可改用 rank 本地 scale，部分和仍用 fp32 归约。
    * 调度及前缀缓存操作须在各 rank 以相同顺序执行，不允许 CUDA graph。
    * ``ep`` 不切任何张量维度：dense、注意力、路由器和共享专家每卡各一份，
      只有专家栈按**专家号**切——前向的归约也就落成「每层一次求和」，而不是
      每个投影一次。任何按头/按通道的切分在 ``ep`` 下都是错的。
"""

from __future__ import annotations

from collections import OrderedDict
import json
import os
import re
import time
from pathlib import Path

import torch
import torch.distributed as dist

from int8_weights import QWeight, QWeightMoE
from w4a8_weights import W4A8Weight


class ParallelContext:
    """一个 rank 的主机通信上下文，进程组由启动器创建。"""

    def __init__(self, rank: int, world_size: int, mode: str = "tp") -> None:
        """核对进程组身份，防止按错误的 rank 读取权重。

        Args:
            rank: 当前进程编号。
            world_size: 并行进程数。
            mode: ``"tp"`` 按张量切分、``"pp"`` 按层切分、``"ep"`` 按专家切分。
                三者共用同一个 Gloo 组，但那个组在 ``pp``/``ep`` 下只用来广播请求，
                数据面各走各的（UDP 点对点 / 同机共享内存）。

        Raises:
            RuntimeError: Gloo 进程组尚未初始化或身份不符。
            ValueError: ``mode`` 不是 ``tp`` / ``pp`` / ``ep``，或 ``ep`` 的共享区
                参数不成立。
        """
        if (not dist.is_initialized() or dist.get_backend() != "gloo"
                or dist.get_world_size() != world_size or dist.get_rank() != rank):
            raise RuntimeError("tensor parallelism requires a matching CPU Gloo group")
        if mode not in {"tp", "pp", "ep"}:
            raise ValueError(
                f"parallel mode must be 'tp', 'pp' or 'ep', got {mode!r}")
        self.rank, self.world_size = rank, world_size
        self.mode = mode
        self._buffers = OrderedDict()
        #: 流水载荷的主机中转缓冲，key 是 ``(numel, dtype)``。每步复用，避免
        #: 每个 token 都新分配一次主机内存（实测一层的 CPU 侧有 8 次 ``empty``）。
        self._relay_hosts = OrderedDict()
        #: pinned 主机内存是否可用，首次 relay 时探一次（部分厂商后端不支持）。
        self._pinned_ok = None
        #: 段间通信的累计秒数与次数。流水把每 token 拆成几次点对点，而"卡在哪"和
        #: "算得慢"在墙上时间上是同一副面孔；这两个计数器让两者可以分开看。
        self.comm_s = 0.0
        self.comm_calls = 0
        self.sum_calls = 0
        self.max_calls = 0
        #: 流水模式的点对点链路；张量并行下为 ``None``，那条路只用 all-reduce。
        self.link = None
        if mode == "pp":
            self.link = PipeLink(rank, world_size,
                                 int(os.environ.get("COFFIN_PP_PORT", 29791)), self)
        #: 专家并行的同机归约件；``tp`` / ``pp`` 下为 ``None``。
        #:
        #: 走共享内存而不是 gloo 是实测的决定：4 个 rank 全在同一台机器上，gloo 一次
        #: 归约 291-420 us（几乎全是固定开销），共享内存 barrier 只要 32.9 us。40 层
        #: 一步就是 14-20 ms 对 1.3 ms —— 前者会把专家并行省下的 0.613 ms/层全部吃掉。
        self.ep = None
        #: ``ep`` 的设备侧集合通信组（ixCCL/NCCL）。有它时 :meth:`combine` 走设备，
        #: 主机不再逐层等归约结果——这一点是 EP 能不能用的分水岭，见 :meth:`combine`。
        self._ep_group = None
        #: ``ep`` 的映射锁页内存归约（``mapped_reduce.cu``）。比集合通信组更省：主机
        #: 只轮询计数，连集合调用的 390 µs 固定开销都没有。
        self._mapped = None
        if mode == "ep":
            kind = os.environ.get("COFFIN_EP_COLLECTIVE", "auto")
            if kind not in {"auto", "mapped", "nccl", "shm"}:
                raise ValueError(
                    "COFFIN_EP_COLLECTIVE must be auto / mapped / nccl / shm, got "
                    f"{kind!r}")
            if kind in {"auto", "mapped"}:
                self._mapped = _ep_mapped(rank, world_size, kind)
            if self._mapped is None and kind in {"auto", "nccl"}:
                self._ep_group = _ep_device_group(world_size, kind)
            if self._mapped is None and self._ep_group is None:
                from models.qwen35.ep import EpShmReduce

                #: 共享区由启动器预先建好，这里只负责映射；载荷容量要盖住一个预填充
                #: 分块的 ``[tokens, hidden]`` fp32（2048 token 是 20 MiB，双缓冲 40 MiB）。
                self.ep = EpShmReduce(
                    rank, world_size,
                    os.environ.get("COFFIN_EP_SHM", "/dev/shm/coffin_ep.bin"),
                    int(os.environ.get("COFFIN_EP_SLOT_BYTES", str(64 << 20))))
        self.scale_mode = os.environ.get("COFFIN_TP_SCALE_MODE", "shared")
        if self.scale_mode not in {"shared", "local", "decode_local"}:
            raise ValueError(
                "COFFIN_TP_SCALE_MODE must be 'shared', 'local', or 'decode_local'")

    def reduce(self, tensor: torch.Tensor, *, maximum: bool = False) -> torch.Tensor:
        """在 CPU 上以 fp32 做全归约，回传本 rank 当前流。

        Args:
            tensor: 本 rank 的浮点输入。
            maximum: 求最大值；否则求和。

        Returns:
            与输入同设备、同形状的 fp32 结果。
        """
        key = (tuple(tensor.shape), maximum)
        entry = self._buffers.pop(key, None)
        if entry is None:
            if len(self._buffers) >= 4:
                self._buffers.popitem(last=False)
            entry = torch.empty(tensor.shape, dtype=torch.float32, pin_memory=True)
        self._buffers[key] = entry
        host = entry
        host.copy_(tensor)
        dist.all_reduce(host, op=dist.ReduceOp.MAX if maximum else dist.ReduceOp.SUM)
        result = host.to(tensor.device)
        if maximum:
            self.max_calls += 1
        else:
            self.sum_calls += 1
        return result

    def relay(self, tensor: torch.Tensor, dst: int) -> None:
        """把一块激活原样送到下一个流水段。

        gloo 只认 CPU 张量，而且不支持 bf16，所以这里经一次 fp32 主机中转。代价是
        实测出来的选择：比 all-reduce 少两个数量级的次数，这点转换成本可以忽略。

        Args:
            tensor: 要送出的激活，``[tokens, hidden]``。
            dst: 目标 rank。
        """
        if self.link is not None:
            # 直接按模型 dtype（bf16）过线：载荷字节减半，并且省掉一次设备侧
            # bf16->fp32 的转换核。pinned 缓冲让 D2H 走 DMA；``non_blocking``
            # 之后必须显式同步当前流，否则读到的是没搬完的旧内容。
            host = self._host_buffer(tensor.numel(), _wire_dtype(tensor.dtype))
            host.copy_(tensor.reshape(-1), non_blocking=True)
            torch.cuda.current_stream().synchronize()
            self.link.send(host, dst)
            return
        # 非流水路径（gloo）只认 CPU 张量、且不支持 bf16，仍经 fp32 中转。
        host = tensor.detach().to(torch.float32).cpu().contiguous()
        t0 = time.perf_counter()
        dist.send(host, dst=dst)
        self.comm_s += time.perf_counter() - t0
        self.comm_calls += 1

    def _host_buffer(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        """取一块长度为 ``numel``、dtype 为 ``dtype`` 的主机中转缓冲（复用）。

        主机内存在这里按 ``(numel, dtype)`` 缓存：一层要发 4 次，每次新分配
        既慢又会在 pageable 上多付一次中间暂存。pinned 是否可用只探一次。

        Args:
            numel: 元素个数，决定缓冲长度。
            dtype: 缓冲的 dtype，决定缓冲身份。

        Returns:
            一维的主机缓冲。
        """
        key = (numel, dtype)
        entry = self._relay_hosts.pop(key, None)
        if entry is None:
            if len(self._relay_hosts) >= 4:
                self._relay_hosts.popitem(last=False)
            if self._pinned_ok is None:
                self._pinned_ok = _pin_probe()
            entry = torch.empty(numel, dtype=dtype, pin_memory=self._pinned_ok)
        self._relay_hosts[key] = entry
        return entry

    def await_relay(self, shape: tuple, src: int, device, dtype) -> torch.Tensor:
        """从上一段/末段收一块激活并送进设备。

        Args:
            shape: 期望形状。
            src: 来源 rank。
            device: 目标设备。
            dtype: 返回 dtype。

        Returns:
            该形状的设备张量。
        """
        if self.link is not None:
            # ``dtype`` 同时用于校验线上字节数：两侧 dtype 不一致时会在这里报
            # 「payload 字节数与期望不符」，而不是拿错位的数据继续算。
            return self.link.recv(tuple(shape), src, _wire_dtype(dtype)).to(
                device=device, dtype=dtype)
        t0 = time.perf_counter()
        host = torch.empty(tuple(shape), dtype=torch.float32)
        dist.recv(host, src=src)
        out = host.to(device=device, dtype=dtype)
        self.comm_s += time.perf_counter() - t0
        self.comm_calls += 1
        return out

    def layer_range(self, num_layers: int) -> tuple[int, int]:
        """本 rank 负责的 decoder 层区间 ``[begin, end)``。

        层数除不尽时前面的段多拿一层：40 层 4 段正好每段 10 层，而 41 层时第一段拿 11。

        Args:
            num_layers: 总层数。

        Returns:
            ``(begin, end)``，半开区间。
        """
        if self.mode != "pp":
            return 0, num_layers
        size, extra = divmod(num_layers, self.world_size)
        begin = self.rank * size + min(self.rank, extra)
        end = begin + size + (1 if self.rank < extra else 0)
        return begin, end

    def expert_range(self, num_experts: int) -> tuple[int, int]:
        """本 rank 装哪一段专家，``[begin, end)``。

        按专家号**连续**切，而不是交错切（``e % world == rank``）。两者负载期望相同，
        但连续切让盘上的 ``[E, K, N]`` 栈可以直接按首轴读一段（safetensors 的
        ``get_slice``），交错切则要整栈读进来再挑行——每层 805 MB，40 层付不起。

        Args:
            num_experts: 全局专家总数。

        Returns:
            ``(begin, end)``；``tp`` / ``pp`` 下是全部 ``(0, num_experts)``，因为那两个
            拓扑的专家是完整的。
        """
        if self.mode != "ep":
            return 0, num_experts
        if num_experts < self.world_size:
            raise ValueError(
                f"{num_experts} experts cannot be split across {self.world_size} ranks")
        size, extra = divmod(num_experts, self.world_size)
        begin = self.rank * size + min(self.rank, extra)
        return begin, begin + size + (1 if self.rank < extra else 0)

    def combine(self, value: torch.Tensor) -> torch.Tensor:
        """专家并行下把各 rank 的部分和加起来。

        每个 rank 只算自己那 1/4 专家命中的槽位，未命中的槽位是零行，所以这里必须是
        **求和**。漏掉它得到的是「每条 token 只剩 1/4 专家」的安静错误——形状全对，
        输出是另一个模型。

        三条路，按实测好坏排：

        1. **映射锁页内存**（``mapped_reduce.cu``）：核直接读写共享区，主机只轮询
           一个 int64 计数，乘加在设备上。这是唯一不付「每次调用 390 µs」的那条。
        2. **设备侧集合通信**（ixCCL/NCCL）：归约留在卡上，但每次调用有约 390 µs 的
           固定开销（实测 10 KB 与 640 KB 同价），一步 decode 40 层就是 15.6 ms。
        3. **主机暂存**（共享内存 barrier）：单次 30-100 µs 看着最快，但它每次都要等
           设备把队列排空，40 层就是 40 次全流同步，实测每层多付约 445 µs——EP4 因此
           从 19.3 掉到比 PP4 的 29.6 tok/s 还慢。

        ``fp32`` 累加、只窄化一次，是因为四个 bf16 直接相加会丢两三位尾数。

        Args:
            value: 本 rank 的部分和，``[tokens, hidden]``，浮点。

        Returns:
            同 dtype、同形状的跨 rank 和。

        Raises:
            ValueError: ``ep`` 上下文两条路都没有建起来（配置错位）。
        """
        if self.mode != "ep":
            return value
        if self._mapped is not None:
            return self._mapped.all_reduce_sum(value)
        if self._ep_group is not None:
            acc = value.to(torch.float32)
            dist.all_reduce(acc, op=dist.ReduceOp.SUM, group=self._ep_group)
            return acc.to(value.dtype)
        if self.ep is None:
            raise ValueError(
                "expert parallel context has neither a mapped region, a device "
                "collective group, nor a shared-memory arena")
        return self.ep.all_reduce_sum(value)


def _ep_mapped(rank: int, world_size: int, kind: str):
    """建一个映射锁页内存的归约器（设备侧求和），拿不到就返回 ``None``。

    这是三条路里唯一不付集合调用开销的：主机只轮询计数，乘加在设备上。

    Args:
        rank: 本进程编号。
        world_size: 进程数。
        kind: ``auto`` 失败时回落并打印原因；``mapped`` 失败时抛。

    Returns:
        :class:`~models.qwen35.ep.MappedReduce` 或 ``None``。

    Raises:
        RuntimeError: ``kind="mapped"`` 而没有配置共享板或注册失败。
    """
    path = os.environ.get("COFFIN_EP_MAPPED", "").strip()
    if not path:
        if kind == "mapped":
            raise RuntimeError(
                "COFFIN_EP_COLLECTIVE=mapped needs COFFIN_EP_MAPPED=<共享板路径>")
        return None
    from models.qwen35.ep import MappedReduce

    rows = int(os.environ.get("COFFIN_EP_ROWS", "2048"))
    hidden = int(os.environ.get("COFFIN_EP_HIDDEN", "2048"))
    try:
        reducer = MappedReduce(rank, world_size, path, rows, hidden)
    except Exception as exc:                                  # noqa: BLE001
        if kind == "mapped":
            raise
        print(f"[ep] mapped reduce unavailable ({type(exc).__name__}: "
              f"{str(exc)[:160]}); falling back", flush=True)
        return None
    if rank == 0:
        print(f"[ep] mapped reduce ready: world={world_size} rows={rows} "
              f"hidden={hidden} board={path}", flush=True)
    return reducer


def _ep_device_group(world_size: int, kind: str):
    """为专家并行建一个设备侧集合通信组，拿不到就返回 ``None``。

    这台卡（天数智芯 BI-V100）的厂商 torch 把 ``nccl`` 接到 ixCCL 上，实测可用；
    ``xccl``/``cxcl`` 都不是它的名字（实测报 ``Unknown c10d backend type XCCL``）。
    请求广播仍走 gloo 的默认组，两个组可以并存——本组只服务 MoE 的归约。

    Args:
        world_size: 组内 rank 数，必须等于并行进程数。
        kind: ``auto`` 先试设备侧、失败回落共享内存；``nccl`` 强制设备侧（建不起来
            就抛）；``shm`` 强制共享内存（对照用）。

    Returns:
        ``ProcessGroup`` 或 ``None``。

    Raises:
        RuntimeError: ``kind="nccl"`` 而设备侧组建不起来。
    """
    if kind in {"shm", "none"}:
        return None
    if not dist.is_initialized():
        return None
    try:
        group = dist.new_group(ranks=list(range(world_size)), backend="nccl")
    except Exception as exc:                                  # noqa: BLE001
        if kind == "nccl":
            raise RuntimeError(
                f"COFFIN_EP_COLLECTIVE=nccl but the device collective group could "
                f"not be created: {type(exc).__name__}: {exc}") from exc
        print(f"[ep] device collective unavailable ({type(exc).__name__}: "
              f"{str(exc)[:120]}); falling back to the shm arena", flush=True)
        return None
    return group


class PipeLink:
    """同一节点上各流水段之间的点对点链路，走 UDP。

    不用 ``torch.distributed`` 的 gloo 传输：它对 4 KB 的小消息实测 **30 ms 一次**，而且与
    payload 大小无关（把回传从 1 MB 的 logits 换成 4 KB 的 hidden，lap 一点没变）——这是
    Nagle 与延迟 ACK 互相等待的签名。流水每 token 有四次这样的消息，120 ms 全花在等 ACK 上。

    UDP 是这里最省事的选择：四个 rank 全在同一台机器上，loopback 上 8 KB 一个包不会分片，
    也不用握手（TCP 那套 connect/accept 在四进程同时启动时还要额外的重试与配对逻辑）。
    一个 deltas 大小 8200 B，远在 65507 B 的 UDP 载荷上限之内。

    Constraints:
        * 所有 rank 必须同机；跨机要回到 gloo，这套链路只适用于单节点部署。
        * 靠调用顺序配对，协议里没有序号：流水是严格锁步的，每条 (src, dst) 上同时只有
          一个包在飞。丢包会撞上 30 s 的接收超时并响亮地失败，而不是给出错序的数据。
    """

    def __init__(self, rank: int, world_size: int, base_port: int, owner) -> None:
        """绑定本 rank 的 UDP 端口。

        Args:
            rank: 当前 rank。
            world_size: 流水段数。
            base_port: 端口基址，rank r 绑定 ``base_port + r``。
            owner: 持有本链路的 :class:`ParallelContext`；计数器写在它身上，这样探针
                只需要读一个地方。

        Raises:
            OSError: 端口被占用。
        """
        import socket

        self.rank, self.world_size = rank, world_size
        self._owner = owner
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 默认 208 KB 的接收缓冲在预填充的长分块（[C, hidden] 最多 16 MB）下会溢出让内核
        # 丢包；锁步协议没有重传，丢包就是挂住，所以这里把缓冲放大到 32 MB。
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32 << 20)
        self.sock.bind(("127.0.0.1", base_port + rank))
        self.sock.settimeout(30.0)
        #: 单个数据报的载荷。32 KB 在 loopback 上一次拷贝的代价可以忽略，也不会撞上
        #: 64 KB 的 UDP 上限。
        self._MTU = 32 * 1024
        #: 每发这么多个分片等一次 ack。``SO_RCVBUF`` 写 32 MB 会被内核按 ``rmem_max``
        #: （常见 208 KB）静默钳回去，所以「一口气发完 4 MB」必然丢包 —— 丢了就是 30 s
        #: 接收超时。窗口取 4（128 KB 在飞）留出安全余量，代价是每 128 KB 一次往返。
        self._WINDOW = 4
        #: 收到但当前调用不等它的报文。等 ack 时可能先收到对端下一条数据的头，
        #: 直接丢弃会让后续 ``_await_header`` 永远等不到 —— 多块预填充必然会撞上，
        #: 因为发送是重叠的。所以按类型取，不匹配的原样留在队列里。
        self._inbox: list[bytes] = []
        #: 超过这个长度的载荷改走 ``/dev/shm``：65507 是 UDP 单包上限，留出头的余量。
        self._UDP_MAX = 60 * 1024
        self._port = base_port
        self._addr = {r: ("127.0.0.1", base_port + r) for r in range(world_size)}

    def _spill(self, src: int, dst: int) -> str:
        """本对 (src, dst) 的大载荷落盘路径。

        ``/dev/shm`` 是 tmpfs，同机进程共享同一份页缓存，写一次读一次就是共享内存传递；
        文件名带路由方向，四段全互联时每条单向链路各有一份，不会互相覆盖。

        Args:
            src: 发送方 rank。
            dst: 接收方 rank。

        Returns:
            文件路径。
        """
        return f"/dev/shm/coffin_pp_{self._port}_{src}_{dst}.bin"

    def _take(self, want: bytes) -> bytes:
        """取第一条类型为 ``want`` 的报文载荷，其余按原顺序留在收件箱里。

        Args:
            want: 期望的类型字节，``b"H"`` / ``b"D"`` / ``b"A"``。

        Returns:
            该报文去类型字节后的载荷。
        """
        held = []
        while True:
            if self._inbox:
                data = self._inbox.pop(0)
            else:
                data, _ = self.sock.recvfrom(65507)
            if data[:1] == want:
                self._inbox[:0] = held
                return data[1:]
            held.append(data)

    def _await_header(self) -> int:
        """等到一条 ``H`` 报文，返回它带的总长（负数表示载荷在 ``/dev/shm``）。"""
        # ``signed=True`` 是必须的：负的总长是「载荷在 /dev/shm 里」的标记，
        # 按无符号解会变成接近 2^64 的天文数字。
        return int.from_bytes(self._take(b"H"), "little", signed=True)

    def _await_data(self) -> bytes:
        """等到一条 ``D`` 报文，返回它的载荷。"""
        return self._take(b"D")

    def _await_ack(self) -> None:
        """等到一条 ``A`` 应答；其余报文（别的对端的数据/头）原地丢弃不合适，

        所以我们只在「本 rank 只可能收到应答」的时机调用它 —— 流水的发送是严格
        串行的，同一时刻一个 rank 只会有一条自己的请求在等应答。
        """
        self._take(b"A")

    def send(self, host: torch.Tensor, dst: int) -> None:
        """把一块**主机侧** float32 连续张量切成若干数据报送出。

        分块预填充的 hidden 是 ``[C, H]``，C=512 就是 4 MB，而单个 UDP 载荷上限是 65507 B，
        所以大块要分片。协议是先发一个 8 字节的总长头，再发若干 ``_MTU`` 大小的分片；
        接收侧按顺序拼回来。loopback 上同一条连接的包序是保留的，因此这个顺序就是全部
        的排序信息，不需要序号字段。

        Args:
            host: CPU 上连续的张量（流水载荷是 bf16）。
            dst: 目标 rank。
        """
        # 走 ``uint8`` 视图而不是 ``host.numpy()``：numpy 没有 bf16，直接
        # ``numpy()`` 会抛 TypeError；视图是按字节发送，与 dtype 无关。
        payload = host.view(torch.uint8).numpy().tobytes()
        addr = self._addr[dst]
        t0 = time.perf_counter()
        # 每条报文带一个类型字节：``H`` 头、``D`` 数据、``A`` 应答。多块预填充时
        # 不同对端的报文会交错到达同一个 socket，按类型等就不会把 ack 当成头、
        # 也不会把某个对端的头当成另一个对端的应答。
        if len(payload) <= self._UDP_MAX:
            self.sock.sendto(b"H" + len(payload).to_bytes(8, "little", signed=True),
                             addr)
            self.sock.sendto(b"D" + payload, addr)
        else:
            # 负的总长表示「载荷在共享内存里」，正数表示「跟着下一个数据报走」。
            with open(self._spill(self.rank, dst), "wb") as handle:
                handle.write(payload)
            # ``to_bytes`` 默认无符号，负长度必须显式 ``signed=True``，否则这一句直接
            # 抛 "can't convert negative int to unsigned"，而报错发生在发送侧、
            # 客户端只看到 KeyError 'text'，很容易误判成接收端的问题。
            self.sock.sendto(b"H" + (-len(payload)).to_bytes(8, "little", signed=True),
                             addr)
        self._await_ack()
        self._owner.comm_s += time.perf_counter() - t0
        self._owner.comm_calls += 1

    def recv(self, shape: tuple, src: int, dtype=torch.float32) -> torch.Tensor:
        """收一块已知形状的张量。

        Args:
            shape: 期望形状；长度取自线上的 8 字节前缀并核对。
            src: 来源 rank。
            dtype: 载荷 dtype，流水模式下是 bf16。它同时是长度校验的依据——
                两侧 dtype 不一致会在这里报字节数不符，避免拿错位数据继续算。

        Returns:
            主机侧连续的同 dtype 张量。

        Raises:
            RuntimeError: 线上长度与期望形状不符（两侧协议不同步）。
            TimeoutError: 30 s 内没收到包（丢包或对端没在跑）。
        """
        t0 = time.perf_counter()
        count = self._await_header()
        if count < 0:
            with open(self._spill(src, self.rank), "rb") as handle:
                payload = handle.read(-count)
        else:
            payload = self._await_data()
        if len(payload) != abs(count):
            raise RuntimeError(
                f"pipe link: got {len(payload)} bytes, header said {abs(count)}")
        width = int(torch.empty((), dtype=dtype).element_size())
        expected = width * int(torch.tensor(shape).prod()) if shape else width
        if abs(count) != expected:
            raise RuntimeError(
                f"pipe link: payload {abs(count)} bytes, expected {expected}")
        self.sock.sendto(b"A", self._addr[src])
        self._owner.comm_s += time.perf_counter() - t0
        self._owner.comm_calls += 1
        buf = torch.frombuffer(bytearray(payload), dtype=torch.uint8).view(dtype)
        return buf.reshape(tuple(shape))

def _wire_dtype(dtype: torch.dtype) -> torch.dtype:
    """流水载荷在线上使用的 dtype。

    默认就是模型 dtype（bf16），载荷字节比 fp32 少一半、也不需要一次设备侧
    dtype 转换。``COFFIN_PP_RELAY_F32=1`` 回到改动前的 fp32 中转，只用于同
    二进制 A/B 对照（两侧都必须读同一个开关，否则字节数校验会报不一致）。

    Args:
        dtype: 激活本身的 dtype。

    Returns:
        线上 dtype。
    """
    if os.environ.get("COFFIN_PP_RELAY_F32") == "1":
        return torch.float32
    return dtype


def _pin_probe() -> bool:
    """探测这套后端是否支持 pinned 主机内存。

    ``pin_memory`` 依赖厂商后端实现，不可用时 ``torch.empty`` 会抛异常。这里
    一次性探好并缓存，避免每个 token 都付一次异常开销；探不到就退回 pageable，
    只是慢一点，不影响正确性。

    Returns:
        可用为 ``True``。
    """
    try:
        probe = torch.empty(8, dtype=torch.bfloat16, pin_memory=True)
        probe.zero_()
        return bool(probe.is_pinned())
    except Exception:
        return False


_context: ParallelContext | None = None


def configure(rank: int, world_size: int, mode: str = "tp") -> ParallelContext:
    """在加载模型前绑定已初始化的 Gloo 进程组。

    Args:
        rank: 当前进程编号。
        world_size: 并行进程数。
        mode: 拓扑，``tp`` / ``pp`` / ``ep``。Gloo 组在三种拓扑下都要建，因为请求
            广播走它；``ep`` 的数据面另走共享内存。

    Returns:
        此进程的上下文，之后所有层共享它。

    Raises:
        RuntimeError: 进程组不匹配，或已经绑定了上下文。
    """
    global _context
    if _context is not None:
        raise RuntimeError("parallel context is already configured")
    _context = ParallelContext(rank, world_size, mode)
    return _context


def context() -> ParallelContext | None:
    """未配置时返回 None，单卡模型保持原有组装路径。"""
    return _context


def local_config(cfg: dict, world_size: int) -> dict:
    """把注意力头数和中间通道变成局部尺寸，隐藏宽度保持全局尺寸。

    Args:
        cfg: checkpoint 文本配置。
        world_size: 分片数。

    Returns:
        独立的局部配置，不修改 checkpoint 配置。

    Raises:
        ValueError: 任意头数或中间通道不能整除并行数。
    """
    if world_size < 1:
        raise ValueError("positive world_size required")
    parallel = context()
    if parallel is not None and parallel.mode != "tp":
        # 流水并行不切张量：头数、中间宽度都保持全局值，只有层号范围变。
        return dict(cfg)
    result = dict(cfg)
    # MoE 的两个中间宽度也要一起切：专家的 gate/up 按输出轴分片、down 按 K 轴分片，
    # 局部宽度不对会让 ``SparseMoE`` 建出一个和权重对不上的融合栈。稠密 checkpoint
    # 没有这两个键，按存在性跳过而不是按族名分支。
    for key in ("num_attention_heads", "num_key_value_heads", "linear_num_key_heads",
                "linear_num_value_heads", "intermediate_size", "moe_intermediate_size",
                "shared_expert_intermediate_size"):
        if key not in cfg:
            continue
        value = int(cfg[key])
        if value % world_size == 0:
            result[key] = value // world_size
        elif key == "num_key_value_heads" and world_size % value == 0:
            # GQA 的 KV 头可以少于卡数：Qwen3.5-MoE 是 2 个 KV 头对 16 个 query 头
            # （rep=8），4 卡时一个 KV 头由 2 张卡共享，每张卡只持有它自己那一个。
            #
            # 不能把两个 KV 头都复制给每张卡：Q 头按块分片后 rank r 拿的是全局头
            # [4r, 4r+4)，而全局 rep=8 让 rank 0/1 只用到 KV 头 0、rank 2/3 只用到
            # KV 头 1。复制两头的写法会让局部 rep 变成 4/2=2，rank 1 和 rank 3 的
            # attention 就配错了 KV——形状全对，输出是垃圾。
            result[key] = 1
        else:
            raise ValueError(f"{key}={value} is not divisible by world_size={world_size}")
    heads = int(result["num_attention_heads"])
    kv_heads = int(result["num_key_value_heads"])
    if heads % kv_heads:
        raise ValueError(
            f"local query heads {heads} must be a multiple of KV heads {kv_heads}: "
            "the attention kernel rebuilds each KV head for an integral number of queries")
    return result


def _ranges(widths: list[int], rank: int, world_size: int) -> list[tuple[int, int]]:
    """分别切分融合矩阵各段，避免把 q 头当作 v 头分给其他 rank。"""
    result = []
    offset = 0
    for width in widths:
        if width % world_size:
            raise ValueError(f"partition width {width} not divisible by {world_size}")
        size = width // world_size
        result.append((offset + rank * size, offset + (rank + 1) * size))
        offset += width
    return result


def shard_weight(key: str, value, cfg: dict, rank: int, world_size: int):
    """在 CPU 上切一项权重，完整矩阵不会先搬到 GPU。

    Args:
        key: checkpoint 参数名。
        value: 稠密张量或 packed W4 权重。
        cfg: 全局文本配置。
        rank: 分片编号。
        world_size: 分片数。

    Returns:
        本 rank 的连续张量或 W4 对象；共享参数原样返回。

    Raises:
        ValueError: rank、融合段宽度或 W4 K 轴边界不满足分片契约。
    """
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid rank/world_size")
    axis, ranges = None, None
    # ``QWeight`` 把形状放在 ``.n`` / ``.k`` 上，没有 ``.shape``；两处都读成 ``[N, K]``，
    # 下面的分片判据因此对稠密、int8、packed W4 三种存储一视同仁。
    shape = (value.n, value.k) if isinstance(value, QWeight) else value.shape
    if ".linear_attn." in key and key.endswith("in_proj_qkv.weight"):
        kd = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
        vd = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
        axis, ranges = 0, _ranges([kd, kd, vd], rank, world_size)
        if shape[0] != 2 * kd + vd:
            raise ValueError(f"{key}: fused q/k/v shape mismatch")
    elif key.endswith((".mlp.down_proj.weight", ".self_attn.o_proj.weight",
                       ".linear_attn.out_proj.weight", ".fc.weight")):
        axis, ranges = 1, _ranges([shape[1]], rank, world_size)
    elif key.endswith(".linear_attn.conv1d.weight"):
        kd = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
        vd = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
        ranges = _ranges([kd, kd, vd], rank, world_size)
        axis = 0
    elif key.endswith((".self_attn.k_proj.weight", ".self_attn.v_proj.weight")):
        # KV 头数少于卡数时（Qwen3.5-MoE 是 2 对 4 卡），一个 KV 头由
        # ``world_size / nkv`` 张卡共享，本 rank 只留它自己那一个头。判据必须与
        # ``local_config`` 逐字一致：两边分叉会得到形状正确、注意力错位的模型。
        kv = int(cfg["num_key_value_heads"])
        if kv % world_size == 0:
            axis, ranges = 0, _ranges([shape[0]], rank, world_size)
        elif world_size % kv == 0:
            hd = int(cfg["head_dim"])
            head = rank // (world_size // kv)
            axis, ranges = 0, [(head * hd, (head + 1) * hd)]
        else:
            raise ValueError(
                f"{key}: num_key_value_heads={kv} cannot be split across {world_size} ranks")
    elif key.endswith((".mlp.gate_proj.weight", ".mlp.up_proj.weight",
                       ".self_attn.q_proj.weight", ".linear_attn.in_proj_z.weight",
                       ".linear_attn.in_proj_a.weight", ".linear_attn.in_proj_b.weight",
                       ".linear_attn.A_log", ".linear_attn.dt_bias")):
        axis, ranges = 0, _ranges([shape[0]], rank, world_size)
    elif key.endswith((".experts.gate_up_stack", ".experts.down_stack")):
        # 融合专家栈：gate_up 按 N 轴（专家内宽度）分片，down 按 K 轴分片。
        #
        # scale 是逐输出行一个、与 K 的长度无关，所以 down 的 K 轴切分不动它；
        # gate_up 切的就是输出轴，它必须跟着切。
        if not isinstance(value, QWeightMoE):
            raise ValueError(
                f"{key}: fused expert stack sharding supports int8 stacks only, got "
                f"{type(value).__name__}")
        # 栈是 ``[E, K, N]``：gate_up 切 N（dim 2），down 切 K（dim 1）。
        gate_up = key.endswith(".experts.gate_up_stack")
        axis = 2 if gate_up else 1
        width = int(value.wq.shape[axis])
        if width % world_size:
            raise ValueError(f"{key}: width {width} not divisible by {world_size}")
        size = width // world_size
        begin = rank * size
        if gate_up:
            return QWeightMoE(value.wq.narrow(2, begin, size).contiguous(),
                              value.scale.narrow(1, begin, size).contiguous(),
                              k=value.k)
        return QWeightMoE(value.wq.narrow(1, begin, size).contiguous(),
                          value.scale, k=size)
    elif ".mlp.experts." in key or ".mlp.shared_expert." in key:
        # 专家的 gate/up 按输出轴（专家内宽度）分片，down 按 K 轴分片；路由器
        # （``.mlp.gate.weight``）与共享专家门（``.mlp.shared_expert_gate.weight``）
        # 故意不命中这两条：它们必须在各 rank 上有同一份，否则各 rank 会选出不同的
        # 专家，而每 rank 只算自己那份的归约结果就是错的。
        if key.endswith((".gate_proj.weight", ".up_proj.weight")):
            axis, ranges = 0, _ranges([shape[0]], rank, world_size)
        elif key.endswith(".down_proj.weight"):
            axis, ranges = 1, _ranges([shape[1]], rank, world_size)
    if axis is None:
        return value
    if isinstance(value, QWeight):
        # int8 的 scale 是逐输出行一个、与 K 长度无关，所以 K 轴（axis 1）切分时它原样
        # 跟着每个 rank 走；只有 N 轴切分才需要把它一起切开。
        # 区间来自 ``ranges``（逐 rank 的 ``(lo, hi)``），不是某个循环里残留的 a/b：
        # 早先这里直接用了 a、b，任何走到该分支的调用都会 NameError，而 PP 下永远
        # 走不到这条分支，所以一直没暴露。
        lo, hi = ranges[0]
        if axis == 0:
            return QWeight(value.wq.narrow(0, lo, hi - lo).contiguous(),
                           value.scale.narrow(0, lo, hi - lo).contiguous())
        return QWeight(value.wq.narrow(1, lo, hi - lo).contiguous(), value.scale)
    if not isinstance(value, W4A8Weight):
        return torch.cat([value.narrow(axis, a, b - a) for a, b in ranges], dim=axis).contiguous()
    if axis == 0:
        packed = torch.cat([value.packed[:, a:b] for a, b in ranges], dim=1).contiguous()
        scale = torch.cat([value.scale[:, a:b] for a, b in ranges], dim=1).contiguous()
        return W4A8Weight(packed, scale, k=value.k, group_size=value.group_size)
    start, stop = ranges[0]
    gs = value.group_size
    if start % gs or start % 2 or (stop != value.k and stop % gs):
        raise ValueError(f"{key}: W4 input partition must align to groups and packed pairs")
    return W4A8Weight(value.packed[start // 2:(stop + 1) // 2].contiguous(),
                       value.scale[start // gs:(stop + gs - 1) // gs].contiguous(),
                       k=stop - start, group_size=gs)


def pipe_keeps(key: str, cfg: dict, parallel: "ParallelContext",
               tied_head: bool = False) -> bool:
    """流水并行下本 rank 该不该读这一项权重。

    首段要 ``embed_tokens``（它把 token 变成 hidden），末段要 ``norm`` 与 ``lm_head``
    （它把 hidden 变成 logits），中间的段两者都不要——多读一份就是几百 MB 白占显存，
    而且会让「谁算 logits」变得不唯一。视觉塔与 MTP 不参与流水，每段各留一份。

    Args:
        key: checkpoint 参数名。
        cfg: 全局文本配置（用来算总层数）。
        parallel: 已配置的流水上下文。
        tied_head: checkpoint 没有独立的 ``lm_head.weight``（共享词嵌入）。此时词嵌入
            同时是输出头，每个 rank 都要有它（见下面的取值规则）。

    Returns:
        应当读进 state dict 时为 ``True``。
    """
    begin, end = parallel.layer_range(int(cfg["num_hidden_layers"]))
    match = re.match(r"^model\.language_model\.layers\.(\d+)\.", key)
    if match is not None:
        return begin <= int(match.group(1)) < end
    if key.startswith("model.language_model.embed_tokens."):
        # ``tied_head`` 的模型没有独立的 ``lm_head.weight``，输出头就是词嵌入本身。而
        # ``_forward`` 是「末段把 4 KB hidden 回给各段、各段自己算 logits」的结构，所以
        # 每个 rank 都要有它——只给 rank 0 会让非末段在算 logits 时拿到 ``None``。
        # int8 下这份是 254 MB/rank，相对换掉的 logits 广播值得。
        return tied_head or parallel.rank == 0
    if key == "model.language_model.norm.weight" or key.startswith("lm_head."):
        # 每个 rank 都留一份输出头：段间只传 4 KB 的 hidden，各 rank 自己算 logits。
        #
        # 反过来（只让末段持有、把 logits 广播出去）实测**更慢**：26.24 -> 21.48 tok/s。
        # 原因是各段的头 GEMV 与流水等待重叠，净成本不到 1 ms；而每 token 把 1 MB 的
        # logits 分发给三个对端要 5 ms 以上。省 508 MB 显存不值这个价。
        return True
    return True


def load_weights(model_dir: str | Path, device: torch.device,
                 parallel: ParallelContext, vision: bool | None = None) -> dict:
    """逐项读取离线量化权重，只把本 rank 的那一份搬到设备。

    三种拓扑各自省不同的东西：``pp`` 只读本段那 10 层，``tp`` 把每个矩阵按头/通道
    切开，``ep`` 原样读全部 dense 与注意力、只把专家栈按首轴裁成本 rank 的那一段。

    Args:
        model_dir: 带 W4 索引的模型目录。
        device: 本进程唯一可见的计算设备。
        parallel: 已配置的并行上下文。
        vision: 是否读视觉塔；None 沿用索引声明。

    Returns:
        与既有加载器相同键名的局部 state dict。

    Raises:
        ValueError: 格式、配置或张量布局不匹配。
        RuntimeError: 调用方要求视觉塔，但产物没有保存它。
    """
    from safetensors import safe_open
    from quantized_load import FORMAT as I8_FORMAT
    from quantized_load import W4A8_FORMAT, _check_source_config, read_index

    model_dir = Path(model_dir)
    index = read_index(model_dir)
    if index["format"] not in (W4A8_FORMAT, I8_FORMAT):
        raise ValueError(
            f"tensor parallel loader needs a pre-quantised directory, got "
            f"format={index['format']!r}")
    _check_source_config(model_dir, index)
    cfg = json.loads((model_dir / "config.json").read_text())["text_config"]
    local_config(cfg, parallel.world_size)
    take_vision = index.get("vision", False) if vision is None else vision
    if take_vision and not index.get("vision"):
        raise RuntimeError("checkpoint has no vision tensors")
    entries = {k: v for k, v in index["tensors"].items()
               if k.startswith(("model.language_model.", "mtp.", "lm_head."))
               or (take_vision and k.startswith("model.visual."))}
    if parallel.mode == "pp":
        # 流水并行只读本段那 10 层的权重：每段 8.5 GB 而不是 34 GB，这也是 PP 能在
        # 一张 32 GB 的卡上装下一半模型之外的其余段的原因。
        # 共享词嵌入的模型要靠这个标志决定 embed_tokens 发给谁，所以先看索引里有没有
        # 独立的输出头。
        tied_head = not any(k.startswith("lm_head.") for k in index["tensors"])
        entries = {k: v for k, v in entries.items()
                   if pipe_keeps(k, cfg, parallel, tied_head)}
    sd = {}
    for shard in sorted({v["shard"] for v in entries.values()}):
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as handle:
            for key, meta in entries.items():
                if meta["shard"] != shard:
                    continue
                # 专家并行必须**在读盘时**就把专家栈裁到本 rank 那一段：整栈每层
                # 805 MB、40 层 32 GB，先读进来再切会把一张 31.7 GiB 的卡撑爆，而
                # 这也正是「连续切专家」而不是「交错切」的理由（见 expert_range）。
                ep_slice = (parallel.mode == "ep" and meta["kind"] == "q8-moe")
                if ep_slice:
                    lo, hi = parallel.expert_range(int(cfg["num_experts"]))
                    value = handle.get_slice(key)[lo:hi]
                else:
                    value = handle.get_tensor(key)
                kind = meta["kind"]
                if kind == "q4":
                    scale = handle.get_tensor(meta["scale_key"])
                    if list(value.shape) != meta["packed_shape"] or list(scale.shape) != meta["scale_shape"]:
                        raise ValueError(f"{key}: packed checkpoint shape mismatch")
                    value = W4A8Weight(value, scale, k=meta["k"], group_size=meta["group_size"])
                elif kind == "q8-moe":
                    # 融合专家栈 `[E, K, N]` int8 + `[E, N]` fp32。漏掉这一条会让它落进
                    # 下面的 "else"，被当成稠密张量原样搬到设备——形状合法、MoE 一跑
                    # 就 mat1/mat2 形状不匹配，而引擎把那次异常当成「这一步没产出」，
                    # 原地空转到看门狗报 KV pool 太小。
                    if ep_slice:
                        scale = handle.get_slice(meta["scale_key"])[lo:hi]
                        expect = (hi - lo, *meta["shape"][1:])
                        if tuple(value.shape) != expect:
                            raise ValueError(
                                f"{key}: expert slice shape {tuple(value.shape)} "
                                f"does not match {expect}")
                    else:
                        scale = handle.get_tensor(meta["scale_key"])
                        if (list(value.shape) != meta.get("shape")
                                or list(scale.shape) != meta.get("scale_shape")):
                            raise ValueError(f"{key}: moe int8 stack shape mismatch")
                    value = QWeightMoE(value, scale, k=int(meta["k"]))
                elif kind == "q8":
                    # 盘上 int8 是 ``[K, N]`` 加逐输出列的 fp32 scale；``from_kn`` 只
                    # 取一个转置视图，不复制也不转置，所以这一趟仍然是「读文件、装箱」。
                    scale = handle.get_tensor(meta["scale_key"])
                    if list(value.shape) != meta["shape"] or int(scale.numel()) != meta["n"]:
                        raise ValueError(f"{key}: int8 checkpoint shape mismatch")
                    value = QWeight.from_kn(value, scale)
                if parallel.mode == "tp":
                    value = shard_weight(key, value, cfg, parallel.rank, parallel.world_size)
                if isinstance(value, W4A8Weight):
                    value = W4A8Weight(value.packed.to(device), value.scale.to(device),
                                       k=value.k, group_size=value.group_size)
                elif isinstance(value, QWeight):
                    value = QWeight(value.wq.to(device), value.scale.to(device))
                elif isinstance(value, QWeightMoE):
                    value = QWeightMoE(value.wq.to(device), value.scale.to(device),
                                       k=value.k)
                else:
                    value = value.to(device)
                sd[key] = value
    return sd


class RowParallelProjection:
    """归约局部 K 点积，向上层保持原有投影接口。"""

    def __init__(self, projection, parallel: ParallelContext) -> None:
        """持有局部投影及对应通信上下文。

        Args:
            projection: 从本 rank 权重构造的 GEMV。
            parallel: 模型共享的进程上下文。

        Raises:
            ValueError: 投影没有 packed W4 权重。
        """
        if not isinstance(projection.w, W4A8Weight):
            raise ValueError("row parallel projection requires packed W4 weights")
        self.w, self.N, self.K = projection.w, projection.N, projection.K
        self.parallel = parallel

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """按上下文选择激活 scale，以 fp32 归约局部整数点积后窄化。"""
        from backends.corex import _w4_rows

        if x.dim() not in (1, 2):
            raise ValueError("row parallel activation shape mismatch")
        if x.shape[-1] != self.K:
            if x.shape[-1] != self.K * self.parallel.world_size:
                raise ValueError("row parallel input width does not match shard count")
            start = self.parallel.rank * self.K
            x = x[..., start:start + self.K].contiguous()
        local_max = x.reshape(-1, self.K).float().abs().amax(-1)
        mode = self.parallel.scale_mode
        if mode not in {"shared", "local", "decode_local"}:
            raise ValueError(f"unknown activation scale mode: {self.parallel.scale_mode}")
        use_local_scale = mode == "local" or (mode == "decode_local" and x.numel() == self.K)
        scale_max = local_max if use_local_scale else self.parallel.reduce(
            local_max, maximum=True)
        scale = scale_max.clamp_min_(1e-5).div_(127)
        partial = _w4_rows(self.w, x.float(), activation_scale=scale)
        return self.parallel.reduce(partial).to(x.dtype)

    def linear_m(self, x: torch.Tensor) -> torch.Tensor:
        """批处理沿用同一全局量化和归约契约。"""
        return self(x)

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """预填充由 CoreX 按形状选择直接消费 packed W4 的张量核。"""
        return self(x)


def row_projection(projection):
    """仅在配置**张量**并行时给行投影添加归约。

    流水并行按层切分，每个 rank 手里的投影都是完整的；这时再加一次归约等于把同一个
    数自己加 world_size 遍——数值翻倍而且完全静默。所以这里按 mode 分叉，不是按
    「有没有上下文」分叉。
    """
    parallel = context()
    if parallel is None or parallel.mode != "tp":
        return projection
    return RowParallelProjection(projection, parallel)
