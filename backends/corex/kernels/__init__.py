"""构建 CoreX 内核扩展，或附着到已有的缓存构建上。

该扩展是用 ``torch.utils.cpp_extension.load`` 从
``backends/corex/kernels/*.cu`` 编出来的 torch JIT 扩展。import 时不编译任何
东西：每个只枚举后端名字的主机都会 import 本模块（包括没有 CoreX 工具链的纯
CPU 主机），所以构建是懒的，失败也被记忆化。

Module: ``coffin/backends/corex``。只 import ``torch``、标准库和扩展加载器，
完全不用 ``coffin`` 里的东西。它构建的内核是紧挨着它的 C++/CUDA 源码；上层的
算子表调用 :func:`load_kernels`。

Constraints:
    * ``TORCH_EXTENSIONS_DIR`` 必须来自环境变量，绝不能在这里写死。每个部署都得
      指向自己的缓存：共享缓存会让并发任务抢同一根构建锁并卡死，而源码树内的
      缓存会让集群上的 checkout 变成不可写。
    * 构建目标是 ``--cuda-gpu-arch=ivcore10``（本部件的 ISA，从驱动读到的
      ``Caps.arch``）。换一个部件要传入不同的架构串，而不是再加一条构建路径。
    * 失败记忆正是 :func:`load_kernels` 的意义：坏工具链在每进程只花一次
      尝试，之后不花代价，算子表回退到参考实现，而不是让进程在请求中途死掉。
    * import 本模块不得在模块作用域 import ``torch.utils.cpp_extension``。加载器
      在 import 时读环境变量，一个永不构建的主机不该被一次枚举重新配置。
    * 扩展按架构编译，所以构建目录以架构串为键：``ivcore10`` 的目标文件绝不能
      附着到跑在不同 ISA 上的进程。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

#: 存放本模块要构建的 ``.cu``/``.cpp`` 源码的目录。
KERNEL_DIR = Path(__file__).resolve().parent

#: 扩展的编译单元，按链接顺序。列在这里而不是 glob，是为了让散落的 ``.cu``
#: （探针、实验）进不了生产构建，也让这份清单可复核。
_SOURCES = ("bindings.cpp", "gemv.cu", "w8a8.cu", "norm.cu", "attn.cu", "linear_attn.cu",
            "delta.cu", "solve.cu")

#: 驱动不报告 ISA 时的回退值。``ivcore10`` 是 BI-V100 的架构
#: （``COREX_PORT.md`` §0）；报告了别的值的卡就用那个值，所以加载器把 ``arch``
#: 作为参数接收。
DEFAULT_ARCH = "ivcore10"

#: 扩展名。后缀是构建方案版本：某次源码改动不能附着到已缓存的目标树时递增
#: 它。每批内核都递增，正是因为陈旧缓存是看不见的——旧目标照样加载并给出答案。
#:
#: 可以用 ``COFFIN_COREX_EXT`` 覆盖，因为同一棵树里改不同 ``.cu`` 的两个人没
#: 法都递增这个值：后提交的人会静默复用前者的目标文件，量的是一个自己没改过的
#: 内核。按 worker 加后缀让各自拥有独立的目标树和独立的构建锁，这也是并发构建
#: 不必串在同一把锁后面的原因。
_EXTENSION = os.environ.get("COFFIN_COREX_EXT", "").strip() or "coffin_corex_ops_v15"

#: 已加载的扩展，首次尝试之前是 ``None``。
_module: Any = None

#: 构建失败留下的 ``<ExceptionType>: <message>`` 字符串，否则 ``None``。记忆化是
#: 为了让坏工具链每进程只花一次尝试。
_error: str | None = None


def _flags(arch: str) -> dict[str, list[str]]:
    """返回某个目标 ISA 的编译选项。

    Args:
        arch: 目标 ISA，即 ``Caps.arch``（BI-V100 上是 ``"ivcore10"``）。

    Returns:
        ``{"cflags": [...], "cuda_cflags": [...]}``。两边都给 ``-O3``，因为 GEMV
        就是一个裸的 load/FMA 循环，没什么可藏；架构选项必须送到设备编译器，否则
        构建会静默地以驱动默认值为目标。
    """
    return {
        "cflags": ["-O3"],
        "cuda_cflags": ["-O3", f"--cuda-gpu-arch={arch}"],
    }


def _clear_stale_build_lock(cache: str) -> None:
    """清掉被中断的进程留下的构建锁。

    ``torch.utils.cpp_extension`` 用 ``<cache>/<name>/lock`` 上的接力文件串行化构建。
    它的 ``wait()`` 一直轮询到文件消失，没有任何机制让它过期：一个在创建该文件之后、
    删除之前被杀掉的任务，会让集群上之后每个任务永久阻塞，两条输出流都没有内容，
    因为等待发生在一次 import 内部。

    这不是理论问题。取消三个仍在解析后端的提交，就把共享缓存卡住了，之后每次运行
    都跑不起来，现象看起来完全像设备代码里的 hang——只有调用栈能指出真凶。

    只删除 *陈旧* 的锁。新的锁属于真的在跑的构建，删掉它会让两个构建写同一份目标
    树。

    Args:
        cache: ``TORCH_EXTENSIONS_DIR`` 的值。

    Returns:
        无。恢复是尽力而为：删不掉的锁交给 ``load`` 处理。
    """
    lock = Path(cache) / _EXTENSION / "lock"
    try:
        age = time.time() - lock.stat().st_mtime
    except OSError:
        return
    # 默认十分钟：比这七个单元任何一次成功构建都长，又远短于一个被阻塞的任务本来
    # 要烧掉的墙钟时间。阈值可以覆盖，因为从一个才卡了几秒的僵局里恢复需要更小的
    # 值，而为了清一把已知坏掉的锁等十分钟本身就是 bug。
    limit = float(os.environ.get("COFFIN_BUILD_LOCK_MAX_AGE", "600"))
    if age < limit:
        return
    try:
        lock.unlink()
    except OSError:
        return


def load_kernels(arch: str = DEFAULT_ARCH) -> Any:
    """构建或附着扩展，每进程一次。

    Args:
        arch: 驱动给出的目标 ISA。只用第一次调用的值：已加载的模块被记忆化，所以
            一个进程不能持有两份构建，传第二个 arch 只会静默复用第一份的目标文件。

    Returns:
        扩展模块。

    Raises:
        RuntimeError: 构建失败，或 ``TORCH_EXTENSIONS_DIR`` 未设置。抛出而不是返回
            ``None``，是为了让调用方只有一处能 catch、只有一条消息要记日志；
            ``backends/corex/__init__.py`` 把它变成后端的 ``reason()`` 和参考实现
            回退。
    """
    global _module, _error
    if _module is not None:
        return _module
    if _error is not None:
        raise RuntimeError(_error)
    cache = os.environ.get("TORCH_EXTENSIONS_DIR", "").strip()
    if not cache:
        # 在这里给默认值会把构建产物丢进共享目录或不可写目录，而这两种失败从外面
        # 看起来一模一样。
        _error = ("TORCH_EXTENSIONS_DIR is not set; the CoreX kernels need a "
                  "per-deployment extension cache (a shared one serialises every "
                  "deployment behind one build lock)")
        raise RuntimeError(_error)
    try:
        from torch.utils.cpp_extension import load

        _clear_stale_build_lock(cache)
        flags = _flags(arch or DEFAULT_ARCH)
        _module = load(
            name=_EXTENSION,
            sources=[str(KERNEL_DIR / name) for name in _SOURCES],
            extra_include_paths=[str(KERNEL_DIR)],
            extra_cflags=flags["cflags"],
            extra_cuda_cflags=flags["cuda_cflags"],
            verbose=False,
        )
    except Exception as exc:                      # pragma: no cover - needs a toolchain
        _error = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(_error) from exc
    return _module


def available() -> bool:
    """报告扩展是否已加载或可加载，不抛异常。

    Returns:
        :func:`load_kernels` 已成功或此刻能够成功时为 ``True``。没有工具链的
        主机上，这会尝试一次构建，之后从记忆里答 ``False``，正是这一点让
        ``backends/corex.usable()`` 可以反复廉价调用。
    """
    if _module is not None:
        return True
    try:
        load_kernels()
    except Exception:                             # noqa: BLE001 - degradation path
        return False
    return True


def last_error() -> str | None:
    """返回记忆下来的构建失败，供后端的 ``reason()`` 使用。

    Returns:
        失败构建留下的 ``<ExceptionType>: <message>`` 字符串；构建成功或从未尝试时为
        ``None``。没有它，启动日志只会显示回退到了参考实现，却看不到原因。
    """
    return _error


def errors() -> list[str]:
    """把构建失败包成单元素列表返回，没有失败则返回空列表。

    Returns:
        构建失败时为 ``[last_error()]``。它存在是为了让诊断脚本可以
        ``"\n".join(...)`` 而不必先查 ``None``——``backends/corex/diag`` 里的诊断
        脚本用的就是这个形状。
    """
    err = last_error()
    return [] if err is None else [err]


def reset() -> None:
    """忘掉记忆下来的模块和错误，让下次调用重新构建。

    Returns:
        无。给刚创建了 ``TORCH_EXTENSIONS_DIR`` 或修好工具链的诊断用，也给需要
        回到「从未尝试」状态的测试用。
    """
    global _module, _error
    _module = None
    _error = None
