"""内核 shim 的 torch-free 构建：厂商 clang 直接编一个普通 ``.so``。

Module: ``coffin/cxrt``。只 import 标准库，**不 import torch**，
也不再需要 ``torch.utils.cpp_extension``：编出来的东西没有 pybind、没有
``TORCH_EXTENSION_NAME``、没有任何 ``at::`` 符号，Python 侧用 ``ctypes`` 装载。

为什么是 clang 而不是 ``nvcc``：本平台 SDK 里的 ``nvcc`` 是空壳（ixblas 的构建脚本
也绕开了它），实际能编 ivcore10 设备代码的是 ``/usr/local/corex/bin/clang++``：
``-x cuda --cuda-path=<root> --cuda-gpu-arch=ivcore10``。

Constraints:
    * 缓存键是「源码 + 头文件 + 编译选项 + 架构」的内容哈希，不是名字。改了 ``.h``
      也必须换目录——只哈希 ``.cu`` 会让所有人共用一个按旧头文件编出来的目标文件，
      表现为「改了半天没变化」。
    * 构建锁用目录创建而不是接力文件：目录创建是原子的，且带时间戳可判陈旧。接力
      文件锁在被杀的任务里会永久残留，之后每个任务都死等。
    * 失败必须带命令行与输出尾部。没有命令行的构建错误在集群上不可复现。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable

from cxrt.errors import BuildError

#: SDK 根目录。``clang++`` 与 ``libcudart.so`` 都在它下面。
_ROOT_ENV = "COREX_ROOT"
_DEFAULT_ROOT = "/usr/local/corex"

#: 缓存根目录。默认在 ``$HOME`` 下，绝不落进源码树：集群上的 checkout 可能只读。
_CACHE_ENV = "COFFIN_RT_DIR"

#: 构建锁的陈旧阈值，秒。默认十分钟：比任何一次成功构建都长，又远短于被阻塞任务的
#: 等待时间。
_LOCK_AGE_ENV = "COFFIN_RT_LOCK_MAX_AGE"

#: 等锁的上限，秒。超时抛 :class:`BuildError`，而不是无限轮询。
_LOCK_WAIT_ENV = "COFFIN_RT_LOCK_WAIT"

#: 主机的 C++ 编译器候选。显式覆盖 > SDK 自带 clang > PATH 上的 clang++。
_CXX_ENV = "COFFIN_RT_CXX"


def corex_root() -> Path:
    """返回 SDK 根目录。

    Returns:
        ``$COREX_ROOT`` 或 ``/usr/local/corex``。
    """
    return Path(os.environ.get(_ROOT_ENV, "").strip() or _DEFAULT_ROOT)


def cache_root() -> Path:
    """返回构建缓存根目录。

    Returns:
        ``$COFFIN_RT_DIR``，否则 ``$HOME/.coffin_corex_rt``。
    """
    explicit = os.environ.get(_CACHE_ENV, "").strip()
    if explicit:
        return Path(explicit)
    return Path.home() / ".coffin_corex_rt"


def compiler() -> str:
    """定位主机编译器。

    Returns:
        可执行文件路径。

    Raises:
        BuildError: 三个候选都不存在。消息里列出试过的路径：集群上「工具链没装」
            与「装在了别的前缀」看起来一样，只有路径能区分。
    """
    explicit = os.environ.get(_CXX_ENV, "").strip()
    if explicit:
        if Path(explicit).exists() or shutil.which(explicit):
            return explicit
        raise BuildError(f"{_CXX_ENV}={explicit} is neither a file nor on PATH")
    candidates = [corex_root() / "bin" / "clang++", Path(shutil.which("clang++") or "")]
    for candidate in candidates:
        if str(candidate) and Path(candidate).exists():
            return str(candidate)
    raise BuildError("no host compiler: set COFFIN_RT_CXX or COREX_ROOT "
                     f"(tried {', '.join(str(c) for c in candidates if str(c))})")


def _collect(paths: Iterable[Path]) -> list[Path]:
    """把目录展开成文件列表，非目录原样保留。

    Args:
        paths: 文件或目录。

    Returns:
        文件路径列表，目录按名称排序后展开（保证哈希稳定）。
    """
    out: list[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            out.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        elif path.is_file():
            out.append(path)
    return out


def cache_key(sources: Iterable[Path], includes: Iterable[Path], arch: str,
              extra_flags: Iterable[str] = ()) -> str:
    """算这次构建的内容哈希。

    Args:
        sources: 源文件或目录。
        includes: 头文件目录，参与哈希。
        arch: 目标 ISA。
        extra_flags: 额外编译选项，参与哈希。

    Returns:
        ``sha256`` 的十六进制串。它的前 16 位做目录名，全串写进 ``meta.json`` 备查。
    """
    digest = hashlib.sha256()
    digest.update(f"arch={arch}\n".encode())
    for flag in extra_flags:
        digest.update(f"flag={flag}\n".encode())
    for path in _collect(list(sources) + list(includes)):
        digest.update(f"file={path.name}\n".encode())
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _acquire_lock(lock: Path) -> bool:
    """尝试建构建锁。

    Args:
        lock: 锁目录路径。

    Returns:
        本次拿到锁为 ``True``；已有锁但尚未陈旧为 ``False``。
    """
    try:
        lock.mkdir()
        return True
    except FileExistsError:
        try:
            age = time.time() - lock.stat().st_mtime
        except OSError:
            # 锁刚被持有者删掉：这一轮算没抢到，下一轮会重新尝试创建。
            return False
        limit = float(os.environ.get(_LOCK_AGE_ENV, "600"))
        if age <= limit:
            return False
        # 陈旧锁：上一次构建被杀掉了。删掉重建，删不掉就让给别的进程。
        try:
            lock.rmdir()
        except OSError:
            return False
        try:
            lock.mkdir()
            return True
        except FileExistsError:
            return False


def build(sources: Iterable[Path], includes: Iterable[Path], arch: str,
          name: str, extra_flags: Iterable[str] = ()) -> Path:
    """编出（或复用）一个 shim ``.so``。

    Args:
        sources: 源文件列表。第一个必须是 C ABI 出口所在的 ``.cu``。
        includes: 头文件目录列表，``-I`` 与哈希都用它。
        arch: 目标 ISA，即 ``Caps.arch``。
        name: 产物名，只用于目录与文件名，**不参与缓存键**。
        extra_flags: 额外编译选项。

    Returns:
        ``.so`` 的绝对路径。

    Raises:
        BuildError: 编译失败、缺工具链，或等锁超时。消息含命令行与输出尾部。
    """
    flags = list(extra_flags) + [
        part for part in os.environ.get("COFFIN_RT_CXXFLAGS", "").split() if part
    ]
    key = cache_key(sources, includes, arch, flags)
    out_dir = cache_root() / f"{name}-{key[:16]}"
    lib = out_dir / f"lib{name}.so"
    if lib.is_file() and (out_dir / "ready").is_file():
        return lib

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = out_dir.parent / f"{name}-{key[:16]}.lock"
    deadline = time.time() + float(os.environ.get(_LOCK_WAIT_ENV, "900"))
    while not _acquire_lock(lock):
        if lib.is_file() and (out_dir / "ready").is_file():
            return lib
        if time.time() > deadline:
            raise BuildError(f"timed out waiting for build lock {lock}")
        time.sleep(0.5)

    try:
        # 拿到锁之后再判一次：等锁期间可能已经被别人编好了。
        if lib.is_file() and (out_dir / "ready").is_file():
            return lib
        out_dir.mkdir(parents=True, exist_ok=True)
        root = corex_root()
        command = [compiler(), "-x", "cuda", f"--cuda-path={root}",
                   f"--cuda-gpu-arch={arch}", "-O3", "-std=c++17", "-shared", "-fPIC"]
        for inc in includes:
            command += ["-I", str(inc)]
        command += [*(str(p) for p in sources),
                    "-L", str(root / "lib64"), "-lcudart",
                    f"-Wl,-rpath,{root / 'lib64'}", "-o", str(lib)]
        command += flags
        started = time.time()
        try:
            proc = subprocess.run(command, capture_output=True, text=True)
        except OSError as exc:
            # 可执行文件不存在、没有执行权限、解释器 shebang 坏掉。``subprocess`` 抛的是
            # ``OSError``，不转成 :class:`BuildError` 会让调用方以为「编译失败了」，
            # 而真实原因是「根本没开始编译」。
            raise BuildError(f"cannot run compiler\n$ {' '.join(command)}\n{exc}") from exc
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-800:]
            raise BuildError(f"compiler failed (rc={proc.returncode})\n"
                             f"$ {' '.join(command)}\n{tail}")
        (out_dir / "meta.json").write_text(json.dumps({
            "key": key, "name": name, "arch": arch, "command": command,
            "seconds": round(time.time() - started, 2),
            "sources": [p.name for p in sources],
        }, indent=2), encoding="utf-8")
        (out_dir / "ready").write_text("ok\n", encoding="utf-8")
        return lib
    finally:
        try:
            lock.rmdir()
        except OSError:
            pass
