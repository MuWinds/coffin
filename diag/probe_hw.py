#!/usr/bin/env python3
"""Synthetic hardware probes for the 27B-int4 / 256k tiered-KV design (DCU Z200SM_80).

No model weights are touched.  Every measurement prints one ``dotted.name value unit``
line so the raw numbers can be pasted/diffed, and the same table is dumped to
``hw_probe.json`` next to this file.

Sections, in the order they run (host-light before GPU-heavy):

1. PCIe H2D/D2H (pinned vs pageable, 8 MB .. 2 GB) + small-transfer latency
2. HBM copy/read/write reference
3. CPU DRAM read and memcpy at 1 and 7 threads (2 GiB fp32)
4. Disk: NFS sequential (warm/cold), local overlay write/sequential, 4 KB random reads
5. VRAM budget: free after init, then a 14.0 GB int8 allocation that is touched
6. CPU/GPU overlap: 2 s GPU matmul loop vs 2 s 7-thread DRAM read

Cache caveats and DCU contention are reported in the ``#`` lines and in
``_conditions`` inside the JSON.
"""

from __future__ import annotations

import glob
import json
import os
import random
import statistics
import subprocess
import threading
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "hw_probe.json")
RESULT: dict = {}
DCU_SAMPLES: list[str] = []

NFS_SHARD = "/root/private_data/models/Qwen3.5-4B/model.safetensors-00001-of-00002.safetensors"
LOCAL_SEQ = "/tmp/hwprobe_seq.bin"
LOCAL_RAND = LOCAL_SEQ  # one 2 GiB file serves both the sequential and random tests
NFS_RAND_SPAN = 2 << 30
MIB = 1 << 20
GIB = 1 << 30


# Shared result-recording helpers.
def rec(name: str, value: float, unit: str) -> None:
    RESULT[name] = {"value": value, "unit": unit}
    if isinstance(value, float):
        print(f"{name} {value:.4g} {unit}", flush=True)
    else:
        print(f"{name} {value} {unit}", flush=True)


def note(msg: str) -> None:
    print(f"# {msg}", flush=True)


def sync() -> None:
    torch.cuda.synchronize()


def median(xs):
    return statistics.median(xs)


def timed(fn, reps: int = 3, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return median(ts)


def sh(cmd: str, timeout: int = 300) -> tuple[int, str]:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def hy_smi() -> str:
    try:
        out = subprocess.run(["hy-smi"], capture_output=True, text=True, timeout=15).stdout
    except Exception as exc:  # pragma: no cover - platform detail
        return f"hy-smi-error:{exc}"
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "0":
            return f"vram={parts[5]} dcu={parts[6]}"
    return "parse-error"


def sample_dcu(tag: str) -> None:
    s = hy_smi()
    DCU_SAMPLES.append(f"{tag}: {s}")
    note(f"hy-smi after {tag}: {s}")


def guarded(name: str, fn) -> None:
    try:
        fn()
    except Exception as exc:  # keep the later sections alive
        import traceback

        traceback.print_exc()
        note(f"section {name} FAILED: {type(exc).__name__}: {exc}")


# Environment and device properties.
def env_info() -> None:
    note("environment")
    p = torch.cuda.get_device_properties(0)
    rec("env.device_total", p.total_memory / GIB, "GiB")
    note(f"torch={torch.__version__} device={p.name} SMs={p.multi_processor_count}")
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    note("cpu=" + line.split(":", 1)[1].strip())
                    break
    except OSError:
        pass
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            rec("env.cgroup_mem_max", int(f.read()) / GIB, "GiB")
    except Exception:
        pass
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            note("cgroup cpu.max=" + f.read().strip())
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    note("meminfo " + line.strip())
                    break
    except OSError:
        pass
    links = []
    for d in sorted(glob.glob("/sys/bus/pci/devices/*")):
        try:
            with open(os.path.join(d, "current_link_speed")) as f:
                speed = f.read().strip()
            with open(os.path.join(d, "current_link_width")) as f:
                width = f.read().strip()
        except OSError:
            continue
        links.append(f"{os.path.basename(d)} {speed} x{width}")
    note("pcie links: " + "; ".join(links))


# 1. Host-to-device copy bandwidth.
def pcie() -> None:
    note("PCIe: torch copies, sync every rep, median of 3 (warmup 1)")
    sizes = [("8MB", 8 * MIB), ("64MB", 64 * MIB), ("256MB", 256 * MIB),
             ("1GB", GIB), ("2GB", 2 * GIB)]
    for pinned in (True, False):
        kind = "pinned" if pinned else "pageable"
        for label, n in sizes:
            try:
                h = torch.empty(n, dtype=torch.uint8, pin_memory=pinned)
            except Exception as exc:
                rec(f"pcie.h2d.{kind}.{label}", 0.0, f"alloc-failed:{type(exc).__name__}")
                note(f"pin_memory={pinned} {label} alloc failed: {exc}")
                continue
            g = torch.empty(n, dtype=torch.uint8, device="cuda")
            h.zero_()
            g.fill_(0)

            def h2d():
                g.copy_(h, non_blocking=pinned)

            def d2h():
                h.copy_(g, non_blocking=pinned)

            sync()
            t = timed(h2d)
            rec(f"pcie.h2d.{kind}.{label}", n / t / 1e9, "GB/s")
            t = timed(d2h)
            rec(f"pcie.d2h.{kind}.{label}", n / t / 1e9, "GB/s")
            del h, g
            torch.cuda.empty_cache()

    for label, n in (("4KB", 4 * 1024), ("1MB", MIB)):
        h = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        g = torch.empty(n, dtype=torch.uint8, device="cuda")
        for direction in ("h2d", "d2h"):
            lat = []
            for _ in range(30):
                t0 = time.perf_counter()
                if direction == "h2d":
                    g.copy_(h, non_blocking=True)
                else:
                    h.copy_(g, non_blocking=True)
                sync()
                lat.append((time.perf_counter() - t0) * 1e6)
            rec(f"pcie.latency.{direction}.{label}", median(lat), "us")
        del h, g
        torch.cuda.empty_cache()


# 2. HBM bandwidth reference.
def hbm() -> None:
    note("HBM reference: 2 GiB uint8 buffers, 1 GiB fp16 sum")
    a = torch.empty(2 * GIB, dtype=torch.uint8, device="cuda")
    b = torch.empty(2 * GIB, dtype=torch.uint8, device="cuda")
    a.fill_(1)
    sync()
    t = timed(lambda: b.copy_(a))
    rec("hbm.copy_2GiB_traffic", 2 * 2 * GIB / t / 1e9, "GB/s")
    t = timed(lambda: a.zero_())
    rec("hbm.write_2GiB", 2 * GIB / t / 1e9, "GB/s")
    del a, b
    torch.cuda.empty_cache()
    x = torch.empty(1 << 29, dtype=torch.float16, device="cuda")
    x.normal_()
    sync()
    t = timed(lambda: x.sum())
    rec("hbm.read_1GiB", GIB / t / 1e9, "GB/s")
    del x
    torch.cuda.empty_cache()


# 3. CPU DRAM bandwidth.
def cpu() -> None:
    note("CPU DRAM: 2 GiB fp32 array (sum = read-only, copy_ = read+write traffic)")
    x = torch.empty(512 * MIB, dtype=torch.float32)
    y = torch.empty_like(x)
    x.normal_()
    for thr in (1, 7):
        torch.set_num_threads(thr)
        t = timed(lambda: x.sum())
        rec(f"cpu.read.{thr}t.2GiB", x.numel() * 4 / t / 1e9, "GB/s")
        t = timed(lambda: y.copy_(x))
        rec(f"cpu.memcpy.{thr}t.2GiB", 2 * x.numel() * 4 / t / 1e9, "GB/s")
    del x, y


# 4. Disk: sequential and random reads.
def drop_cache(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except Exception as exc:
        note(f"fadvise(DONTNEED) failed on {path}: {exc}")


def time_cmd(cmd: str, nbytes: int, name: str) -> None:
    t0 = time.perf_counter()
    rc, out = sh(cmd)
    dt = time.perf_counter() - t0
    if rc != 0:
        note(f"{name} rc={rc} out={out.strip()[-300:]}")
        rec(name, 0.0, "GB/s")
        return
    rec(name, nbytes / dt / 1e9, "GB/s")
    rec(name + ".seconds", dt, "s")


def rand_read(path: str, n: int = 2000, bs: int = 4096, span: int = NFS_RAND_SPAN,
              seed: int = 7) -> dict:
    size = os.path.getsize(path)
    limit = max(1, min(span, size) - bs)
    rng = random.Random(seed)
    offs = [rng.randrange(0, limit) for _ in range(n)]
    fd = os.open(path, os.O_RDONLY)
    lat = []
    t0 = time.perf_counter()
    try:
        for off in offs:
            a = time.perf_counter()
            os.pread(fd, bs, off)
            lat.append(time.perf_counter() - a)
    finally:
        os.close(fd)
    dt = time.perf_counter() - t0
    lat.sort()
    return {
        "mean_us": sum(lat) / n * 1e6,
        "p50_us": lat[n // 2] * 1e6,
        "p99_us": lat[int(n * 0.99)] * 1e6,
        "iops": n / dt,
    }


def disk() -> None:
    note("disk: NFS shard 5.33 GB, local overlay /tmp 2 GiB file")
    try:
        note(f"nfs shard bytes={os.path.getsize(NFS_SHARD)}")
    except OSError as exc:
        note(f"nfs shard missing: {exc}")

    # NFS sequential, warm page cache first (the 4B model was used today)
    time_cmd(f"dd if={NFS_SHARD} of=/dev/null bs=1M count=1500 status=none",
             1500 * MIB, "disk.nfs.seq.warm")
    drop_cache(NFS_SHARD)
    time_cmd(f"dd if={NFS_SHARD} of=/dev/null bs=1M count=1500 status=none",
             1500 * MIB, "disk.nfs.seq.cold_best_effort")

    # local overlay: write 2 GiB (fsync), then random 4k, then sequential read
    time_cmd(f"dd if=/dev/zero of={LOCAL_SEQ} bs=1M count=2048 conv=fsync status=none",
             2 * GIB, "disk.local.write")
    drop_cache(LOCAL_SEQ)
    r = rand_read(LOCAL_SEQ)
    rec("disk.local.rand4k.lat_mean", r["mean_us"], "us")
    rec("disk.local.rand4k.lat_p50", r["p50_us"], "us")
    rec("disk.local.rand4k.lat_p99", r["p99_us"], "us")
    rec("disk.local.rand4k.iops", r["iops"], "IOPS")
    drop_cache(LOCAL_SEQ)
    time_cmd(f"dd if={LOCAL_SEQ} of=/dev/null bs=1M count=2048 status=none",
             2 * GIB, "disk.local.seq.read")

    # NFS random 4k over the first 2 GiB of the shard (page cache partly warm)
    try:
        r = rand_read(NFS_SHARD)
        rec("disk.nfs.rand4k.lat_mean", r["mean_us"], "us")
        rec("disk.nfs.rand4k.lat_p50", r["p50_us"], "us")
        rec("disk.nfs.rand4k.lat_p99", r["p99_us"], "us")
        rec("disk.nfs.rand4k.iops", r["iops"], "IOPS")
    except OSError as exc:
        note(f"nfs rand_read failed: {exc}")
    try:
        os.remove(LOCAL_SEQ)
    except OSError:
        pass


# 5. VRAM capacity.
def vram() -> None:
    free0, total = torch.cuda.mem_get_info()
    rec("vram.total", total / GIB, "GiB")
    rec("vram.free_after_init", free0 / GIB, "GiB")
    rec("vram.driver_plus_context", (total - free0) / GIB, "GiB")
    torch.cuda.empty_cache()
    nbytes = 14_000_000_000  # 14.0 GB decimal = 13.04 GiB
    t = torch.empty(nbytes, dtype=torch.int8, device="cuda")
    t.fill_(1)
    sync()
    rec("vram.allocated_14GB", torch.cuda.memory_allocated() / GIB, "GiB")
    rec("vram.reserved_14GB", torch.cuda.memory_reserved() / GIB, "GiB")
    free1, _ = torch.cuda.mem_get_info()
    rec("vram.free_after_14GB", free1 / GIB, "GiB")
    rec("vram.weights_that_fit", (nbytes + free1) / GIB, "GiB")
    del t
    torch.cuda.empty_cache()


# 6. CPU/GPU overlap.
def overlap() -> None:
    torch.set_num_threads(7)
    note("overlap: 4096^2 fp16 matmul loop (GPU) vs 2 GiB fp32 sum loop (7 CPU threads)")
    a = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")

    def gpu_iters(n: int) -> None:
        for _ in range(n):
            a @ b
        sync()

    gpu_iters(2)
    sync()
    t0 = time.perf_counter()
    gpu_iters(2)
    one_gpu = (time.perf_counter() - t0) / 2
    gpu_n = max(3, int(2.0 / max(one_gpu, 1e-3)))

    x = torch.empty(512 * MIB, dtype=torch.float32)
    x.normal_()

    def cpu_iters(n: int) -> None:
        for _ in range(n):
            x.sum()

    cpu_iters(1)
    t0 = time.perf_counter()
    cpu_iters(1)
    one_cpu = time.perf_counter() - t0
    cpu_n = max(1, int(2.0 / max(one_cpu, 1e-3)))

    t0 = time.perf_counter()
    gpu_iters(gpu_n)
    t_gpu = time.perf_counter() - t0
    t0 = time.perf_counter()
    cpu_iters(cpu_n)
    t_cpu = time.perf_counter() - t0

    th = threading.Thread(target=lambda: cpu_iters(cpu_n))
    t0 = time.perf_counter()
    th.start()
    gpu_iters(gpu_n)
    th.join()
    t_both = time.perf_counter() - t0

    rec("overlap.gpu_only", t_gpu, "s")
    rec("overlap.cpu_only", t_cpu, "s")
    rec("overlap.both", t_both, "s")
    eff = (t_gpu + t_cpu - t_both) / max(1e-9, min(t_gpu, t_cpu)) * 100.0
    rec("overlap.efficiency", max(0.0, min(100.0, eff)), "%")
    note(f"iters gpu={gpu_n} cpu={cpu_n} one_gpu={one_gpu * 1e3:.1f}ms one_cpu={one_cpu * 1e3:.1f}ms")
    del a, b, x
    torch.cuda.empty_cache()


def main() -> None:
    note("start " + time.strftime("%Y-%m-%d %H:%M:%S"))
    note(f"script={os.path.abspath(__file__)}")
    guarded("env", env_info)
    sample_dcu("init")
    guarded("pcie", pcie)
    sample_dcu("pcie")
    guarded("hbm", hbm)
    sample_dcu("hbm")
    guarded("cpu", cpu)
    sample_dcu("cpu")
    guarded("disk", disk)
    sample_dcu("disk")
    guarded("vram", vram)
    sample_dcu("vram")
    guarded("overlap", overlap)
    sample_dcu("overlap")
    RESULT["_conditions"] = {
        "dcu_samples": DCU_SAMPLES,
        "note": "synthetic tensors only; page cache warm unless marked cold_best_effort",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(JSON_PATH, "w") as f:
        json.dump(RESULT, f, indent=1)
    note(f"wrote {JSON_PATH}")
    note("done")


if __name__ == "__main__":
    main()
