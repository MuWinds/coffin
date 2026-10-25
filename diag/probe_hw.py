#!/usr/bin/env python3
"""Synthetic hardware probes for the 27B-int4 / 256k tiered-KV design (DCU Z200SM_80).

Answers the capacity questions the tiered-KV layout rests on, with no model
weights touched:

* ``vram.weights_that_fit`` — what stays resident once the weights and the CUDA
  context are paid for. Compare it against the 14.0 GB int8 size to decide
  whether a tier fits at all.
* ``hbm.*`` — the bandwidth roof every kernel is measured against.
* ``pcie.*`` — H2D/D2H rate for pinned vs pageable memory, plus small-transfer
  latency, which is what a per-token weight fetch actually pays.
* ``cpu.*`` and ``overlap.efficiency`` — whether CPU DRAM work can hide behind
  GPU work (near 100 % means the engine may scan CPU-resident tiers for free) or
  will serialise with it.
* ``disk.*`` — NFS and local-overlay cost per 4 KB random read on the KV shard.

Run inside the container, from the ``coffin/`` root; a laptop run measures
nothing this design can use, because the DCU, the cgroup limits and the NFS
mount all change the answer:

    python3 diag/probe_hw.py

Every measurement prints one ``dotted.name value unit`` line so raw numbers can
be pasted/diffed, and the same table is dumped to ``hw_probe.json`` next to this
file. A ``# ...`` line is context, not a measurement. Sections run host-light
before GPU-heavy (env, PCIe, HBM, CPU, disk, VRAM, overlap) so a GPU failure
still leaves the CPU and disk numbers behind.

Module: ``coffin/diag``. Standalone probe: imports torch and the standard
library only, and nothing under ``coffin/`` imports it.

Constraints:
    * Never import an engine module. A probe must run while the engine is
      mid-refactor, and these numbers are the baseline evidence for that code.
    * Measurements are synthetic and single-box: the page cache is warm unless a
      key says ``cold_best_effort``, and DCU contention from other tenants is
      only visible in the ``_conditions`` block of the JSON. A number is
      comparable across runs on this host, not across hosts.
    * Every section stays wrapped in ``guarded``. One missing capability (pinned
      memory, ``hy-smi``, the NFS shard) must not void the rest of the run.
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


# Shared result-recording helpers. Each measurement goes to stdout and to RESULT
# at once, so a crash mid-run still leaves the earlier numbers on the console.
def rec(name: str, value: float, unit: str) -> None:
    """Record one measurement under its dotted key and echo it as a single line.

    Args:
        name: Dotted key such as ``pcie.h2d.pinned.2GB``. Callers diff keys
            across runs, so renaming one invalidates the stored JSON.
        value: Measured number. ``0.0`` with a non-bandwidth ``unit`` such as
            ``alloc-failed:Pinned`` marks a probe that could not run; do not
            read it as a zero-cost result.
        unit: Unit string appended verbatim, e.g. ``GB/s``, ``us``, ``GiB``.
    """
    RESULT[name] = {"value": value, "unit": unit}
    if isinstance(value, float):
        print(f"{name} {value:.4g} {unit}", flush=True)
    else:
        print(f"{name} {value} {unit}", flush=True)


def note(msg: str) -> None:
    """Print a ``#``-prefixed context line, ignored when the output is parsed.

    Args:
        msg: Free text describing conditions (device name, cache state, failure
            reason). Measurement lines stay machine-parseable because they never
            start with ``#``.
    """
    print(f"# {msg}", flush=True)


def sync() -> None:
    """Block until the DCU queue drains.

    Every timed region is bracketed by this: ``torch`` copy and matmul calls are
    asynchronous, so a timer stopped earlier would report launch latency instead
    of the transfer or kernel time.
    """
    torch.cuda.synchronize()


def median(xs):
    """Return the median of the samples, the statistic used for every timing.

    Args:
        xs: Timings of one repetition. The median rather than the mean, because
            a single preempted sample on a shared DCU skews a mean badly.

    Returns:
        Middle sample in seconds (mean of the two middle samples for an even
        count); it is not scaled, so the caller keeps the unit ``fn`` produced.
    """
    return statistics.median(xs)


def timed(fn, reps: int = 3, warmup: int = 1) -> float:
    """Time ``fn`` with one untimed warmup and return the median of ``reps`` runs.

    Args:
        fn: Zero-argument callable performing the measured work.
        reps: Timed repetitions. Three is enough to reject one outlier while
            keeping the multi-GiB sections short.
        warmup: Untimed calls first, covering first-touch allocation and kernel
            module load, which would otherwise land in the first sample.

    Returns:
        Median elapsed seconds per repetition, including the final ``sync``.
    """
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
    """Run one shell command and return its exit code with combined output.

    Args:
        cmd: Command line passed to the shell, so globs and redirections work.
        timeout: Seconds before the command is killed. The default bounds a hung
            ``dd`` on an NFS mount that stopped responding; without it the whole
            probe would wait forever.

    Returns:
        ``(returncode, stdout + stderr)``. Output is never raised away: the disk
        section prints the tail of a failed ``dd``, which is how a mount problem
        gets diagnosed.
    """
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def hy_smi() -> str:
    """Read VRAM and DCU utilisation from ``hy-smi`` as a short one-line summary.

    Returns:
        ``vram=<n> dcu=<n>`` for device 0, or ``hy-smi-error:<exc>`` /
        ``parse-error`` when the tool is absent or its column layout changed.
        A parse failure is reported rather than raised, because VRAM sampling is
        corroboration and must not abort the timed sections.
    """
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
    """Append one ``hy-smi`` sample labelled with the section that just finished.

    Args:
        tag: Section name (``init``, ``pcie``, ...). The sequence in
            ``_conditions.dcu_samples`` shows whether another tenant's DCU load
            overlapped a measurement, which is the first thing to check when a
            bandwidth number comes out low.
    """
    s = hy_smi()
    DCU_SAMPLES.append(f"{tag}: {s}")
    note(f"hy-smi after {tag}: {s}")


def guarded(name: str, fn) -> None:
    """Run one section, printing the traceback and carrying on if it fails.

    Args:
        name: Section name used in the failure line.
        fn: Zero-argument section entry point. Capabilities differ between DCU
            hosts and container images, so a missing one loses its own section
            only; aborting would throw away the VRAM and bandwidth numbers
            already collected.
    """
    try:
        fn()
    except Exception as exc:  # keep the later sections alive
        import traceback

        traceback.print_exc()
        note(f"section {name} FAILED: {type(exc).__name__}: {exc}")


# Environment and device properties.
def env_info() -> None:
    """Record device, cgroup and PCIe-link facts that decide the other numbers.

    ``env.cgroup_mem_max`` is the container's real ceiling, not the host's RAM
    total, and it is what caps CPU-resident tiers. The PCIe link width and speed
    bound every transfer measured by ``pcie``, so a slow link must be visible
    next to it.
    """
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
    """Measure H2D/D2H bandwidth across a size ladder, pinned and pageable.

    The ladder matters more than any single point: pageable copies fall back to
    a staging buffer, so the gap against pinned appears only at small sizes,
    while at 2 GB both saturate the link. The 4 KB and 1 MB rows bound the
    per-token cost of fetching a weight or a KV page that is not resident.
    """
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
    """Measure the copy, write and read HBM roof with 2 GiB buffers.

    Every kernel number is divided by one of these, so the three are kept apart:
    a copy pays read plus write traffic (hence the doubling), a ``zero_`` pays
    write only, and a ``sum`` pays read only. Using the wrong one as the
    denominator inflates or deflates a kernel's reported efficiency by 2x.
    """
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
    """Measure CPU DRAM read and memcpy bandwidth at 1 and 7 threads.

    The 1-thread row is the floor for any per-token CPU work that cannot be
    parallelised; 7 threads is what the engine can actually use on this host.
    ``copy_`` is counted as read plus write traffic, so its GB/s is not
    comparable with the read-only row without the factor of 2 already applied.
    """
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
    """Best-effort eviction of ``path`` from the page cache before a cold read.

    Args:
        path: File to evict. ``POSIX_FADV_DONTNEED`` only hints the kernel and a
            shared NFS client may refill it, which is why the cold keys are named
            ``cold_best_effort`` and must not be quoted as guaranteed cold.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except Exception as exc:
        note(f"fadvise(DONTNEED) failed on {path}: {exc}")


def time_cmd(cmd: str, nbytes: int, name: str) -> None:
    """Run a shell transfer command, record its throughput and elapsed seconds.

    Args:
        cmd: The transfer, normally a ``dd`` with ``status=none`` so its own
            summary does not pollute the parsed output.
        nbytes: Bytes the command moves. Passed explicitly because ``dd`` output
            is suppressed; nothing else knows the transfer size.
        name: Dotted key prefix for both ``<name>`` (GB/s) and
            ``<name>.seconds`` (s). On a non-zero exit the rate is recorded as
            ``0.0`` and the output tail goes to a ``#`` line, so a failure cannot
            be mistaken for a fast run.
    """
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
    """Issue ``os.pread`` calls at random offsets and summarise the latency.

    Args:
        path: File to read; its size clips ``span`` so a short file cannot
            produce an offset past EOF.
        n: Number of reads.
        bs: Bytes per read. 4096 is the block every filesystem here is formatted
            for, so a larger value would measure streaming, not random access.
        span: Bytes of the front of the file to sample. Bounded so the NFS
            random test does not pull a whole multi-GB shard over the wire.
        seed: PRNG seed, fixed so two runs touch the same offsets and the latency
            difference is the storage, not a different random draw.

    Returns:
        Dict with ``mean_us``, ``p50_us``, ``p99_us`` and ``iops``. The percentile
        is what a KV page fetch under load actually experiences; the mean hides
        the tail that causes a decode stall.
    """
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
    """Measure NFS and local-overlay disk cost per sequential and 4 KB read.

    This is the tier that decides whether paged KV can live on disk at all: a
    4 KB random read is the smallest unit the block manager fetches, so
    ``disk.*.rand4k.lat_p99`` is the stall a decode step pays on a miss. The warm
    NFS row comes first on purpose — the shard is usually in cache because the
    model was just loaded — and is then contrasted with the cold row.
    """
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
    """Measure how much VRAM is left for weights after the context is created.

    ``vram.weights_that_fit`` is the number the tier decision uses: it adds back
    the probe allocation to the free figure, so it answers "how large may the
    checkpoint be and still leave room to run". The 14_000_000_000 allocation is
    decimal GB (13.04 GiB), matching how checkpoint sizes are quoted, and it is
    touched so lazy allocation cannot report VRAM as free when it is committed.
    """
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
    """Measure whether CPU DRAM work hides behind GPU work or serialises with it.

    Both loops are sized to about 2 s from a single timed iteration, so the
    comparison is not biased by a short side finishing early.
    ``overlap.efficiency`` near 100 % means the engine may scan a CPU-resident
    tier during decode without costing tokens; near 0 % means the two contend for
    memory bandwidth on this host and the tier must be prefetched instead.
    """
    torch.set_num_threads(7)
    note("overlap: 4096^2 fp16 matmul loop (GPU) vs 2 GiB fp32 sum loop (7 CPU threads)")
    a = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")

    def gpu_iters(n: int) -> None:
        """Run ``n`` 4096^2 fp16 matmuls and drain the queue.

        Args:
            n: Iterations. The trailing ``sync`` is inside the loop count's scope
                so the caller's timer includes kernel time, not launches.
        """
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
        """Run ``n`` full reductions of a 2 GiB fp32 array on the CPU.

        Args:
            n: Iterations. Deliberately the same array the single-threaded CPU
                section sized, so this measures DRAM traffic and not allocation.
        """
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
    """Run every section in order and write ``hw_probe.json``.

    A section that raises is caught by ``guarded``, so the JSON is written with
    as many keys as the host allowed; a missing key means that capability was
    unavailable, not that it measured zero.
    """
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
