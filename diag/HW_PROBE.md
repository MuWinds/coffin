# Hardware probe — 1× DCU Z200SM_80 (16 GB), 2026-10-22 23:48 CST

Synthetic measurements for the **Qwen3.8-27B int4 / 256k context** design. No model weights
were read or downloaded. Script: [`probe_hw.py`](probe_hw.py) (remote copy at
`/root/private_data/engine/diag/probe_hw.py`; JSON at `.../hw_probe.json`).

Reproduce in the container:

```sh
cd /root/private_data/engine
MKL_THREADING_LAYER=GNU PYTHONPATH=$PWD TORCH_EXTENSIONS_DIR=/root/private_data/torch_ext \
    python3 diag/probe_hw.py
```

## Conditions

- Pod `cgrp-v87dg651`, 1× DCU `Z200SM_80` (gfx906, 64 CU, 16 GB), cgroup **7 cores / 60 GiB**,
  DTK 26.04, `torch 2.7.1+das.opt1.dtk2604`.
- **GPU idle throughout**: `hy-smi` samples at every section boundary show `DCU% = 0.0`
  (no compile / kernel work from other agents overlapped the run). No re-run needed.
- Host: Hygon C86 7380 32-core, node 251 GiB; PCIe links as enumerated in `/sys` are mostly
  `8.0 GT/s x16` (PCIe 3.0-class) plus some `16.0 GT/s` entries (unidentified devices —
  see gaps).
- `MKL_THREADING_LAYER=GNU`; `OMP`/torch threads pinned to 1 or 7 explicitly.
- NFS = `/root/private_data` (hstor, 442 GB free); local = `/tmp` on overlayfs.
- Page cache warm (the 4B model was used today) unless a row says `cold_best_effort`;
  `fadvise(DONTNEED)` was applied before the cold rows.

## Headline constants

| Constant | Value | Notes |
| --- | --- | --- |
| HBM read (fp16 sum, 1 GiB) | **630 GB/s** | matches the 4B engine's ~700 GB/s copy |
| HBM D2D copy / write | **700 / 678 GB/s** | 2 GiB buffers |
| PCIe H2D pinned | **13.1 GB/s** | flat 8 MB→2 GB, so not a startup artifact |
| PCIe D2H pinned | **14.1 GB/s** | |
| PCIe H2D pageable | 7.1 GB/s | D2H pageable is 14.0 GB/s (staged) |
| PCIe copy latency (pinned) | **47.6 µs @ 4 KB, 108 µs @ 1 MB** | per copy + sync |
| CPU DRAM read, 1 / 7 threads | **20.3 / 35.5 GB/s** | 2 GiB fp32 `sum()` |
| CPU DRAM memcpy, 1 / 7 threads | 11.6 / 12.2 GB/s | torch `copy_` is single-thread-bound |
| CPU/GPU overlap efficiency | **88.6 %** | 2 s GPU matmul ∥ 2 s 7-thread DRAM read |
| VRAM total / free after init | 15.98 / **15.50 GiB** | driver+context = 0.49 GiB |
| 14.0 GB int8 alloc leaves | **2.46 GiB free** | so 14.0 GB decimal ≈ max weight blob |
| NFS sequential read (warm / attempted-cold) | **0.564 / 0.517 GB/s** | `dd bs=1M count=1500` |
| NFS 4 KB random | p50 7.1 µs, mean 697 µs, **p99 14.9 ms**, 1434 IOPS | p50 is page-cache hits |
| Local overlay write / seq read | **0.264 / 0.469 GB/s** | `/tmp`, 2 GiB file, fsync |
| Local 4 KB random | p50 171 µs, 6016 IOPS | no page-cache hits after fadvise |

## Full raw output

```
# start 2026-10-22 23:48:08
# script=/root/private_data/engine/diag/probe_hw.py
# environment
env.device_total 15.98 GiB
# torch=2.7.1 device=Z200SM_80 SMs=64
# cpu=Hygon C86 7380 32-core Processor
env.cgroup_mem_max 60 GiB
# cgroup cpu.max=700000 100000
# meminfo MemAvailable:   255848300 kB
# PCIe: torch copies, sync every rep, median of 3 (warmup 1)
pcie.h2d.pinned.8MB 11.98 GB/s
pcie.d2h.pinned.8MB 12.83 GB/s
pcie.h2d.pinned.64MB 12.97 GB/s
pcie.d2h.pinned.64MB 13.96 GB/s
pcie.h2d.pinned.256MB 13.07 GB/s
pcie.d2h.pinned.256MB 14.06 GB/s
pcie.h2d.pinned.1GB 13.12 GB/s
pcie.d2h.pinned.1GB 14.12 GB/s
pcie.h2d.pinned.2GB 13.13 GB/s
pcie.d2h.pinned.2GB 14.13 GB/s
pcie.h2d.pageable.8MB 6.385 GB/s
pcie.d2h.pageable.8MB 5.247 GB/s
pcie.h2d.pageable.64MB 5.078 GB/s
pcie.d2h.pageable.64MB 13.87 GB/s
pcie.h2d.pageable.256MB 7.101 GB/s
pcie.d2h.pageable.256MB 14.01 GB/s
pcie.h2d.pageable.1GB 7.122 GB/s
pcie.d2h.pageable.1GB 14.01 GB/s
pcie.h2d.pageable.2GB 7.088 GB/s
pcie.d2h.pageable.2GB 14.02 GB/s
pcie.latency.h2d.4KB 47.64 us
pcie.latency.d2h.4KB 47.56 us
pcie.latency.h2d.1MB 108.1 us
pcie.latency.d2h.1MB 101.6 us
# HBM reference: 2 GiB uint8 buffers, 1 GiB fp16 sum
hbm.copy_2GiB_traffic 699.6 GB/s
hbm.write_2GiB 678.3 GB/s
hbm.read_1GiB 630 GB/s
# CPU DRAM: 2 GiB fp32 array (sum = read-only, copy_ = read+write traffic)
cpu.read.1t.2GiB 20.29 GB/s
cpu.memcpy.1t.2GiB 11.58 GB/s
cpu.read.7t.2GiB 35.47 GB/s
cpu.memcpy.7t.2GiB 12.17 GB/s
# disk: NFS shard 5.33 GB, local overlay /tmp 2 GiB file
# nfs shard bytes=5329398688
disk.nfs.seq.warm 0.5641 GB/s
disk.nfs.seq.warm.seconds 2.788 s
disk.nfs.seq.cold_best_effort 0.517 GB/s
disk.nfs.seq.cold_best_effort.seconds 3.042 s
disk.local.write 0.2635 GB/s
disk.local.write.seconds 8.151 s
disk.local.rand4k.lat_mean 165.8 us
disk.local.rand4k.lat_p50 170.9 us
disk.local.rand4k.lat_p99 204.6 us
disk.local.rand4k.iops 6016 IOPS
disk.local.seq.read 0.4693 GB/s
disk.local.seq.read.seconds 4.576 s
disk.nfs.rand4k.lat_mean 696.9 us
disk.nfs.rand4k.lat_p50 7.12 us
disk.nfs.rand4k.lat_p99 1.488e+04 us
disk.nfs.rand4k.iops 1434 IOPS
vram.total 15.98 GiB
vram.free_after_init 15.5 GiB
vram.driver_plus_context 0.4874 GiB
vram.allocated_14GB 13.04 GiB
vram.reserved_14GB 13.04 GiB
vram.free_after_14GB 2.458 GiB
vram.weights_that_fit 15.5 GiB
# hy-smi after overlap: vram=2% dcu=0.0%
overlap.gpu_only 1.981 s
overlap.cpu_only 2.357 s
overlap.both 2.583 s
overlap.efficiency 88.58 %
# iters gpu=292 cpu=33 one_gpu=6.8ms one_cpu=59.6ms
# wrote /root/private_data/engine/diag/hw_probe.json
```

(PCIe-link enumeration and per-section `hy-smi` samples are in `hw_probe.json` on the pod;
the `# pcie links:` line is omitted here for length.)

## What this means for 27B int4 / 256k

### Byte budget

Qwen3.8-27B: 64 layers = 48 gated-DeltaNet + **16 full-attention** (`24 Q / 4 KV / d=256`,
partial RoPE 64). Only the 16 full layers need KV.

- **KV per context token (int8, incl. per-token scales)**: `16 layers × (4 kv × 256 × 2) B
  + 16×4×2×2 B ≈ 33 KB/token`.
  | context | 4k | 32k | 64k | 128k | **256k** |
  | --- | --- | --- | --- | --- | --- |
  | int8 KV | 0.14 GB | 1.08 GB | 2.16 GB | 4.33 GB | **8.66 GB** |
  | int4 KV (group 32) | 0.075 GB | 0.60 GB | 1.21 GB | 2.42 GB | **4.83 GB** |
- **Weights**: 27.78 B params @ int4 symmetric group 128 + fp16 scale ≈ 14.32 GB
  (13.34 GiB) + MTP ≈ 14.5 GB total; text-only figures are ~1 B lower.
- **RSS state**: 48 DeltaNet layers × 48 V-heads × 128×128 fp32 = **151 MB** (not per-token).
- **VRAM** usable = **15.50 GiB** after driver/context. int4 weights + MTP leave
  ~**1.1–1.5 GiB** on a single card: ≈ 32–40k tokens of int8 KV, nothing more.

### Per-token decode cost at 256k (derived; constants above)

| design | bytes/token | bottleneck | est. |
| --- | --- | --- | --- |
| 1 card, 32k hot + 224k cold int8 on host, **CPU attention** | 14.5 GB HBM + 7.6 GB host | CPU 35.5 GB/s | **214 ms → ~4.6 tok/s** |
| 1 card, cold int4 KV on host, CPU attention | 14.5 + 4.2 GB | CPU | 119 ms → ~8.4 tok/s |
| 1 card, cold int8 KV **streamed over PCIe** | 7.6 GB @ 13.1 GB/s | PCIe | 580 ms → ~1.7 tok/s |
| 1 card, H2O/Quest evict to 32k (all in HBM) | 15.6 GB | HBM | 25–29 ms → 30–40 tok/s (quality risk) |
| 2 cards, **pipeline** (32 layers/card, int8 KV) | 2 × 11.6 GB serially | HBM | 37 ms → ~27 tok/s |
| 2 cards, **tensor parallel** (int4 weights, int8 KV) | 11.6 GB/card | HBM, parallel | **18–23 ms → 25–40 tok/s** |
| any card, disk-resident KV in the loop | 8.66 GB @ 0.5 GB/s | disk | ~17 s/token — unusable |

Notes: CPU attention traffic is `2 × 4 kv × 256 × N` per layer (K is shared by the 6 Q heads
per KV head); its arithmetic is ~100 GFLOP at 256k, i.e. ~10 % of the memory time, so it is
memory-bound. The 88.6 % overlap number means CPU cold attention and GPU weight GEMV can run
concurrently, but the CPU side is still the wall. 2-card figures assume per-card 7.25 GB
weights + 4.33 GB KV and ignore the (tiny) TP all-reduce: 2×5120×2 B per layer ≈ 1.3 MB/token,
0.2 ms even host-staged at 13 GB/s — **no RCCL/P2P needed for 2-way TP**.

### Prefill at 256k (derived)

Total FLOPs ≈ `2·P·N + 16·12288·N²` ≈ 1.46e16 + 1.35e16 = **2.8e16**. The 4B engine's
measured 18 TFLOP/s fp16 GEMM, scaled: 1 card ≈ 26 min, 2 cards ≈ 13 min at peak;
budget **20–45 min**. Single-card prefill would place the 8.66 GB KV on the host (D2H 14 GB/s
≈ 0.6 s to write, ~550 GB of PCIe reads over the whole prefill ≈ 42 s) — noise next to compute.
2-card prefill keeps KV in HBM. This is a one-time cost per unique prefix; radix reuse matters
more than any kernel micro-optimization.

### Disk-tier verdict

NFS and local disk are both ~0.26–0.56 GB/s sequential and have 0.17–15 ms 4 KB tails.
Use disk only for **prefix snapshots / model weights**, never in the per-token path. Host RAM
(60 GiB, soon 120 GiB with 2 cards) is the only sane second tier: 8.66 GB of KV fits with
plenty of page cache to spare.

## Recommendation in one line each

1. **Use the second DCU (quota is 2/2 free) and tensor-parallel the 27B**: it turns 256k
   decode from ~4 tok/s (host-KV) into ~25–40 tok/s and gives 2× prefill, with KV fully in HBM.
2. If 1 card is a hard constraint: hot KV in VRAM (~32k tokens), cold KV in **host RAM** read
   by **CPU-side attention** (35.5 GB/s beats 13.1 GB/s PCIe by 2.7×), int4 for the cold tier
   (≈8.5 tok/s at 256k); NFS only for snapshots. Expect ~4 tok/s with int8 cold KV.
3. Do not design around PCIe-streamed KV or disk-resident KV: 47 µs/copy and 1.4k IOPS make
   both non-viable inside a decode step.
4. int4 weights are forced on either card count (14.3 GB); plan a W4A16 dequant GEMM for
   prefill and a dequant GEMV for decode, with fp16 dequant scratch kept small.
5. Keep the radix/prefix cache: a 256k prefill is 20–45 min, and agent workloads re-send the
   same prefix.

## Gaps / caveats

- GEMV effective bandwidth for large int4 matrices was **not** measured here (needs the new
  kernel); decode tables use 500–630 GB/s as the HBM-bound range and the 4B int8 engine's
  317 GB/s as the pessimistic end.
- 2-card TP and inter-card peer bandwidth were not measured (requires a 2-card pod); the TP
  estimate assumes PCIe 3.0-class ~13 GB/s, which is adequate because all-reduce is ~1.3 MB/token.
- L2/PCIe: `/sys` shows a mix of 8.0 and 16.0 GT/s links; the measured 13.1/14.1 GB/s is the
  operative number regardless of which BDF is the DCU.
- CPU bandwidth with 14 cores (2-card pod) is unmeasured; 7-thread scaling was only 1.75× over
  1 thread, so do not assume 2× from doubling cores.
- NFS random-read p50 (7 µs) is page-cache hits; the p99 (14.9 ms) is the honest cold latency.
- Local overlay write (0.26 GB/s) was slower than NFS; treat both as bulk-only.
