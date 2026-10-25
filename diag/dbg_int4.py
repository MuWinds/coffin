"""Tiny correctness debug for the int4 probe kernel (K=256, N=2).

Answers whether the int4 GEMV probe kernel's packing and scale convention match
the Python reference, on a matrix small enough to print in full. This is the
first thing to run when ``probe_gemv_int4.py`` shows a bandwidth-plausible but
numerically wrong result: with K=256, N=2 every intermediate fits on screen, so
a packing or scale-layout error is visible by eye instead of inferred from a
tolerance.

Read it by comparing the printed rows: ``kernel`` must equal ``ref``. If they
differ only by a constant factor, the scale placement is wrong (a per-row scale
applied where the kernel expects a per-128-group one); if ``ref`` matches
``no-scale ref``, the scales themselves were quantised away. Both rows are
printed precisely so the failure mode can be told apart.

Run inside the container, on the DCU, after ``build_int4.py`` has populated the
extension cache:

    python3 diag/dbg_int4.py

Module: ``coffin/diag``. Standalone probe: builds and imports ``gemv_int4_ext``
and reaches into it directly, importing nothing from ``coffin/``.

Constraints:
    * The input is deterministic arithmetic, not random, so the same mismatch
      reproduces every run and the printed tables can be compared across hosts.
    * ``gfx906`` and the extensions directory are defaulted before torch imports,
      for the same reasons as ``build_int4.py``.
    * K=256 is two 128-element quantisation groups on purpose: one group would
      not show a group-indexing error at all.
"""
import os
import sys

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx906")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/root/private_data/torch_ext")
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = load(name="gemv_int4_probe_ext", sources=[os.path.join(HERE, "gemv_int4_ext.cu")],
           extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--offload-arch=gfx906"])

N, K = 2, 256
w = torch.arange(N * K, dtype=torch.float32, device="cuda").reshape(N, K) % 17 - 8
w = (w * 0.01).to(torch.bfloat16)
x = (torch.arange(K, dtype=torch.float32, device="cuda") % 7 - 3).to(torch.bfloat16)


def q4(w):
    n, k = w.shape
    wf = w.float().reshape(n, k // 128, 128)
    s = wf.abs().amax(-1).clamp_min(1e-8) / 7.0
    q = (wf / s[..., None]).round().clamp_(-8, 7).to(torch.int8)
    u = (q + 8).reshape(n, k).to(torch.uint8).to(torch.int16)
    packed = ((u[:, 0::2] | (u[:, 1::2] << 4)) & 0xFF).to(torch.uint8).contiguous()
    return packed, s.to(torch.float16).contiguous()


w4, s4 = q4(w)
xs = (x.float().reshape(-1, 8).sum(-1) * 8.0).contiguous()
y = EXT.gemv_w4a16(w4, s4, xs, x, 1).float()
torch.cuda.synchronize()

# reconstruct from packed bytes
wd = w4.to(torch.int64)
u = torch.zeros(N, K, dtype=torch.int64, device="cuda")
u[:, 0::2] = wd & 0xF
u[:, 1::2] = (wd >> 4) & 0xF
s4f = s4.float().repeat_interleave(128, dim=1)
ref = ((u - 8).float() * s4f) @ x.float()
print("w4 shapes", w4.shape, s4.shape, xs.shape)
print("s4", s4.cpu())
print("xs first 2", xs[:2].cpu())
print("kernel", y.cpu())
print("ref   ", ref.cpu())
print("no-scale ref", ((u - 8).float() @ x.float()).cpu())
