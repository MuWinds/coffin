"""Build the standalone int4 probe extension once (and print where it landed).

    MKL_THREADING_LAYER=GNU PYTHONPATH=/root/private_data/engine \
    TORCH_EXTENSIONS_DIR=/root/private_data/torch_ext python3 diag/build_int4.py
"""
import os

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx906")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/root/private_data/torch_ext")

from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
mod = load(
    name="gemv_int4_probe_ext",
    sources=[os.path.join(HERE, "gemv_int4_ext.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--offload-arch=gfx906"],
    verbose=True,
)
print("BUILD_OK", mod)
