"""Build the standalone int4 probe extension once (and print where it landed).

Answers whether ``gemv_int4_ext.cu`` compiles on this card at all, before an
int4 probe run blames the kernel for a build failure. Run it first when
``dbg_int4.py`` or ``probe_gemv_int4.py`` reports that the extension is missing:

    MKL_THREADING_LAYER=GNU PYTHONPATH=/root/private_data/engine \
    TORCH_EXTENSIONS_DIR=/root/private_data/torch_ext python3 diag/build_int4.py

A successful build prints ``BUILD_OK`` followed by the loaded module; a
compilation failure raises out of ``load`` with the compiler log on stderr,
which is the actual compiler diagnosis to read.

Module: ``coffin/diag``. Standalone build helper: compiles one ``.cu`` next to
it through ``torch.utils.cpp_extension.load`` and imports nothing from
``coffin/``.

Constraints:
    * ``PYTORCH_ROCM_ARCH`` is set to ``gfx906`` as a default before torch is
      imported. Torch reads the arch when it initialises, so setting it after the
      import would silently build for whatever the environment already had.
    * ``TORCH_EXTENSIONS_DIR`` must stay per-deployment. A shared directory makes
      concurrent runs rebuild and contend for the same build baton, which looks
      like a hang rather than a build.
    * ``setdefault`` rather than assignment: an operator who points these at a
      different arch or cache keeps control.
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
