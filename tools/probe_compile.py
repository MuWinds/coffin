"""诊断用：把全部 ``cxrt`` ABI 库编一遍，错误压成一行打印。

由 ``card_verify --model-cmd`` 调（``push_run.py`` 的变体），只为把被报告截掉的
``error:`` 行捞出来。不进产物、不进测试。

torch JIT 扩展删掉之后，唯一的编译面就是 ``abi/*.cu`` 这六七个库，所以这里逐个建。
"""
import os
import sys
import traceback

sys.path.insert(0, os.environ.get("COFFIN_ROOT", "."))

LOG = "/data/home/2024240384035/coffin-work/cerr.txt"
text = ""
rc = 0
try:
    from cxrt import open_runtime

    rt = open_runtime()
    for name in ("norm", "gemv", "w8a8", "gemm", "attn", "moe", "conv", "solve"):
        getattr(rt, "build_" + name)()
except Exception:                       # noqa: BLE001 - 诊断脚本
    text = traceback.format_exc()
    rc = 1

with open(LOG, "w", encoding="utf-8") as fh:
    fh.write(text)

lines = [ln for ln in text.splitlines() if "error:" in ln]
print("ERR_LINES=%d" % len(lines))
chunks = []
for ln in lines[:4]:
    chunks.append(ln.split("kernels/")[-1][:95])
print("@".join(chunks)[-400:])
print("PROBE_RC=%d" % rc)
