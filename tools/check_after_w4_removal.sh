set -u

# 删掉 4-bit 全链路之后的静态校验：qwen4exp 还能不能 import、参考后端的算子表还全不全。
# 这两件事都不需要真权重，也不需要编内核，所以它跑得比任何一次推理都便宜，适合放在
# 每次动完 `models/` 或 `backends/protocol.py` 之后。
#
# 注意：上传的脚本不能含 `#!/bin/bash`（网关会把文件过滤成 0 字节并返回成功）。

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${COFFIN_ROOT:-$HERE/coffin}
[ -d "$ROOT" ] || ROOT=$HOME/coffin-work/base/coffin
PY=${PY:-/data/software/Anaconda3/2024.10-1/envs/py310/bin/python}

cd "$ROOT" || { echo "MISSING_ROOT=$ROOT"; exit 66; }
export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export PYTHONPATH=$ROOT
export TOKENIZERS_PARALLELISM=false

echo "[env] host=$(hostname)"

"$PY" -B - <<'PYCODE'
import importlib
mods = [
    "backends", "backends.protocol", "backends.reference", "backends.corex",
    "ops.gemv", "ops.embed", "quantized_load", "int8_weights",
    "models.qwen35.moe", "models.qwen35.parallel",
    "models.qwen4exp.proj", "models.qwen4exp.moe", "models.qwen4exp.gdn",
    "models.qwen4exp.weights", "models.qwen4exp.engine",
]
bad = 0
for name in mods:
    try:
        importlib.import_module(name)
        print(f"[import] ok   {name}")
    except Exception as exc:                                  # noqa: BLE001
        bad += 1
        print(f"[import] FAIL {name}: {type(exc).__name__}: {exc}")
from backends.protocol import Ops
print(f"[ops] names={len(Ops.NAMES)} has_moe_grouped_gemm="
      f"{'moe_grouped_gemm' in Ops.NAMES}")
import sys
sys.exit(1 if bad else 0)
PYCODE
RC=$?
echo "[imports] rc=$RC"

"$PY" -B -u tests/test_backends.py
RC2=$?
echo "[tests] rc=$RC2"
exit $(( RC != 0 || RC2 != 0 ))
