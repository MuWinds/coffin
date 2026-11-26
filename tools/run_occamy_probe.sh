set -u

# Occamy 1.0 W8A8 的装载探针：无 torch 进程里问一次「量化产物能不能装」。
#
# 期望看到的是 engine.load_weights 的 NotImplementedError——那条 lane 明确没接。
# 这条脚本存在的意义是把「不能跑」变成一条带作业号的事实，而不是读代码得出的推断。
#
# 注意：上传的脚本不能含 `#!/bin/bash`（网关会把文件过滤成 0 字节并返回成功）。

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${COFFIN_ROOT:-$HERE/coffin}
[ -d "$ROOT" ] || ROOT=$HOME/coffin-work/base/coffin
PY=${PY:-/data/software/Anaconda3/2024.10-1/envs/py310/bin/python}
CKPT=${CKPT:-$HOME/models/occamy-1.0-W8A8}

cd "$ROOT" || { echo "MISSING_ROOT=$ROOT"; exit 66; }
export PYTHONPATH=$ROOT
export COFFIN_BACKEND=${COFFIN_BACKEND:-reference}

echo "[env] host=$(hostname) root=$ROOT ckpt=$CKPT"
ls "$CKPT" | head -3

"$PY" -B -u - "$CKPT" <<'PYEOF'
import sys, traceback
ckpt = sys.argv[1]
import cxrt.install as inst
inst.install()
print("[alias]", sys.modules["torch"].__name__, flush=True)
from models.qwen4exp import engine
print("[probe] engine:", engine.__name__, flush=True)
try:
    w = engine.load_weights(ckpt, quant=True)
    print("[probe] quant=True returned keys:", sorted(w), flush=True)
except Exception as exc:
    print(f"[probe] quant=True refused: {type(exc).__name__}: {exc}", flush=True)
try:
    w = engine.load_weights(ckpt, quant=False)
    src = w["source"]
    print("[probe] bf16 lane source keys:", len(list(src.keys())), flush=True)
except Exception as exc:
    print(f"[probe] bf16 lane: {type(exc).__name__}: {exc}", flush=True)
    traceback.print_exc()
PYEOF
RC=$?
echo "[run] rc=$RC"
exit 0
