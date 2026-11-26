set -u

# 无 torch 跑一次真实对话：装别名 -> 导入 run_chat -> 贪心解码 4 个 token。
#
# 目的不是「跑得快」，而是把「模型层还能不能在没有真 torch 的进程里走完一次前向」
# 这个事实钉下来。失败时这里只打 traceback，不做兜底——兜底会把真正缺的那一步藏掉。
#
# 注意：上传的脚本不能含 `#!/bin/bash`（网关会把文件过滤成 0 字节并返回成功）。

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${COFFIN_ROOT:-$HERE/coffin}
[ -d "$ROOT" ] || ROOT=$HOME/coffin-work/base/coffin
PY=${PY:-/data/software/Anaconda3/2024.10-1/envs/py310/bin/python}
CKPT=${CKPT:-$HOME/models/Qwen3.5-4B-int8-nk}

cd "$ROOT" || { echo "MISSING_ROOT=$ROOT"; exit 66; }
export PYTHONPATH=$ROOT
export COREX_ROOT=${COREX_ROOT:-/usr/local/corex}
export COFFIN_RT_DIR=${COFFIN_RT_DIR:-$HOME/.coffin_corex_rt}
export COFFIN_RT_CXX=${COFFIN_RT_CXX:-$COREX_ROOT/bin/clang++}
export COFFIN_BACKEND=${COFFIN_BACKEND:-reference}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "[env] host=$(hostname) root=$ROOT ckpt=$CKPT"

"$PY" -B -u - "$CKPT" <<'PYEOF'
import sys
import traceback

ckpt = sys.argv[1]

import cxrt.install as inst
inst.install()
print("[alias] torch ->", sys.modules["torch"].__name__, flush=True)

try:
    import torch
    print("[alias] verify", inst.verify(), "zeros", tuple(torch.zeros(2, 2).shape), flush=True)
except Exception:
    traceback.print_exc()
    raise SystemExit(3)

sys.argv = ["run_chat", "--model", ckpt, "--int8", "--raw",
            "The capital of France is", "-n", "4", "--no-bench", "--no-graph"]
try:
    import run_chat
    run_chat.main()
except SystemExit as exc:
    print("[exit]", exc.code, flush=True)
except Exception:
    print("[fail] traceback follows", flush=True)
    traceback.print_exc()
    raise SystemExit(4)
PYEOF
RC=$?
echo "[run] rc=$RC"
exit $RC
