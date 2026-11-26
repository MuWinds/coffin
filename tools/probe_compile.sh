#!/usr/bin/env bash
# 诊断用：只编 torch 扩展（不跑模型），把编译错误压成一行写到 stdout 尾部。
set -u
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOG=/data/home/2024240384035/coffin-work/cerr.txt

cd "$ROOT"
COFFIN_BACKEND=corex TORCH_EXTENSIONS_DIR=/data/home/2024240384035/torch_ext_coffin_corex \
    "$PY" -B -u -c "import backends.corex as m; m._kernels()" > "$LOG" 2>&1
RC=$?

echo "ERR_LINES=$(grep -c 'error:' "$LOG")"
grep 'error:' "$LOG" | sed 's#^.*/kernels/##' | cut -c1-95 | tr '\n' '@' | tail -c 420
echo
echo "PROBE_RC=$RC"
