set -u

# 对拍分组 int8 MoE 投影（GEMV 与张量核两档）vs 逐专家 oracle。Occamy 的形状量级。
#
# 注意：上传的脚本不能含 `#!/bin/bash`（网关会把文件过滤成 0 字节并返回成功）。

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${COFFIN_ROOT:-$HERE/coffin}
[ -d "$ROOT" ] || ROOT=$HOME/coffin-work/base/coffin
PY=${PY:-/data/software/Anaconda3/2024.10-1/envs/py310/bin/python}

cd "$ROOT" || { echo "MISSING_ROOT=$ROOT"; exit 66; }
export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export PYTHONPATH=$ROOT
export COFFIN_BACKEND=corex
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$HOME/torch_ext_coffin_corex}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "[env] host=$(hostname)"
"$PY" -B -u tools/probe_corex_moe_i8.py
RC=$?
echo "[moe] rc=$RC"
exit "$RC"
