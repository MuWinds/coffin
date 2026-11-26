set -u

# 对拍 corex 的两支投影内核：int8 nk（gemv_nk_i8*）与 bf16 kn（gemv_kn*）。
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
"$PY" -B -u tools/probe_corex_gemv.py "$@"
RC=$?
echo "[gemv] rc=$RC"
exit "$RC"
