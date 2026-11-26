set -u

# 无 torch 端到端跑 Qwen3.5-4B：装别名 -> 折稠密权重 -> 建模型 -> prefill -> 贪心解码。
# 默认 24 个 token，用来判断「第一个 token 对了之后还能不能继续对」。
#
# 注意：上传的脚本不能含 `#!/bin/bash`（网关会把文件过滤成 0 字节并返回成功）。

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${COFFIN_ROOT:-$HERE/coffin}
[ -d "$ROOT" ] || ROOT=$HOME/coffin-work/base/coffin
PY=${PY:-/data/software/Anaconda3/2024.10-1/envs/py310/bin/python}
CKPT=${CKPT:-$HOME/models/Qwen3.5-4B-int8-nk}
NEW=${NEW:-24}
SEQ=${SEQ:-256}

cd "$ROOT" || { echo "MISSING_ROOT=$ROOT"; exit 66; }
export PYTHONPATH=$ROOT
export COFFIN_BACKEND=${COFFIN_BACKEND:-reference}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "[env] host=$(hostname) root=$ROOT ckpt=$CKPT new=$NEW"

"$PY" -B -u -m cxrt.run_qwen35 --model "$CKPT" -n "$NEW" --max-seq "$SEQ"
RC=$?
echo "[run] rc=$RC"
exit $RC
