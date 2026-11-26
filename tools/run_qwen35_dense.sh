set -u

# 无 torch 端到端出一条 token 链：装别名 -> 折稠密权重 -> 建 Qwen35 -> prefill -> 贪心解码。
#
# 与 run_torchfree_chat.sh 的区别：那条走 quantized_load（要求 kn 后端），这条走模型
# 自己的稠密路径（weights= 入口），因此不需要适配层，也不需要 23 个还没搬的算子。
# 失败时只打 traceback，不做兜底。
#
# 注意：上传的脚本不能含 `#!/bin/bash`（网关会把文件过滤成 0 字节并返回成功）。

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${COFFIN_ROOT:-$HERE/coffin}
[ -d "$ROOT" ] || ROOT=$HOME/coffin-work/base/coffin
PY=${PY:-/data/software/Anaconda3/2024.10-1/envs/py310/bin/python}
CKPT=${CKPT:-$HOME/models/Qwen3.5-4B-int8-nk}
NEW=${NEW:-8}
SEQ=${SEQ:-256}
LIMIT=${LIMIT:-}

cd "$ROOT" || { echo "MISSING_ROOT=$ROOT"; exit 66; }
export PYTHONPATH=$ROOT
export COREX_ROOT=${COREX_ROOT:-/usr/local/corex}
export COFFIN_RT_DIR=${COFFIN_RT_DIR:-$HOME/.coffin_corex_rt}
export COFFIN_RT_CXX=${COFFIN_RT_CXX:-$COREX_ROOT/bin/clang++}
export COFFIN_BACKEND=${COFFIN_BACKEND:-reference}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "[env] host=$(hostname) root=$ROOT ckpt=$CKPT new=$NEW seq=$SEQ limit=${LIMIT:-none}"
df -h "$HOME" | tail -1
free -g | head -2

ARGS=(--model "$CKPT" -n "$NEW" --max-seq "$SEQ")
if [ -n "$LIMIT" ]; then ARGS+=(--limit-gib "$LIMIT"); fi

"$PY" -B -u -m cxrt.run_qwen35 "${ARGS[@]}"
RC=$?
echo "[run] rc=$RC"
exit $RC
