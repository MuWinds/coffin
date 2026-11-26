set -u

# 真 torch + corex 后端，直接读 nk 预量化目录跑 4B：这条是内核路（int8 权重不被反量化成
# 稠密），与 cxrt 那条无 torch 的稠密验证路径互补。它成立的前提正是「加载器不再要求
# kn」——盘上是 [N, K]，后端要的 [K, N] 由 pack.to_kn 在打包时转一次。
#
# 注意：上传的脚本不能含 `#!/bin/bash`（网关会把文件过滤成 0 字节并返回成功）。

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${COFFIN_ROOT:-$HERE/coffin}
[ -d "$ROOT" ] || ROOT=$HOME/coffin-work/base/coffin
PY=${PY:-/data/software/Anaconda3/2024.10-1/envs/py310/bin/python}
MODEL=${MODEL:-$HOME/models/Qwen3.5-4B-int8-nk}
RAW=${RAW:-"The capital of France is"}
NEW=${NEW:-16}
INT8=${INT8:-1}
GRAPH=${GRAPH:-1}
EXTRA=${EXTRA:-}

cd "$ROOT" || { echo "MISSING_ROOT=$ROOT"; exit 66; }
export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export PYTHONPATH=$ROOT
export COFFIN_BACKEND=${COFFIN_BACKEND:-corex}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$HOME/torch_ext_coffin_corex}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false

echo "[env] host=$(hostname) backend=$COFFIN_BACKEND model=$MODEL n=$NEW int8=$INT8"

FLAG=()
[ "$INT8" = "1" ] && FLAG=(--int8)
# ``EXTRA`` 会被 sync 的 ``--env`` 当成单个赋值传下来，所以带空格的开关必须从
# 环境里翻译成参数，不能直接塞进 EXTRA（``export "--no-graph"`` 是语法错）。
NOGRAPH=()
[ "$GRAPH" = "0" ] && NOGRAPH=(--no-graph)
"$PY" -B -u run_chat.py --model "$MODEL" --raw "$RAW" -n "$NEW" --max-seq 256 \
    "${FLAG[@]}" "${NOGRAPH[@]}" $EXTRA
RC=$?
echo "[chat] rc=$RC"
exit "$RC"
