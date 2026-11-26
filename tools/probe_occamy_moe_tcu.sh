set -u

# 单卡把 Occamy 第 0 层 MoE 按行数扫一遍：张量核分组 / dp4a 分组 / 稠密 bf16 三方对拍。
#
# 注意：上传的脚本不能含 `#!/bin/bash`（网关会把文件过滤成 0 字节并返回成功）。

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${COFFIN_ROOT:-$HERE/coffin}
[ -d "$ROOT" ] || ROOT=$HOME/coffin-work/base/coffin
PY=${PY:-/data/software/Anaconda3/2024.10-1/envs/py310/bin/python}
MODEL=${MODEL:-$HOME/models/occamy-1.0-W8A8-nk}
LAYER=${LAYER:-0}
ROWS=${ROWS:-8,256,512,1024,2048}

cd "$ROOT" || { echo "MISSING_ROOT=$ROOT"; exit 66; }
export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export PYTHONPATH=$ROOT
export COFFIN_BACKEND=${COFFIN_BACKEND:-corex}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$HOME/torch_ext_coffin_corex}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false

echo "[env] host=$(hostname) model=$MODEL layer=$LAYER rows=$ROWS"
"$PY" -B -u tools/probe_occamy_moe_tcu.py --model "$MODEL" --layer "$LAYER" --rows "$ROWS"
RC=$?
echo "[moe-tcu] rc=$RC"
exit "$RC"
