set -u

# Occamy 1.0 → nk int8 预量化目录。走的就是 4B 那条 `tools/quant_disk.py`：它的
# `architectures` 是 Qwen3_5MoeForConditionalGeneration，逐专家张量
# `layers.{i}.mlp.experts.{e}.{proj}.weight` 会按 `_EXPERT_KEY` 融合成
# `[E, N, K]` 的 q8-moe 栈（gate_up 沿 N 拼 gate/up，down 单独一条）。
#
# 注意：上传的脚本不能含 `#!/bin/bash`（网关会把文件过滤成 0 字节并返回成功）。

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${COFFIN_ROOT:-$HERE/coffin}
[ -d "$ROOT" ] || ROOT=$HOME/coffin-work/base/coffin
PY=${PY:-/data/software/Anaconda3/2024.10-1/envs/py310/bin/python}
MODEL=${MODEL:-$HOME/models/occamy-1.0}
OUT=${OUT:-$HOME/models/occamy-1.0-W8A8-nk}
LAYOUT=${LAYOUT:-nk}
DEVICE=${DEVICE:-cuda:0}

cd "$ROOT" || { echo "MISSING_ROOT=$ROOT"; exit 66; }
export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export PYTHONPATH=$ROOT
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$HOME/torch_ext_coffin_corex}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false

echo "[env] host=$(hostname) src=$MODEL out=$OUT layout=$LAYOUT device=$DEVICE"
df -h "$HOME/models" | tail -2

"$PY" -B -u tools/quant_disk.py --checkpoint "$MODEL" --out "$OUT" --layout "$LAYOUT" \
    --device "$DEVICE"
RC=$?
echo "QUANT_RC=$RC"
df -h "$HOME/models" | tail -2
exit "$RC"
