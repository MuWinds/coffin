set -u

# Qwen3.5-4B（nk int8 预量化目录）的吞吐与显存扫描：预填充 128/512/2048/4096 四档，
# 每档之后再解码 32 个 token，最后报峰值显存。一律关图（当前工作树的 rms_norm 正在改，
# 图捕获在 capture() -> prefill 上必挂）。
#
# 注意：上传的脚本不能含 `#!/bin/bash`（网关会把文件过滤成 0 字节并返回成功）。

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${COFFIN_ROOT:-$HERE/coffin}
[ -d "$ROOT" ] || ROOT=$HOME/coffin-work/base/coffin
PY=${PY:-/data/software/Anaconda3/2024.10-1/envs/py310/bin/python}
MODEL=${MODEL:-$HOME/models/Qwen3.5-4B-int8-nk}
LENS=${LENS:-128,512,2048,4096}
DECODE=${DECODE:-32}

cd "$ROOT" || { echo "MISSING_ROOT=$ROOT"; exit 66; }
export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export PYTHONPATH=$ROOT
export COFFIN_BACKEND=${COFFIN_BACKEND:-corex}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$HOME/torch_ext_coffin_corex}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false

echo "[env] host=$(hostname) backend=$COFFIN_BACKEND model=$MODEL lens=$LENS"

"$PY" -B -u tools/bench_qwen35_nk.py --model "$MODEL" --max-seq 4096 \
    --lens "$LENS" --decode "$DECODE" --int8
RC=$?
echo "[bench] rc=$RC"
exit "$RC"
