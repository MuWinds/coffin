set -u

# 把 Qwen3.5-4B 的 bf16 原模型量化成 int8，落盘布局是 nk（int8 [N, K] + fp32 [N]）。
#
# 为什么是 nk：解码侧的分组/单流内核按行索引，一行输出通道要是一段连续字节；kn 那份
# 得在运行时转置，白占一份显存。这里一次性按行落盘，之后加载器直接读。
#
# 做完立刻自检（tools/check_nk_disk.py）：形状、标尺方向、反量化相对误差。
#
# 注意：上传的脚本不能含 `#!/bin/bash`（网关会把文件过滤成 0 字节并返回成功）。

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${COFFIN_ROOT:-$HERE/coffin}
[ -d "$ROOT" ] || ROOT=$HOME/coffin-work/base/coffin
PY=${PY:-/data/software/Anaconda3/2024.10-1/envs/py310/bin/python}
SRC=${SRC:-$HOME/models/Qwen3.5-4B}
OUT=${OUT:-$HOME/models/Qwen3.5-4B-int8-nk}
LAYOUT=${LAYOUT:-nk}

cd "$ROOT" || { echo "MISSING_ROOT=$ROOT"; exit 66; }
export PYTHONPATH=$ROOT
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "[env] host=$(hostname) src=$SRC out=$OUT layout=$LAYOUT"
ls "$SRC" | head -4
df -h "$HOME" | tail -1

if [ -e "$OUT" ]; then
    if [ -f "$OUT/quant_index.json" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "[quant] OUTPUT_ALREADY_EXISTS=$OUT (自检后重跑请先删掉它，或 FORCE=1)"
        exit 73
    fi
    # 没有索引说明上一次跑到一半失败，没有任何可用东西；FORCE=1 是显式要求覆盖重跑。
    echo "[quant] removing $OUT (force=${FORCE:-0})"
    rm -rf "$OUT"
fi

"$PY" -B -u tools/quant_disk.py \
    --checkpoint "$SRC" \
    --out "$OUT" \
    --weight-dtype bf16 \
    --layout "$LAYOUT" \
    --device cuda:0
RC=$?
echo "[quant] rc=$RC"
if [ "$RC" -ne 0 ]; then exit "$RC"; fi

echo "[quant] on-disk:"
du -sh "$OUT"
ls "$OUT" | head -5

"$PY" -B -u tools/check_nk_disk.py --checkpoint "$SRC" --out "$OUT" --sample 4
RC2=$?
echo "[check] rc=$RC2"
exit "$RC2"
