#!/usr/bin/env bash
set -u

# 一条作业跑完「去 torch + bf16 GEMM + 新口径」这套改动的全部上卡验收。
#
# 判据只有两份：``ixblas/docs/README.md`` 的 PERF 行，和 ``docs/COREX-GEMM-NK.md`` §2。
# 这里做的事就是把文档里那张表拿去和卡上的实测对一遍，外加静态审计与离机单测。
#
# 选址是逐卡跑一个 8x8 bf16 矩阵乘，5 秒超时：同机箱里每次会话坏哪张卡都不一样，
# 节点名不作判据（见 platform/scow/AGENTS.md §2）。

H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
ENGINE_ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$H/coffin-work/card_report.md

export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
export COFFIN_BACKEND=corex
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$ENGINE_ROOT${PYTHONPATH:+:$PYTHONPATH}

echo "HOST=$(hostname)  ENGINE=$ENGINE_ROOT"
echo "PY=$PY"

AVAILABLE=${CUDA_VISIBLE_DEVICES:-0}
PICKED=
for DEV in $(echo "$AVAILABLE" | tr ',' ' '); do
    if CUDA_VISIBLE_DEVICES=$DEV timeout --foreground -k 2 5 "$PY" -B -u -c \
        "import torch; a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda'); b=a@a; torch.cuda.synchronize(); raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then PICKED=$DEV; break; fi
    echo "DEAD_CARD=$DEV"
done
if [ -z "$PICKED" ]; then echo NO_HEALTHY_CARD; exit 64; fi
export CUDA_VISIBLE_DEVICES=$PICKED
echo "PICKED_CARD=$PICKED"

# 端到端那一档由调用方给（``sync.py push --env CARD_MODEL_CMD=...``）：本脚本不替人
# 挑模型与提示词。给了就两面都跑一遍（同一命令两次，只看确定性），报告 §6 才有内容。
EXTRA=()
if [ -n "${CARD_MODEL_CMD:-}" ]; then
    echo "MODEL_CMD=$CARD_MODEL_CMD"
    EXTRA=(--model-cmd "$CARD_MODEL_CMD")
fi

"$PY" -B -u "$ENGINE_ROOT/tools/card_verify.py" --out "$OUT" "${EXTRA[@]}"
RC=$?
echo "----- report -----"
cat "$OUT" 2>/dev/null
echo "----- rc=$RC -----"
exit $RC
