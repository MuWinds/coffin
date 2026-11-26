#!/usr/bin/env bash
set -u
# 同一条命令连跑三遍：kn 全删之后，这条 nk-only 链路的输出是否可复现。
H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
ENGINE=$H/coffin-work/base/coffin-dv
export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
export COFFIN_BACKEND=corex
export PYTHONPATH=$ENGINE${PYTHONPATH:+:$PYTHONPATH}

AVAILABLE=${CUDA_VISIBLE_DEVICES:-0}
for DEV in $(echo "$AVAILABLE" | tr ',' ' '); do
    if CUDA_VISIBLE_DEVICES=$DEV timeout --foreground -k 2 5 "$PY" -B -u -c \
        "import torch; a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda'); b=a@a; torch.cuda.synchronize(); raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then export CUDA_VISIBLE_DEVICES=$DEV; echo "PICKED_CARD=$DEV"; break; fi
done

for i in 1 2 3; do
    echo "===== run $i ====="
    "$PY" -B -u "$ENGINE/run_chat.py" --model "$H/models/Qwen3.5-4B-int8-nk" \
        --int8 --no-graph --raw "The capital of France is" -n 16 2>&1 | sed -n '/output:/,$p'
done
