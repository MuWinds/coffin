#!/usr/bin/env bash
set -u
# 三件事：门槛自己跑一遍；再让**普通入口**（run_chat）在 COFFIN_NO_TORCH=1 下跑一次，
# 看它死在哪一步——那是「链路里还剩 torch」的下一处现场。
H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
ENGINE=$H/coffin-work/base/coffin-dv
export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export PYTHONPATH=$ENGINE${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=16
M=$H/models/Qwen3.5-4B-int8-nk

echo "## 1) 无 torch 门槛"
"$PY" -B -u "$ENGINE/tools/audit_no_torch.py" --runtime
echo "rc_gate=$?"

echo "## 2) 普通入口 + COFFIN_NO_TORCH=1（稠密反量化）"
COFFIN_NO_TORCH=1 timeout -k 5 900 "$PY" -B -u "$ENGINE/run_chat.py" --model "$M" \
    --no-graph --raw "The capital of France is" -n 4
echo "rc_chat_dense=$?"

echo "## 3) 普通入口 + COFFIN_NO_TORCH=1（--int8）"
COFFIN_NO_TORCH=1 timeout -k 5 900 "$PY" -B -u "$ENGINE/run_chat.py" --model "$M" --int8 \
    --no-graph --raw "The capital of France is" -n 4
echo "rc_chat_int8=$?"
