#!/usr/bin/env bash
set -u
# 无 torch 全量跑一次：证明「推理链路里没有真 torch 也能出 token」。
H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
ENGINE=$H/coffin-work/base/coffin-dv
export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export PYTHONPATH=$ENGINE${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=16
echo "host=$(hostname) mem_gib=$(free -g | awk '/Mem:/{print $2}')"
/usr/bin/time -f 'wall=%es rss=%MKB' "$PY" -B -u -m cxrt.run_qwen35 \
    --model "$H/models/Qwen3.5-4B-int8-nk" --raw "The capital of France is" -n 8
echo "rc=$?"
