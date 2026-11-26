#!/usr/bin/env bash
set -u
# 「torch 完全不在场」的现场证据：本进程里 torch 是壳、真包从未被导入，模型能不能跑。
H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
ENGINE=$H/coffin-work/base/coffin-dv
export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export PYTHONPATH=$ENGINE${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

HR=$(printf '=%.0s' $(seq 1 60))
echo "$HR"
echo "1) 身份探针：装壳之后 torch 到底是谁"
"$PY" -B -u - <<'EOF'
import sys
import cxrt.install as inst
inst.install()
import torch
print("verify:", inst.verify())
print("torch 对象:", torch)
print("torch 文件:", getattr(torch, "__file__", "?"))
print("真 torch 在 sys.modules 里:", "torch" in sys.modules and not inst.verify())
print("torch.Tensor:", getattr(torch, "Tensor", "?"))
print("torch.cuda.is_available():", torch.cuda.is_available())
EOF

echo "$HR"
echo "2) 逐层对拍用的稠密折表（限 2 GiB，只验通路）"
"$PY" -B -u -m cxrt.run_qwen35 --model "$H/models/Qwen3.5-4B-int8-nk" \
    --raw "The capital of France is" -n 4 --limit-gib 2
echo "rc_limit=$?"
