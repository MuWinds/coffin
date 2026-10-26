set -u

# Repeat the capture+replay attempt in one process, to tell a race from cumulative
# corruption. Two identical runs disagreed on the same node: one finished, one stopped
# at "graph replay" with no traceback. The index of the last completed attempt is what
# separates the two explanations.
#
# No cap around the repeat: a hang here is data (which attempt it died in), and killing
# it early throws that away. The job's own time limit bounds it.
#
# The 8x8 matmul gate runs first, because a node whose context already spins would
# produce a "hang" that says nothing about the graph.

export NODE=${NODE:-}
H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export PATH
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
export COFFIN_BACKEND=corex
export ENGINE_MODEL=$H/models/Qwen3.5-4B
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

if [ -n "${COFFIN_ROOT:-}" ]; then
    cd "$COFFIN_ROOT" || exit 1
else
    cd "$(dirname "$0")/../../.." || exit 1
fi
echo "HOST=$(hostname)  PWD=$(pwd)"

echo "--- card sanity ---"
timeout --foreground -k 10 60 "$PY" -B -u -c \
  "import torch; a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda'); b=a@a; torch.cuda.synchronize(); print('SANITY_OK', bool(torch.isfinite(b).all()), flush=True)"
if [ $? -ne 0 ]; then
    echo "CARD_UNHEALTHY on $(hostname); publish to another node"
    exit 64
fi

echo "--- repeat capture+replay ---"
"$PY" -B -u diag/corex_probe/probe_graph_repeat.py
echo "REPEAT_EXIT=$?"
