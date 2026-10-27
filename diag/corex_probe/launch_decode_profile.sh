set -u

# Where does the decode step's time go? Attribution, not tuning.
#
# The port is correct but 4x slower than gfx906 for the same model, and the two obvious
# explanations are already accounted for (the graph is on, worth 1.47x; int8 weights are
# worth ~1.43x on gfx906). Guessing at which of the remaining ~30 ops dominates would
# waste jobs, so this measures the per-shape projection budget, the real step, and the
# profiler's own ranking, and lets the numbers pick the next target.
#
# Card sanity first: a node whose context is spinning times out on an 8x8 matmul, and
# that failure looks identical to "the probe hung", so it is checked before anything
# else and exits without loading a model.

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
    echo "CARD_UNHEALTHY on $(hostname)"
    exit 64
fi

echo "--- decode attribution ---"
"$PY" -B -u diag/corex_probe/probe_decode_profile.py
echo "PROFILE_EXIT=$?"
