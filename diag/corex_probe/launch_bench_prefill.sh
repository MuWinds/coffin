set -u

# Prefill throughput on this card: headline, chunk sweep, profiler, per-shape GEMM.
#
# Nothing in the tree has ever measured prefill here. The decode path is fully attributed
# (PERF_COREX.md) and the only prefill figure is "1056.5 ms (5 tokens)", which routes
# through the decode path and measures nothing about batched prefill.
#
# Card selection is shared with the other launchers; see coffin/diag/_pick_card.sh. It is
# sourced by this file's own path because the job's working directory is the parent of
# coffin, not coffin itself.

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
    cd "$(dirname "$0")/../.." || exit 1
fi
echo "HOST=$(hostname)  PWD=$(pwd)"

echo "--- card selection ---"
. "$(dirname "$0")/../../diag/_pick_card.sh"
pick_card || exit 64

echo "--- prefill benchmark ---"
"$PY" -B -u diag/corex_probe/bench_prefill.py
echo "PREFILL_EXIT=$?"
