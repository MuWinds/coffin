set -u

# Split one decode step into host dispatch time and device time, on the int8 serving path.
#
# The question: a C++ entry point that issues every launch itself would remove Python, ATen
# dispatch and pybind from every op. That only helps if host time is on the critical path, so
# this measures the split on the wall clock instead of assuming it.
#
# Card selection is shared with the other launchers; see coffin/diag/_pick_card.sh.

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

echo "--- host budget ---"
"$PY" -B -u diag/corex_probe/probe_host_budget.py
echo "HOST_BUDGET_EXIT=$?"
