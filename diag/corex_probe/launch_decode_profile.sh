set -u

# Where does the decode step's time go? Attribution, not tuning.
#
# The port is correct but 4x slower than gfx906 for the same model, and the two obvious
# explanations are already accounted for (the graph is off by default and is a 0.93x
# regression here anyway; int8 weights are worth ~1.08x on this part). Guessing at which of
# the remaining ops dominates would waste jobs, so this measures the per-shape projection
# budget, the real step, and the profiler's own ranking.
#
# The step this reports is the DENSE path: create_model does not pass quant=True, so the GEMV
# it times is the bf16 kernel. For the int8 path that serving actually uses, run
# backends/corex/diag/launch_bench_step_i8.sh instead.
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

echo "--- decode attribution ---"
"$PY" -B -u diag/corex_probe/probe_decode_profile.py
echo "PROFILE_EXIT=$?"
