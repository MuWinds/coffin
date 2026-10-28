set -u

# What a kernel launch actually costs here, measured with device events rather than a trace.
#
# This is the number that decides whether removing launches can help, which PERF_COREX §8
# previously answered "no" to from a profiler sum that over-counted. No model is loaded, so
# it costs a couple of minutes and can be re-run after any dispatch change.
#
# Card selection is shared with the other launchers; see coffin/diag/_pick_card.sh.

H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export PATH
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
export COFFIN_BACKEND=corex
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

echo "--- launch floor ---"
"$PY" -B -u diag/corex_probe/probe_launch_floor.py
echo "FLOOR_EXIT=$?"
