set -u

# Conformance: every CoreX op against an independent host oracle, plus per-op bandwidth.
#
# This is the gate that catches a kernel change that is fast and wrong. The int8 GEMV
# dispatch changed (PERF_COREX.md §10) along with the int8 kernel family, so it runs after
# them, not only at port time.
#
# Card selection is shared with the other launchers; see coffin/diag/_pick_card.sh.

H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export PATH
# A per-deployment extension cache: a shared TORCH_EXTENSIONS_DIR makes concurrent jobs
# contend on one build baton and hang on this cluster. The path comes from the environment,
# never from the backend's source, so this script is the only place it is decided.
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
export COFFIN_BACKEND=corex
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

if [ -n "${COFFIN_ROOT:-}" ]; then
    cd "$COFFIN_ROOT" || exit 1
else
    cd "$(dirname "$0")/../../.." || exit 1
fi
echo "HOST=$(hostname)  PWD=$(pwd)"

echo "--- card selection ---"
. "$(dirname "$0")/../../../diag/_pick_card.sh"
pick_card || exit 64

echo "--- conformance ---"
"$PY" -B -u backends/corex/diag/check_ops.py
echo "OPS_EXIT=$?"
