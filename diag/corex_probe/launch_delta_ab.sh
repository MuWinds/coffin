set -u

# Same-session A/B of the fused delta kernel against the three-op chain it replaced.
#
# Cross-job comparison cannot settle this: the two paths differ by ~0.3 ms on a ~20 ms step,
# and the wall varies by about that much between nodes and runs. Measuring both in one
# process, alternating, puts them under the same clocks.
#
# Card selection is shared with the other launchers; see coffin/diag/_pick_card.sh.

H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export PATH
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
export COFFIN_BACKEND=corex
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export COFFIN_COREX_EXT=${COFFIN_COREX_EXT:-coffin_corex_ops_v14}

if [ -n "${COFFIN_ROOT:-}" ]; then
    cd "$COFFIN_ROOT" || exit 1
else
    cd "$(dirname "$0")/../.." || exit 1
fi
echo "HOST=$(hostname)  PWD=$(pwd)"

echo "--- card selection ---"
. "$(dirname "$0")/../../diag/_pick_card.sh"
pick_card || exit 64

echo "--- delta A/B ---"
"$PY" -B -u diag/corex_probe/probe_delta_ab.py
echo "DAB_EXIT=$?"
