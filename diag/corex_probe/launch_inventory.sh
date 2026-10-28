set -u

# Attribution for one eager int8 decode step, at launch granularity.
#
# The three optimisations this backs (fusing the split-K reduce, removing the glue
# operators, and cutting the delta writeback) are all launch-count reductions, so the probe
# that decides them has to produce a launch inventory rather than a bandwidth table. It also
# prints the greedy ids, which is the regression bar for a change that is supposed to be
# arithmetically identical.
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

echo "--- launch inventory ---"
"$PY" -B -u diag/corex_probe/probe_launch_inventory.py
echo "INVENTORY_EXIT=$?"
