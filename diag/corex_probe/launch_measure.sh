set -u

# End-to-end decode measurement: the launch inventory, then the greedy continuation.
#
# This is the single job that judges the whole fusion batch, so it measures the number the
# work was aimed at (launches per step) and the number that says the arithmetic did not
# move (the greedy token ids). Keeping both in one process matters: a launch count from one
# build and a token list from another is how a stale extension disguises itself as a win.
#
# The extension cache is overridable so this can be run against a per-agent build before
# the batch is reconciled to one name -- see FUSION_PLAN.md section 4.
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

# Which build to exercise. Defaults to the reconciled production name; a per-agent run
# passes COFFIN_COREX_EXT=... in the job environment.
export COFFIN_COREX_EXT=${COFFIN_COREX_EXT:-coffin_corex_ops_v14}

if [ -n "${COFFIN_ROOT:-}" ]; then
    cd "$COFFIN_ROOT" || exit 1
else
    cd "$(dirname "$0")/../.." || exit 1
fi
echo "HOST=$(hostname)  PWD=$(pwd)"
echo "EXT=${COFFIN_COREX_EXT}"
echo "EXT_DIRS=$(ls -d "$TORCH_EXTENSIONS_DIR"/"$COFFIN_COREX_EXT" 2>/dev/null || echo '<not built yet>')"

echo "--- card selection ---"
. "$(dirname "$0")/../../diag/_pick_card.sh"
pick_card || exit 64

echo "--- launch inventory + greedy ids ---"
"$PY" -B -u diag/corex_probe/probe_launch_inventory.py
echo "INVENTORY_EXIT=$?"
