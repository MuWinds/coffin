set -u

# glue-attn's acceptance run: B1 (kv_quant), B2 (silu_mul) and the fused rope, measured on
# the same probe that produced the 23.527 ms / 2407-launch baseline. Three passes because the
# three questions are different:
#
#   1. probe_launch_inventory.py -- launches/step, BASELINE step_ms= and the greedy ids.
#      The ids are the bar: every one of these three changes is supposed to be arithmetically
#      identical to the torch chain it replaces, so a changed id is a bug and not a trade.
#   2. check_ops.py -- per-op conformance against the host oracle. Its attention section is
#      the one that reads the int8 cache, so it is also the only place that would catch
#      kv_quant and attn_slice_kernel disagreeing about the scale.
#   3. a direct A/B of the three new kernels against their torch chains, at the model's real
#      shapes. The inventory says whether the step got faster; this says *which* kernel did
#      it and whether the arithmetic is bit-identical, which is much cheaper to interpret
#      than a 70-launch delta inside a profile.
#
# COFFIN_COREX_EXT isolates the build: without it a job can silently attach to another agent's
# cached objects and measure a kernel nobody changed.

H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export PATH
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
export COFFIN_BACKEND=corex
export COFFIN_COREX_EXT=v13attn
export ENGINE_MODEL=$H/models/Qwen3.5-4B
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

if [ -n "${COFFIN_ROOT:-}" ]; then
    cd "$COFFIN_ROOT" || exit 1
else
    cd "$(dirname "$0")/../.." || exit 1
fi
echo "HOST=$(hostname)  PWD=$(pwd)  EXT=${COFFIN_COREX_EXT}"

echo "--- card selection ---"
. "$(dirname "$0")/../../diag/_pick_card.sh"
pick_card || exit 64

echo ""
echo "=== 1. launch inventory + greedy ids ==="
"$PY" -B -u diag/corex_probe/probe_launch_inventory.py
echo "INVENTORY_EXIT=$?"

echo ""
echo "=== 2. per-op conformance ==="
"$PY" -B -u backends/corex/diag/check_ops.py
echo "CHECK_OPS_EXIT=$?"

echo ""
echo "=== 3. A/B of the three fused glue kernels against their torch chains ==="
"$PY" -B -u diag/corex_probe/probe_glue_ab.py
echo "GLUE_AB_EXIT=$?"

echo ""
echo "GLUE_ATTN_DONE"
