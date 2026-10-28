set -u

# Three prefill costs a fix can remove, measured before anyone writes the fix:
#
#   * the five GEMMs that cost ~10x a tensor-core GEMM (non-tensor-core fallback shapes),
#   * the delta scan's chunk length, which is the hardcoded default 64 regardless of the
#     model-level prefill chunk,
#   * the per-call dequantisation that linear_prefill does on int8 weights.
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

echo "--- prefill attribution ---"
"$PY" -B -u diag/corex_probe/probe_prefill_attr.py
echo "ATTR_EXIT=$?"
