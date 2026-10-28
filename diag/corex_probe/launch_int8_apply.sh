set -u

# End-to-end validation of the int8 prefill fixes: the current spelling, the cheaper
# cast-then-scale spelling, and the upper bound with every weight pre-dequantised.
#
# The micro-benchmark said the cheap spelling is 1.88x on the dequant itself; this says
# what that is worth on a real prompt, and how it scales when the prompt is longer than
# one chunk (the prefill loop is chunk-major, so each weight is dequantised once per
# chunk).
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

echo "--- int8 prefill, end to end ---"
"$PY" -B -u diag/corex_probe/probe_int8_apply.py
echo "I8A_EXIT=$?"
