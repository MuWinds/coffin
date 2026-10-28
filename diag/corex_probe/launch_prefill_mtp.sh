set -u

# Close two evidence gaps left by the prefill measurement:
#
#   * what the MTP drafter's prefill costs op by op (it is worth 221.8 ms at n=2048, the
#     largest no-kernel-work win found so far, but the five slow GEMMs only account for
#     ~52 ms of it),
#   * bf16 + paged + mtp_prefill=True, the one headline cell that was extrapolated.
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

echo "--- mtp prefill cost ---"
"$PY" -B -u diag/corex_probe/probe_prefill_mtp.py
echo "MTP_EXIT=$?"
