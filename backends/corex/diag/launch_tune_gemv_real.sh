set -u

# Sweep the int8 GEMV against the shapes a real decode step calls, and locate the split-K
# crossover.
#
# Card selection is shared with the other launchers; see ``coffin/diag/_pick_card.sh``.

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
    cd "$(dirname "$0")/../../.." || exit 1
fi
echo "HOST=$(hostname)  PWD=$(pwd)"

echo "--- card selection ---"
. "$(dirname "$0")/../../../diag/_pick_card.sh"
pick_card || exit 64

echo "--- real-shape sweep ---"
"$PY" -B -u -c "
import sys; sys.path.insert(0, '.')
from backends.corex.diag import tune_gemv_real
sys.exit(tune_gemv_real.main())
"
echo "REAL_TUNE_EXIT=$?"
