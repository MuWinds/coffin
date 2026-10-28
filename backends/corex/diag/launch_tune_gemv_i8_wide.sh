set -u

# Sweep the W8A16 GEMV over weight-load width x threads x split-K on the shapes a real
# decode step calls.
#
# Card selection is shared with the other launchers; see ``coffin/diag/_pick_card.sh``.
#
# Pass shape names as arguments to sweep a subset, which is the cheap way to retry one
# shape without paying for ``lm_head`` again:
#     bash launch_tune_gemv_i8_wide.sh mlp_up
#
# This launcher must stay LF-only. A CRLF here fails silently with a 0-byte log that reads
# like a scheduler fault rather than a script fault.

H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export PATH
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
export COFFIN_BACKEND=corex
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# The sweep passes shape names through to the script; an empty argument list means all of
# them, which is why this is built rather than passed inline.
SHAPE_ARGS=""
for _shape in "$@"; do
    SHAPE_ARGS="$SHAPE_ARGS '$_shape',"
done

if [ -n "${COFFIN_ROOT:-}" ]; then
    cd "$COFFIN_ROOT" || exit 1
else
    cd "$(dirname "$0")/../../.." || exit 1
fi
echo "HOST=$(hostname)  PWD=$(pwd)"

echo "--- card selection ---"
. "$(dirname "$0")/../../../diag/_pick_card.sh"
pick_card || exit 64

echo "--- width x threads x split sweep ---"
"$PY" -B -u -c "
import sys; sys.path.insert(0, '.')
from backends.corex.diag import tune_gemv_i8_wide
sys.exit(tune_gemv_i8_wide.main([$SHAPE_ARGS]))
"
echo "I8_WIDE_TUNE_EXIT=$?"
