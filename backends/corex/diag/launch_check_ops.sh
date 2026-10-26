set -x
H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
# A per-deployment extension cache: a shared TORCH_EXTENSIONS_DIR makes concurrent
# jobs contend on one build baton and hang on this cluster. The path comes from the
# environment, never from the backend's source, so this script is the only place it
# is decided.
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
export COFFIN_BACKEND=corex
cd "$(dirname "$0")/coffin"
echo "HOST=$(hostname)  PWD=$(pwd)"
$PY -B -u backends/corex/diag/check_ops.py 2>&1
