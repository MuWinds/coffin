set -x
H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
# Same per-deployment extension cache as the conformance run: one build serves every
# script, and a fresh one would pay for a recompile of unchanged sources.
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
export COFFIN_BACKEND=corex
cd "$(dirname "$0")/coffin"
echo "HOST=$(hostname)  PWD=$(pwd)"
$PY -B -u backends/corex/diag/tune_gemv.py 2>&1
