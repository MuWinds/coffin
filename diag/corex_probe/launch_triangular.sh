set -x
H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
cd "$(dirname "$0")/coffin"
echo "HOST=$(hostname)"
$PY -B -u diag/corex_probe/probe_triangular.py 2>&1