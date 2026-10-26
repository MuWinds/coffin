set -x
H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_import
cd "$(dirname "$0")/coffin"
echo "HOST=$(hostname)  PWD=$(pwd)"
# The engine is imported as a set of top-level packages (ops, models, backends...),
# exactly as the entry points do it: coffin/ itself is the sys.path root.
$PY -B -u diag/corex_probe/probe_import.py 2>&1
