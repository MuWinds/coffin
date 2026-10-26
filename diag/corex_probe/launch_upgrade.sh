set -x
H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
# Separate extension cache per deployment: a shared one makes concurrent jobs
# contend on a single build baton and hang (observed on the sibling SCNet cluster).
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_upgrade
cd "$(dirname "$0")"
cd corex_probe
echo "HOST=$(hostname)  PWD=$(pwd)"
$PY -B -u probe_torch_upgrade.py 2>&1
