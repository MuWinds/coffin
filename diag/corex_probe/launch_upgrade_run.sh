set -x
H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
cd "$(dirname "$0")/corex_probe"
echo "HOST=$(hostname)"
$PY -B -u probe_torch_upgrade_run.py 2>&1