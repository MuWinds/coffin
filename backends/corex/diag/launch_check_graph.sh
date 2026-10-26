set -x
H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
# Same cache as the conformance run: one build serves both scripts, and keeping them
# separate would pay for two compiles of the same sources.
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
export COFFIN_BACKEND=corex
# Scope note for whoever reads this log: the script checks two ops in isolation and says
# nothing about whether the whole decode step is capturable. A green run here is NOT
# permission to set _GRAPH_VERIFIED = True -- that was tried once and the replay hung the
# device until the scheduler killed the job at its wall-clock limit.
cd "$(dirname "$0")/coffin"
echo "HOST=$(hostname)  PWD=$(pwd)"
$PY -B -u backends/corex/diag/check_graph.py 2>&1
