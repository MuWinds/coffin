set -u

# Load-path attribution with a hard cap. Self-contained (one uploaded script).
#
# The symptom: an end-to-end run stopped before its first "loaded in" line, while the
# same run had finished in seconds earlier. Between the two, the reducers gained
# closing barriers -- and a barrier inside a reducer reached from a divergent context
# deadlocks silently. That is a device-side spin: no traceback, no stderr, and it runs
# until the *job* limit unless something kills it. So everything here is capped.
#
# First a plain matmul, because the very first question is whether the card itself is
# healthy. Then the load, with and without the autotune sweep: bench=False skips the
# load-time kernel selection, so if the plain load passes and the bench load hangs,
# the barrier is in the sweep's call shapes rather than in weight loading.

CAP=${CAP:-150}
H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export PATH
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
export COFFIN_BACKEND=corex
export ENGINE_MODEL=$H/models/Qwen3.5-4B
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

cd "$(dirname "$0")/coffin" || exit 1
echo "HOST=$(hostname)  PWD=$(pwd)"

echo ""
echo "=== A. card sanity (8x8 matmul) ==="
timeout --foreground -k 10 90 "$PY" -B -u -c \
  "import torch; a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda'); b=a@a; torch.cuda.synchronize(); print('MATMUL_OK', bool(torch.isfinite(b).all()), flush=True)"
echo "A_EXIT=$?"

echo ""
echo "=== B. load WITHOUT autotune (bench=False) ==="
timeout --foreground -k 10 "$CAP" "$PY" -B -u diag/corex_probe/probe_loadcheck.py
echo "B_EXIT=$?"

echo ""
echo "=== C. load WITH autotune (bench=True) ==="
timeout --foreground -k 10 "$CAP" "$PY" -B -u diag/corex_probe/probe_loadcheck.py --bench
echo "C_EXIT=$?"

echo ""
echo "LOADCHECK_DONE"
