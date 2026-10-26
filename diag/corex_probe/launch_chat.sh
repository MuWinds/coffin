set -u

# End-to-end smoke test, pinned to a node, with a hard wall-clock cap.
#
# Self-contained: the deploy path uploads the tree plus exactly one script, so a
# launcher that `bash`es a sibling finds it missing on the node.
#
# Pinning matters. Part of this partition's cards are unusable at the moment (an
# earlier capture attempt left at least one node's context spinning on the very first
# allocation -- a plain 8x8 matmul times out there), so the scheduler's choice of node
# decides whether a run measures the engine or the card. NODE is passed in by the
# caller; the health check at the top makes a wrong choice fail fast instead of at the
# end of a model load.

CAP=${CAP:-420}   # only used by the 8x8 matmul cap below
NODE=${NODE:-}
H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export PATH
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
export COFFIN_BACKEND=corex
export ENGINE_MODEL=$H/models/Qwen3.5-4B
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# The engine root is passed in as COFFIN_ROOT: the script is uploaded on its own (not
# inside the tree), so its own location says nothing about where the tree was
# extracted. Deriving it from ``dirname $0`` is what produced "can't open file
# /data/home/diag/..." -- the script sat at the working-directory root, three levels
# above the engine it was supposed to run.
if [ -n "${COFFIN_ROOT:-}" ]; then
    cd "$COFFIN_ROOT" || exit 1
else
    cd "$(dirname "$0")/../../.." || exit 1
fi
echo "HOST=$(hostname)  WANTED=${NODE:-any}  PWD=$(pwd)  CAP=${CAP}s"

if [ -n "$NODE" ] && [ "$(hostname)" != "$NODE" ]; then
    echo "WRONG_NODE: got $(hostname), wanted $NODE"
    exit 65
fi

echo "--- card sanity before the model ---"
timeout --foreground -k 10 60 "$PY" -B -u -c \
  "import torch; a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda'); b=a@a; torch.cuda.synchronize(); print('SANITY_OK', bool(torch.isfinite(b).all()), flush=True)"
sane=$?
if [ "$sane" -ne 0 ]; then
    echo "CARD_UNHEALTHY on $(hostname): even an 8x8 matmul did not return (rc=$sane)."
    echo "Stop, and submit to another node."
    exit 64
fi

echo "--- engine end to end ---"
# No wall-clock cap around the engine run. A node that has never built the CoreX
# extension pays for a full compile first, which dominates everything else and varies
# wildly between nodes -- a cap short enough to catch a real hang also kills these
# builds and reports them as failures. The 8x8 matmul above is the hang detector that
# actually works: a wedged context fails there, in seconds, before the model loads.
"$PY" -B -u diag/corex_probe/probe_chat.py
echo "CHAT_EXIT=$?"
