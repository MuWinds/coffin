set -u

# Step-budget attribution on one node with several cards, picking a healthy card first.
#
# One job asks for several accelerators and decides inside the job which of the cards it
# actually received can do arithmetic. Submitting one job per card is worse in both
# directions: the scheduler keeps handing successive submissions to whichever node it just
# freed -- repeatedly one with a wedged context -- and each submission is another chance for
# the WebVPN gateway to answer with a 403 block page.
#
# Card health is a real question on this partition. An earlier device-side hang left several
# cards unable to finish an 8x8 matmul, and the symptom -- a process that prints nothing and
# never exits -- is indistinguishable in the log from a wedged kernel.
#
# The checks run concurrently with a short cap, so an unkillable card cannot hold the job:
# an 8x8 matmul is microseconds of device work, so the only budget a healthy card needs is
# the interpreter plus the CUDA context. A corpse simply never wins the race.
#
# The tree is assumed to be in place: this launcher runs from a persistent checkout, so
# there is no unpack step and no per-run upload of the engine.

H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
export COFFIN_BACKEND=corex
export ENGINE_MODEL=$H/models/Qwen3.5-4B
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PROBE=${PROBE:-diag/corex_probe/probe_step_budget.py}

cd coffin || exit 1
echo "HOST=$(hostname)  PWD=$(pwd)"
echo "EXT_CACHE=$(ls -d "$TORCH_EXTENSIONS_DIR"/coffin_corex_ops_v* 2>/dev/null | tr '\n' ' ')"

N=$(timeout -s KILL 60 "$PY" -B -c "import torch; print(torch.cuda.device_count())" \
    2>/dev/null || echo 0)
echo "CARDS_VISIBLE=$N"
if [ "$N" -eq 0 ]; then
    echo "NO_CARDS: the job holds no usable accelerator"
    exit 64
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

IMPORT_MS=$("$PY" -B -c "import time
t = time.time()
import torch
print(int((time.time() - t) * 1000))" 2>/dev/null || echo 4000)
CAP=$(( (IMPORT_MS + 2000) / 1000 + 1 ))
if [ "$CAP" -lt 3 ]; then CAP=3; fi
if [ "$CAP" -gt 8 ]; then CAP=8; fi
echo "CARD_CHECK_CAP=${CAP}s (import measured ${IMPORT_MS}ms)"

for i in $(seq 0 $((N - 1))); do
    (
        CUDA_VISIBLE_DEVICES=$i timeout -s KILL "$CAP" "$PY" -B -c "
import torch
a = torch.randn(8, 8, dtype=torch.bfloat16, device='cuda')
b = torch.randn(8, 8, dtype=torch.bfloat16, device='cuda')
c = a @ b
torch.cuda.synchronize()
print('OK' if bool(torch.isfinite(c).all()) else 'NONFINITE')
" >"$TMP/c$i" 2>&1
    ) &
done

GOOD=""
i=0
while [ "$i" -lt 20 ]; do
    for c in $(seq 0 $((N - 1))); do
        if [ -f "$TMP/c$c" ] && grep -qx "OK" "$TMP/c$c" 2>/dev/null; then
            GOOD=$c
            break 2
        fi
    done
    sleep 0.5
    i=$((i + 1))
done

for c in $(seq 0 $((N - 1))); do
    verdict="(no answer)"
    if [ -f "$TMP/c$c" ]; then
        verdict=$(tr '\n' ' ' <"$TMP/c$c" | sed 's/  */ /g')
    fi
    echo "  card $c: ${verdict:-<empty>}"
done

# No waiting on the losers. Each is already capped by its own timeout, and waiting on a
# wedged context is the one thing that cannot complete.
if [ -z "$GOOD" ]; then
    echo "NO_HEALTHY_CARD on $(hostname) among $N visible card(s)"
    exit 65
fi

echo ""
echo "USING_CARD=$GOOD on $(hostname)"
export CUDA_VISIBLE_DEVICES=$GOOD

echo "--- $PROBE ---"
# stderr merged into stdout on purpose. A torch extension build reports progress on stderr
# only, and this portal's file API refuses to serve the stderr stream (500), so keeping them
# apart makes a several-minute compile look exactly like a hang -- which is how several jobs
# were wasted. Every diagnostic below is worth more interleaved than separated.
"$PY" -B -u "$PROBE" 2>&1
echo "PROBE_EXIT=$?"
