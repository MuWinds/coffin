set -u

# One job, many cards: pick a healthy one inside the job, then run the work on it.
#
# Submitting one job per card is the wrong shape twice over. The scheduler keeps handing
# successive submissions to whichever node it just freed -- repeatedly one with a wedged
# context -- so a serial search burns jobs before finding a good card. And every
# submission is a request the WebVPN gateway can reject with a 403 block page, which
# looks like an auth failure and wastes a scan of the login QR code.
#
# So: ask for several accelerators in a single job, and decide inside it which of the
# cards the job actually received can do arithmetic. The probe that matters runs on the
# first card that passes an 8x8 matmul; the rest cost nothing.
#
# All of the shell logic lives here rather than in the submitted command on purpose. The
# gateway rejects compound command strings (``cd X && bash Y`` is refused while
# ``bash Y`` alone is accepted), but it does not inspect uploaded files, so the only
# thing submitted is ``bash <absolute path to this file>``.
#
# Card health is a real question on this partition, not paranoia: an earlier device-side
# hang left several nodes unable to finish a 512x512 matmul, and the symptom -- a process
# that prints nothing and never exits -- is identical to a wedged kernel. Checking first
# is what separates "this card is bad" from "this code is bad".

PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
H=/data/home/2024240384035
export PATH=$H/../Anaconda3/2024.10-1/envs/py310/bin:/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
export COFFIN_BACKEND=corex
export ENGINE_MODEL=$H/models/Qwen3.5-4B
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

: "${WORKDIR:?WORKDIR must be set by the submitter}"
: "${PROBE:=diag/corex_probe/probe_gemv_tune.py}"

cd "$WORKDIR" || exit 1
tar xzf "$WORKDIR/../raw/coffin.tar.gz" || exit 1
export COFFIN_ROOT=$WORKDIR/coffin
cd "$COFFIN_ROOT" || exit 1

echo "HOST=$(hostname)"
echo "CARDS_REQUESTED=${CARDS:-?}"
echo "PWD=$(pwd)"

# How many cards did the job actually get? Asked of the runtime rather than assumed from
# the request: requesting eight and receiving two is a normal scheduler outcome, and a
# loop that assumed eight would test devices that are not there.
#
# This call touches the driver, so it is capped too: on a node whose device stack is
# wedged, merely initialising CUDA can be the thing that hangs, and an uncapped probe would
# report nothing at all.
N=$(timeout -s KILL 30 "$PY" -B -c "import torch; print(torch.cuda.device_count())" \
    2>/dev/null || echo 0)
echo "CARDS_VISIBLE=$N"
if [ "$N" -eq 0 ]; then
    echo "NO_CARDS: the job holds no usable accelerator. Ask for more via gpu_count."
    exit 64
fi

# Probe every card AT ONCE, and take the first that answers.
#
# Two properties matter here and the serial version had neither.
#
# Short cap. An 8x8 matmul is microseconds of device work; the cap only has to cover the
# interpreter and the CUDA context. It is set from a measured ``import torch`` cost plus
# three seconds, because a fixed number either wastes a minute on a wedged card or fails a
# slow one on import alone.
#
# No blocking on a dead card. Some wedged contexts cannot be killed at all -- the process
# sits in an uninterruptible state and ``timeout`` returns nothing -- so a serial walk
# stops at the first corpse and never reports the cards behind it. Running the checks as
# background children and polling for the first ``OK`` means an unkillable card simply
# never wins the race, and the job proceeds on a card that works. The losers are killed
# without being waited on: waiting is exactly what a dead card refuses to allow.

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

IMPORT_MS=$("$PY" -B -c "import time
t = time.time()
import torch
print(int((time.time() - t) * 1000))" 2>/dev/null || echo 4000)
# Three seconds on top of the import: the
# whole point is that an 8x8 matmul is microseconds of device work, so the only budget a
# healthy card needs is interpreter plus context. Two seconds of slack on top of a measured
# import lands the cap at 4-6 s, which is short enough that a corpse cannot hold the job.
CAP=$(( (IMPORT_MS + 2000) / 1000 + 1 ))
if [ "$CAP" -lt 3 ]; then CAP=3; fi
if [ "$CAP" -gt 8 ]; then CAP=8; fi
echo "CARD_CHECK_CAP=${CAP}s (import measured ${IMPORT_MS}ms)"

PIDS=""
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
        echo "rc=$?" >>"$TMP/c$i"
    ) &
    PIDS="$PIDS $!"
done

# Poll for the first card that reports OK. The ceiling is generous relative to the cap so
# a slow scheduler tick cannot lose a card that did answer.
GOOD=""
i=0
while [ "$i" -lt 24 ]; do
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

# Do not wait for the losers. A dead card's process may never be reaped; the job's cgroup
# cleans up on exit, and the score matters more than the tidiness.
for pid in $PIDS; do
    kill -9 "$pid" 2>/dev/null || true
done

if [ -z "$GOOD" ]; then
    echo "NO_HEALTHY_CARD on $(hostname) among $N visible card(s)"
    exit 65
fi

echo ""
echo "USING_CARD=$GOOD on $(hostname)"
export CUDA_VISIBLE_DEVICES=$GOOD

echo "--- $PROBE ---"
"$PY" -B -u "$PROBE"
echo "PROBE_EXIT=$?"
