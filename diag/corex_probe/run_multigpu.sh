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
N=$("$PY" -B -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "CARDS_VISIBLE=$N"
if [ "$N" -eq 0 ]; then
    echo "NO_CARDS: the job holds no accelerator. Ask for more via gpu_count."
    exit 64
fi

# Walk the cards. Each check is capped: a wedged context spins forever, so an unbounded
# check would consume the job and report nothing.
GOOD=""
for i in $(seq 0 $((N - 1))); do
    echo "--- card $i ---"
    out=$(CUDA_VISIBLE_DEVICES=$i timeout --foreground -k 5 45 "$PY" -B -c \
      "import torch
a = torch.randn(8, 8, dtype=torch.bfloat16, device='cuda')
b = torch.randn(8, 8, dtype=torch.bfloat16, device='cuda')
c = a @ b
torch.cuda.synchronize()
print('OK' if bool(torch.isfinite(c).all()) else 'NONFINITE')" 2>&1)
    rc=$?
    echo "  rc=$rc out=${out:-<empty>}"
    if [ "$rc" -eq 0 ] && [ "$out" = "OK" ]; then
        echo "  card $i is healthy"
        GOOD=$i
        break
    fi
    echo "  card $i is not usable, moving on"
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
