# Select a card that actually computes, and make it the only visible device.
#
# Sourced by the launchers under ``diag/corex_probe/`` and ``backends/corex/diag/``.
#
# This partition loses cards to a wedged context: the first kernel never returns, so the job
# makes no output on either stream and looks exactly like a probe bug. Node name is not a
# usable signal -- one dead card and three healthy ones sit in the same chassis, and which
# one is dead changes between sessions. So the test runs a kernel.
#
# Each candidate gets its own process under ``timeout``. Once the driver spins, the process is
# uninterruptible and an in-process check cannot return control, so a single-process sweep
# would burn its whole wall clock instead of moving to the next card.
#
# The bound is short because the two outcomes are far apart: a healthy card finishes an 8x8
# bf16 matmul in milliseconds, a wedged one never finishes. There is nothing in between that
# a longer wait would discover. The one concession is the retry: if every candidate fails at
# the short bound the loop repeats once with a generous one, so a slow interpreter start-up
# cannot be reported as "every card is dead".

_PICK_PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
_PICK_TIMEOUT=5
_PICK_RETRY=60

# One bf16 matmul: the cheapest kernel that still proves the context schedules work and
# returns finite numbers.
_PICK_SNIPPET="import torch; a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda'); b=a@a; torch.cuda.synchronize(); raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)"

pick_card() {
    _visible=${CUDA_VISIBLE_DEVICES:-0}
    for _bound in $_PICK_TIMEOUT $_PICK_RETRY; do
        if [ "$_bound" = "$_PICK_RETRY" ]; then
            echo "RETRY_SLOW_STARTUP bound=${_bound}s"
        fi
        for _dev in $(echo "$_visible" | tr ',' ' '); do
            _t0=$(date +%s%N)
            if CUDA_VISIBLE_DEVICES=$_dev timeout --foreground -k 2 "$_bound" \
                "$_PICK_PY" -B -u -c "$_PICK_SNIPPET" >/dev/null 2>&1; then
                _ms=$(( ($(date +%s%N) - _t0) / 1000000 ))
                export CUDA_VISIBLE_DEVICES=$_dev
                echo "PICKED_CARD=$_dev visible=[$_visible] probe=${_ms}ms host=$(hostname)"
                return 0
            fi
            _ms=$(( ($(date +%s%N) - _t0) / 1000000 ))
            echo "DEAD_CARD=$_dev bound=${_bound}s waited=${_ms}ms"
        done
    done
    echo "NO_HEALTHY_CARD visible=[$_visible] host=$(hostname)"
    return 64
}
