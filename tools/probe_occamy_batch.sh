set -u

H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
JOB_ROOT=$(cd "$(dirname "$0")" && pwd)
ENGINE_ROOT=$JOB_ROOT/coffin
MODEL=$H/models/occamy-1.0-W8A8-kn
WORLD=4
AVAILABLE=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
CARDS=()

export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
export COFFIN_BACKEND=corex
export COFFIN_PARALLEL=pp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$ENGINE_ROOT${PYTHONPATH:+:$PYTHONPATH}
# 单流基线也固定走分页 KV，否则它会与并发路径比较两种不同的缓存布局。
export COFFIN_NO_DENSE_PREFILL=1

if [ ! -d "$MODEL" ]; then
    echo "MODEL_MISSING=$MODEL"
    exit 66
fi

for DEV in $(echo "$AVAILABLE" | tr ',' ' '); do
    if [ ${#CARDS[@]} -ge "$WORLD" ]; then
        break
    fi
    if CUDA_VISIBLE_DEVICES=$DEV timeout --foreground -k 2 5 "$PY" -B -u -c \
        "import torch; a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda'); b=a@a; torch.cuda.synchronize(); raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then
        CARDS+=("$DEV")
    else
        echo "DEAD_CARD=$DEV"
    fi
done
if [ ${#CARDS[@]} -lt "$WORLD" ]; then
    echo "NOT_ENOUGH_HEALTHY_CARDS got=${#CARDS[@]} need=$WORLD"
    exit 64
fi
echo "PICKED=${CARDS[*]}"

"$PY" -B -u -c "import sys; sys.path.insert(0, '$ENGINE_ROOT'); import backends; print(backends.status())" \
    || { echo BACKEND_WARMUP_FAILED; exit 65; }

run_level() {
    LEVEL=$1
    PORT=$((29700 + LEVEL))
    SOCK=/tmp/occamy_batch_probe_$LEVEL.sock
    LOGDIR=$JOB_ROOT/logs_$LEVEL
    mkdir -p "$LOGDIR"
    rm -f "$SOCK"
    PIDS=()
    for ((R = 0; R < WORLD; R++)); do
        CUDA_VISIBLE_DEVICES=${CARDS[$R]} RANK=$R WORLD_SIZE=$WORLD \
        MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT COFFIN_PP_PORT=$((PORT + 2)) \
            "$PY" -B -u -m entrypoints.cli.runtime --serve \
            --model "$MODEL" --sock "$SOCK" --max-seq 4096 \
            --num-blocks 1024 --max-num-seqs "$LEVEL" --no-prefix-cache \
            --no-bench --no-vision --no-graph --prefill-quantum 2048 \
            >"$LOGDIR/rank$R.log" 2>&1 &
        PIDS+=($!)
    done
    READY=0
    for _ in $(seq 1 600); do
        if [ -S "$SOCK" ]; then READY=1; break; fi
        if ! kill -0 "${PIDS[0]}" 2>/dev/null; then break; fi
        sleep 1
    done
    if [ "$READY" != 1 ]; then
        echo "SERVICE_START_FAILED level=$LEVEL"
        for ((R = 0; R < WORLD; R++)); do
            echo "=== level=$LEVEL rank=$R ==="
            tail -30 "$LOGDIR/rank$R.log"
        done
        for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
        return 1
    fi
    grep -h "^\[serve\]" "$LOGDIR"/rank*.log || true
    set +e
    "$PY" -B -u "$JOB_ROOT/probe_occamy_batch.py" --sock "$SOCK" --level "$LEVEL"
    RESULT=$?
    set -e
    timeout 60 "$PY" -B -u -m entrypoints.cli.runtime --sock "$SOCK" stop >/dev/null 2>&1 || true
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
    return "$RESULT"
}

run_level 4 || exit $?
run_level 8 || exit $?
