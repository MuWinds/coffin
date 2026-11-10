#!/usr/bin/env bash
set -u

# 四卡张量并行把 Occamy-1.0（W4A8 预量化目录）起成常驻服务，并跑一遍冒烟请求。
#
# 四进程各只看到一张卡：这个引擎的张量并行按"一个进程一张健康 GPU"设计
# （models/qwen35/parallel.py），进程组是 CPU 上的 Gloo，跨 rank 只有 fp32 求和与
# 一次广播。所以这里不能用 torchrun 的 device 分配，必须每个 rank 自己设
# CUDA_VISIBLE_DEVICES 并把 RANK/WORLD_SIZE 写进环境。
#
# 用法：
#   bash tools/launch_occamy_tp4.sh            # 起服务 + 冒烟 + 停服务
#   OCCAMY_KEEP=1 bash tools/launch_occamy_tp4.sh   # 只起服务，不走冒烟也不停

H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
ENGINE_ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${OCCAMY_MODEL:-$H/models/occamy-1.0-W4A8-g128}
SOCK=${OCCAMY_SOCK:-/tmp/occamy_tp4.sock}
WORLD=${OCCAMY_WORLD:-4}
MAX_SEQ=${OCCAMY_SEQ:-16384}
LOGDIR=${OCCAMY_LOGS:-$ENGINE_ROOT/../occamy_tp4_logs}
PORT=${OCCAMY_PORT:-29531}

export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
export COFFIN_BACKEND=corex
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$ENGINE_ROOT${PYTHONPATH:+:$PYTHONPATH}

echo "HOST=$(hostname)  ENGINE=$ENGINE_ROOT"
echo "MODEL=$MODEL  WORLD=$WORLD  MAX_SEQ=$MAX_SEQ"
if [ ! -d "$MODEL" ]; then
    echo "MODEL_MISSING=$MODEL"
    exit 66
fi

# 卡的健康是会话级事实：同一机箱里会有一张卡挂死，每次坏的是哪张都不一样，所以逐卡
# 用 8x8 matmul 实测，只取通过的那几张。timeout 只写 5 秒——挂死的表现是永不返回。
AVAILABLE=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
CARDS=()
echo "--- card selection (need $WORLD) ---"
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

# 四个 rank 同时 import 后端会在共用的 TORCH_EXTENSIONS_DIR 上抢同一个 build lock：
# 谁先把 ``coffin_corex_ops_v20/lock`` 建出来，其余三个有时会撞上「目录刚被删」的窗口，
# 报 FileNotFoundError 并在起服务前就退出（实测 790 的 rank2 就是这样死的，而其余三个
# 静默等在 Gloo 里）。所以先用**一个**进程把扩展编译/校验出来，再起四个 rank。
echo "--- warm the corex extension (single process) ---"
"$PY" -B -u -c "import sys; sys.path.insert(0, '$ENGINE_ROOT'); import backends; print(backends.status())" \
    || { echo "BACKEND_WARMUP_FAILED"; exit 65; }

mkdir -p "$LOGDIR"
rm -f "$SOCK"
PIDS=()
for ((R = 0; R < WORLD; R++)); do
    CUDA_VISIBLE_DEVICES=${CARDS[$R]} RANK=$R WORLD_SIZE=$WORLD \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT \
        "$PY" -B -u -m entrypoints.cli.runtime --serve \
        --model "$MODEL" --sock "$SOCK" --max-seq "$MAX_SEQ" \
        --no-bench --no-vision --prefill-quantum 2048 --max-num-seqs 1 \
        > "$LOGDIR/rank$R.log" 2>&1 &
    PIDS+=($!)
    echo "STARTED rank=$R card=${CARDS[$R]} pid=$!"
done

cleanup() {
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null
    done
}
trap cleanup EXIT

echo "--- waiting for socket (up to 600 s) ---"
for _ in $(seq 1 600); do
    if [ -S "$SOCK" ]; then
        break
    fi
    if ! kill -0 "${PIDS[0]}" 2>/dev/null; then
        echo "RANK0_DIED"
        tail -40 "$LOGDIR/rank0.log"
        exit 70
    fi
    sleep 1
done
if [ ! -S "$SOCK" ]; then
    echo "SOCKET_TIMEOUT"
    for ((R = 0; R < WORLD; R++)); do
        echo "=== rank$R tail ==="
        tail -20 "$LOGDIR/rank$R.log"
    done
    exit 71
fi
echo "SOCKET_READY=$SOCK"

grep -h "^\[serve\]" "$LOGDIR"/rank*.log || true

echo "--- ping ---"
"$PY" -B -u -m entrypoints.cli.runtime --sock "$SOCK" ping || echo "PING_FAILED=$?"

echo "--- generate ---"
"$PY" -B -u -m entrypoints.cli.runtime --sock "$SOCK" generate \
    --prompt "The capital of France is" -n 24 || echo "GENERATE_FAILED=$?"

echo "--- chat ---"
"$PY" -B -u -m entrypoints.cli.runtime --sock "$SOCK" chat \
    --user "用一句话介绍你自己" -n 64 || echo "CHAT_FAILED=$?"

if [ "${OCCAMY_KEEP:-0}" = "1" ]; then
    echo "OCCAMY_KEEP=1 -> 保持服务运行，PIDS=${PIDS[*]}"
    trap - EXIT
    wait
    exit 0
fi

echo "--- stop ---"
"$PY" -B -u -m entrypoints.cli.runtime --sock "$SOCK" stop || true
wait
echo "TP4_SERVE_DONE"
