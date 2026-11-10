#!/usr/bin/env bash
set -u

# 层切分管线（PP）把 Occamy-1.0 起成常驻服务：每卡只装模型的一段层，token 在环上走一圈。
#
# 为什么不用张量并行（TP）：这张卡的跨卡归约实测 1 ms 量级（跨组写 2.4 GB/s、NCCL 不可用、
# 只能主机中转），而 MoE 有 40 个需要归约的层——TP4 下每 token 80 次归约，实测把 decode
# 从单卡的 82 ms/步拖到 170 ms/步（12.2 → 5.9 tok/s），卡越多越慢。
# PP 每 token 只有三次 4 KB 的激活传递加一次 logits 回传（约 0.5 MB），归约一次都没有。
#
# 用法：
#   OCCAMY_WORLD=2 bash tools/launch_occamy_pp.sh     # 先验两段
#   OCCAMY_WORLD=4 bash tools/launch_occamy_pp.sh     # 四段

H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
ENGINE_ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${OCCAMY_MODEL:-$H/models/occamy-1.0-W8A8-kn}
WORLD=${OCCAMY_WORLD:-4}
SOCK=${OCCAMY_SOCK:-/tmp/occamy_pp.sock}
LOGDIR=${OCCAMY_LOGS:-$H/coffin-work/base/occamy_pp_logs}
PORT=${OCCAMY_PORT:-29611}
MAX_SEQ=${OCCAMY_SEQ:-4096}
BLOCKS=${OCCAMY_BLOCKS:-256}
# 前缀缓存默认关：流水并行下它要搬线性注意力状态快照，而那套假设「一个 rank 持有整
# 个模型」。先让「能不能吐字」成立，再单独验缓存的命中与正确性。
if [ "${OCCAMY_PREFIX:-0}" = "1" ]; then PREFIX_FLAG=""; else PREFIX_FLAG="--no-prefix-cache"; fi

export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
export COFFIN_BACKEND=corex
export COFFIN_PARALLEL=pp
# 段间 UDP 链路的端口基址。每次跑给不同的值：上一个进程刚被杀时它的端口可能还在
# TIME_WAIT 里，绑不上就会让整次启动失败。
export COFFIN_PP_PORT=$((PORT + 2))
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$ENGINE_ROOT${PYTHONPATH:+:$PYTHONPATH}

echo "HOST=$(hostname)  ENGINE=$ENGINE_ROOT"
echo "MODEL=$MODEL  WORLD=$WORLD  MAX_SEQ=$MAX_SEQ"
if [ ! -d "$MODEL" ]; then echo MODEL_MISSING; exit 66; fi

echo "--- warm the corex extension (single process) ---"
"$PY" -B -u -c "import sys; sys.path.insert(0, '$ENGINE_ROOT'); import backends; print(backends.status())" \
    || { echo BACKEND_WARMUP_FAILED; exit 65; }

AVAILABLE=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
CARDS=()
for DEV in $(echo "$AVAILABLE" | tr ',' ' '); do
    [ ${#CARDS[@]} -ge "$WORLD" ] && break
    if CUDA_VISIBLE_DEVICES=$DEV timeout --foreground -k 2 5 "$PY" -B -u -c \
        "import torch; a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda'); b=a@a; torch.cuda.synchronize(); raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then CARDS+=("$DEV"); else echo "DEAD_CARD=$DEV"; fi
done
[ ${#CARDS[@]} -lt "$WORLD" ] && { echo NOT_ENOUGH_CARDS; exit 64; }
echo "PICKED=${CARDS[*]}"

mkdir -p "$LOGDIR"
rm -f "$SOCK"
PIDS=()
for ((R = 0; R < WORLD; R++)); do
    CUDA_VISIBLE_DEVICES=${CARDS[$R]} RANK=$R WORLD_SIZE=$WORLD \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT \
        "$PY" -B -u -m entrypoints.cli.runtime --serve \
        --model "$MODEL" --sock "$SOCK" --max-seq "$MAX_SEQ" \
        --num-blocks "$BLOCKS" $PREFIX_FLAG \
        --no-bench --no-vision --prefill-quantum 2048 --max-num-seqs 1 \
        > "$LOGDIR/rank$R.log" 2>&1 &
    PIDS+=($!)
done
trap 'for pid in ${PIDS[@]:-}; do kill $pid 2>/dev/null; done' EXIT

echo "--- waiting for socket ---"
for _ in $(seq 1 600); do [ -S "$SOCK" ] && break; sleep 1; done
if [ ! -S "$SOCK" ]; then
    echo SOCKET_TIMEOUT
    for ((R = 0; R < WORLD; R++)); do echo "=== rank$R ==="; tail -12 "$LOGDIR/rank$R.log"; done
    exit 71
fi
grep -h "^\[serve\]" "$LOGDIR"/rank*.log || true

# 每个客户端调用都套 timeout：引擎一旦卡在集合通信上，没有超时的调用会把整个队列
# 占满 1800 s 的上限（804 那次就是这样把后面全部饿死的）。
echo "=== ping ==="
timeout 60 "$PY" -B -u -m entrypoints.cli.runtime --sock "$SOCK" ping; echo "PING=$?"
echo "=== generate: 原始续写 ==="
timeout 240 "$PY" -B -u -m entrypoints.cli.runtime --sock "$SOCK" generate \
    --prompt "The capital of France is" -n 24
echo "G1=$?"
echo "=== chat --think: 数学 ==="
timeout 240 "$PY" -B -u -m entrypoints.cli.runtime --sock "$SOCK" chat --think \
    --user "What is 2+2? Reply with the number only." -n 96
echo "G2=$?"
echo "=== chat --think: 中文 ==="
timeout 240 "$PY" -B -u -m entrypoints.cli.runtime --sock "$SOCK" chat --think \
    --user "用一句话介绍你自己" -n 96
echo "G3=$?"

timeout 60 echo "=== generate: 长 prompt（约 1800 token，量分块预填充）==="
LONG=$("$PY" -c "print('The quick brown fox jumps over the lazy dog. ' * 200)")
timeout 600 "$PY" -B -u -m entrypoints.cli.runtime --sock "$SOCK" generate \
    --prompt "$LONG" -n 8
echo "G4=$?"

timeout 60 "$PY" -B -u -m entrypoints.cli.runtime --sock "$SOCK" stop || true
# 不要 ``wait``：流水一旦卡在环上的 recv，四个 rank 都不会自己退出，而 ``wait`` 会让
# 这个脚本一直占着队列直到 1800 s 上限（833 就是这样把后面饿死的）。直接 kill。
for pid in ${PIDS[@]:-}; do kill -9 "$pid" 2>/dev/null; done
sleep 1
echo "PP_DONE"
exit 0
