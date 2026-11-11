set -u

# 专家并行（EP）把 Occamy-1.0 起成常驻服务：每卡装全部 40 层，只把专家切 4 份。
#
# 与 PP4 的差别只在拓扑：PP 每步 39.4 ms 是四段严格串行（每段 10 层，段间 3 次 UDP
# 接力），EP 每卡自己把 40 层走一遍，但每层专家只算 1/4，代价是每层一次同机归约
# （8 行 x 2560 bf16，共享内存 barrier 33 us；同样的事走 gloo 要 291-420 us）。
# 按实测的每层 876 us 拆：专家 613 us + dense 263 us，EP 下变成 263 + 153 + 33 us。
#
# 专家栈按专家号**连续**切（rank r 拿 [r*E/4, (r+1)*E/4)），所以加载器可以对
# safetensors 用 get_slice 只读自己那一段——整栈每层 805 MB，40 层 32 GB，先读进来
# 再切会把一张 31.7 GiB 的卡撑爆。
#
# 用法：
#   OCCAMY_WORLD=2 bash tools/launch_occamy_ep.sh     # 先验两卡
#   OCCAMY_WORLD=4 bash tools/launch_occamy_ep.sh

H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
ENGINE_ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${OCCAMY_MODEL:-$H/models/occamy-1.0-W8A8-kn}
WORLD=${OCCAMY_WORLD:-4}
SOCK=${OCCAMY_SOCK:-/tmp/occamy_ep.sock}
LOGDIR=${OCCAMY_LOGS:-$H/coffin-work/base/occamy_ep_logs}
PORT=${OCCAMY_PORT:-30611}
MAX_SEQ=${OCCAMY_SEQ:-4096}
BLOCKS=${OCCAMY_BLOCKS:-256}
MAX_SEQS=${OCCAMY_SEQS:-1}
# 每个 rank 的共享区容量。它要盖住一次预填充分块的 [tokens, hidden] fp32：
# prefill-quantum 2048 x 2560 x 4 B = 20 MiB，双缓冲 40 MiB，64 MiB 留一倍余量。
SLOT_BYTES=${OCCAMY_EP_SLOT_BYTES:-67108864}
SHM=${OCCAMY_EP_SHM:-/dev/shm/coffin_ep_$PORT.bin}
# 前缀缓存默认关：它要搬线性注意力状态快照，而那套假设「一个 rank 持有整个模型」。
if [ "${OCCAMY_PREFIX:-0}" = "1" ]; then PREFIX_FLAG=""; else PREFIX_FLAG="--no-prefix-cache"; fi

export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
export COFFIN_BACKEND=corex
export COFFIN_PARALLEL=ep
export COFFIN_EP_SHM=$SHM
export COFFIN_EP_SLOT_BYTES=$SLOT_BYTES
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$ENGINE_ROOT${PYTHONPATH:+:$PYTHONPATH}

echo "HOST=$(hostname)  ENGINE=$ENGINE_ROOT"
echo "MODEL=$MODEL  WORLD=$WORLD  MAX_SEQ=$MAX_SEQ  EP_SHM=$SHM slot=$SLOT_BYTES"
if [ ! -d "$MODEL" ]; then echo MODEL_MISSING; exit 66; fi

# 共享区必须在任何进程启动前就建好：ep.EpShmReduce 只负责映射，自己不 ftruncate。
# 在这里截断会让四个进程互相拆台（mmap length is greater than file size）。
# 长度 = 头部 (world+1)*8 字节 + world*一个 rank 的载荷区，头部那 40 字节不额外算够
# 的话 EpShmReduce 会以「共享板太短」拒绝启动，所以多给一整块。
rm -f "$SHM"
dd if=/dev/zero of="$SHM" bs=4096 count=$(( (WORLD * SLOT_BYTES) / 4096 + 1 )) \
    status=none || { echo SHM_CREATE_FAILED; exit 67; }

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
    # 各 rank 错峰启动：四个进程同时导入扩展会抢同一个构建锁，其余 rank 干等
    # 就撞上 30 s 的接收超时（PP 那边踩过同一个坑）。
    [ "$R" -gt 0 ] && sleep 4
    CUDA_VISIBLE_DEVICES=${CARDS[$R]} RANK=$R WORLD_SIZE=$WORLD \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT \
        "$PY" -B -u -m entrypoints.cli.runtime --serve \
        --model "$MODEL" --sock "$SOCK" --max-seq "$MAX_SEQ" \
        --num-blocks "$BLOCKS" $PREFIX_FLAG \
        --no-bench --no-vision --prefill-quantum 2048 --max-num-seqs "$MAX_SEQS" \
        > "$LOGDIR/rank$R.log" 2>&1 &
    PIDS+=($!)
done
trap 'for pid in ${PIDS[@]:-}; do kill $pid 2>/dev/null; done' EXIT

echo "--- waiting for socket ---"
for _ in $(seq 1 900); do [ -S "$SOCK" ] && break; sleep 1; done
if [ ! -S "$SOCK" ]; then
    echo SOCKET_TIMEOUT
    for ((R = 0; R < WORLD; R++)); do echo "=== rank$R ==="; tail -14 "$LOGDIR/rank$R.log"; done
    exit 71
fi
grep -h "^\[serve\]" "$LOGDIR"/rank*.log || true

# 每个客户端调用都套 timeout：引擎一旦卡在集合通信上，没有超时的调用会把整个队列
# 占满 1800 s 上限。
echo "=== ping ==="
timeout 60 "$PY" -B -u -m entrypoints.cli.runtime --sock "$SOCK" ping; echo "PING=$?"
echo "=== generate: 原始续写（贪心，用来和 PP4 对拍） ==="
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

timeout 60 echo "=== generate: 长 prompt（约 1800 token，量分块预填充） ==="
LONG=$("$PY" -c "print('The quick brown fox jumps over the lazy dog. ' * 200)")
timeout 900 "$PY" -B -u -m entrypoints.cli.runtime --sock "$SOCK" generate \
    --prompt "$LONG" -n 8
echo "G4=$?"

timeout 60 "$PY" -B -u -m entrypoints.cli.runtime --sock "$SOCK" stop || true
# 不要 ``wait``：barrier 卡住时四个 rank 都不会自己退出，``wait`` 会让脚本占着队列
# 直到 1800 s 上限。直接 kill。
for pid in ${PIDS[@]:-}; do kill -9 "$pid" 2>/dev/null; done
sleep 1
rm -f "$SHM"
echo "EP_DONE"
exit 0
