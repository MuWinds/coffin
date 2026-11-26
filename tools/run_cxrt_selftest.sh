set -u

# CoreX 无 torch 通道的卡上自检：不发作业命令，只用 qjob 的 --script 放进来。
#
# 做三件事：挑一张能算的卡（会话级事实，不能按节点名判断）、把 corex 工具链与运行时
# 缓存指到本机路径、跑 `python -m cxrt.selftest`——它会用厂商 clang 现编
# norm_abi.cu / solve_abi.cu 并对拍数值。
#
# 注意：上传的脚本不能含 `#!/bin/bash`（网关会把文件过滤成 0 字节并返回成功），
# 因此这里以 `set -u` 起头。

# 树从 qjob --tree 打进私有作业目录，所以默认取 run.sh 旁边的 coffin/；
# 退回到常驻共享检出，方便手工对同一棵树复跑。
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${COFFIN_ROOT:-$HERE/coffin}
[ -d "$ROOT" ] || ROOT=$HOME/coffin-work/base/coffin
PY=${PY:-/data/software/Anaconda3/2024.10-1/envs/py310/bin/python}

cd "$ROOT" || { echo "MISSING_ROOT=$ROOT"; exit 66; }
export PYTHONPATH=$ROOT
export COREX_ROOT=${COREX_ROOT:-/usr/local/corex}
export COFFIN_RT_DIR=${COFFIN_RT_DIR:-$HOME/.coffin_corex_rt}
export COFFIN_RT_CXX=${COFFIN_RT_CXX:-$COREX_ROOT/bin/clang++}

echo "[env] host=$(hostname) root=$ROOT py=$PY"
"$PY" -c "import sys; print('[env] python', sys.version.split()[0])"

AVAILABLE=${CUDA_VISIBLE_DEVICES:-0}
PICKED=
for DEV in $(echo "$AVAILABLE" | tr ',' ' '); do
    if CUDA_VISIBLE_DEVICES=$DEV timeout --foreground -k 5 5 "$PY" -B -u -c \
        "import torch; a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda'); b=a@a; torch.cuda.synchronize(); raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then PICKED=$DEV; break; fi
    echo "[card] dead=$DEV"
done
if [ -z "$PICKED" ]; then echo "NO_HEALTHY_CARD"; exit 64; fi
export CUDA_VISIBLE_DEVICES=$PICKED
echo "[card] picked=$PICKED"

# 位模式诊断只在排查时打开（COFFIN_DIAG=1），平时留着它只会把日志稀释。
if [ "${COFFIN_DIAG:-0}" = "1" ]; then
    echo "[diag] norm bit patterns"
    "$PY" -B -u -m cxrt.diag_norm --rows 4 --cols 2560 --dtype bf16 2>&1 | tail -40
fi

"$PY" -B -u -m cxrt.selftest --dtype bf16 --repeat 0
RC=$?
echo "[selftest] rc=$RC"
exit $RC
