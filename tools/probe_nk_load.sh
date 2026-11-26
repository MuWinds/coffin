set -u

# 在真 torch 上过一遍 nk 目录的另一条读取器：quantized_load → QWeight → pack.to_kn。
# 稠密路（cxrt.dense_weights）已经在无 torch 进程里跑通了，但那是模型自己的稠密入口；
# 后端实际吃的是 QWeight，这条链有没有把 nk 读对，只有这里能回答。
#
# 注意：上传的脚本不能含 `#!/bin/bash`（网关会把文件过滤成 0 字节并返回成功）。

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${COFFIN_ROOT:-$HERE/coffin}
[ -d "$ROOT" ] || ROOT=$HOME/coffin-work/base/coffin
PY=${PY:-/data/software/Anaconda3/2024.10-1/envs/py310/bin/python}
MODEL=${MODEL:-$HOME/models/Qwen3.5-4B-int8-nk}
SOURCE=${SOURCE:-$HOME/models/Qwen3.5-4B}

cd "$ROOT" || { echo "MISSING_ROOT=$ROOT"; exit 66; }
# ninja 与编译器都装在 py310 那个 env 里，PATH 不带上它 torch 的 cpp_extension 直接失败。
export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export PYTHONPATH=$ROOT
export COFFIN_BACKEND=${COFFIN_BACKEND:-corex}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# corex 的 pybind 扩展要一个按部署隔离的编译缓存，否则注册表直接判定不可用。
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$HOME/torch_ext_coffin_corex}

echo "[env] host=$(hostname) backend=$COFFIN_BACKEND model=$MODEL"

"$PY" -B -u tools/probe_nk_load.py --model "$MODEL" --source "$SOURCE" --sample 4
RC=$?
echo "[probe] rc=$RC"
exit "$RC"
