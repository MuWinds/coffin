set -u

# 卡上探针：确认 cxrt.store / cxrt.safetensors 能读**真实** checkpoint。
#
# 与自检脚本的区别：这条不需要卡，也不设工具链；放在作业里跑只是为了用集群的 py310
# 与 $HOME/models 下的真实权重（主机上没有 torch、没有这些权重）。
#
# 注意：上传的脚本不能含 `#!/bin/bash`（网关会把文件过滤成 0 字节并返回成功）。

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${COFFIN_ROOT:-$HERE/coffin}
[ -d "$ROOT" ] || ROOT=$HOME/coffin-work/base/coffin
PY=${PY:-/data/software/Anaconda3/2024.10-1/envs/py310/bin/python}

cd "$ROOT" || { echo "MISSING_ROOT=$ROOT"; exit 66; }
export PYTHONPATH=$ROOT

echo "[env] host=$(hostname) root=$ROOT"

echo "[probe] qwen35 4B int8-nk"
"$PY" -B -u -m cxrt.probe_store "$HOME/models/Qwen3.5-4B-int8-nk" --limit 12

echo "[probe] occamy 1.0 W8A8"
"$PY" -B -u -m cxrt.probe_store "$HOME/models/occamy-1.0-W8A8" --limit 12

echo "[probe] done"
exit 0
