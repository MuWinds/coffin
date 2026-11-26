set -u

# Occamy 1.0 的配置/索引探测：只读 config.json 与 safetensors 索引，不碰权重。
#
# 注意：上传的脚本不能含 `#!/bin/bash`（网关会把文件过滤成 0 字节并返回成功）。

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${COFFIN_ROOT:-$HERE/coffin}
[ -d "$ROOT" ] || ROOT=$HOME/coffin-work/base/coffin
PY=${PY:-/data/software/Anaconda3/2024.10-1/envs/py310/bin/python}
MODEL=${MODEL:-$HOME/models/occamy-1.0}

cd "$ROOT" || { echo "MISSING_ROOT=$ROOT"; exit 66; }
export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export PYTHONPATH=$ROOT

echo "[env] host=$(hostname) model=$MODEL"
"$PY" -B -u tools/probe_occamy_cfg.py --checkpoint "$MODEL"
RC=$?
echo "[cfg] rc=$RC"
exit "$RC"
