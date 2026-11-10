#!/usr/bin/env bash
set -u

# Occamy-1.0（Qwen3.5-MoE 架构，40 层 / 256 专家 / top-8）离线量化为分组 W4A8。
#
# 与 launch_qwen38_w4a8.sh 的区别只有三处：源模型换成 occamy、量化产物另起目录、
# 以及必须带上 --include-vision——checkpoint 声明了 vision_config，weights.py 在
# 找不到任何 model.visual.* 张量时会直接报错，而不是起一个纯文本的引擎。

H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
ENGINE_ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=$H/models/occamy-1.0
OUT=$H/models/occamy-1.0-W4A8-g128

export PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
export COFFIN_BACKEND=corex
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$ENGINE_ROOT${PYTHONPATH:+:$PYTHONPATH}

echo "HOST=$(hostname)  ENGINE=$ENGINE_ROOT"
echo "SOURCE=$MODEL"
echo "OUTPUT=$OUT"

if [ ! -d "$MODEL" ]; then
    echo "SOURCE_MODEL_MISSING"
    exit 66
fi
if [ -e "$OUT" ]; then
    echo "OUTPUT_ALREADY_EXISTS; refusing to overwrite"
    exit 73
fi

AVAILABLE=${CUDA_VISIBLE_DEVICES:-0}
PICKED=
echo "--- card selection ---"
for DEV in $(echo "$AVAILABLE" | tr ',' ' '); do
    if CUDA_VISIBLE_DEVICES=$DEV timeout --foreground -k 2 5 "$PY" -B -u -c \
        "import torch; a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda'); b=a@a; torch.cuda.synchronize(); raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then
        PICKED=$DEV
        break
    fi
    echo "DEAD_CARD=$DEV"
done
if [ -z "$PICKED" ]; then
    echo "NO_HEALTHY_CARD visible=[$AVAILABLE]"
    exit 64
fi
export CUDA_VISIBLE_DEVICES=$PICKED
echo "PICKED_CARD=$PICKED visible=[$AVAILABLE]"

echo "--- W4A8 disk conversion ---"
"$PY" -B -u "$ENGINE_ROOT/tools/quant_disk_w4a8.py" \
    --checkpoint "$MODEL" --out "$OUT" --weight-dtype bf16 \
    --device auto --group-size 128 --include-vision || exit $?

echo "--- artifact verification ---"
"$PY" -B - "$MODEL" "$OUT" <<'PY'
import json
import sys
from pathlib import Path

from safetensors import safe_open

source, output = map(Path, sys.argv[1:])
with (source / "model.safetensors.index.json").open(encoding="utf-8") as f:
    source_map = json.load(f)["weight_map"]
with (output / "quant_index.json").open(encoding="utf-8") as f:
    report = json.load(f)

prefixes = ("model.language_model.", "mtp.", "lm_head.", "model.visual.")
expected = {key for key in source_map if key.startswith(prefixes)}
actual = set(report["tensors"])
missing = expected - actual
if missing:
    raise SystemExit(f"MISSING_SOURCE_TENSORS={len(missing)} first={sorted(missing)[0]}")
if report.get("format") != "coffin-w4a8-kn-v1" or report.get("kinds", {}).get("q4", 0) == 0:
    raise SystemExit("INVALID_W4_INDEX")

# 路由器与共享专家门必须留在稠密档：它们是 top-k 的选择依据与共享分支的幅度，
# 量化噪声在这里是「选错专家」而不是「精度略差」。
dense_kept = [key for key in actual
              if key.endswith(("mlp.gate.weight", "mlp.shared_expert_gate.weight"))
              and report["tensors"][key]["kind"] == "dense"]
routers = [key for key in actual if key.endswith("mlp.gate.weight")]
if len(dense_kept) < len(routers):
    raise SystemExit(f"ROUTER_NOT_DENSE count={len(routers) - len(dense_kept)}")

json.dump({
    "result": "W4A8_DISK_ARTIFACT_OK",
    "format": report["format"],
    "source_tensors": len(expected),
    "indexed_tensors": len(actual),
    "kinds": report["kinds"],
    "totals": report["totals"],
    "vision_preserved": report["vision"],
    "output": str(output),
}, sys.stdout, indent=2)
PY
VERIFY=$?
if [ "$VERIFY" -ne 0 ]; then
    echo "ARTIFACT_VERIFICATION_FAILED=$VERIFY"
    exit "$VERIFY"
fi

du -sh "$OUT"
df -h "$H/models"
echo "W4A8_DISK_ARTIFACT_OK=$OUT"
