#!/usr/bin/env bash
set -u

# Occamy-1.0 离线量化为 W8A8（int8 权重 + int8 激活）预量化目录。
#
# 与 W4A8 那条路的区别只有两处：用的是 quant_disk.py（逐输出列 absmax /127），
# 以及产物 35 GB 而不是 19 GB——一张 32 GB 的卡放不下，所以 W8A8 只能四卡（或两卡）跑。
# 选 W8A8 的理由是精度：真权重上 4-bit 的 RMS 相对误差是 11.8%，int8 是 0.4% 量级，
# 而本卡 int8 的 GEMM 速率是 49.07 TINT8-OPS/s，预填充侧并不吃亏。

H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
ENGINE_ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=$H/models/occamy-1.0
OUT=$H/models/occamy-1.0-W8A8-kn

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

if [ ! -d "$MODEL" ]; then echo SOURCE_MODEL_MISSING; exit 66; fi
if [ -e "$OUT" ]; then echo OUTPUT_ALREADY_EXISTS; exit 73; fi

AVAILABLE=${CUDA_VISIBLE_DEVICES:-0}
PICKED=
for DEV in $(echo "$AVAILABLE" | tr ',' ' '); do
    if CUDA_VISIBLE_DEVICES=$DEV timeout --foreground -k 2 5 "$PY" -B -u -c \
        "import torch; a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda'); b=a@a; torch.cuda.synchronize(); raise SystemExit(0 if bool(torch.isfinite(b).all()) else 1)" \
        >/dev/null 2>&1; then PICKED=$DEV; break; fi
    echo "DEAD_CARD=$DEV"
done
if [ -z "$PICKED" ]; then echo NO_HEALTHY_CARD; exit 64; fi
export CUDA_VISIBLE_DEVICES=$PICKED
echo "PICKED_CARD=$PICKED"

"$PY" -B -u "$ENGINE_ROOT/tools/quant_disk.py" \
    --checkpoint "$MODEL" --out "$OUT" --weight-dtype bf16 \
    --device auto --include-vision || exit $?

"$PY" -B - "$MODEL" "$OUT" <<'PY'
import json, re, sys
from pathlib import Path
source, output = map(Path, sys.argv[1:])
with (source / "model.safetensors.index.json").open(encoding="utf-8") as f:
    source_map = json.load(f)["weight_map"]
with (output / "quant_index.json").open(encoding="utf-8") as f:
    report = json.load(f)
prefixes = ("model.language_model.", "mtp.", "lm_head.", "model.visual.")
# 逐专家张量在产物里被融合成两个三维栈，所以期望集合要把 30720 个 ``experts.{e}.*``
# 折成每层两个栈名；否则校验必然报 MISSING，而权重其实是齐的。
EXP = re.compile(r"^(?P<mlp>.+)\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)\.weight$")
expected = set()
for k in source_map:
    if not k.startswith(prefixes):
        continue
    m = EXP.match(k)
    if m:
        expected.add(m.group("mlp") + ".experts.gate_up_stack")
        expected.add(m.group("mlp") + ".experts.down_stack")
    else:
        expected.add(k)
actual = set(report["tensors"])
missing = expected - actual
if missing:
    raise SystemExit(f"MISSING={len(missing)} first={sorted(missing)[0]}")
if report.get("format") != "coffin-int8-kn-v1" or report.get("kinds", {}).get("int8", 0) == 0:
    raise SystemExit("INVALID_INDEX")
routers = [k for k in actual if k.endswith("mlp.gate.weight")]
dense_kept = [k for k in routers if report["tensors"][k]["kind"] == "dense"]
if len(dense_kept) != len(routers):
    raise SystemExit(f"ROUTER_NOT_DENSE {len(routers) - len(dense_kept)}")
stacks = [k for k in actual if k.endswith((".experts.gate_up_stack", ".experts.down_stack"))]
if len(stacks) != 2 * sum(1 for k in expected if k.endswith(".experts.gate_up_stack")):
    raise SystemExit(f"EXPERT_STACKS_INCOMPLETE {len(stacks)}")
print(json.dumps({"result": "W8A8_DISK_ARTIFACT_OK",
                  "indexed": len(actual), "expert_stacks": len(stacks),
                  "kinds": report["kinds"], "totals": report["totals"],
                  "output": str(output)}, indent=1))
PY
VERIFY=$?
if [ "$VERIFY" -ne 0 ]; then echo ARTIFACT_VERIFICATION_FAILED=$VERIFY; exit "$VERIFY"; fi
du -sh "$OUT"
echo "W8A8_DISK_ARTIFACT_OK=$OUT"
