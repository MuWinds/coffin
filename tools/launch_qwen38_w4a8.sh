#!/usr/bin/env bash
set -u

H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
ENGINE_ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=$H/models/Qwen3.8-27B
OUT=$H/models/Qwen3.8-27B-W4A8-g128

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
import hashlib
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

handles = {}
keysets = {}
try:
    for key, meta in report["tensors"].items():
        shard = output / meta["shard"]
        if not shard.is_file():
            raise SystemExit(f"MISSING_SHARD={meta['shard']}")
        if meta["shard"] not in handles:
            handle = safe_open(str(shard), framework="pt", device="cpu")
            handle.__enter__()
            handles[meta["shard"]] = handle
            keysets[meta["shard"]] = set(handle.keys())
        keys = keysets[meta["shard"]]
        if key not in keys:
            raise SystemExit(f"MISSING_TENSOR={key}")
        if meta["kind"] == "q4" and meta["scale_key"] not in keys:
            raise SystemExit(f"MISSING_SCALE={key}")
finally:
    for handle in handles.values():
        handle.__exit__(None, None, None)

for name in ("config.json", "tokenizer.json", "preprocessor_config.json"):
    if not (output / name).is_file():
        raise SystemExit(f"MISSING_SIDECAR={name}")
digest = hashlib.sha256((source / "config.json").read_bytes()).hexdigest()
if report.get("source_config_sha256") != digest:
    raise SystemExit("SOURCE_CONFIG_DIGEST_MISMATCH")

print(json.dumps({
    "result": "W4A8_DISK_ARTIFACT_OK",
    "format": report["format"],
    "source_tensors": len(expected),
    "indexed_tensors": len(actual),
    "kinds": report["kinds"],
    "totals": report["totals"],
    "vision_preserved": report["vision"],
    "output": str(output),
}, indent=2))
PY
VERIFY=$?
if [ "$VERIFY" -ne 0 ]; then
    echo "ARTIFACT_VERIFICATION_FAILED=$VERIFY"
    exit "$VERIFY"
fi

du -sh "$OUT"
df -h "$H/models"
echo "W4A8_DISK_ARTIFACT_OK=$OUT"
