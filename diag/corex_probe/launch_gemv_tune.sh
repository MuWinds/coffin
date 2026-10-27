set -u
H=/data/home/2024240384035
PY=/data/software/Anaconda3/2024.10-1/envs/py310/bin/python
PATH=/data/software/Anaconda3/2024.10-1/envs/py310/bin:$PATH
export PATH
export TORCH_EXTENSIONS_DIR=$H/torch_ext_coffin_corex
export COFFIN_BACKEND=corex
export ENGINE_MODEL=$H/models/Qwen3.5-4B
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [ -n "${COFFIN_ROOT:-}" ]; then cd "$COFFIN_ROOT" || exit 1; else cd "$(dirname "$0")/../../.." || exit 1; fi
echo "HOST=$(hostname)  PWD=$(pwd)"
timeout --foreground -k 10 60 "$PY" -B -u -c "import torch; a=torch.randn(8,8,dtype=torch.bfloat16,device='cuda'); b=a@a; torch.cuda.synchronize(); print('SANITY_OK', bool(torch.isfinite(b).all()), flush=True)"
if [ $? -ne 0 ]; then echo "CARD_UNHEALTHY on $(hostname)"; exit 64; fi
"$PY" -B -u diag/corex_probe/probe_gemv_tune.py
echo "TUNE_EXIT=$?"