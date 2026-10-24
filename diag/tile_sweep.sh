#!/bin/sh
# Sweep the causal-attention query-tile width (prefill) and re-confirm the paged
# suite, which also runs the tiled path.
cd /root/private_data/engine/diag || exit 1
export PYTHONPATH=/root/private_data/engine
export TORCH_EXTENSIONS_DIR=/root/private_data/torch_ext
export MKL_THREADING_LAYER=GNU
for q in 256 512 1024 2048; do
    printf 'TILE=%-5s ' "$q"
    PREFILL_ATTN_TILE=$q python3 bench_prefill2.py 2>&1 | grep -E '^ *2048' | head -1
done
echo '[sweep] paged'
python3 check_paged.py 2>&1 | grep -E 'PASS|FAIL|DONE'
echo '[sweep] ALLDONE'
