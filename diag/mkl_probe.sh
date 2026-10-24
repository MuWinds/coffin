#!/bin/sh
# MKL refuses to initialize inside the container since the host changed under us;
# find which knob brings torch back (the model only needs the DCU, not the CPU
# BLAS, so any working setting is fine).
for v in "MKL_THREADING_LAYER=GNU" "MKL_ENABLE_INSTRUCTIONS=AVX2" "MKL_DEBUG_CPU_TYPE=5" \
         "MKL_SERVICE_FORCE_INTEL=1" "MKL_THREADING_LAYER=GNU MKL_ENABLE_INSTRUCTIONS=AVX2" \
         "MKL_THREADING_LAYER=GNU MKL_ENABLE_INSTRUCTIONS=AVX2 MKL_DEBUG_CPU_TYPE=5"; do
    printf '%-64s ' "$v"
    env $v python3 -c 'import torch; print("OK", torch.__version__, torch.cuda.is_available())' 2>&1 | tail -1
done
echo '--- cpu ---'
grep -m1 'model name' /proc/cpuinfo
grep -o -m1 'avx512[a-z]*' /proc/cpuinfo | head -1
echo '--- ldpaths ---'
ls /usr/local/lib/python3.11/site-packages/torch/lib/ | grep -i mkl
