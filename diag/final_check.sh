#!/bin/sh
# One-command functional regression.
#
# The correctness checks moved to tests/ (unittest); this runs the three heaviest
# modules and prints an area name before each, so a failure says where to look.
# The full suite is:
#     MKL_THREADING_LAYER=GNU PYTHONPATH=$PWD python3 -m unittest discover -s tests -v
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$(pwd)"
export MKL_THREADING_LAYER=GNU
export TORCH_EXTENSIONS_DIR=/root/private_data/torch_ext
for mod in test_paging test_mtp test_spec; do
    echo "[check] $mod"
    timeout 2400 python3 -m unittest discover -s tests -p "$mod.py" -v 2>&1 | tail -15
done
echo '[check] ALLDONE'
