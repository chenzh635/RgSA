#!/bin/bash
# Wait for the M9 stochastic control to finish, then rerun M9 fixed with
# the train.py fix (global-index-keyed corruption matrix).
cd "$(dirname "$0")/../.." || exit 1

echo "[$(date)] waiting for M9 stochastic..."
while ! grep -q "POSTCHAIN2_DONE" experiments/overnight2.log 2>/dev/null; do
    sleep 20
done
sleep 10
echo "===== M9 fixed rerun ($(date)) ====="
python3 experiments/scripts/rerun_sparse_val.py \
    --seeds 42 --tag m9_fixed --flip-frac 0.2 --flip-mode fixed 2>&1
echo "===== M9_FIXED_RERUN_DONE ($(date)) ====="
