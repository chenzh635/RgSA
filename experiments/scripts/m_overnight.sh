#!/bin/bash
# M-sprint overnight chain (launched 2026-10-09 evening).
# Waits for faithful seeds 456/789, then: M1 GT+email, M1 LOO,
# M10 condensed lambda sweep (seed 42, val protocol), M9 fixed corruption.
# No set -e: later steps run even if an earlier one fails.
cd "$(dirname "$0")/../.." || exit 1
LOG=experiments/m1_faithful_seed456_789.log

echo "[$(date)] waiting for 456/789 to finish..."
while [ "$(grep -c 'done. best val acc' "$LOG" 2>/dev/null)" -lt 2 ]; do
    sleep 30
done
sleep 10
echo "[$(date)] seeds done; starting chain"

echo "===== STEP 1: M1 GT + email metrics ($(date)) ====="
python3 experiments/scripts/m1_gt_eval.py --seeds 123,456,789 2>&1

echo "===== STEP 2: M1 LOO ($(date)) ====="
python3 experiments/scripts/m1_loo.py --seeds 123,456,789 2>&1

echo "===== STEP 3: M10 lambda sweep ($(date)) ====="
python3 experiments/scripts/rerun_sparse_val.py \
    --lambdas 1.0,2.0,3.0,3.5 --tag m10_lambda 2>&1

echo "===== STEP 4: M9 fixed corruption ($(date)) ====="
python3 experiments/scripts/rerun_sparse_val.py \
    --seeds 42 --tag m9_fixed --flip-frac 0.2 --flip-mode fixed 2>&1

echo "===== OVERNIGHT_CHAIN_DONE ($(date)) ====="
