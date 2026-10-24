#!/bin/bash
# Wait for the M9 fixed rerun to finish, then retrain final seed 456 from
# scratch (its earlier run was killed at epoch 4). Append to final_seeds.log
# so the eval chain's marker count advances to 2.
cd "$(dirname "$0")/../.." || exit 1

while ! grep -q "M9_FIXED_RERUN_DONE" experiments/m9_fixed_rerun.log 2>/dev/null; do
    sleep 15
done
sleep 5
echo "===== final seed456 resume ($(date)) ====="
python3 experiments/scripts/rerun_sparse_val.py \
    --seeds 456 --lambdas 1.0 --tag final --force-seed-suffix \
    2>&1 | tee -a experiments/final_seeds.log
echo "===== FINAL_SEED456_DONE ($(date)) ====="
