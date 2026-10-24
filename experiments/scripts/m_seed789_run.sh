#!/bin/bash
# Train final seed 789 (never started before the earlier kill). Appends to
# final_seeds.log; its "Training complete" is the 3rd marker that releases
# the evaluation chain.
cd "$(dirname "$0")/../.." || exit 1

echo "===== final seed789 run ($(date)) ====="
python3 experiments/scripts/rerun_sparse_val.py \
    --seeds 789 --lambdas 1.0 --tag final --force-seed-suffix \
    2>&1 | tee -a experiments/final_seeds.log
echo "===== FINAL_SEED789_DONE ($(date)) ====="
