#!/bin/bash
# Post-training evaluation chain. Waits for the final 3-seed training and the
# M9 fixed rerun, then runs every scoring step sequentially.
cd "$(dirname "$0")/../.." || exit 1
LOG=experiments/eval_chain.log
: > "$LOG"

echo "[$(date)] waiting for final 3-seed training..." | tee -a "$LOG"
while [ "$(grep -c 'Training complete' experiments/final_seeds.log 2>/dev/null)" != "3" ]; do
    sleep 20
done
echo "[$(date)] waiting for M9 fixed rerun..." | tee -a "$LOG"
while ! grep -q "M9_FIXED_RERUN_DONE" experiments/m9_fixed_rerun.log 2>/dev/null; do
    sleep 20
done
sleep 5

echo "===== STEP 1: final 3-seed GT eval ($(date)) =====" | tee -a "$LOG"
python3 experiments/scripts/m_sparse_gt_eval.py \
    --tag final --seeds 123,456,789 2>&1 | tee -a "$LOG"

echo "===== STEP 2: M9 fixed GT eval ($(date)) =====" | tee -a "$LOG"
python3 experiments/scripts/m_sparse_gt_eval.py \
    --tag m9_fixed --seeds 42 --suffix lam3 2>&1 | tee -a "$LOG"

echo "===== STEP 3: M9 stochastic GT eval ($(date)) =====" | tee -a "$LOG"
python3 experiments/scripts/m_sparse_gt_eval.py \
    --tag m9_stoch --seeds 42 --suffix lam3 2>&1 | tee -a "$LOG"

echo "===== STEP 4: M10 four-lambda GT range ($(date)) =====" | tee -a "$LOG"
python3 experiments/scripts/m10_gt_range.py 2>&1 | tee -a "$LOG"

echo "===== STEP 5: LOO final sparse ($(date)) =====" | tee -a "$LOG"
python3 experiments/scripts/m_generic_loo.py --model sparse \
    --tag final --seeds 123,456,789 \
    --out experiments/final_loo.json 2>&1 | tee -a "$LOG"

echo "===== STEP 6: LOO ABMIL ($(date)) =====" | tee -a "$LOG"
python3 experiments/scripts/m_generic_loo.py --model abmil \
    --dir experiments/multi_seed_abmil_seed{seed} --seeds 123,456,789 \
    --out experiments/abmil_loo.json 2>&1 | tee -a "$LOG"

echo "===== STEP 7: proxy refresh ($(date)) =====" | tee -a "$LOG"
python3 experiments/scripts/m_proxy_refresh.py 2>&1 | tee -a "$LOG"

echo "===== STEP 8: unified table data ($(date)) =====" | tee -a "$LOG"
python3 experiments/scripts/m_table_data.py 2>&1 | tee -a "$LOG"

echo "===== EVAL_CHAIN_DONE ($(date)) =====" | tee -a "$LOG"
