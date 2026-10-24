#!/bin/bash
# Post-chain waiter: after the main overnight chain completes, run the
# M9 stochastic-corruption control (same seed/split/protocol as fixed).
cd "$(dirname "$0")/../.." || exit 1

echo "[$(date)] waiting for main chain..."
while ! grep -q "OVERNIGHT_CHAIN_DONE" experiments/overnight.log 2>/dev/null; do
    sleep 30
done
sleep 10
echo "===== STEP 5: M9 stochastic control ($(date)) ====="
python3 experiments/scripts/rerun_sparse_val.py \
    --seeds 42 --tag m9_stoch --flip-frac 0.2 --flip-mode stochastic 2>&1
echo "===== POSTCHAIN2_DONE ($(date)) ====="
