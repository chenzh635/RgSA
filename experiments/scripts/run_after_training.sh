#!/bin/bash
# Run this after both training runs complete
# It re-trains each model briefly (5 epochs) with model-specific checkpoint names

echo "=== Re-training Sparse ABMIL (5 epochs) ==="
python3 main.py --model sparse --mode train

echo ""
echo "=== Re-training ABMIL baseline (5 epochs) ==="
python3 main.py --model abmil --mode train

echo ""
echo "=== Running full evaluation ==="
python3 collect_results.py

echo ""
echo "=== Generating qualitative analysis ==="
python3 -m src.visualize

echo ""
echo "=== Updating training curves ==="
python3 -m src.plot_training

echo ""
echo "=== Done ==="
echo "Results saved to experiments/results.json"
echo "Plots in experiments/ and experiments/analysis/"
