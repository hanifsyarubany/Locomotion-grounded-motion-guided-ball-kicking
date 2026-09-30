#!/usr/bin/env bash
# Checkpoint sweep for every policy reported in the paper (7 specialists + 2 distillations).
# 8 checkpoints each, 30 iterations x 10 trials = 300 attempts per checkpoint.
# Run from the repo root with the hssim env active.
set -u
cd "$(dirname "$0")/.."

CFGS=(
  sweep-skill-012 sweep-skill-016 sweep-skill-017 sweep-skill-018
  sweep-skill-019 sweep-skill-020 sweep-skill-011
  sweep-unified-7 sweep-unified-6
)

mkdir -p out/sweep_logs
for c in "${CFGS[@]}"; do
  echo "=== $(date +%H:%M:%S)  START $c ==="
  python src/holosoma/holosoma/sim2sim_eval.py \
      --config "configs/sim2sim_eval/${c}.yaml" \
      2>&1 | tee "out/sweep_logs/${c}.log"
  echo "=== $(date +%H:%M:%S)  DONE  $c (exit ${PIPESTATUS[0]}) ==="
done
echo "All sweeps finished. Aggregate with:  python scripts/aggregate_sweep.py"
