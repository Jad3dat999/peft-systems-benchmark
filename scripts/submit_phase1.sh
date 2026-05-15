#!/bin/bash
# Submit Phase 1: 6-method comparison experiments
# Run this on the NOTS login node after environment is set up.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

echo "Submitting Phase 1: Method Comparison (6 jobs)"
echo ""

EXPERIMENTS=(full_ft lora_r16 qlora_r16 dora_r16 qdora_r16 adalora_r16)

for exp in "${EXPERIMENTS[@]}"; do
    echo -n "  ${exp}... "
    JOB_ID=$(sbatch --job-name="p1-${exp}" --parsable scripts/run_single.sh "$exp")
    echo "submitted (job ${JOB_ID})"
    sleep 2
done

echo ""
echo "Phase 1 submitted. Monitor with:  squeue -u \$USER"
echo "Check results with:  ls results/*/metrics.json | wc -l"
