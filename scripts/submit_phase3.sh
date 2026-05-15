#!/bin/bash
# Submit Phase 3: gradient-checkpointing interactions

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

echo "Submitting Phase 3: Gradient Checkpointing (4 jobs)"
echo ""

EXPERIMENTS=(lora_r16_gc dora_r16_gc qlora_r16_gc full_ft_gc)

for exp in "${EXPERIMENTS[@]}"; do
    echo -n "  ${exp}... "
    JOB_ID=$(sbatch --job-name="p3-${exp}" --parsable scripts/run_single.sh "$exp")
    echo "submitted (job ${JOB_ID})"
    sleep 2
done

echo ""
echo "Phase 3 submitted."
