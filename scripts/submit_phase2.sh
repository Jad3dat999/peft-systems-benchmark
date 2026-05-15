#!/bin/bash
# Submit Phase 2: LoRA rank sensitivity sweep
# Prereq: Phase 1 lora_r16 completed successfully (validates setup)

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

# Check that lora_r16 from Phase 1 succeeded
if [[ ! -f results/lora_r16/metrics.json ]]; then
    echo "ERROR: results/lora_r16/metrics.json not found."
    echo "Run Phase 1 first and verify lora_r16 completed."
    exit 1
fi

echo "Submitting Phase 2: Rank Sweep (5 jobs, r=16 reused from Phase 1)"
echo ""

for r in 4 8 32 64 128; do
    echo -n "  lora_r${r}... "
    JOB_ID=$(sbatch --job-name="p2-r${r}" --parsable scripts/run_single.sh "lora_r${r}")
    echo "submitted (job ${JOB_ID})"
    sleep 2
done

echo ""
echo "Phase 2 submitted."
