#!/bin/bash
#SBATCH --job-name=peft-profile
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:lovelace:1
#SBATCH --output=logs/profile_%j.out
#SBATCH --error=logs/profile_%j.err
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module load CUDA/12.1.1
export PATH="/scratch/$USER/conda/envs/peft/bin:$PATH"
export HF_HOME="/scratch/$USER/huggingface_cache"
export PIP_CACHE_DIR="/scratch/$USER/pip_cache"

cd "$SHARED_SCRATCH/$USER/peft-systems-analysis"

echo ""
echo "  PEFT Kernel Profiling"
echo "  GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

METHODS=(full_ft lora_r16 qlora_r16 dora_r16)

for exp in "${METHODS[@]}"; do
    echo ""
    echo "-- Profiling ${exp} at $(date '+%H:%M:%S') "
    python src/profile_run.py --experiment "$exp" --output_dir results/
    echo "-- Done ${exp} "
done

echo ""
echo ""
echo "  All profiling complete at $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Traces: results/profile_*/trace.json"
echo ""
