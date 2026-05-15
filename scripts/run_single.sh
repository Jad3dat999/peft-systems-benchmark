#!/bin/bash
#SBATCH --job-name=peft-bench
# Note: override with --job-name=p1-<experiment> when submitting directly
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:lovelace:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=ky65@rice.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Environment
module load CUDA/12.1.1
export PATH="/scratch/ky65/conda/envs/peft/bin:$PATH"
export HF_HOME="/scratch/ky65/huggingface_cache"
export PIP_CACHE_DIR="/scratch/ky65/pip_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPERIMENT="${1:?Usage: sbatch run_single.sh <experiment_name>}"
OUTPUT_DIR="${2:-results}"

# Banner
echo ""
echo "  Experiment : ${EXPERIMENT}"
echo "  GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Node       : $(hostname)"
echo "  Start      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  CUDA       : $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'N/A')"
echo ""

# Run
cd "$SHARED_SCRATCH/$USER/peft-systems-analysis"
python src/train.py --experiment "${EXPERIMENT}" --output_dir "${OUTPUT_DIR}"

echo ""
echo ""
echo "  End        : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results    : ${OUTPUT_DIR}/${EXPERIMENT}/metrics.json"
echo ""
