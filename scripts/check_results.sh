#!/bin/bash
# Quick validation: check all Phase 1 results and print summary table

set -euo pipefail
cd "$(dirname "$0")/.."

echo ""
echo "PEFT Benchmark — Results Summary"
echo ""

python3 << 'PYEOF'
import json, glob, os

files = sorted(glob.glob('results/*/metrics.json'))
if not files:
    print('No results found in results/*/metrics.json')
    exit(1)

header = f"{'Experiment':<22} {'Method':<8} {'Rank':<6} {'Mem(GB)':<9} {'Speed(s/s)':<11} {'StepMs':<9} {'Loss':<8} {'Train%':<8}"
print(header)
print('-' * len(header))

for f in files:
    name = os.path.basename(os.path.dirname(f))
    d = json.load(open(f))
    rank = str(d.get('rank', 'N/A'))
    loss = d.get('final_loss')
    loss_str = f"{loss:<8.3f}" if loss is not None else "N/A     "
    print(
        f"{name:<22} {d['method']:<8} {rank:<6} "
        f"{d['peak_memory_gb']:<9.2f} "
        f"{d['throughput_samples_per_sec']:<11.2f} "
        f"{d['avg_step_time_ms']:<9.1f} "
        f"{loss_str} "
        f"{d.get('trainable_pct', 100):<8.4f}"
    )
print()
print(f'Total experiments: {len(files)}')
PYEOF
