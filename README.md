# PEFT Systems Benchmark: GPU Kernel & Memory Profiling

Systems-level benchmark of six parameter-efficient fine-tuning (PEFT) methods on Microsoft Phi-2 (2.7B), measuring GPU memory, training throughput, and kernel-level CUDA cost on the Rice NOTS HPC cluster.

**Methods compared:** LoRA, QLoRA, DoRA, QDoRA, AdaLoRA, full fine-tuning
**Dataset:** Stanford Alpaca (52K instruction-following samples)
**Hardware:** NVIDIA L40S (48 GB, Lovelace) on the NOTS `commons` partition
**Project report:** see [`report.pdf`](report.pdf) for motivation, methodology, full results, and figures

---

## What this project answers

1. **How do PEFT methods compare in peak GPU memory and training throughput?** — 6-method comparison at fixed rank, identical hardware/dataset/seed.
2. **Where does GPU compute time go for each method?** — `torch.profiler` decomposition into GEMM / attention / quantization / optimizer / other kernel categories.
3. **How does LoRA rank affect the memory–throughput tradeoff?** — sweep over r ∈ {4, 8, 16, 32, 64, 128}.
4. **How do PEFT methods interact with system optimizations?** — ablations over Flash-Attention 2 and gradient checkpointing.

Each experiment writes a `metrics.json` (peak memory, throughput, loss, trainable params) plus a `loss_curve.json`. Profile runs additionally write `kernel_categories.json` and `kernel_summary.txt`.

---

## Repository Layout

```
src/
  config.py         experiment configurations
  train.py          training loop with metrics collection
  profile_run.py    torch.profiler harness (Phase 4)
scripts/
  run_single.sh     SLURM job wrapper for one experiment
  submit_phaseN.sh  batch-submit a phase of experiments
  run_profiling.sh  SLURM job for kernel profiling
  check_results.sh  print a summary table of all results
analysis/
  plot_results.py   regenerate all figures from results/
results/
  <exp_name>/metrics.json + loss_curve.json
  profile_<method>/kernel_categories.json, kernel_data.json, kernel_summary.txt
report.pdf          full writeup
```

## Environment

```
Python 3.10, PyTorch 2.5.1 + CUDA 12.1
transformers 4.44, peft 0.12, bitsandbytes 0.43
datasets 2.x, accelerate 0.x
```

Install:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft bitsandbytes datasets accelerate matplotlib
```

## Running

```bash
# List options
python src/train.py --list

# One experiment, locally
python src/train.py --experiment lora_r16

# One experiment, via SLURM
sbatch scripts/run_single.sh lora_r16 results

# Full phase
bash scripts/submit_phase1.sh   # 6-method comparison
bash scripts/submit_phase2.sh   # LoRA rank sweep
bash scripts/submit_phase3.sh   # gradient-checkpointing x Flash-Attention
sbatch scripts/run_profiling.sh # kernel profiling (Phase 4)

# Regenerate figures
python analysis/plot_results.py
```

## Hardware note

SLURM scripts request `--gres=gpu:lovelace:1` to pin jobs to L40S GPUs on NOTS. Replace with `gpu:1` to run on any available GPU; mixing GPU types across experiments will produce non-comparable memory/throughput numbers.

---

*Course project for COMP 568 (Deep Learning System Design and Optimization), Rice University, Spring 2026.*
