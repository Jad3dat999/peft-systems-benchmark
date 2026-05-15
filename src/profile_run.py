#!/usr/bin/env python3
"""Profile PEFT training steps with torch.profiler and extract kernel-level data.

Runs a short manual training loop under torch.profiler to capture GPU kernel
timings, memory allocation, and FLOP estimates. Produces:
  - trace.json           Chrome-compatible trace for chrome://tracing
  - kernel_summary.txt   Human-readable table of kernel averages
  - kernel_data.json     Machine-readable per-kernel statistics
  - kernel_categories.json  Kernels grouped by functional category with time totals

Usage:
    python src/profile_run.py --experiment lora_r16 --output_dir results/
    python src/profile_run.py --all --output_dir results/
    python src/profile_run.py --list
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader

# Ensure sibling modules are importable regardless of working directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_model, apply_peft, prepare_dataset
from config import EXPERIMENTS, ExperimentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# The four canonical methods for --all profiling
KEY_METHODS: list[str] = ["full_ft", "lora_r16", "qlora_r16", "dora_r16"]

# Profiler schedule constants
PROF_WAIT = 1
PROF_WARMUP = 2
PROF_ACTIVE = 8
PROF_REPEAT = 1
PROF_TOTAL_STEPS = (PROF_WAIT + PROF_WARMUP + PROF_ACTIVE) * PROF_REPEAT

# Kernel category mapping

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "GEMM": ["gemm", "cublas", "matmul", "addmm", "linear", "mm_"],
    "Attention": ["attention", "flash", "sdpa", "softmax", "_attn"],
    "Quantization": ["quantize", "dequant", "nf4", "bitsandbytes", "bnb_", "int4"],
    "Normalization": ["norm", "layer_norm", "rms_norm", "dora", "weight_norm"],
    "Optimizer": ["adam", "optimizer", "step", "_foreach"],
    "Elementwise": ["add_", "mul_", "gelu", "silu", "relu", "dropout", "embedding"],
}


def classify_kernel(name: str) -> str:
    """Map a kernel name to a functional category based on keyword matching.

    Args:
        name: Kernel or operator name from the profiler.

    Returns:
        Category string (one of CATEGORY_KEYWORDS keys, or "Other").
    """
    name_lower = name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return category
    return "Other"


def _get_cuda_time(event) -> float:
    """Safely get CUDA time from a profiler event.

    PyTorch 2.5+ renamed cuda_time_total to self_cuda_time_total on
    FunctionEventAvg objects. This helper tries both attributes.
    """
    for attr in ("cuda_time_total", "self_cuda_time_total", "device_time_total"):
        if hasattr(event, attr):
            return getattr(event, attr)
    return 0.0


def _get_cuda_mem(event) -> int:
    """Safely get CUDA memory usage from a profiler event."""
    for attr in ("cuda_memory_usage", "device_memory_usage", "self_cuda_memory_usage"):
        if hasattr(event, attr):
            return getattr(event, attr)
    return 0


# Manual training loop


def build_dataloader(
    tokenizer,
    cfg: ExperimentConfig,
    num_samples: int | None = None,
) -> DataLoader:
    """Build a DataLoader from the Alpaca dataset.

    Args:
        tokenizer: HuggingFace tokenizer.
        cfg: Experiment configuration.
        num_samples: Optional cap on dataset size (speeds up profiling).

    Returns:
        A PyTorch DataLoader.
    """
    dataset = prepare_dataset(tokenizer, cfg)

    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )


def manual_training_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    use_bf16: bool,
) -> float:
    """Execute a single forward + backward + optimizer step.

    Args:
        model: The model to train.
        batch: Dictionary with input_ids, attention_mask, labels.
        optimizer: Optimizer instance.
        device: Target device.
        scaler: GradScaler for mixed-precision (None if bf16).
        use_bf16: Whether to use bfloat16 autocast.

    Returns:
        Loss value as a float.
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    optimizer.zero_grad(set_to_none=True)

    dtype = torch.bfloat16 if use_bf16 else torch.float16
    with torch.amp.autocast("cuda", dtype=dtype):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return loss.item()


# Profiling


def profile_experiment(
    experiment_name: str,
    output_root: str,
) -> None:
    """Profile a single experiment and write all output artifacts.

    Args:
        experiment_name: Key into EXPERIMENTS dict.
        output_root: Root directory for output files.
    """
    cfg = EXPERIMENTS[experiment_name]
    exp_dir = os.path.join(output_root, f"profile_{experiment_name}")
    os.makedirs(exp_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("PROFILING: %s", experiment_name)
    log.info("  method=%s  rank=%s  4bit=%s  gc=%s  fa=%s",
             cfg.method, cfg.lora_rank, cfg.load_in_4bit,
             cfg.gradient_checkpointing, cfg.flash_attention)
    log.info("  schedule: wait=%d  warmup=%d  active=%d  repeat=%d  (total=%d steps)",
             PROF_WAIT, PROF_WARMUP, PROF_ACTIVE, PROF_REPEAT, PROF_TOTAL_STEPS)
    log.info("=" * 60)

    # Reset GPU state
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log.warning("No CUDA device found. Profiling will run on CPU only.")

    # Load model + tokenizer
    model, tokenizer = load_model(cfg)
    model, _ = apply_peft(model, cfg)

    if cfg.gradient_checkpointing and not cfg.load_in_4bit:
        model.gradient_checkpointing_enable()
        log.info("Gradient checkpointing enabled")

    model.train()

    # Prepare data
    # Only load enough samples for profiling (avoids tokenizing entire dataset)
    min_samples = PROF_TOTAL_STEPS * cfg.batch_size + cfg.batch_size
    dataloader = build_dataloader(tokenizer, cfg, num_samples=min_samples * 2)

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # bf16 does not need GradScaler; fp16 does
    scaler = None if cfg.bf16 else torch.amp.GradScaler("cuda")

    # Profiler setup
    trace_path = os.path.join(exp_dir, "trace.json")

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    prof_schedule = schedule(
        wait=PROF_WAIT,
        warmup=PROF_WARMUP,
        active=PROF_ACTIVE,
        repeat=PROF_REPEAT,
    )

    log.info("Starting profiled training loop (%d steps)...", PROF_TOTAL_STEPS)
    wall_start = time.perf_counter()

    data_iter = iter(dataloader)

    with profile(
        activities=activities,
        schedule=prof_schedule,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        for step_idx in range(PROF_TOTAL_STEPS):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            loss = manual_training_step(
                model, batch, optimizer, device, scaler, cfg.bf16,
            )

            prof.step()

            if step_idx % 3 == 0 or step_idx == PROF_TOTAL_STEPS - 1:
                log.info("  step %2d/%d  loss=%.4f", step_idx + 1, PROF_TOTAL_STEPS, loss)

    wall_elapsed = time.perf_counter() - wall_start
    log.info("Profiled training complete in %.1f s", wall_elapsed)

    # Export Chrome trace
    prof.export_chrome_trace(trace_path)
    log.info("Chrome trace saved to %s", trace_path)

    # Extract and save kernel data
    _save_kernel_summary(prof, exp_dir)
    _save_kernel_data(prof, exp_dir)
    _save_kernel_categories(prof, exp_dir)

    log.info("All profiling artifacts written to %s", exp_dir)


# Kernel data extraction


def _save_kernel_summary(prof: profile, output_dir: str) -> None:
    """Write human-readable kernel summary table.

    Args:
        prof: Completed profiler context.
        output_dir: Directory for output file.
    """
    summary_path = os.path.join(output_dir, "kernel_summary.txt")
    table = prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=100,
    )
    with open(summary_path, "w") as f:
        f.write(table)
    log.info("Kernel summary written to %s", summary_path)


def _save_kernel_data(prof: profile, output_dir: str) -> None:
    """Write machine-readable per-kernel statistics to JSON.

    Extracts: name, cuda_time_us, cpu_time_us, calls, self_cuda_time_us,
    cuda_memory_usage for every kernel in the profiler's key_averages.

    Args:
        prof: Completed profiler context.
        output_dir: Directory for output file.
    """
    data_path = os.path.join(output_dir, "kernel_data.json")
    kernels: list[dict[str, Any]] = []

    for event in prof.key_averages():
        kernels.append({
            "name": event.key,
            "cuda_time_us": round(_get_cuda_time(event), 2),
            "cpu_time_us": round(event.cpu_time_total, 2),
            "calls": event.count,
            "self_cuda_time_us": round(_get_cuda_time(event), 2),
            "cuda_memory_usage": _get_cuda_mem(event),
        })

    # Sort by total CUDA time descending
    kernels.sort(key=lambda k: k["cuda_time_us"], reverse=True)

    with open(data_path, "w") as f:
        json.dump(kernels, f, indent=2)
    log.info("Kernel data (%d entries) written to %s", len(kernels), data_path)


def _save_kernel_categories(prof: profile, output_dir: str) -> None:
    """Categorize kernels and write per-category time breakdown.

    Each kernel is classified into one of: GEMM, Attention, Quantization,
    Normalization, Optimizer, Elementwise, Other. The output includes
    per-kernel classification and aggregate statistics.

    Args:
        prof: Completed profiler context.
        output_dir: Directory for output file.
    """
    categories_path = os.path.join(output_dir, "kernel_categories.json")

    # Collect per-kernel data with categories
    categorized_kernels: list[dict[str, Any]] = []
    category_totals: dict[str, float] = {}

    for event in prof.key_averages():
        cat = classify_kernel(event.key)
        cuda_us = _get_cuda_time(event)

        categorized_kernels.append({
            "name": event.key,
            "category": cat,
            "cuda_time_us": round(cuda_us, 2),
            "calls": event.count,
        })

        category_totals[cat] = category_totals.get(cat, 0.0) + cuda_us

    # Compute percentages
    total_cuda_us = sum(category_totals.values())
    category_summary: list[dict[str, Any]] = []

    for cat in list(CATEGORY_KEYWORDS.keys()) + ["Other"]:
        total_us = category_totals.get(cat, 0.0)
        pct = (total_us / total_cuda_us * 100) if total_cuda_us > 0 else 0.0
        category_summary.append({
            "category": cat,
            "total_cuda_time_us": round(total_us, 2),
            "percentage": round(pct, 2),
        })

    # Sort summary by time descending
    category_summary.sort(key=lambda c: c["total_cuda_time_us"], reverse=True)

    output = {
        "total_cuda_time_us": round(total_cuda_us, 2),
        "category_summary": category_summary,
        "kernels": categorized_kernels,
    }

    with open(categories_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Kernel categories written to %s", categories_path)


# CLI


def main() -> None:
    """Entry point: parse arguments and run profiling."""
    parser = argparse.ArgumentParser(
        description="Profile PEFT training steps with torch.profiler",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name from config.EXPERIMENTS (e.g., lora_r16)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Root output directory (default: ./results)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Profile all key methods: {', '.join(KEY_METHODS)}",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available experiments and exit",
    )
    args = parser.parse_args()

    # List mode
    if args.list:
        print(f"\nAvailable experiments ({len(EXPERIMENTS)}):\n")
        print(f"{'Name':<22} {'Method':<8} {'Rank':<6} {'4bit':<6} {'GC':<6} {'FA':<6}")
        print("-" * 60)
        for name, cfg in EXPERIMENTS.items():
            marker = " *" if name in KEY_METHODS else ""
            print(
                f"{name:<22} {cfg.method:<8} "
                f"{cfg.lora_rank if cfg.method != 'full' else 'all':<6} "
                f"{str(cfg.load_in_4bit):<6} "
                f"{str(cfg.gradient_checkpointing):<6} "
                f"{str(cfg.flash_attention):<6}{marker}"
            )
        print(f"\n  * = included in --all ({', '.join(KEY_METHODS)})")
        return

    # Determine which experiments to profile
    if args.all:
        targets = KEY_METHODS
    elif args.experiment:
        if args.experiment not in EXPERIMENTS:
            parser.error(
                f"Unknown experiment '{args.experiment}'. Use --list to see options."
            )
        targets = [args.experiment]
    else:
        parser.error("Provide --experiment <name> or --all (use --list to see options)")
        return  # unreachable, satisfies type checker

    # Run profiling
    log.info("Profiling %d experiment(s): %s", len(targets), ", ".join(targets))

    for i, name in enumerate(targets, 1):
        log.info("[%d/%d] Starting profiling for: %s", i, len(targets), name)
        try:
            profile_experiment(name, args.output_dir)
        except Exception:
            log.exception("Failed to profile experiment '%s'", name)
            if len(targets) == 1:
                raise
            log.info("Continuing with remaining experiments...")
            continue

        # Free GPU memory between experiments
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log.info("Profiling complete. Results in %s/", args.output_dir)


if __name__ == "__main__":
    main()
