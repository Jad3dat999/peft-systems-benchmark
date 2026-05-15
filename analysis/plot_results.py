#!/usr/bin/env python3
"""Generate publication-quality figures from PEFT benchmark results.

Reads metrics, loss curves, and kernel profiling data from the results
directory and produces six figures covering method comparison, rank sweeps,
kernel breakdowns, system-level interactions, training dynamics, and a
summary table.

Usage:
    python analysis/plot_results.py --results_dir results/ --figures_dir figures/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Style & palette


matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

METHOD_COLORS: dict[str, str] = {
    "Full FT":  "#2D4057",
    "LoRA":     "#3A86FF",
    "QLoRA":    "#8338EC",
    "DoRA":     "#FF6B6B",
    "QDoRA":    "#FF006E",
    "AdaLoRA":  "#FB5607",
}

KERNEL_COLORS: dict[str, str] = {
    "GEMM":           "#3A86FF",
    "Attention":      "#FF6B6B",
    "Quantization":   "#8338EC",
    "Normalization":  "#06D6A0",
    "Optimizer":      "#FB5607",
    "Elementwise":    "#FFD166",
    "Other":          "#AAAAAA",
}

# Mapping from experiment directory names to display labels.
PHASE1_EXPERIMENTS: list[tuple[str, str]] = [
    ("full_ft",      "Full FT"),
    ("lora_r16",     "LoRA"),
    ("qlora_r16",    "QLoRA"),
    ("dora_r16",     "DoRA"),
    ("qdora_r16",    "QDoRA"),
    ("adalora_r16",  "AdaLoRA"),
]


# IO helpers


def _load_json(path: Path) -> Any | None:
    """Load a JSON file; return *None* on any failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"  [WARN] Cannot load {path}: {exc}")
        return None


def _load_metrics(results_dir: Path, experiment: str) -> dict[str, Any] | None:
    return _load_json(results_dir / experiment / "metrics.json")


def _load_loss_curve(results_dir: Path, experiment: str) -> list[dict] | None:
    return _load_json(results_dir / experiment / "loss_curve.json")


def _load_kernel_categories(results_dir: Path, experiment: str) -> dict | None:
    return _load_json(results_dir / f"profile_{experiment}" / "kernel_categories.json")


def _save_figure(fig: matplotlib.figure.Figure, figures_dir: Path, stem: str) -> None:
    """Save *fig* as both PNG and PDF under *figures_dir*."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = figures_dir / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"  Saved {out}")
    plt.close(fig)


def _add_bar_labels(
    ax: matplotlib.axes.Axes,
    bars: matplotlib.container.BarContainer,
    fmt: str = "{:.1f}",
    fontsize: int = 8,
) -> None:
    """Add value labels above each bar."""
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                fmt.format(height),
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )


# Figure 1 -- Method comparison (1x3 bar chart)


def plot_fig1_method_comparison(results_dir: Path, figures_dir: Path) -> None:
    """1x3 subplot bar chart: memory, throughput, step time across methods."""
    print("Figure 1: method comparison ...")

    names: list[str] = []
    memory: list[float] = []
    throughput: list[float] = []
    step_time: list[float] = []
    colors: list[str] = []

    for exp_name, label in PHASE1_EXPERIMENTS:
        m = _load_metrics(results_dir, exp_name)
        if m is None:
            print(f"  [WARN] Skipping {exp_name} (missing data)")
            continue
        names.append(label)
        memory.append(m["peak_memory_gb"])
        throughput.append(m["throughput_samples_per_sec"])
        step_time.append(m["avg_step_time_ms"])
        colors.append(METHOD_COLORS.get(label, "#999999"))

    if not names:
        print("  [WARN] No data for fig1 -- skipping.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(names))

    # Memory
    bars = axes[0].bar(x, memory, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_ylabel("Peak Memory (GB)")
    axes[0].set_title("Peak GPU Memory")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right")
    _add_bar_labels(axes[0], bars)

    # Throughput
    bars = axes[1].bar(x, throughput, color=colors, edgecolor="white", linewidth=0.5)
    axes[1].set_ylabel("Throughput (samples/sec)")
    axes[1].set_title("Training Throughput")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    _add_bar_labels(axes[1], bars)

    # Step time
    bars = axes[2].bar(x, step_time, color=colors, edgecolor="white", linewidth=0.5)
    axes[2].set_ylabel("Avg Step Time (ms)")
    axes[2].set_title("Average Step Time")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=30, ha="right")
    _add_bar_labels(axes[2], bars, fmt="{:.0f}")

    fig.suptitle("PEFT Method Comparison", fontsize=16, y=1.02)
    fig.tight_layout()
    _save_figure(fig, figures_dir, "fig1_method_comparison")


# Figure 2 -- Rank sweep (dual Y-axis)


def plot_fig2_rank_sweep(results_dir: Path, figures_dir: Path) -> None:
    """Dual-Y line plot of memory and throughput vs. LoRA rank."""
    print("Figure 2: rank sweep ...")

    ranks_target = [4, 8, 16, 32, 64, 128]
    ranks_found: list[int] = []
    mem_vals: list[float] = []
    thr_vals: list[float] = []

    for r in ranks_target:
        m = _load_metrics(results_dir, f"lora_r{r}")
        if m is None:
            print(f"  [WARN] Skipping rank {r} (missing data)")
            continue
        ranks_found.append(r)
        mem_vals.append(m["peak_memory_gb"])
        thr_vals.append(m["throughput_samples_per_sec"])

    if not ranks_found:
        print("  [WARN] No rank-sweep data -- skipping.")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    x = np.array(ranks_found)
    ax1.plot(x, mem_vals, "o-", color="#3A86FF", linewidth=2, markersize=8, label="Peak Memory (GB)")
    ax2.plot(x, thr_vals, "s--", color="#FF6B6B", linewidth=2, markersize=8, label="Throughput (samp/s)")

    ax1.set_xlabel("LoRA Rank")
    ax1.set_ylabel("Peak Memory (GB)", color="#3A86FF")
    ax2.set_ylabel("Throughput (samples/sec)", color="#FF6B6B")

    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.set_xticks(ranks_target)
    ax1.get_xaxis().set_major_formatter(ticker.FormatStrFormatter("%d"))

    ax1.tick_params(axis="y", labelcolor="#3A86FF")
    ax2.tick_params(axis="y", labelcolor="#FF6B6B")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Effect of LoRA Rank on Memory and Throughput")
    fig.tight_layout()
    _save_figure(fig, figures_dir, "fig2_rank_sweep")


# Figure 3 -- Kernel breakdown (horizontal stacked bar)


def plot_fig3_kernel_breakdown(results_dir: Path, figures_dir: Path) -> None:
    """Horizontal stacked bar chart of kernel time breakdown per method."""
    print("Figure 3: kernel breakdown ...")

    methods_cfg = [
        ("full_ft",   "Full FT"),
        ("lora_r16",  "LoRA"),
        ("qlora_r16", "QLoRA"),
        ("dora_r16",  "DoRA"),
    ]

    kernel_order = ["GEMM", "Attention", "Quantization", "Normalization",
                    "Optimizer", "Elementwise", "Other"]

    method_labels: list[str] = []
    pct_data: list[list[float]] = []  # rows=methods, cols=kernel categories

    for exp_name, label in methods_cfg:
        kc = _load_kernel_categories(results_dir, exp_name)
        if kc is None:
            print(f"  [WARN] Skipping kernel data for {exp_name}")
            continue
        method_labels.append(label)
        # kernel_categories.json has {"category_summary": [{"category": "GEMM", "percentage": ...}, ...]}
        cat_pcts = {entry["category"]: entry["percentage"]
                    for entry in kc.get("category_summary", [])}
        row = [cat_pcts.get(k, 0.0) for k in kernel_order]
        pct_data.append(row)

    if not method_labels:
        print("  [WARN] No kernel data -- skipping.")
        return

    data = np.array(pct_data)  # shape (n_methods, n_kernels)
    fig, ax = plt.subplots(figsize=(10, 4))
    y = np.arange(len(method_labels))
    bar_height = 0.5

    left = np.zeros(len(method_labels))
    for j, kernel in enumerate(kernel_order):
        color = KERNEL_COLORS.get(kernel, "#CCCCCC")
        bars = ax.barh(y, data[:, j], left=left, height=bar_height,
                       label=kernel, color=color, edgecolor="white", linewidth=0.5)
        # Annotate segments wider than 8 %
        for i, bar in enumerate(bars):
            w = bar.get_width()
            if w > 8:
                cx = bar.get_x() + w / 2
                ax.text(cx, bar.get_y() + bar.get_height() / 2,
                        f"{w:.0f}%", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
        left += data[:, j]

    ax.set_yticks(y)
    ax.set_yticklabels(method_labels)
    ax.set_xlabel("Percentage of GPU Time (%)")
    ax.set_title("Kernel Time Breakdown by Method")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9)
    ax.set_xlim(0, 100)

    fig.tight_layout()
    _save_figure(fig, figures_dir, "fig3_kernel_breakdown")


# Figure 4 -- System interactions (1x2 grouped bar)


def plot_fig4_system_interactions(results_dir: Path, figures_dir: Path) -> None:
    """Grouped bar chart showing effect of gradient checkpointing & flash attn."""
    print("Figure 4: system interactions ...")

    configs = [
        ("lora_r16",          "Base"),
        ("lora_r16_gc",       "+GradCkpt"),
        ("lora_r16_nofa",     "-FlashAttn"),
        ("lora_r16_gc_nofa",  "+GC-FA"),
    ]

    labels: list[str] = []
    mem_vals: list[float] = []
    thr_vals: list[float] = []

    for exp_name, label in configs:
        m = _load_metrics(results_dir, exp_name)
        if m is None:
            print(f"  [WARN] Skipping {exp_name}")
            continue
        labels.append(label)
        mem_vals.append(m["peak_memory_gb"])
        thr_vals.append(m["throughput_samples_per_sec"])

    if not labels:
        print("  [WARN] No system-interaction data -- skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(len(labels))
    width = 0.5
    color_mem = "#3A86FF"
    color_thr = "#FF6B6B"

    bars = axes[0].bar(x, mem_vals, width, color=color_mem, edgecolor="white")
    axes[0].set_ylabel("Peak Memory (GB)")
    axes[0].set_title("Memory: System Config Effects")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    _add_bar_labels(axes[0], bars)

    bars = axes[1].bar(x, thr_vals, width, color=color_thr, edgecolor="white")
    axes[1].set_ylabel("Throughput (samples/sec)")
    axes[1].set_title("Throughput: System Config Effects")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    _add_bar_labels(axes[1], bars)

    fig.suptitle("Effect of Gradient Checkpointing & Flash Attention", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_figure(fig, figures_dir, "fig4_system_interactions")


# Figure 5 -- Loss curves


def plot_fig5_loss_curves(results_dir: Path, figures_dir: Path) -> None:
    """Training loss curves for all Phase-1 methods."""
    print("Figure 5: loss curves ...")

    fig, ax = plt.subplots(figsize=(9, 5))
    any_plotted = False

    for exp_name, label in PHASE1_EXPERIMENTS:
        lc = _load_loss_curve(results_dir, exp_name)
        if lc is None:
            print(f"  [WARN] Skipping loss curve for {exp_name}")
            continue
        steps = [pt["step"] for pt in lc]
        losses = [pt["loss"] for pt in lc]
        color = METHOD_COLORS.get(label, "#999999")
        ax.plot(steps, losses, linewidth=1.8, label=label, color=color)
        any_plotted = True

    if not any_plotted:
        print("  [WARN] No loss-curve data -- skipping.")
        plt.close(fig)
        return

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_figure(fig, figures_dir, "fig5_loss_curves")


# Figure 6 -- Summary table


def plot_fig6_summary_table(results_dir: Path, figures_dir: Path) -> None:
    """Rendered summary table of key metrics for every Phase-1 method."""
    print("Figure 6: summary table ...")

    col_labels = ["Method", "Rank", "Memory (GB)", "Throughput\n(samp/s)",
                  "Step Time\n(ms)", "Final Loss", "Trainable %"]

    rows: list[list[str]] = []
    row_colors: list[str] = []

    for exp_name, label in PHASE1_EXPERIMENTS:
        m = _load_metrics(results_dir, exp_name)
        if m is None:
            print(f"  [WARN] Skipping {exp_name} in table")
            continue

        rank_str = str(m.get("rank", "-")) if m.get("rank") is not None else "-"
        rows.append([
            label,
            rank_str,
            f"{m['peak_memory_gb']:.1f}",
            f"{m['throughput_samples_per_sec']:.1f}",
            f"{m['avg_step_time_ms']:.1f}",
            f"{m['final_loss']:.3f}" if m.get('final_loss') is not None else "N/A",
            f"{m['trainable_pct']:.3f}",
        ])
        row_colors.append(METHOD_COLORS.get(label, "#CCCCCC"))

    if not rows:
        print("  [WARN] No data for summary table -- skipping.")
        return

    fig, ax = plt.subplots(figsize=(12, 1.0 + 0.5 * len(rows)))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)

    # Style header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2D4057")
        cell.set_text_props(color="white", fontweight="bold")

    # Tint data rows with a light wash of the method colour
    for i, color in enumerate(row_colors, start=1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            # Convert hex to RGBA with low alpha
            r = int(color[1:3], 16) / 255
            g = int(color[3:5], 16) / 255
            b = int(color[5:7], 16) / 255
            cell.set_facecolor((r, g, b, 0.12))

    ax.set_title("PEFT Benchmark Summary", fontsize=14, pad=20)
    fig.tight_layout()
    _save_figure(fig, figures_dir, "fig6_summary_table")


# CLI entry point


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate publication-quality PEFT benchmark figures.",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results"),
        help="Root directory containing experiment subdirectories (default: results/)",
    )
    parser.add_argument(
        "--figures_dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for generated figures (default: figures/)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point: parse args, apply style, generate all figures."""
    args = parse_args(argv)
    results_dir: Path = args.results_dir
    figures_dir: Path = args.figures_dir

    if not results_dir.is_dir():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    figures_dir.mkdir(parents=True, exist_ok=True)

    # Apply style -- fall back gracefully if the exact name is unavailable.
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            print("  [WARN] seaborn style not found; using default.")

    print(f"Results dir : {results_dir.resolve()}")
    print(f"Figures dir : {figures_dir.resolve()}")
    print()

    plot_fig1_method_comparison(results_dir, figures_dir)
    plot_fig2_rank_sweep(results_dir, figures_dir)
    plot_fig3_kernel_breakdown(results_dir, figures_dir)
    plot_fig4_system_interactions(results_dir, figures_dir)
    plot_fig5_loss_curves(results_dir, figures_dir)
    plot_fig6_summary_table(results_dir, figures_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
