"""Train a PEFT (or full) fine-tuning run and record system metrics.

Typical usage:
    python src/train.py --experiment lora_r16 --output_dir results/
    python src/train.py --list
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import EXPERIMENTS, ExperimentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_model(cfg: ExperimentConfig):
    """Load the base model, optionally in 4-bit, and a tokenizer."""
    kwargs: dict = {
        "torch_dtype": torch.bfloat16 if cfg.bf16 else torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    # Fall back to the default attention implementation if flash-attn is not installed.
    if cfg.flash_attention:
        try:
            import flash_attn  # noqa: F401
            kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            log.warning("flash-attn not installed; using default sdpa attention")
            cfg.flash_attention = False

    if cfg.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        )

    log.info(
        "Loading model %s (4-bit=%s, flash_attn=%s)",
        cfg.model_name, cfg.load_in_4bit, cfg.flash_attention,
    )
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def apply_peft(model, cfg: ExperimentConfig):
    """Wrap the base model with the requested PEFT method."""
    if cfg.method == "full":
        total = sum(p.numel() for p in model.parameters())
        log.info("Full fine-tuning: %d trainable params", total)
        return model, None

    from peft import (
        AdaLoraConfig, LoraConfig, TaskType,
        get_peft_model, prepare_model_for_kbit_training,
    )

    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing,
        )

    use_dora = cfg.method in ("dora", "qdora")

    if cfg.method == "adalora":
        peft_config = AdaLoraConfig(
            init_r=cfg.lora_rank,
            target_r=max(4, cfg.lora_rank // 2),
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
            task_type=TaskType.CAUSAL_LM,
            total_step=cfg.max_steps,
        )
    else:
        peft_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
            use_dora=use_dora,
        )

    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(
        "PEFT applied: method=%s rank=%d dora=%s trainable=%s (%.4f%%)",
        cfg.method, cfg.lora_rank, use_dora, f"{trainable:,}", 100 * trainable / total,
    )
    model.print_trainable_parameters()

    return model, peft_config


ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "{input_section}"
    "### Response:\n{output}"
)
ALPACA_TEMPLATE_INPUT = "### Input:\n{input}\n\n"


def prepare_dataset(tokenizer, cfg: ExperimentConfig):
    """Load and tokenize Alpaca."""
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    log.info("Loaded Alpaca: %d examples", len(dataset))

    def _format_and_tokenize(example):
        input_section = (
            ALPACA_TEMPLATE_INPUT.format(input=example["input"])
            if example["input"] else ""
        )
        text = ALPACA_TEMPLATE.format(
            instruction=example["instruction"],
            input_section=input_section,
            output=example["output"],
        )
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=cfg.max_seq_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = dataset.map(
        _format_and_tokenize,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
        num_proc=4,
    )
    dataset.set_format("torch")
    return dataset


class SystemMetricsCallback(TrainerCallback):
    """Capture per-step wall-clock time and loss values."""

    def __init__(self):
        self.step_times: list[float] = []
        self.losses: list[dict] = []
        self._step_start: float | None = None

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self._step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_start is not None:
            torch.cuda.synchronize()
            self.step_times.append(time.perf_counter() - self._step_start)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append({"step": state.global_step, "loss": logs["loss"]})


def collect_metrics(cfg, model, cb, wall_start):
    """Gather all system-level metrics into a single dict."""
    effective_batch = cfg.batch_size * cfg.gradient_accumulation_steps
    avg_step = sum(cb.step_times) / len(cb.step_times) if cb.step_times else 0

    metrics: dict = {
        "method": cfg.method,
        "rank": cfg.lora_rank if cfg.method != "full" else "N/A",
        "load_in_4bit": cfg.load_in_4bit,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "flash_attention": cfg.flash_attention,
        "model": cfg.model_name,
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "peak_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
        "total_wall_sec": round(time.perf_counter() - wall_start, 1),
        "total_time_sec": round(time.perf_counter() - wall_start, 1),
        "avg_step_time_ms": round(avg_step * 1000, 2),
        "throughput_samples_per_sec": round(effective_batch / avg_step, 2) if avg_step else 0,
        "loss_history": cb.losses,
        "final_loss": cb.losses[-1]["loss"] if cb.losses else None,
    }

    total = sum(p.numel() for p in model.parameters())
    if cfg.method != "full":
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        metrics["trainable_params"] = trainable
        metrics["total_params"] = total
        metrics["trainable_pct"] = round(100 * trainable / total, 4)
    else:
        metrics["trainable_params"] = total
        metrics["total_params"] = total
        metrics["trainable_pct"] = 100.0

    if torch.cuda.is_available():
        metrics["gpu_name"] = torch.cuda.get_device_name(0)
        metrics["gpu_memory_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1,
        )

    return metrics


def save_results(metrics: dict, output_dir: str) -> None:
    """Write metrics.json and loss_curve.json."""
    os.makedirs(output_dir, exist_ok=True)

    loss_history = metrics.pop("loss_history", [])

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Saved metrics to %s", metrics_path)

    loss_path = os.path.join(output_dir, "loss_curve.json")
    with open(loss_path, "w") as f:
        json.dump(loss_history, f, indent=2)
    log.info("Saved loss curve to %s", loss_path)


def run_experiment(experiment_name: str, output_root: str) -> dict:
    cfg = EXPERIMENTS[experiment_name]
    exp_dir = os.path.join(output_root, experiment_name)

    log.info("EXPERIMENT: %s", experiment_name)
    log.info(
        "  method=%s  rank=%s  4bit=%s  gc=%s  fa=%s",
        cfg.method, cfg.lora_rank, cfg.load_in_4bit,
        cfg.gradient_checkpointing, cfg.flash_attention,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    wall_start = time.perf_counter()

    model, tokenizer = load_model(cfg)
    model, _ = apply_peft(model, cfg)

    # prepare_model_for_kbit_training already handles GC for quantized models
    if cfg.gradient_checkpointing and not cfg.load_in_4bit:
        model.gradient_checkpointing_enable()
        log.info("Gradient checkpointing enabled")

    dataset = prepare_dataset(tokenizer, cfg)

    training_args = TrainingArguments(
        output_dir=exp_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_steps=cfg.max_steps,
        warmup_steps=cfg.warmup_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_strategy="no",
        report_to="tensorboard",
        logging_dir=os.path.join(exp_dir, "tb_logs"),
        seed=cfg.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    cb = SystemMetricsCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[cb],
    )

    log.info("Starting training for %d steps...", cfg.max_steps)
    trainer.train()
    log.info("Training complete.")

    metrics = collect_metrics(cfg, model, cb, wall_start)
    save_results(metrics, exp_dir)

    log.info("RESULTS: %s", experiment_name)
    for k in ("peak_memory_gb", "avg_step_time_ms",
              "throughput_samples_per_sec", "final_loss", "trainable_pct"):
        log.info("  %-30s %s", k, metrics.get(k))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="PEFT systems benchmark")
    parser.add_argument("--experiment", type=str,
                        help="Experiment name from config.EXPERIMENTS")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Root output directory")
    parser.add_argument("--list", action="store_true",
                        help="List all available experiments")
    args = parser.parse_args()

    if args.list:
        print(f"\nAvailable experiments ({len(EXPERIMENTS)}):\n")
        header = f"{'Name':<22} {'Method':<8} {'Rank':<6} {'4bit':<6} {'GC':<6} {'FA':<6}"
        print(header)
        print("-" * len(header))
        for name, cfg in EXPERIMENTS.items():
            rank_str = str(cfg.lora_rank) if cfg.method != "full" else "all"
            print(
                f"{name:<22} {cfg.method:<8} {rank_str:<6} "
                f"{str(cfg.load_in_4bit):<6} "
                f"{str(cfg.gradient_checkpointing):<6} "
                f"{str(cfg.flash_attention):<6}"
            )
        return

    if not args.experiment:
        parser.error("--experiment is required (use --list to see options)")
    if args.experiment not in EXPERIMENTS:
        parser.error(f"Unknown experiment '{args.experiment}'. Use --list.")

    run_experiment(args.experiment, args.output_dir)


if __name__ == "__main__":
    main()
