"""Experiment configurations for the PEFT benchmark.

Every experiment is a single `ExperimentConfig` instance stored in
`EXPERIMENTS`. The key is the experiment name used on the command line
(e.g. `python src/train.py --experiment lora_r16`).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """All knobs for one training run."""

    model_name: str = "microsoft/phi-2"

    # Method selector. Valid values:
    #   full     - full fine-tuning
    #   lora     - vanilla LoRA
    #   qlora    - LoRA + 4-bit NF4 base weights
    #   dora     - weight-decomposed LoRA
    #   qdora    - DoRA + 4-bit NF4
    #   adalora  - adaptive-rank LoRA
    method: str = "lora"

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "dense",
        "fc1", "fc2",
    ])

    # Quantization (only used when method is qlora or qdora)
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_steps: int = 500
    warmup_steps: int = 50
    max_seq_length: int = 512
    weight_decay: float = 0.01

    # System optimizations
    gradient_checkpointing: bool = False
    flash_attention: bool = True
    bf16: bool = True

    # Logging
    logging_steps: int = 10
    output_dir: str = "./results"
    seed: int = 42


def _build_experiments() -> dict[str, ExperimentConfig]:
    exps: dict[str, ExperimentConfig] = {}

    # Phase 1: method comparison
    exps["full_ft"] = ExperimentConfig(
        method="full",
        learning_rate=2e-5,
        batch_size=1,
        gradient_accumulation_steps=16,
    )
    exps["lora_r16"] = ExperimentConfig(method="lora", lora_rank=16, lora_alpha=32)
    exps["qlora_r16"] = ExperimentConfig(
        method="qlora", lora_rank=16, lora_alpha=32, load_in_4bit=True,
    )
    exps["dora_r16"] = ExperimentConfig(
        method="dora", lora_rank=16, lora_alpha=32, learning_rate=1e-4,
    )
    exps["qdora_r16"] = ExperimentConfig(
        method="qdora", lora_rank=16, lora_alpha=32,
        load_in_4bit=True, learning_rate=1e-4,
    )
    exps["adalora_r16"] = ExperimentConfig(method="adalora", lora_rank=16, lora_alpha=32)

    # Phase 2: rank sensitivity (r=16 is reused from Phase 1)
    for r in (4, 8, 32, 64, 128):
        exps[f"lora_r{r}"] = ExperimentConfig(
            method="lora", lora_rank=r, lora_alpha=2 * r,
        )

    # Phase 3: gradient checkpointing interactions
    exps["lora_r16_gc"] = ExperimentConfig(
        method="lora", lora_rank=16, lora_alpha=32,
        gradient_checkpointing=True,
    )
    exps["dora_r16_gc"] = ExperimentConfig(
        method="dora", lora_rank=16, lora_alpha=32,
        learning_rate=1e-4,
        gradient_checkpointing=True,
    )
    exps["qlora_r16_gc"] = ExperimentConfig(
        method="qlora", lora_rank=16, lora_alpha=32,
        load_in_4bit=True, gradient_checkpointing=True,
    )
    exps["full_ft_gc"] = ExperimentConfig(
        method="full",
        learning_rate=2e-5,
        batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
    )

    return exps


EXPERIMENTS: dict[str, ExperimentConfig] = _build_experiments()
