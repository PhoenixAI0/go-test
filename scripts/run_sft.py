from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from go_rl.config import SftConfig, load_config
from go_rl.logging_utils import setup_logging
from go_rl.run_logging import write_run_log
from go_rl.training_utils import (
    configure_torch_for_speed,
    default_num_workers,
    get_optimal_attention_backend,
    quantization_kwargs,
)
from go_rl.wandb_utils import configure_wandb, finish_wandb, log_system_metrics

logger = logging.getLogger(__name__)


def format_example(example: Dict[str, str]) -> str:
    return f"{example['prompt']}\n{example['response']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases reporting.",
    )
    parser.add_argument("--wandb-project", help="Weights & Biases project name.")
    parser.add_argument("--wandb-entity", help="Weights & Biases entity/team.")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing for memory efficiency",
    )
    parser.add_argument(
        "--use-fp8",
        action="store_true",
        help="Use FP8 quantization (1.4x faster, 60%% less VRAM, requires RTX 40/50/H100)",
    )
    parser.add_argument(
        "--use-dynamic-4bit",
        action="store_true",
        help="Use Unsloth Dynamic 4-bit quantization (best accuracy, <10%% more VRAM than BnB)",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["sdpa", "flash_attention_2", "xformers", "auto"],
        default="auto",
        help="Attention backend (auto=SDPA on new GPUs, FA2 on old GPUs)",
    )
    parser.add_argument(
        "--enable-compile",
        action="store_true",
        help="Enable torch.compile for 10%% faster training",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.info("Loading SFT config from %s", args.config)
    cfg: SftConfig = load_config(args.config, SftConfig)
    logger.debug("Loaded SFT config: %s", asdict(cfg))
    try:
        from datasets import load_dataset
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError("Missing dependencies for SFT training.") from exc

    # Configure PyTorch for optimal performance
    attention_backend = args.attention_backend if args.attention_backend != "auto" else get_optimal_attention_backend()
    configure_torch_for_speed(
        enable_compile=args.enable_compile,
        attention_backend=attention_backend,
    )

    # Prepare wandb configuration with hyperparameters
    run_name = f"sft_{Path(args.config).stem}"
    quant_type = "fp8" if args.use_fp8 else "dynamic-4bit" if args.use_dynamic_4bit else f"{cfg.quantization_bits}bit"
    wandb_config = {
        "model_name": cfg.model_name,
        "max_seq_length": cfg.max_seq_length,
        "quantization": quant_type,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "use_packing": cfg.use_packing,
        "use_8bit_optimizer": cfg.use_8bit_optimizer,
        "gradient_checkpointing": args.gradient_checkpointing,
        "attention_backend": attention_backend or "default",
        "torch_compile": args.enable_compile,
    }
    wandb_tags = ["sft", quant_type]
    if args.use_fp8:
        wandb_tags.append("fp8")
    if args.enable_compile:
        wandb_tags.append("torch-compile")

    wandb_enabled = configure_wandb(
        args.wandb,
        run_name,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        config=wandb_config,
        tags=wandb_tags,
        notes=f"SFT training with {quant_type} quantization on {cfg.model_name}",
    )

    logger.info("Loading model %s with %s quantization", cfg.model_name, quant_type)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=None,  # Auto-select optimal dtype
        **quantization_kwargs(
            cfg.quantization_bits,
            use_dynamic=args.use_dynamic_4bit,
            use_fp8=args.use_fp8,
        ),
    )

    # Apply LoRA with optimized parameters
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth" if args.gradient_checkpointing else False,
        random_state=cfg.seed,
        use_rslora=False,
        loftq_config=None,
    )

    logger.info("Loading dataset from %s", cfg.dataset_path)
    dataset = load_dataset("json", data_files=cfg.dataset_path, split="train")
    if args.smoke:
        smoke_count = min(len(dataset), 32)
        logger.info("Smoke run enabled: using %d rows", smoke_count)
        dataset = dataset.select(range(smoke_count))
    logger.info("Dataset rows: %d", len(dataset))
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = write_run_log(
        cfg.log_dir,
        run_name,
        args.config,
        asdict(cfg),
        extra={
            "smoke": args.smoke,
            "dataset_rows": len(dataset),
            "wandb": {
                "enabled": wandb_enabled,
                "project": args.wandb_project,
                "entity": args.wandb_entity,
                "mode": args.wandb_mode,
            },
        },
        packages=["datasets", "trl", "unsloth", "torch", "transformers"],
    )
    logger.info("Wrote run log to %s", log_path)
    logger.info("Starting SFT training run %s", run_name)
    num_workers = default_num_workers()
    trainer_args = {
        "output_dir": cfg.output_dir,
        "num_train_epochs": 1 if args.smoke else cfg.epochs,
        "per_device_train_batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "max_seq_length": cfg.max_seq_length,
        "logging_steps": 1 if args.smoke else 10,
        "max_steps": 1 if args.smoke else -1,
        "dataset_num_proc": num_workers,
        "dataloader_num_workers": num_workers,
        "dataloader_pin_memory": True,
    }
    # Packing optimization: 3-5x speedup
    if cfg.use_packing:
        trainer_args["packing"] = True
    # 8-bit optimizer for memory efficiency
    if cfg.use_8bit_optimizer:
        trainer_args["optim"] = "adamw_8bit"
    # Use Unsloth's optimized gradient checkpointing
    if args.gradient_checkpointing:
        trainer_args["gradient_checkpointing"] = "unsloth"
        trainer_args["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if wandb_enabled:
        trainer_args["report_to"] = ["wandb"]
        trainer_args["run_name"] = run_name
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=format_example,
        args=SFTConfig(**trainer_args),
    )
    # Train with periodic GPU metrics logging
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully")
    finally:
        # Log final GPU metrics
        if wandb_enabled:
            log_system_metrics()

    logger.info("Saving model to %s", cfg.output_dir)
    trainer.save_model(cfg.output_dir)

    # Finish wandb run
    if wandb_enabled:
        finish_wandb()


if __name__ == "__main__":
    main()
