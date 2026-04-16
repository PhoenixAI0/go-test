from __future__ import annotations

import argparse
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict

from go_rl.config import SftConfig, load_config, run_name_for_config
from go_rl.reasoning import (
    build_chat_template_kwargs,
    build_final_only_mask,
    detect_reasoning_profile,
)
from go_rl.logging_utils import setup_logging
from go_rl.run_logging import write_run_log
from go_rl.training_utils import (
    configure_torch_for_speed,
    default_num_workers,
    get_optimal_attention_backend,
    quantization_kwargs,
)
from go_rl.wandb_utils import (
    configure_wandb,
    finish_wandb,
    log_system_metrics,
    log_training_metrics,
)
from venv_utils import ensure_virtualenv

logger = logging.getLogger(__name__)


class _ChatMessage(TypedDict):
    role: str
    content: str


class _SftExample(TypedDict):
    prompt: str
    response: str


class _ReasoningSample(TypedDict):
    prompt: list[_ChatMessage]
    completion: list[_ChatMessage]
    raw_completion: str


class _MaskedSample(TypedDict):
    input_ids: list[int]
    completion_mask: list[int]


def _convert_to_messages(example: _SftExample, eos_token: str) -> _ReasoningSample:
    prompt = example["prompt"]
    response = example["response"]
    completion = f"{response}{eos_token}"
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "completion": [{"role": "assistant", "content": completion}],
        "raw_completion": response,
    }


def _add_reasoning_completion_mask(
    example: _ReasoningSample,
    tokenizer: object,
    profile: str,
    malformed_policy: str,
    reasoning_enabled: bool,
    optimize_final_only: bool,
    chat_template_kwargs: dict[str, object],
) -> _MaskedSample:
    prompt_messages = example["prompt"]
    completion_messages = example["completion"]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
        **chat_template_kwargs,
    )
    prompt_completion_processed = tokenizer.apply_chat_template(
        prompt_messages + completion_messages,
        tokenize=True,
        return_dict=True,
        **chat_template_kwargs,
    )
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    prompt_completion_ids = prompt_completion_processed["input_ids"]
    if prompt_completion_ids and isinstance(prompt_completion_ids[0], list):
        prompt_completion_ids = prompt_completion_ids[0]
    completion_ids = prompt_completion_ids[len(prompt_ids) :]
    if not reasoning_enabled or not optimize_final_only:
        completion_mask = [1] * len(completion_ids)
    else:
        completion_mask, malformed, _ = build_final_only_mask(
            completion=example["raw_completion"],
            completion_ids=completion_ids,
            tokenizer=tokenizer,
            profile=profile,
            malformed_policy=malformed_policy,
        )
        if malformed:
            logger.debug("SFT sample marked malformed reasoning boundary; masking completion")
    return {
        "input_ids": prompt_completion_ids,
        "completion_mask": [0] * len(prompt_ids) + completion_mask,
    }


def _safe_perplexity(loss_value: float) -> float:
    return math.exp(min(loss_value, 20.0))


def _log_perplexity_history(log_history: list[dict], wandb_enabled: bool) -> None:
    if not wandb_enabled:
        return
    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        metrics: dict[str, float] = {}
        if "loss" in entry:
            train_loss = float(entry["loss"])
            metrics["train/loss"] = train_loss
            metrics["train/perplexity"] = _safe_perplexity(train_loss)
        if "eval_loss" in entry:
            val_loss = float(entry["eval_loss"])
            metrics["val/loss"] = val_loss
            metrics["val/perplexity"] = _safe_perplexity(val_loss)
        if metrics:
            log_training_metrics(step=int(step), metrics=metrics, commit=False)


def main() -> None:
    ensure_virtualenv()
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
    run_name = run_name_for_config("sft", args.config)
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
        "eval_split_ratio": cfg.eval_split_ratio,
        "eval_steps": cfg.eval_steps,
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

    reasoning_profile = detect_reasoning_profile(
        model_name=cfg.model_name,
        tokenizer=tokenizer,
        configured_profile=cfg.reasoning.profile,
    )
    chat_template_kwargs = build_chat_template_kwargs(cfg.reasoning, reasoning_profile)
    logger.info("Using reasoning profile: %s", reasoning_profile)

    logger.info("Loading dataset from %s", cfg.dataset_path)
    dataset = load_dataset("json", data_files=cfg.dataset_path, split="train")
    if args.smoke:
        smoke_count = min(len(dataset), 32)
        logger.info("Smoke run enabled: using %d rows", smoke_count)
        dataset = dataset.select(range(smoke_count))

    eos_token = tokenizer.eos_token or ""
    dataset = dataset.map(
        _convert_to_messages,
        fn_kwargs={"eos_token": eos_token},
        remove_columns=[col for col in dataset.column_names if col not in {"prompt", "response"}],
    )
    dataset = dataset.map(
        _add_reasoning_completion_mask,
        fn_kwargs={
            "tokenizer": tokenizer,
            "profile": reasoning_profile,
            "malformed_policy": cfg.reasoning.malformed_policy,
            "reasoning_enabled": cfg.reasoning.enabled,
            "optimize_final_only": cfg.reasoning.optimize_final_only,
            "chat_template_kwargs": chat_template_kwargs,
        },
        remove_columns=["raw_completion", "prompt", "completion"],
    )

    train_dataset = dataset
    eval_dataset = None
    if cfg.eval_split_ratio > 0.0 and len(dataset) > 1:
        eval_size = max(1, int(round(len(dataset) * cfg.eval_split_ratio)))
        eval_size = min(eval_size, len(dataset) - 1)
        split = dataset.train_test_split(
            test_size=eval_size,
            seed=cfg.seed,
            shuffle=True,
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
    logger.info(
        "Dataset rows: total=%d train=%d eval=%d",
        len(dataset),
        len(train_dataset),
        len(eval_dataset) if eval_dataset is not None else 0,
    )
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = write_run_log(
        cfg.log_dir,
        run_name,
        args.config,
        asdict(cfg),
        extra={
            "smoke": args.smoke,
            "dataset_rows_total": len(dataset),
            "dataset_rows_train": len(train_dataset),
            "dataset_rows_eval": len(eval_dataset) if eval_dataset is not None else 0,
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
        "per_device_eval_batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "max_seq_length": cfg.max_seq_length,
        "logging_steps": 1 if args.smoke else 10,
        "max_steps": 1 if args.smoke else -1,
        "dataset_num_proc": num_workers,
        "dataloader_num_workers": num_workers,
        "dataloader_pin_memory": True,
        "completion_only_loss": True,
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
    if eval_dataset is not None:
        trainer_args["eval_strategy"] = "steps"
        trainer_args["eval_steps"] = 1 if args.smoke else cfg.eval_steps
    if wandb_enabled:
        trainer_args["report_to"] = ["wandb"]
        trainer_args["run_name"] = run_name
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(**trainer_args),
    )
    # Train with periodic GPU metrics logging
    logger.info("Starting training...")
    try:
        train_result = trainer.train()
        logger.info("Training completed successfully")
        _log_perplexity_history(trainer.state.log_history, wandb_enabled=wandb_enabled)
        if wandb_enabled:
            final_train_loss = train_result.metrics.get("train_loss")
            if final_train_loss is not None:
                log_training_metrics(
                    step=int(train_result.metrics.get("global_step", 0)),
                    metrics={
                        "train/final_loss": float(final_train_loss),
                        "train/final_perplexity": _safe_perplexity(float(final_train_loss)),
                    },
                )
        if eval_dataset is not None:
            eval_metrics = trainer.evaluate()
            logger.info("Final validation metrics: %s", eval_metrics)
            if wandb_enabled and "eval_loss" in eval_metrics:
                eval_loss = float(eval_metrics["eval_loss"])
                log_training_metrics(
                    step=int(train_result.metrics.get("global_step", 0)),
                    metrics={
                        "val/final_loss": eval_loss,
                        "val/final_perplexity": _safe_perplexity(eval_loss),
                    },
                )
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
