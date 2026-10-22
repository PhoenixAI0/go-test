from __future__ import annotations

import argparse
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict, Mapping, Any

from go_rl.config import SftConfig, load_config, run_name_for_config
from go_rl.data_prep import split_dataset_by_game, board_from_example
from go_rl.reasoning import (
    build_chat_template_kwargs,
    build_final_only_mask,
    detect_reasoning_profile,
)
from go_rl.logging_utils import setup_logging
from go_rl.run_logging import write_run_log
from go_rl.training_utils import (
    add_common_training_args,
    configure_torch_for_speed,
    default_num_workers,
    get_optimal_attention_backend,
    load_training_model,
    precision_kwargs,
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


def _convert_to_messages(example: Mapping[str, Any], input_encoding: str = "legacy") -> _ReasoningSample:
    # The chat template supplies the assistant EOS exactly once.
    response = example["response"]
    prompt = example["prompt"]
    if input_encoding != "legacy":
        from go_rl.model_interface import render_position

        if example.get("komi") is None or not example.get("rules"):
            raise ValueError("Modern SFT inputs require explicit komi and rules metadata; "
                             "regenerate or relabel the dataset before training.")
        prompt = render_position(board_from_example(example), input_encoding,
                                 komi=float(example["komi"]), rules=example["rules"])
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "completion": [{"role": "assistant", "content": response}],
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
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
        **chat_template_kwargs,
    )
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    # Some chat templates insert an empty thinking block only in the generation
    # prefix. Encode the answer separately so the prompt boundary stays exact.
    completion_ids = tokenizer.encode(example["raw_completion"], add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        completion_ids.append(tokenizer.eos_token_id)
    prompt_completion_ids = prompt_ids + completion_ids
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


def _has_complete_supervision(example: _MaskedSample, max_seq_length: int) -> bool:
    """Reject partial move/EOS targets instead of teaching truncated answers."""
    return len(example["input_ids"]) <= max_seq_length and any(example["completion_mask"])


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
            log_training_metrics(step=int(step), metrics=metrics)


def main() -> None:
    ensure_virtualenv()
    parser = argparse.ArgumentParser()
    add_common_training_args(parser)
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.info("Loading SFT config from %s", args.config)
    cfg: SftConfig = load_config(args.config, SftConfig)
    logger.debug("Loaded SFT config: %s", asdict(cfg))
    try:
        from datasets import load_dataset
        from trl import SFTConfig, SFTTrainer
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
    quant_type = f"{cfg.quantization_bits}bit"
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
    model, tokenizer = load_training_model(
        cfg, attention_backend=attention_backend,
        gradient_checkpointing=args.gradient_checkpointing,
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
        if "split" in dataset.column_names:
            dataset = dataset.filter(lambda row: row["split"] == "train")
        smoke_count = min(len(dataset), 32)
        logger.info("Smoke run enabled: using %d rows", smoke_count)
        dataset = dataset.select(range(smoke_count))

    dataset = dataset.map(
        _convert_to_messages,
        fn_kwargs={"input_encoding": cfg.input_encoding},
        remove_columns=[col for col in dataset.column_names if col not in {"prompt", "response", "game_id", "split"}],
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
        remove_columns=[col for col in dataset.column_names if col not in {"game_id", "split"}],
    )

    # Keep the full board and complete answer, including its stopping token.
    dataset = dataset.filter(_has_complete_supervision, fn_kwargs={"max_seq_length": cfg.max_seq_length})
    if not len(dataset):
        raise ValueError("No SFT examples have supervised tokens within max_seq_length.")
    train_dataset, eval_dataset = split_dataset_by_game(dataset, cfg.eval_split_ratio, cfg.seed)
    # Provenance is for partitioning/audits, not a model input or loss target.
    metadata_columns = [col for col in ("game_id", "split") if col in dataset.column_names]
    train_dataset = train_dataset.remove_columns(metadata_columns)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.remove_columns(metadata_columns)
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
        packages=["datasets", "trl", "peft", "torch", "transformers"],
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
        "max_length": cfg.max_seq_length,
        "logging_steps": 1 if args.smoke else 10,
        "max_steps": 1 if args.smoke else -1,
        "dataset_num_proc": num_workers,
        "dataloader_num_workers": num_workers,
        "dataloader_pin_memory": True,
        "completion_only_loss": True,
        "seed": cfg.seed,
        "report_to": [],
        "torch_compile": args.enable_compile,
        **precision_kwargs(cfg.quantization_bits),
    }
    # Packing requires a compatible attention backend.
    if cfg.use_packing:
        trainer_args["packing"] = True
    # 8-bit optimizer for memory efficiency
    if cfg.use_8bit_optimizer:
        trainer_args["optim"] = "adamw_8bit"
    # Use non-reentrant PyTorch gradient checkpointing
    if args.gradient_checkpointing:
        trainer_args["gradient_checkpointing"] = True
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
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(cfg.output_dir)

    # Finish wandb run
    if wandb_enabled:
        finish_wandb()


if __name__ == "__main__":
    main()
