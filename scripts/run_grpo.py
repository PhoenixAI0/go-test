from __future__ import annotations

import argparse
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from go_rl.board import BoardState, gtp_to_coords
from go_rl.config import KataGoConfig, RewardConfig, RolloutConfig, TrainingConfig
from go_rl.katago import KataGoAnalysisClient
from go_rl.logging_utils import setup_logging
from go_rl.prompting import parse_prompt
from go_rl.rewards import evaluate_move
from go_rl.rollout import normalize_move
from go_rl.run_logging import write_run_log
from go_rl.training_utils import (
    configure_torch_for_speed,
    default_num_workers,
    get_optimal_attention_backend,
    quantization_kwargs,
)
from go_rl.wandb_utils import configure_wandb, finish_wandb, log_system_metrics
from venv_utils import ensure_virtualenv

logger = logging.getLogger(__name__)


def load_yaml(path: str) -> Dict[str, object]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for GRPO config loading.") from exc
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping")
    return data


def build_reward_fn(
    katago: KataGoAnalysisClient,
    reward_cfg: RewardConfig,
    rollout_cfg: RolloutConfig,
) -> callable:
    def winrate_for_player(winrate: float, to_move: str, player: str) -> float:
        if to_move.upper() == player.upper():
            return winrate
        return 1.0 - winrate

    def reward_fn(
        prompts: List[str], completions: List[str], **_: object
    ) -> List[float]:
        # CRITICAL: In multi-GPU training, reward computation MUST only run on rank 0
        # TRL's GRPOTrainer handles reward computation on rank 0 and broadcasts to workers
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank != 0:
            raise RuntimeError(
                f"Reward function called on non-main process (rank {local_rank}). "
                "This should never happen - rewards must be computed on rank 0 only."
            )

        if katago is None:
            raise RuntimeError(
                "KataGo not initialized on rank 0. This is a configuration error. "
                "Ensure KataGo client is started before creating the reward function."
            )
        rewards: List[float] = []
        for prompt, completion in zip(prompts, completions):
            board = parse_prompt(prompt)
            player = board.next_player()
            base_analysis = katago.analyze_position(board)
            base_winrate = winrate_for_player(base_analysis.winrate, player, player)
            try:
                move = normalize_move(completion, stop_at_pass=rollout_cfg.stop_at_pass)
            except Exception:
                logger.debug(
                    "Failed to normalize completion: %r",
                    completion,
                    exc_info=True,
                )
                breakdown = evaluate_move(
                    board, completion, 0.0, reward_cfg, player=player
                )
                rewards.append(breakdown.total)
                continue
            try:
                coords = gtp_to_coords(move, board.size)
                valid_coords = True
            except Exception:
                logger.debug("Invalid move coords: %r", move, exc_info=True)
                coords = None
                valid_coords = False
            if not valid_coords:
                breakdown = evaluate_move(board, move, 0.0, reward_cfg, player=player)
                rewards.append(breakdown.total)
                continue
            if coords is None:
                is_legal = True
            elif reward_cfg.strict_legality:
                is_legal = board.is_strict_legal(player, coords)
            else:
                is_legal = board.is_basic_legal(coords)
            if not is_legal:
                breakdown = evaluate_move(board, move, 0.0, reward_cfg, player=player)
                rewards.append(breakdown.total)
                continue
            candidate = BoardState(size=board.size, moves=list(board.moves))
            try:
                candidate.apply_move(player, move)
                candidate_analysis = katago.analyze_position(candidate)
                candidate_winrate = winrate_for_player(
                    candidate_analysis.winrate,
                    candidate.next_player(),
                    player,
                )
                delta = candidate_winrate - base_winrate
                breakdown = evaluate_move(board, move, delta, reward_cfg, player=player)
                rewards.append(breakdown.total)
            except Exception:
                logger.debug("Candidate analysis failed", exc_info=True)
                breakdown = evaluate_move(board, move, 0.0, reward_cfg, player=player)
                rewards.append(breakdown.total)
        return rewards

    return reward_fn


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
        help="Use FP8 quantization (1.4x faster RL inference, 60%% less VRAM, 2x longer context)",
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
        help="Enable torch.compile for 10%% faster training + 2x faster compile time",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.info("Loading GRPO config from %s", args.config)
    cfg = load_yaml(args.config)
    train_cfg = TrainingConfig(**cfg["training"])
    katago_cfg = KataGoConfig(**cfg["katago"])
    reward_cfg = RewardConfig(**cfg["reward"])
    rollout_cfg = RolloutConfig(**cfg["rollout"])
    dataset_path = os.path.expanduser(os.path.expandvars(cfg["dataset"]))
    logger.debug("Training config: %s", asdict(train_cfg))
    logger.debug("KataGo config: %s", asdict(katago_cfg))
    logger.debug("Reward config: %s", asdict(reward_cfg))
    logger.debug("Rollout config: %s", asdict(rollout_cfg))
    logger.debug("Dataset path: %s", dataset_path)

    try:
        from datasets import load_dataset
        from trl import GRPOConfig, GRPOTrainer
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError("Missing dependencies for GRPO training.") from exc

    # Configure PyTorch for optimal performance (RL-specific optimizations)
    attention_backend = args.attention_backend if args.attention_backend != "auto" else get_optimal_attention_backend()
    configure_torch_for_speed(
        enable_compile=args.enable_compile,
        attention_backend=attention_backend,
    )

    # Prepare wandb configuration with hyperparameters
    run_name = f"grpo_{Path(args.config).stem}"
    quant_type = "fp8" if args.use_fp8 else "dynamic-4bit" if args.use_dynamic_4bit else f"{train_cfg.quantization_bits}bit"
    wandb_config = {
        "model_name": train_cfg.model_name,
        "max_seq_length": train_cfg.max_seq_length,
        "quantization": quant_type,
        "use_lora": train_cfg.use_lora,
        "lora_rank": train_cfg.lora_rank if train_cfg.use_lora else None,
        "lora_alpha": train_cfg.lora_alpha if train_cfg.use_lora else None,
        "learning_rate": train_cfg.learning_rate,
        "batch_size": train_cfg.batch_size,
        "gradient_accumulation": train_cfg.gradient_accumulation,
        "num_steps": train_cfg.num_steps,
        "use_8bit_optimizer": train_cfg.use_8bit_optimizer,
        "gradient_checkpointing": args.gradient_checkpointing,
        "attention_backend": attention_backend or "default",
        "torch_compile": args.enable_compile,
        "grpo_mini_batch": train_cfg.grpo_mini_batch,
        "grpo_logit_chunk_multiplier": train_cfg.grpo_logit_chunk_multiplier,
        "num_samples": rollout_cfg.num_samples,
        "temperature": rollout_cfg.temperature,
        "reward_config": asdict(reward_cfg),
    }
    wandb_tags = ["grpo", "rl", quant_type]
    if args.use_fp8:
        wandb_tags.extend(["fp8", "fp8-rl"])
    if args.enable_compile:
        wandb_tags.append("torch-compile")
    if train_cfg.grpo_mini_batch or train_cfg.grpo_logit_chunk_multiplier:
        wandb_tags.append("memory-optimized")

    wandb_enabled = configure_wandb(
        args.wandb,
        run_name,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        config=wandb_config,
        tags=wandb_tags,
        notes=f"GRPO RL training with {quant_type} quantization on {train_cfg.model_name}",
    )

    logger.info("Loading model %s with %s quantization", train_cfg.model_name, quant_type)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=train_cfg.model_name,
        max_seq_length=train_cfg.max_seq_length,
        dtype=None,  # Auto-select optimal dtype
        **quantization_kwargs(
            train_cfg.quantization_bits,
            use_dynamic=args.use_dynamic_4bit,
            use_fp8=args.use_fp8,
        ),
        fast_inference=True,  # Enable fast inference for RL rollouts
    )

    # Apply LoRA with optimized parameters (only if use_lora=True)
    if train_cfg.use_lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=train_cfg.lora_rank if train_cfg.lora_rank else 64,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=train_cfg.lora_alpha,
            lora_dropout=train_cfg.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth"
            if args.gradient_checkpointing
            else False,
            random_state=train_cfg.seed,
            use_rslora=False,
            loftq_config=None,
        )

    logger.info("Loading dataset from %s", dataset_path)
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    if args.smoke:
        smoke_count = min(len(dataset), 32)
        logger.info("Smoke run enabled: using %d rows", smoke_count)
        dataset = dataset.select(range(smoke_count))
    logger.info("Dataset rows: %d", len(dataset))

    # Check if we're in distributed training (multi-GPU)
    # CRITICAL: Only start KataGo on the main process (rank 0) to avoid:
    # - Resource contention (multiple KataGo instances competing)
    # - File system conflicts (simultaneous writes)
    # - Redundant computation (same analysis on multiple ranks)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main_process = local_rank == 0

    if world_size > 1:
        logger.info(
            "Multi-GPU training detected: rank %d of %d", local_rank, world_size
        )

    katago = None
    try:
        if is_main_process:
            katago = KataGoAnalysisClient(katago_cfg)
            logger.info(
                "Starting KataGo on main process (rank %d/%d)", local_rank, world_size
            )
            katago.start()
            logger.info("KataGo client initialized successfully on rank 0")
        else:
            logger.info(
                "Skipping KataGo on worker process (rank %d/%d)", local_rank, world_size
            )
            logger.info(
                "Reward computation will be handled by rank 0 and broadcasted"
            )
            # Note: In multi-GPU training, reward computation happens on main process
            # and rewards are broadcasted to other processes by the trainer

        Path(train_cfg.output_dir).mkdir(parents=True, exist_ok=True)
        log_path = write_run_log(
            train_cfg.log_dir,
            run_name,
            args.config,
            {
                "training": asdict(train_cfg),
                "katago": asdict(katago_cfg),
                "reward": asdict(reward_cfg),
                "rollout": asdict(rollout_cfg),
                "dataset": dataset_path,
            },
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
        logger.info("Starting GRPO training run %s", run_name)
        num_workers = default_num_workers()
        trainer_args = {
            "output_dir": train_cfg.output_dir,
            "max_seq_length": train_cfg.max_seq_length,
            "num_train_epochs": 1,
            "per_device_train_batch_size": train_cfg.batch_size,
            "gradient_accumulation_steps": train_cfg.gradient_accumulation,
            "learning_rate": train_cfg.learning_rate,
            "num_generations": rollout_cfg.num_samples,
            "temperature": rollout_cfg.temperature,
            "top_p": rollout_cfg.top_p,
            "max_completion_length": rollout_cfg.max_tokens,
            "logging_steps": 1 if args.smoke else 10,
            "dataloader_num_workers": num_workers,
            "dataloader_pin_memory": True,
        }
        max_steps = 1 if args.smoke else train_cfg.num_steps
        trainer_args["max_steps"] = max_steps if max_steps > 0 else -1
        # GRPO-specific Unsloth optimizations (saves ~5GB VRAM)
        if train_cfg.grpo_mini_batch is not None:
            trainer_args["unsloth_grpo_mini_batch"] = train_cfg.grpo_mini_batch
        if train_cfg.grpo_logit_chunk_multiplier is not None:
            trainer_args["unsloth_logit_chunk_multiplier"] = (
                train_cfg.grpo_logit_chunk_multiplier
            )
        # 8-bit optimizer for memory efficiency
        if train_cfg.use_8bit_optimizer:
            trainer_args["optim"] = "adamw_8bit"
        # Use Unsloth's optimized gradient checkpointing
        if args.gradient_checkpointing:
            trainer_args["gradient_checkpointing"] = "unsloth"
            trainer_args["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        if wandb_enabled:
            trainer_args["report_to"] = ["wandb"]
            trainer_args["run_name"] = run_name
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=build_reward_fn(katago, reward_cfg, rollout_cfg),
            train_dataset=dataset,
            args=GRPOConfig(**trainer_args),
        )

        # Train with periodic GPU metrics logging
        logger.info("Starting GRPO training...")
        try:
            trainer.train()
            logger.info("GRPO training completed successfully")
        finally:
            # Log final GPU metrics
            if wandb_enabled and is_main_process:
                log_system_metrics()

        logger.info("Saving model to %s", train_cfg.output_dir)
        trainer.save_model(train_cfg.output_dir)

        # Finish wandb run on main process
        if wandb_enabled and is_main_process:
            finish_wandb()

    finally:
        if is_main_process and katago is not None:
            logger.info("Stopping KataGo")
            katago.close()


if __name__ == "__main__":
    main()
