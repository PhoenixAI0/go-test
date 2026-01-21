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
from go_rl.wandb_utils import configure_wandb

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
) -> callable:
    def winrate_for_player(
        winrate: float, to_move: str, player: str
    ) -> float:
        if to_move.upper() == player.upper():
            return winrate
        return 1.0 - winrate

    def reward_fn(
        prompts: List[str], completions: List[str], **_: object
    ) -> List[float]:
        # In multi-GPU training, this should only be called on main process
        # TRL's GRPOTrainer handles reward computation on rank 0 and broadcasts
        if katago is None:
            raise RuntimeError(
                "KataGo not initialized. In multi-GPU training, "
                "this should only happen on non-main processes."
            )
        rewards: List[float] = []
        for prompt, completion in zip(prompts, completions):
            board = parse_prompt(prompt)
            player = board.next_player()
            base_analysis = katago.analyze_position(board)
            base_winrate = winrate_for_player(
                base_analysis.winrate, player, player
            )
            move = normalize_move(completion)
            try:
                coords = gtp_to_coords(move, board.size)
                valid_coords = True
            except Exception:
                coords = None
                valid_coords = False
            if not valid_coords:
                breakdown = evaluate_move(
                    board, move, 0.0, reward_cfg, player=player
                )
                rewards.append(breakdown.total)
                continue
            if coords is None:
                is_legal = True
            elif reward_cfg.strict_legality:
                is_legal = board.is_strict_legal(player, coords)
            else:
                is_legal = board.is_basic_legal(coords)
            if not is_legal:
                breakdown = evaluate_move(
                    board, move, 0.0, reward_cfg, player=player
                )
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
                breakdown = evaluate_move(
                    board, move, delta, reward_cfg, player=player
                )
                rewards.append(breakdown.total)
            except Exception:
                breakdown = evaluate_move(
                    board, move, 0.0, reward_cfg, player=player
                )
                rewards.append(breakdown.total)
        return rewards

    return reward_fn


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

    run_name = f"grpo_{Path(args.config).stem}"
    wandb_enabled = configure_wandb(
        args.wandb,
        run_name,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
    )

    logger.info("Loading model %s", train_cfg.model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=train_cfg.model_name,
        max_seq_length=train_cfg.max_seq_length,
        dtype=None,
        load_in_4bit=train_cfg.use_lora,
        fast_inference=True,
    )

    # Apply LoRA with optimized parameters (only if use_lora=True)
    if train_cfg.use_lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=train_cfg.lora_rank if train_cfg.lora_rank else 64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=train_cfg.lora_alpha,
            lora_dropout=train_cfg.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth" if args.gradient_checkpointing else False,
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
    # Only start KataGo on the main process (rank 0) to avoid conflicts
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = local_rank == 0

    if is_main_process:
        katago = KataGoAnalysisClient(katago_cfg)
        logger.info("Starting KataGo on main process (rank %d)", local_rank)
        katago.start()
    else:
        logger.info("Skipping KataGo on worker process (rank %d)", local_rank)
        # Note: In multi-GPU training, reward computation happens on main process
        # and rewards are broadcasted to other processes by the trainer
        katago = None

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
    }
    # GRPO-specific Unsloth optimizations (saves ~5GB VRAM)
    if train_cfg.grpo_mini_batch is not None:
        trainer_args["unsloth_grpo_mini_batch"] = train_cfg.grpo_mini_batch
    if train_cfg.grpo_logit_chunk_multiplier is not None:
        trainer_args["unsloth_logit_chunk_multiplier"] = train_cfg.grpo_logit_chunk_multiplier
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
        reward_funcs=build_reward_fn(katago, reward_cfg),
        train_dataset=dataset,
        args=GRPOConfig(**trainer_args),
    )

    trainer.train()
    logger.info("Saving model to %s", train_cfg.output_dir)
    trainer.save_model(train_cfg.output_dir)

    if is_main_process and katago is not None:
        logger.info("Stopping KataGo")
        katago.close()


if __name__ == "__main__":
    main()
