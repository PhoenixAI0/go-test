from __future__ import annotations

import argparse
import logging
import os
import random
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

from go_rl.board import BoardState
from go_rl.data_prep import board_from_example, state_record_from_board
from go_rl.reward_context import reward_position
from go_rl.config import (
    FullGameConfig,
    KataGoConfig,
    GRPOObjectiveConfig,
    ReasoningConfig,
    RewardConfig,
    RolloutConfig,
    TrainingConfig,
    load_yaml_mapping,
    resolve_config_base_dir,
    run_name_for_config,
)
from go_rl.fullgame_rollout import (
    align_board_to_player_turn,
    choose_model_color,
    simulate_full_game,
)
from go_rl.katago import KataGoResult
from go_rl.katago_pool import build_katago_service
from go_rl.logging_utils import setup_logging
from go_rl.prompting import parse_prompt
from go_rl.run_logging import write_run_log
from go_rl.shared_types import KataGoService
from go_rl.training_utils import (
    add_common_training_args,
    configure_torch_for_speed,
    default_num_workers,
    get_distributed_context,
    get_optimal_attention_backend,
    load_training_model,
    precision_kwargs,
    render_training_prompt,
)
from go_rl.wandb_utils import (
    configure_wandb,
    finish_wandb,
    log_prefixed_history,
    log_system_metrics,
)
from venv_utils import ensure_virtualenv

logger = logging.getLogger(__name__)


def _build_seed_board(prompt: str, full_cfg: FullGameConfig, rng: random.Random) -> BoardState:
    board = parse_prompt(prompt)
    if rng.random() < full_cfg.seed_empty_board_prob:
        return BoardState.empty(board.size)
    return board


def build_fullgame_reward_fn(
    model: object,
    tokenizer: object,
    katago: KataGoService,
    reward_cfg: RewardConfig,
    rollout_cfg: RolloutConfig,
    full_cfg: FullGameConfig,
    reasoning_cfg: ReasoningConfig | None = None,
    model_name: str = "",
    input_encoding: str = "compact",
    default_rules: str | dict[str, object] | None = None,
    default_komi: float | None = None,
) -> Callable[..., list[float]]:
    """Score root moves using continuations; only root completion tokens train."""
    state = {
        "episode_index": 0,
        "truncated": 0,
        "illegal": 0,
        "fallback_passes": 0,
    }
    analysis_cache: OrderedDict[tuple, KataGoResult] = OrderedDict()
    analysis_cache_max_size = 4096

    def reward_fn(prompts: list[str], completions: list[str], **kwargs: object) -> list[float]:
        # Score the raw board, not the chat-templated prompt the model saw.
        prompts = kwargs.get("board_prompt", prompts)
        if len(prompts) != len(completions):
            raise ValueError("Prompts and completions must have matching lengths.")
        rewards: list[float] = []
        for index, (prompt, completion) in enumerate(zip(prompts, completions)):
            position = reward_position(prompt, kwargs, index, len(prompts), reward_cfg,
                                       default_rules, default_komi)
            board = position.board

            # Score the exact position and side used to generate this completion.
            # Resampling here changes the task after the policy has acted.
            model_color = board.next_player()
            state["episode_index"] += 1
            result = simulate_full_game(
                board=board,
                initial_completion=completion,
                model=model,
                tokenizer=tokenizer,
                katago=katago,
                reward_cfg=position.reward_config,
                rollout_cfg=rollout_cfg,
                fullgame_cfg=full_cfg,
                model_color=model_color,
                reasoning_cfg=reasoning_cfg,
                reasoning_model_name=model_name,
                input_encoding=input_encoding,
                analysis_cache=analysis_cache,
                max_cache_size=analysis_cache_max_size,
                **position.analysis_options,
            )
            rewards.append(float(result.total_reward))
            if result.truncated:
                state["truncated"] += 1
            if result.terminated_by_illegal:
                state["illegal"] += 1
            state["fallback_passes"] += result.katago_fallback_passes

        total_episodes = state["episode_index"]
        if total_episodes and total_episodes % 50 == 0:
            logger.info(
                "Full-game rollout stats: episodes=%d truncated=%d illegal=%d fallback_passes=%d",
                total_episodes,
                state["truncated"],
                state["illegal"],
                state["fallback_passes"],
            )
        return rewards

    return reward_fn


def main() -> None:
    ensure_virtualenv()
    parser = argparse.ArgumentParser()
    add_common_training_args(parser)
    parser.add_argument(
        "--max-plies",
        type=int,
        help="Override full_game.max_plies from config.",
    )
    parser.add_argument(
        "--max-model-turns",
        type=int,
        help="Override full_game.max_model_turns from config.",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.info("Loading full-game GRPO config from %s", args.config)
    cfg = load_yaml_mapping(
        args.config,
        "PyYAML is required for GRPO config loading.",
    )
    config_base_dir = resolve_config_base_dir(args.config)

    train_cfg = TrainingConfig(**cfg["training"])
    katago_cfg = KataGoConfig(config_base_dir=config_base_dir, **cfg["katago"])
    reward_cfg = RewardConfig(**cfg["reward"])
    rollout_cfg = RolloutConfig(**cfg["rollout"])
    objective_cfg = GRPOObjectiveConfig(**(cfg.get("grpo") or {}))
    if rollout_cfg.num_samples < 2:
        raise ValueError("GRPO requires rollout.num_samples >= 2.")
    full_cfg = FullGameConfig(**(cfg.get("full_game") or {}))
    if args.max_plies is not None:
        full_cfg.max_plies = args.max_plies
    if args.max_model_turns is not None:
        full_cfg.max_model_turns = args.max_model_turns
    full_cfg.__post_init__()
    if not full_cfg.enabled:
        logger.warning("full_game.enabled=false in config; skipping full-game GRPO run.")
        return
    dataset_path = os.path.expanduser(os.path.expandvars(cfg["dataset"]))

    logger.debug("Training config: %s", asdict(train_cfg))
    logger.debug("KataGo config: %s", asdict(katago_cfg))
    logger.debug("Reward config: %s", asdict(reward_cfg))
    logger.debug("Rollout config: %s", asdict(rollout_cfg))
    logger.info("GRPO objective: %s", asdict(objective_cfg))
    if rollout_cfg.top_p != 1.0 or rollout_cfg.top_k != 0:
        logger.warning("Truncated sampling differs from the temperature-softmax policy "
                       "used for log probabilities; prefer top_p=1 and top_k=0 for RL.")
    logger.debug("Full-game config: %s", asdict(full_cfg))
    logger.debug("Dataset path: %s", dataset_path)

    logger.warning(
        "This trainer optimizes the initial completion only. Continuation moves are "
        "value-estimation rollouts, not full-trajectory policy-gradient samples. "
        "KataGo endpoint scores are estimates and never count as verified wins."
    )
    # One opponent query per turn, plus root/endpoint score estimates.
    logger.info(
        "Full-game cost: up to %d KataGo queries per sample (tune full_game.max_model_turns).",
        full_cfg.max_model_turns + 2,
    )

    try:
        from datasets import load_dataset
        from trl import GRPOConfig
    except ImportError as exc:
        raise RuntimeError("Missing dependencies for GRPO training.") from exc

    attention_backend = (
        args.attention_backend
        if args.attention_backend != "auto"
        else get_optimal_attention_backend()
    )
    configure_torch_for_speed(
        enable_compile=args.enable_compile,
        attention_backend=attention_backend,
    )

    run_name = run_name_for_config("grpo_fullgame", args.config)
    quant_type = f"{train_cfg.quantization_bits}bit"
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
        "num_samples": rollout_cfg.num_samples,
        "temperature": rollout_cfg.temperature,
        "reward_config": asdict(reward_cfg),
        "grpo_objective": asdict(objective_cfg),
        "full_game_config": asdict(full_cfg),
    }
    wandb_tags = ["grpo", "rl", "fullgame", quant_type]
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
        notes=f"Full-game GRPO RL training with {quant_type} on {train_cfg.model_name}",
    )

    logger.info("Loading model %s with %s quantization", train_cfg.model_name, quant_type)
    model, tokenizer = load_training_model(
        train_cfg, attention_backend=attention_backend,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    logger.info("Loading dataset from %s", dataset_path)
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    if args.smoke:
        smoke_count = min(len(dataset), 32)
        logger.info("Smoke run enabled: using %d rows", smoke_count)
        dataset = dataset.select(range(smoke_count))
    seed_rng = random.Random(train_cfg.seed)
    def prepare_seed(example, index):
        board = board_from_example(example)
        if seed_rng.random() < full_cfg.seed_empty_board_prob:
            board = BoardState.empty(board.size)
        color = choose_model_color(board, full_cfg, index)
        return state_record_from_board(align_board_to_player_turn(board, color))

    dataset = dataset.map(prepare_seed, with_indices=True, load_from_cache_file=False)
    dataset = dataset.map(
        render_training_prompt,
        fn_kwargs={"tokenizer": tokenizer, "reasoning_cfg": train_cfg.reasoning,
                   "model_name": train_cfg.model_name,
                   "input_encoding": train_cfg.input_encoding,
                   "rules": katago_cfg.rules, "komi": katago_cfg.komi},
    )
    logger.info("Dataset rows: %d", len(dataset))

    local_rank, world_size, is_main_process = get_distributed_context()

    if world_size > 1:
        logger.info("Multi-GPU training detected: rank %d of %d", local_rank, world_size)

    katago: Optional[KataGoService] = None
    try:
        # TRL invokes reward functions on each rank's local completions.
        katago = build_katago_service(katago_cfg)
        katago.start()

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
                "grpo": asdict(objective_cfg),
                "full_game": asdict(full_cfg),
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
            packages=["datasets", "trl", "peft", "torch", "transformers"],
        )
        logger.info("Wrote run log to %s", log_path)

        num_workers = default_num_workers()
        trainer_args = {
            "output_dir": train_cfg.output_dir,
            "max_prompt_length": train_cfg.max_seq_length - rollout_cfg.max_tokens,
            "num_train_epochs": 1,
            "per_device_train_batch_size": train_cfg.batch_size,
            "gradient_accumulation_steps": train_cfg.gradient_accumulation,
            "learning_rate": train_cfg.learning_rate,
            "num_generations": rollout_cfg.num_samples,
            "temperature": rollout_cfg.temperature,
            "top_p": rollout_cfg.top_p,
            "top_k": rollout_cfg.top_k,
            "max_completion_length": rollout_cfg.max_tokens,
            "logging_steps": 1 if args.smoke else 10,
            "dataloader_num_workers": num_workers,
            "dataloader_pin_memory": True,
            "seed": train_cfg.seed,
            "report_to": [],
            "torch_compile": args.enable_compile,
            **precision_kwargs(train_cfg.quantization_bits),
            **asdict(objective_cfg),
        }
        max_steps = 1 if args.smoke else train_cfg.num_steps
        trainer_args["max_steps"] = max_steps if max_steps > 0 else -1

        if train_cfg.use_8bit_optimizer:
            trainer_args["optim"] = "adamw_8bit"
        if args.gradient_checkpointing:
            trainer_args["gradient_checkpointing"] = True
            trainer_args["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        if wandb_enabled:
            trainer_args["report_to"] = ["wandb"]
            trainer_args["run_name"] = run_name

        from go_rl.reasoning_grpo_trainer import ReasoningGRPOTrainer

        trainer = ReasoningGRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=build_fullgame_reward_fn(
                model=model,
                tokenizer=tokenizer,
                katago=katago,
                reward_cfg=reward_cfg,
                rollout_cfg=rollout_cfg,
                full_cfg=full_cfg,
                reasoning_cfg=train_cfg.reasoning,
                model_name=train_cfg.model_name,
                input_encoding=train_cfg.input_encoding,
                default_rules=katago_cfg.rules, default_komi=katago_cfg.komi,
            ),
            train_dataset=dataset,
            args=GRPOConfig(**trainer_args),
            reasoning_config=train_cfg.reasoning,
            reasoning_model_name=train_cfg.model_name,
        )

        logger.info("Starting full-game GRPO training...")
        try:
            trainer.train()
            logger.info("Full-game GRPO training completed successfully")
            if is_main_process:
                log_prefixed_history(
                    trainer.state.log_history,
                    wandb_enabled,
                    "grpo_fullgame/",
                )
        finally:
            if wandb_enabled and is_main_process:
                log_system_metrics()

        logger.info("Saving model to %s", train_cfg.output_dir)
        trainer.save_model(train_cfg.output_dir)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(train_cfg.output_dir)

        if wandb_enabled and is_main_process:
            finish_wandb()

    finally:
        if katago is not None:
            logger.info("Stopping KataGo")
            katago.close()


if __name__ == "__main__":
    main()
