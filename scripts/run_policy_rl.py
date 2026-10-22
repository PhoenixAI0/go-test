#!/usr/bin/env python3
"""Bounded categorical self-play PPO; use larger explicit budgets for training.

Example (substitute the local KataGo paths):
  uv run python scripts/run_policy_rl.py --model outputs/policy_distill \
    --output outputs/policy_rl_smoke --max-plies 16 --updates 1 \
    --referee-command 'katago gtp -model NETWORK -config CONFIG'

The short default is a plumbing experiment, not a full-game strength benchmark.
Run actual completed games with --max-plies 1083 and a larger episode budget.
For a separate endgame experiment, --seed-data TRAIN_JSONL --min-seed-move 250
replays eligible training histories before sampling new model actions. Seeded
continuations do not measure full-game skill from an empty board.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from go_rl.gtp_match import GtpEngine
from go_rl.data_prep import board_from_example
from go_rl.experiment_utils import file_sha256, read_records, position_key
from go_rl.policy_model import load_policy_model, save_policy_model
from go_rl.policy_rollout import collect_selfplay_episode, optimize_selfplay_episodes
from go_rl.rules import normalize_rules
from go_rl.seed_positions import prepare_seed_records, sample_training_seed
from go_rl.training_checkpoint import model_fingerprints


def validate_run_destination(model_path: str, output_path: str) -> None:
    destination = Path(output_path)
    if Path(model_path).resolve() == destination.resolve():
        raise ValueError("Use a separate output directory to preserve the starting checkpoint")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Output directory is nonempty; use a fresh experiment path (resume is not implemented)")


def resolve_run_encoding(metadata: dict, requested: str | None) -> str:
    encoding = requested or metadata.get("encoding") or "compact"
    if metadata.get("readout", "last") in {"cells", "cell_attention"} and encoding != "cells":
        raise ValueError("Cell-based readout checkpoints require cells encoding; text overrides change the architecture's inputs")
    return encoding


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Categorical policy checkpoint, usually distilled first")
    parser.add_argument("--output", required=True)
    parser.add_argument("--referee-command", required=True, help="KataGo GTP command for legality and final_score")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--encoding", default=None, choices=("legacy", "grid", "compact", "stones", "cells"))
    parser.add_argument("--rules", default=None, help="Defaults to checkpoint rules, otherwise positional area/no-suicide")
    parser.add_argument("--komi", type=float, default=None)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--episodes-per-update", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=64)
    parser.add_argument("--seed-data", help="Pinned training states with complete histories; changes experiment to seeded self-play")
    parser.add_argument("--min-seed-move", type=int, default=250)
    parser.add_argument("--minibatch-size", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-kl", type=float, default=0.03, help="Stop additional PPO epochs beyond this sampled KL")
    parser.add_argument("--normalize-advantages", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("updates", "episodes_per_update", "max_plies", "minibatch_size", "ppo_epochs"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("learning_rate", "temperature", "max_grad_norm", "target_kl"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("gamma", "gae_lambda"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if not 0 <= args.clip_epsilon < 1:
        parser.error("--clip-epsilon must be in [0, 1)")
    for name in ("value_coefficient", "entropy_coefficient"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.komi is not None and not math.isfinite(args.komi):
        parser.error("--komi must be finite")
    if args.min_seed_move < 0:
        parser.error("--min-seed-move must be nonnegative")
    try:
        validate_run_destination(args.model, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, tokenizer, metadata = load_policy_model(args.model, args.device, trainable=True)
    encoding = resolve_run_encoding(metadata, args.encoding)
    rules = args.rules or metadata.get("rules") or "koPOSITIONALscoreAREAtaxNONEsui0"
    komi = args.komi if args.komi is not None else float(metadata.get("komi") if metadata.get("komi") is not None else 7.5)
    seed_records, seed_rejections = [], {}
    if args.seed_data:
        if metadata.get("rules") is None or metadata.get("komi") is None:
            raise ValueError("Seeded self-play requires a checkpoint with explicit rules and komi")
        if normalize_rules(metadata["rules"]) != normalize_rules(rules) or float(metadata["komi"]) != komi:
            raise ValueError("Seeded self-play cannot override the checkpoint objective")
        seed_records, seed_rejections = prepare_seed_records(
            read_records(args.seed_data), board_size=model.board_size,
            rules=rules, komi=komi, min_move=args.min_seed_move)
    seed_rng = random.Random(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    run_config = {**vars(args), "encoding": encoding, "rules": rules, "komi": komi,
                  "board_size": model.board_size,
                  "method": "categorical_selfplay_ppo", "outcome_source": "katago_gtp_final_score",
                  "truncation_target": "detached_next_player_value", "all_model_turns_optimized": True,
                  "start_distribution": "training_seed_positions" if seed_records else "empty_board",
                  "seed_sampling": "uniform_games_then_positions" if seed_records else None,
                  "seed_dataset_sha256": file_sha256(args.seed_data) if args.seed_data else None,
                  "eligible_seed_positions": len(seed_records), "seed_rejections": seed_rejections,
                  "starting_checkpoint": {"path": args.model, "fingerprint_observed_at_collection": True,
                                          "files_sha256": model_fingerprints(Path(args.model))}}
    (output / "rl_run.json").write_text(json.dumps(run_config, indent=2) + "\n")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=args.learning_rate, weight_decay=0.0)
    start = time.monotonic()
    with (output / "referee.log").open("w") as engine_log:
        referee = GtpEngine("rl_referee", args.referee_command, stderr=engine_log)
        try:
            with (output / "metrics.jsonl").open("w") as metrics_file, (output / "episodes.jsonl").open("w") as episode_file:
                for update in range(args.updates):
                    episodes = []
                    for index in range(args.episodes_per_update):
                        seed_record = sample_training_seed(seed_records, seed_rng) if seed_records else None
                        episode = collect_selfplay_episode(
                            model, tokenizer, referee, max_plies=args.max_plies, encoding=encoding,
                            rules=rules, komi=komi, temperature=args.temperature,
                            start_board=board_from_example(seed_record) if seed_record is not None else None,
                        )
                        episodes.append(episode)
                        summary = {"update": update, "episode": index, **episode.summary()}
                        if seed_record is not None:
                            summary.update(seed_game_id=seed_record["game_id"],
                                           seed_move_number=seed_record["move_number"],
                                           seed_position_key=position_key(seed_record))
                        episode_file.write(json.dumps(summary) + "\n")
                        episode_file.flush()
                        print(json.dumps({"event": "episode", **{k: v for k, v in summary.items()
                                                                 if k not in {"moves", "start_position"}}}), flush=True)
                    metrics = optimize_selfplay_episodes(
                        model, tokenizer, optimizer, episodes, minibatch_size=args.minibatch_size,
                        epochs=args.ppo_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda,
                        clip_epsilon=args.clip_epsilon, value_coefficient=args.value_coefficient,
                        entropy_coefficient=args.entropy_coefficient, max_grad_norm=args.max_grad_norm,
                        normalize_advantages=args.normalize_advantages, target_kl=args.target_kl,
                    )
                    metrics.update(update=update, elapsed_seconds=time.monotonic() - start,
                                   completed_games=sum(ep.black_outcome is not None for ep in episodes),
                                   truncated_games=sum(ep.black_outcome is None for ep in episodes))
                    if not metrics["completed_games"]:
                        print("No adjudicated game in this update; targets use learned value bootstraps at caps.", flush=True)
                    metrics_file.write(json.dumps(metrics) + "\n")
                    metrics_file.flush()
                    print(json.dumps({"event": "update", **metrics}), flush=True)
                    save_policy_model(model, tokenizer, output, {
                        **metadata, "encoding": encoding, "rules": rules, "komi": komi,
                        "training": run_config, "completed_updates": update + 1,
                    })
        finally:
            referee.close()


if __name__ == "__main__":
    main()
