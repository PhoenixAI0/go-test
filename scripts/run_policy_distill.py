"""Bounded, reproducible Qwen policy/value distillation on 19x19 positions."""
from __future__ import annotations

import argparse
from itertools import chain
import json
import math
from pathlib import Path
import platform
import random
import time

import torch

from go_rl.data_prep import board_from_example, transform_teacher_example
from go_rl.experiment_utils import file_sha256, policy_baselines, validate_splits
from go_rl.indexed_jsonl import IndexedJsonlRecords, iter_augmented_records, write_record_cache
from go_rl.model_interface import choose_device, legal_action_mask, move_to_action, synchronize_device
from go_rl.policy_model import (
    collate_policy_inputs, create_policy_model, encode_policy_position,
    load_policy_model, policy_value_loss, save_policy_model,
)
from go_rl.rules import legality_superko
from go_rl.training_checkpoint import (
    preserve_best_snapshot, restore_best_snapshot, restore_training_state, save_training_state,
)


def prepare_records(records: list[dict], tokenizer, encoding: str, target: str,
                    max_seq_length: int, board_size: int) -> list[dict]:
    prepared = []
    for record in records:
        board = board_from_example(record)
        if board.size != board_size:
            raise ValueError("All records must match the checkpoint board size")
        rules, komi = record["rules"], float(record["komi"])
        legal = legal_action_mask(board, superko=legality_superko(rules))
        if "legal_mask" in record and record["legal_mask"] != legal:
            raise ValueError("Stored teacher legality differs from reconstructed state/rules")
        ids = encode_policy_position(board, tokenizer, encoding, rules=rules, komi=komi)
        if len(ids) > max_seq_length:
            raise ValueError(f"Position needs {len(ids)} tokens, exceeds {max_seq_length}; never truncate a board")
        action = move_to_action(record["response"], board_size)
        if not legal[action]:
            raise ValueError("Supervised move is illegal under the declared rules")
        if target == "policy":
            probabilities = record["policy_target"]
        elif target == "policy_argmax":
            soft = torch.tensor(record["policy_target"], dtype=torch.float32)
            if soft.shape != (board_size * board_size + 1,) or not torch.isfinite(soft).all() or (soft < 0).any():
                raise ValueError("policy_argmax requires a finite nonnegative policy_target")
            if not torch.isclose(soft.sum(), torch.tensor(1.0), atol=1e-4) or (soft[~torch.tensor(legal)] > 0).any():
                raise ValueError("policy_argmax requires a normalized legal policy_target")
            hard_action = int(soft.argmax())
            probabilities = [float(index == hard_action) for index in range(board_size * board_size + 1)]
        else:
            probabilities = [float(index == action) for index in range(board_size * board_size + 1)]
        prepared.append({"ids": ids, "legal": legal, "target": probabilities,
                         "action": action, "value": record.get("teacher_value"),
                         "game_id": record["game_id"]})
    return prepared


def make_batch(records: list[dict], tokenizer, device):
    inputs = collate_policy_inputs([record["ids"] for record in records], tokenizer, device)
    inputs["legal_mask"] = torch.tensor([record["legal"] for record in records], dtype=torch.bool, device=device)
    targets = torch.tensor([record["target"] for record in records], dtype=torch.float32, device=device)
    values = [record["value"] for record in records]
    if any(value is None for value in values) and not all(value is None for value in values):
        raise ValueError("A batch cannot mix labeled and unlabeled values")
    values = None if values[0] is None else torch.tensor(values, dtype=torch.float32, device=device)
    return inputs, targets, values


@torch.inference_mode()
def evaluate(model, tokenizer, records: list[dict], batch_size: int, value_weight: float) -> dict:
    was_training = model.training
    model.eval()
    totals = {"loss": 0., "policy_loss": 0., "value_loss": 0., "top1": 0., "top5": 0.,
              "target_probability_of_prediction": 0., "policy_kl": 0., "entropy": 0.}
    started = time.perf_counter()
    for offset in range(0, len(records), batch_size):
        chunk = records[offset:offset + batch_size]
        inputs, targets, values = make_batch(chunk, tokenizer, model.device)
        outputs = model(**inputs)
        loss, losses = policy_value_loss(outputs, targets, value_targets=values, value_weight=value_weight)
        probabilities = outputs.logits.softmax(-1)
        log_probs = outputs.logits.log_softmax(-1).masked_fill(~inputs["legal_mask"], 0)
        prediction = outputs.logits.argmax(-1)
        expected = torch.tensor([record["action"] for record in chunk], device=model.device)
        target_entropy = -(targets * targets.clamp_min(1e-30).log()).sum(-1).mean().item()
        metrics = {
            "loss": loss.item(), **losses,
            "top1": (prediction == expected).float().mean().item(),
            "top5": (outputs.logits.topk(min(5, outputs.logits.shape[-1]), -1).indices == expected[:, None]).any(-1).float().mean().item(),
            "target_probability_of_prediction": targets.gather(1, prediction[:, None]).mean().item(),
            "policy_kl": losses["policy_loss"] - target_entropy,
            "entropy": -(probabilities * log_probs).sum(-1).mean().item(),
        }
        for key, value in metrics.items():
            totals[key] += value * len(chunk)
    synchronize_device(model.device.type)
    model.train(was_training)
    return {**{key: value / len(records) for key, value in totals.items()},
            "positions": len(records), "seconds": time.perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--resume-from", help="Model+optimizer checkpoint from the same planned run; use a fresh output directory")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Save resumable snapshots every N steps; 0 disables periodic snapshots")
    parser.add_argument("--encoding", choices=["legacy", "grid", "compact", "stones", "cells"], default="compact")
    parser.add_argument("--target", choices=["policy", "policy_argmax", "move"], default="policy",
                        help="policy=soft target; policy_argmax=same target made onehot; move=searched best move")
    parser.add_argument("--readout", choices=["last", "cells", "cell_attention"], default="last")
    parser.add_argument("--attention", choices=["causal", "bidirectional"], default="causal")
    parser.add_argument("--head-init-std", type=float, default=.001)
    parser.add_argument("--symmetries", type=int, choices=[1, 8], default=1,
                        help="Training-only D4 augmentation, after game/state isolation")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--value-weight", type=float, default=.25)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--head-only", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.init_checkpoint and args.resume_from:
        parser.error("Choose init-checkpoint for a new experiment or resume-from for exact continuation")
    if args.checkpoint_every < 0:
        parser.error("checkpoint-every must be nonnegative")
    if args.init_checkpoint and args.head_only:
        parser.error("--init-checkpoint and --head-only cannot be combined; checkpoint trainability is preserved")
    if min(args.steps, args.batch_size, args.gradient_accumulation, args.eval_every) < 1:
        parser.error("steps, batch sizes, accumulation, and eval-every must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in (args.learning_rate, args.head_learning_rate)):
        parser.error("Learning rates must be finite and positive")
    if args.readout != "last" and args.encoding != "cells":
        parser.error("The cell readout requires --encoding cells")
    if any(not math.isfinite(value) or value < 0 for value in (args.head_init_std, args.value_weight)):
        parser.error("Head initialization scale and value weight must be finite and nonnegative")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is nonempty; use a fresh experiment path")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    train_records, eval_records = IndexedJsonlRecords(args.train_data), IndexedJsonlRecords(args.eval_data)
    validate_splits(train_records, eval_records)
    source_train_positions = len(train_records)
    if args.init_checkpoint or args.resume_from:
        model, tokenizer, metadata = load_policy_model(args.resume_from or args.init_checkpoint, device, trainable=True)
        if metadata.get("encoding") != args.encoding:
            parser.error("Checkpoint input encoding differs; start a fresh model for this ablation")
        if metadata.get("readout", "last") != args.readout or metadata.get("attention", "causal") != args.attention:
            parser.error("Checkpoint architecture differs; start a fresh model for this ablation")
    else:
        model, tokenizer, metadata = create_policy_model(
            args.model, device=device, revision=args.revision, lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha, train_backbone=not args.head_only,
            readout=args.readout, attention=args.attention, head_init_std=args.head_init_std,
        )
    rules = {json.dumps(record["rules"], sort_keys=True) for record in chain(train_records, eval_records)}
    komis = {record["komi"] for record in chain(train_records, eval_records)}
    metadata.update(encoding=args.encoding, rules=json.loads(next(iter(rules))) if len(rules) == 1 else None,
                    komi=next(iter(komis)) if len(komis) == 1 else None)
    augmented = iter_augmented_records(train_records, args.symmetries, transform_teacher_example)
    train = write_record_cache(
        (prepare_records([row], tokenizer, args.encoding, args.target, args.max_seq_length, model.board_size)[0]
         for row in augmented), output / "prepared_train.jsonl")
    validation = write_record_cache(
        (prepare_records([row], tokenizer, args.encoding, args.target, args.max_seq_length, model.board_size)[0]
         for row in eval_records), output / "prepared_validation.jsonl")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": [parameter for parameter in model.backbone.parameters() if parameter.requires_grad], "lr": args.learning_rate},
        {"params": list(model.head_parameters()), "lr": args.head_learning_rate},
    ], weight_decay=.01)
    warmup = max(1, min(10, args.steps // 10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup, 1.) * .5 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps)))
    rng = random.Random(args.seed)
    run = {"arguments": vars(args), "model": metadata, "platform": platform.platform(),
           "torch": torch.__version__, "device": device,
           "train_sha256": file_sha256(args.train_data),
           "eval_sha256": file_sha256(args.eval_data),
           "train_positions": len(train), "eval_positions": len(validation),
           "source_train_positions": source_train_positions,
           "prepared_data": {"storage": "indexed_jsonl", "train_bytes": train.path.stat().st_size,
                             "eval_bytes": validation.path.stat().st_size,
                             "train_sha256": file_sha256(train.path), "eval_sha256": file_sha256(validation.path),
                             "index_bytes": train.index_bytes + validation.index_bytes},
           "train_games": len({record['game_id'] for record in train}),
           "eval_games": len({record['game_id'] for record in validation}),
           "mean_prompt_tokens": sum(len(record['ids']) for record in train) / len(train),
           "trainable_parameters": sum(parameter.numel() for parameter in parameters),
           "non_model_baselines": policy_baselines(validation, train)}
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    contract = {name: getattr(args, name) for name in (
        "steps", "batch_size", "gradient_accumulation", "learning_rate", "head_learning_rate",
        "value_weight", "target", "encoding", "readout", "attention", "symmetries", "seed", "max_seq_length", "eval_every")}
    contract.update(train_sha256=run["train_sha256"], eval_sha256=run["eval_sha256"],
                    train_prepared_sha256=run["prepared_data"]["train_sha256"],
                    eval_prepared_sha256=run["prepared_data"]["eval_sha256"],
                    torch=str(torch.__version__), trainer_version=2)
    first_step = restore_training_state(args.resume_from, optimizer, scheduler, rng,
                                       contract=contract, device=device) if args.resume_from else 0
    if first_step >= args.steps:
        parser.error("Checkpoint already reached the planned steps; use init-checkpoint for a new training budget")
    started = time.perf_counter()
    baseline = evaluate(model, tokenizer, validation, args.batch_size, args.value_weight)
    best = baseline["policy_loss"]
    restored_best = bool(args.resume_from and restore_best_snapshot(args.resume_from, output / "best"))
    if restored_best:
        previous_best = json.loads((output / "best" / "go_policy.json").read_text())["validation"]["policy_loss"]
        best = min(best, previous_best)
    if not restored_best or baseline["policy_loss"] <= best:
        save_policy_model(model, tokenizer, output / "best", {**metadata, "step": first_step, "validation": baseline})
    history = [{"step": first_step, "eval": baseline, "elapsed_seconds": time.perf_counter() - started}]
    print(json.dumps(history[-1]), flush=True)
    with (output / "metrics.jsonl").open("w") as log:
        log.write(json.dumps(history[-1]) + "\n")
        model.train()
        for step in range(first_step + 1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.
            for _ in range(args.gradient_accumulation):
                chunk = [train[rng.randrange(len(train))] for _ in range(args.batch_size)]
                inputs, targets, values = make_batch(chunk, tokenizer, device)
                prediction = model(**inputs)
                loss, _ = policy_value_loss(prediction, targets, value_targets=values, value_weight=args.value_weight)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                (loss / args.gradient_accumulation).backward()
                step_loss += loss.detach().item() / args.gradient_accumulation
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Nonfinite gradient norm")
            optimizer.step()
            scheduler.step()
            entry = {"step": step, "train_loss": step_loss, "gradient_norm": gradient_norm.item(),
                     "elapsed_seconds": time.perf_counter() - started}
            if step % args.eval_every == 0 or step == args.steps:
                entry["eval"] = evaluate(model, tokenizer, validation, args.batch_size, args.value_weight)
                if entry["eval"]["policy_loss"] < best:
                    best = entry["eval"]["policy_loss"]
                    save_policy_model(model, tokenizer, output / "best", {**metadata, "step": step, "validation": entry["eval"]})
            history.append(entry)
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(json.dumps(entry), flush=True)
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                snapshot = output / f"checkpoint-{step:08d}"
                save_policy_model(model, tokenizer, snapshot, {**metadata, "step": step})
                preserve_best_snapshot(snapshot, output / "best")
                save_training_state(snapshot, optimizer, scheduler, rng, step=step, contract=contract, device=device)
    save_policy_model(model, tokenizer, output / "final", {**metadata, "step": args.steps, "validation": history[-1]["eval"]})
    preserve_best_snapshot(output / "final", output / "best")
    save_training_state(output / "final", optimizer, scheduler, rng, step=args.steps, contract=contract, device=device)
    (output / "summary.json").write_text(json.dumps({"baseline": baseline, "final": history[-1]["eval"],
        "non_model_baselines": run["non_model_baselines"],
        "best_validation_policy_loss": best, "seconds": time.perf_counter() - started,
        "best_scope": "full_run" if not args.resume_from or restored_best else "continuation_segment_only",
        "limitation": "Teacher imitation diagnostics, not match Elo or full-game strength."}, indent=2) + "\n")


if __name__ == "__main__":
    main()
