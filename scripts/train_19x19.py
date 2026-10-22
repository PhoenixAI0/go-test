#!/usr/bin/env python3
"""Run direct 19x19 teacher → Qwen text/categorical policy → evaluation.

All configured paths are relative to the repository root. Dry runs import no
model libraries, start no engines, download nothing, and create no output files.
Text SFT is a provisional small-sample choice; categorical heads remain an ablation.
Optional trajectory PPO requires the categorical policy architecture.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import yaml

from go_rl.rules import legality_superko, normalize_rules

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/policy_19x19_qwen06.yaml"


def file_hash(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def resolve_path(value: str) -> str:
    expanded = os.path.expanduser(os.path.expandvars(value))
    if "$" in expanded:  # Retain readable placeholders in a dry-run plan.
        return expanded
    path = Path(expanded)
    return str((path if path.is_absolute() else ROOT / path).resolve())


def load_recipe(args) -> dict:
    recipe = yaml.safe_load(Path(args.config).read_text())
    if not isinstance(recipe, dict):
        raise ValueError("Recipe must be a YAML mapping")
    recipe = deepcopy(recipe)
    recipe["evaluation"].setdefault("text_num_beams", 1)
    for section in ("selfplay", "evaluation"):
        if not isinstance(recipe[section]["enabled"], bool):
            raise ValueError(f"{section}.enabled must be true or false")
    for key in ("dataset", "train_data", "eval_data", "output_dir", "device", "train_kind",
                "resume_from", "init_checkpoint"):
        if getattr(args, key, None) is not None:
            recipe[key] = getattr(args, key)
    if args.selfplay:
        recipe["selfplay"]["enabled"] = True
    if args.evaluate:
        recipe["evaluation"]["enabled"] = True
    if recipe["train_kind"] not in {"text", "categorical"}:
        raise ValueError("train_kind must be text or categorical")
    if getattr(args, "encoding", None):
        recipe[recipe["train_kind"] + "_training"]["encoding"] = args.encoding
    if recipe["train_kind"] == "text" and recipe["selfplay"]["enabled"]:
        raise ValueError("Categorical self-play PPO cannot train a text model; choose --train-kind categorical. Text GRPO is a separate workflow.")
    if args.smoke:
        recipe["teacher"].update(train_positions=8, eval_positions=4,
                                 positions_per_game=1, max_visits=1, batch_size=2)
        recipe["training"].update(steps=2, batch_size=1, gradient_accumulation=1, eval_every=1,
                                  checkpoint_every=1)
        recipe["text_training"].update(generation_positions=2)
        recipe["selfplay"].update(updates=1, episodes_per_update=1, max_plies=8,
                                  minibatch_size=2, ppo_epochs=1)
        recipe["evaluation"].update(max_positions=4, max_visits=1, num_games=2,
                                    max_moves=12, main_time=60, byoyomi_time=10)
    if recipe.get("board_size") != 19:
        raise ValueError("This recipe requires board_size: 19; no smaller-board curriculum is used")
    if not recipe.get("model") or not recipe.get("revision"):
        raise ValueError("Specify a model and pinned revision")
    if not math.isfinite(float(recipe["komi"])):
        raise ValueError("komi must be finite")
    legality_superko(recipe["rules"])
    # A JSON rules argument preserves every normalized field at CLI boundaries.
    recipe["rules"] = json.dumps(normalize_rules(recipe["rules"]), sort_keys=True, separators=(",", ":"))
    if bool(recipe.get("train_data")) != bool(recipe.get("eval_data")):
        raise ValueError("Supply both train_data and eval_data, or neither")
    if recipe.get("resume_from") and recipe.get("init_checkpoint"):
        raise ValueError("resume_from and init_checkpoint are mutually exclusive")
    if recipe.get("resume_from") and not recipe.get("train_data"):
        raise ValueError("Resume requires explicit existing train_data and eval_data; never regenerate teacher labels during resume")
    if not recipe.get("train_data") and not recipe.get("dataset"):
        raise ValueError("Teacher generation requires dataset")
    if recipe["device"] not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("Unsupported device")
    categorical = recipe["categorical_training"]
    for key, allowed in {
        "encoding": {"legacy", "grid", "compact", "stones", "cells"},
        "readout": {"last", "cells"}, "attention": {"causal", "bidirectional"},
        "target": {"policy", "policy_argmax", "move"}, "symmetries": {1, 8},
    }.items():
        if categorical[key] not in allowed:
            raise ValueError(f"Unsupported categorical_training.{key}: {categorical[key]}")
    if categorical["readout"] == "cells" and categorical["encoding"] != "cells":
        raise ValueError("Cell readout requires cells encoding")
    if recipe["text_training"]["encoding"] not in {"legacy", "grid", "compact", "stones"}:
        raise ValueError("Text generation requires legacy, grid, compact or stones encoding")
    if recipe["text_training"]["symmetries"] not in {1, 8}:
        raise ValueError("text_training.symmetries must be 1 or 8")
    for section, fields in {
        "teacher": ("train_positions", "eval_positions", "positions_per_game", "max_visits", "batch_size"),
        "training": ("steps", "batch_size", "gradient_accumulation", "eval_every", "checkpoint_every",
                     "lora_rank", "max_seq_length"),
        "text_training": ("generation_positions", "max_new_tokens"),
        "selfplay": ("updates", "episodes_per_update", "max_plies", "minibatch_size", "ppo_epochs"),
        "evaluation": ("max_positions", "max_visits", "num_games", "max_moves", "text_num_beams"),
    }.items():
        for field in fields:
            value = recipe[section][field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{section}.{field} must be a positive integer")
    for section, fields in {
        "training": ("learning_rate",),
        "categorical_training": ("head_learning_rate",),
        "selfplay": ("learning_rate",),
    }.items():
        for field in fields:
            value = float(recipe[section][field])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{section}.{field} must be finite and positive")
    for field in ("head_init_std", "value_weight"):
        value = float(categorical[field])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"categorical_training.{field} must be finite and nonnegative")
    teacher = recipe["teacher"]
    if not 0 <= float(teacher["prior_mix"]) <= 1:
        raise ValueError("teacher.prior_mix must be between zero and one")
    if teacher["sampling"] not in {"game", "game-phase"}:
        raise ValueError("teacher.sampling must be game or game-phase")
    if (not isinstance(teacher["min_move"], int) or teacher["min_move"] < 0
            or (teacher["max_move"] is not None
                and (not isinstance(teacher["max_move"], int) or teacher["max_move"] < teacher["min_move"]))):
        raise ValueError("Teacher move window must contain ordered nonnegative integers")
    evaluation = recipe["evaluation"]
    for field in ("main_time", "byoyomi_time", "byoyomi_periods"):
        if not math.isfinite(float(evaluation[field])) or evaluation[field] < 0:
            raise ValueError(f"evaluation.{field} must be finite and nonnegative")
    if evaluation["main_time"] + evaluation["byoyomi_time"] * evaluation["byoyomi_periods"] <= 0:
        raise ValueError("Evaluation time controls must allow thinking time")
    if evaluation["enabled"] and evaluation["num_games"] % 2:
        raise ValueError("Paired-color evaluation requires an even number of games")
    for key in ("dataset", "train_data", "eval_data", "output_dir", "resume_from", "init_checkpoint"):
        if recipe.get(key):
            recipe[key] = resolve_path(recipe[key])
    for key, value in recipe["katago"].items():
        expanded = os.path.expanduser(os.path.expandvars(value))
        if key == "binary" and "$" not in expanded and "/" not in expanded:
            recipe["katago"][key] = shutil.which(expanded) or expanded
        else:
            recipe["katago"][key] = resolve_path(value)
    recipe["smoke"] = args.smoke
    return recipe


def command(script: str, **options) -> list[str]:
    argv = [sys.executable, str(ROOT / "scripts" / script)]
    for key, value in options.items():
        if isinstance(value, bool):
            if value:
                argv.append("--" + key.replace("_", "-"))
        elif value is not None:
            argv.extend(["--" + key.replace("_", "-"), str(value)])
    return argv


def build_plan(recipe: dict) -> tuple[list[dict], dict | None]:
    output = Path(recipe["output_dir"])
    engine = recipe["katago"]
    stages = []
    train_data, eval_data = recipe.get("train_data"), recipe.get("eval_data")
    if train_data and recipe["smoke"]:
        train_data, eval_data = str(output / "smoke_train.jsonl"), str(output / "smoke_validation.jsonl")
    if not train_data:
        train_data, eval_data = str(output / "teacher_train.jsonl"), str(output / "teacher_validation.jsonl")
        teacher = recipe["teacher"]
        for split, path, count in (("train", train_data, teacher["train_positions"]),
                                   ("validation", eval_data, teacher["eval_positions"])):
            stages.append({"name": "teacher_" + split, "argv": command(
                "build_teacher_dataset.py", dataset=recipe["dataset"], output=path,
                split=split, max_positions=count, positions_per_game=teacher["positions_per_game"],
                min_move=teacher["min_move"], max_move=teacher["max_move"],
                sampling=teacher["sampling"],
                seed=recipe["seed"], max_visits=teacher["max_visits"], prior_mix=teacher["prior_mix"],
                batch_size=teacher["batch_size"], relabel_rules=recipe["rules"], relabel_komi=recipe["komi"],
                katago_bin=engine["binary"], katago_model=engine["model"], katago_config=engine["analysis_config"],
            )})
    model_kind = recipe["train_kind"]
    settings = {**recipe["training"], **recipe[model_kind + "_training"]}
    stages.append({"name": "distill", "argv": command(
        "run_text_distill.py" if model_kind == "text" else "run_policy_distill.py",
        train_data=train_data, eval_data=eval_data,
        output_dir=str(output / "distill"), model=recipe["model"], revision=recipe["revision"],
        resume_from=recipe.get("resume_from"), init_checkpoint=recipe.get("init_checkpoint"),
        device=recipe["device"], seed=recipe["seed"], **settings,
    )})
    checkpoint = str(output / "distill" / "best")
    # One trusted GTP process verifies rules and terminal outcomes during PPO.
    referee = shlex.join([engine["binary"], "gtp", "-model", engine["model"],
                          "-config", engine["gtp_config"], "-override-config", "maxVisits=1,numSearchThreads=1"])
    if recipe["selfplay"]["enabled"]:
        settings = {key: value for key, value in recipe["selfplay"].items() if key != "enabled"}
        stages.append({"name": "selfplay", "argv": command(
            "run_policy_rl.py", model=checkpoint, output=str(output / "selfplay"),
            referee_command=referee, device=recipe["device"], seed=recipe["seed"],
            rules=recipe["rules"], komi=recipe["komi"], **settings,
        )})
        checkpoint = str(output / "selfplay")
    match_config = None
    if recipe["evaluation"]["enabled"]:
        evaluation = recipe["evaluation"]
        model_options = ({"text_model": "policy=" + checkpoint, "encoding": settings["encoding"],
                          "revision": recipe["revision"], "max_new_tokens": settings["max_new_tokens"],
                          "text_beam_widths": evaluation["text_num_beams"]}
                         if model_kind == "text" else {"policy_checkpoint": "policy=" + checkpoint})
        stages.append({"name": "move_quality", "argv": command(
            "eval_move_quality.py", dataset=eval_data, train_data=train_data,
            **model_options, output=str(output / "move_quality.json"),
            device=recipe["device"], seed=recipe["seed"], max_positions=evaluation["max_positions"],
            max_visits=evaluation["max_visits"], katago_bin=engine["binary"],
            katago_model=engine["model"], katago_config=engine["eval_config"],
        )})
        gtp_options = ({"model": checkpoint, "encoding": settings["encoding"],
                        "revision": recipe["revision"], "max_new_tokens": settings["max_new_tokens"],
                        "num_beams": evaluation["text_num_beams"]}
                       if model_kind == "text" else {"checkpoint": checkpoint})
        model_command = command("model_gtp.py", **gtp_options, device=recipe["device"],
                                board_size=19, rules=recipe["rules"], komi=recipe["komi"], seed=recipe["seed"])
        opponent_command = [engine["binary"], "gtp", "-model", engine["model"],
                            "-config", engine["gtp_config"], "-override-config",
                            f"maxVisits={evaluation['max_visits']}"]
        match_config = {"board_size": 19, "num_games": evaluation["num_games"],
                        "model": {"name": "Qwen19Policy", "command": shlex.join(model_command)},
                        "katago": {"name": "KataGo", "command": shlex.join(opponent_command)},
                        "rules": recipe["rules"], "komi": recipe["komi"],
                        "sgf_output_dir": str(output / "match_sgfs"),
                        **{key: evaluation[key] for key in
                           ("main_time", "byoyomi_time", "byoyomi_periods", "max_moves")}}
        stages.append({"name": "matches", "argv": command(
            "eval_elo.py", match_config=str(output / "match_config.json"),
            log_file=str(output / "match.log"), summary_json=str(output / "match_summary.json"),
        )})
    return stages, match_config


def preflight(recipe: dict) -> dict[str, str]:
    """Validate resources before an expensive child process can be launched."""
    output = Path(recipe["output_dir"])
    if output.exists():
        raise ValueError(f"Output must be a fresh path: {output}")
    inputs = {key: recipe[key] for key in ("train_data", "eval_data") if recipe.get(key)}
    if not inputs:
        inputs["dataset"] = recipe["dataset"]
    resources = dict(inputs)
    needs_teacher = not recipe.get("train_data")
    if needs_teacher or recipe["selfplay"]["enabled"] or recipe["evaluation"]["enabled"]:
        resources["katago_model"] = recipe["katago"]["model"]
        binary = recipe["katago"]["binary"]
        if "$" in binary or not shutil.which(binary):
            raise ValueError(f"KataGo executable not found: {binary}; set KATAGO_BIN")
        resources["katago_binary"] = shutil.which(binary)
    for needed, key in ((needs_teacher, "analysis_config"),
                        (recipe["evaluation"]["enabled"], "eval_config"),
                        (recipe["selfplay"]["enabled"] or recipe["evaluation"]["enabled"], "gtp_config")):
        if needed:
            resources[key] = recipe["katago"][key]
    for name, path in resources.items():
        if "$" in path or not Path(path).is_file():
            raise ValueError(f"Missing {name}: {path}; configure paths or export the referenced environment variable")
    game_splits: dict[str, str] = {}
    for name, path in inputs.items():
        splits = set()
        count = 0
        with Path(path).open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("board_size") != 19:
                    raise ValueError(f"{name} requires explicit board_size=19 on every record")
                if not record.get("game_id"):
                    raise ValueError(f"{name} requires game_id on every record")
                splits.add(record.get("split"))
                split = record.get("split") if name == "dataset" else name
                previous = game_splits.setdefault(record["game_id"], split)
                if split != previous:
                    raise ValueError("Training/validation game leakage in recipe inputs")
                if name != "dataset":
                    if normalize_rules(record["rules"]) != normalize_rules(recipe["rules"]) or float(record["komi"]) != float(recipe["komi"]):
                        raise ValueError(f"{name} does not match the configured target rules/komi")
                    if (recipe["train_kind"] == "categorical"
                            and recipe["categorical_training"]["target"] in {"policy", "policy_argmax"}
                            and "policy_target" not in record):
                        raise ValueError(f"{name} needs policy_target; use a raw dataset for teacher generation")
                count += 1
        if not count:
            raise ValueError(f"{name} contains no records")
        if name == "dataset" and not {"train", "validation"}.issubset(splits):
            raise ValueError("dataset needs game-separated train and validation split metadata")
        expected_split = {"train_data": "train", "eval_data": "validation"}.get(name)
        if expected_split is not None and splits != {expected_split}:
            raise ValueError(f"{name} requires explicit {expected_split} split metadata on every record")
    for source in ("resume_from", "init_checkpoint"):
        if recipe.get(source):
            path = Path(recipe[source])
            if not path.is_dir():
                raise ValueError(f"Missing {source} checkpoint directory: {path}")
            if source == "resume_from" and not (path / "training_state.pt").is_file():
                raise ValueError(f"Resume checkpoint lacks training_state.pt: {path}; use init_checkpoint for a new optimizer")
            files = sorted(child for child in path.rglob("*") if child.is_file())
            if not files:
                raise ValueError(f"Empty {source}: {path}")
            resources.update({f"{source}/{child.relative_to(path)}": str(child) for child in files})
    return {name: file_hash(path) for name, path in resources.items()}


def write_smoke_inputs(recipe: dict) -> dict:
    """Select a deterministic, game-balanced subset of provided real labels."""
    from go_rl.experiment_utils import select_game_balanced_indices
    from go_rl.indexed_jsonl import IndexedJsonlRecords

    output = Path(recipe["output_dir"])
    selected = {}
    for source, destination, count in (
        ("train_data", "smoke_train.jsonl", recipe["teacher"]["train_positions"]),
        ("eval_data", "smoke_validation.jsonl", recipe["teacher"]["eval_positions"]),
    ):
        records = IndexedJsonlRecords(recipe[source])
        rows = records.select(select_game_balanced_indices(records, count, recipe["seed"]))
        path = output / destination
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        selected[source] = {"path": str(path), "sha256": file_hash(path),
                            "positions": len(rows), "games": len({row["game_id"] for row in rows})}
    return selected


def run_recipe(recipe: dict, stages: list[dict], match_config: dict | None, config_path: str) -> None:
    hashes = preflight(recipe)
    output = Path(recipe["output_dir"])
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    resolved_config = output / "recipe.yaml"
    resolved_config.write_text(yaml.safe_dump(recipe, sort_keys=False))
    manifest = {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "running", "recipe": recipe, "config_sha256": file_hash(config_path),
                "resolved_config": "recipe.yaml", "resolved_config_sha256": file_hash(resolved_config),
                "input_sha256": hashes, "python": sys.version, "commands": stages,
                "code_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in
                                [*sorted((ROOT / "go_rl").glob("*.py")),
                                 *sorted((ROOT / "scripts").glob("*.py")), ROOT / "uv.lock"] if path.is_file()},
                "stages": [], "limitations": "Research recipe; smoke truncations are not wins and imitation is not Elo."}
    manifest_path = output / "manifest.json"

    def save_manifest():
        temporary = output / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)

    save_manifest()
    try:
        if recipe["smoke"] and recipe.get("train_data"):
            manifest["smoke_inputs"] = write_smoke_inputs(recipe)
            save_manifest()
        if match_config is not None:
            (output / "match_config.json").write_text(json.dumps(match_config, indent=2) + "\n")
        for stage in stages:
            entry = {"name": stage["name"], "status": "running"}
            manifest["stages"].append(entry)
            save_manifest()
            start = time.monotonic()
            print(f"\n[{stage['name']}] {shlex.join(stage['argv'])}", flush=True)
            with (output / "logs" / (stage["name"] + ".log")).open("w") as log:
                result = subprocess.run(stage["argv"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            entry.update(returncode=result.returncode, seconds=time.monotonic() - start,
                         status="complete" if result.returncode == 0 else "failed")
            save_manifest()
            if result.returncode:
                raise RuntimeError(f"Stage {stage['name']} failed; see {output / 'logs' / (stage['name'] + '.log')}")
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "failed"
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_manifest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset")
    parser.add_argument("--train-data")
    parser.add_argument("--eval-data")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from", help="Resume distillation optimizer/RNG state; requires the same explicit teacher data")
    parser.add_argument("--init-checkpoint", help="Warm-start distillation model weights with a new optimizer")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--train-kind", choices=("text", "categorical"))
    parser.add_argument("--encoding", choices=("legacy", "grid", "compact", "stones", "cells"),
                        help="Override the selected model kind's input representation")
    parser.add_argument("--selfplay", action="store_true", help="Enable categorical self-play PPO after distillation")
    parser.add_argument("--evaluate", action="store_true", help="Enable move-quality diagnostics and paired-color GTP matches")
    parser.add_argument("--smoke", action="store_true", help="Use tiny real budgets; capped games provide no strength evidence")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without loading models, creating files, or downloading")
    args = parser.parse_args()
    try:
        recipe = load_recipe(args)
        stages, match_config = build_plan(recipe)
        print(f"19x19 {recipe['train_kind']} research recipe; small experiments do not establish full-game strength.")
        if args.smoke:
            print("Smoke mode: integration checks only; capped games are Void, not wins.")
        for stage in stages:
            print(f"[{stage['name']}] {shlex.join(stage['argv'])}")
        if match_config:
            print("Generated match_config.json:\n" + json.dumps(match_config, indent=2))
        if not args.dry_run:
            run_recipe(recipe, stages, match_config, args.config)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
