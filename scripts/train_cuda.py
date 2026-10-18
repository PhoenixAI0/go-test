"""Run the CUDA curriculum with optional match evaluation and Hub upload."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    ("sft_9x9", "run_sft.py", "sft_warm_start.yaml", 9),
    ("grpo_9x9", "run_grpo.py", "grpo_9x9.yaml", 9),
    ("grpo_19x19", "run_grpo.py", "grpo_19x19.yaml", 19),
    ("grpo_fullgame_19x19", "run_grpo_fullgame.py", "grpo_fullgame_19x19.yaml", 19),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sgf-9x9", type=Path, default=Path("data/sgf_9x9"))
    parser.add_argument("--sgf-19x19", type=Path, default=Path("data/sgf_19x19"))
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Base Hugging Face model ID or local checkpoint")
    parser.add_argument("--quantization-bits", type=int, choices=[4, 8, 16, 32], default=4)
    parser.add_argument("--optimizer", choices=["adamw_8bit", "adamw"], default="adamw_8bit")
    parser.add_argument("--output-dir", type=Path, help="New run directory (default: outputs/cuda_TIMESTAMP)")
    parser.add_argument("--max-positions", type=int, default=200000, help="Maximum positions per board size")
    parser.add_argument("--no-download-data", action="store_true", help="Require local SGFs; never download games")
    parser.add_argument("--download-games", type=int, default=10000, help="Maximum downloaded games per board size (default: 10000)")
    parser.add_argument("--data-cache", type=Path, default=Path("data/downloads"), help="Cache for verified SGF archives")
    parser.add_argument("--skip-setup", action="store_true", help="Reuse the installed environment and KataGo")
    parser.add_argument("--skip-data", action="store_true", help="Reuse prepared JSONL files from --data-dir")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Prepared dataset directory for --skip-data")
    parser.add_argument("--skip-sft", action="store_true", help="Start GRPO from --model (supply an existing SFT adapter if desired)")
    parser.add_argument("--skip-fullgame", action="store_true", help="Stop after 19x19 position GRPO")
    parser.add_argument("--smoke", action="store_true", help="One update per stage using a tiny synthetic SGF fixture")
    parser.add_argument("--dry-run", action="store_true", help="Print stages without setup, file writes, or a GPU")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B (disabled by default)")
    parser.add_argument("--wandb-project", default="go-rl")
    parser.add_argument("--no-eval", action="store_true", help="Disable match evaluations and promotion gates")
    parser.add_argument("--model-command-template", default=os.environ.get("MODEL_GTP_COMMAND_TEMPLATE", ""),
                        help="Enable evaluation using a GTP command containing {MODEL_DIR}; optional {BOARD_SIZE}")
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--eval-main-time", type=int, default=60)
    parser.add_argument("--eval-byoyomi-time", type=int, default=5)
    parser.add_argument("--eval-byoyomi-periods", type=int, default=5)
    parser.add_argument("--eval-komi", type=float, default=7.5)
    parser.add_argument("--eval-rules", default="chinese")
    parser.add_argument("--eval-sgf-root", type=Path, help="Default: RUN_DIR/match_sgfs")
    parser.add_argument("--eval-log-root", type=Path, help="Default: RUN_DIR/logs/matches")
    parser.add_argument("--fullgame-smoke-plies", type=int, help="Override full-game maximum plies")
    parser.add_argument("--upload-to-hub", action="store_true")
    parser.add_argument("--hf-repo-id")
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--hf-base-model", help="Model card base model (default: --model)")
    parser.add_argument("--preflight-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.max_positions < 1 or args.download_games < 1:
        parser.error("--max-positions and --download-games must be positive")
    if args.upload_to_hub and not args.hf_repo_id:
        parser.error("--hf-repo-id is required when --upload-to-hub is set")
    if args.eval_games < 1 or args.eval_main_time < 0 or args.eval_byoyomi_time < 0 or args.eval_byoyomi_periods < 1:
        parser.error("Evaluation games/periods must be positive and times nonnegative")
    if args.fullgame_smoke_plies is not None and args.fullgame_smoke_plies < 1:
        parser.error("--fullgame-smoke-plies must be positive")
    args.run_eval = bool(args.model_command_template) and not args.no_eval
    if args.run_eval and "{MODEL_DIR}" not in args.model_command_template:
        parser.error("--model-command-template must include {MODEL_DIR}")
    return args


def check_inputs(args: argparse.Namespace) -> None:
    if args.output_dir and args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        raise ValueError(f"Run directory is not empty: {args.output_dir}. Choose a new --output-dir.")
    if args.skip_data:
        for filename, _ in required_datasets(args):
            path = args.data_dir / filename
            if not path.is_file():
                raise ValueError(f"Prepared dataset not found: {path}")
    elif not args.smoke:
        if args.sgf_9x9.resolve() == args.sgf_19x19.resolve():
            raise ValueError("Each board size needs a separate SGF directory.")
        for path in (args.sgf_9x9, args.sgf_19x19):
            if path.exists() and not path.is_dir():
                raise ValueError(f"SGF path is not a directory: {path}")
            if not args.no_download_data:
                continue
            if not path.is_dir() or not next(path.rglob("*.sgf"), None):
                raise ValueError(f"No SGF games in {path}. Supply both board sizes, or use --smoke to test setup.")


def required_datasets(args: argparse.Namespace) -> list[tuple[str, int]]:
    datasets = [("rl_prompts_9x9.jsonl", 9), ("rl_prompts_19x19.jsonl", 19)]
    if not args.skip_sft:
        datasets.append(("sft_9x9.jsonl", 9))
    return datasets


def run(command: list[str]) -> None:
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def cuda_info(bits: int = 4, optimizer: str = "adamw_8bit") -> dict:
    # Release the probe's CUDA context before the training subprocess starts.
    command = [sys.executable, "-m", "go_rl.cuda_check", "--quantization-bits", str(bits),
               "--optimizer", optimizer]
    result = subprocess.run(command, text=True, capture_output=True, cwd=REPO_ROOT)
    if result.returncode:
        raise RuntimeError(f"CUDA preflight failed:\n{result.stdout}\n{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def validate_dataset(path: Path, size: int) -> None:
    from go_rl.prompting import parse_prompt

    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            board = parse_prompt(row["prompt"])
            if board.size != size:
                raise ValueError(f"Expected {size}x{size} positions in {path}; found {board.size}x{board.size}")
            count += 1
    if not count:
        raise ValueError(f"Dataset is empty: {path}")


def write_smoke_games(directory: Path, size: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    # Smoke data exercises parsing and checkpoint handoff, not playing strength.
    for i in range(4):
        moves = "".join(f";{'B' if j % 2 == 0 else 'W'}[{chr(97 + j)}{chr(97 + i)}]" for j in range(8))
        (directory / f"smoke_{i}.sgf").write_text(f"(;GM[1]FF[4]SZ[{size}]KM[7.5]{moves})\n")


def evaluate_checkpoint(args: argparse.Namespace, run_dir: Path, checkpoint: Path,
                        size: int, games: int) -> dict:
    label = checkpoint.name
    log_root = (args.eval_log_root or run_dir / "logs/matches").resolve()
    sgf_root = (args.eval_sgf_root or run_dir / "match_sgfs").resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    sgf_root.mkdir(parents=True, exist_ok=True)
    match_config = log_root / f"match_{label}.cfg"
    summary_path = log_root / f"{label}_summary.json"
    model_command = args.model_command_template.replace("{MODEL_DIR}", str(checkpoint)).replace(
        "{BOARD_SIZE}", f"{size}x{size}")
    run([sys.executable, "scripts/create_match_config.py",
         "--output", str(match_config), "--board-size", str(size),
         "--num-games", str(min(games, 8) if args.smoke else games),
         "--model-command", model_command,
         "--katago-bin", os.environ["KATAGO_BIN"],
         "--katago-model", os.environ["KATAGO_MODEL"],
         "--katago-config", os.environ.get("KATAGO_GTP_CONFIG", str(REPO_ROOT / "config/katago_gtp_eval.cfg")),
         "--sgf-output-dir", str(sgf_root / label),
         "--main-time", str(args.eval_main_time),
         "--byoyomi-time", str(args.eval_byoyomi_time),
         "--byoyomi-periods", str(args.eval_byoyomi_periods),
         "--komi", str(args.eval_komi), "--rules", args.eval_rules,
         "--model-name", f"Model_{label}", "--katago-name", "KataGoBaseline"])
    command = [sys.executable, "scripts/eval_elo.py",
               "--match-config", str(match_config),
               "--log-file", str(log_root / f"{label}.log"),
               "--sgf-dir", str(sgf_root / label), "--summary-json", str(summary_path),
               "--model-name", f"Model_{label}", "--katago-name", "KataGoBaseline"]
    if args.wandb:
        command.extend(["--wandb", "--wandb-project", args.wandb_project,
                        "--wandb-prefix", f"eval/{label}", "--wandb-run-name", f"eval_{label}"])
    # A successful evaluation must produce a fresh summary even with custom log roots.
    summary_path.unlink(missing_ok=True)
    run(command)
    return json.loads(summary_path.read_text())


def promotion_passed(summary: dict, gate: dict) -> bool:
    metric = gate.get("metric", "elo_delta_or_winrate")
    elo_passed = float(summary["elo_delta"]) >= float(gate.get("threshold_elo", 150))
    winrate_passed = float(summary["win_rate"]) >= float(gate.get("threshold_winrate", 0.55))
    outcomes = {"elo_delta_or_winrate": elo_passed or winrate_passed,
                "elo_delta_and_winrate": elo_passed and winrate_passed,
                "elo_delta": elo_passed, "winrate": winrate_passed}
    if metric not in outcomes:
        raise ValueError(f"Unknown promotion metric: {metric}")
    passed = outcomes[metric]
    print(f"Promotion gate {'PASSED' if passed else 'FAILED'}: metric={metric}, "
          f"elo_delta={summary['elo_delta']}, win_rate={summary['win_rate']}", flush=True)
    return passed


def upload_checkpoint(args: argparse.Namespace, checkpoint: Path) -> None:
    command = [sys.executable, "scripts/upload_to_hub.py", "--model", str(checkpoint),
               "--repo-id", args.hf_repo_id, "--board-size", "19",
               "--base-model", args.hf_base_model or args.model, "--push-only"]
    if args.hf_private:
        command.append("--private")
    run(command)


def main() -> None:
    os.chdir(REPO_ROOT)
    args = parse_args()
    stages = STAGES[:-1] if args.skip_fullgame else STAGES
    if args.skip_sft:
        stages = stages[1:]
    if args.dry_run:
        print(f"Base model: {args.model}; setup: {not args.skip_setup}; smoke: {args.smoke}")
        print(f"Data: prepared files in {args.data_dir}" if args.skip_data else
              "Data: synthetic smoke fixtures" if args.smoke else f"Data: {args.sgf_9x9}, {args.sgf_19x19}")
        print(f"Download missing SGFs: {not args.no_download_data and not args.smoke and not args.skip_data}")
        print(" -> ".join(stage[0] for stage in stages))
        print("Each stage loads the preceding saved adapter.")
        print(f"Match evaluation and promotion gates: {args.run_eval}; Hub upload: {args.upload_to_hub}")
        return
    check_inputs(args)
    if args.preflight_only:
        return
    print(f"Match evaluation and promotion gates: {'enabled' if args.run_eval else 'disabled'}", flush=True)

    import yaml
    from go_rl.board import BoardState
    from go_rl.config import KataGoConfig
    from go_rl.katago import KataGoAnalysisClient

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("The full pipeline launches one GPU. Use the individual stage scripts for DDP.")
    gpu = cuda_info(args.quantization_bits, args.optimizer)
    print(f"Training on {gpu['gpu']}", flush=True)
    for key in ("KATAGO_BIN", "KATAGO_MODEL"):
        if not os.environ.get(key) or not Path(os.environ[key]).is_file():
            raise ValueError(f"Set {key} to an existing file, or run setup_environment.sh.")
    katago_settings = {
        "binary_path": os.environ["KATAGO_BIN"],
        "model_path": os.environ["KATAGO_MODEL"],
        "config_path": os.environ.get("KATAGO_TRAIN_CONFIG", str(REPO_ROOT / "config/katago_analysis_train.cfg")),
        "request_timeout_seconds": 120.0,
    }
    client = KataGoAnalysisClient(KataGoConfig(**katago_settings))
    try:
        client.start()
        for size in (9, 19):
            print(f"KataGo {size}x{size} preflight: {client.analyze_position(BoardState.empty(size)).winrate:.3f}", flush=True)
    finally:
        client.close()

    data_sources = {}
    if not args.skip_data and not args.smoke and not args.no_download_data:
        from go_rl.sgf_download import ensure_sgf_games
        data_sources = ensure_sgf_games(
            {9: args.sgf_9x9, 19: args.sgf_19x19}, args.data_cache, args.download_games)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (args.output_dir or Path(f"outputs/{'smoke' if args.smoke else 'cuda'}_{stamp}")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_dir = run_dir / "config"
    config_dir.mkdir()
    data_dir = args.data_dir.resolve() if args.skip_data else run_dir / "data"
    for size, source in (() if args.skip_data else ((9, args.sgf_9x9), (19, args.sgf_19x19))):
        if args.smoke:
            source = data_dir / f"sgf_{size}x{size}"
            write_smoke_games(source, size)
        run([sys.executable, "scripts/prepare_datasets.py", "--sgf-dir", str(source.resolve()),
             "--output-dir", str(data_dir), "--tag", f"{size}x{size}",
             "--max-positions", str(32 if args.smoke else args.max_positions)])
        validate_dataset(data_dir / f"sft_{size}x{size}.jsonl", size)
    for filename, size in required_datasets(args):
        validate_dataset(data_dir / filename, size)

    previous_model = args.model
    # Persist all resolved configs before training starts for reproducibility.
    commands = []
    for label, script, config_name, size in stages:
        config = yaml.safe_load((REPO_ROOT / "config" / config_name).read_text())
        training = config if script == "run_sft.py" else config["training"]
        training.update(model_name=previous_model, output_dir=str(run_dir / label),
                        log_dir=str(run_dir / "logs" / label),
                        quantization_bits=args.quantization_bits,
                        use_8bit_optimizer=args.optimizer == "adamw_8bit")
        if script == "run_sft.py":
            config["dataset_path"] = str(data_dir / "sft_9x9.jsonl")
        else:
            config["dataset"] = str(data_dir / f"rl_prompts_{size}x{size}.jsonl")
            config["katago"].update(katago_settings)
        path = config_dir / config_name
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        command = [sys.executable, f"scripts/{script}", "--config", str(path), "--gradient-checkpointing"]
        if args.smoke:
            command.append("--smoke")
            if script == "run_grpo_fullgame.py":
                command.extend(["--max-model-turns", "2"])
        if script == "run_grpo_fullgame.py":
            max_plies = args.fullgame_smoke_plies or (4 if args.smoke else None)
            if max_plies is not None:
                command.extend(["--max-plies", str(max_plies)])
        if args.wandb:
            command.extend(["--wandb", "--wandb-project", args.wandb_project])
        gate = None
        if label == "grpo_9x9":
            gate = config.get("promotion_gate") or {}
        elif label == "grpo_19x19" and not args.skip_fullgame:
            gate = {"metric": "winrate", **(config.get("promotion_gate_fullgame") or {})}
        commands.append((command, Path(training["output_dir"]), size, gate))
        previous_model = training["output_dir"]
    (run_dir / "run.json").write_text(json.dumps({"model": args.model, "smoke": args.smoke,
        "stages": [label for label, *_ in stages], "sgf_sources": data_sources, **gpu}, indent=2) + "\n")
    final_checkpoint = None
    for command, checkpoint, size, gate in commands:
        run(command)
        if not all((checkpoint / name).is_file() for name in
                   ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json")):
            raise RuntimeError(f"Stage did not save its LoRA adapter: {checkpoint}")
        final_checkpoint = checkpoint
        if args.run_eval:
            games = int(gate.get("evaluation_games", 200)) if gate is not None else args.eval_games
            summary = evaluate_checkpoint(args, run_dir, checkpoint, size, games)
            if gate is not None and not promotion_passed(summary, gate):
                if size == 9:
                    raise RuntimeError("9x9 promotion gate failed; stopping before 19x19 training.")
                print("Skipping full-game GRPO stage because the promotion gate did not pass.")
                break
    if args.upload_to_hub:
        upload_checkpoint(args, final_checkpoint)
    print(f"Training complete. Final adapter: {final_checkpoint}\nRun configs and data: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Training stopped: {exc}", file=sys.stderr)
        sys.exit(1)
