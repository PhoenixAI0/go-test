"""Run the complete CUDA curriculum; invoked by train_cuda.sh."""
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
    parser.add_argument("--skip-fullgame", action="store_true", help="Stop after 19x19 position GRPO")
    parser.add_argument("--smoke", action="store_true", help="One update per stage using a tiny synthetic SGF fixture")
    parser.add_argument("--dry-run", action="store_true", help="Print stages without setup, file writes, or a GPU")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B (disabled by default)")
    parser.add_argument("--preflight-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.max_positions < 1 or args.download_games < 1:
        parser.error("--max-positions and --download-games must be positive")
    return args


def check_inputs(args: argparse.Namespace) -> None:
    if args.output_dir and args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        raise ValueError(f"Run directory is not empty: {args.output_dir}. Choose a new --output-dir.")
    if not args.smoke:
        if args.sgf_9x9.resolve() == args.sgf_19x19.resolve():
            raise ValueError("Each board size needs a separate SGF directory.")
        for path in (args.sgf_9x9, args.sgf_19x19):
            if path.exists() and not path.is_dir():
                raise ValueError(f"SGF path is not a directory: {path}")
            if not args.no_download_data:
                continue
            if not path.is_dir() or not next(path.rglob("*.sgf"), None):
                raise ValueError(f"No SGF games in {path}. Supply both board sizes, or use --smoke to test setup.")


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


def main() -> None:
    os.chdir(REPO_ROOT)
    args = parse_args()
    stages = STAGES[:-1] if args.skip_fullgame else STAGES
    if args.dry_run:
        print(f"Base model: {args.model}; setup: {not args.skip_setup}; smoke: {args.smoke}")
        print("Data: synthetic smoke fixtures" if args.smoke else f"Data: {args.sgf_9x9}, {args.sgf_19x19}")
        print(f"Download missing SGFs: {not args.no_download_data and not args.smoke}")
        print(" -> ".join(stage[0] for stage in stages))
        print("Each stage loads the preceding saved adapter. Match evaluation and Hub upload are separate.")
        return
    check_inputs(args)
    if args.preflight_only:
        return

    import yaml
    from go_rl.board import BoardState
    from go_rl.config import KataGoConfig
    from go_rl.katago import KataGoAnalysisClient

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("train_cuda.sh launches one GPU. Use the individual stage scripts for DDP.")
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
    if not args.smoke and not args.no_download_data:
        from go_rl.sgf_download import ensure_sgf_games
        data_sources = ensure_sgf_games(
            {9: args.sgf_9x9, 19: args.sgf_19x19}, args.data_cache, args.download_games)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (args.output_dir or Path(f"outputs/{'smoke' if args.smoke else 'cuda'}_{stamp}")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_dir = run_dir / "config"
    config_dir.mkdir()
    data_dir = run_dir / "data"
    for size, source in ((9, args.sgf_9x9), (19, args.sgf_19x19)):
        if args.smoke:
            source = data_dir / f"sgf_{size}x{size}"
            write_smoke_games(source, size)
        run([sys.executable, "scripts/prepare_datasets.py", "--sgf-dir", str(source.resolve()),
             "--output-dir", str(data_dir), "--tag", f"{size}x{size}",
             "--max-positions", str(32 if args.smoke else args.max_positions)])
        validate_dataset(data_dir / f"sft_{size}x{size}.jsonl", size)

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
                command.extend(["--max-plies", "4", "--max-model-turns", "2"])
        if args.wandb:
            command.append("--wandb")
        commands.append((command, Path(training["output_dir"])))
        previous_model = training["output_dir"]
    (run_dir / "run.json").write_text(json.dumps({"model": args.model, "smoke": args.smoke,
        "stages": [label for label, *_ in stages], "sgf_sources": data_sources, **gpu}, indent=2) + "\n")
    for command, checkpoint in commands:
        run(command)
        if not all((checkpoint / name).is_file() for name in
                   ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json")):
            raise RuntimeError(f"Stage did not save its LoRA adapter: {checkpoint}")
    print(f"Training complete. Final adapter: {previous_model}\nRun configs and data: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Training stopped: {exc}", file=sys.stderr)
        sys.exit(1)
