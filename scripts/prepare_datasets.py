from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from go_rl.data_prep import run_go_parser, prepare_dataset_files
from go_rl.logging_utils import setup_logging
from venv_utils import ensure_virtualenv

logger = logging.getLogger(__name__)


def main() -> None:
    ensure_virtualenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgf-dir", required=True, help="Directory containing SGF files")
    parser.add_argument("--output-dir", default="data", help="Output directory for datasets")
    parser.add_argument("--tag", default="9x9", help="Tag for output filenames")
    parser.add_argument("--max-positions", type=int, default=200000, help="Maximum positions to collect")
    parser.add_argument("--positions-per-game", type=int, default=64,
                        help="Uniform positions per game (0 = all); avoids opening-only subsets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.05,
                        help="Fraction of whole games reserved from both SFT and RL")
    parser.add_argument("--symmetries", type=int, choices=[1, 8], default=1,
                        help="D4 augmentation of training records after the split")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Go parser worker count (0 = auto-detect)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Deprecated compatibility option; writes now stream in bounded memory",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.info("Preparing SFT and RL datasets")
    logger.debug("Batch size: %d", args.batch_size)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    go_parser_output = output_root / f"__go_parser_{args.tag}.jsonl"
    logger.info("Building intermediate dataset with Go parser")
    run_go_parser(
        sgf_dir=args.sgf_dir,
        output_path=str(go_parser_output),
        max_positions=args.max_positions,
        num_workers=args.num_workers,
        seed=args.seed,
        positions_per_game=args.positions_per_game,
    )
    report = prepare_dataset_files(
        go_parser_output, output_root, args.tag, eval_ratio=args.eval_ratio,
        seed=args.seed, symmetries=args.symmetries,
    )

    if go_parser_output.exists():
        go_parser_output.unlink()

    logger.info("Dataset manifest: %s", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
