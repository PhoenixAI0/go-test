from __future__ import annotations

import argparse
import json
import logging
import os

from go_rl.data_prep import build_sft_dataset
from go_rl.logging_utils import setup_logging
from venv_utils import ensure_virtualenv, run_go_parser

logger = logging.getLogger(__name__)


def _resolve_num_workers(requested: int) -> int:
    if requested and requested > 0:
        return requested
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def main() -> None:
    ensure_virtualenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgf-dir", required=True, help="Directory containing SGF files")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--max-positions", type=int, default=200000, help="Maximum positions to collect")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of parallel workers (0 = auto-detect, 1 = sequential)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing",
    )
    parser.add_argument(
        "--no-go-parser",
        action="store_true",
        help="Disable Go parser (use Python parsing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for buffered writes",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.info("Building SFT dataset")
    used_go_parser = False
    num_workers = _resolve_num_workers(args.num_workers)
    if not args.no_go_parser:
        used_go_parser = run_go_parser(
            sgf_dir=args.sgf_dir,
            output_path=args.output,
            max_positions=args.max_positions,
            num_workers=num_workers,
        )

    if used_go_parser:
        # Normalize to expected schema if needed
        rewritten = 0
        tmp_output = f"{args.output}.tmp"
        with open(args.output, "r", encoding="utf-8") as src, open(
            tmp_output, "w", encoding="utf-8"
        ) as dst:
            for line in src:
                if not line.strip():
                    continue
                sample = json.loads(line)
                record = {
                    "prompt": sample["prompt"],
                    "response": sample["response"],
                    "board_size": sample.get("board_size"),
                }
                dst.write(json.dumps(record) + "\n")
                rewritten += 1
        os.replace(tmp_output, args.output)
        logger.info("Wrote %d samples to %s", rewritten, args.output)
        return

    count = build_sft_dataset(
        args.sgf_dir,
        args.output,
        args.max_positions,
        num_workers=args.num_workers,
        use_parallel=not args.no_parallel,
        batch_size=args.batch_size,
    )
    logger.info("Wrote %d samples to %s", count, args.output)


if __name__ == "__main__":
    main()
