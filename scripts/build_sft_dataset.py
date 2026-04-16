from __future__ import annotations

import argparse
import logging

from go_rl.data_prep import build_sft_dataset
from go_rl.logging_utils import setup_logging
from venv_utils import ensure_virtualenv

logger = logging.getLogger(__name__)


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
        help="Go parser worker count (0 = auto-detect)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.info("Building SFT dataset")
    count = build_sft_dataset(
        args.sgf_dir,
        args.output,
        args.max_positions,
        num_workers=args.num_workers,
    )
    logger.info("Wrote %d samples to %s", count, args.output)


if __name__ == "__main__":
    main()
