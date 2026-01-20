from __future__ import annotations

import argparse
import logging

from go_rl.data_prep import build_sft_dataset
from go_rl.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgf-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-positions", type=int, default=200000)
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.info("Building SFT dataset")
    count = build_sft_dataset(args.sgf_dir, args.output, args.max_positions)
    logger.info("Wrote %d samples to %s", count, args.output)


if __name__ == "__main__":
    main()
