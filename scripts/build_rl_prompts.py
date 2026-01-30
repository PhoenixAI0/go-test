from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from go_rl.sgf_utils import iter_sgf_samples
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
    logger.info("Building RL prompt dataset")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for sample in iter_sgf_samples(args.sgf_dir, max_positions=args.max_positions):
            handle.write(json.dumps({"prompt": sample["prompt"]}) + "\n")
            count += 1
    logger.info("Wrote %d prompts to %s", count, args.output)


if __name__ == "__main__":
    main()
