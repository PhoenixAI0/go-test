from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

from go_rl.sgf_utils import iter_sgf_samples
from go_rl.logging_utils import setup_logging
from venv_utils import ensure_virtualenv

logger = logging.getLogger(__name__)


def _write_jsonl_batch(handle, records: List[dict]) -> None:
    """Write a batch of records to JSONL file efficiently."""
    if not records:
        return
    lines = [json.dumps(record) + "\n" for record in records]
    handle.write("".join(lines))


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
    logger.info("Building RL prompt dataset")
    logger.debug("Parallel processing: %s (workers: %d)", not args.no_parallel, args.num_workers)
    logger.debug("Batch size: %d", args.batch_size)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    batch: List[dict] = []

    with output.open("w", encoding="utf-8", buffering=8192 * 16) as handle:
        for sample in iter_sgf_samples(
            args.sgf_dir,
            max_positions=args.max_positions,
            num_workers=args.num_workers,
            use_parallel=not args.no_parallel,
        ):
            batch.append({"prompt": sample["prompt"]})
            count += 1

            # Write batch when full
            if len(batch) >= args.batch_size:
                _write_jsonl_batch(handle, batch)
                batch = []

                if count % 10000 == 0:
                    logger.info("Written %d prompts...", count)

        # Write remaining batch
        if batch:
            _write_jsonl_batch(handle, batch)

    logger.info("Wrote %d prompts to %s", count, args.output)


if __name__ == "__main__":
    main()
