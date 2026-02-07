from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List

from go_rl.logging_utils import setup_logging
from venv_utils import ensure_virtualenv, run_go_parser

logger = logging.getLogger(__name__)


def _resolve_num_workers(requested: int) -> int:
    if requested and requested > 0:
        return requested
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def _write_jsonl_batch(handle, records: List[dict]) -> None:
    if not records:
        return
    handle.writelines(json.dumps(record) + "\n" for record in records)


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
    logger.debug("Batch size: %d", args.batch_size)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    num_workers = _resolve_num_workers(args.num_workers)
    count = 0
    batch: List[dict] = []

    with output.open("w", encoding="utf-8", buffering=8192 * 16) as handle:
        go_parser_output = output.parent / f"__go_parser_{output.stem}.jsonl"
        logger.info("Building intermediate dataset with Go parser")
        run_go_parser(
            sgf_dir=args.sgf_dir,
            output_path=str(go_parser_output),
            max_positions=args.max_positions,
            num_workers=num_workers,
        )

        with go_parser_output.open("r", encoding="utf-8") as src:
            for line in src:
                if not line.strip():
                    continue
                sample = json.loads(line)
                batch.append({"prompt": sample["prompt"]})
                count += 1

                if len(batch) >= args.batch_size:
                    _write_jsonl_batch(handle, batch)
                    batch = []

                    if count % 10000 == 0:
                        logger.info("Written %d prompts...", count)
        go_parser_output.unlink(missing_ok=True)

        if batch:
            _write_jsonl_batch(handle, batch)

    logger.info("Wrote %d prompts to %s", count, args.output)


if __name__ == "__main__":
    main()
