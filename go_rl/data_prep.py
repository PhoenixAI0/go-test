from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from .sgf_utils import load_sgf_dir

logger = logging.getLogger(__name__)


def build_sft_dataset(sgf_dir: str, output_path: str, max_positions: int) -> int:
    logger.info("Loading SGF samples from %s", sgf_dir)
    logger.debug("Max positions: %d", max_positions)
    samples: List[dict] = load_sgf_dir(sgf_dir, max_positions=max_positions)
    logger.info("Preparing to write %d samples", len(samples))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Writing dataset to %s", output)
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            record = {
                "prompt": sample["prompt"],
                "response": sample["target"],
                "board_size": sample["board_size"],
            }
            handle.write(json.dumps(record) + "\n")
    logger.info("Finished writing dataset to %s", output)
    return len(samples)
