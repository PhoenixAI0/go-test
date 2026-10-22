"""Recalculate an existing move-quality report after excluding terminal histories.

This is a post-hoc data audit, not a fresh evaluation. Original engine/model
observations remain unchanged; the original report is never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from go_rl.data_prep import board_from_example
from go_rl.experiment_utils import file_sha256
from go_rl.move_quality import summarize_move_quality, terminal_history


def audit_report(source: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Audit output exists; preserve earlier reports")
    original = json.loads(source.read_text())
    retained, excluded = [], []
    for index, row in enumerate(original["details"]):
        if terminal_history(board_from_example(row["position"])):
            excluded.append({"original_evaluation_row": index, "position_id": row["position_id"],
                             "game_id": row["game_id"],
                             "reason": "history_reaches_two_pass_termination"})
        else:
            retained.append(row)
    report = dict(original)
    report.update(details=retained, positions=len(retained), games=len({row["game_id"] for row in retained}),
                  summary=summarize_move_quality(retained,
                      bootstrap_samples=original["arguments"]["bootstrap_samples"],
                      seed=original["arguments"]["seed"]))
    report["post_hoc_terminal_audit"] = {
        "source_report": str(source), "source_report_sha256": file_sha256(source),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "original_positions": original["positions"], "retained_positions": len(retained),
        "excluded": excluded, "new_engine_or_model_queries": 0,
        "source_query_counts_and_runtime_preserved": True,
        "limitation": "Post-hoc exclusion on the already sampled states; no replacement and no independent holdout.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_report(Path(args.source), Path(args.output))
    print(json.dumps({"output": args.output, "audit": report["post_hoc_terminal_audit"],
                      "summary": report["summary"]}))


if __name__ == "__main__":
    main()
