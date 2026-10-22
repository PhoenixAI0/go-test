"""Compare match SGF results with raw-board area counts, without adjudicating games.

KataGo's pass-alive scoring can legitimately differ from counting every remaining
stone as its own color. These diagnostics never replace RE, rewards, or match
results. Move caps and other unscored games never acquire an outcome here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from sgfmill import sgf

from go_rl.board import BoardState, coords_to_gtp
from go_rl.game_result import parse_game_result
from go_rl.rules import legality_superko, normalize_rules
from go_rl.scoring import count_board_area, terminal_board_area_result, validate_board_area_rules


def _numeric_black_margin(result: str) -> float | None:
    if result == "0":
        return 0.0
    try:
        return float(result[2:]) * (1 if result.startswith("B+") else -1)
    except ValueError:
        return None


def audit_sgf(data: bytes, *, source: str) -> dict:
    """Replay one ordinary match strictly; reject edits, variations and bad history."""
    game = sgf.Sgf_game.from_bytes(data)
    root = game.get_root()
    if root.has_property("GM") and root.get("GM") != 1:
        raise ValueError("The SGF does not describe Go")
    if not root.has_property("RU") or not root.has_property("KM"):
        raise ValueError("Match SGFs must explicitly declare rules and komi")
    rules, komi = root.get("RU"), float(root.get("KM"))
    superko = legality_superko(rules)
    if not math.isfinite(komi):
        raise ValueError("SGF komi must be finite")
    raw_result = root.get("RE") if root.has_property("RE") else ""
    if raw_result.strip().upper() in {"", "VOID", "?"}:
        result, outcome = None, None
    else:
        result, outcome = parse_game_result(raw_result)
    comment = root.get("C") if root.has_property("C") else ""
    match = re.search(r"(?:^|\s)termination=([a-z_]+)(?:\s|$)", comment)
    termination = match.group(1) if match else "unspecified"
    if termination == "move_cap" and result is not None:
        raise ValueError("A move-capped SGF cannot have an adjudicated outcome")

    size = game.get_size()
    matrix = [[0] * size for _ in range(size)]
    black, white, empty = root.get_setup_stones()
    if black & white or (black | white) & empty:
        raise ValueError("Conflicting SGF setup stones")
    for value, points in ((1, black), (2, white)):
        for row, column in points:
            matrix[size - 1 - row][column] = value
    player = (root.get("PL").upper() if root.has_property("PL")
              else "W" if root.has_property("HA") and root.get("HA") > 1 else "B")
    board = BoardState.from_matrix(size, matrix, next_player=player)
    for node in game.get_main_sequence():
        if len(node) > 1:
            raise ValueError("Match SGFs cannot contain variations")
        if node is not root and (node.has_setup_stones() or node.has_property("PL")):
            raise ValueError("Mid-game setup/player edits are not match trajectories")
        if node.has_property("B") and node.has_property("W"):
            raise ValueError("An SGF node cannot contain two moves")
        color, coords = node.get_move()
        if color is None:
            continue
        player = color.upper()
        if player != board.next_player():
            raise ValueError("SGF move history does not alternate players")
        if board.consecutive_passes() >= 2:
            raise ValueError("SGF contains moves after two-pass termination")
        point = (size - 1 - coords[0], coords[1]) if coords is not None else None
        move = coords_to_gtp(point, size)
        if move not in board.legal_moves(player, superko=superko):
            raise ValueError(f"Illegal SGF action at ply {len(board.moves)}: {player} {move}")
        board.apply_move(player, move, strict=True)

    row = {
        "source": source, "sha256": hashlib.sha256(data).hexdigest(),
        "board_size": size, "rules": normalize_rules(rules), "komi": komi,
        "original_result": raw_result, "adjudicated_result": result,
        "adjudicated_black_outcome": outcome, "termination": termination,
        "plies": len(board.moves), "two_passes": board.consecutive_passes() >= 2,
        "raw_board_result": None,
    }
    if termination == "move_cap":
        row["status"] = "move_cap"
    elif result is None:
        row["status"] = "unscored"
    elif _numeric_black_margin(result) is None:
        row["status"] = "nonnumeric_adjudication"
    elif not row["two_passes"]:
        row["status"] = "not_two_pass_completion"
    else:
        try:
            validate_board_area_rules(rules)
        except ValueError as exc:
            row.update(status="unsupported_raw_area_rules", reason=str(exc))
        else:
            raw_board_result = terminal_board_area_result(board, komi=komi, rules=rules)
            raw_margin = _numeric_black_margin(raw_board_result)
            adjudicated_margin = _numeric_black_margin(result)
            row.update(
                status="compared", raw_board_result=raw_board_result,
                raw_board_area=asdict(count_board_area(board.current_board())),
                adjudicated_black_margin=adjudicated_margin, raw_board_black_margin=raw_margin,
                adjudicated_minus_raw_black_margin=adjudicated_margin - raw_margin,
                margin_matches=adjudicated_margin == raw_margin,
                winner_matches=outcome == parse_game_result(raw_board_result)[1],
            )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sgf-dir", type=Path, action="append", default=[])
    parser.add_argument("--sgf", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Audit output already exists; preserve earlier reports")
    sources = set(path.resolve() for path in args.sgf)
    for directory in args.sgf_dir:
        if not directory.is_dir():
            parser.error(f"SGF directory does not exist: {directory}")
        sources.update(path.resolve() for path in directory.rglob("*.sgf"))
    if not sources:
        parser.error("Provide at least one match via --sgf or --sgf-dir")
    rows = []
    for source in sorted(sources):
        try:
            rows.append(audit_sgf(source.read_bytes(), source=str(source)))
        except (ValueError, OSError) as exc:
            rows.append({"source": str(source), "status": "invalid", "reason": str(exc)})
    counts = Counter(row["status"] for row in rows)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine_or_model_queries": 0,
        "interpretation": "Diagnostic only. Raw-board area differences can be legitimate under KataGo pass-alive scoring; original adjudications remain unchanged.",
        "counts": dict(counts),
        "different_margins": sum(row.get("margin_matches") is False for row in rows),
        "different_winners": sum(row.get("winner_matches") is False for row in rows),
        "games": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "counts": dict(counts),
                      "different_margins": report["different_margins"],
                      "different_winners": report["different_winners"]}))
    if counts["invalid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
