"""Validate adjudicated Go results without inventing outcomes for void games."""
from __future__ import annotations

import math


def parse_game_result(score: str) -> tuple[str, float]:
    """Return normalized GTP/SGF result and Black's outcome in {-1, 0, 1}.

    The caller is responsible for obtaining a trusted adjudication. Parsing a
    string is not itself a score verifier. Void, unknown and malformed results
    raise rather than becoming draws or wins.
    """
    result = score.strip().upper()
    if result in {"0", "DRAW", "JIGO"}:
        return "0", 0.0
    if len(result) > 2 and result[:2] in {"B+", "W+"}:
        margin = result[2:]
        if margin not in {"R", "RESIGN", "T", "TIME", "F", "FORFEIT"}:
            try:
                points = float(margin)
            except ValueError as exc:
                raise ValueError(f"Unscored result: {score!r}") from exc
            if not math.isfinite(points) or points <= 0:
                raise ValueError(f"Invalid winning margin: {score!r}")
        return result, 1.0 if result[0] == "B" else -1.0
    raise ValueError(f"Unscored result: {score!r}")
