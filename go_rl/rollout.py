from __future__ import annotations

import re

# Training targets are a bare GTP move, so the answer must be exactly one.
_MOVE_ANSWER = re.compile(r"PASS|[A-HJ-T](?:[1-9]|1[0-9])")


def parse_move_answer(answer: str) -> str:
    """Return the move if the final answer is exactly a canonical GTP move, else "".

    Any preamble, punctuation, lowercase or extra text is a format error that the
    reward penalizes like an illegal move, so the policy learns the format rather
    than relying on a lenient parser. Board bounds are checked by the caller.
    """
    answer = answer.strip()
    return answer if _MOVE_ANSWER.fullmatch(answer) else ""
