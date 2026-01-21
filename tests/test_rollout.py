import pytest

from go_rl.board import GtpParseError
from go_rl.rollout import normalize_move


def test_normalize_move_basic():
    assert normalize_move("d4") == "D4"
    assert normalize_move("pass") == "PASS"


def test_normalize_move_stop_at_pass():
    assert normalize_move("D4 PASS", stop_at_pass=True) == "PASS"
    assert normalize_move("D4 PASS", stop_at_pass=False) == "D4"


def test_normalize_move_empty():
    with pytest.raises(GtpParseError):
        normalize_move("   ")
