import pytest

from go_rl.board import GtpParseError
from go_rl.rollout import normalize_move


def test_normalize_move_basic():
    assert normalize_move("d4") == "D4"
    assert normalize_move("pass") == "PASS"


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("d4", "D4"),
        ("D4", "D4"),
        ("pass", "PASS"),
        ("PASS", "PASS"),
        ("Pass", "PASS"),
        ("q16", "Q16"),
        ("Q16", "Q16"),
        ("a1", "A1"),
        ("t19", "T19"),
        ("k10", "K10"),
    ],
)
def test_normalize_move_case_insensitive(input_text, expected):
    """Test that normalize_move handles case-insensitive input."""
    assert normalize_move(input_text) == expected


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("  d4  ", "D4"),  # Leading/trailing whitespace
        ("\td4\t", "D4"),  # Tabs
        ("\nd4\n", "D4"),  # Newlines
        ("  pass  ", "PASS"),
    ],
)
def test_normalize_move_strips_whitespace(input_text, expected):
    """Test that normalize_move strips surrounding whitespace."""
    assert normalize_move(input_text) == expected


@pytest.mark.parametrize(
    "input_text,stop_at_pass,expected",
    [
        ("D4 PASS", True, "PASS"),
        ("D4 PASS", False, "D4"),
        ("PASS D4", True, "PASS"),
        ("PASS D4", False, "PASS"),
        ("D4 Q16 PASS", True, "PASS"),
        ("D4 Q16 PASS", False, "D4"),
        ("Some text with PASS in it", True, "PASS"),
        ("Some text with PASS in it", False, "SOME"),  # Uppercased
    ],
)
def test_normalize_move_stop_at_pass_parametrized(input_text, stop_at_pass, expected):
    """Test stop_at_pass parameter with various inputs."""
    assert normalize_move(input_text, stop_at_pass=stop_at_pass) == expected


def test_normalize_move_extracts_first_token():
    """Test that normalize_move extracts the first valid-looking token."""
    assert normalize_move("D4 and some other text") == "D4"
    assert normalize_move("Q16 followed by more") == "Q16"
    assert normalize_move("move: D4") == "MOVE:"  # First token, uppercased


def test_normalize_move_empty():
    with pytest.raises(GtpParseError):
        normalize_move("   ")


@pytest.mark.parametrize(
    "invalid_input",
    [
        "",  # Empty string
        "   ",  # Only whitespace
        "\t\t",  # Only tabs
        "\n\n",  # Only newlines
    ],
)
def test_normalize_move_empty_inputs(invalid_input):
    """Test that empty or whitespace-only inputs raise GtpParseError."""
    with pytest.raises(GtpParseError):
        normalize_move(invalid_input)


def test_normalize_move_with_newlines():
    """Test handling of text with newlines."""
    assert normalize_move("D4\nQ16") == "D4"
    assert normalize_move("move:\nD4") == "MOVE:"  # Uppercased


def test_normalize_move_with_markdown():
    """Test handling of model outputs with markdown formatting."""
    # Common LLM output patterns - all uppercased
    assert normalize_move("```D4```") == "```D4```"  # First token, uppercased
    assert normalize_move("Move: D4") == "MOVE:"  # First token, uppercased
    assert normalize_move("I'll play D4") == "I'LL"  # First token, uppercased


def test_normalize_move_multiword_with_pass():
    """Test that PASS detection works in multi-word strings."""
    assert normalize_move("I will PASS this turn", stop_at_pass=True) == "PASS"
    assert normalize_move("Let me PASS", stop_at_pass=True) == "PASS"
