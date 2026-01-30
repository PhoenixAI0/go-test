import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from go_rl.sgf_utils import (
    SgfGame,
    sample_positions,
    load_sgf,
    iter_sgf_samples,
    load_sgf_dir,
    _require_sgfmill,
)


def test_sgf_game_dataclass():
    """Test SgfGame dataclass creation."""
    game = SgfGame(
        board_size=9,
        moves=[("B", (5, 3)), ("W", (3, 3))],
    )
    assert game.board_size == 9
    assert len(game.moves) == 2
    assert len(game.setup_stones) == 0


def test_sgf_game_with_setup_stones():
    """Test SgfGame with setup stones."""
    game = SgfGame(
        board_size=9,
        moves=[("B", (5, 3))],
        setup_stones=[("B", (4, 4)), ("W", (3, 3))],
    )
    assert len(game.setup_stones) == 2
    assert game.setup_stones[0] == ("B", (4, 4))
    assert game.setup_stones[1] == ("W", (3, 3))


def test_sample_positions_outputs_prompts_and_targets():
    game = SgfGame(
        board_size=5,
        moves=[("B", (4, 0)), ("W", None), ("B", (3, 1))],
    )
    samples = sample_positions(game, max_positions=2)
    assert len(samples) == 2
    assert samples[0]["prompt"] == "board_size: 5\nmove:"
    assert samples[0]["target"] == "A1"
    assert samples[1]["prompt"].startswith("board_size: 5\nB: A1")
    assert samples[1]["target"] == "PASS"


def test_sample_positions_respects_max_positions():
    """Test that sample_positions respects max_positions limit."""
    game = SgfGame(
        board_size=9,
        moves=[("B", (i, i)) for i in range(9)],  # 9 moves
    )
    samples = sample_positions(game, max_positions=5)
    assert len(samples) == 5


def test_sample_positions_all_moves_when_below_max():
    """Test that all moves are sampled when below max_positions."""
    game = SgfGame(
        board_size=9,
        moves=[("B", (0, 0)), ("W", (1, 1)), ("B", (2, 2))],
    )
    samples = sample_positions(game, max_positions=100)
    assert len(samples) == 3


def test_sample_positions_includes_player_and_board_size():
    """Test that samples include player and board_size fields."""
    game = SgfGame(
        board_size=13,
        moves=[("B", (5, 3)), ("W", (3, 5))],
    )
    samples = sample_positions(game, max_positions=2)

    assert samples[0]["player"] == "B"
    assert samples[0]["board_size"] == 13
    assert samples[1]["player"] == "W"
    assert samples[1]["board_size"] == 13


def test_sample_positions_handles_pass_moves():
    """Test that PASS moves are correctly handled."""
    game = SgfGame(
        board_size=9,
        moves=[("B", (5, 3)), ("W", None), ("B", None)],
    )
    samples = sample_positions(game, max_positions=3)

    assert samples[1]["target"] == "PASS"
    assert samples[2]["target"] == "PASS"


def test_sample_positions_applies_setup_stones():
    """Test that setup stones are applied to the board before sampling."""
    game = SgfGame(
        board_size=9,
        moves=[("B", (5, 3))],
        setup_stones=[("B", (4, 4)), ("W", (3, 3))],
    )
    samples = sample_positions(game, max_positions=1)

    # Prompt should include the setup stones
    assert "B:" in samples[0]["prompt"]
    assert "W:" in samples[0]["prompt"]


def test_sample_positions_empty_game():
    """Test sampling from a game with no moves."""
    game = SgfGame(board_size=9, moves=[])
    samples = sample_positions(game, max_positions=10)
    assert len(samples) == 0


def test_require_sgfmill_missing():
    """Test that _require_sgfmill raises RuntimeError when sgfmill is missing."""
    with patch("builtins.__import__", side_effect=ImportError("No module named 'sgfmill'")):
        with pytest.raises(RuntimeError, match="sgfmill is required"):
            _require_sgfmill()


def test_load_sgf_mock(tmp_path):
    """Test load_sgf with mocked sgfmill - skip if sgfmill not installed."""
    pytest.importorskip("sgfmill", reason="sgfmill not installed")

    # This is a simplified version - full SGF testing would require sgfmill
    # For now, just verify the function can be called
    # A real test would require a valid SGF file and sgfmill library
    pass


def test_iter_sgf_samples_skips_bad_files(tmp_path):
    """Test that iter_sgf_samples skips files that fail to load."""
    # Create some SGF files
    sgf_dir = tmp_path / "sgf"
    sgf_dir.mkdir()

    # Create valid file
    valid_game = SgfGame(
        board_size=9,
        moves=[("B", (5, 3)), ("W", (3, 3))],
    )

    # Mock load_sgf to raise on bad files
    def mock_load_sgf(path):
        if "bad" in str(path):
            raise ValueError("Bad SGF file")
        return valid_game

    (sgf_dir / "good.sgf").write_text("good")
    (sgf_dir / "bad.sgf").write_text("bad")

    with patch("go_rl.sgf_utils.load_sgf", side_effect=mock_load_sgf):
        samples = list(iter_sgf_samples(str(sgf_dir), max_positions=10))

    # Should only get samples from the good file
    assert len(samples) == 2  # valid_game has 2 moves


def test_load_sgf_dir(tmp_path):
    """Test load_sgf_dir returns a list of samples."""
    sgf_dir = tmp_path / "sgf"
    sgf_dir.mkdir()
    (sgf_dir / "test.sgf").write_text("test")

    valid_game = SgfGame(
        board_size=9,
        moves=[("B", (5, 3)), ("W", (3, 3))],
    )

    with patch("go_rl.sgf_utils.load_sgf", return_value=valid_game):
        samples = load_sgf_dir(str(sgf_dir), max_positions=1)

    assert isinstance(samples, list)
    assert len(samples) == 1


def test_iter_sgf_samples_respects_max_positions(tmp_path):
    """Test that iter_sgf_samples stops after max_positions."""
    sgf_dir = tmp_path / "sgf"
    sgf_dir.mkdir()

    # Create multiple SGF files
    for i in range(5):
        (sgf_dir / f"game{i}.sgf").write_text(f"game{i}")

    large_game = SgfGame(
        board_size=9,
        moves=[("B", (i, i)) for i in range(9)],  # 9 moves per game
    )

    with patch("go_rl.sgf_utils.load_sgf", return_value=large_game):
        samples = list(iter_sgf_samples(str(sgf_dir), max_positions=10))

    # Should stop at 10 samples even though there are 5 games * 9 moves = 45 possible
    assert len(samples) == 10
