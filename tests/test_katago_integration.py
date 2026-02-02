"""Integration tests for KataGo analysis client.

These tests verify that the KataGo integration works correctly.
Most tests use mocked KataGo to avoid requiring real KataGo binaries.
Tests marked with @pytest.mark.katago require real KataGo setup.
"""

from __future__ import annotations

import pytest

from go_rl.board import BoardState
from go_rl.katago import KataGoAnalysisClient


class TestMockedKataGo:
    """Tests using mocked KataGo client (no real binary required)."""

    def test_analyze_empty_board(self, mock_katago_client, sample_board_9x9):
        """Test analyzing an empty board position."""
        empty_board = BoardState(size=9)
        result = mock_katago_client.analyze_position(empty_board)

        assert result is not None
        assert 0.0 <= result.winrate <= 1.0
        assert result.raw is not None

    def test_analyze_board_with_moves(self, mock_katago_client, sample_board_9x9):
        """Test analyzing a board with some moves played."""
        result = mock_katago_client.analyze_position(sample_board_9x9)

        assert result is not None
        assert 0.0 <= result.winrate <= 1.0
        # Winrate should increase with more moves (based on mock logic)
        assert result.winrate > 0.5

    def test_evaluate_multiple_moves(self, mock_katago_client, sample_board_9x9):
        """Test evaluating multiple candidate moves."""
        moves = ["D5", "E4", "F3"]
        results = mock_katago_client.evaluate_moves(
            sample_board_9x9, moves, player="B"
        )

        assert len(results) == 3
        for move in moves:
            assert move in results
            assert 0.0 <= results[move] <= 1.0

    def test_client_lifecycle(self, mock_katago_client):
        """Test client start and close lifecycle."""
        # Should not raise errors
        mock_katago_client.start()
        mock_katago_client.close()

        # Verify methods were called
        mock_katago_client.start.assert_called_once()
        mock_katago_client.close.assert_called_once()

    def test_winrate_progression(self, mock_katago_client):
        """Test that winrate changes sensibly as game progresses."""
        board = BoardState(size=9)

        # Analyze at different stages
        results = []
        for i in range(5):
            result = mock_katago_client.analyze_position(board)
            results.append(result.winrate)

            # Play a move
            player = "B" if i % 2 == 0 else "W"
            move_coords = (3 + i, 3)
            board.apply_move(player, move_coords)

        # Results should be monotonically increasing (based on mock logic)
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]


@pytest.mark.katago
class TestRealKataGo:
    """Tests using real KataGo binary (requires KATAGO_BIN setup).

    Run with: pytest tests/test_katago_integration.py -m katago

    Skip with: pytest tests/test_katago_integration.py -m "not katago"
    """

    def test_real_katago_startup(self, skip_if_no_katago, real_katago_config):
        """Test that real KataGo can start and close."""
        client = KataGoAnalysisClient(real_katago_config)

        try:
            client.start()
            # If we get here, KataGo started successfully
            assert client._process is not None
        finally:
            client.close()
            assert client._process is None

    def test_real_katago_analysis(self, skip_if_no_katago, real_katago_config):
        """Test analyzing a position with real KataGo."""
        client = KataGoAnalysisClient(real_katago_config)
        board = BoardState(size=9)

        try:
            client.start()
            result = client.analyze_position(board)

            # Verify result structure
            assert result is not None
            assert 0.0 <= result.winrate <= 1.0
            assert result.raw is not None
            assert "rootInfo" in result.raw
            assert "winrate" in result.raw["rootInfo"]

            # Empty board should be roughly even
            assert 0.4 <= result.winrate <= 0.6

        finally:
            client.close()

    def test_real_katago_evaluate_moves(
        self, skip_if_no_katago, real_katago_config
    ):
        """Test evaluating multiple moves with real KataGo."""
        client = KataGoAnalysisClient(real_katago_config)
        board = BoardState(size=9)

        try:
            client.start()

            # Test evaluating opening moves
            moves = ["D4", "E5", "C3"]
            results = client.evaluate_moves(board, moves, player="B")

            # Verify results
            assert len(results) == len(moves)
            for move in moves:
                assert move in results
                assert 0.0 <= results[move] <= 1.0

        finally:
            client.close()

    def test_real_katago_game_sequence(
        self, skip_if_no_katago, real_katago_config
    ):
        """Test analyzing a sequence of moves."""
        client = KataGoAnalysisClient(real_katago_config)
        board = BoardState(size=9)

        try:
            client.start()

            # Play a short game and analyze at each step
            moves = [
                ("B", "D4"),
                ("W", "F6"),
                ("B", "C7"),
                ("W", "G3"),
            ]

            winrates = []
            for player, move in moves:
                board.apply_move(player, move)
                result = client.analyze_position(board)
                winrates.append(result.winrate)

            # All winrates should be valid
            for wr in winrates:
                assert 0.0 <= wr <= 1.0

            # Winrates should vary as position changes
            # (but we can't predict exact values)
            assert len(set(winrates)) > 1  # Not all the same

        finally:
            client.close()


class TestKataGoErrorHandling:
    """Tests for error handling in KataGo client."""

    def test_analyze_without_start(self, mock_katago_config):
        """Test that analyzing without starting client raises error."""
        # Use a real client (not mocked) to test error handling
        client = KataGoAnalysisClient(mock_katago_config)
        board = BoardState(size=9)

        # Should raise error because process is not started
        with pytest.raises(RuntimeError, match="KataGo process not started"):
            # Directly call _send to trigger the error
            # (analyze_position would be mocked in mock_katago_client)
            client._send({"test": "data"})

    def test_double_start(self, mock_katago_client):
        """Test that starting client twice is safe."""
        mock_katago_client.start()
        mock_katago_client.start()  # Should not raise

        # Start should only be called once due to check
        assert mock_katago_client.start.call_count == 2

    def test_close_without_start(self, mock_katago_client):
        """Test that closing without starting is safe."""
        # Should not raise
        mock_katago_client.close()


def test_katago_config_validation():
    """Test that KataGo config validates paths."""
    from go_rl.config import KataGoConfig

    # Should raise on placeholder paths
    with pytest.raises(ValueError, match="must be set to a real path"):
        KataGoConfig(
            binary_path="/path/to/katago",
            model_path="/path/to/model",
            config_path="/path/to/config",
        )


def test_katago_result_structure(mock_katago_client, sample_board_9x9):
    """Test that KataGoResult has expected structure."""
    result = mock_katago_client.analyze_position(sample_board_9x9)

    # Test attributes exist
    assert hasattr(result, "winrate")
    assert hasattr(result, "raw")

    # Test types
    assert isinstance(result.winrate, float)
    assert isinstance(result.raw, dict)
