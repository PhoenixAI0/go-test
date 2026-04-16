"""Integration tests for the KataGo analysis client."""

from __future__ import annotations

import pytest

from go_rl.board import BoardState
from go_rl.katago import KataGoAnalysisClient


class TestMockedKataGo:
    def test_analyze_empty_board(self, mock_katago_client, sample_board_9x9):
        empty_board = BoardState(size=9)
        result = mock_katago_client.analyze_position(empty_board)

        assert result is not None
        assert 0.0 <= result.winrate <= 1.0
        assert result.raw is not None

    def test_analyze_board_with_moves(self, mock_katago_client, sample_board_9x9):
        result = mock_katago_client.analyze_position(sample_board_9x9)

        assert result is not None
        assert 0.0 <= result.winrate <= 1.0
        assert result.winrate > 0.5

    def test_evaluate_multiple_moves(self, mock_katago_client, sample_board_9x9):
        moves = ["D5", "E4", "F3"]
        results = mock_katago_client.evaluate_moves(
            sample_board_9x9, moves, player="B"
        )

        assert len(results) == 3
        for move in moves:
            assert move in results
            assert 0.0 <= results[move] <= 1.0

    def test_client_lifecycle(self, mock_katago_client):
        mock_katago_client.start()
        mock_katago_client.close()

        mock_katago_client.start.assert_called_once()
        mock_katago_client.close.assert_called_once()

    def test_winrate_progression(self, mock_katago_client):
        board = BoardState(size=9)

        results = []
        for i in range(5):
            result = mock_katago_client.analyze_position(board)
            results.append(result.winrate)

            player = "B" if i % 2 == 0 else "W"
            move_coords = (3 + i, 3)
            board.apply_move(player, move_coords)

        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]


@pytest.mark.katago
class TestRealKataGo:
    def test_real_katago_startup(self, skip_if_no_katago, real_katago_config):
        client = KataGoAnalysisClient(real_katago_config)

        try:
            client.start()
            assert client._process is not None
        finally:
            client.close()
            assert client._process is None

    def test_real_katago_analysis(self, skip_if_no_katago, real_katago_config):
        client = KataGoAnalysisClient(real_katago_config)
        board = BoardState(size=9)

        try:
            client.start()
            result = client.analyze_position(board)

            assert result is not None
            assert 0.0 <= result.winrate <= 1.0
            assert result.raw is not None
            assert "rootInfo" in result.raw
            assert "winrate" in result.raw["rootInfo"]

            assert 0.4 <= result.winrate <= 0.6

        finally:
            client.close()

    def test_real_katago_evaluate_moves(
        self, skip_if_no_katago, real_katago_config
    ):
        client = KataGoAnalysisClient(real_katago_config)
        board = BoardState(size=9)

        try:
            client.start()

            moves = ["D4", "E5", "C3"]
            results = client.evaluate_moves(board, moves, player="B")

            assert len(results) == len(moves)
            for move in moves:
                assert move in results
                assert 0.0 <= results[move] <= 1.0

        finally:
            client.close()

    def test_real_katago_game_sequence(
        self, skip_if_no_katago, real_katago_config
    ):
        client = KataGoAnalysisClient(real_katago_config)
        board = BoardState(size=9)

        try:
            client.start()

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

            for wr in winrates:
                assert 0.0 <= wr <= 1.0

            assert len(set(winrates)) > 1

        finally:
            client.close()


class TestKataGoErrorHandling:
    def test_analyze_without_start(self, mock_katago_config):
        client = KataGoAnalysisClient(mock_katago_config)

        with pytest.raises(RuntimeError, match="KataGo process not started"):
            client._send({"test": "data"})

    def test_double_start(self, mock_katago_client):
        mock_katago_client.start()
        mock_katago_client.start()

        assert mock_katago_client.start.call_count == 2

    def test_close_without_start(self, mock_katago_client):
        mock_katago_client.close()


def test_katago_config_validation():
    from go_rl.config import KataGoConfig

    with pytest.raises(ValueError, match="must be set to a real path"):
        KataGoConfig(
            binary_path="/path/to/katago",
            model_path="/path/to/model",
            config_path="/path/to/config",
        )


def test_katago_result_structure(mock_katago_client, sample_board_9x9):
    result = mock_katago_client.analyze_position(sample_board_9x9)

    assert hasattr(result, "winrate")
    assert hasattr(result, "raw")

    assert isinstance(result.winrate, float)
    assert isinstance(result.raw, dict)
