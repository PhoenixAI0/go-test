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

    @pytest.mark.parametrize("board_size", [9, 19])
    def test_real_katago_analysis(self, skip_if_no_katago, real_katago_config, board_size):
        client = KataGoAnalysisClient(real_katago_config)
        board = BoardState(size=board_size)

        try:
            client.start()
            result = client.analyze_position(board)

            assert result is not None
            assert 0.0 <= result.winrate <= 1.0
            assert result.raw is not None
            assert "rootInfo" in result.raw
            assert "winrate" in result.raw["rootInfo"]

            # Komi and board size can strongly favor one side on an empty board.
            assert result.score_lead is not None
            assert result.best_move is not None

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


@pytest.mark.katago
def test_fullsize_legal_masks_match_real_katago(skip_if_no_katago, real_katago_config):
    import random
    from go_rl.board import coords_to_gtp

    rng = random.Random(404)
    board = BoardState.empty(19)
    samples = []
    for ply in range(321):
        if ply in {0, 20, 80, 160, 240, 320}:
            samples.append(board.clone())
        legal = board.legal_moves(superko=True, include_pass=False)
        board.apply_move(board.next_player(), rng.choice(legal) if legal else "PASS", strict=True)
    client = KataGoAnalysisClient(real_katago_config)
    try:
        client.start()
        analyses = client.analyze_positions(samples, max_visits=1, include_policy=True,
                                            rules="koPOSITIONALscoreAREAtaxNONEsui0", komi=7.5)
        for position, analysis in zip(samples, analyses):
            oracle = {coords_to_gtp(divmod(index, 19), 19) if index < 361 else "PASS"
                      for index, probability in enumerate(analysis.raw["policy"]) if probability >= 0}
            assert oracle == set(position.legal_moves(superko=True)), len(position.moves)
    finally:
        client.close()


@pytest.mark.katago
def test_snapshot_ko_ban_reaches_real_katago(skip_if_no_katago, real_katago_config):
    board = BoardState.empty(19)
    for color, move in [("B", "B3"), ("B", "D3"), ("B", "C4"),
                        ("W", "C3"), ("W", "B2"), ("W", "C1"), ("W", "D2")]:
        board.apply_move(color, move, strict=True)
    previous = board.current_board()
    board.apply_move("B", "C2", strict=True)
    snapshot = BoardState.from_matrix(19, board.current_board(), "W", previous)
    client = KataGoAnalysisClient(real_katago_config)
    try:
        client.start()
        original, restored = client.analyze_positions([board, snapshot], max_visits=1,
                                                       include_policy=True)
        recapture_action = (19 - 3) * 19 + 2  # C3, row-major from A19.
        assert original.raw["policy"][recapture_action] == -1
        assert restored.raw["policy"][recapture_action] == -1
        assert [value >= 0 for value in original.raw["policy"]] == [
            value >= 0 for value in restored.raw["policy"]]
    finally:
        client.close()
