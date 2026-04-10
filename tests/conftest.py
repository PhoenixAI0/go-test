"""Pytest configuration and fixtures for Go RL tests."""

from __future__ import annotations

import os
from typing import Dict
from unittest.mock import Mock, patch

import pytest

from go_rl.board import BoardState
from go_rl.config import KataGoConfig
from go_rl.katago import KataGoAnalysisClient, KataGoResult


@pytest.fixture
def mock_katago_config() -> KataGoConfig:
    """Create a mock KataGo config for testing."""
    return KataGoConfig(
        binary_path="/mock/katago",
        model_path="/mock/model.bin.gz",
        config_path="/mock/config.cfg",
        board_size=9,
        rules="chinese",
        komi=7.5,
    )


@pytest.fixture
def mock_katago_response() -> Dict[str, object]:
    """Create a mock KataGo analysis response."""
    return {
        "id": 1,
        "turnNumber": 0,
        "rootInfo": {
            "winrate": 0.55,
            "scoreLead": 2.5,
            "utility": 0.1,
            "visits": 100,
        },
        "moveInfos": [
            {
                "move": "Q4",
                "winrate": 0.56,
                "scoreMean": 3.0,
                "visits": 80,
            }
        ],
    }


@pytest.fixture
def mock_katago_client(mock_katago_config, mock_katago_response):
    """Create a mock KataGo client that returns deterministic results.

    This fixture mocks the KataGoAnalysisClient to return fixed winrates
    without requiring a real KataGo binary.
    """
    client = Mock(spec=KataGoAnalysisClient)
    client.config = mock_katago_config

    # Mock analyze_position to return deterministic results
    def mock_analyze(board: BoardState) -> KataGoResult:
        # Return different winrates based on move count for variety
        base_winrate = 0.5
        move_bonus = len(board.stones()) * 0.01
        winrate = min(0.95, base_winrate + move_bonus)

        return KataGoResult(
            winrate=winrate,
            raw=mock_katago_response,
        )

    client.analyze_position = Mock(side_effect=mock_analyze)

    # Mock evaluate_moves
    def mock_evaluate(board: BoardState, moves, player=None):
        results = {}
        for i, move in enumerate(moves):
            # Give different winrates to different moves
            results[move] = 0.5 + (i * 0.05)
        return results

    client.evaluate_moves = Mock(side_effect=mock_evaluate)

    # Mock start/close
    client.start = Mock()
    client.close = Mock()

    return client


@pytest.fixture
def real_katago_config() -> KataGoConfig | None:
    """Create a real KataGo config from environment variables.

    Returns None if KataGo is not configured (allows skipping tests).
    """
    binary_path = os.environ.get("KATAGO_BIN")
    model_path = os.environ.get("KATAGO_MODEL")
    config_path = os.environ.get("KATAGO_CONFIG")

    if not all([binary_path, model_path, config_path]):
        return None

    # Check if paths actually exist
    if not os.path.exists(binary_path):
        return None

    try:
        return KataGoConfig(
            binary_path=binary_path,
            model_path=model_path,
            config_path=config_path,
            board_size=9,
            rules="chinese",
            komi=7.5,
        )
    except Exception:
        return None


@pytest.fixture
def skip_if_no_katago(real_katago_config):
    """Skip test if KataGo is not available.

    Usage:
        def test_something(skip_if_no_katago):
            # Test code that requires KataGo
            pass
    """
    if real_katago_config is None:
        pytest.skip("KataGo not available (set KATAGO_BIN, KATAGO_MODEL, KATAGO_CONFIG)")


@pytest.fixture
def sample_board_9x9() -> BoardState:
    """Create a sample 9x9 board with some moves."""
    board = BoardState(size=9)
    # Play a few opening moves
    board.apply_move("B", "D4")
    board.apply_move("W", "F6")
    board.apply_move("B", "C7")
    return board


@pytest.fixture
def sample_board_19x19() -> BoardState:
    """Create a sample 19x19 board with some moves."""
    board = BoardState(size=19)
    # Play a few opening moves
    board.apply_move("B", "Q16")
    board.apply_move("W", "D4")
    board.apply_move("B", "Q4")
    board.apply_move("W", "D16")
    return board
