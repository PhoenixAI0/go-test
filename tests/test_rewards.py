import pytest

from go_rl.board import BoardState
from go_rl.config import RewardConfig
from go_rl.rewards import evaluate_move, length_reward, outcome_reward, winrate_reward


def test_length_reward_rules():
    config = RewardConfig(length_penalty=-0.25)
    assert length_reward("PASS", config) == 0.0
    assert length_reward("A1", config) == 0.0
    assert length_reward("ABCD", config) == -0.25  # len("ABCD") = 4, > 3, so penalized


@pytest.mark.parametrize(
    "move,length_penalty,expected",
    [
        ("PASS", -0.1, 0.0),  # PASS always returns 0
        ("A1", -0.1, 0.0),  # len=2, not penalized
        ("A9", -0.1, 0.0),  # len=2, not penalized
        ("J1", -0.1, 0.0),  # len=2, not penalized
        ("A10", -0.1, 0.0),  # len=3, not penalized (<=3)
        ("D4", -0.5, 0.0),  # len=2, not penalized
        ("K10", -0.5, 0.0),  # len=3, not penalized (<=3)
        ("pass", -0.25, -0.25),  # lowercase pass is len=4, penalized
        ("PASS", -0.1, 0.0),  # PASS
        ("  A1  ", -0.1, 0.0),  # Whitespace stripped, len=2
        ("ABCD", -0.25, -0.25),  # len=4, penalized
        ("A10B", -0.1, -0.1),  # len=4, penalized
    ],
)
def test_length_reward_parametrized(move, length_penalty, expected):
    """Test length reward calculation for various moves (penalizes len > 3 after strip)."""
    config = RewardConfig(length_penalty=length_penalty)
    result = length_reward(move, config)
    assert result == expected, f"length_reward({move!r}) = {result}, expected {expected}"


def test_outcome_reward_rules():
    config = RewardConfig(full_game_bonus=0.5)
    assert outcome_reward(None, config) == 0.0
    assert outcome_reward(1.0, config) == 0.5
    assert outcome_reward(-1.0, config) == -0.5


@pytest.mark.parametrize(
    "outcome,full_game_bonus,expected",
    [
        (None, 0.5, 0.0),  # No outcome
        (1.0, 0.5, 0.5),  # Win
        (-1.0, 0.5, -0.5),  # Loss
        (1.0, 1.0, 1.0),  # Win with higher bonus
        (-1.0, 1.0, -1.0),  # Loss with higher penalty
        (1.0, 0.0, 0.0),  # Win with no bonus
        (-1.0, 0.0, 0.0),  # Loss with no penalty
        (0.0, 0.5, 0.0),  # Draw (treated as no outcome)
    ],
)
def test_outcome_reward_parametrized(outcome, full_game_bonus, expected):
    """Test outcome reward calculation for various scenarios."""
    config = RewardConfig(full_game_bonus=full_game_bonus)
    assert outcome_reward(outcome, config) == expected


@pytest.mark.parametrize(
    "winrate_delta,winrate_scale,expected",
    [
        (0.0, 1.0, 0.0),
        (0.1, 1.0, 0.1),
        (-0.1, 1.0, -0.1),
        (0.5, 1.0, 0.5),
        (0.1, 2.0, 0.2),  # Scaled by 2x
        (0.1, 0.5, 0.05),  # Scaled by 0.5x
        (-0.3, 2.0, -0.6),  # Negative delta scaled
    ],
)
def test_winrate_reward_parametrized(winrate_delta, winrate_scale, expected):
    """Test winrate reward calculation with various scales."""
    config = RewardConfig(winrate_scale=winrate_scale)
    assert winrate_reward(winrate_delta, config) == expected


def test_evaluate_move_illegal_sets_penalty():
    config = RewardConfig(legal_bonus=0.2, illegal_penalty=-1.0)
    board = BoardState.empty(5)
    breakdown = evaluate_move(board, "Z99", winrate_delta=0.1, config=config)
    assert breakdown.legality == -1.0
    assert breakdown.winrate == 0.1


def test_evaluate_move_strict_legality():
    config = RewardConfig(legal_bonus=0.2, illegal_penalty=-1.0, strict_legality=True)
    board = BoardState.empty(3)
    board.apply_move("W", "A2")
    board.apply_move("W", "B1")
    board.apply_move("W", "C2")
    board.apply_move("W", "B3")
    breakdown = evaluate_move(board, "B2", winrate_delta=0.0, config=config)
    assert breakdown.legality == -1.0


def test_evaluate_move_legal_gets_bonus():
    """Test that legal moves get the legal bonus."""
    config = RewardConfig(legal_bonus=0.5, illegal_penalty=-1.0)
    board = BoardState.empty(9)
    breakdown = evaluate_move(board, "D4", winrate_delta=0.0, config=config)
    assert breakdown.legality == 0.5


def test_evaluate_move_total_reward():
    """Test that total reward is sum of all components."""
    config = RewardConfig(
        winrate_scale=1.0,
        legal_bonus=0.2,
        illegal_penalty=-1.0,
        length_penalty=-0.1,
        full_game_bonus=0.5,
    )
    board = BoardState.empty(9)
    breakdown = evaluate_move(
        board,
        "D4",
        winrate_delta=0.3,
        game_result=None,
        config=config,
    )

    # winrate: 0.3, legality: 0.2, length: 0.0, outcome: 0.0
    expected_total = 0.3 + 0.2 + 0.0 + 0.0
    assert breakdown.total == expected_total
    assert breakdown.winrate == 0.3
    assert breakdown.legality == 0.2
    assert breakdown.length == 0.0
    assert breakdown.outcome == 0.0


def test_evaluate_move_with_all_components():
    """Test evaluation with all reward components non-zero."""
    config = RewardConfig(
        winrate_scale=2.0,
        legal_bonus=0.5,
        illegal_penalty=-2.0,
        length_penalty=-0.25,
        full_game_bonus=1.0,
    )
    board = BoardState.empty(19)
    breakdown = evaluate_move(
        board,
        "ABCD",  # Long move (len=4, > 3, so penalized)
        winrate_delta=0.2,
        game_result=1.0,  # Win
        config=config,
    )

    # winrate: 0.2 * 2.0 = 0.4
    # legality: -2.0 (illegal move "ABCD")
    # length: -0.25 (len > 3)
    # outcome: 1.0 (win)
    assert breakdown.winrate == 0.4
    assert breakdown.legality == -2.0  # Illegal move
    assert breakdown.length == -0.25
    assert breakdown.outcome == 1.0
    assert breakdown.total == 0.4 + (-2.0) - 0.25 + 1.0


@pytest.mark.parametrize(
    "move,is_legal,winrate_delta,game_result",
    [
        ("D4", True, 0.1, None),
        ("PASS", True, 0.0, None),
        ("F6", True, 0.3, 1.0),  # Use valid 9x9 move
        ("Z99", False, 0.1, None),  # Illegal move
    ],
)
def test_evaluate_move_various_scenarios(move, is_legal, winrate_delta, game_result):
    """Test evaluate_move with various scenarios."""
    config = RewardConfig(
        legal_bonus=0.2,
        illegal_penalty=-1.0,
        length_penalty=-0.1,
        full_game_bonus=0.5,
    )
    board = BoardState.empty(9)
    breakdown = evaluate_move(
        board,
        move,
        winrate_delta=winrate_delta,
        game_result=game_result,
        config=config,
    )

    # Check that legality is correctly assigned
    if is_legal:
        assert breakdown.legality == 0.2
    else:
        assert breakdown.legality == -1.0

    # Check winrate is assigned
    assert breakdown.winrate == winrate_delta

    # Check that total includes all components
    assert breakdown.total == (
        breakdown.winrate + breakdown.legality + breakdown.length + breakdown.outcome
    )
