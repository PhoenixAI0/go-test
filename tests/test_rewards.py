from go_rl.board import BoardState
from go_rl.config import RewardConfig
from go_rl.rewards import evaluate_move, length_reward, outcome_reward


def test_length_reward_rules():
    config = RewardConfig(length_penalty=-0.25)
    assert length_reward("PASS", config) == 0.0
    assert length_reward("A1", config) == 0.0
    assert length_reward("Q16", config) == -0.25


def test_outcome_reward_rules():
    config = RewardConfig(full_game_bonus=0.5)
    assert outcome_reward(None, config) == 0.0
    assert outcome_reward(1.0, config) == 0.5
    assert outcome_reward(-1.0, config) == -0.5


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
