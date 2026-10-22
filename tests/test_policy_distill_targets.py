import importlib.util
from pathlib import Path

import pytest

from go_rl.board import BoardState
from go_rl.data_prep import state_record_from_board

spec = importlib.util.spec_from_file_location("policy_distill_targets", Path(__file__).resolve().parents[1] / "scripts/run_policy_distill.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
prepare_records = module.prepare_records


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]


def test_hard_policy_argmax_uses_same_distribution_not_searched_best_move():
    target = [.03] * 10
    target[4] = .73
    record = {**state_record_from_board(BoardState.empty(3)), "game_id": "g", "rules": "chinese", "komi": 7.5,
              "response": "A3", "policy_target": target}
    soft = prepare_records([record], Tokenizer(), "compact", "policy", 1024, 3)[0]
    hard_policy = prepare_records([record], Tokenizer(), "compact", "policy_argmax", 1024, 3)[0]
    hard_move = prepare_records([record], Tokenizer(), "compact", "move", 1024, 3)[0]
    assert soft["target"] == target
    assert hard_policy["target"][4] == 1 and sum(hard_policy["target"]) == 1
    assert hard_move["target"][0] == 1
    assert hard_policy["action"] == 0  # Diagnostic agreement still uses searched reference.


@pytest.mark.parametrize("target", [[0.] * 10, [float("nan")] * 10, [1.]])
def test_hard_policy_target_validates_distribution(target):
    record = {**state_record_from_board(BoardState.empty(3)), "game_id": "g", "rules": "chinese", "komi": 7.5,
              "response": "A3", "policy_target": target}
    with pytest.raises(ValueError, match="policy_argmax"):
        prepare_records([record], Tokenizer(), "compact", "policy_argmax", 1024, 3)
