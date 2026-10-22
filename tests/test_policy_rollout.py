import json

import pytest
import torch
from transformers import Qwen3Config, Qwen3Model

from go_rl.board import BoardState
from go_rl.data_prep import board_from_example
from go_rl.learner_data import episode_training_records
from go_rl.policy_model import GoPolicyValueModel
from go_rl.policy_optimization import categorical_statistics
from go_rl.policy_rollout import (
    collect_selfplay_episode, optimize_selfplay_episodes, referee_black_outcome,
)


class TinyTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        # Include current state in the tiny observation without needing hub files.
        text = messages[0]["content"]
        return [1, 2 if "to_play: B" in text else 3, 4 + text.count("X") % 16]


class FakeReferee:
    def __init__(self, score="B+1.5"):
        self.commands = []
        self.score = score

    def send(self, command, timeout=600):
        self.commands.append(command)
        return self.score if command == "final_score" else ""


def tiny_model():
    torch.manual_seed(123)
    return GoPolicyValueModel(Qwen3Model(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        pad_token_id=0, attention_dropout=0.0,
    )), board_size=3)


def sample_actions(monkeypatch, actions):
    choices = iter(actions)
    monkeypatch.setattr(torch, "multinomial", lambda probabilities, n: torch.tensor([next(choices)], device=probabilities.device))


def test_full_game_keeps_both_players_and_scores_only_after_two_passes(monkeypatch):
    model, referee = tiny_model(), FakeReferee("W+2.5")
    sample_actions(monkeypatch, [0, 9, 9])
    episode = collect_selfplay_episode(model, TinyTokenizer(), referee, max_plies=5, temperature=.7)
    assert [step.player for step in episode.transitions] == ["B", "W", "B"]
    assert [step.reward for step in episode.transitions] == [0, 0, -1]
    assert episode.transitions[-1].terminated and not episode.transitions[-1].truncated
    assert episode.black_outcome == -1 and episode.outcome_source == "katago_gtp_final_score"
    assert episode.raw_board_result == "B+1.5"  # Does not replace the referee reward.
    assert referee.commands[-3:] == ["play w PASS", "play b PASS", "final_score"]
    rules_command = next(command for command in referee.commands if command.startswith("kata-set-rules"))
    assert json.loads(rules_command.split(" ", 1)[1])["ko"] == "POSITIONAL"
    for step in episode.transitions:
        output = model(torch.tensor([step.input_ids]), torch.ones((1, len(step.input_ids)), dtype=torch.long))
        logp, _ = categorical_statistics(output.logits, torch.tensor([step.action]),
                                        torch.tensor([step.legal_mask]), temperature=.7)
        assert step.old_log_prob == pytest.approx(logp.item(), abs=1e-7)


def test_cap_bootstraps_without_scored_win_or_terminal_reward(monkeypatch):
    model, referee = tiny_model(), FakeReferee()
    with torch.no_grad():
        model.value_head.bias.fill_(.4)
    sample_actions(monkeypatch, [0, 1])
    episode = collect_selfplay_episode(model, TinyTokenizer(), referee, max_plies=2)
    assert len(episode.transitions) == 2
    assert episode.black_outcome is None and episode.final_score is None
    assert episode.raw_board_result is None
    assert "final_score" not in referee.commands
    assert episode.transitions[-1].truncated and not episode.transitions[-1].terminated
    assert episode.transitions[-1].next_value == pytest.approx(torch.tensor(.4).tanh().item())
    assert all(step.reward == 0 for step in episode.transitions)


def test_real_optimizer_updates_all_collected_model_turns(monkeypatch):
    model = tiny_model()
    sample_actions(monkeypatch, [0, 9, 9])
    episode = collect_selfplay_episode(model, TinyTokenizer(), FakeReferee(), max_plies=3)
    before = model.policy_head.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005)
    metrics = optimize_selfplay_episodes(model, TinyTokenizer(), optimizer, [episode],
                                        minibatch_size=2, epochs=1, gae_lambda=1.0)
    assert metrics["collected_moves"] == metrics["optimized_move_samples"] == 3
    assert metrics["optimizer_steps"] == 2
    assert metrics["return_mean"] == pytest.approx(1 / 3)
    assert torch.isfinite(model.policy_head.weight).all()
    assert not torch.equal(before, model.policy_head.weight)


def test_referee_rejection_aborts_instead_of_fabricating_a_reward(monkeypatch):
    class RejectRules(FakeReferee):
        def send(self, command, timeout=600):
            if command.startswith("kata-set-rules"):
                raise RuntimeError("unsupported rules")
            return super().send(command, timeout)

    with pytest.raises(RuntimeError, match="unsupported rules"):
        collect_selfplay_episode(tiny_model(), TinyTokenizer(), RejectRules())


@pytest.mark.parametrize("rules", [
    "japanese", "chinese",
    {"ko": "POSITIONAL", "scoring": "AREA", "tax": "NONE", "hasButton": True},
    {"ko": "POSITIONAL", "scoring": "AREA", "tax": "NONE", "friendlyPassOk": True},
])
def test_unsupported_endpoint_rules_fail_before_model_or_referee_work(rules):
    class NoForward:
        def __getattr__(self, name):
            raise AssertionError("Unsupported terminal rules must fail before model access")

    referee = FakeReferee()
    with pytest.raises(ValueError):
        collect_selfplay_episode(NoForward(), TinyTokenizer(), referee, rules=rules)
    assert referee.commands == []


@pytest.mark.parametrize("score", ["Void", "?", "B+nan", "W+-1", "B+0", "B+"])
def test_unscored_referee_result_cannot_become_an_outcome(score):
    with pytest.raises(ValueError, match="result|margin"):
        referee_black_outcome(score)


def test_adjudicated_scores_draw_and_resign():
    assert referee_black_outcome("0") == 0
    assert referee_black_outcome("W+R") == -1
    assert referee_black_outcome("B+1.5") == 1


def test_seed_history_is_replayed_and_prior_pass_can_terminate_after_one_model_move(monkeypatch):
    start = BoardState.empty(3)
    start.apply_move("B", "A3", strict=True)
    start.apply_move("W", "PASS", strict=True)
    sample_actions(monkeypatch, [9])
    referee = FakeReferee("W+2.5")
    episode = collect_selfplay_episode(tiny_model(), TinyTokenizer(), referee,
                                       start_board=start, max_plies=4)
    assert len(start.moves) == 2 and start.consecutive_passes() == 1
    assert len(episode.transitions) == 1 and episode.moves == [("B", "PASS")]
    assert episode.transitions[0].terminated and episode.transitions[0].reward == -1.
    assert episode.summary()["starting_plies"] == 2
    assert referee.commands[-4:] == ["play b A3", "play w PASS", "play b PASS", "final_score"]
    summary = {**episode.summary(), "seed_game_id": "training-game", "update": 0, "episode": 0}
    rows = episode_training_records(summary, {"board_size": 3, "rules": "chinese", "komi": 7.5},
                                    source_run_id="seeded-run", episode_row=0)
    assert len(rows) == 1 and rows[0]["moves"] == [["B", "A3"], ["W", "PASS"]]
    assert rows[0]["learner"]["ply"] == 2 and rows[0]["learner"]["rollout_step"] == 0
    assert rows[0]["learner"]["seed_game_id"] == "training-game"
    assert board_from_example(rows[0]).consecutive_passes() == 1


def test_seeded_cap_optimizes_only_new_turns_and_keeps_seed_history_for_export(monkeypatch):
    start = BoardState.empty(3)
    for player, move in [("B", "A3"), ("W", "B2"), ("B", "C1")]:
        start.apply_move(player, move, strict=True)
    sample_actions(monkeypatch, [1, 9])
    referee = FakeReferee()
    episode = collect_selfplay_episode(tiny_model(), TinyTokenizer(), referee,
                                       start_board=start, max_plies=2)
    assert [step.player for step in episode.transitions] == ["W", "B"]
    assert episode.black_outcome is None and episode.transitions[-1].truncated
    assert len(episode.moves) == len(episode.transitions) == 2
    assert len(episode.start_position["moves"]) == 3
    assert "final_score" not in referee.commands
    rows = episode_training_records(episode.summary(), {"board_size": 3, "rules": "chinese", "komi": 7.5},
                                    source_run_id="seeded-run", episode_row=0)
    assert [len(row["moves"]) for row in rows] == [3, 4]
    assert all("source_result" not in row for row in rows)


def test_invalid_seed_aborts_before_referee_or_model_work():
    referee = FakeReferee()
    start = BoardState.from_matrix(3, [[1, 0, 0], [0, 0, 0], [0, 0, 0]], "W")
    with pytest.raises(ValueError, match="initial setup"):
        collect_selfplay_episode(tiny_model(), TinyTokenizer(), referee, start_board=start)
    assert referee.commands == []
