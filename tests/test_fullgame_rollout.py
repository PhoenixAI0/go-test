from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import pytest

from go_rl.board import BoardState, gtp_to_coords
from go_rl.config import FullGameConfig, RewardConfig, RolloutConfig
from go_rl.fullgame_rollout import (
    align_board_to_player_turn,
    choose_model_color,
    scorelead_from_model_perspective,
    simulate_full_game,
)
from go_rl.katago import KataGoResult


@dataclass
class _DummyModel:
    device: str = "cpu"


class _DummyTokenizer:
    eos_token_id = 0

    def __call__(self, *_args, **_kwargs):
        raise RuntimeError("Tokenizer should not be called in this test scenario")

    def decode(self, *_args, **_kwargs):
        return "PASS"


class _StubKataGo:
    def __init__(self, score_lead: float = 0.0, move: str = "PASS"):
        self.score_lead = score_lead
        self.move = move

    def analyze_position(self, _board: BoardState) -> KataGoResult:
        return KataGoResult(
            winrate=0.5,
            raw={"rootInfo": {"winrate": 0.5, "scoreLead": self.score_lead}},
            score_lead=self.score_lead,
            best_move=self.move,
        )

    def choose_move(self, _board: BoardState, player: str | None = None) -> str:
        return self.move


def test_scorelead_from_model_perspective():
    assert scorelead_from_model_perspective(3.0, "B") == 3.0
    assert scorelead_from_model_perspective(3.0, "W") == -3.0
    assert scorelead_from_model_perspective(None, "B") == 0.0


def test_choose_model_color_alternate_deterministic():
    board = BoardState.empty(19)
    cfg = FullGameConfig(model_color_mode="alternate")
    assert choose_model_color(board, cfg, episode_index=0) == "B"

    board.apply_move("B", "D4")
    assert choose_model_color(board, cfg, episode_index=1) == "W"


def test_choose_model_color_honors_forced_modes():
    board = BoardState.empty(19)
    board.apply_move("B", "D4")  # W to move
    assert choose_model_color(board, FullGameConfig(model_color_mode="black"), 0) == "B"
    assert choose_model_color(board, FullGameConfig(model_color_mode="white"), 0) == "W"


def test_align_board_to_player_turn_adds_pass_when_needed():
    board = BoardState.empty(19)
    aligned = align_board_to_player_turn(board, "W")
    assert aligned.next_player() == "W"
    assert len(aligned.moves) == 1
    assert aligned.moves[0][0] == "B"
    assert aligned.moves[0][1] is None


def test_align_board_to_player_turn_noop_when_already_aligned():
    board = BoardState.empty(19)
    aligned = align_board_to_player_turn(board, "B")
    assert aligned.moves == board.moves
    assert aligned.next_player() == "B"


def test_align_board_to_player_turn_rejects_unknown_color():
    board = BoardState.empty(19)
    with pytest.raises(ValueError, match="Unknown player color"):
        align_board_to_player_turn(board, "X")


def test_simulate_full_game_two_pass_termination():
    board = BoardState.empty(19)
    katago = _StubKataGo(score_lead=0.0, move="PASS")
    result = simulate_full_game(
        board=board,
        initial_completion="PASS",
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        katago=katago,
        reward_cfg=RewardConfig(strict_legality=True),
        rollout_cfg=RolloutConfig(),
        fullgame_cfg=FullGameConfig(max_plies=50, max_model_turns=10),
        model_color="B",
    )

    assert result.pass_terminated is True
    assert result.model_turns >= 1
    assert result.plies_played == 2
    assert result.final_outcome is None
    assert result.terminal_reward == 0.0
    assert result.outcome_source == "katago_estimate"


@pytest.mark.parametrize("continue_after_endpoint", [False, True])
def test_fullgame_rejects_root_at_or_past_two_pass_endpoint(continue_after_endpoint):
    board = BoardState.empty(19)
    board.apply_move("B", "PASS")
    board.apply_move("W", "PASS")
    if continue_after_endpoint:
        board.apply_move("B", "D4")
    before = list(board.moves)
    with pytest.raises(ValueError, match="root must precede"):
        simulate_full_game(
            board=board, initial_completion="PASS", model=_DummyModel(), tokenizer=_DummyTokenizer(),
            katago=object(), reward_cfg=RewardConfig(), rollout_cfg=RolloutConfig(),
            fullgame_cfg=FullGameConfig(), model_color=board.next_player())
    assert board.moves == before


def test_simulate_full_game_rejects_mismatched_model_turn():
    board = BoardState.empty(19)
    board.apply_move("B", "D4")  # W to move
    katago = _StubKataGo(score_lead=0.0, move="PASS")
    with pytest.raises(ValueError, match="does not match model_color"):
        simulate_full_game(
            board=board,
            initial_completion="PASS",
            model=_DummyModel(),
            tokenizer=_DummyTokenizer(),
            katago=katago,
            reward_cfg=RewardConfig(strict_legality=True),
            rollout_cfg=RolloutConfig(),
            fullgame_cfg=FullGameConfig(max_plies=50, max_model_turns=10),
            model_color="B",
        )


def test_simulate_full_game_truncates_on_max_plies():
    board = BoardState.empty(19)
    katago = _StubKataGo(score_lead=1.0, move="Q16")
    result = simulate_full_game(
        board=board,
        initial_completion="D4",
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        katago=katago,
        reward_cfg=RewardConfig(strict_legality=True),
        rollout_cfg=RolloutConfig(),
        fullgame_cfg=FullGameConfig(max_plies=1, max_model_turns=10),
        model_color="B",
    )

    assert result.truncated is True
    assert result.plies_played == 1
    assert result.final_outcome is None
    assert result.terminal_reward == 0.0
    assert result.estimated_outcome == 1.0


def test_simulate_full_game_illegal_move_ends_game():
    board = BoardState.empty(19)
    katago = _StubKataGo(score_lead=0.0, move="PASS")
    result = simulate_full_game(
        board=board,
        initial_completion="Z99",
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        katago=katago,
        reward_cfg=RewardConfig(
            illegal_penalty=-2.0,
            strict_legality=True,
            illegal_move_ends_game=True,
        ),
        rollout_cfg=RolloutConfig(),
        fullgame_cfg=FullGameConfig(),
        model_color="B",
    )

    assert result.terminated_by_illegal is True
    assert result.final_outcome == -1.0
    assert result.total_reward < 0.0


def _suicide_board() -> BoardState:
    board = BoardState.empty(5)
    board.apply_move("B", "A1", strict=True)
    board.apply_move("W", "C2", strict=True)
    board.apply_move("B", "A2", strict=True)
    board.apply_move("W", "C4", strict=True)
    board.apply_move("B", "A3", strict=True)
    board.apply_move("W", "B3", strict=True)
    board.apply_move("B", "A4", strict=True)
    board.apply_move("W", "D3", strict=True)
    # C3 is basic-legal (empty) but suicide for Black.
    assert board.is_basic_legal(gtp_to_coords("C3", board.size)) is True
    assert board.is_strict_legal("B", gtp_to_coords("C3", board.size)) is False
    return board


def test_simulate_full_game_rejects_suicide_at_basic_legality():
    # KataGo plays under real rules, so applying a basic-legal suicide would
    # corrupt the game. It must be penalized and never reach the board.
    board = _suicide_board()
    result = simulate_full_game(
        board=board,
        initial_completion="C3",
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        katago=_StubKataGo(score_lead=0.0, move="PASS"),
        reward_cfg=RewardConfig(strict_legality=False, illegal_move_ends_game=True),
        rollout_cfg=RolloutConfig(),
        fullgame_cfg=FullGameConfig(max_model_turns=1, max_plies=10),
        model_color="B",
    )

    assert result.terminated_by_illegal is True
    assert result.plies_played == 0
    assert result.legality_reward_total == RewardConfig().illegal_penalty
    assert board.current_board() == _suicide_board().current_board()


def test_simulate_full_game_illegal_move_without_ending_does_not_count_as_ply(monkeypatch):
    analyzed: list[BoardState] = []

    class _RecordingKataGo(_StubKataGo):
        def analyze_position(self, board: BoardState) -> KataGoResult:
            analyzed.append(board.clone())
            return super().analyze_position(board)

    monkeypatch.setattr("go_rl.fullgame_rollout.generate_model_move", lambda **_: "PASS")
    result = simulate_full_game(
        board=_suicide_board(),
        initial_completion="C3",
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        katago=_RecordingKataGo(score_lead=0.0, move="PASS"),
        reward_cfg=RewardConfig(strict_legality=True, illegal_move_ends_game=False),
        rollout_cfg=RolloutConfig(),
        fullgame_cfg=FullGameConfig(max_model_turns=2, max_plies=10),
        model_color="B",
    )

    # The illegal C3 is retried as a PASS from the same position: two model
    # turns, but only the PASS is a ply.
    assert result.terminated_by_illegal is False
    assert result.model_turns == 2
    assert result.plies_played == 1
    # The rejected move never reaches KataGo.
    c3 = gtp_to_coords("C3", 5)
    assert analyzed and all(("B", c3) not in item.moves for item in analyzed)


def test_simulate_full_game_analysis_cache_reuses_final_position():
    class _CountingKataGo(_StubKataGo):
        def __init__(self):
            super().__init__(score_lead=0.0, move="PASS")
            self.calls = 0

        def analyze_position(self, _board: BoardState) -> KataGoResult:
            self.calls += 1
            return super().analyze_position(_board)

    board = BoardState.empty(19)
    katago = _CountingKataGo()
    cache: OrderedDict[tuple, KataGoResult] = OrderedDict()
    result = simulate_full_game(
        board=board,
        initial_completion="D4",
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        katago=katago,
        reward_cfg=RewardConfig(strict_legality=True),
        rollout_cfg=RolloutConfig(),
        fullgame_cfg=FullGameConfig(max_plies=1, max_model_turns=10),
        model_color="B",
        analysis_cache=cache,
        max_cache_size=64,
    )

    assert result.truncated is True
    assert katago.calls == 2


def test_simulate_full_game_analysis_cache_ordered_dict_handles_eviction():
    class _CountingKataGo(_StubKataGo):
        def __init__(self):
            super().__init__(score_lead=0.0, move="PASS")
            self.calls = 0

        def analyze_position(self, _board: BoardState) -> KataGoResult:
            self.calls += 1
            return super().analyze_position(_board)

    board = BoardState.empty(19)
    katago = _CountingKataGo()
    cache: OrderedDict[tuple, KataGoResult] = OrderedDict()
    result = simulate_full_game(
        board=board,
        initial_completion="D4",
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        katago=katago,
        reward_cfg=RewardConfig(strict_legality=True),
        rollout_cfg=RolloutConfig(),
        fullgame_cfg=FullGameConfig(max_plies=1, max_model_turns=10),
        model_color="B",
        analysis_cache=cache,
        max_cache_size=1,
    )

    assert result.truncated is True
    assert katago.calls == 2
    assert len(cache) == 1


@pytest.mark.parametrize("fail", [False, True])
def test_generation_uses_cache_and_restores_training_mode(monkeypatch, fail):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from go_rl.fullgame_rollout import generate_model_move

    class Model(torch.nn.Module):
        device = "cpu"

        def generate(self, **kwargs):
            assert not self.training
            assert kwargs["use_cache"] is True
            assert kwargs["top_k"] == 0
            # Thinking controls belong to the chat template; generate rejects them.
            assert "enable_thinking" not in kwargs
            assert not torch.is_grad_enabled()
            if fail:
                raise RuntimeError("generation failed")
            return torch.tensor([[1, 2]])

    model = Model()
    tokenizer = SimpleNamespace(eos_token_id=0, decode=lambda *args, **kwargs: "PASS")
    template_kwargs = []
    monkeypatch.setattr("go_rl.fullgame_rollout.build_generation_inputs",
                        lambda **kwargs: template_kwargs.append(kwargs["chat_template_kwargs"])
                        or {"input_ids": torch.tensor([[1]])})
    thinking = {"enable_thinking": False}
    if fail:
        with pytest.raises(RuntimeError, match="generation failed"):
            generate_model_move(model, tokenizer, BoardState.empty(9), RolloutConfig(), "qwen3", thinking)
    else:
        assert generate_model_move(model, tokenizer, BoardState.empty(9), RolloutConfig(), "qwen3", thinking) == "PASS"
    assert template_kwargs == [thinking]
    assert model.training


def test_generation_inherits_autocast_for_mixed_precision_head(monkeypatch):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from go_rl.fullgame_rollout import generate_model_move

    class Model(torch.nn.Module):
        device = "cpu"

        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(2, 2, bias=False)

        def generate(self, **kwargs):
            # K-bit training keeps the output head in FP32 while transformer
            # activations can be BF16. Nested rollout generation needs the
            # enclosing trainer's autocast context just like the first move.
            hidden_states = torch.ones(1, 2, dtype=torch.bfloat16)
            logits = self.lm_head(hidden_states)
            assert logits.dtype == torch.bfloat16
            assert not self.training
            assert not torch.is_grad_enabled()
            return torch.cat((kwargs["input_ids"], logits.argmax(-1, keepdim=True)), dim=1)

    model = Model()
    tokenizer = SimpleNamespace(eos_token_id=0, decode=lambda *args, **kwargs: "PASS")
    monkeypatch.setattr("go_rl.fullgame_rollout.build_generation_inputs",
                        lambda **kwargs: {"input_ids": torch.tensor([[1]])})
    with pytest.raises(RuntimeError, match="dtype|scalar type"):
        generate_model_move(model, tokenizer, BoardState.empty(9), RolloutConfig(), "plain")

    with torch.autocast("cpu", dtype=torch.bfloat16):
        move = generate_model_move(model, tokenizer, BoardState.empty(9), RolloutConfig(), "plain")

    assert move == "PASS"
    assert model.training
    assert model.lm_head.weight.dtype == torch.float32


@pytest.mark.parametrize("completion,illegal", [("I'll play D4", True), ("d4", True), ("D4", False)])
def test_simulate_full_game_requires_bare_move_answer(completion, illegal):
    result = simulate_full_game(
        board=BoardState.empty(9),
        initial_completion=completion,
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        katago=_StubKataGo(score_lead=0.0, move="PASS"),
        reward_cfg=RewardConfig(strict_legality=True),
        rollout_cfg=RolloutConfig(),
        fullgame_cfg=FullGameConfig(max_plies=1),
        model_color="B",
    )
    assert result.terminated_by_illegal is illegal


def test_endpoint_reward_includes_opponent_reply_and_only_two_analyses():
    class ChangingKataGo(_StubKataGo):
        def __init__(self):
            super().__init__(move="Q16")
            self.analysis_plies = []

        def analyze_position(self, board):
            self.analysis_plies.append(len(board.moves))
            # The opponent reply changes the endpoint substantially.
            lead = {0: 1.0, 1: -4.0, 2: 3.0}[len(board.moves)]
            return KataGoResult(winrate=0.5, raw={}, score_lead=lead)

    katago = ChangingKataGo()
    result = simulate_full_game(
        board=BoardState.empty(19), initial_completion="D4",
        model=_DummyModel(), tokenizer=_DummyTokenizer(), katago=katago,
        reward_cfg=RewardConfig(scorelead_scale=0.5, legal_bonus=0.0),
        rollout_cfg=RolloutConfig(), fullgame_cfg=FullGameConfig(max_plies=2),
        model_color="B",
    )
    assert result.scorelead_delta_sum == 2.0
    assert result.scorelead_reward == 1.0
    assert result.total_reward == 1.0
    assert katago.analysis_plies == [0, 2]
    assert result.terminal_reward == 0.0


def test_continuation_length_does_not_multiply_legal_bonus(monkeypatch):
    monkeypatch.setattr("go_rl.fullgame_rollout.generate_model_move", lambda **_: "PASS")
    result = simulate_full_game(
        board=BoardState.empty(19), initial_completion="D4",
        model=_DummyModel(), tokenizer=_DummyTokenizer(),
        katago=_StubKataGo(move="PASS"),
        reward_cfg=RewardConfig(legal_bonus=0.2), rollout_cfg=RolloutConfig(),
        fullgame_cfg=FullGameConfig(max_plies=20), model_color="B",
    )
    assert result.model_turns == 2
    assert result.legality_reward_total == 0.2
    assert result.final_outcome is None


def test_pass_termination_includes_pass_before_seed():
    board = BoardState.empty(19)
    board.apply_move("B", "PASS")
    result = simulate_full_game(
        board=board, initial_completion="PASS", model=_DummyModel(),
        tokenizer=_DummyTokenizer(), katago=_StubKataGo(move="Q16"),
        reward_cfg=RewardConfig(), rollout_cfg=RolloutConfig(),
        fullgame_cfg=FullGameConfig(), model_color="W",
    )
    assert result.pass_terminated
    assert result.plies_played == 1


def test_missing_engine_score_is_not_silently_zero():
    class MissingScore(_StubKataGo):
        def analyze_position(self, board):
            return KataGoResult(winrate=0.5, raw={})

    with pytest.raises(RuntimeError, match="scoreLead"):
        simulate_full_game(
            board=BoardState.empty(19), initial_completion="D4",
            model=_DummyModel(), tokenizer=_DummyTokenizer(), katago=MissingScore(),
            reward_cfg=RewardConfig(), rollout_cfg=RolloutConfig(),
            fullgame_cfg=FullGameConfig(max_plies=1), model_color="B",
        )
