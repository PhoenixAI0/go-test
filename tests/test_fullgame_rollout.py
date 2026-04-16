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


def test_simulate_full_game_respects_basic_legality_application():
    board = BoardState.empty(5)
    board.apply_move("B", "A1", strict=True)
    board.apply_move("W", "C2", strict=True)
    board.apply_move("B", "A2", strict=True)
    board.apply_move("W", "C4", strict=True)
    board.apply_move("B", "A3", strict=True)
    board.apply_move("W", "B3", strict=True)
    board.apply_move("B", "A4", strict=True)
    board.apply_move("W", "D3", strict=True)

    assert board.is_basic_legal(gtp_to_coords("C3", board.size)) is True
    assert board.is_strict_legal("B", gtp_to_coords("C3", board.size)) is False

    katago = _StubKataGo(score_lead=0.0, move="PASS")
    result = simulate_full_game(
        board=board,
        initial_completion="C3",
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        katago=katago,
        reward_cfg=RewardConfig(strict_legality=False, illegal_move_ends_game=True),
        rollout_cfg=RolloutConfig(),
        fullgame_cfg=FullGameConfig(max_model_turns=1, max_plies=10),
        model_color="B",
    )

    assert result.terminated_by_illegal is False
    assert result.plies_played == 1


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
