from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from go_rl.config import ReasoningConfig, RewardConfig
from go_rl.katago import KataGoResult


def _load_script_module(filename: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script_path = scripts_dir / filename
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _BatchOnlyKataGo:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def analyze_positions(self, boards):
        board_list = list(boards)
        self.batch_sizes.append(len(board_list))
        return [
            KataGoResult(
                winrate=0.5 + (0.01 * len(board.stones())),
                raw={"rootInfo": {"winrate": 0.5}},
            )
            for board in board_list
        ]

    def analyze_position(self, _board):
        raise AssertionError("Expected batched analyze_positions path")


PROMPT = "\n".join(
    [
        "board_size: 9",
        "to_play: B",
        "board:",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        "move:",
    ]
)


def test_run_grpo_reward_deduplicates_base_and_candidate(monkeypatch):
    module = _load_script_module("run_grpo.py", "run_grpo_script_reward_cache")
    monkeypatch.setenv("LOCAL_RANK", "0")

    katago = _BatchOnlyKataGo()
    reward_fn = module.build_reward_fn(
        katago=katago,
        reward_cfg=RewardConfig(),
        reasoning_cfg=ReasoningConfig(profile="qwen3"),
        model_name="qwen-3-8b",
        tokenizer=None,
    )

    rewards = reward_fn(
        prompts=[PROMPT, PROMPT],
        completions=["D4", "D4"],
    )

    assert len(rewards) == 2
    assert katago.batch_sizes == [1, 1]


def test_run_grpo_reward_skips_candidate_batch_for_illegal_moves(monkeypatch):
    module = _load_script_module("run_grpo.py", "run_grpo_script_reward_illegal")
    monkeypatch.setenv("LOCAL_RANK", "0")

    katago = _BatchOnlyKataGo()
    reward_fn = module.build_reward_fn(
        katago=katago,
        reward_cfg=RewardConfig(),
        reasoning_cfg=ReasoningConfig(profile="qwen3"),
        model_name="qwen-3-8b",
        tokenizer=None,
    )

    rewards = reward_fn(
        prompts=[PROMPT, PROMPT],
        completions=["Z99", "Z99"],
    )

    assert len(rewards) == 2
    assert katago.batch_sizes == [1]


def test_run_grpo_reward_reasoning_prefixed_same_as_plain(monkeypatch):
    module = _load_script_module("run_grpo.py", "run_grpo_script_reward_reasoning")
    monkeypatch.setenv("LOCAL_RANK", "0")

    katago = _BatchOnlyKataGo()
    reward_fn = module.build_reward_fn(
        katago=katago,
        reward_cfg=RewardConfig(),
        reasoning_cfg=ReasoningConfig(profile="qwen3"),
        model_name="qwen-3-8b",
        tokenizer=None,
    )

    rewards = reward_fn(
        prompts=[PROMPT, PROMPT],
        completions=["D4", "<think>analysis</think> D4"],
    )
    assert rewards[0] == rewards[1]


def test_reward_uses_black_perspective_for_both_sides(monkeypatch):
    import pytest
    module = _load_script_module("run_grpo.py", "reward_perspective")
    monkeypatch.setenv("LOCAL_RANK", "1")
    reward = module.build_reward_fn(_BatchOnlyKataGo(), RewardConfig())
    scores = reward(prompts=[PROMPT, PROMPT.replace("to_play: B", "to_play: W")],
                    completions=["D4", "D4"])
    assert scores == pytest.approx([0.21, 0.19])


def test_reward_scores_original_board_when_prompt_is_chat_rendered():
    module = _load_script_module("run_grpo.py", "reward_chat_prompt")
    reward = module.build_reward_fn(_BatchOnlyKataGo(), RewardConfig())
    assert reward(prompts=["<chat>...</chat>"], completions=["D4"], board_prompt=[PROMPT])[0] > 0


def test_reward_penalizes_suicide_without_crashing():
    from go_rl.board import BoardState
    module = _load_script_module("run_grpo.py", "reward_suicide")
    board = BoardState.from_matrix(3, [[0, 2, 0], [2, 0, 2], [0, 2, 0]], "B")
    reward = module.build_reward_fn(_BatchOnlyKataGo(), RewardConfig())
    assert reward(prompts=[board.to_prompt()], completions=["B2"]) == [-1.0]


def test_reward_penalizes_any_answer_format_but_a_bare_move():
    import pytest
    module = _load_script_module("run_grpo.py", "reward_strict_format")
    reward = module.build_reward_fn(_BatchOnlyKataGo(), RewardConfig(), model_name="plain-model")
    sloppy = ["d4", "D4.", "Move: D4", "I'll play D4", "D4 D5", "pass", "PASS please", "", "D 4"]
    assert reward(prompts=[PROMPT] * len(sloppy), completions=sloppy) == [-1.0] * len(sloppy)
    # Surrounding whitespace (e.g. after a reasoning block or before EOS) is fine.
    assert reward(prompts=[PROMPT] * 2, completions=["\nD4\n", " PASS"]) == pytest.approx([0.21, 0.2])


def test_reward_keeps_rules_and_komi_separate_in_engine_batches():
    class RecordingKataGo(_BatchOnlyKataGo):
        def __init__(self):
            super().__init__()
            self.options = []

        def analyze_positions(self, boards, **options):
            self.options.append(options)
            return super().analyze_positions(boards)

    module = _load_script_module("run_grpo.py", "reward_rule_context")
    katago = RecordingKataGo()
    reward = module.build_reward_fn(katago, RewardConfig())
    reward([PROMPT, PROMPT], ["D4", "D4"], rules=["chinese", "chinese"], komi=[6.5, 7.5])
    assert katago.batch_sizes == [1, 1, 1, 1]
    assert [item["komi"] for item in katago.options] == [6.5, 7.5, 6.5, 7.5]
    assert all(item["rules"] == "chinese" for item in katago.options)


def test_reward_reconstructs_ko_history_from_dataset_columns():
    from go_rl.board import BoardState, gtp_to_coords
    from go_rl.data_prep import state_record_from_board

    matrix = [[0] * 5 for _ in range(5)]
    for color, move in [(1, "B3"), (1, "D3"), (1, "C4"),
                        (2, "C3"), (2, "B2"), (2, "C1"), (2, "D2")]:
        row, column = gtp_to_coords(move, 5)
        matrix[row][column] = color
    board = BoardState.from_matrix(5, matrix, "B")
    board.apply_move("B", "C2", strict=True)
    record = state_record_from_board(board)

    class HistoryKataGo(_BatchOnlyKataGo):
        def analyze_positions(self, boards, **options):
            boards = list(boards)
            assert all(len(item.moves) == 1 for item in boards)
            return super().analyze_positions(boards)

    module = _load_script_module("run_grpo.py", "reward_ko_metadata")
    katago = HistoryKataGo()
    reward = module.build_reward_fn(katago, RewardConfig(strict_legality=True))
    columns = {key: [value] for key, value in record.items() if key != "prompt"}
    result = reward([record["prompt"]], ["C3"], **columns,
                    rules=["koPOSITIONALscoreAREAtaxNONEsui0"], komi=[7.5])
    assert result == [-1.0]
    assert katago.batch_sizes == [1]


def test_unsupported_rules_and_corrupt_history_are_data_errors():
    import pytest
    module = _load_script_module("run_grpo.py", "reward_bad_metadata")
    reward = module.build_reward_fn(_BatchOnlyKataGo(), RewardConfig())
    with pytest.raises(ValueError, match="suicide"):
        reward([PROMPT], ["D4"], rules=["koPOSITIONALscoreAREAtaxNONEsui1"])
    with pytest.raises(ValueError, match="both moves and initial_stones"):
        reward([PROMPT], ["D4"], moves=[[['B', 'D4']]])


def test_area_reward_rejects_past_terminal_source_before_engine_queries():
    import pytest
    from go_rl.board import BoardState
    from go_rl.data_prep import state_record_from_board

    board = BoardState.empty(9)
    for player, move in [("B", "PASS"), ("W", "PASS"), ("B", "D4")]:
        board.apply_move(player, move)
    record = state_record_from_board(board)
    columns = {key: [value] for key, value in record.items() if key != "prompt"}
    module = _load_script_module("run_grpo.py", "reward_terminal_history")
    engine = _BatchOnlyKataGo()
    reward = module.build_reward_fn(engine, RewardConfig(), default_rules="chinese")
    with pytest.raises(ValueError, match="precede two-pass"):
        reward([record["prompt"]], ["D5"], **columns)
    assert engine.batch_sizes == []
