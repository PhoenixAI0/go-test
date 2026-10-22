import json
import pytest

from go_rl.data_prep import (build_sft_dataset, game_split_indices, split_dataset_by_game,
                             board_from_example, transform_example, prepare_dataset_files,
                             state_record_from_board)
from go_rl.board import BoardState


def test_build_sft_dataset_invokes_go_parser(tmp_path, monkeypatch):
    output_path = tmp_path / "data" / "sft.jsonl"
    written = [
        {
            "prompt": "\n".join(
                [
                    "board_size: 5",
                    "to_play: B",
                    "board:",
                    ".....",
                    ".....",
                    ".....",
                    ".....",
                    ".....",
                    "move:",
                ]
            ),
            "response": "A1",
            "board_size": 5,
        },
        {
            "prompt": "\n".join(
                [
                    "board_size: 5",
                    "to_play: W",
                    "board:",
                    ".....",
                    ".....",
                    ".....",
                    ".....",
                    "X....",
                    "move:",
                ]
            ),
            "response": "PASS",
            "board_size": 5,
        },
    ]

    def fake_run_go_parser(sgf_dir, output_path, max_positions, num_workers):
        assert sgf_dir == "/unused"
        assert max_positions == 2
        assert num_workers == 0
        output_path = tmp_path / "data" / "sft.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            for record in written:
                handle.write(json.dumps(record) + "\n")
        return len(written)

    monkeypatch.setattr("go_rl.data_prep.run_go_parser", fake_run_go_parser)

    count = build_sft_dataset("/unused", str(output_path), max_positions=2)
    assert count == 2

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]
    assert records == written


def _example(game_id="game", response="D2"):
    initial = [[0] * 5 for _ in range(5)]
    initial[0][0] = 1
    board = BoardState.from_matrix(5, initial, "W")
    board.apply_move("W", "C3", strict=True)
    return {"prompt": board.to_prompt(), "response": response, "board_size": 5,
            "game_id": game_id, "initial_stones": [["B", "A5"]],
            "initial_player": "W", "to_play": "B", "moves": [["W", "C3"]],
            "komi": 0.5, "rules": "Chinese", "move_number": 1}


def test_game_split_keeps_positions_and_symmetries_together():
    game_ids = [f"game-{i // 5}" for i in range(100)]
    train, validation = game_split_indices(game_ids, 0.2, 7)
    train_games = {game_ids[i] for i in train}
    validation_games = {game_ids[i] for i in validation}
    assert not train_games & validation_games
    assert len(validation_games) == 4
    reversed_ids = list(reversed(game_ids))
    _, reversed_validation = game_split_indices(reversed_ids, 0.2, 7)
    assert {reversed_ids[i] for i in reversed_validation} == validation_games
    assert game_split_indices(["one", "one"], 0.2) == ([0, 1], [])
    with pytest.raises(ValueError, match="game_id"):
        game_split_indices([""])


def test_board_from_example_replays_history_and_rejects_mismatch():
    example = _example()
    board = board_from_example(example)
    assert len(board.moves) == 1
    assert board.next_player() == "B"
    assert board.current_board()[0][0] == 1
    assert board.current_board()[2][2] == 2
    example["moves"] = [["W", "B3"]]
    with pytest.raises(ValueError, match="reconstruct"):
        board_from_example(example)


def test_state_record_round_trips_setup_and_history():
    board = board_from_example(_example())
    board.apply_move("B", "PASS", strict=True)
    record = state_record_from_board(board)
    reconstructed = board_from_example(record)
    assert reconstructed.current_board() == board.current_board()
    assert reconstructed.moves == board.moves
    assert record["initial_player"] == "W"
    assert record["to_play"] == "W"


def test_all_symmetries_preserve_setup_history_action_alignment():
    source = _example(response="A4")
    boards = set()
    for symmetry in range(8):
        transformed = transform_example(source, symmetry)
        assert transformed["game_id"] == source["game_id"]
        assert transformed["komi"] == 0.5
        board = board_from_example(transformed)
        board.apply_move("B", transformed["response"], strict=True)
        assert sum(value == 1 for row in board.current_board() for value in row) == 2
        boards.add((transformed["prompt"], transformed["response"]))
    assert len(boards) == 8
    assert transform_example(_example(response="PASS"), 5)["response"] == "PASS"


def test_preparation_holds_validation_out_of_rl_and_augments_only_training(tmp_path):
    source = tmp_path / "raw.jsonl"
    records = [_example(f"game-{index}") for index in range(10)]
    source.write_text("".join(json.dumps(record) + "\n" for record in records))
    report = prepare_dataset_files(source, tmp_path, "fixture", eval_ratio=0.2, symmetries=8)
    assert report["games"] == {"train": 8, "validation": 2}
    assert report["counts"] == {"source_train": 8, "source_validation": 2,
                                "written_train": 64, "written_validation": 2}
    sft = [json.loads(line) for line in (tmp_path / "sft_fixture.jsonl").read_text().splitlines()]
    rl = [json.loads(line) for line in (tmp_path / "rl_prompts_fixture.jsonl").read_text().splitlines()]
    validation_ids = {record["game_id"] for record in sft if record["split"] == "validation"}
    assert not validation_ids & {record["game_id"] for record in rl}
    assert all(record["moves"] and "komi" in record and "response" not in record for record in rl)
    assert len(report["source_sha256"]) == 64


def test_hf_split_honors_pinned_validation_and_rejects_leakage():
    from datasets import Dataset

    dataset = Dataset.from_list([{**_example("a"), "split": "train"},
                                 {**_example("b"), "split": "validation"}])
    train, validation = split_dataset_by_game(dataset, 0.8)
    assert list(train["game_id"]) == ["a"]
    assert list(validation["game_id"]) == ["b"]
    leaked = dataset.add_item({**_example("a"), "split": "validation"})
    with pytest.raises(ValueError, match="multiple"):
        split_dataset_by_game(leaked, 0.2)
    with pytest.raises(ValueError, match="requires game_id"):
        split_dataset_by_game(dataset.remove_columns(["game_id", "split"]), 0.2)
