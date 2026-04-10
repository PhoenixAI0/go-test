import json

from go_rl.data_prep import build_sft_dataset


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
