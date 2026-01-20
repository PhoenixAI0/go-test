import json

from go_rl.data_prep import build_sft_dataset


def test_build_sft_dataset_writes_jsonl(tmp_path, monkeypatch):
    samples = [
        {
            "prompt": "board_size: 5\nmove:",
            "target": "A1",
            "player": "B",
            "board_size": 5,
        },
        {
            "prompt": "board_size: 5\nB: A1\nmove:",
            "target": "PASS",
            "player": "W",
            "board_size": 5,
        },
    ]

    def fake_load_sgf_dir(_sgf_dir, max_positions):
        assert max_positions == 2
        return samples

    monkeypatch.setattr("go_rl.data_prep.load_sgf_dir", fake_load_sgf_dir)

    output_path = tmp_path / "data" / "sft.jsonl"
    count = build_sft_dataset("/unused", str(output_path), max_positions=2)
    assert count == 2

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]
    assert records == [
        {"prompt": samples[0]["prompt"], "response": "A1", "board_size": 5},
        {"prompt": samples[1]["prompt"], "response": "PASS", "board_size": 5},
    ]
