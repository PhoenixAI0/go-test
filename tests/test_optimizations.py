#!/usr/bin/env python3
"""Quick integration checks for Go-parser dataset generation."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from go_rl.data_prep import build_sft_dataset


def create_test_sgf(path: Path, board_size: int = 9, num_moves: int = 10) -> None:
    sgf_content = f"(;GM[1]FF[4]SZ[{board_size}]"
    for i in range(num_moves):
        sgf_content += ";B[aa]" if i % 2 == 0 else ";W[bb]"
    sgf_content += ")"
    path.write_text(sgf_content, encoding="utf-8")


@pytest.mark.skipif(shutil.which("go") is None, reason="go toolchain not installed")
def test_go_parser_worker_consistency():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for i in range(10):
            create_test_sgf(tmpdir / f"test_{i:03d}.sgf", board_size=9, num_moves=5)

        out_one = tmpdir / "out_one.jsonl"
        out_many = tmpdir / "out_many.jsonl"

        count_one = build_sft_dataset(str(tmpdir), str(out_one), max_positions=100, num_workers=1)
        count_many = build_sft_dataset(str(tmpdir), str(out_many), max_positions=100, num_workers=4)

        assert count_one == count_many
        lines_one = out_one.read_text(encoding="utf-8").splitlines()
        lines_many = out_many.read_text(encoding="utf-8").splitlines()
        assert len(lines_one) == count_one
        assert len(lines_many) == count_many
        assert [json.loads(line) for line in lines_one] == [json.loads(line) for line in lines_many]


@pytest.mark.skipif(shutil.which("go") is None, reason="go toolchain not installed")
def test_go_parser_output_schema():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for i in range(5):
            create_test_sgf(tmpdir / f"test_{i:03d}.sgf", board_size=9, num_moves=3)

        output_file = tmpdir / "output.jsonl"
        count = build_sft_dataset(str(tmpdir), str(output_file), max_positions=100, num_workers=2)
        assert output_file.exists()
        assert count > 0

        records = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
        assert len(records) == count
        assert all("prompt" in record for record in records)
        assert all("response" in record for record in records)
        assert all("board_size" in record for record in records)
