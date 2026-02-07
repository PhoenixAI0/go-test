import json
import shutil
from pathlib import Path

import pytest

from go_rl.data_prep import build_sft_dataset


@pytest.mark.skipif(shutil.which("go") is None, reason="go toolchain not installed")
def test_build_sft_dataset_uses_go_parser(tmp_path):
    sgf_dir = tmp_path / "sgf"
    sgf_dir.mkdir()

    (sgf_dir / "game1.sgf").write_text(
        """(;GM[1]FF[4]SZ[5]AB[aa:ab]AW[cc]
;B[dd];W[];B[ee])
""",
        encoding="utf-8",
    )

    output_path = tmp_path / "sft.jsonl"
    count = build_sft_dataset(
        str(sgf_dir),
        str(output_path),
        max_positions=100,
        num_workers=1,
    )

    assert count > 0
    assert output_path.exists()
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == count
    assert all("prompt" in record for record in records)
    assert all("response" in record for record in records)
    assert all("board_size" in record for record in records)
