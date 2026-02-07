import json
import shutil
import subprocess
from pathlib import Path

import pytest

from go_rl.data_prep import build_sft_dataset


@pytest.mark.skipif(shutil.which("go") is None, reason="go toolchain not installed")
def test_go_parser_matches_python(tmp_path):
    pytest.importorskip("sgfmill", reason="sgfmill not installed")

    sgf_dir = tmp_path / "sgf"
    sgf_dir.mkdir()

    (sgf_dir / "game1.sgf").write_text(
        """(;GM[1]FF[4]SZ[5]AB[aa:ab]AW[cc]
;B[dd];W[];B[ee])
""",
        encoding="utf-8",
    )

    (sgf_dir / "game2.sgf").write_text(
        """(;GM[1]FF[4]SZ[5]
;B[aa](;W[bb];B[cc])(;W[dd];B[ee]))
""",
        encoding="utf-8",
    )

    (sgf_dir / "game3.sgf").write_text(
        """(;GM[1]FF[3]SZ[19]
;B[dd];W[tt];B[pp])
""",
        encoding="utf-8",
    )

    python_out = tmp_path / "python.jsonl"
    count = build_sft_dataset(
        str(sgf_dir),
        str(python_out),
        max_positions=100,
        num_workers=1,
        use_parallel=False,
        batch_size=10,
    )
    assert count > 0

    go_parser_dir = Path(__file__).resolve().parents[1] / "go_parser"
    go_bin = go_parser_dir / "sgf_parser"
    subprocess.run(
        ["go", "build", "-o", str(go_bin), "."],
        cwd=str(go_parser_dir),
        check=True,
    )

    go_out = tmp_path / "go.jsonl"
    subprocess.run(
        [
            str(go_bin),
            "-sgf-dir",
            str(sgf_dir),
            "-output",
            str(go_out),
            "-max-positions",
            "100",
            "-workers",
            "2",
        ],
        check=True,
    )

    def load_jsonl(path: Path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    python_records = load_jsonl(python_out)
    go_records = load_jsonl(go_out)

    assert python_records == go_records
