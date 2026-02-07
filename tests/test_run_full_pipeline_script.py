from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_pipeline_no_eval_still_runs_fullgame_stage(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    test_repo = tmp_path / "repo"
    scripts_dir = test_repo / "scripts"
    scripts_dir.mkdir(parents=True)

    shutil.copy2(repo_root / "scripts" / "run_full_pipeline.sh", scripts_dir / "run_full_pipeline.sh")

    _write_executable(
        scripts_dir / "train_grpo.sh",
        """#!/bin/bash
set -euo pipefail
board_size="9x9"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --board-size)
      board_size="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
mkdir -p "outputs/grpo_${board_size}"
mkdir -p logs
echo "train_grpo:${board_size}" >> logs/train_calls.log
""",
    )

    _write_executable(
        scripts_dir / "train_grpo_fullgame.sh",
        """#!/bin/bash
set -euo pipefail
mkdir -p outputs/grpo_fullgame_19x19
echo "called" > fullgame_called.txt
""",
    )

    run_env = dict(os.environ)
    run_env["VIRTUAL_ENV"] = "/tmp/fake_venv_for_test"

    result = subprocess.run(
        [
            "bash",
            "scripts/run_full_pipeline.sh",
            "--skip-setup",
            "--skip-data",
            "--skip-sft",
            "--no-eval",
        ],
        cwd=test_repo,
        env=run_env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "Bypassing full-game promotion gate because --no-eval was set." in result.stdout
    assert (test_repo / "fullgame_called.txt").is_file()
    assert (test_repo / "outputs" / "grpo_fullgame_19x19").is_dir()

