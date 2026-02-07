from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_go_binary() -> str | None:
    go_bin = shutil.which("go")
    if go_bin:
        return go_bin
    local_go = Path.home() / ".local" / "go" / "bin" / "go"
    if local_go.exists():
        return str(local_go)
    return None


def _build_go_parser_binary() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    parser_dir = repo_root / "go_parser"
    binary_path = parser_dir / "sgf_parser"

    if not parser_dir.exists():
        raise RuntimeError(f"Go parser directory not found: {parser_dir}")

    go_bin = _find_go_binary()
    if not go_bin:
        install_script = repo_root / "scripts" / "install_go.sh"
        if install_script.exists() and os.access(install_script, os.X_OK):
            logger.info("Go toolchain not found, attempting install via %s", install_script)
            try:
                subprocess.run([str(install_script)], check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError("Failed to install Go toolchain required for SGF parsing.") from exc
            go_bin = _find_go_binary()

    if not go_bin:
        raise RuntimeError("Go toolchain is required for dataset parsing but was not found.")

    env = os.environ.copy()
    env.setdefault("GOCACHE", "/tmp/go-build")
    try:
        subprocess.run(
            [go_bin, "build", "-o", str(binary_path), "."],
            cwd=str(parser_dir),
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to build Go SGF parser in {parser_dir}") from exc

    if not binary_path.exists() or not os.access(binary_path, os.X_OK):
        raise RuntimeError(f"Go parser binary is missing or not executable: {binary_path}")

    return binary_path


def run_go_parser(
    sgf_dir: str,
    output_path: str,
    max_positions: int,
    num_workers: int = 0,
) -> int:
    parser_bin = _build_go_parser_binary()
    workers = num_workers if num_workers > 0 else max(1, (os.cpu_count() or 1) - 1)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(parser_bin),
        "-sgf-dir",
        sgf_dir,
        "-output",
        str(output),
        "-max-positions",
        str(max_positions),
        "-workers",
        str(workers),
    ]
    logger.info("Running Go SGF parser with %d workers", workers)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Go SGF parser execution failed.") from exc

    with output.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def build_sft_dataset(
    sgf_dir: str,
    output_path: str,
    max_positions: int,
    num_workers: int = 0,
    use_parallel: bool = True,
    batch_size: int = 1000,
) -> int:
    if not use_parallel:
        logger.warning("--no-parallel is deprecated; Go parser parallelism is always managed by --num-workers.")
    if batch_size != 1000:
        logger.warning("batch_size is ignored when using the Go parser path.")
    logger.info("Building dataset from SGFs with Go parser: %s", sgf_dir)
    count = run_go_parser(
        sgf_dir=sgf_dir,
        output_path=output_path,
        max_positions=max_positions,
        num_workers=num_workers,
    )
    logger.info("Finished writing %d samples to %s", count, output_path)
    return count
