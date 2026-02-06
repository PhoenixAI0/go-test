from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def in_virtualenv() -> bool:
    if os.environ.get("VIRTUAL_ENV"):
        return True
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    return sys.prefix != base_prefix


def ensure_virtualenv() -> None:
    if in_virtualenv():
        return
    print(
        "Error: This script must run inside a virtual environment.\n"
        "Run: uv venv .venv && source .venv/bin/activate\n"
        "Or:  ./scripts/setup_environment.sh",
        file=sys.stderr,
    )
    sys.exit(1)


def ensure_uv() -> None:
    if shutil.which("uv"):
        return
    print(
        "Error: uv is required to manage dependencies.\n"
        "Install uv: https://astral.sh/uv",
        file=sys.stderr,
    )
    sys.exit(1)


def uv_pip_install(packages: Iterable[str]) -> None:
    ensure_virtualenv()
    ensure_uv()
    cmd = ["uv", "pip", "install", *packages]
    subprocess.run(cmd, check=True)


def _find_go_binary() -> str | None:
    go_bin = shutil.which("go")
    if go_bin:
        return go_bin
    local_go = Path.home() / ".local" / "go" / "bin" / "go"
    if local_go.exists():
        return str(local_go)
    return None


def ensure_go_parser_binary() -> Path | None:
    repo_root = Path(__file__).resolve().parent.parent
    parser_dir = repo_root / "go_parser"
    binary_path = parser_dir / "sgf_parser"

    if not parser_dir.exists():
        print(
            "Warning: go_parser directory not found; falling back to Python SGF parsing.",
            file=sys.stderr,
        )
        return None

    go_bin = _find_go_binary()
    if not go_bin and binary_path.exists() and os.access(binary_path, os.X_OK):
        print(
            "Warning: Go not found; using existing parser binary.",
            file=sys.stderr,
        )
        return binary_path

    if not go_bin:
        install_script = repo_root / "scripts" / "install_go.sh"
        if install_script.exists() and os.access(install_script, os.X_OK):
            try:
                subprocess.run([str(install_script)], check=True)
            except subprocess.CalledProcessError:
                print(
                    "Warning: Go install failed; falling back to Python SGF parsing.",
                    file=sys.stderr,
                )
                return None
            go_bin = _find_go_binary()
        if not go_bin:
            if binary_path.exists() and os.access(binary_path, os.X_OK):
                print(
                    "Warning: Go not found; using existing parser binary.",
                    file=sys.stderr,
                )
                return binary_path
            print(
                "Warning: Go not found; falling back to Python SGF parsing.",
                file=sys.stderr,
            )
            return None

    try:
        subprocess.run(
            [go_bin, "build", "-o", str(binary_path), "."],
            cwd=str(parser_dir),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"Warning: Failed to build Go parser ({exc}); falling back to Python.",
            file=sys.stderr,
        )
        return None

    if binary_path.exists() and os.access(binary_path, os.X_OK):
        return binary_path

    print(
        "Warning: Go parser build did not produce an executable; falling back to Python.",
        file=sys.stderr,
    )
    return None


def run_go_parser(
    sgf_dir: str,
    output_path: str,
    max_positions: int,
    num_workers: int,
) -> bool:
    parser_bin = ensure_go_parser_binary()
    if not parser_bin:
        return False

    cmd = [
        str(parser_bin),
        "-sgf-dir",
        sgf_dir,
        "-output",
        output_path,
        "-max-positions",
        str(max_positions),
        "-workers",
        str(num_workers),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            f"Warning: Go parser failed ({exc}); falling back to Python.",
            file=sys.stderr,
        )
        return False
    return True
