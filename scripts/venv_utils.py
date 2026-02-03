from __future__ import annotations

import os
import shutil
import subprocess
import sys
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
