from __future__ import annotations

import os
import sys


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
