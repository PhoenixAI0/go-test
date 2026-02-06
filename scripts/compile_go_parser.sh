#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PARSER_DIR="$REPO_ROOT/go_parser"
PARSER_BIN="$PARSER_DIR/sgf_parser"

if [ ! -d "$PARSER_DIR" ]; then
    echo "Warning: go_parser directory not found; cannot compile Go parser." >&2
    exit 1
fi

if ! command -v go &> /dev/null; then
    if [ -x "$SCRIPT_DIR/install_go.sh" ]; then
        "$SCRIPT_DIR/install_go.sh" || true
    fi
    if [ -f "$SCRIPT_DIR/go_env.sh" ]; then
        # shellcheck source=/dev/null
        source "$SCRIPT_DIR/go_env.sh"
    fi
fi

if ! command -v go &> /dev/null; then
    if [ -x "$PARSER_BIN" ]; then
        echo "Go not found; using existing parser binary at $PARSER_BIN" >&2
        exit 0
    fi
    echo "Warning: Go not found and no parser binary exists; cannot compile parser." >&2
    exit 1
fi

echo "Compiling Go SGF parser..."
(cd "$PARSER_DIR" && go build -o sgf_parser .)

if [ ! -x "$PARSER_BIN" ]; then
    echo "Warning: Go parser compilation did not produce an executable binary." >&2
    exit 1
fi

echo "Go SGF parser ready: $PARSER_BIN"
