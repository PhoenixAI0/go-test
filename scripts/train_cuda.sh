#!/usr/bin/env bash
# Compatibility alias for run_full_pipeline.sh.
exec "$(dirname "${BASH_SOURCE[0]}")/run_full_pipeline.sh" "$@"
