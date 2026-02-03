# Issues Report (Static Review + Pytest)

Date: 2026-02-03

## Test Execution

Command:
```
python3 -m pytest -q
```

Result:
- 16 failed, 4 errors, 4 skipped, 1 warning.

Key error signatures:
- `TypeError: BoardState.__init__() missing 1 required positional argument: 'moves'`
- `module 'sgfmill.sgf' has no attribute 'Sgf_collection'`
- `IndexError: list index out of range` (after SGF parsing skipped all files)
- `json.decoder.JSONDecodeError: Expecting value` (empty JSONL output)

## Problems Found (Pytest + Static Analysis)

### P0
- **`BoardState` requires `moves` but many call sites instantiate with only `size`.**
  - Breaks fixtures and many tests; also affects runtime usage in scripts.
  - Likely fix: default `moves` to empty list (e.g., `field(default_factory=list)`) or add an alternate constructor.
  - Files: `go_rl/board.py`, `tests/conftest.py`, `tests/test_grpo_rewards.py`, `go_rl/self_play.py`, `scripts/play_interactive.py`, `docs/EVALUATION_GUIDE.md`.

- **SGF parsing uses a non-existent `sgfmill.sgf.Sgf_collection`.**
  - With sgfmill 1.1.1, only `Sgf_game` exists; this causes all SGF loads to fail.
  - Cascades into dataset building tests and optimization tests (zero samples).
  - Files: `go_rl/sgf_utils.py`, `test_optimizations.py`, `tests/test_sgf_utils.py`, `tests/test_data_prep.py`.

### P1
- **Missing `parse_move` and mismatched `build_prompt` signature.**
  - `parse_move` is imported in multiple modules but does not exist.
  - `build_prompt` only accepts `(board)` but call sites pass `(board, player)`.
  - Affects: `go_rl/self_play.py`, `scripts/play_interactive.py`, `docs/EVALUATION_GUIDE.md`.
  - File: `go_rl/prompting.py`.

- **Interactive UI accesses `board.board`, which doesn’t exist.**
  - `BoardState` stores move history, not a board grid.
  - Script fails on first render.
  - File: `scripts/play_interactive.py`.

### P2
- **Self-play SGF coordinate axes are swapped.**
  - `gtp_to_coords` returns `(row, col)` but `_build_sgf` treats them as `(x, y)`.
  - Leads to transposed SGF coordinates.
  - File: `go_rl/self_play.py`.

- **`RewardConfig.legality_level` and `go_rl.rules` are unused.**
  - Configuration advertises `basic/ko/superko`, but reward logic only checks `strict_legality`.
  - Documentation and config suggest features that do not execute.
  - Files: `go_rl/config.py`, `go_rl/rewards.py`, `go_rl/rules.py`, `scripts/run_grpo.py`.

## Detailed Pytest Failures (Grouped)

### BoardState missing default `moves`
Triggered in:
- `tests/conftest.py` fixtures
- `tests/test_grpo_rewards.py`
- `tests/test_katago_integration.py`
- `tests/test_multi_gpu_setup.py`
- `tests/test_data_prep.py`

### SGF parsing API mismatch (`Sgf_collection`)
Triggered in:
- `test_optimizations.py` (both tests)
- `tests/test_sgf_utils.py` (`iter_sgf_samples` tests)
- `tests/test_data_prep.py`

### Downstream failures from empty SGF samples
Triggered in:
- `test_optimizations.py::test_parallel_processing` (IndexError)
- `test_optimizations.py::test_buffered_writing` (JSON decode on empty output)

## Notes
- A `pytest.mark.katago` is used without registration; currently only a warning.
- Environment uses Python 3.13 with deps from `requirements.txt`.
