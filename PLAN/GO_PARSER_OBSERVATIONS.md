# Go SGF Parser Observations (go_parser)

This note summarizes all technical observations about the current Go implementation in `go_parser/`.

## Files and Entry Points

- `go_parser/main.go` is the sole Go source and the CLI entry point.
- `go_parser/sgf_parser` is a prebuilt binary (ELF 64-bit, statically linked, with debug info). It is produced by `go build -o sgf_parser main.go`.
- `go_parser/README.md` documents expected performance claims and usage.

## High-Level Pipeline (Go)

- CLI flags:
  - `-sgf-dir` (required): directory to scan for `*.sgf` (case-insensitive extension via `strings.ToLower`).
  - `-output` (required): JSONL output file path.
  - `-max-positions` (default 200000): maximum total positions across all games.
  - `-workers` (default `runtime.NumCPU()`): number of goroutines.
- The process:
  1. Walks `sgf-dir` recursively with `filepath.Walk` and collects `.sgf` files.
  2. Creates output file and a large buffered writer (1 MB buffer).
  3. Spawns `workers` goroutines; each worker reads file paths from a `jobs` channel and emits slices of `Sample` on a `results` channel.
  4. Consumes `results`, marshals each `Sample` to JSON, writes as JSONL until `max-positions` is reached.

## SGF Parsing Logic (Simplified)

- Board size parsing:
  - Uses regex `SZ\[(\d+)\]` to parse size from the SGF content.
  - Defaults to `19` if no size is found.
- Move parsing:
  - Uses two regexes:
    - `;B\[([a-s]{2}|)\]`
    - `;W\[([a-s]{2}|)\]`
  - Extracts all black moves and all white moves independently.
  - Interleaves these two lists in order, assuming alternating play.
  - The code does **not** parse the SGF node tree; it simply finds all matching move tags in the raw file string.
  - `coord == ""` is treated as a pass.
  - Coordinate decoding: `row = coord[1] - 'a'`, `col = coord[0] - 'a'`.
- The parser ignores:
  - Variations (`(;...)(;...)`) and node structure.
  - Setup stones (`AB`, `AW`) and other root properties.
  - Handicap, komi, time, ranks, and rules.
  - Any move tags outside `;B[]`/`;W[]` or non-`a-s` coordinate ranges.
  - Edge cases such as invalid coordinates, mixed board sizes, or multiple games in a collection.

## Board State and Prompting (Go)

- There is **no board state tracking** or legality checking in the Go parser.
- Prompt creation is a placeholder:
  - `generateBoardPrompt(boardSize, moveNum)` returns the string:
    - `"Board size: %dx%d, Move: %d"`
  - This does **not** match the Python prompt format (`BoardState.to_prompt()`) which lists all previous moves and ends with `"move:"`.
- Response generation:
  - Uses `coordsToGTP` to convert `(row, col)` to GTP.
  - Implements GTP coordinate mapping with `A-T` skipping `I`, and rows numbered from bottom.

## JSON Schema and Sample Fields (Go)

- `Sample` is serialized as:
  - `prompt` (string)
  - `response` (string)
  - `player` (string)
  - `board_size` (int)
- The Go pipeline emits `player` and `board_size` for each sample, but the Python SFT builder writes only `prompt`, `response`, `board_size` (and uses `target` internally from sgf utils).

## Concurrency and Distribution Details

- Work distribution:
  - `maxSamplesPerFile := (maxPositions / len(sgfFiles)) + 1`.
  - This limits per-file samples and is used by all workers; it does not account for uneven game lengths.
- Channels:
  - `jobs` and `results` are buffered with capacity 100.
  - Workers push slices of `Sample` (`[]Sample`) onto `results`.
- Termination:
  - A goroutine waits for `wg.Wait()` then closes `results`.
  - The main goroutine breaks out of the results loop once `max-positions` is reached. If it breaks early, remaining worker output is ignored, but workers still run to completion.

## Error Handling

- File read errors or parse errors are logged and skipped.
- JSON marshal errors are logged and skipped per sample.
- No retry or error aggregation; errors are reported via `log.Printf` only.

## Performance-Oriented Choices

- Uses `os.ReadFile` and regex on full file contents (no streaming parsing).
- Single-pass JSONL writing via a large buffered writer.
- Concurrency via goroutines rather than processes.

## Parity Gaps vs Python Pipeline

- SGF parsing:
  - Go parser does not match `sgfmill` semantics used in `go_rl/sgf_utils.py`.
  - No support for setup stones, collections, or root metadata.
  - Regex extraction can incorrectly interleave if SGF contains comments with `;B[]`/`;W[]` or variations.
- Board legality:
  - Python uses `BoardState` to apply moves, handle captures, and stop on illegal moves.
  - Go parser does not validate or apply moves at all.
- Prompt format:
  - Go prompts are placeholders and are not the same as Python `BoardState.to_prompt()`.
- Output schema:
  - Go emits `player` in output; Python SFT output does not include it by default.

## Integration Notes

- `scripts/fast_build_dataset.py` will use the Go parser via `scripts/venv_utils.py` if a working `go_parser/sgf_parser` binary is present; otherwise it falls back to Python.
- `scripts/prepare_data.sh` builds the Go parser binary if missing (`go build -o sgf_parser main.go`).
- The Go parser is explicitly described in `go_parser/README.md` as a **simplified** parser and not fully SGF-complete.

## Opportunities / Implications

- The current Go implementation prioritizes speed over correctness or parity with Python data generation.
- To make Go output equivalent to Python SFT datasets, the Go parser would need:
  - A real SGF parser (either a Go SGF library or custom parser aware of node structure).
  - Board state tracking with capture logic (matching `go_rl/board.py`).
  - Prompt formatting that matches `BoardState.to_prompt()`.
  - Handling of setup stones and edge cases mirrored from `sgf_utils.py`.
