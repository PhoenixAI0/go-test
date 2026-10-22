# SGF dataset parser

The Go parser reconstructs main-line game states and produces metadata-rich JSONL for supervised training and RL starts. Python entry points compile it with the installed Go toolchain.

```bash
cd go_parser
go build -o sgf_parser .
go test ./...
./sgf_parser -sgf-dir ../data/sgf_19x19 -output ../data/raw.jsonl \
  -max-positions 200000 -positions-per-game 64 -seed 42 -workers 8
```

Files are shuffled reproducibly. A per-game position limit samples the complete validated main line rather than taking only openings; `-positions-per-game 0` retains all positions. Worker count does not change the output. `-max-positions 0` writes an empty file.

Each record contains `prompt`, `response`, `board_size`, `game_id`, `source_file`, `move_number`, `to_play`, `initial_player`, `komi`, `rules`, `result`, `handicap`, `initial_stones`, and `moves`. `moves` is the prefix **before** the target action, and setup stones are separate. `opponent_response` optionally contains the next validated move for auxiliary supervision. Coordinates use GTP (skip I; row 1 at the bottom).

`game_id` hashes the canonical game across eight board symmetries, ignoring annotations and filenames. Duplicate games with identical objectives are removed; differently annotated copies still belong to the same train/validation group. Missing komi is null and unspecified rules remain empty. No scoring objective is inferred from a directory name.

The parser supports square boards up to 19×19, setup ranges, pass moves, captures, simple ko, and no-suicide validation. It reads the first game and first variation. Unsupported mid-game setup/player changes and malformed metadata are rejected. Games requiring unsupported move legality stop before the first illegal label; preserve and check rules metadata before engine analysis or constrained decoding.

Use the Python preparation command to split whole games, preserve metadata, and optionally apply symmetry augmentation only to training data:

```bash
uv run python scripts/prepare_datasets.py --sgf-dir data/sgf_19x19 \
  --tag 19x19 --positions-per-game 64 --eval-ratio 0.05 --symmetries 8
```

See [the research and data audit](../PLAN/DATA_RESEARCH_2026.md) for schema consumers, measured leakage, and limitations. Performance claims require a workload-specific benchmark.
