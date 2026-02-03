# Fast SGF Parser (Go)

This is an optional high-performance SGF parser written in Go for maximum speed, with output parity to the Python pipeline.

## Why Go?

- **10-50x faster** than Python for SGF parsing
- **Native parallelism** with goroutines
- **Zero Python dependencies** - runs as standalone binary
- **Low memory footprint** - efficient concurrent processing

## Building

```bash
cd go_parser
go build -o sgf_parser .
```

This creates a `sgf_parser` binary that can be used from Python or directly.

## Usage

### Direct Usage

```bash
./sgf_parser -sgf-dir ../data/sgf_9x9 -output ../data/fast_parsed.jsonl -max-positions 200000 -workers 8
```

### From Python

The Python data processing scripts will automatically detect and use this binary if available, falling back to Python parsing if not.

## Performance

On a typical dataset:
- **Python (sequential)**: ~1000 files/minute
- **Python (parallel, 8 cores)**: ~5000 files/minute
- **Go (8 goroutines)**: ~20,000+ files/minute

For very large datasets (100k+ SGF files), the Go parser provides significant speedup.

## Parity Notes

The Go parser now mirrors the Python dataset builder behavior:

- Main-line SGF parsing (first variation).
- Setup stones (`AB`/`AW`, including ranges like `aa:cc`).
- Board state updates with capture/suicide handling (as in `go_rl.board`).
- Prompt formatting identical to `BoardState.to_prompt()`.
- JSONL schema matches Python (`prompt`, `response`, `board_size`).
