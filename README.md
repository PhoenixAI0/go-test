# Learning full-size Go with Qwen

This repository trains and evaluates language-model Go policies on **19×19 boards**.
The current starting point is **Qwen3-0.6B**, teacher-labeled positions, and short
controlled experiments before expensive training. The direct recipe pins model
revision `c1899de289a04d12100db370d81485cdf75e47ca`.

The implementation supports complete games, exact supported-rule legality, and
independent KataGo evaluation. That does **not** mean the small experimental
checkpoints are strong players. Read [experiment results](PLAN/EXPERIMENT_RESULTS_2026.md)
for measured improvements, negative results, budgets, and remaining uncertainty.
Use the [experiment protocol](docs/EXPERIMENT_PROTOCOL.md) for a stepwise handoff
covering frozen final data, controlled sweeps, full games, and learner-state refresh.
The older 9×9 → 19×19 CUDA curriculum remains documented in
[Legacy curriculum](docs/LEGACY_CURRICULUM.md).

## Install

Use Python 3.12 or 3.13; the locked setup uses 3.12. Run from the repository root:

```bash
uv sync --locked --python 3.12
./scripts/compile_go_parser.sh
```

The direct experimental runners accept `--device auto|cuda|mps|cpu`; the bounded
Qwen experiments were exercised on Apple MPS. CPU supports tests and tiny models.
For Linux NVIDIA setup, use `./scripts/setup_environment.sh`; the existing x86_64
and aarch64 package choices, system prerequisites, and CUDA smoke checks are in
the [legacy setup guide](docs/LEGACY_CURRICULUM.md#run-the-entire-training-curriculum-on-cuda).
Validate CUDA execution on the target machine before a long run.

Teacher generation and evaluation need a working KataGo binary and network:

```bash
export KATAGO_BIN=/absolute/path/to/katago
export KATAGO_MODEL=/absolute/path/to/network.bin.gz
```

Linux setup writes these paths to `.env`. To use that file with Python commands,
export its values using `set -a; source .env; set +a`. The direct recipe reads the
exported environment; it does not source shell files. Analysis and GTP use separate
configs in `config/`. Teacher network hashes, engine versions, and visit budgets
are recorded so a cheap teacher is never confused with a stronger evaluation.

## Prepare real 19×19 positions

Import the checksum-pinned public KataGo archive, or put your own SGFs in the same
directory. Preserve a separate final-test corpus before repeated model selection.

```bash
uv run python scripts/download_sgf.py --download --board-size 19 --max-games 10000
uv run python scripts/prepare_datasets.py \
  --sgf-dir data/sgf_19x19 --output-dir data --tag 19x19 \
  --max-positions 200000 --positions-per-game 64 \
  --eval-ratio 0.05 --seed 3407
```

The downloader reuses an existing nonempty SGF directory; increasing `--max-games`
does not expand an earlier small import. For a larger fresh corpus, pass a new
`--output-dir` to the downloader and that same directory to `--sgf-dir`. Keep its
`katago_2025_01_01/source.json`: it records the archive URL/SHA-256 and selected
content-hashed SGF filenames. Dataset preparation records the parser-output hash,
split seed and validation game IDs; training records the exact input JSONL hashes.
Retain these manifests with the original SGFs so the source selection can be rebuilt.

The parser preserves semantic `game_id`, source provenance, rules, komi, setup
stones, and move history. Sampling spans games rather than taking only opening
prefixes. `prepare_datasets.py` writes `sft_19x19.jsonl`, separate train/validation
files, training-only RL prompts, and a dataset manifest. Whole games are assigned
to one split before optional D4 augmentation; experimental trainers also reject
visible-state overlap up to symmetry. Do not use random row splits.

Audit selected training coverage from the empty board through opening, middlegame,
endgame, and passing. The recipe includes move zero, but `game-phase` sampling alone
does not guarantee enough empty-board or early-opening examples. An earlier model
trained without the first ten plies passed on 34 of 35 Black turns in a real game;
validation beginning later in games missed that failure. Before a long run, inspect
phase counts and use an explicit opening quota when needed. The bounded research
helper [prepare_opening_budget.py](scripts/prepare_opening_budget.py) freezes 512
positions with 64 opening states (including up to eight real empty-board states)
and 448 later states, excludes held-out games/states, and records source hashes.
Its fixed budget is an experimental example to adapt, not a tuned production ratio.

The source games have varied objectives. Teacher generation replays each history,
rejects histories incompatible with the requested rules, and reanalyzes under a
fixed target: **positional superko, area scoring, no suicide, komi 7.5**. Source
outcomes are retained as provenance, not reused as labels for that new objective.
The named KataGo ruleset `chinese` uses simple ko and is not an alias for this target.

## Direct training recipe

Inspect [config/policy_19x19_qwen06.yaml](config/policy_19x19_qwen06.yaml), then print
the exact commands. Dry-run loads no models, starts no engines, and writes nothing:

```bash
uv run python scripts/train_19x19.py --dry-run --smoke \
  --output-dir outputs/qwen06_smoke
```

Execute the same bounded integration check on real positions:

```bash
uv run python scripts/train_19x19.py --smoke \
  --output-dir outputs/qwen06_smoke
```

This creates up to eight training and four validation teacher positions and runs
two updates. It is an integration check, not a useful training budget. Every run
requires a fresh output path and records resolved commands, input/code hashes,
stage logs, a reusable `recipe.yaml`, and a completion/failure manifest. Reusing
labeled data is supported:

```bash
uv run python scripts/train_19x19.py --smoke \
  --train-data data/teacher_train.jsonl --eval-data data/teacher_validation.jsonl \
  --output-dir outputs/qwen06_reuse_smoke
```

To build teacher files independently:

```bash
uv run python scripts/build_teacher_dataset.py \
  --dataset data/sft_19x19.jsonl --split train \
  --output data/teacher_train.jsonl --max-positions 32768 \
  --positions-per-game 32 --sampling game-phase --min-move 0 --max-visits 32 --prior-mix 0.5 \
  --relabel-rules koPOSITIONALscoreAREAtaxNONEsui0 --relabel-komi 7.5 \
  --katago-config config/katago_analysis_train.cfg
uv run python scripts/build_teacher_dataset.py \
  --dataset data/sft_19x19.jsonl --split validation \
  --output data/teacher_validation.jsonl --max-positions 512 \
  --positions-per-game 32 --sampling game-phase --min-move 0 --max-visits 32 --prior-mix 0.5 \
  --relabel-rules koPOSITIONALscoreAREAtaxNONEsui0 --relabel-komi 7.5 \
  --katago-config config/katago_analysis_train.cfg
```

The longer starting recipe uses up to 32,768 teacher positions and 2,000 updates
at effective batch 32. These are **unvalidated starting budgets**, not tuned
optima; the long run is left to the operator:

```bash
uv run python scripts/train_19x19.py --output-dir outputs/qwen06_19x19
```

Default `train_kind: text` uses compact board inputs and completion-only LoRA SFT.
It is a provisional choice based on small move-quality experiments. Select the
categorical ablation with `--train-kind categorical`; it uses a 362-action
policy/value head and experimental cell readout with bidirectional attention.
`--encoding` overrides the selected input representation. Compare alternatives at
matched data and compute; text-token NLL and categorical policy CE are not directly
comparable. Both trainers save `distill/best`, `distill/final`, metrics, and periodic
`distill/checkpoint-XXXXXXXX/` snapshots.

Source histories and D4 augmentation stream into `prepared_train.jsonl` and
`prepared_validation.jsonl` caches. Training keeps compact row offsets in memory
and reads token/target records as sampled; it does not retain eight copies of every
game history. Allow disk space for these derived caches, especially for D4 runs.

Resume the same planned schedule, data, hyperparameters, and device type into a
fresh directory; `--steps` in the YAML remains the original total budget:

```bash
uv run python scripts/train_19x19.py \
  --config outputs/qwen06_19x19/recipe.yaml \
  --train-data outputs/qwen06_19x19/teacher_train.jsonl \
  --eval-data outputs/qwen06_19x19/teacher_validation.jsonl \
  --resume-from outputs/qwen06_19x19/distill/checkpoint-00000500 \
  --output-dir outputs/qwen06_19x19_resumed
```

The checkpoint includes optimizer, scheduler, sampling RNG, and device RNG state
bound to the saved model files. It also preserves the earlier best model, so
resuming does not discard a better checkpoint from before the interruption.
A compatible snapshot without that copy reports best selection over the
continuation segment only. Resume also verifies prepared-data hashes and the
evaluation schedule. Older trainer-contract versions, changed schedules, or new
objectives require `--init-checkpoint PATH` and a fresh optimizer. The recipe
defaults to saving every 500 updates.

## Model interface and evaluation

Inputs contain the current board, side to play, normalized rules, komi, and recent
moves. Full history stays in the verifier for ko/superko; the model is never fed a
silently truncated board. Text output is exactly one GTP coordinate or `PASS`.
Legal-token decoding constrains syntax and legality, while the model still chooses
the move. Greedy decoding selects the best next token at each prefix. Text engines
also accept `--num-beams 4` or `8`, ranking candidate complete move strings by raw
model log-probability including EOS, without a length penalty. Finite beam search
is approximate; neither interface is exact global conditioning over legal strings.
Categorical policies instead emit one masked action distribution.

The bundled GTP engine serves either checkpoint type:

```bash
uv run python scripts/model_gtp.py \
  --model outputs/qwen06_19x19/distill/best --encoding compact
# Categorical: replace --model with --checkpoint and omit --encoding.
```

Evaluate move quality on held-out positions using a fixed-budget engine, including
uniform and fresh-search references:

```bash
uv run python scripts/eval_move_quality.py \
  --dataset outputs/qwen06_19x19/teacher_validation.jsonl \
  --train-data outputs/qwen06_19x19/teacher_train.jsonl \
  --text-model qwen=outputs/qwen06_19x19/distill/best --encoding compact \
  --max-positions 128 --max-visits 64 \
  --katago-config config/katago_analysis_eval.cfg \
  --output logs/qwen06_move_quality.json
# Categorical: use --policy-checkpoint qwen=PATH instead of --text-model.
# Text decoding ablation: add --text-beam-widths 1 4 8 for shared engine estimates.
```

These are approximate teacher estimates with game-cluster uncertainty, not exact
regret or Elo. Imitation agreement and legality alone do not measure playing strength.
Use actual games for that assessment:

```bash
uv run python scripts/create_match_config.py \
  --output config/qwen06_match.json --board-size 19 --num-games 20 \
  --rules koPOSITIONALscoreAREAtaxNONEsui0 --komi 7.5 \
  --model-command 'uv run python scripts/model_gtp.py --model outputs/qwen06_19x19/distill/best --encoding compact'
uv run python scripts/eval_elo.py --match-config config/qwen06_match.json \
  --log-file logs/qwen06_matches.log --summary-json logs/qwen06_matches.json
```

Colors alternate. The trusted engine validates moves and scores two-pass endings;
illegal actions, resignation, and clock failures are explicit outcomes. Capped
games are `RE[Void]`, excluded from scored win rates/Elo, and reported with cap
rates. A few games or games against one weak opponent cannot establish a robust
rating. Add `--evaluate` to the training recipe to run both diagnostics and matches
automatically; smoke caps remain integration-only outcomes.

## RL and iterative improvement

`run_grpo.py` supports explicit GRPO-family objective and sampling choices. The
legacy `run_grpo_fullgame.py` assigns a continuation reward to the initial text
completion; it does **not** optimize every later action. Its engine potential is
an estimate, and truncation is not a terminal win.

`run_policy_rl.py` is a separate categorical self-play PPO path. It trains all
collected decisions with actual behavior probabilities, alternating-player value
signs, and final outcomes from a KataGo GTP referee. Caps bootstrap from the next
value estimate rather than inventing a winner. Use
`train_19x19.py --train-kind categorical --selfplay` to append it; text models cannot
silently enter this trainer. Text GRPO remains a separately configured experiment.

`scripts/export_learner_positions.py` exports learner-visited states for fresh
teacher annotation and another distillation round, excluding declared holdouts.
Learner actions are provenance, not expert targets. Track this loop against a
fixed holdout and independent full-game opponents before scaling it. Keep the
final-test games unused until model and checkpoint selection are frozen.

## Tests and research records

```bash
uv run --locked pytest -m 'not katago'
uv run --locked pytest -m katago  # Requires exported KataGo paths.
(cd go_parser && go test ./...)
```

Tests include actual tiny Qwen/TRL/PEFT training, legal-action and history checks,
checkpoint reload/continuation, reward perspectives, dataset isolation, and GTP
contracts. Real 0.6B experiment outcomes are recorded separately from unit tests.

- [Experiment results](PLAN/EXPERIMENT_RESULTS_2026.md) and [experiment journal](autoresearch.jsonl).
- [RL and verifiable-reward papers](PLAN/RL_RESEARCH_2026.md).
- [Model interfaces and representation papers](PLAN/MODEL_INTERFACE_RESEARCH_2026.md).
- [Data and distillation papers](PLAN/DATA_RESEARCH_2026.md).
- [Go engine correctness](PLAN/GO_ENGINE_RESEARCH_2026.md).
- [Legacy CUDA curriculum, multi-GPU wrappers, and upload commands](docs/LEGACY_CURRICULUM.md).

`go_rl/` contains the library, `go_parser/` the SGF parser, `scripts/` entry points,
`config/` recipes, and `tests/` validation. Generated data, checkpoints, and logs
belong under `data/`, `outputs/`, and `logs/` and stay out of version control.
