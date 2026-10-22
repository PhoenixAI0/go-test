# Run a reproducible 19×19 experiment

This is an operator-run protocol, not a report of new results. Start with bounded
experiments, then spend a larger budget only after development evidence supports
the choice. The 32,768-position/2,000-update production recipe is an **unvalidated
starting budget**. See [measured results](../PLAN/EXPERIMENT_RESULTS_2026.md) for
what has actually been tested.

Run commands from the repository root after [setup](../README.md#install). Use
fresh paths throughout; do not reuse the example names for a second experiment.
All commands below are sequential so training and KataGo do not compete for the
same GPU. Replace the engine paths with installed files:

```bash
export KATAGO_BIN=/absolute/path/to/katago
export GO_TEACHER_MODEL=/absolute/path/to/teacher-network.bin.gz
export GO_EVAL_MODEL=/absolute/path/to/independent-evaluator-network.bin.gz
export GO_REVISION=c1899de289a04d12100db370d81485cdf75e47ca
mkdir -p logs/protocol_v1
git rev-parse HEAD > logs/protocol_v1/commit.txt
git diff --binary > logs/protocol_v1/tracked_changes.patch
uv pip freeze > logs/protocol_v1/packages.txt
```

Keep the actual source tree as well: a Git diff does not include untracked files.
The trainer and teacher manifests record input/model/config hashes, resolved
arguments, seed, and engine version. Preserve `uv.lock`, source manifests,
checkpoints, stage logs, and reports. A fixed seed is not a promise of bitwise
reproducibility across GPU types or driver/library versions.

## 1. Freeze development and final-test sources before model selection

This small example imports 128 games. `--max-games` is a cap, not a guarantee;
the downloader reuses a nonempty destination without expanding it. Use a new
directory for a larger import. Reserve final games before consuming the entire
archive for development.

```bash
uv run --locked python scripts/download_sgf.py --download --board-size 19 \
  --max-games 128 --output-dir data/protocol_v1/sgf --cache-dir data/cache
uv run --locked python scripts/prepare_datasets.py \
  --sgf-dir data/protocol_v1/sgf --output-dir data/protocol_v1 --tag 19x19 \
  --max-positions 8192 --positions-per-game 64 --eval-ratio 0.125 --seed 3407
uv run --locked python scripts/prepare_final_test.py \
  --exclude-dataset data/protocol_v1/sft_19x19.jsonl \
  --exclude-source-manifest data/protocol_v1/sgf/katago_2025_01_01/source.json \
  --cache-dir data/cache --games 32 --positions-per-game 4 --seed 20261023 \
  --output data/protocol_v1/final_source.jsonl
uv run --locked python scripts/prepare_eval_objective.py \
  --dataset data/protocol_v1/final_source.jsonl \
  --source-manifest data/protocol_v1/final_source.manifest.json \
  --rules koPOSITIONALscoreAREAtaxNONEsui0 --komi 7.5 \
  --output data/protocol_v1/final_target.jsonl
```

The final-source selector excludes imported SGF hashes and semantic game IDs.
The objective preparation only replays histories and checks validity; it does
not query a teacher or model. It removes incompatible/post-terminal histories
without replacing them based on quality. Retain the original and target files
with both manifests. Use the final file only for leakage checks until the model,
checkpoint, decoder and evaluation budget are frozen.

Split whole games before augmentation. The trainers/evaluator additionally reject
visible-state overlap up to D4 symmetry, with side-to-play, rules and komi included.
Resolve any such overlap during data preparation, before examining model results.
Do not enlarge the development corpus later with the reserved final games.

## 2. Freeze state coverage and label under one objective

Every teacher query uses positional superko, area scoring, no suicide/tax/button/
handicap bonus, `friendlyPassOk=false`, and komi 7.5. A named `chinese` preset is
different. Source responses and outcomes are provenance after relabeling.

First build a fixed development label set. Then select the bounded training
source, with an explicit opening quota, without using model quality:

```bash
uv run --locked python scripts/build_teacher_dataset.py \
  --dataset data/protocol_v1/sft_19x19.jsonl --split validation \
  --output data/protocol_v1/validation.jsonl --max-positions 128 \
  --positions-per-game 8 --sampling game-phase --min-move 0 \
  --max-visits 64 --prior-mix 0.5 --seed 3407 \
  --relabel-rules koPOSITIONALscoreAREAtaxNONEsui0 --relabel-komi 7.5 \
  --katago-model "$GO_TEACHER_MODEL" --katago-config config/katago_analysis_train.cfg
uv run --locked python scripts/prepare_opening_budget.py \
  --dataset data/protocol_v1/sft_19x19.jsonl \
  --background data/protocol_v1/sft_19x19.jsonl \
  --exclude-dataset data/protocol_v1/validation.jsonl \
  --exclude-dataset data/protocol_v1/final_target.jsonl \
  --seed 3407 --output data/protocol_v1/train_source.jsonl
```

`prepare_opening_budget.py` is a fixed small-study helper: 512 positions, with
64 at moves 0–20 (up to eight genuine empty starts) and 448 later positions. It
requires coverage of every source training game and can fail on a larger corpus.
Adapt the sampling design before scaling; do not assume that simply setting
`min_move=0` supplies enough openings. Inspect its manifest's actual quotas,
phase counts, per-game counts and exclusions. If empty starts are absent, revise
the source sampling before training. A forced `PASS` policy is legal but useless.

Label **the same frozen states** at two teacher budgets:

```bash
for visits in 8 64; do
  uv run --locked python scripts/build_teacher_dataset.py \
    --dataset data/protocol_v1/train_source.jsonl --split train \
    --output "data/protocol_v1/train_v${visits}.jsonl" --max-positions 512 \
    --positions-per-game 512 --sampling game --min-move 0 \
    --max-visits "$visits" --prior-mix 0.5 --seed 3407 \
    --relabel-rules koPOSITIONALscoreAREAtaxNONEsui0 --relabel-komi 7.5 \
    --katago-model "$GO_TEACHER_MODEL" --katago-config config/katago_analysis_train.cfg
done
```

Before comparing arms, verify equal selected `(game_id, move_number)` sets and
coverage, no terminal histories, and the intended network/config/rules hashes.
The teacher response budget changes here; the state sample, target objective,
validation labels and mixture coefficient do not. Different target conventions
require a separate experiment. Keep original artifacts when correcting a dataset.

## 3. Run bounded learning and decoder comparisons

Run the recipe smoke first using the new labels. It performs two updates on a
tiny real subset; success verifies the pipeline, not playing strength.

```bash
uv run --locked python scripts/train_19x19.py --smoke --dry-run \
  --train-data data/protocol_v1/train_v64.jsonl \
  --eval-data data/protocol_v1/validation.jsonl --output-dir outputs/protocol_v1/smoke
uv run --locked python scripts/train_19x19.py --smoke \
  --train-data data/protocol_v1/train_v64.jsonl \
  --eval-data data/protocol_v1/validation.jsonl --output-dir outputs/protocol_v1/smoke
```

A four-arm example crosses teacher visits and learning rate. These values are
experimental budgets, not claimed optima. Keep D4, update count, effective batch,
LoRA size, representation and checkpoint-selection cadence fixed:

```bash
for visits in 8 64; do
  for lr in 1e-4 5e-4; do
    uv run --locked python scripts/run_text_distill.py \
      --train-data "data/protocol_v1/train_v${visits}.jsonl" \
      --eval-data data/protocol_v1/validation.jsonl \
      --model Qwen/Qwen3-0.6B --revision "$GO_REVISION" \
      --encoding compact --symmetries 8 --steps 120 --batch-size 4 \
      --gradient-accumulation 1 --learning-rate "$lr" --seed 3407 \
      --lora-rank 16 --lora-alpha 32 --max-seq-length 1024 \
      --eval-every 20 --checkpoint-every 60 --generation-positions 8 \
      --output-dir "outputs/protocol_v1/t${visits}_lr${lr}_s3407"
  done
done
```

Repeat promising arms with additional training seeds before promotion. Change
representation (`compact`/`grid`), D4 (1/8), categorical architecture, or reasoning
in separate matched comparisons; do not combine every change into one arm.
Track wall time, tokens and peak memory along with quality. Token NLL and
categorical policy CE are different objectives and cannot rank each other.

Compare checkpoints and decoder widths on identical development states and one
fixed independent engine budget. Add all candidate models to the same call where
practical; include every training source through repeated `--train-data`:

```bash
uv run --locked python scripts/eval_move_quality.py \
  --dataset data/protocol_v1/validation.jsonl \
  --train-data data/protocol_v1/train_v8.jsonl --train-data data/protocol_v1/train_v64.jsonl \
  --text-model t8_lr1e4=outputs/protocol_v1/t8_lr1e-4_s3407/best \
  --text-model t8_lr5e4=outputs/protocol_v1/t8_lr5e-4_s3407/best \
  --text-model t64_lr1e4=outputs/protocol_v1/t64_lr1e-4_s3407/best \
  --text-model t64_lr5e4=outputs/protocol_v1/t64_lr5e-4_s3407/best \
  --text-beam-widths 1 4 8 --max-positions 128 --max-visits 64 \
  --seed 3407 --katago-model "$GO_EVAL_MODEL" \
  --katago-config config/katago_analysis_eval.cfg \
  --output logs/protocol_v1/development_quality.json
```

The evaluator always includes a seeded uniform legal reference and searches each
chosen child explicitly. Its game-bootstrap intervals reflect position/game
sampling, not engine error or repeated model selection. Beam width is part of the
policy and inference budget; raw complete-string likelihood is not a guarantee
of stronger Go. A good diagnostic result must survive games from the empty board.

## 4. Play paired full games, then test learner-state refresh

The following concrete command uses one candidate and beam 1; substitute the
development choice deliberately. Use a fixed opponent network/config, an even
game count, and the same time controls for every candidate. Twenty games is a
starting evaluation budget, not automatically precise strength evidence.

```bash
uv run --locked python scripts/create_match_config.py \
  --output logs/protocol_v1/development_match.json --board-size 19 --num-games 20 \
  --rules koPOSITIONALscoreAREAtaxNONEsui0 --komi 7.5 \
  --model-name qwen-protocol --katago-model "$GO_EVAL_MODEL" \
  --katago-config config/katago_gtp_eval.cfg \
  --main-time 300 --byoyomi-time 10 --byoyomi-periods 5 \
  --sgf-output-dir outputs/protocol_v1/development_games \
  --model-command 'uv run --locked python scripts/model_gtp.py --model outputs/protocol_v1/t64_lr1e-4_s3407/best --encoding compact --num-beams 1'
uv run --locked python scripts/eval_elo.py \
  --match-config logs/protocol_v1/development_match.json \
  --log-file logs/protocol_v1/development_games.log \
  --summary-json logs/protocol_v1/development_games.json
```

Colors alternate. Inspect SGFs, early pass behavior, caps, forfeits and completion
rates, not just aggregate wins. A move cap is `Void`, never a scored loss/win.
Zero/all wins do not justify a finite Elo estimate. Finished KataGo scores use
the pinned engine's rule semantics; analysis `scoreLead` is only an estimate.

Refresh from **development games only**, using the exact immutable checkpoint
that collected them. Export the learner's turns, then obtain new expert labels;
the learner's sampled actions are not imitation targets:

```bash
uv run --locked python scripts/export_learner_sgf.py \
  --sgf-dir outputs/protocol_v1/development_games --learner-name qwen-protocol \
  --checkpoint outputs/protocol_v1/t64_lr1e-4_s3407/best \
  --exclude-dataset data/protocol_v1/validation.jsonl \
  --exclude-dataset data/protocol_v1/final_target.jsonl \
  --max-positions 256 --per-game 32 --seed 3407 \
  --output data/protocol_v1/learner_source.jsonl
uv run --locked python scripts/build_teacher_dataset.py \
  --dataset data/protocol_v1/learner_source.jsonl --split train \
  --output data/protocol_v1/learner_v64.jsonl --max-positions 256 \
  --positions-per-game 32 --sampling game-phase --min-move 0 \
  --max-visits 64 --prior-mix 0.5 --seed 3407 \
  --relabel-rules koPOSITIONALscoreAREAtaxNONEsui0 --relabel-komi 7.5 \
  --katago-model "$GO_TEACHER_MODEL" --katago-config config/katago_analysis_train.cfg
uv run --locked python scripts/prepare_refresh_ablation.py \
  --expert-data data/protocol_v1/train_v64.jsonl \
  --learner-data data/protocol_v1/learner_v64.jsonl \
  --validation-data data/protocol_v1/validation.jsonl \
  --test-data data/protocol_v1/final_target.jsonl \
  --warm-start-checkpoint outputs/protocol_v1/t64_lr1e-4_s3407/best \
  --phase-match-control --seed 3407 --output-dir data/protocol_v1/refresh
```

Inspect `refresh/manifest.json` before running its `training_commands.json`:
require the intended learner quota, equal total examples, the same shared expert
prefix and warm start, and `exact_phase_match=true` with `fallback_positions=0`.
The helper reports shortages and may otherwise use fallback expert states; that
is a different comparison and must not be presented as exact phase matching.
Its current controlled recipe requires 64-visit labels with identical teacher
network/config/objective and a compact text checkpoint. Its generated output
paths are fixed research names: choose fresh paths before executing them.

Both arms need identical optimizer/update and decoder budgets. Report per-game
weights/effective games: many states from a few learner games can dominate the
mixture. This is offline teacher imitation on learner-visited states, not PPO or
a full DAgger reproduction. Never refresh from final-test games or select refresh
examples using final-test quality.

## 5. Scale the development winner, then release the final test once

Long training remains an operator decision. Copy and edit the YAML using the
development evidence, then print the complete plan before launching:

```bash
cp config/policy_19x19_qwen06.yaml config/protocol_v1_long.yaml
uv run --locked python scripts/train_19x19.py \
  --config config/protocol_v1_long.yaml --dry-run \
  --output-dir outputs/protocol_v1/long
```

Set the larger source/label paths, representation, augmentation, batch, schedule,
and teacher budget explicitly. Freezing more labels requires enough independent
games and measured phase coverage. Do not repeatedly cycle 512 positions and
call that scaling the data. The YAML's 32,768/2,000 defaults remain unvalidated.
Run the same command without `--dry-run` only when ready for the long job.
Use [the documented resume command](../README.md#direct-training-recipe) for an
interrupted identical schedule; a changed objective or budget uses
`--init-checkpoint` with a new optimizer and fresh output.

Before final evaluation, record the chosen checkpoint directory and file hashes,
all training dataset hashes (including refresh sources), model revision,
representation, decoder width, random seeds, opponent network/config, search
budget and time controls in an immutable selection record. Freeze these choices
using development data only. The example below evaluates the concrete earlier
candidate; replace it with that frozen selection, not a list of contenders:

```bash
uv run --locked python scripts/eval_move_quality.py \
  --dataset data/protocol_v1/final_target.jsonl --split test \
  --train-data data/protocol_v1/train_v64.jsonl \
  --text-model selected=outputs/protocol_v1/t64_lr1e-4_s3407/best \
  --text-beam-widths 1 --max-positions 128 --max-visits 64 --seed 20261023 \
  --katago-model "$GO_EVAL_MODEL" --katago-config config/katago_analysis_eval.cfg \
  --output logs/protocol_v1/final_quality.json
```

Repeat `--train-data` for every source actually used by the selected model,
including pretraining adapters and learner refresh. The final target has no
teacher labels; this evaluator queries reference/selected moves directly. Also
run a new, predeclared paired full-game match with the frozen policy/opponent;
the held-out-position report and complete-game evidence answer different questions.
Do not tune after seeing either final result. Any subsequent iteration needs a
new untouched final-test set. Keep negative results, capped games, and all
artifact manifests in the report.
