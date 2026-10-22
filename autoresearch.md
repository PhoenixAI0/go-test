# Full-size Go research session

Started 2026-10-22 23:15 Asia/Singapore. Work actively for at least three hours
(through approximately 2026-10-23 03:20 after the usage interruption); do not
fill the budget with waiting.

## Objective and decision rule

Train Qwen3-0.6B to play complete 19x19 Go games. Optimize held-out move quality
and eventually paired match win rate at fixed inference/teacher budgets. Legal
move rate, token count, latency, loss, and teacher agreement are diagnostics;
none alone demonstrates playing strength. No hand-coded tactical move selector.

Small local experiments run on a 32 GB Apple Silicon Mac. Long CUDA training is
prepared for the user, not run here. Keep reproducible seeds, dataset/game IDs,
model revisions, commands, results, failures, and limits. Never tune on the final
test split. Keep synthetic fixtures separate from real-data quality experiments.

## Baseline

- Commit: d7ec6f4; clean initial worktree.
- Branch: autoresearch/go-19x19-20261022.
- `uv run --locked pytest -m 'not katago' -q`: 373 passed, 3 skipped,
  6 deselected, 18.57 seconds.
- Torch 2.9.1, Transformers 4.57.6, TRL 0.24.0; MPS available.
- Model: Qwen/Qwen3-0.6B, revision
  c1899de289a04d12100db370d81485cdf75e47ca.

## Research and work allocation

- Root: model interface, local experiments, categorical policy/value alternative,
  integration and final run recipes.
- rl_research: RL literature, GRPO objective and root-continuation reward fixes.
- go_correctness: rule/verifier correctness, engine performance and matches.
- data_research: parser metadata, game-level splits, symmetry, real dataset.

Research notes live in PLAN/*RESEARCH_2026.md. Results are append-only in
autoresearch.jsonl; large generated data/checkpoints/logs stay in ignored paths.

## Initial findings

- Existing full-game GRPO trains the initial completion only, not every move.
- Per-move survival bonuses dominate terminal outcome reward.
- Move-limit truncations are incorrectly labeled as terminal wins/losses.
- Matrix prompts discard historical legality context; engine payloads can lose ko.
- Row-random SFT split leaks positions from the same game across partitions.
- No bundled model GTP engine; match evaluation requires a user implementation.

## Experiment queue

1. Correct rules/rewards/splits and retain regression tests.
2. Compare text encodings and legal constrained decoding on held-out real games.
3. Compare categorical Qwen policy/value heads with text SFT at bounded budgets.
4. Compare hard move targets with dense teacher policy/value supervision.
5. Run small on-policy learning/verification tests and complete GTP games.
6. Review failures, run another targeted ablation, and prepare long-run recipes.

## Timed side task

At 01:15 on 2026-10-23 Asia/Singapore, inspect interactive Codex `/usage` and
redeem only the reset expiring approximately 05:00 that morning, if still present.
Never consume later resets. The T3 task was deleted and absence verified. The
authorized attempts are complete; zero resets were consumed. Evidence is in
PLAN/USAGE_RESET_2026-10-23.md. Do not retry or recreate the schedule.

## Resume

Read this file, autoresearch.jsonl, PLAN research notes, `git diff --stat`, and
agent messages. Preserve all in-flight changes and avoid repeating experiments.

## Resumed after usage-limit interruption

The goal runtime stopped at a usage limit around 00:08 Singapore (53 minutes of
recorded active goal time). The one-time reminder resumed this thread at 01:15,
after the regular weekly reset. To preserve the requested substantial active
research time rather than count the inactive interval, continue useful research
and bounded experiments through approximately **03:20 Singapore** if permitted.
The reset side-task authorization still ends around 02:00; do not extend it.

## Active state at 02:44, October 23

The reset schedule is deleted and its absence was verified. Two authorized TUI
attempts selected only the October 23 05:01 reset; Codex declined both because
usage did not need resetting. All three remained. Zero consumed; do not retry.

The compact text v2 beam-4 development winner lost both full games: Black
W+40.5 after 70 plies (34/35 model turns were PASS), White B+337.5 after 509.
Both completed by two passes without caps or forfeits. The old train sets had
no empty boards and validation began at move 18. Greedy also passed from empty.

Four 120-update compact/D4 arms on the new 512-state opening-inclusive v4 source
are finished: teacher visits 8/64 crossed with learning rates 1e-4/5e-4, seed
3407. Source has eight empty starts, 56 other positions at moves 1–20, and 448
later positions, from 112 training games. All held-out game/D4 exclusions pass.
Commands and source snapshots: logs/research/teacher_budget_queue.json and
logs/research/source_snapshots/. Outputs: text_v4_visits{8,64}_lr{0.0001,0.0005}_s3407.
Best NLLs: 1.586295, 1.608759, 1.620971, 1.712080, respectively. Do not rank their
Go quality by those losses; validation labels were generated at eight visits.

GPU is now reserved for logs/research/run_refresh_queue.py, session 25384,
launched around 02:40. It warms the old compact-v2 checkpoint on equal 256-row
control/mix datasets under data/refresh_ablation_v2. Both share 192 expert rows;
the remaining 64 expert or learner rows are exactly matched by six game phases.
Both have five empty states. Learner labels come from 99 states exported from
the two actual failed games and queried at 64 visits. Exact training commands
are logs/research/refresh_queue_commands.json (120 steps, D4×8, lr1e-4, seed3407,
eval every60, generation8). v1 prepared datasets were never trained.

After refresh, give rl_research the GPU for prepared
logs/research/run_v4_quality.py: all six new beam-4 text candidates, old-v2 beam4,
categorical D4=1/8, static frequencies, same32 valid development states, b18@32.
It also runs four opening probes. White passing after Black passes from empty
wins W+7.5, so do not mislabel all opening passes as failures. Selection protocol
is frozen in logs/research/v4_selection_protocol.json: all legal + empty Black
non-PASS, then highest equal-game score advantage. No eligible arm means no
promotion claim. Test the choice in two color-swapped complete games.

Next: optionally repeat the chosen training seed; freeze model/decoder choice,
then evaluate the untouched 123-position /32-game target final set once. Also run
actual two-step train_19x19 smoke with teacher_v4_visits64 and val_v3, plus seeded
PPO (four actual sampled continuations, at most128 new plies each) from prepared
logs/research/seeded_ppo_v4_command.json. Caps remain bootstraps, never outcomes.
Defer all long training to the user. Continue useful work through at least03:20.

Nine real KataGo1.18.2 integration tests passed in10.25 seconds. The Benson
fixture produces formal B+353.5 vs raw-board B+347.5, confirming the separate
score conventions. Latest CPU suite before final metadata edits:737 passed.
Final full CPU suite, Ruff, and diff checks remain after all code changes.

New text checkpoints now persist validated board size/rules/komi/encoding; GTP
restores them. Spatial heads reject incompatible input encodings. Existing
compact full-game commands passed explicit objective/encoding and are unaffected.

Scientific results: PLAN/EXPERIMENT_RESULTS_2026.md; journal through run30.
Earlier32-position quality reports each contained one post-terminal state;
use31-position post-hoc audits or later preselection-filtered32-position reports.
Final target SHA: b2cc7402fb88bbb1595323e884854a2e21f99892c9a5a00e4db10375f6486070.
