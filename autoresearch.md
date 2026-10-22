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

## Bounded research round completed, 03:30 October 23

Branch: `autoresearch/go-19x19-20261022`. Source commits through `5723396`;
final documentation/journal commit follows. All bounded GPU jobs are finished.
Work ran from 23:15 to 03:30 with a usage-limit interruption at 00:08–01:15:
about three hours of active research, implementation and experiments excluding
that pause. Long training and larger tournaments remain for the user.

Primary report: `PLAN/EXPERIMENT_RESULTS_2026.md`. Research notes cover 38 distinct
primary papers. The journal through run 40 records commands, inputs, snapshots,
metrics and negative results. Generated artifacts remain ignored/local.

The development-selected recipe is teacher64/lr1e-4, compact text, D4×8,
120 updates, beam4, seed3407. Its fixed repeat uses seed3408. The chosen checkpoint
is `outputs/research/text_v4_visits64_lr0.0001_s3407/best`. Pilot recipe:
`config/policy_19x19_qwen06_pilot.yaml`. Larger unvalidated budget:
`config/policy_19x19_qwen06.yaml`; see `docs/EXPERIMENT_PROTOCOL.md`.

The frozen final test was released once: 123 positions/32 games, b18 at 32 visits,
all beam4, fixed selected/repeat/old/base plus controls. There were 891 queries;
the measured evaluation took 522.58 seconds. Selected minus base was
+4.968 [3.410, 6.833]; minus old +0.109 [−1.182, 1.398]; minus uniform
+1.150 [0.031, 2.317]. Repeat minus uniform was +0.580 [−0.655, 1.923].
All predictions were legal. These are engine estimates, not Elo. No model choice
was made using final outcomes. Raw report:
`outputs/research/final_quality_b18_frozen123_s3407.json`; paired statistics:
`outputs/research/final_quality_contrasts.json`; standalone figures:
`outputs/research/final_quality_frozen123.png` and `.svg`.

An independent numerical audit reproduced 22 summaries; 17 statistical-helper
tests passed. The repeat fingerprint audit excludes only the report-only
`training_state.pt` extra; all 22 frozen model/tokenizer/metadata hashes match.
Raw artifacts were unchanged. The b18 reference is selected and scored by the
same engine; independence means the evaluator differs from the b6 teacher.

The selected model lost both full games: Black W+368.5 after 530 plies, White
B+336.5 after 511. Model passes were 11/265 and 11/255, first pass at plies
489/474. Participation improved over the old empty-Black pass collapse;
playing-strength improvement is unproven. Old v2 also lost both games. No finite
Elo is reported. Another 128 learner states were exported for future training
only; no new labels or training used them.

Seeded categorical PPO produced four 128-ply continuations, 512 legal actions,
one PASS and zero endings. There were 128 optimizer steps in 390.61 seconds,
approximate KL 0.03454 and clip fraction 0.28516. Caps use critic bootstraps;
316 captures reopened space. There was no observed terminal-reward learning.
Learner refresh and categorical inference D4 also failed their bounded quality
comparisons.

Final verification: 778 CPU tests in 16.00 seconds, nine real KataGo tests in
10.25 seconds, Go race tests passed, Ruff F/E9 and diff checks clean. A late
input-role guard rejects marked final-test rows for checkpoint selection;
generic final evaluation remains supported. Earlier valid training inputs and
frozen weights are unaffected. Periodic/final checkpoints contain optimizer/RNG
state; `best` supports inference or warm starts. Use `--init-checkpoint` for a
new training budget.

Next research round: acquire many more independent games and reserve a NEW
untouched final set. Audit opening/endgame coverage; match teacher budget,
learning rate, data and decoder compute; repeat seeds; evaluate completed,
color-balanced games. Measure natural terminal-reward coverage before expensive
RL variant sweeps. Do not tune on this final sample again.

Reset side task is complete: schedule deleted and verified empty at 03:09;
zero banked resets consumed. Authorized TUI attempts selected only October23
05:01, and the service declined twice. No later reset was selected. Do not retry.
`PLAN/USAGE_RESET_2026-10-23.md` is the record.
