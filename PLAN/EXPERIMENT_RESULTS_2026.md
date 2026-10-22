# Qwen3-0.6B, full-size Go: research and measurements

Session: October 22–23, 2026, Asia/Singapore. Baseline commit `d7ec6f4`;
working branch `autoresearch/go-19x19-20261022`. The experiment journal is
[`autoresearch.jsonl`](../autoresearch.jsonl). Generated datasets, checkpoints,
reports and exact source snapshots remain in ignored local directories.

## Current conclusion

The repository now has an executable 19×19 learning and evaluation path rather
than relying on a missing external model GTP implementation. Compact text SFT
is the provisional starting recipe. A categorical policy/value model and
complete-turn PPO remain useful controlled alternatives. None of the small
checkpoints should yet be described as a strong Go player.

The first encouraging move-quality result did not establish a robust gain
across training seeds. Lower imitation loss also failed to predict better moves.
The practical outcome is a much more reliable experiment system, measured
interface and efficiency improvements, and explicit evidence about which
quality hypotheses still need larger runs. Long training is left to the user. The one-time final
comparison found +4.97 estimated points over base Qwen but only +0.11 over the
previous checkpoint, with that latter interval crossing zero. Both selected
full games were losses. There is no demonstrated large playing-strength gain.

## Research translated into code

The reading notes cite 38 distinct primary papers: 36 unique arXiv papers plus
DAgger and ALLIE in conference proceedings, excluding the engineering blog and
source-code/documentation links. They record method sections, assumptions and
counterevidence, and distinguish implemented ideas
from deferred hypotheses and do not claim full reproductions:

- [RL/GRPO, reasoning, verifiable rewards and recent distillation](RL_RESEARCH_2026.md)
- [Data, curriculum, sampling and learner-state refresh](DATA_RESEARCH_2026.md)
- [Go rules, engine correctness and performance](GO_ENGINE_RESEARCH_2026.md)
- [Qwen interfaces, finite actions and bidirectional encoding](MODEL_INTERFACE_RESEARCH_2026.md)
- [Spatial readout review](SPATIAL_MODEL_REVIEW_2026.md)

The main implemented consequences are complete state/history preservation;
game-level and D4 state isolation; fixed teacher rules/komi; searched and dense
policy targets; completion-only text supervision; categorical policy/value
heads; explicit GRPO reductions and reasoning-token credit; signed alternating
player returns; real behavior probabilities; and strict separation of two-pass
completion from rollout caps. These address actual failures in the old code,
not just configuration modernization.

No manual tactical move selector is added. The model chooses among legal moves.
Compute goes toward teacher labels, model adaptation, controlled decoding, and
independent evaluation. The new paper-inspired readouts are ablations, not
presumed improvements.

## Reproducibility and data

- Hardware: Apple Silicon, 32 GB shared memory, MPS; Torch 2.9.1,
  Transformers 4.57.6, PEFT 0.18.1, TRL 0.24.0.
- Model: `Qwen/Qwen3-0.6B`, immutable revision
  `c1899de289a04d12100db370d81485cdf75e47ca`.
- Teacher: KataGo 1.18.2, b6c96 network
  `kata1-b6c96-s175395328-d26788732.txt.gz`; independent evaluator: frozen b18c384
  `kata1-b18c384nbt-s9996604416-d4316597426.bin.gz` with a configured 32-visit cap per root/child.
- Target: positional superko, area scoring, no suicide/tax/button/handicap bonus,
  `friendlyPassOk=false`, komi 7.5. Named `chinese` is a different ruleset.
- Original public archive: `https://katagoarchive.org/kata1/traininggames/2025-01-01sgfs.tar.bz2`,
  SHA256 `98dc06610ccad000ed532fa00c7daf76157d150a1319fb147493c1cf26c5b4ca`.
- Initial corpus: 8,166 full-history positions from 128 games; 112 training
  games and 16 development games. A separate next 32 unused archive games were
  frozen before the b18 development comparisons, after the first categorical
  training run. They have not been used for training or development-quality choices.
- v1 teacher data: 256 training positions, 64 validation positions, b6 at 8
  visits, equal prior/search mixture. v2 adds phase-balanced training positions
  through the endgame: 1,024 source positions, optionally eight D4 transforms.

Source games have varied rules, komi and cleanup continuations. A later audit
found histories at or beyond the first two-pass boundary in 5/256 v1 training,
78/1024 v2 training, and 1/64 validation rows. Such continuation can be meaningful
in the source protocol but is invalid as another decision in our fixed target
game. Original artifacts remain immutable. Future builders reject these states.

The corrected validation file is `data/teacher_19x19_val_v3.jsonl`: 63 positions,
the same 16 development games and unchanged retained labels, SHA256
`e367af48c3c967d4318c7a586e9d715501acff4ecf5ef73ee93fcfaedf8f7b62`.
It is not a new independent holdout. The target frozen until final evaluation has 123 positions
from all 32 frozen games after five terminal-history exclusions, SHA256
`b2cc7402fb88bbb1595323e884854a2e21f99892c9a5a00e4db10375f6486070`.
No replacements were selected based on labels or model outputs.

## Measured interface and training results

All early training arms use LoRA rank 16/alpha 32 and batch 4. Unless stated
otherwise they use 60 updates, learning rate 1e-4, seed 3407, and v1 data. Text
token NLL and categorical policy CE measure different objectives. Timings are
local run measurements including periodic evaluation, not isolated GPU-kernel times.

| Arm | Initial → final validation diagnostic | Local training seconds | Interpretation |
|---|---:|---:|---|
| Global categorical head, initialization std .01 | policy CE 6.234 → 5.294 | 148 | Worse than uniform legal CE 5.205 |
| Global head, std .001 | 5.215 → 5.288 | 136 | Best checkpoint was initial; reject presumed gain |
| Shared cell head, causal | 5.199 → 5.120; best 5.107 | 157 | Early cells lack later-board context |
| Shared cell head, bidirectional | 5.198 → 5.036 | 156 | Better imitation, not demonstrated stronger play |
| Shared cell head, bidirectional, seeds 3408/3409 | final 5.066 / 5.003 | 135 / 135 | CE improvement repeats |
| Global-query cell attention, causal | 5.213 → 5.148 | 135 | Did not beat bidirectional cell CE |
| Compact text | completion NLL 5.786 → 1.704 | 158 | Format learning plus uncertain move-quality gain |
| Compact text, seeds 3408/3409 | final 1.740 / 1.728 | 140 / 140 | Similar imitation across seeds |
| Compact text, v2 + D4, 180 updates | 5.786 → 1.567 | 359 | Lower NLL did not establish better moves |
| Grid text, v2 + D4, 180 updates | 5.510 → 1.562 | 620 | Nearly equal NLL at about 1.7× time |

Raw base-model output produced zero legal moves on the initial 32 encoding/position
trials, all truncated at 16 tokens. Legal-trie decoding produced 32/32 legal
moves, but no teacher-move agreement. This solves an interface failure; it does
not prove strategic skill. After the first compact SFT run, raw legality was
4/8, versus 8/8 with the constraint. Larger v2 runs had only 1/8 raw legality
despite lower NLL, another reason to evaluate actual behavior.

An explicit base-model reasoning trial used four development positions and
128 thinking tokens. All four exhausted the budget before a final move;
average latency was 5.47 seconds versus 0.33 seconds for a direct constrained
move. This is a failure at that budget, not a claim that longer or trained
reasoning cannot help. Evidence:
`logs/research/interfaces_qwen3_0_6b_thinking128_v1.json`.

Optional beam widths 4/8 maximize approximate complete legal-move likelihood,
including EOS, without local renormalization or length normalization. Tests
exercise actual Transformers generation and catch prefix-greedy failures and
inherited generation settings. Beam search is inference only; its chosen action
must not be assigned an ordinary sampled-policy likelihood in PPO.

## Independent move-quality evaluation

Every selected model move, reference move and seeded uniform move receives an
explicit child-position query at the same b18 visit cap. Unsearched actions are
never assigned fabricated regret. Scores are from the original player's
perspective; confidence intervals resample games, not individual correlated
positions. They omit engine approximation error.

The initial 32-position reports each contained one post-terminal history.
Separate 31-position audit reports remove that row without new queries and
preserve the original reports and hashes. Do not cite the old 32-position
summary without this qualification. Future evaluations filter before sampling.

In the corrected repeat study, text score advantage over one uniform legal move
was **+3.095, +1.199, +0.110 points** for seeds 3407, 3408, 3409. The corresponding
bidirectional categorical results were **+0.326, −0.662, −0.354**. All categorical
game-bootstrap intervals and two of three text intervals included zero.

A crossed bootstrap resampling both the three training seeds and 16 games gives:

| Comparison | Mean score difference | Exploratory 95% interval |
|---|---:|---:|
| Text versus uniform | +1.47 | [−1.11, +3.50] |
| Bidirectional categorical versus uniform | −0.23 | [−2.14, +1.42] |
| Text versus categorical, paired seeds/games | +1.70 | [−0.50, +4.17] |

Evidence: `outputs/research/seed_game_bootstrap_v1.json` and its source report.
Three seeds are a small sample; repeated development selection adds uncertainty.
These are estimated move-quality differences, not Elo or verified game outcomes.

A subsequent decoder study filtered terminal histories before sampling 32 valid
states and querying them, retaining the same 16 development games. It contains
the corrected study's 31 states plus one replacement; differences between those
reports are not paired model improvements. Compact-v2 beam 4 scored +3.95 points versus a
seeded uniform move, with a game-bootstrap interval [1.73, 6.57]. Its paired
difference from greedy was +2.34 [0.04, 6.14], at roughly twice the inference
time. Beam 4 hurt the v1 checkpoint instead, and base Qwen beam 8 passed on
31/32 positions. Approximate complete-string likelihood is therefore an
inference option to measure, not a universally better policy.

A training-only coordinate-frequency control scored −0.65 [−3.85, 1.91]. The
v1 greedy model exceeded it by +3.95 [0.68, 7.78]. This argues against a static
coordinate prior explaining the entire measured gain, without establishing
deep Go reasoning. These development comparisons are exploratory and their
intervals are not adjusted for selecting among multiple arms. Evidence:
`outputs/research/move_quality_b18_beams_frequency_valid32_s3407.json`.

## Full games changed the optimization direction

The development winner, compact-v2 beam 4, played two actual 19×19 games from
empty against b6 KataGo at eight visits, swapping colors. Both finished by two
passes, with zero caps or forfeits. It lost both:

| Model color | Result | Plies | Model passes / turns |
|---|---|---:|---:|
| Black | W+40.5 | 70 | 34 / 35 |
| White | B+337.5 | 509 | 19 / 254 |

No finite Elo is inferred from two losses. The Black model passed on its first
18 turns. Both greedy and beam 4 also passed on a direct empty-board diagnostic.
The existing training sets excluded the opening: neither included an empty
board, and validation started at move 18. Positive sampled-position results
had missed an essential part of the actual task. SGFs, scores, pass statistics,
and collection-time checkpoint fingerprints are retained under
`outputs/research/text_fullgames_compact_v2_beam4/`.

The next controlled source contained 512 states from 112 training
games: eight real empty starts, 56 other positions at moves 1–20, and 448
later positions. It preserves all held-out game and D4-state exclusions. The
same states receive b6 labels at eight and 64 visits, followed by four equal
120-update runs crossing teacher budget and learning rates 1e-4/5e-4. The
teacher's chosen move changes on 250/512 states. Neither teacher budget passes
in the 64 opening states. Future builders default to move zero.

A separate learner-state refresh exported 99 model-turn states from those
two failed games and labeled them at 64 visits. The bounded comparison uses
256 examples per arm: a shared 192 expert states, then either 64 learner states
or 64 additional expert states matched exactly in six move-number bins. These
expert states are not necessarily unseen by the shared warm-start checkpoint. Both arms
have five empty-board rows and the same optimizer budget. Their different game
concentration remains a limitation: the learner data comes from only two games.
The immutable v2 manifests document selection and hashes; v1 was prepared but
not trained. This implements a small DAgger-inspired data-distribution test,
not a complete reproduction or proof of on-policy distillation gains.

All six runs completed. Each of the four base-initialized arms selected its
final step 120 by validation loss. Both refresh arms instead retained their
initial weights as best by that loss. Before any Go-quality queries, the
protocol was amended to compare **both refresh final checkpoints**, preserving
equal update budgets. Nothing was selected using final-test outcomes.

The independent b18 evaluation reused the same 32 valid development states:

| Training arm, all beam 4 | Best validation NLL | Score vs uniform, 95% game interval |
|---|---:|---:|
| Teacher 8 visits, lr 1e-4 | 1.5863 | +1.11 [−1.58, 3.87] |
| Teacher 64 visits, lr 1e-4 | 1.6088 | +4.54 [1.34, 8.98] |
| Teacher 8 visits, lr 5e-4 | 1.6210 | +3.64 [1.05, 6.78] |
| Teacher 64 visits, lr 5e-4 | 1.7121 | +1.71 [−0.53, 4.22] |
| Expert-only refresh, final 120 | 1.5669 (initial) | +3.00 [1.53, 4.72] |
| Learner-state refresh, final 120 | 1.5669 (initial) | +1.32 [−0.74, 3.22] |
| Previous compact-v2 model | 1.5669 on corrected validation | +4.01 [1.77, 6.63] |

The selected 64-visit/lr1e-4 model plays `Q16` from empty. All six new candidates
made legal opening probes and avoided the old empty-Black PASS. These include
known training states and demonstrate coverage, not held-out generalization.
White can correctly pass to win immediately after Black passes from empty;
the selected model misses that opportunity, choosing `R17` instead. It remains
a weak policy despite fixing the most obvious opening symptom.

Selected minus previous model is only **+0.52 [−1.02, 2.57]** on paired positions,
so no midgame improvement over the previous model is established. Learner
refresh minus the phase-matched control is **−1.69 [−3.78, −0.06]** in this
one-seed pilot. Optional categorical D4 inference changed +0.30 versus uniform
to −1.73; its paired change is −2.03 [−6.04, 1.22], costing about five times
the measured inference duration. Neither refresh nor this D4 ensemble earns a
quality recommendation from these measurements. Training D4 and inference D4
are distinct experiments.

Evidence: `outputs/research/move_quality_v4_b18_beam4_d4_valid32_s3407.json`,
`opening_probe_v4_beam4_s3407.json`, and `v4_selection_and_contrasts.json`.
The teacher-budget effect depends on learning rate in this small study;
64 visits is not shown universally superior. Repeated development selection,
one seed per new arm, and approximate engine values limit these intervals.

## Selected v4 model: completed games remain two losses

The frozen 64-visit/lr1e-4, seed 3407 checkpoint at step 120 played two
from-empty 19×19 games with beam 4, swapping colors against KataGo b6 at
eight visits. Both ended by two passes. There were no caps, forfeits, void
results, or scoring failures; the 1,083-ply cap would have produced Void.

| Model color | Official result | Total plies | Model passes / turns | First model pass |
|---|---|---:|---:|---:|
| Black | W+368.5 | 530 | 11 / 265 | Ply 489 |
| White | B+336.5 | 511 | 11 / 255 | Ply 474 |

The summary is **zero wins and two losses**, with Elo left undefined. The
model now participates from the opening: the previous compact-v2 model
passed on 34/35 Black turns starting at ply 1, while this candidate first
passed at ply 489. This change does **not** establish stronger full-game play.
Both models lost both games, and the new Black loss was larger. Early-pass
losing margins are not a useful substitute for winning or strategic competence.
The model remains very weak against this opponent.

Independent CPU raw-area counting matched both official margins. The Black
game ended with zero Black stones, 262 White stones and 99 White territory.
The White game ended with 254 Black stones, 98 Black territory, eight White
stones and one neutral point. This agreement in these games does not erase
the legitimate raw-area/Benson difference documented below; the diagnostic
never replaced the referee result.

Evidence, complete SGFs, logs, collection-time checkpoint fingerprints and
source snapshots are in
`outputs/research/text_fullgames_v4_v64_lr0.0001_beam4/`, including
`summary.json`, `behavior_audit.json` and `raw_area_audit.json`.

A CPU-only export retained 128 learner pre-action states, 64 per game, from
520 eligible learner turns. It found no D4 duplicates or overlaps with the
validation and frozen final-test states. The training-only file
`data/text_learner_v4_v64_lr0.0001_beam4_source_v1.jsonl` has SHA256
`6da6bb6e3dd45ffdebaf4323a0a98c896c9a1a7ed19f44bd72bb6769e64655b8`.
These are unlabeled learner states with behavior/checkpoint provenance for
future controlled refresh. No teacher queries or training used this export,
and it does not alter the frozen comparison or current quality claims.

## Correctness, efficiency and RL

Board replay caching and bulk legality reduced the representative board benchmark
from roughly 235 ms to about 1 ms. Real KataGo checks agreed on tested masks and
normalized rules; unsupported rule semantics fail explicitly.

The teacher builder previously retained every historical board for every
selected position. On 64 real records, traced allocations fell from 36.75 MB to
7.71 MB with batch size eight. Its memory now scales with replay batch size.
Indexed selection reproduced the exact previous 1,024-row selection SHA while
using 2.53 MB traced peak on the 8,166-row source. Streamed D4 preparation reduced
a 64→512-row fixture from 16.98 MB to 0.50 MB, with identical prepared output.
These measurements exclude model/native allocations; prepared caches trade disk
space for memory. No giant training run was needed to verify the mechanism.

The categorical PPO path collects and optimizes every new model turn, preserves
the actual masked behavior distribution, flips value perspective across players,
and bootstraps caps without inventing wins. A real Qwen 16-ply smoke completed
eight optimizer minibatches in 16 seconds, with finite gradients, approximate
KL .00638 and clip fraction .0625. It capped before termination; this is plumbing
evidence, not a reward-learning or strength result. Seeded late-game continuations
are explicitly distinguished from games starting empty.

A second bounded PPO experiment sampled four genuine continuations from training
positions at plies 283, 341, 302 and 314, with a limit of 128 new model moves each.
All four capped. Replay independently verified all 512 actions under full-history
positional superko, alternating colors, exact seed provenance, and no missed
two-pass ending. There was **one pass in 512 moves**. Captures reopened space:
316 stones were captured, 259 in the second halves of the trajectories; ending
empty points were 89, 33, 30 and 64. Late move number did not ensure closeness to
termination. These trajectories have null outcomes and supply bootstrapped
critic targets, not verified terminal win/loss learning. This uses the weaker
categorical checkpoint and does not update the selected text model. The update
finished in 390.61 seconds (398.29 seconds for the process), optimizing all 512
moves through 128 minibatches with finite gradients. Approximate KL was 0.03454
and clip fraction 0.28516. The KL guard limits additional epochs; this one-epoch
run completed its entire first pass, as configured. These drift diagnostics
provide another reason not to treat a finite loss as a useful policy improvement.

For a future reward-coverage experiment, preregister training-only seed strata
by legal-action count, board occupancy and prior-pass state, then measure the
fraction of **naturally sampled** completions at a fixed budget. Preserve an
empty-board evaluation distribution. Do not manufacture terminal rewards by
forcing passes, search-scoring caps, or discarding failed groups without counting
their compute. The current min-move curriculum alone did not solve this problem.
Evidence: `outputs/research/policy_seeded_terminal_v4_s3407/episodes.jsonl` and
`episode_replay_diagnostic.json`.

For KataGo 1.18.2, genuinely finished area games with `friendlyPassOk=false` use
deterministic terminal scoring, including Benson pass-alive territory. Other
GTP situations can invoke anticipated-score estimates. Analysis `scoreLead` and
caps are never verified outcomes. A separate raw-board flood-fill diagnostic
counts every remaining stone; it can legitimately disagree with KataGo and
never replaces the reward. See the pinned-source audit in the RL notes.

The real-engine regression suite subsequently passed all nine tests. Its frozen
19×19 fixture finishes by two passes and returns `B+353.5` twice through
KataGo's `final_score`, while independently counting every present stone gives
`B+347.5`. This six-point difference is expected from pass-alive territory
scoring. The fixture performs no search or neural-score fallback. It verifies
why the two score conventions must remain separate in the implementation.

Periodic resume checkpoints bind weights, tokenizer, historical best model, optimizer,
scheduler, RNG state, prepared-data hashes and evaluation cadence. Real tiny
Qwen tests verify bit-exact CPU continuation for both trainers with and without
D4. Periodic checkpoints and `final` contain optimizer/RNG state and the
preserved best snapshot; `best` is for inference or a new warm start. Resuming
an already finished schedule does not extend it: use a new run with
`--init-checkpoint` for a different budget. Exact MPS/CUDA numerical equivalence
across machines is not claimed.

## Reproduce and continue

Use the [README](../README.md) and
[`config/policy_19x19_qwen06.yaml`](../config/policy_19x19_qwen06.yaml). Start with
`scripts/train_19x19.py --dry-run --smoke`, then run the bounded smoke. The longer
32,768-position/2,000-update recipe is a transparent starting budget, **not an
experimentally tuned optimum**. Keep teacher budgets, inference compute, data
splits and outcome conventions fixed when comparing candidates.

The larger experiments should prioritize more independent games, a stronger
teacher budget, repeated seeds, completed color-balanced matches, and learner-
visited states with expert replay. Reserve a new untouched final split before the next round, and keep it unused until
choosing the development winner. Promote by independent quality and games,
never merely by syntax, token count or training loss.

## One-time frozen final evaluation

After the development choice and fixed seed3408 repeat were recorded, the
123-position/32-game final set was released once. All four Qwen arms used
nonthinking constrained beam 4, with the same position order and independent
b18 evaluator at a 32-visit cap per root and unique child. The run performed
123 root and 768 unique-child queries (891 total); the measured evaluation
phase took 522.58 seconds after input auditing. All predictions were legal;
all paired comparisons retain every position and game.

| Fixed arm | Estimated score advantage vs uniform | 95% game interval |
|---|---:|---:|
| Selected teacher64/lr1e-4, seed3407 | +1.15 | [+0.03, +2.32] |
| Fixed repeat, seed3408 | +0.58 | [-0.65, +1.92] |
| Previous compact-v2 | +1.04 | [-0.40, +2.67] |
| Base Qwen3-0.6B | -3.82 | [-5.07, -2.66] |
| Training-only coordinate frequency | -1.28 | [-4.24, +0.59] |
| Fresh evaluator-selected move | +7.30 | [+5.54, +9.25] |

The primary paired contrasts are selected minus previous **+0.109
[−1.182, 1.398]**, selected minus base **+4.968 [3.410, 6.833]**, and selected
minus uniform **+1.150 [0.031, 2.317]**. Training improved this diagnostic over
base Qwen, but the new selected recipe does **not** demonstrate a gain over the
previous checkpoint. Its margin over uniform is narrow and does not repeat
as a positive score interval for seed3408. Selected minus the static frequency
control is **+2.430 [−0.066, 5.923]**; superiority to that control is also
inconclusive on this final sample. Repeat minus selected is **−0.570
[−1.995, 0.830]**. Neither seed is promoted using final results.

As secondary engine diagnostics, estimated win-probability differences versus
uniform were +4.51 percentage points [0.83, 8.59] for selected and +3.55
[0.64, 6.63] for the repeat. These are **not measured game win rates** and do
not override the two full-game losses. The reference action remains well ahead
on estimated score. The same b18 engine both chooses that reference and scores
its child, so it is an internal search reference, not independent ground truth.
Evaluator independence here means separate from the b6 training teacher. All intervals use 5,000 paired, equal-game bootstrap draws;
they are pointwise, not adjusted for multiple contrasts, and exclude engine
approximation error and general training-seed uncertainty.

The release manifest binds dataset, protocol, chosen/repeat model files, source
snapshot and development selection hashes. The independent statistical audit
reproduced all 22 score/winrate summaries with a second NumPy implementation.
A schema check initially compared all reported checkpoint files with a
model-only manifest: the repeat's `final` directory additionally contains
`training_state.pt`. The summary now projects out only that known optimizer/RNG
file, records its hash, and still rejects changed, missing or extra model files.
All 22 released repeat model/tokenizer/metadata hashes match; the raw report and
release manifest were not changed. Seventeen CPU checks cover the summary.

Evidence: `outputs/research/final_quality_b18_frozen123_s3407.json`,
`outputs/research/final_quality_contrasts.json`,
`logs/research/final_quality_numerical_audit.json`, and the frozen protocol/source
manifest under `logs/research/`. Standalone figures are
`outputs/research/final_quality_frozen123.png` and `.svg`.

## Decisions for the next larger run

1. Keep complete histories, fixed rules/komi, whole-game/D4 isolation and legal
   decoding. These fix demonstrated correctness and interface failures.
2. Use compact text as the measured starting interface. Grid had similar NLL
   at about 1.7× the local training time. Preserve beam 4 as a measured inference
   option; it helped one checkpoint and hurt another. Raw text legality remains
   too unreliable to remove the constraint.
3. Scale independent source games and audit opening/endgame coverage before
   scaling updates. The selected D4/teacher64 recipe bundles several choices;
   this study does not isolate a universal D4 benefit or prove 64 visits always
   beats eight. Keep teacher-budget/LR comparisons matched and repeat seeds.
4. Keep categorical soft-target learning as an explicit alternative, with
   matched target-source controls. Its lower imitation loss did not establish
   stronger play. Ownership/score auxiliary heads remain unimplemented hypotheses.
5. Measure natural terminal-reward coverage before spending a large budget on
   GRPO/PPO variants. The sampled late-game test produced no outcomes. Consider
   broader expert replay and a preregistered ending-state curriculum, evaluating
   the eventual policy in full games from empty.
6. Treat learner refresh and D4 inference as hypotheses to revisit, not adopted
   improvements: both failed their bounded development comparisons. Use more
   learner games and fixed budgets before drawing broad negative conclusions.

The final evaluation above is a one-time report. Any subsequent optimization round
needs a new untouched final sample, with promotion based on repeated seeds and
color-balanced completed games against a fixed opponent ladder. Do not turn
this small study's estimated score differences into Elo claims.

## Verification and artifact status

The final CPU suite passed **778 tests** in 16.00 seconds. All **nine real
KataGo integration tests** passed in 10.25 seconds, including the terminal
scoring fixture. The Go parser passed `go test -race ./...`; Ruff's fatal/error
checks and `git diff --check` passed. These establish tested contracts, not
playing strength. Logs are `logs/research/cpu_final_20261023_role_guard.log` and
`logs/research/katago_integration_20261023.log`.

The last review closed another leakage boundary: recipe training inputs now
require explicit train/validation roles, and both direct trainers reject
marked final-test rows before loading a model. Generic standalone evaluation
still accepts test data. Twelve new regressions cover the boundary and legacy
compatibility. The completed frozen evaluation used the earlier saved source
snapshot and fixed checkpoint hashes; this input guard changes neither those
weights nor the previously valid training inputs. The fix is commit `5723396`.

An actual Qwen/MPS smoke through `train_19x19.py` completed two updates with
8 training and 4 validation positions in 21.20 seconds. The selected 120-update
pilot is separately exposed in `config/policy_19x19_qwen06_pilot.yaml`.
Documentation checks covered 32 command flag sets, 19 links/anchors, and both
pilot/long-recipe dry-runs. Direct runners use a single model device; the legacy
multi-GPU wrappers are a separate path.

A fixed repeat of the selected training recipe with seed 3408 completed all
120 updates in 245.25 seconds. Final/best completion NLL was 1.710322 versus
1.608759 for seed 3407. Raw legality on eight development states was 5/8, while
constrained legality was 8/8. This repeat's final step was chosen in the protocol
before any final-test queries; the final set must not choose between seeds.

Source changes and the experiment journal are versioned. Large checkpoints,
JSONL datasets, engine networks, source snapshots and per-position reports remain
in ignored local paths. Transfer those artifacts and verify recorded hashes
when reproducing this exact study on another machine.
