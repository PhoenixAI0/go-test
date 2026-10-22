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
quality hypotheses still need larger runs. Long training is left to the user.

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
It is not a new independent holdout. The untouched final target has 123 positions
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

The next controlled source therefore contains 512 states from 112 training
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
D4. The `best` and `final` inference directories alone do not contain optimizer
or RNG state. Exact MPS/CUDA numerical equivalence across machines is not claimed.

## Reproduce and continue

Use the [README](../README.md) and
[`config/policy_19x19_qwen06.yaml`](../config/policy_19x19_qwen06.yaml). Start with
`scripts/train_19x19.py --dry-run --smoke`, then run the bounded smoke. The longer
32,768-position/2,000-update recipe is a transparent starting budget, **not an
experimentally tuned optimum**. Keep teacher budgets, inference compute, data
splits and outcome conventions fixed when comparing candidates.

The larger experiments should prioritize more independent games, a stronger
teacher budget, repeated seeds, completed color-balanced matches, and learner-
visited states with expert replay. Preserve the untouched final split until
choosing the development winner. Promote by independent quality and games,
never merely by syntax, token count or training loss.

Further bounded experiment results are appended below as they complete.
