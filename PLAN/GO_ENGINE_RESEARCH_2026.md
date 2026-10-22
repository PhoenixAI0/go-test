# Go engine, rules, and evaluation audit — 2026-10-22

## What was measured

Baseline before this work: 84 targeted tests passed, one real-engine match test skipped. The principal CPU bottleneck was replaying the complete game for every position read and every legality query. Snapshot-backed boards also dropped their move history, and KataGo received the latest stones as a fresh setup, losing ko and pass context.

Reproducible benchmark:

```sh
uv run python scripts/benchmark_board.py --board-size 19 --plies 160 --seed 20261022 --repeats 7 --output logs/engine_audit/board_benchmark.json
```

On this macOS arm64 host, Python 3.12.13, the same 201 legal intersections were found by every method. Median times per 361-candidate scan:

| Method | Median milliseconds | Relative speed |
| --- | ---: | ---: |
| Reconstruct full history for each candidate | 234.793 | 1.0x |
| Cached positions, individual legality calls | 1.094 | 214.6x |
| Bulk `BoardState.legal_moves()` | 0.819 | 286.8x |

The original unmodified implementation was separately measured at 234.753 ms/scan (five repetitions), consistent with the replay baseline. These are board-operation speedups, **not model-quality or training-throughput results**. The reproducible benchmark reconstructs uncached boards rather than retaining a second implementation in production.

KataGo v1.18.2 was installed from a prebuilt Homebrew bottle. It uses Metal, so inference shares the GPU with local Qwen tests; do not interpret concurrent runs as clean speed measurements. A 4.9 MB b6c96 network was downloaded to ignored `katago/` for bounded correctness probes. Its strength is not a substitute for a stronger final evaluation network.

* First 19x19 analysis, two visits and 362 policy probabilities: 2.481 s including first startup.
* Warm local engine, nine 19x19 positions, one visit each: 0.355 s including startup. Every legal-action mask agreed with KataGo: plies 0/20/80/160/240/320/360 plus full-history and snapshot versions of a ko position.
* Six rule normalizations were compared against real `kata-get-rules`, with exact agreement, including Chinese, Chinese-OGS, Japanese, AGA, and two compact SGF rule strings.
* Independent `sgfmill` differential tests play 162 plies on 9x9 and 722 on 19x19 and compare every resulting matrix, including captures.

Detailed local results are in ignored `logs/engine_audit/{board_baseline,board_benchmark,real_engine_verification}.json`.

## Changes implemented

1. `BoardState` retains an initial setup plus subsequent move history. Cached matrices are shared safely between branches and copied when returned to callers. Mutation of the public move list invalidates the cache. Previous-state and trailing-pass APIs now work on both ordinary and setup-backed games.
2. Legacy snapshots with a previous matrix reconstruct a verifiable last move/pass for the KataGo request. Inconsistent predecessor context fails explicitly. Full original histories remain preferable: two matrices cannot reconstruct ancient superko.
3. KataGo analysis supports per-query `rules`, `komi`, `max_visits`, `include_policy`, and `include_ownership`, including pool forwarding. Warning-only messages no longer masquerade as completed results. Abandoned batches release request handles and terminate outstanding search. Requests after a reader failure fail promptly.
4. Cache identity includes initial setup and predecessor context. Same visible stones are insufficient for an analysis cache key.
5. Rule normalization preserves exact ko, scoring, tax, suicide, button, handicap compensation, and friendly-pass semantics. Local legality explicitly rejects multi-stone-suicide and situational/Spight ko rather than silently using the wrong mask.
6. A move cap produces `RE[Void]`, never an inferred win. Match summaries exclude void/unknown results from win/loss/draw, report cap rate, and return null strength metrics if no games finished.

## Rule consistency is a training requirement

KataGo's **`chinese` preset uses SIMPLE ko**. `chinese-ogs` and `chinese-kgs` use POSITIONAL ko. The raw string `koPOSITIONALscoreAREAtaxNONEsui0` differs from Chinese-OGS in handicap compensation and friendly-pass settings. Japanese uses simple ko and territory scoring; local main-phase legality does not implement territory-scoring encore phases or adjudication. These details were checked against the installed engine and [KataGo's rule parser](https://github.com/lightvector/KataGo/blob/v1.18.2/cpp/game/rules.cpp).

The [analysis protocol](https://github.com/lightvector/KataGo/blob/v1.18.2/docs/Analysis_Engine.md) recommends original moves over fresh setup stones when known. Its dense `policy` array is row-major from top left, with pass last and negative entries for illegal actions. Values and ownership follow the configured reporting perspective. This client fixes perspective to Black. A player-relative value target must reverse White's sign/probability; an action target must not.

The initial 128-game corpus includes 44 rule strings. Only one of those games uses the exact no-tax, no-suicide, positional-area target string. A game-level validation split cannot be created from one game. Either obtain more matching games or deliberately relabel diverse source positions under one declared target ruleset. In the latter case discard original game outcomes as target values, retain original metadata, recheck full-history legality, and recompute teacher targets. Never quietly rebrand source labels as target-rule truth.

## Papers read and implementation decisions

### AlphaZero — Silver et al., 2017/2018

[Primary paper, methods and learning objective](https://arxiv.org/pdf/1712.01815). The model predicts a policy and outcome value; MCTS supplies improved policy targets, while completed games provide outcome targets. Search-guided self-play and terminal outcomes are distinct training signals. The methods mask illegal actions using rules. The paper also distinguishes AlphaZero from AlphaGo Zero's explicit symmetry augmentation and gating.

Application here: use the inexpensive exact game transition system, expose a compact 362-action policy, separate inferred teacher values from completed-game outcomes, and evaluate whole games. A Qwen backbone with policy/value heads is a testable alternative to serial token generation; this audit does not establish its superiority.

### KataGo — Wu, 2019/2020

[Primary paper, sections 3–4 and appendices](https://arxiv.org/html/1902.10565v5). Playout-cap randomization spends larger search budgets on a subset of turns, retaining those policy samples while generating more games with cheap moves. Forced exploration and policy-target pruning separate discovery from imitation of exploratory noise. Auxiliary ownership, score, and future-policy targets add supervision beyond the noisy binary game result. The ablations, rather than the headline aggregate speedup, justify testing these independently.

Application here: optional dense policy/ownership output and per-query search budgets enable controlled teacher-data experiments. First compare teacher label quality at several visit counts on a fixed validation set. Do not multiply KataGo's reported gains into expected Qwen gains, nor add complex MCTS heuristics before the policy has learned useful priors.

### LoGos — Ma et al., 2026

[Primary paper, sections 2–4](https://arxiv.org/html/2601.16447v1). It mixes Go data and general reasoning for a cold start, then uses GRPO with tiered move-ranking and predicted-winrate rewards. Its board-state-rendering experiment directly addresses degradation when models must reconstruct long game histories. The main Go benchmark tests membership among engine-annotated candidate moves, not full-game win rate. Thus its professional-level framing cannot be treated as evidence that a small language model wins professional games.

Application here: make current board state explicit, retain full history for the verifier, compare encodings/token costs empirically, and report move regret separately from game wins. Test dense/smooth rewards against rank tiers; matching one engine's candidate list is only a proxy. Long thought traces need an equal-compute ablation, especially at 0.6B parameters.

### MAV planning — Schultz et al., ICML 2025

[Primary paper, sections 2–5](https://arxiv.org/html/2412.12119v3). Multi-action-value modeling combines transition, action-value, and best-action tasks. It trains a best-action output partly to resolve saturated win-value ties and compares expectation versus mode for bucketed values. Both external MCTS and internally serialized search improve game performance in its tested domains: chess, Chess960, Connect Four, and Hex, **not Go**. Its Elo calibration is also explicitly not a blitz-time guarantee.

Application here: retain both best-action and distributional targets, and benchmark inference cost rather than only search-node counts. Go has many more legal actions, making per-candidate language-model generation particularly costly. Efficient batch policy/value calls and deterministic transitions should precede internal natural-language search trees.

### Adversarial Go — Wang et al., 2023

[Primary paper, attack methods and rules appendix](https://arxiv.org/html/2211.00241v4). The authors train policies against fixed Go engines and demonstrate that a high win rate against a strong victim can reflect exploitation rather than broad Go strength. Passing/scoring choices and unusual cyclic groups matter; adding far more victim search did not eliminate all failures. The adversary explicitly models the opponent policy during search.

Application here: use an opponent population and varied openings, keep a separate stronger verifier for evaluation, and audit passing/endgame behavior. Label terminal adjudication precisely. Track cap/forfeit/pass rates, score regret, and fixed-opponent wins separately; a single victim score is not sufficient to claim general improvement.

## Next experiments, without long unattended runs

Teacher record schema version 2 corrects the metadata for `teacher_value` to
`value_perspective: side_to_move`. Raw engine fields remain Black-relative, as
declared by `teacher.perspective`. The first generated v1 teacher files contain
correct numerical player-relative values but incorrectly say `black` for this
one field. Those immutable experiment inputs and their recorded hashes are
preserved; consumers of legacy files must use this documented correction rather
than flip White's value a second time. A `prior_mix=1` policy target is the raw
network prior; `response`/top-1 accuracy still use the searched best move. They
are distinct objectives and must be reported separately.

* Train a small fixed-budget policy/value-head baseline and token-output baseline on the same game-disjoint source positions and same teacher. Compare held-out teacher cross-entropy, top-move regret, legal rate, and milliseconds per move.
* Compare current-board encodings at identical context and output budgets. A legal mask changes the policy distribution and must be declared in both training and evaluation.
* Measure teacher agreement and regret at 1/8/32/128 visits. Low-visit b6c96 is useful for debugging; final ranking should use a stronger network/search configuration absent from training labels where feasible.
* Test auxiliary score/ownership losses one at a time. Record throughput and avoid spending more generation tokens on verbose labels than learning signal warrants.
* Run full 19x19 matches only after the policy can legally finish games. Use paired colors, fixed seeds/openings, multiple opponents, honest uncertainty, and a separate count of truncations. Long training and sufficiently powered match tournaments remain user-run jobs.
