# Full-board Go: data, representations, and distillation

Research and local measurements: 22 October 2026. This note separates methods reported in papers from changes implemented here and from untested proposals. No small local run establishes professional-level Go strength.

## Findings that changed the implementation

The previous SFT split randomized positions, not games. On the 128-game real 19×19 corpus below, reproducing that split put **408/408 validation positions in games also present in training**. A game-disjoint split is now enforced before the trainer consumes examples; semantic game IDs survive rotations and annotations. Prepared SFT files pin their split, while RL files contain training games only. Adjacent positions cannot cross the boundary.

The archive is not a fixed-rule Chinese Go dataset. Its 128 sampled games contain **44 raw rule combinations** and komi from **−29.5 to 70**. Dropping `RU` and `KM` gives the same visible board contradictory objectives. The parser now preserves rules, komi, setup stones, full preceding moves, player to move, result, handicap, source file, and semantic game ID. Missing komi stays null. No default is silently fabricated.

The old parallel extractor changed which examples it selected when the worker count changed, because it divided the remaining position budget across a worker-dependent batch. Extraction is now reproducible across worker counts. Whole games are shuffled with a local seed; a per-game budget uniformly samples positions across the complete validated game. This replaces the implicit opening-only curriculum created by taking each game's prefix.

Setup stones are placed simultaneously, rather than played as moves that could capture one another. Non-Go SGFs, invalid numeric metadata, conflicting move properties, unsupported mid-game setup/player changes, and nonalternating moves are rejected. Import remains main-line only. The local Go parser uses simple-ko/no-suicide validation, so games requiring other legality rules may be truncated at the first unsupported move; preserved metadata must still be checked before a verifier or constrained policy uses a record.

## Primary papers read

1. **Wu, Accelerating Self-Play Learning in Go (2019; revised 2020).** Read sections 3–4 and training/input appendices. KataGo separates cheap trajectory generation from expensive policy improvement through randomized search budgets, trains against pruned search distributions rather than raw exploration counts, and adds ownership, score, and opponent-policy supervision. The opponent auxiliary loss has weight 0.15. Application: preserve distributions and value targets when querying the teacher; optionally predict the next opponent move using already available SGF labels. `opponent_response` is now emitted only after that future move validates and is transformed with board symmetries. It is metadata for a future controlled ablation, not yet a trained head. [Paper](https://arxiv.org/html/1902.10565v5)

2. **Silver et al., Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (2017).** Read the policy/value objective, MCTS targets, and symmetry discussion. Training minimizes value error plus cross-entropy to search visit probabilities. The paper distinguishes AlphaGo Zero's eight Go symmetries from AlphaZero's removal of symmetry augmentation for a generic multi-game algorithm. Application: D4 transformations are valid for Go, so `--symmetries 8` now transforms grid, action, setup, and full history consistently, after splitting games. Color exchange is deliberately excluded because komi breaks that symmetry. [Paper](https://arxiv.org/html/1712.01815v1)

3. **Tian and Zhu, Better Computer Go Player with Neural Network and Long-term Prediction (2015).** Read sections 2.1–2.4. Darkforest predicts several subsequent actions from one position using separate outputs; its minibatch sampler initializes games at different phases and applies eight spatial symmetries. This supports cheap auxiliary supervision and mixed game phases, rather than hand-written strategic rewards. Application: per-game uniform position sampling and future-opponent labels are implemented. Whether auxiliary move prediction helps a pretrained Qwen backbone must be measured; its reported convolutional-network improvement does not establish that result. [Paper](https://arxiv.org/html/1511.06410v1)

4. **Ruoss et al., Amortized Planning with Large-Scale Transformers: A Case Study on Chess (2024; initially Grandmaster-Level Chess Without Search).** Read dataset construction, tokenization, predictor targets, ablations, and limitations. The work distills engine action-values into transformers up to 270M parameters, using fixed-position board tokens and 128 value bins. Its dataset contains approximately 530M states and 15B action-value labels. Different months separate train and test games, but opening positions still overlap; the paper reports this explicitly. Application: a 362-action policy head and distributional value target are serious alternatives to autoregressive coordinate text. Model size alone is not the bottleneck, but the annotation scale is far beyond our smoke corpus. Full Go history must remain available to the legality engine. [Paper](https://arxiv.org/html/2402.04494v2)

5. **Schultz et al., Mastering Board Games by External and Internal Planning with Language Models (2024/2025).** Read section 2's state/action-value format, section 3's search, and internal-search construction. MAV keeps board cells at fixed token positions, jointly supervises transitions and action-values, randomizes candidate ordering, and uses 64 value buckets. Its internal reasoning is trained on explicit search operations; external search uses learned evaluations and bounded MCTS. Application: measure fixed-cell or spaced-grid representations against BPE-compressed grids, and distill useful candidate/value information before adding long free-form explanations. A rules engine already exists here, so predicting transitions is optional auxiliary supervision, not a replacement for exact verification. [Paper](https://arxiv.org/html/2412.12119v1)

6. **Zhang et al., Transcendence: Generative Models Can Outperform The Experts That Train Them (2024).** Read theoretical assumptions and experimental sections 4.1–4.2. The experiments train 50M autoregressive chess models on character-level game transcripts and test sampling temperatures. Lower temperatures can suppress inconsistent errors in a diverse imitation dataset, but this is neither guaranteed for every model nor equivalent to improving coverage of unfamiliar states. Their evaluation allows retries before declaring illegal-output losses. Application: compare deterministic decoding with sampled decoding using explicit retry accounting. Never compare a five-retry policy with a one-attempt policy as if their inference budgets matched. Our interface benchmark defaults to one attempt. [Paper](https://arxiv.org/html/2406.11741v3)

7. **Ma et al., Mixing Expert Knowledge: Bring Human Thoughts Back To the Game of Go / LoGos (NeurIPS 2025; arXiv 2026).** Read sections 2–4 and appendices B–C. The recipe uses 10M teacher-annotated positions, multi-candidate/value/variation supervision, mixed reasoning cold start, then GRPO on Qwen2.5 7B/32B. Explicit board matrices improve long-game inputs. Important limits: their next-move benchmark is a proxy; reported professional comparisons do not establish 0.6B scaling or our own playing strength. Their compute and response budgets are much larger. Application: preserve board state and dense teacher information; compare direct policy learning against structured reasoning under equal budgets. Do not infer that longer unverified prose makes a better policy. [Paper](https://arxiv.org/html/2601.16447v1)

8. **Tan and Hong, Self-Supervised On-Policy Distillation for Reasoning Language Models (May 2026).** Read section 3's objective and witness selection. SSOPD uses the shortest correct completion as privileged teacher context and distills at prefixes of the longest incorrect completion, with frontier weighting. It adds local distributional supervision to sparse GRPO outcomes. Application is a hypothesis: query KataGo on states visited by the student's actual mistakes, then distill its action distribution there. Go engine action-values are approximate search estimates; they are not binary mathematical proof certificates. This does not justify labelling every teacher disagreement as an objectively wrong move. [Paper](https://arxiv.org/html/2605.17497v1)

## Local data and representation experiment

Corpus: the repository's pinned [KataGo 2025-01-01 archive](https://katagoarchive.org/kata1/traininggames/2025-01-01sgfs.tar.bz2), SHA256 `98dc06610ccad000ed532fa00c7daf76157d150a1319fb147493c1cf26c5b4ca`. Extracted 128 real 19×19 games; generated files remain under ignored `data/`. This is a small experiment corpus, not a competitive training set.

```bash
uv run python scripts/prepare_datasets.py \
  --sgf-dir data/sgf_19x19 --tag 19x19_small \
  --max-positions 8192 --positions-per-game 64 \
  --num-workers 4 --eval-ratio 0.125
```

The run produced 8,166 records: 7,142 from 112 training games and 1,024 from 16 validation games. All 8,166 replayed from setup/history to exactly the visible matrix and player. There were zero shared game IDs and zero exact visible-board-plus-objective overlaps between these partitions. Those zero counts are corpus-specific, not a guarantee for larger archives. Move-number percentiles were 0/75/152/232/311/411 at 0/25/50/75/95/100 percentiles, demonstrating broad phase coverage.

A 128-position CPU tokenizer probe using `Qwen/Qwen3-0.6B` measured raw input lengths before the chat template:

| Encoding | Mean tokens | Median | 95th percentile | Maximum |
|---|---:|---:|---:|---:|
| Legacy unlabelled grid | 177.3 | 195 | 218 | 228 |
| Spaced grid with context | 556.2 | 556.5 | 566 | 571 |
| Compact grid with context | 344.5 | 362 | 388 | 399 |
| Stone coordinate lists | 537.9 | 576 | 808 | 851 |

The legacy input omits the objective and is not an equally informative representation. Compact inputs are cheaper than spaced grids; spaced grids keep cell locations more regular. Token counts alone cannot choose the better learner. Stone lists are inexpensive in openings but increasingly costly later. The probe predates canonical rule-string rendering, so later prompt lengths can differ slightly. Measurement artifact: `logs/research/data_audit_2026.json`.

## Bounded model experiments

The first standardized teacher files contain 256 training positions from 112 games and 64 validation positions from 16 games, reanalyzed under positional-ko area rules, no suicide/tax, komi 7.5. Labels combine an eight-visit search distribution and dense prior with equal weight. They are approximate estimates, not verified optimal moves. Frozen model revision: `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`.

The untrained model was tested on eight game-balanced validation positions across all four text encodings and unconstrained/constrained decoding: 64 total predictions, one attempt each, a 16-token completion budget. Every unconstrained arm produced zero legal moves and truncated all eight responses. Every constrained arm produced eight legal moves but zero teacher-move matches. Compact constrained prediction averaged 0.261 seconds versus 0.301 for grid and 0.302 for stone lists. None of the selected moves had engine search coverage in the saved eight-visit analyses, so their regret is **unknown**, not zero. Artifact: `logs/research/interfaces_qwen3_0_6b_base_v1.json`.

The compact text baseline trained a rank-16/alpha-32 LoRA for 60 steps, batch size 4, learning rate 1e-4, on the same teacher files. It projected only completion hidden states into Qwen's large vocabulary. A tiny-Qwen regression test compares both loss and every parameter gradient against ordinary full-logit causal training, including unequal answer lengths. This saves vocabulary projection work without changing the objective; inputs and the full target including EOS are never truncated.

| Text metric | Before | After 60 steps |
|---|---:|---:|
| Validation completion-token NLL, 64 positions | 5.78624 | 1.70420 |
| Validation whole-move plus EOS NLL | 20.25184 | 5.96469 |
| Raw legal moves, 8 positions | 0/8 | 4/8 |
| Raw truncated responses | 8/8 | 0/8 |
| Constrained legal moves | 8/8 | 8/8 |
| Teacher top-move matches, either decoding arm | 0/8 | 0/8 |

The run took 158 seconds on the local MPS device, approximately 1.6 seconds per training step. It demonstrates format learning, not Go strength. The eight-position generation subset is too small for ranking close models, and completion-token NLL cannot be directly compared with a 362-action soft-target cross-entropy. Full artifacts and loadable adapters: `outputs/text_distill_compact_v1/`; runnable entry point: `scripts/run_text_distill.py`.

An important inexpensive control: uniform probabilities over legal actions achieve **5.20491** validation soft-target cross-entropy. Frequencies fitted only to training teacher distributions achieve **5.38296**, with 0/64 top-one and top-five matches. The initial compact dense policy's 5.2943 loss after 60 steps is still worse than uniform on this measure. Uniform randomized predictions have expected top-one accuracy 0.00590 and top-five accuracy 0.02948 across these varying legal sets. These controls are implemented in `go_rl/experiment_utils.py`; they prevent reporting optimization progress alone as game skill. Artifact: `logs/research/policy_baselines_v1.json`.

The teacher exporter's initial maximum move number of 280 excludes **847/7,142 (11.9%)** training positions and **163/232 (70.3%)** source pass labels. Validation loses 112/1,024 positions and 25/33 pass labels. The small teacher training set has just one pass target. A model intended to finish full games needs later phases and passing decisions; remove the upper cutoff and compare a game-balanced, phase-balanced sampler against uniform positions. Source pass counts diagnose coverage only: a pass in the original source objective is not automatically optimal after changing komi or rules.

## Untouched final test

`scripts/prepare_final_test.py` freezes the next unused archive games in archive order, excluding both original SGF content hashes and D4-invariant canonical game IDs. The first frozen set is **32 games / 128 positions**, every row pinned to `split: test`. No engine/model evaluation, teacher labels, or label-quality selection was performed. Files: `data/final_test_19x19_v1.jsonl` and `data/final_test_19x19_v1.manifest.json`; dataset SHA256 `5e361c77bed11ccc0ca14b50bd4acdbe2a8932634ee9537851c2a1f77cbeea2b`. Freeze architecture, hyperparameters and checkpoint choice before annotating or evaluating this set. Game separation does not guarantee no repeated opening states; final auditing should report any state overlap without tuning on test labels.

```bash
uv run python scripts/prepare_final_test.py \
  --exclude-dataset data/sft_19x19_small.jsonl \
  --exclude-source-manifest data/sgf_19x19/katago_2025_01_01/source.json \
  --output data/final_test_19x19_v1.jsonl
```

## What to run next

1. Reanalyze mixed-source positions with an explicit target ruleset and komi. Preserve `source_*` metadata, replace the move with the teacher target, and remove the source outcome/future-move labels from target supervision. Reject histories that are illegal under the target rules. This is relabelling a state distribution, not pretending all source games used Chinese rules.
2. Compare Qwen 0.6B autoregressive move text with a finite-action head on identical game splits and equal optimizer/teacher budgets. First compare one-hot versus soft policy supervision. Add value or opponent-action auxiliary losses separately so their contributions are measurable.
3. Compare compact/grid/fixed-cell inputs by held-out teacher loss, legal output rate, evaluated-move regret and coverage, and complete-game win rate. Preserve per-position outputs. A benchmark harness is provided at `scripts/benchmark_interfaces.py`; it balances games, filters unsupported rules explicitly, records source checksums/model revisions, and never assigns invented regret to unsearched moves.
4. Refresh part of the training set with student's visited states after the first useful policy emerges. That addresses distribution shift more directly than repeatedly fitting static expert openings. Annotate diverse games and phases before spending many more visits on already easy examples.
5. Treat 9×9 as a debugging and curriculum ablation, not evidence of 19×19 quality. Keep a persistent 19×19 holdout from the beginning; separately report late-game positions and complete-game termination.

## Implemented interfaces and remaining limits

`board_from_example` rebuilds setup and move history and checks it against the visible prompt. `state_record_from_board` replaces stale state metadata when rollouts change the starting board. `transform_example` applies D4 to raw examples; `transform_teacher_example` additionally permutes complete policy distributions, masks, ownership and engine move/PV labels, preserving pass and scalar values. Split first; every augmentation retains its source game ID and partition. `game_split_indices` and `split_dataset_by_game` enforce whole-game partitions. Modern SFT input rendering is selected with `input_encoding: grid`, `compact`, or `stones`; it requires explicit rules and komi. `legacy` remains an explicit baseline.

Prepared `sft_<tag>.jsonl` contains both pinned train and validation records for `run_sft.py`; `sft_<tag>_train.jsonl` and `sft_<tag>_validation.jsonl` are also written. `rl_prompts_<tag>.jsonl` contains only training records. A manifest records the split seed, game IDs, objectives and counts. Regenerate old position-only data before enabling SFT validation; silently falling back to row splits would invalidate the comparison.

Full histories increase JSONL size. Current preparation streams records and holds game IDs plus a bounded parser batch, but per-game histories still cost quadratic bytes in game length. For millions of games, use sharded columnar state/teacher records plus a shared per-game history table or packed fixed-cell inputs. Dataset scaling and strength claims need separate larger runs; they are not established by parser tests.

## Broader training sample and learner refresh

The CPU-selected `data/teacher_19x19_train_v2_source.jsonl` contains 1,024 positions from the original 112 training games (96 games contribute nine positions, 16 contribute ten). Fixed move-phase counts are 291 at moves 0–60, 277 at 61–180, 258 at 181–280, and 198 at 281+. Selection uses `select_phase_balanced(..., seed=3407, per_game=16)`, with minimum move ten and no upper cutoff. All 1,024 histories replay legally under the target positional rules; no game IDs or D4-equivalent visible states overlap the original validation teacher set after normalizing the target objective. Its SHA256 is `22d2b14c5298f69e7ec7d092e311fe903a643d7b6baf32e1e8702d3ea4df946a`. The 58 source pass actions describe source coverage, not new teacher labels. The frozen final test was not inspected or modified for this expansion.

`scripts/export_learner_positions.py` reconstructs every pre-action state from the categorical PPO runner's `episodes.jsonl` and `rl_run.json`. It requires complete legal alternating histories, explicit rules/komi, and a consistent termination boundary. It exports both players' states, pins every row to training, and records run checksums, seed, update, episode, ply, sampled action and starting-checkpoint fingerprints. Fingerprints observed after the run are marked as observations at export time; an unavailable original checkpoint is reported rather than assigned a fictional digest. The sampled move is stored as `learner_move`, never as an expert `response`. Capped games have no source outcome, and finished results remain source provenance until teacher reanalysis.

```bash
uv run python scripts/export_learner_positions.py \
  --run-dir outputs/research/policy_ppo_smoke_s3407 \
  --exclude-dataset data/teacher_19x19_val_v1.jsonl \
  --output data/learner_refresh_source.jsonl

uv run python scripts/build_teacher_dataset.py \
  --dataset data/learner_refresh_source.jsonl \
  --output data/learner_refresh_teacher.jsonl \
  --split train --min-move 0 --max-positions 256 --positions-per-game 256 \
  --max-visits 8 --prior-mix 0.5 \
  --katago-model "$KATAGO_MODEL" --katago-config katago/analysis_smoke.cfg
```

The exporter deduplicates visible states up to D4, side to play, rules and komi, retaining the first complete history. This conservatively collapses different histories with the same visible board; it is a diversity filter rather than a claim that their Markov states are identical. Optional validation exclusions inspect state fields only, never move quality. Run split validation again after any change of annotation objective. The existing 16-ply PPO smoke trajectory produced 16 distinct replayable positions with no validation exclusions; it does not establish that such a small refresh improves learning.

Both categorical and text distillation runners support `--symmetries 8` after split validation. Teacher policies, legal masks, ownership and move/PV labels rotate together; pass, komi, scalar values and game IDs remain invariant. Validation and generation subsets remain unaugmented. Pinned validation/test rows are now rejected if passed as training data, and pinned training rows are rejected as evaluation data; unpinned legacy records remain supported with game/state isolation checks.

For the text policy, collect its own states rather than using categorical-policy trajectories as if they came from the text model. `scripts/export_learner_sgf.py` selects pre-action positions where the exact SGF `PB` or `PW` name equals `--learner-name`. Opponent moves remain in the complete replay history, but their turns are not learner queries. It preserves both player names, SGF checksums, the recorded result and the optional checkpoint fingerprint. A `Void` result has no outcome; a move-cap comment paired with a claimed win is rejected. Source scores remain provenance and never become teacher targets merely because a result string parses.

```bash
uv run python scripts/export_learner_sgf.py \
  --sgf-dir outputs/research/text_fullgames_v1/sgf \
  --learner-name Qwen06BTextSFTv1 \
  --checkpoint outputs/text_distill_compact_v1/best \
  --exclude-dataset data/teacher_19x19_val_v1.jsonl \
  --output data/text_refresh_source_v1.jsonl
```

Both refresh exporters produce state datasets for expert reanalysis, not an exact off-policy importance-sampling dataset: persisted categorical episode summaries do not contain every action's behavior log probability, and match SGFs do not record the full model sampling distribution. The original checkpoint is observed at export time unless the run saved stronger collection-time provenance. Do not infer retrospective importance weights from a later checkpoint.

## Seeded endgame RL experiment

`run_policy_rl.py --seed-data ... --min-seed-move 250` changes the starting distribution from empty boards to held-in training positions. Candidate rows must explicitly be pinned to training, contain their complete history, match the model's board size, and use the checkpoint's exact normalized rules and komi. Unknown or changed objectives and corrupted histories fail validation. Initial setup stones are currently unsupported by this replay path; such candidates and histories that have already crossed a two-pass boundary are excluded with counts. Sampling first chooses a source game uniformly, then one of its eligible positions, so long games do not automatically dominate.

The original v2 teacher file provides **165 eligible positions from 70 games** at minimum move 250. Of 1,024 total rows, 754 precede the minimum, 62 late rows have initial setup stones, and 43 late rows contain an earlier two-pass boundary. These counts concern a frozen experimental dataset; its bytes were not changed. The source continuation/encore histories explain why a late-game cutoff alone is insufficient for choosing seeds under a two-pass termination convention.

The collector recreates the seed from an empty board and sends every prior move to the GTP referee after setting rules and komi. Its existing pass count and repetition history remain intact. Only newly sampled model actions receive policy gradients; both players' actions are retained. An existing single pass plus a sampled pass can terminate naturally, while a capped continuation still has no scored outcome. Episode summaries store `start_position`, `starting_plies`, the source game/position identity, and the newly sampled move list separately. The learner exporter reconstructs that prefix instead of treating a seeded trajectory as an empty-board game. New RL runs also persist starting-checkpoint fingerprints at collection time; exports prefer those over later observations.

This is a seeded 19×19 endgame learning experiment, not a complete-game strength evaluation from empty. It may provide terminal-reward observations within a smaller budget, but cannot establish opening or full-game quality. No engine selects the model's continuation moves, no passing action is forced, and no winner is manufactured if the rollout reaches its cap.

## Teacher-selection memory audit

Teacher annotation now indexes the source JSONL, filters it during a streaming pass, and keeps selected row indices rather than decoded histories. The phase sampler retains only the lowest-hash `min(max_positions, positions_per_game)` candidates per game and phase, which is sufficient to reproduce its original round-robin selection exactly. The legacy game sampler preserves its original Python RNG shuffle using compact integer arrays. The selected view loads records on demand, and board reconstruction is limited to the current annotation batch. File reads use context managers; no source file handle or parsed-row cache persists between indexed lookups.

On the real 8,166-row starter file, the source index uses **130,656 bytes** and phase selection peaked at **2,532,329 traced Python bytes**. Selecting 1,024 rows and hashing their serialized output took 3.52 seconds on this machine. The resulting SHA256 exactly matches the earlier v2 source selection (`22d2b14c5298f69e7ec7d092e311fe903a643d7b6baf32e1e8702d3ea4df946a`), so this optimization does not change that experiment's data. A separate synthetic 1,000-row corpus with 32 KiB prompts selects under 2 MB rather than retaining over 32 MB of row contents. Tests compare bounded reservoirs against the original complete-sort specification across seeds, budgets, missing/exhausted phases, and duplicate rows.

These are bounded-memory improvements, not constant-memory claims: file offsets grow with row count and phase-reservoir metadata grows with the number of games and configured per-game budget. The measurement uses `tracemalloc`, not total native-process RSS. Artifact: `logs/research/teacher_sampling_memory_2026.json`.

## Two-pass boundary audit and fixed validation subset

Replaying a history legally is insufficient to establish that it is an active state under the target game's termination convention. The frozen teacher files contain **5/256** histories in `train_v1`, **78/1,024** in `train_v2`, and **1/64** in `val_v1` that reach or continue after two consecutive passes. Original territory-scoring source games can legitimately contain encore or cleanup continuations; those states cannot be treated as active positions in the target area-scoring game that ends at the first two-pass boundary. The original datasets and reported runs remain immutable. Future teacher builds reject these histories before querying the engine, and seeded rollouts already reject them before collection.

Both existing 32-position B18 move-quality reports selected the affected validation row. Their original artifacts must remain intact; any post-hoc 31-position diagnostic needs an explicit exclusion record and cannot be presented as a prospectively specified new test. The independent RL review is recording those corrected aggregations. This issue also means the earlier claim that all v2 histories replay legally does not imply all v2 rows represent valid decision points in the target full game.

`data/teacher_19x19_val_v3.jsonl` freezes the **63 remaining positions from the same 16 validation games**, retaining every field and existing teacher label for each selected record. Its SHA256 is `e367af48c3c967d4318c7a586e9d715501acff4ecf5ef73ee93fcfaedf8f7b62`. No engine/model was queried, labels/objective were not changed, and no dropped row was replaced. This is a corrected subset of the existing validation corpus, not a fresh independent holdout. Its manifest records the old dataset/manifest hashes, retained source row indices, rejected row, and unchanged-label declaration. Future v3 comparisons should use this common 63-position validation set.

```bash
uv run python scripts/filter_terminal_positions.py \
  --dataset data/teacher_19x19_val_v1.jsonl \
  --source-manifest data/teacher_19x19_val_v1.manifest.json \
  --output data/teacher_19x19_val_v3.jsonl --split validation
```

## Frozen final-test target objective

State-only preparation of the previously frozen 128-position / 32-game source produced `data/final_test_19x19_target_v1.jsonl`: **123 positions from all 32 games**, target rules `koPOSITIONALscoreAREAtaxNONEsui0`, komi 7.5. Five source histories crossed the two-pass boundary; no other history was incompatible with the target rules. Selection retains original source order, without replacement or resampling and without consulting source move labels or outcomes. Original rules, komi, response, opponent response and result remain under `source_*`; there are no active move/outcome targets and `teacher_labeled` is false. Source outcomes do not apply to the new target objective.

The original source remains byte-identical with SHA256 `5e361c77bed11ccc0ca14b50bd4acdbe2a8932634ee9537851c2a1f77cbeea2b`. Target SHA256 is `b2cc7402fb88bbb1595323e884854a2e21f99892c9a5a00e4db10375f6486070`. The target manifest includes the complete original source manifest, both source hashes, every retained source index, rejection categories and detailed rejected records. A CPU-only audit found no shared game IDs or D4-equivalent visible states/objectives with `teacher_19x19_train_v2.jsonl`. No engine/model evaluation or annotation has been performed on either final-test file. Architecture, hyperparameters and checkpoint choice must still be fixed before evaluating this target set; `eval_move_quality.py` can consume its state records without teacher labels.

```bash
uv run python scripts/prepare_eval_objective.py \
  --dataset data/final_test_19x19_v1.jsonl \
  --source-manifest data/final_test_19x19_v1.manifest.json \
  --output data/final_test_19x19_target_v1.jsonl
```

The preparation tests verify source immutability and manifest hashes, exact retention of existing validation labels, full-history and two-pass filtering, target-rule rejection provenance, pinned splits, and invariance of selection to changed source move/result labels. Eight CPU tests passed. Independent review of the indexed JSONL, streamed distillation, bounded teacher batches and exact-resume integrations passed a further 44 focused tests. Prepared caches publish atomically; teacher datasets publish only after all accepted batches finish successfully, while failed runs leave an identifiable temporary file. Offset metadata scales with row count; prepared token caches still require proportional disk space and D4 preprocessing work.

## D4 inference compute allocation hypothesis

`go_rl/policy_inference.py` optionally evaluates all eight rotations/reflections in one padded batch per group of source positions. It computes legality from the complete original history, permutes the mask with each board, applies a separate masked softmax to every orientation, maps all probabilities back into the original action frame, and takes their arithmetic mean. Pass stays at index 361 on a 19×19 board. Values retain the side-to-move perspective and are averaged without changing sign. No engine supplies moves, and teacher labels are excluded from transformed inference inputs.

`eval_move_quality.py --policy-symmetries 1 8` exposes separate `/sym1` and `/sym8` model arms on the same selected positions. The ensemble's action probabilities define a different inference policy; they are not substituted into PPO rollout log probabilities. Every source position costs eight transformer sequences in the ensemble arm. Reports retain total orientation token counts, padded forward batch size, forward count and measured forward time, so extra inference compute can be compared with autoregressive beam decoding. This is a testable compute-allocation hypothesis, not a demonstrated quality improvement.

CPU regressions independently invert action geometry with NumPy on both 3×3 and 19×19 positions, verify every D4-transformed probability distribution and invariant averaged values, preserve exact zeros for illegal actions, and keep pass unchanged. A ko-capture setup checks transformed masks against fresh full-history legality. A real local tiny Qwen3 model's batch-of-eight output matches eight separate forwards after inverse mapping. Inputs, source labels, parameters and eval mode remain unchanged; training mode, invalid output values, context overflow, mismatched board sizes and incompatible cell-readout encodings fail explicitly.

## Opening coverage after the full-game failure

The first compact-v2 beam-four game exposed an opening-distribution gap: the learner passed on 34 of its 35 Black turns, while the original teacher source sampler started at move ten. This is evidence for an opening-coverage experiment, not evidence that adding opening examples will solve full-game play. `scripts/prepare_opening_budget.py` freezes a new **512-position / 112-training-game** source at `data/teacher_budget_source_v4.jsonl`. Its quota is eight real empty-board move-zero states from eight different source games, 56 states at moves 1–20, and 448 states after move 20, with the latter balanced across games and phases. There are **23 positions at moves 0–5 and 41 at moves 6–20**. All positions come from existing training games; none were fabricated or selected by teacher/model outputs.

Exactly repeated `(game_id, move_number)` rows are removed. Identical opening states from different real training games remain, preserving an explicit training-frequency quota. Target-normalized D4 state and game isolation checks pass against the 63-position validation subset and 123-position final-test target. All histories are legal under the target rules and precede the first two-pass boundary. Raw source fields remain intact for subsequent teacher annotation. The new source SHA256 is `125307c7d2c68d89934f6644a467d209b501d5029d828eda61ce85c036f19b99`; the earlier v3 source is unchanged. Its manifest records source hashes, exact selected game/move pairs, eligible counts, exclusions and the coverage quota.

## Controlled learner-state refresh comparison

The learner SGF exporter now accepts optional `--max-positions`, `--per-game` and `--seed`. Game-balanced sampling occurs after D4 state deduplication and held-out exclusions. Omitting the budget flags retains all eligible positions in source order. Manifests separately record input, duplicate, held-out-excluded, eligible, budget-excluded and selected counts. A real-SGF regression verifies two-game balancing, duplicate/exclusion order, reproducibility, full-history replay and unlabelled training provenance.

The two compact-v2 beam-four matches produced 99 exported learner states: 35 from its Black game and 64 from its White game after the per-game cap. The initial v1 design below is **superseded and unused**; its frozen files remain intact. `scripts/prepare_refresh_ablation.py` prepared matched **256-position** text-imitation arms using the expert-v4 and learner-state teacher files, both labelled by the same B6 network/config at 64 visits and the same target objective:

- `data/refresh_ablation_v1/expert_control.jsonl`: 256 game-balanced expert positions, SHA256 `4501a83b626842277eceb351b255487297413a53434c1517bf55d8279f74b295`.
- `data/refresh_ablation_v1/learner_mix.jsonl`: the first 192 entries of that same expert ordering plus 64 learner positions, exactly 32 from each match; SHA256 `afffe4f7471820a4aa6261be5c04eb1c7ef792cc4e1ab86fcb9d7df18bdbf349`.

Both arms warm-start `outputs/research/text_compact_v2_d4_s3407/best` and use 120 optimizer updates, batch four, D4 augmentation, learning rate `1e-4`, seed 3407, and the same 63-position validation set. Exact argument arrays are saved in `data/refresh_ablation_v1/training_commands.json`; preparation does not launch training. Checkpoint hashes match the checkpoint observed when the two learner games were collected. Both complete source datasets pass game and D4 state isolation against validation and final test. If fewer than 64 distinct learner positions were available, the builder would retain all and fill the remainder with the same expert ordering, keeping the total budget fixed.

The mixture deliberately changes game weights: expert control has 112 games and an inverse-Herfindahl effective game count of 107.79, while the mixture has 114 games but effective count 27.31 because each learner game contributes 32/256 positions. Both selected arms contain five empty-board rows and four repeated visible states. Teacher pass targets number two in the control and six in the mixture; these are post-selection descriptions, not selection criteria. The manifest records these counts and move-phase distributions. `warm_start_overlap.json` separately audits visible-state reuse from the prior v2 training set without using that audit for selection.

This is one offline teacher-response cross-entropy refresh on learner-visited states, analogous to a bounded dataset-aggregation iteration. It is not an on-policy policy-gradient update, and its two learner games are not an independent strength sample. Controlled quality evaluation and complete-game testing are still required to establish any benefit.

The active **v2 design** tightens the control with `--phase-match-control`. Both arms retain the same 192 shared expert records. The mixture's 64 learner records remain byte-identical; the control replaces them with 64 distinct expert records matching the learner quotas exactly: one at move zero, ten at moves 1–20, 23 at 21–60, eight at 61–180, seven at 181–280, and 15 at 281+. No shortage, fallback or repeated exact row was needed. Both complete arms consequently have the same six phase counts: **5, 26, 72, 60, 52, 41**. This controls opening and broad phase coverage while preserving the intended learner-state distribution difference. Game concentration still differs: effective game counts are 88.32 for control and 27.31 for mixture.

Use `data/refresh_ablation_v2/training_commands.json` for both runs, with the same previously specified training budgets/output names. The v2 expert-control SHA256 is `14175a3e8197962d4d756219f412b9b41ec6e88c4780205e2af7943d19433700`; mixture SHA256 remains `afffe4f7471820a4aa6261be5c04eb1c7ef792cc4e1ab86fcb9d7df18bdbf349`. The manifest records requested/selected replacement bins, zero fallback, both source hashes, teacher configuration, warm-start fingerprints and game-weight distributions. Six CPU tests cover exact quotas, distinct replacements, deterministic selection, explicit shortage fallback, matched total budgets and source/holdout guards.

## Deferred ownership auxiliary hypothesis

KataGo's [self-play learning paper, §4.1](https://arxiv.org/html/1902.10565) reports that ownership and score prediction improve learning efficiency by supplying more localized feedback than a binary outcome alone. A deferred experiment would add a 361-point ownership head and a score target to the categorical model, comparing against identical data, training budget and policy/value losses. Teacher ownership estimates must retain their approximate-label provenance and target rules/komi; reliable terminal ownership labels require completed, validated games, never capped trajectories. Existing D4 target transforms already support ownership vectors. This head and loss are **not implemented or tested here**; its benefit for a small Qwen backbone remains a hypothesis rather than a transferred result from KataGo.

## Training-to-GTP integration review

Independent review found two deployment mismatches. Text GTP startup previously ignored a local adapter's saved encoding; a grid/stones checkpoint would receive compact observations unless its launch command repeated the encoding. The server now reads `go_text.json`, matching the move-quality evaluator. Categorical GTP startup now rejects text encodings for cell-based readout heads before any forward pass. Completed compact-v2 matches explicitly supplied compact encoding and target rules/komi, so their results are unaffected.

New text-distillation runs validate one common board size, normalized rules and finite komi across training and evaluation, then persist that context alongside encoding. Existing checkpoint artifacts remain unchanged. GTP defaults use any saved context; explicit CLI options override startup defaults, and subsequent GTP match commands govern the actual game. Existing older text adapters lack these context fields and still require explicit rules/komi when their training objective differs from the server fallback. Warm-start initialization continues to permit its documented new-objective transfer; exact resume retains dataset/contract checks.

CPU tests replay a nine-ply capture/ko position through GTP and compare the exact inference token prefix with the prepared training prefix for grid, compact and stones encodings. They check player, komi, recent history, ko exclusion and move-plus-EOS completion. Existing real tiny-Qwen loss/gradient, beam/EOS and exact-resume tests pass; metadata persists through the real save/resume path. The seeded categorical collector review found consistent sampled behavior probabilities, alternating-perspective GAE, terminal masking and cap bootstraps. No GPU experiment was used for this review.

The opening-quota helper now requires every exclusion record to be explicitly validation/test and use the fixed target objective. Previously, passing a raw SGF-rule holdout could produce a different identity hash from the target-normalized candidates and miss a visible-state overlap. Current v4 heldouts already meet the guard; its frozen dataset and manifest were not regenerated.
