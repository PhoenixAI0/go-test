# RL research and objective audit — 22 October 2026

## Decision

Prioritize a reliable 19×19 learning problem, strong data, and useful gradient per unit compute. Optimizer variants cannot compensate for incomplete state, an invalid action interface, mislabeled terminal outcomes, or the absence of rewarding trajectories. Start Qwen's 0.6B checkpoint with compact observations and measurable legal action learning; compare a categorical 362-action policy with coordinate-text generation. Use KataGo as an explicitly identified teacher and value estimator. Measure actual finished games separately from teacher agreement and estimated move quality.

The initial architectural hypothesis was a shared Qwen backbone with a legal-action policy head and a value/action-value head. This retains the chosen model while eliminating most language decoding overhead. Its promise is an inference from the finite action structure and the research below, not an already demonstrated playing-strength gain. Long textual reasoning should earn its inference cost in a controlled experiment rather than be assumed useful. The subsequent bounded experiments below favored constrained coordinate-text distillation over the tested categorical heads; the categorical path remains useful for efficient, correctly specified trajectory RL, not the demonstrated strength winner.

## Reading record: primary sources and actual sections inspected

These entries represent method-section reading, not title/abstract collection. They do not claim exhaustive reading of every appendix. Reported math/model results are evidence for hypotheses, not evidence of transfer to 0.6B Go. Versioned links identify what was inspected. Entries 1–12 are papers; entry 13 is a primary technical report/blog and is not counted as a peer-reviewed paper.

1. **DeepSeekMath** (2024), §§4.1.1–4.1.4, 4.2. GRPO replaces a learned critic with rewards from multiple completions of the same prompt; outcome advantages apply to **all generated response tokens**. The paper includes tokenwise likelihood ratios, per-response normalization, and a reference-policy KL term. Practical takeaway: define the complete action and behavior policy precisely; reward parsing must not silently remove credit for sampled reasoning. [Primary text](https://arxiv.org/html/2402.03300v3#S4).

2. **DeepSeek-R1** (2025), §§2.2.1–2.2.3 and introduction's distillation comparison. Rule-based accuracy/format rewards support RL without prescribing reasoning content. The paper separately studies cold-start data and distilling a stronger model into smaller students. Practical takeaway: a tiny Go model needs a useful initial distribution; direct zero-RL is an experimental arm, not a principle that forbids expert data. KataGo estimates are neural feedback, not the deterministic outcome verifier used in the central R1 examples. [Primary text](https://arxiv.org/html/2501.12948v1#S2).

3. **Understanding R1-Zero-Like Training / Dr. GRPO** (2025), §§2.1–2.2, 3.1–3.4. Removing per-response length normalization avoids asymmetric weighting of long negative and short positive samples. Removing within-prompt standard-deviation division avoids reweighting questions by sampled reward variance. Prompt/template choice also changes apparent starting capability. Practical takeaway: test fixed-denominator token loss and unscaled centered rewards; do not interpret longer reasoning or large recovery from a poor template as new Go capability. [Primary text](https://arxiv.org/html/2503.20783v2#S3).

4. **DAPO** (2025), §§2.2–2.4, 3.1–3.4. The recipe combines asymmetric clipping, retaining informative reward groups, global token normalization, and handling overlong completions. Dynamic sampling changes sampling cost and prompt selection; it is not synonymous with the DAPO loss reduction. Practical takeaway: expose clip bounds and reduction, log zero-variance groups, and give any resampling a firm budget. Token-limit truncation and a game move-limit truncation are different events. [Primary text](https://arxiv.org/html/2503.14476v2#S3).

5. **Group Sequence Policy Optimization / GSPO** (2025), §§3–4.3. Its ratio is the exponential of the **mean** token log-ratio, with sequence-level clipping; this is not an unnormalized product ratio. Its clipping scale differs from tokenwise PPO, and its paper particularly emphasizes long responses and MoE stability. Practical takeaway: sequence ratios are a controlled option, not a reason to blindly reuse PPO's 0.2 bounds or presume an advantage for two-token coordinates. The broad claim that a single-sample importance estimator is inherently invalid should not be treated as a general mathematical theorem. [Primary text](https://arxiv.org/html/2507.18071v2#S4).

6. **MiniMax-M1 / CISPO** (2025), §§2.1–2.2, 3.1–3.2. CISPO clips a **detached importance weight** multiplying advantage and log probability, retaining gradients for tokens that the clipped PPO surrogate would suppress. The report also identifies rollout/training numerical mismatch. Its many-update, long-response setting differs substantially from one-step updates on coordinate answers. Practical takeaway: do not implement CISPO by merely increasing PPO clipping, and do not expect its reported gains without an actual local ablation. [Primary text](https://arxiv.org/html/2506.13585v1#S3).

7. **Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?** (2025), §§2.1–2.2 and experimental framing in §1. The paper compares low- and high-sample coverage, observing improved low-sample accuracy but reduced high-sample coverage in its evaluated RLVR settings. Practical takeaway: track both greedy/best single policy quality and useful-action coverage among sampled moves. A finite pass@k comparison is a measurement of those models and budgets, not a proof that RL can never discover new strategies. [Primary text](https://arxiv.org/html/2504.13837v2#S2).

8. **ProRL** (2025), §§2.1–2.3.1, 3.1–3.2. This supplies counterevidence to universal ceiling claims: prolonged training, broad verifiable tasks, exploration-preserving choices, KL regularization, and reference resets support continued gains in the studied models. The authors distinguish their initialized reasoning checkpoint from base-model zero-RL. Practical takeaway: retain optional KL and measure entropy/coverage over time. Do not hardcode periodic reference resets into a tiny smoke run or equate two thousand updates with a universally sufficient budget. [Primary text](https://arxiv.org/html/2505.24864v1#S2).

9. **Beyond the 80/20 Rule** (2025), §§3–5.3, especially §5.1 and Table 2. Policy-gradient updates restricted to high-entropy positions improve large-Qwen reasoning in its experiments, with smaller gains at 8B. Selection is based on entropy of the vocabulary distribution at a position, not the surprise of the sampled token. Practical takeaway: expose the existing TRL entropy-quantile option, leave it at 1.0 initially, and do not delete 80% of a two-token move or mistake final-only masking for entropy selection. [Primary text](https://arxiv.org/html/2506.01939v2#S5).

10. **Learning to Reason under Off-Policy Guidance / LUFFY** (2025), §§2–3.1 and method framing in §1. Mixing teacher demonstrations with learner samples changes advantage construction and introduces an off-policy objective; regularized importance sampling is used to avoid shallow imitation. Practical takeaway: teacher moves can supply signal where learner groups all fail. Label them as teacher samples; do not insert them into an ordinary on-policy group and pretend they were generated by the learner. Warm-start cross-entropy is a simple first control. [Primary text](https://arxiv.org/html/2504.14945v5#S3).

11. **Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm** (2025), §§1–2 framing, 4.1–4.3. The authors analyze a surrogate interpretation and show that regularization and data-distribution shaping can matter more than simplistic “on-policy/off-policy” labels. This does not assert arbitrary replay produces an unbiased policy gradient. Practical takeaway: record behavior policy and selection mechanism, compare replay under an explicitly defined surrogate, and distinguish empirical stability claims from exact importance-sampling correction. [Primary text](https://arxiv.org/html/2509.24203v1#S4).

12. **Demystifying GRPO: Its Policy Gradient Is a U-Statistic** (2026), §3.2 and the setup of §4.1. The analyzed group-relative estimator uses a leave-one-out baseline; using the full group mean differs by the factor G/(G−1). Main analysis simplifies production features, including reward standardization and multiple-update importance ratios. Practical takeaway: group size changes variance, number of distinct states, and effective learning-rate scale; sweep it at a fixed total sampling budget. Do not cite an asymptotic oracle result as proof that the repository's complete clipped trainer is unbiased or optimal. [Primary text](https://arxiv.org/html/2603.01162v3#S3).

13. **Your Efficient RL Framework Secretly Brings You Off-Policy RL Training** (2025), visible “Mismatch Problem” and system-mismatch sections. Different generation and training backends can produce different probabilities even with shared weights; actual post-transformation sampling probabilities matter. Practical takeaway: preserve behavior log probabilities and backend metadata when introducing vLLM. The current Transformers baseline should use top_p=1, top_k=0 before adding sampling truncation as a measured choice. This is a primary engineering report, not a separate peer-reviewed algorithm paper. [Primary report](https://fengyao.notion.site/off-policy-rl).

14. **Resource-Efficient Model-Free RL for Board Games / KLENT** (2026), §§4.1–4.3, 5.2–5.3. For finite actions, the paper computes a KL/entropy-regularized policy improvement target in closed form, trains policy and Q heads, and uses intermediate λ-returns. It includes 19×19 Go experiments using a 20-block ResNet and an ELF OpenGo anchor, while explicitly limiting its claims about asymptotic performance. Practical takeaway: a categorical Go interface permits full-distribution targets and exact action sums, avoiding language-token sampling variance. Porting the recipe to Qwen is a new experiment, not reproduction of their architecture or result. [Primary text](https://arxiv.org/html/2602.10894v1#S4).

## Repository audit and changes

### 1. “Full game” previously optimized only the root response

`scripts/run_grpo_fullgame.py` gives TRL a normal prompt/completion pair. Its reward function calls `simulate_full_game`, whose later model moves are generated under `torch.no_grad()` and never returned as optimization samples. The gradient therefore has the form

\[
\hat R(s_0,a_0)\nabla_\theta\log\pi_\theta(a_0\mid s_0),
\]

not the complete-trajectory estimator

\[
\sum_{t\in\text{model turns}}(G_t-b(s_t))\nabla_\theta\log\pi_\theta(a_t\mid s_t).
\]

The former can be useful as root-action policy iteration with a frozen continuation policy. It does not directly train later decisions or supply the total derivative of the policy-controlled return. Code/logs now identify this honestly. A complete trajectory implementation must preserve every observation, action, behavior probability, player perspective, termination reason, and return; replaying later moves as independent prompts while normalizing unrelated states into one GRPO group is not a valid shortcut.

### 2. Duration reward overwhelmed winning

The old full-game configuration added +0.2 for every legal model move. At 150 turns this supplied +30, versus a ±1 outcome term. This creates a strong incentive to remain in a losing game and makes reward changes confounded by duration. The implementation now applies the legacy legal bonus once to the root action. Continuation legal actions add zero; invalid actions still incur their configured penalty. New full-game recipes should set `legal_bonus: 0.0` altogether.

### 3. Estimated score is not a verifiable terminal result

Previously the sign of KataGo's `scoreLead` became ±1 even after a move cap. A truncated game was effectively reported as a won/lost terminal sample. Two passes also do not turn a neural score estimate into an exact rules adjudicator, particularly where unsettled/dead groups need resolution.

The result now distinguishes:

- `truncated`: a rollout budget was exhausted.
- `pass_terminated`: two passes ended this continuation.
- `final_outcome`: a verified result only; currently an invalid-action forfeit gives −1, otherwise `None`.
- `estimated_outcome`: sign of an available endpoint estimate, separately labeled.
- `outcome_source`: explicit provenance.

`terminal_reward` is zero without a verified result. Actual match evaluation should use a rules-aware GTP scoring/referee path and preserve adjudication metadata. “All remaining stones alive” flood-fill scoring must not be advertised as general Chinese scoring.

### 4. Local score deltas did not telescope

The previous sum contained only transitions caused by model moves, excluding opponent replies. It was neither the total endpoint improvement nor conventional potential shaping across the trajectory. The revised continuation estimator uses

\[
R_{estimate}=c[\Phi_{model}(s_T)-\Phi_{model}(s_0)].
\]

Only root and endpoint require score analysis; opponent move queries remain. This cuts analyses from approximately two per model turn to two per continuation. With a fixed root, subtracting its potential is a baseline. This is an estimated continuation objective, **not** a guarantee of policy-invariant terminal reward shaping. That guarantee would require the appropriate γΦ(s′)−Φ(s) construction and terminal treatment for the actual MDP. Missing required scores now fail visibly rather than silently becoming zero. Earlier pass history is counted at a seeded position.

### 5. Final-only masking was not reasoning RL

For a sampled reasoning prefix z and move a,

\[
\nabla\log\pi_\theta(z,a\mid s)=\nabla\log\pi_\theta(z\mid s)+\nabla\log\pi_\theta(a\mid s,z).
\]

Masking z drops its policy-gradient credit. It can still affect later activations, but is not trained by its own sampled log-likelihood term. Worse, masking every malformed answer removes the direct gradient that would discourage it. `ReasoningConfig.optimize_final_only` therefore defaults to false; explicit final-only use remains an ablation with a warning. All sampled tokens and EOS retain outcome credit by default. Existing YAML flags must be updated too, because explicit old values override dataclass defaults.

### 6. Pin algorithm choices, do not inherit unnoticed defaults

The installed dependency is TRL 0.24.0. Its actual default was already `loss_type=dapo`, not original per-response GRPO. It also defaults to group reward scaling. The new top-level `grpo:` settings are logged and passed explicitly:

```yaml
grpo:
  loss_type: dr_grpo
  scale_rewards: none
  beta: 0.0
  epsilon: 0.2
  epsilon_high: 0.28
  num_iterations: 1
  importance_sampling_level: token
  mask_truncated_completions: false
  top_entropy_quantile: 1.0
rollout:
  num_samples: 8
  temperature: 1.0
  top_p: 1.0
  top_k: 0
  max_tokens: 16
```

This is a deliberately simple **Dr. GRPO-style reduction with asymmetric clipping baseline**, not a new named algorithm and not full DAPO (no dynamic sampling was added). `mask_truncated_completions` concerns answer-token limits, not game move limits. For tiny coordinate outputs, retaining malformed/truncated gradients is a useful baseline. A later long-reasoning arm should compare explicit truncation handling without confusing it with a verified wrong move.

The fixed maximum-length denominator changes gradient scale when changing generation budget; match effective update magnitude when comparing 16-token direct moves with longer reasoning. `scale_rewards: none` also changes magnitude relative to normalized rewards, so copying the learning rate without a small calibration can confound the comparison.

### 7. Sampling and scoring must refer to the same policy

Top-p/top-k sampling renormalizes a truncated support. Temperature-softmax log probabilities alone describe a different policy. The new explicit `top_k: 0` removes an easy-to-miss generation default, and both scripts warn on truncated sampling. For the clean baseline use top_p=1, top_k=0. If using hard legal masks, teacher action injection, search, or another backend later, store the true behavior distribution and define the trained target policy accordingly. A legal categorical softmax can apply the same mask during sampling and loss; it avoids this mismatch without asking the language model to rediscover syntax rules.

## What to test first

### Representation and sample efficiency

Use identical game-disjoint 19×19 train/dev/test states and the same Qwen checkpoint. Compare compact row-major board text, coordinate lists, and a categorical action head. A snapshot requires side to move, rule/komi context, relevant ko/history, and pass/phase information; a beautiful compact format that loses these is not a sufficient state representation. Preserve a full-state record outside the prompt for exact legality checks.

Compare these outputs under matched time/token budgets:

1. Bare GTP move with thinking disabled.
2. Bare GTP move under a legal output constraint, with consistent masked likelihood.
3. One categorical distribution over 361 intersections plus pass.
4. Short reasoning plus move, retaining gradients on the whole sampled sequence.

A finite action head can train on every component of a teacher policy vector instead of receiving one scalar reward for each of eight sampled strings. A shared value/Q head has marginal parameters and does not require the separate full-size critic that motivates GRPO for language models. These are structural efficiency arguments; measure wall time, memory, move-quality improvements, and played games before declaring a winner.

Teacher warm-start should compare hard best-move targets with a soft distribution based on search visits. Soft targets preserve plausible alternatives and provide signal outside the model's initial sampled support. Include policy labels from learner-visited states, not only expert-game states, to address exposure shift. Do not present KataGo teacher agreement as self-play discovery.

### Learning experiments

Run short reproducible experiments first, keeping at least three seeds for promoted arms. Suggested order:

1. Verify the model can overfit a tiny set of legal 19×19 actions and can preserve format under inference.
2. Compare held-out likelihood/teacher-policy KL for hard targets versus soft targets.
3. Compare the categorical policy with text generation at equal compute; inspect action entropy and diverse useful moves, not only teacher top-1 match.
4. Compare DAPO reduction/group scaling with fixed reduction/no scaling, matching learning-rate scale. Test groups 4, 8, 16 at equal total generated moves and report number of distinct states.
5. Compare zero KL with small KL only after a meaningful warm start; log actual KL, gradient norm, clip fractions, and coverage collapse.
6. Add short-horizon continuations and later true full trajectories. Use a mixture of opponent strengths so wins are not almost surely impossible. Keep a fixed held-out opponent ladder for evaluation.
7. Only then compare GSPO-style sequence ratios, CISPO, entropy-selected token updates, or longer reasoning. A novel acronym is not a substitute for greater measured strength.

Do not run unbounded dynamic sampling while all groups fail. Count attempted samples, engine calls and discarded groups against the budget. A saturation report should trigger easier initial positions/opponents or more teacher coverage, not infinite resampling. Conversely, all legal but different-quality moves can still give informative continuous teacher rewards even if a binary win signal is constant.

### Finite-action policy improvement experiment

The KLENT paper motivates an explicit alternative to token GRPO. For a learned current policy p and action values q, a target can be computed as

\[
p^*(a\mid s)\propto\exp\{[q(s,a)+\beta\log p(a\mid s)]/(\alpha+\beta)\}
\]

over legal actions. Fit it with cross-entropy while learning value targets from actual trajectories. If q comes from KataGo rather than a learned on-policy Q function, call this **teacher-guided regularized policy improvement**, not a reproduction of KLENT self-play. All target distributions must be detached during policy fitting. In alternating-player value backups, change sign at opponent turns; treating both players' values as the same perspective is a severe but easy-to-miss bug. Ended episodes and truncated rollouts need different bootstrap masks. Logged behavior policy is essential if mixing replay or opponents.

## Metrics and evaluation guardrails

- Primary: win/loss/draw rate in completed 19×19 games against fixed opponents with color-balanced seeds and confidence intervals.
- Report forfeits, illegal actions, pass endings, resignations, truncated games, scoring failures, and engine fallback passes separately; never silently turn all of them into wins/losses.
- Secondary: legal rate, teacher policy cross-entropy, score-loss distribution, useful-action coverage@k, action entropy, and symmetry consistency.
- Efficiency: training wall time, generated tokens, useful state/action labels, engine queries/visits, peak memory, and inference latency per move.
- Split by source game **before** extracting/augmenting positions. Rotate/reflection-related positions and neighboring positions from a game must not leak into validation.
- Single-move score gains and successful unit tests establish components, not playing strength. Small match counts only rule out large regressions. Defer expensive repeated-seed training and hundreds-game tournaments to the user with exact configs and commands.

## Validation performed for the RL changes

The focused suite completed with **135 tests passing**: fullgame rollout, scalar reward construction, rewards/config validation, reasoning masking, and the new objective tests. The new TRL integration test constructs a local tiny causal model/tokenizer without downloads and checks that the actual trainer receives the fixed reduction, unscaled rewards, asymmetric clipping, top_p=1, top_k=0, and full response masking. Endpoint tests check opponent-transition inclusion, exact query count, unverified outcomes, missing-score failure, prior-pass termination, and absence of repeated legal bonuses. These are implementation checks; no model strength gain is inferred from them.

Remaining deliberate limitation: the root-continuation training path still is not full-trajectory RL, and KataGo analysis endpoints still are estimates. Those limitations are now visible in the API/logging rather than hidden by an outcome label.

## Implemented finite-action trajectory path

`go_rl/policy_optimization.py` now provides the detached finite-action regularized improvement target, masked categorical cross-entropy, signed generalized advantage estimates, and a categorical PPO clipped surrogate. The target operator is an experimental primitive; it is not presented as a reproduced KLENT result. The PPO implementation uses the actual normalized legal-action sampling distribution at the recorded temperature, including the same support in both old and new likelihoods. It reports exact categorical entropy and a sampled approximate KL diagnostic.

`go_rl/policy_rollout.py` and `scripts/run_policy_rl.py` now collect and optimize complete **sets of model turns**, including both colors in self-play. This is distinct from the legacy root-action continuation scorer: every collected move receives a gradient on the first PPO epoch. Subsequent epochs can stop for a sampled KL budget. The policy stays in evaluation mode for collection and optimization (gradients remain enabled during optimization), avoiding stochastic dropout changing the meaning of the behavior likelihood. No search, tactical heuristic, or teacher move selects self-play actions.

For alternating sides, the implemented temporal-difference error is

\[
\delta_t=r_t-\gamma(1-d_t)V(s_{t+1})-V(s_t),
\qquad A_t=\delta_t-\gamma\lambda(1-d_t)(1-c_t)A_{t+1}.
\]

Here \(d_t\) is genuine termination and \(c_t\) is a collection cap. A cap bootstraps the next player's detached value and ends recursion; a terminal move uses the referee outcome and no bootstrap. An unscored game never receives a binary outcome. The GTP referee must accept the full normalized rules, checks each move, and supplies `final_score` only after two passes. Such results are labeled `katago_gtp_final_score`. Their exact semantics depend on the engine version, rules and completion status; the source audit below distinguishes deterministic terminal scoring from neural adjudication. Rejected rules, illegal actions, and unscored/void referee responses abort visibly rather than becoming fabricated wins or losses.

**KataGo 1.18.2 scoring audit:** in a finished AREA game with `friendlyPassOk=false`, the GTP implementation returns the already computed history score; it does not call neural lead estimation. Human-friendly or incomplete-game scoring instead uses a search-based estimate. The project's raw positional-area rules normalize to `friendlyPassOk=false`. [Pinned GTP implementation](https://github.com/lightvector/KataGo/blob/v1.18.2/cpp/command/gtp.cpp#L1354-L1377).

After the second pass, `BoardHistory` invokes terminal area scoring; TAX_NONE calls `calculateArea` with all area flags enabled. This is deterministic scoring under KataGo's formal convention. [Pinned history implementation](https://github.com/lightvector/KataGo/blob/v1.18.2/cpp/game/boardhistory.cpp#L573-L582), [terminal transition](https://github.com/lightvector/KataGo/blob/v1.18.2/cpp/game/boardhistory.cpp#L1057-L1062). That routine includes Benson pass-alive territory, so it can differ from counting every remaining stone plus flood-filled empty regions. [Pinned board implementation](https://github.com/lightvector/KataGo/blob/v1.18.2/cpp/game/board.cpp#L1721-L1741). A separate raw-board-area diagnostic therefore must not enforce equality with KataGo or silently replace its reward. Source inspection corrects the overly broad assumption that every GTP score is a neural estimate; it does not make analysis `scoreLead` or a capped game a verified outcome.

The source finding now has an actual-engine regression: `tests/fixtures/benson_raw_area_19x19.sgf` contains two corner White stones inside a Black pass-alive territory and ends with two passes. On KataGo 1.18.2, repeated `final_score` returns B+353.5 while the independent raw-board scorer returns B+347.5. The six-point difference is expected because the three-point corner changes owner. The test uses `loadsgf`/`final_score`, with no search or analysis query; it verifies the loaded rules and komi separately. Thus raw-score equality is a diagnostic observation in the two model games, not a universal invariant.

The default script budget is deliberately a bounded plumbing experiment (one update, one episode, 64 plies). A short run that produces only caps mostly tests bootstrapped value consistency and entropy optimization; it does **not** establish learning from new completed-game rewards. Use full-length completed games and a substantial episode budget for the actual terminal-reward experiment, after warm-starting the policy and value head. Current self-play is synchronous and sequential; batching multiple active boards is the next throughput optimization if measurements justify it. No population league or opponent mixture is implemented yet.

The additional focused suite now passes **84 tests** across root rewards, fullgame semantics, TRL objective wiring, masking, categorical objectives, trajectory collection, and checkpoints. The new trajectory tests verify both-color credit, opponent-perspective signs, actual stored mask/temperature likelihoods, no cap win labels, required referee rules, and a real tiny-Qwen optimizer update over all three sampled moves. Local tiny-Qwen checkpoint tests perform an update and verify exact policy/value restoration for both LoRA and frozen-backbone head probes. These tests do not measure Go strength.

### Real Qwen trajectory and protocol smoke

Starting from `outputs/research/policy_compact_soft_s3407/best`, the new runner collected 16 actual Qwen3-0.6B self-play moves on 19×19 and optimized all 16 in eight minibatches. Collection plus optimization took **15.99 seconds** on the local MPS device. The game hit its explicit 16-ply cap: **zero adjudicated games, one truncation, no win label**. The detached next-player bootstrap was −0.204. Mean legal-action entropy was 5.538 nats, sampled approximate KL 0.00638, and clip fraction 0.0625. These numbers validate the training path; with no completed-game reward they do not show improvement in playing strength.

The saved checkpoint at `outputs/research/policy_ppo_smoke_s3407` was then loaded through the actual GTP CLI. It played N5, accepted a protocol-test White move A19, played O5, and reproduced O5 after undo. It correctly reported `final_score` unsupported. The protocol check took 2.79 seconds. The real KataGo referee accepted normalized JSON rules and checked every action during the PPO smoke. Both processes closed after their bounded checks.

### Measuring more than teacher agreement

`scripts/eval_move_quality.py` now compares actual categorical and text-model predictions on the same game-balanced held-out positions. It audits source-game and symmetry-normalized state separation against the supplied training file. A fresh root search selects a reference move, then the reference, each model choice, and a seeded uniform legal choice all receive explicit child-position analysis at the same configured visit cap (default 32). Even a move absent from root `moveInfos` is evaluated. Duplicate child actions share exactly the same estimate.

Reported differences are chosen-child minus reference/uniform-child score lead and winrate, in the original player's perspective. They are approximate paired engine diagnostics, **not exact regret or terminal rewards**. Legality includes every prediction; quality summaries explicitly condition on legal predictions. Portable JSON includes full selected state histories, model and dataset hashes, engine provenance, visit budgets, predictions, and estimates. Confidence intervals resample source-game clusters after averaging each game's positions, so source games receive equal weight. They do not account for engine bias, training-seed variance, or selection of a best model on the same validation set. A final untouched test set and completed matches remain necessary.

The distillation runner additionally exposes `--target policy_argmax`: onehot the same stored policy distribution to isolate target softness. Existing `--target move` uses the searched best move, which can differ from the argmax of prior or mixed targets. Comparing soft prior/mixed targets with searched-best-move SFT changes the supervision source as well as softness; descriptions must preserve that distinction. Four target tests and five evaluator tests cover distribution validation, explicit analysis of unsearched actions, perspective changes, duplicate caching, illegal predictions, deterministic uniform draws, and clustered uncertainty.

### Stronger frozen evaluator result

**Audit correction:** the original table below is preserved as historical evidence. One selected history had already reached two passes; the corrected, matched 31-position results below supersede its quantitative interpretation. No new engine or model calls were used for that correction.

The first matched evaluation used the b18 `s9996604416-d4316597426` network, 32 visits per query, 32 positions from 16 held-out games, and 248 total root/unique-child queries. It finished in 89.63 seconds. Every prediction was legal under constrained inference. Estimated score differences against the seeded uniform-legal baseline were:

| Model | Mean score difference | Game-cluster 95% interval |
|---|---:|---:|
| Text SFT | +3.04 | [+1.18, +5.34] |
| Compact categorical | +1.08 | [−0.94, +3.12] |
| Causal cells | −1.35 | [−4.06, +1.11] |
| Bidirectional cells | +0.27 | [−1.59, +1.96] |
| Base Qwen constrained text | −2.02 | [−3.57, −0.73] |

Text SFT exceeded base Qwen by an estimated 5.06 points [2.84, 8.41] on this sample. Its estimated advantage over the compact categorical model was 1.96 [−0.19, 4.31], which is inconclusive. All models were materially worse than the fresh b18 reference action: the closest, text SFT, averaged −5.80 estimated points versus that reference. Thus lower distillation loss did not establish superior move quality, and categorical output has not yet beaten the text baseline. These are 32-position engine diagnostics, not completed-game strength. Full provenance and individual estimates are in `outputs/research/move_quality_b18_32pos_s3407.json`.

### Terminal-history audit and training-seed replication

The source corpus contains post-game cleanup histories. Auditing the actual two-pass endpoint found 5/256 such training rows in v1, 78/1024 in v2, and 1/64 validation rows. The same invalid position occurred in both 32-position evaluation samples. `audit_move_quality_terminal.py` preserves each original report, removes only that position, and recomputes game-cluster summaries from the saved predictions and child estimates. Its new report records the source hash, excluded position, post-hoc status, and zero additional queries. The corrected original comparison on 31 positions from 16 games gives text SFT +2.972 points [1.091, 5.280] versus uniform and base Qwen −2.093 [−3.643, −0.797].

The replication run evaluated three independently trained text and bidirectional-cell seeds, a query-cell model, and larger-data text models with the same b18@32 evaluator. Its original 32-root/268-unique-child run took 143.39 seconds; the following figures use the same 31 preterminal positions after the audit:

| Model | Estimated score difference vs uniform | Game-cluster 95% interval |
|---|---:|---:|
| Text v1, seed 3407 | +3.095 | [+1.217, +5.431] |
| Text v1, seed 3408 | +1.199 | [−1.132, +3.034] |
| Text v1, seed 3409 | +0.110 | [−3.287, +2.636] |
| Bidirectional cells, seed 3407 | +0.326 | [−1.507, +1.975] |
| Bidirectional cells, seed 3408 | −0.662 | [−3.106, +1.554] |
| Bidirectional cells, seed 3409 | −0.354 | [−2.060, +1.181] |
| Query cells causal, seed 3407 | −0.806 | [−3.426, +1.458] |
| Compact text v2 + D4, seed 3407 | +1.576 | [−0.742, +3.750] |
| Grid text v2 + D4, seed 3407 | +1.185 | [−1.284, +3.373] |

The intervals describe position/game sampling, not uncertainty across training seeds. Choosing seed 3407 after seeing these results incurs validation selection bias. Neither more data nor lower imitation NLL established better move quality here. The terminal rows in the training data also limit causal interpretation; new teacher-data experiments exclude histories at or past the two-pass endpoint before sampling. The evaluator now applies that filter before selecting positions, reports exclusions, and rejects any terminal root passed to its lower-level scorer.

The next matched decoder comparison adds an empirical coordinate-frequency control. It fits only explicit teacher `response` counts from pinned training rows, chooses the most common legal coordinate with a deterministic row-major tie break, and never reads held-out labels. Optional D4 expansion transforms coordinates; pass is invariant. This baseline intentionally retains the original training rows, including terminal-history rows, to match the actual model's training distribution, and records their count. Every frequency source also participates in the game/state leakage audit. It can expose a static-location explanation for apparent model gains without prescribing Go tactics.

## Learner-state refresh: latest distillation research and next experiment

Additional primary methods read on October 23, 2026:

- [OPSD, January/March 2026](https://arxiv.org/html/2601.18734v3), §3.2 and §4.3.5: evaluates a privileged teacher on the student's sampled prefixes, with detached teacher targets and full-distribution divergence as an alternative to sampled-token advantages. For this repo, student-visited **board states** are the relevant distribution. Dense 362-action targets are inexpensive enough that token-vocabulary approximations are unnecessary. This is a design analogy, not a reproduction of OPSD self-teaching.
- [SSOPD, May 2026](https://arxiv.org/html/2605.17497v1), §3 and Appendix C: uses shortest-correct/longest-wrong witnesses, a frozen teacher, and a success-mixture weight. Its auxiliary signal disappears when groups contain no successes. A weak Go model lacks the reliable successful witnesses and language understanding needed to transfer that recipe directly; do not convert an engine score estimate into a fake binary witness.
- [Entropy-Aware OPD, March 2026](https://arxiv.org/html/2603.07079v1), §4 and Algorithm 1: adds forward KL where teacher entropy is high while retaining the sampled reverse-KL term. Its experiments include Qwen3-0.6B-Base. This motivates measuring teacher/student entropy and preserving plausible alternative moves. It does not establish that a thresholded objective beats our existing full-action forward-KL baseline in Go.
- [Does OPD Really Distill?, August 2026](https://arxiv.org/html/2608.31046v1), §§3.2–4.2: reports gains from suppressing selected low-probability reasoning tokens without teacher labels. This is an important control for claims about teacher signal. It gives no Go-specific reason to penalize low-probability legal moves indiscriminately; rare good moves are exactly what a weak Go policy may need to discover.
- [Does self-distillation improve reasoning?, October 13, 2026](https://arxiv.org/html/2610.18613v1), §§4.1–4.4 and Appendix C: re-evaluates OPSD/SDPO, separating unseen-problem performance and Qwen thinking-mode transfer from privileged-context effects. Its findings caution against attributing training gains to self-correction without those controls. Our fixed mode, game-disjoint states, stronger frozen evaluator, and eventual completed games address analogous confounds; they do not guarantee generalization.
- [DAgger, 2011](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf), §3/Algorithm 3.1: collect states visited by the learner, query expert actions there, aggregate, and refit. This is the most direct foundation for a Go learner-state refresh loop. Its theoretical assumptions do not automatically hold for nonlinear Qwen optimization or a finite-budget approximate KataGo teacher.

The proposed first refresh experiment keeps the **same teacher network, visit budget, target mixture, training steps, encoding, and validation** as the static-data control. Changing both teacher strength and state distribution would confound the result. A stronger b18 evaluator can remain frozen for both arms.

1. Freeze checkpoint θₖ while collecting a bounded set of its stochastic 19×19 self-play episodes. Existing `collect_selfplay_episode` already stores the full action sequence and genuine terminal-versus-cap status. Reconstruct each pre-action board by replaying `episode.moves`, then serialize it with `state_record_from_board`; do not try to infer stones/history from token IDs.
2. Assign a new episode-level `game_id` from checkpoint hash, seed, rules, komi, and initial state. All positions and symmetries from that episode remain in the same group. Preserve behavior checkpoint, action, actual log probability, temperature, and episode boundary as collection provenance. Teacher labels are explicitly off-policy **targets at learner-visited states**, not PPO behavior actions.
3. Uniformly select a fixed number of states per collected game before teacher querying. For a bounded pilot, four 64-ply episodes and at most 16 labels per episode produce 64 fresh states. The cap means no full-game reward claim. An optional training-game seed curriculum can reach later stages faster, but must be labeled as a changed starting-state distribution; held-out games cannot seed collection.
4. Remove states overlapping any validation/test state under D4 normalization before querying the teacher. Query `make_teacher_record` targets with the exact recorded rules/komi/history. Keep approximate teacher values separate from any true episode outcomes. Save immutable raw learner-state and labeled JSONL files plus hashes.
5. Compare a fixed 80/20 expert/learner-state training mixture against an 80/20 expert/fresh-expert-label control at equal teacher-query and gradient budgets. Retain earlier learner-state records in a bounded, game-balanced reservoir on later iterations. Initially use existing dense forward-KL fitting; an entropy-aware loss is a separate later ablation, not bundled with the distribution change.
6. Re-evaluate the frozen held-out move-quality set and color-balanced completed games. Track state novelty, teacher entropy, student entropy, pass/closeout behavior, cap rates, and score differences. Only promote an arm after repeated seeds; otherwise preserve the negative result. No GPU refresh experiment has been run yet.

### Implemented collection-to-teacher bridge and current limits

`export_learner_positions.py` reconstructs sampled self-play states from the saved episode histories, and `export_learner_sgf.py` selects only the named model's turns from actual GTP match SGFs. Opponent moves remain in the replay history but are not mistaken for learner queries. Both exporters pin all output to the training split, preserve full state/rules/komi, exclude specified held-out D4-equivalent states, and keep the learner action as provenance rather than an expert response. A later `build_teacher_dataset.py` call supplies labeled targets with explicit engine provenance. Completed results are checked for valid syntax and consistent outcomes; capped trajectories have no adjudicated outcome. Teacher annotation preserves learner metadata and marks the record as labeled.

Unlike the initial proposal above, the implemented game-group identity is canonicalized from the entire move trajectory and starting stones across D4 orientations, plus rules and komi. This groups replayed equivalent trajectories across runs. Run/checkpoint provenance remains separately recorded. Existing exported checkpoint hashes are explicitly observations at **export time**, not proof of the exact collection weights. The PPO episode summaries do not persist every behavior log probability or every intermediate weight snapshot. The bridge therefore supplies states for fresh teacher annotation; it is not an importance-weighted off-policy replay dataset. The first exported 16-state categorical smoke remains separate from the planned text-model refresh, since their collection policies differ.

The updated PPO collector additionally records initial checkpoint fingerprints at collection time. It accepts optional game-balanced late positions from an explicitly training-only source with the same rules/komi. Seeds must reconstruct completely from empty; unsupported initial setup stones and already ended histories are excluded. The full seed prefix is replayed to the referee and retained in exports, including a preceding pass. Only newly sampled model moves enter the PPO objective. This supports a bounded terminal-reward plumbing test from late positions; it changes the start-state distribution and is not evidence of learning to win a whole game from empty.

### Complete-move decoding as a controlled interface ablation

The default text interface chooses each constrained token greedily. Coordinate prefixes can share probability mass: if P(A)=0.6, P(B)=0.4, P(1|A)=0.51, and P(1|B)=0.99, greedy selects A1 while B1 has the greater complete-string probability, assuming equal EOS likelihood. Forcing a legal EOS does not make its original model probability equal to one. This matters when comparing two- and three-token coordinates, and PASS.

`TextGoPolicy(num_beams=4)` or 8 now provides an optional approximate maximum-likelihood legal complete move. The installed Transformers beam implementation was inspected directly: it applies full-vocabulary log-softmax before the prefix constraint processor adds zero or negative infinity. The new settings retain these raw sequence scores (`renormalize_logits=False`), include EOS, disable length normalization (`length_penalty=0`), and use the nonheuristic beam stopping bound (`early_stopping="never"`). Beam decoding is deterministic and restricted to non-thinking constrained text. It does not sample from the globally conditioned legal-move distribution, and finite beam width does not guarantee its exact mode.

CPU tests use the actual tiny-Qwen/Transformers cache and beam machinery with prescribed token probabilities. They demonstrate recovery from greedy prefix mistakes and verify that low original EOS probability affects the result rather than being renormalized away. An additional test uncovered Transformers' automatic inheritance of checkpoint generation defaults even with a fresh `GenerationConfig`; the beam path uses `use_model_defaults=False` to prevent saved suppression or repetition processors from changing its objective. The GTP CLI exposes `--num-beams`; the interface benchmark accepts `--beam-widths 1 4 8`; the move-quality evaluator accepts `--text-beam-widths 1 4 8`, comparing all decoder choices against shared child-position estimates. No actual Go-quality gain is claimed before the held-out evaluation and inference-cost measurement. This deterministic beam option is for inference: its selected move must never be treated as a sample from the original policy for an on-policy RL update.

### Measured beam and coordinate-frequency comparison

The next b18@32 run excluded the terminal source row **before** deterministic sampling, producing 32 valid positions from 16 games. All 271 root/unique-child queries and ten prediction variants completed in 300.20 seconds. This selection replaces one earlier sampled state, so its means should not be subtracted directly from the earlier 31-position post-hoc reports. All paired differences below use shared child estimates within this run.

| Text policy | Decoder | Score difference vs uniform (95% game CI) | Model inference seconds / position |
|---|---|---:|---:|
| v1 seed 3407 | Greedy | +3.301 [1.459, 5.564] | 0.429 |
| v1 seed 3407 | Beam 4 | +2.090 [0.398, 3.469] | 1.009 |
| v1 seed 3407 | Beam 8 | +2.883 [1.753, 4.116] | 1.497 |
| Compact v2 + D4 seed 3407 | Greedy | +1.616 [−0.627, 3.597] | 0.392 |
| Compact v2 + D4 seed 3407 | Beam 4 | +3.953 [1.730, 6.567] | 0.819 |
| Compact v2 + D4 seed 3407 | Beam 8 | +3.794 [1.633, 6.399] | 1.380 |
| Base Qwen | Greedy | −1.990 [−3.606, −0.700] | 0.271 |
| Base Qwen | Beam 4 | −2.303 [−3.918, −0.926] | 0.682 |
| Base Qwen | Beam 8 | −3.677 [−5.484, −1.908] | 1.198 |

The fitted v1 coordinate-frequency policy scored −0.646 [−3.847, 1.914] versus uniform. The v1 greedy model exceeded that control by a paired +3.947 [0.675, 7.779], providing evidence against this simple static coordinate prior as the entire explanation of the gain. This does not establish sophisticated board reasoning: occupancy, shape priors, recency and other learned features remain possible explanations.

Compact-v2 beam 4 exceeded its own greedy decoder by a paired +2.336 points [0.038, 6.143], at roughly twice the inference time. Beam 4 on v1 instead changed the estimate by −1.211 [−3.227, 0.245]. Base Qwen beam 8 passed on 31/32 positions and performed worse. Thus closer approximation to the most probable complete string does not universally improve Go decisions; it exposes what the model learned. These exploratory intervals are not corrected for trying multiple models/decoders on one small validation sample. Compact-v2 beam 4 was selected for the bounded full-game test by the highest observed mean, not a statistically established claim of superiority. Source report: `outputs/research/move_quality_b18_beams_frequency_valid32_s3407.json`.

The categorical interface additionally supports optional D4 probability averaging (`--policy-symmetries 1 8`). It transforms complete state histories, evaluates all orientations in one expanded batch, maps legal action probabilities back to the original frame, and averages probabilities and side-to-move values. PASS stays fixed. Reports count every orientation's tokens and forward sequence, including the eightfold expanded batch. Tiny-Qwen and independent geometry tests cover correct inverse mapping, ko masks, and parity with eight separate forwards. This defines a different inference policy; it does not change PPO sampling, and no actual D4 quality measurement has yet been made.

## Actual full-size games exposed a missing training distribution

The selected compact-v2 beam-4 policy played two complete 19×19 games **from the empty board**, swapping colors against the b6 KataGo network at eight visits. Both ended by two passes; neither was truncated, forfeited, or assigned a synthetic result. The 1,083-ply cap, if reached, would have produced Void. The outcomes were:

| Model color | Official result | Total plies | Model passes / turns | First model pass |
|---|---|---:|---:|---:|
| Black | W+40.5 | 70 | 34 / 35 | Ply 1 |
| White | B+337.5 | 509 | 19 / 254 | Ply 472 |

The model lost both games; no finite Elo estimate is reported from this all-loss sample. The Black game ended with only one black stone and 34 white stones, with 326 neutral points. The White game ended with seven white stones against 254 black stones and 98 black territory points. Independent raw-board flood-fill scoring matched both official margins exactly; this agreement in these games does not erase the general Benson-versus-raw-area distinction above. Complete SGFs, engine log, result summary, raw-score audit, and checkpoint/config/code hashes are in `outputs/research/text_fullgames_compact_v2_beam4/`.

This is a substantial negative result for the current policy. Its validation pass rate was 1/32, yet it passed on 34/35 Black turns from empty. A quick fixed-state diagnostic reproduced the failure with **both greedy and beam-4 decoding**:

| Observation | Greedy | Beam 4 |
|---|---|---|
| Empty board, Black to play | PASS | PASS |
| Empty board after Black passes, White to play | PASS | PASS |
| Black D4, White to play | B10 | C12 |
| Black passes then White D4, Black to play | PASS | PASS |

These are constructed diagnostic observations, not a statistical strength test. They rule out beam selection alone as the cause of the Black opening failure. **White passing immediately after Black's empty-board pass is correct under this task:** it ends the game and wins by the 7.5 komi. The four probes must not be ranked by total non-pass frequency. Black's opening pass allows that immediate forced loss; the important coverage checks distinguish the player and previous move. The original training builders defaulted to a minimum of move 10: v1 contained only one row at move ≤10, v2 contained nine, and neither contained the empty board. Validation began at move 18. Compact tokenization additionally compresses the empty board to 201 input tokens, compared with more varied occupied boards; this is a plausible representation shift, not an established causal diagnosis. Input/output correctness and positive midgame engine diagnostics were insufficient evidence that the model could start and complete a useful game.

The next teacher-budget/learning-rate comparison therefore uses a newly frozen v4 source with 64 opening states (eight empty-board source rows and 56 early positions) and 448 other valid phase-balanced states. The original frozen source is preserved, and no prior labels were reused under a misleading name. Validation and final-test D4 state overlaps are excluded. This corrects task coverage with data instead of adding a hand-coded opening move or pass prohibition. It still needs another completed-game test, since seeing opening training states does not establish robust gameplay.

The SGF bridge exported **99 learner pre-action states**: all 35 Black-game states and a game-balanced cap of 64 White-game states, selected from 289 eligible states after D4 deduplication and exclusion against the validation and untouched final-test corpora. No overlaps were found. `data/text_learner_refresh_compact_v2_beam4_source_v1.jsonl` has SHA256 `3c7dcfbbe7c7c58935f047efff8b8d5e1edab41c585b8b4c57f22a45a525d6e2`. Learner actions, game results, and checkpoint observations remain provenance, not expert response labels. Fresh teacher annotation can now compare learner-state coverage against a static-source control at the same teacher budget.

The endpoint audit also closed a remaining legacy-RL hole: root-continuation rollouts configured to stop after two passes now reject already-ended histories, and scalar area-scoring reward contexts reject them before engine queries. `BoardState.has_two_pass_history()` tracks the entire known history, including an identifiable pass before a snapshot; checking only the current trailing passes would miss a cleanup move after the endpoint. Territory scoring is not indiscriminately declared finished by this area-specific scalar guard, because its phase semantics can differ. These checks establish input correctness, not an improvement in learned play.

The categorical collector now also rejects unsupported terminal semantics **before** model or referee work. It is deliberately restricted to area scoring with no tax, button, handicap bonus, or friendly-pass shortcut. Two local passes are not a sufficient terminal-state representation for territory-scoring encore phases. The active positional-area experiments satisfy this restriction; they are unchanged. Four regression cases cover territory rules, named Chinese defaults with friendly passing/handicap semantics, button rules, and explicitly enabled friendly passing. The rollout/seed/runner focused suite passed 32 tests after this guard.

### Predeclared bounded comparisons after opening coverage repair

The four v4 models cross teacher search at 8 or 64 visits with learning rates `1e-4` or `5e-4`. They use the same 512 original training positions, D4 augmentation, and 120 optimizer updates. All four selected NLL checkpoints are the final step 120. Teacher responses agree on only 262/512 positions (51.2%); their searched-label supervision therefore differs materially despite the shared source. Validation NLL uses the earlier 8-visit labels and can favor imitation of that budget. The independent b18@32 child-position comparison is the principal development quality diagnostic.

The separate refresh experiment uses the same old compact-v2 warm start and 120 updates on 256 examples per arm: 192 shared expert states plus either 64 learner-visited states or 64 expert controls matched by move-number bin. Both arms include five empty-board rows. Its expert-control NLL-best checkpoint was the unchanged initialization. Before any new quality inference, the frozen protocol was amended to compare **both final step-120 checkpoints**, preserving equal update budgets. The original protocol, amendment timestamp, command hashes, and source snapshot are retained under `logs/research/`. This amendment uses training diagnostics to preserve the causal comparison; it does not select checkpoints by later Go-quality outcomes.

The next 32-position evaluation uses the exact same valid position IDs as the earlier beam comparison, fixed beam width four, the same teacher objective and b18@32 budget, plus the categorical model with one/eight D4 orientations and fitted response-frequency controls. Selection among the six new text candidates requires legal outputs and a non-pass move for Black on the empty-board probe, then ranks equal-game mean score difference from uniform. If no candidate meets that opening constraint, there is no promotion claim. The probe states are known task-coverage checks, not held-out generalization evidence; the selected model still needs two completed from-empty games. The final test remains untouched at this stage.

### Measured v4 budget, refresh, and D4 results

The fixed evaluation completed in 297.91 seconds with 32 root queries and 269 unique child queries, all at b18/32 visits. Its 32 position IDs exactly match the previous valid beam evaluation. Every prediction was legal. Equal-game mean score differences from the shared seeded uniform baseline were:

| Candidate, fixed beam 4 unless categorical | Estimated points vs uniform | 95% game-cluster interval |
|---|---:|---:|
| Teacher 8 visits, LR 1e-4 | +1.109 | [−1.577, 3.870] |
| Teacher 64 visits, LR 1e-4 | +4.537 | [1.344, 8.985] |
| Teacher 8 visits, LR 5e-4 | +3.635 | [1.050, 6.784] |
| Teacher 64 visits, LR 5e-4 | +1.709 | [−0.529, 4.225] |
| Previous compact v2 | +4.013 | [1.769, 6.625] |
| Refresh expert control, final step 120 | +3.000 | [1.528, 4.715] |
| Refresh learner mixture, final step 120 | +1.315 | [−0.744, 3.224] |
| Bidirectional categorical, direct | +0.299 | [−1.544, 1.928] |
| Bidirectional categorical, eight-orientation ensemble | −1.729 | [−5.891, 1.848] |
| Static response frequency, 8-visit labels + D4 | −0.367 | [−2.710, 1.888] |
| Static response frequency, 64-visit labels + D4 | +0.670 | [−1.601, 3.123] |

Teacher search and learning rate interact in this small sample: 64−8 visits at LR 1e-4 was +3.428 [−0.725, 9.755], but at LR 5e-4 was −1.926 [−6.074, 0.785]. Neither paired interval establishes a teacher-budget gain. The learner-state refresh underperformed its phase-matched control by −1.685 [−3.783, −0.059]. That negative result is limited to one seed, 64 selected learner states from two games, this mixture, and this update budget; it does not refute learner-state aggregation generally. Intervals omit training-seed variation and are not corrected for repeated exploratory comparisons.

D4 probability averaging changed the categorical estimate by −2.028 [−6.037, 1.221]. It consumed 256 forward sequences and 117,718 input tokens versus 32 sequences and 14,690 tokens; measured forward time rose from 6.57 to 41.38 seconds. It therefore provided no supported gain for this checkpoint, despite its larger inference budget. The result does not establish that symmetry ensembles are generally harmful.

All six new text models passed the predeclared legality/non-pass-empty-Black eligibility test. The selected model was the 64-visit/LR1e-4 checkpoint at step 120, with fixed beam four. It exceeded old-v2 by only +0.524 [−1.017, 2.571], so a midgame improvement over the previous model is **not established**. It exceeded the matching static frequency baseline by +3.866 [1.332, 7.019]. Its opening probe moves were Q16 (empty Black), R17 (White after Black passes), D1 (White after Black D4), and D1 (Black after passing then White D4). It now starts playing as Black, but still misses White's immediately winning pass. The learner-refresh model recognized that White pass while still passing as Black in the final probe, illustrating why total pass frequency is not a useful objective.

Selection and paired contrasts were frozen in `outputs/research/v4_selection_and_contrasts.json` before starting two new full games. The raw comparison is `outputs/research/move_quality_v4_b18_beam4_d4_valid32_s3407.json`; its SHA256 is `98a9c3cd69ac6a7335710e0dd9f824eb8a2682720bb29752abfe9ad3962366e0`. These remain development results, with a deliberately untouched final test at selection time.

### Selected v4 model: two completed games, still no strength success

The frozen 64-visit/LR1e-4/beam-4 candidate then played two from-empty 19×19 games against the same b6@8 opponent, with colors swapped. Both ended by two passes, with zero caps, forfeits, or scoring failures:

| Model color | Official result | Plies | Model passes / turns | First model pass |
|---|---|---:|---:|---:|
| Black | W+368.5 | 530 | 11 / 265 | Ply 489 |
| White | B+336.5 | 511 | 11 / 255 | Ply 474 |

The model lost both; the summary correctly leaves Elo undefined for this all-loss sample. The Black game ended with zero Black stones and 262 White stones plus 99 White territory. Independent raw-board scoring agrees with both official margins. Source snapshots, collection-time checkpoint fingerprints, complete SGFs, logs, behavior audits, and scoring diagnostics are preserved in `outputs/research/text_fullgames_v4_v64_lr0.0001_beam4/`.

This confirms a change in participation: the earlier model passed on 34/35 Black turns starting at ply 1, while this candidate first passed at ply 489. It does **not** establish a playing-strength improvement. Both candidates lost both games, and the new Black loss was larger. A smaller losing margin from an early-pass game is itself an unsuitable substitute for win-rate or strategic competence. The selected model remains very weak against this opponent; the development single-move advantage and removal of the opening failure are insufficient to claim the final goal of strong full-size Go play.

A CPU-only export retained 128 new learner pre-action states (64 from each game) from 520 eligible learner turns. No D4 duplicate states or validation/final-test overlaps were found. `data/text_learner_v4_v64_lr0.0001_beam4_source_v1.jsonl` has SHA256 `6da6bb6e3dd45ffdebaf4323a0a98c896c9a1a7ed19f44bd72bb6769e64655b8`. These records are pinned to training and carry behavior/checkpoint provenance; they have no newly queried teacher labels. They are available for a future controlled refresh without altering the already frozen comparison.

Finally, the present full-trajectory PPO implementation trains the categorical policy/value interface, whereas the strongest measured development candidates use coordinate-text distillation. A successful categorical seeded-PPO smoke therefore checks actual trajectory optimization and terminal rewards; it does not demonstrate RL improvement of the selected text model. A future bridge—initializing the categorical backbone from a text-trained adapter and fitting its heads, or implementing text-trajectory policy gradients—requires a separate measured experiment. It is not part of the completed quality claims above.
