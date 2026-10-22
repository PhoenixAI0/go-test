# Independent model review and spatial learning research — 2026-10-23

## Review result

Reviewed `model_interface.py`, `policy_model.py`, `teacher_data.py`,
`build_teacher_dataset.py`, `run_policy_distill.py`, and the shared experiment
utilities. The policy soft-target loss correctly handles zero mass on masked
actions, the value conversion correctly changes Black-relative engine values
to the player to move, and current split validation rejects shared games and
visible states up to D4 symmetry. These checks do not establish playing strength.

Confirmed issues fixed during review:

* Teacher metadata incorrectly called the already converted `teacher_value`
  Black-relative. New schema version 2 declares `side_to_move`; raw engine
  fields remain Black-relative. Existing v1 generated datasets are unchanged
  so their recorded hashes remain valid. Their numerical values are correct;
  consumers must account for the documented metadata mistake.
* `--init-checkpoint --head-only` silently ignored the head-only request and
  trained restored adapters. This combination now fails before loading data
  or models, with a regression test.
* Teacher provenance omitted the engine configuration hash. Newly generated
  datasets include `engine_config_sha256` alongside model and input hashes.

Added actual Transformers/PEFT persistence tests using a locally created tiny
Qwen model. After training, frozen and LoRA backbones both reproduce policy and
value outputs through save/load, for last-token/causal and cell/bidirectional
architectures. The four round trips pass on CPU. The relevant model, interface,
teacher, experiment utility, persistence, and CLI tests passed (24 tests in the
combined selection at review time). The first real compact experiment's mean
input length is reproduced exactly by current preprocessing: 352.453125 tokens.

Scientific interpretation:

* With `prior_mix=1`, soft policy targets are the raw network prior. `response`
  and top-1 accuracy instead refer to the searched best move. A regression test
  deliberately makes them disagree; report both objectives explicitly.
* Lower loss than a random initialized head is insufficient. The first run's
  final policy CE 5.2943 is worse than uniform legal CE 5.2049. It is a failed
  quality experiment, despite learning relative to its initialization.
* Cell readouts originally omitted recent history that compact prompts retain.
  Match history/pass context when comparing representations. Keep exact full
  history in the rules engine even when the neural observation is abbreviated.
* Small validation sets of correlated positions require game-level uncertainty
  and an untouched final test set after adaptive experiment selection. Avoid
  treating several similar checkpoints tested on 16 games as independent proof.
* Pin preprocessing schema, backbone revision, datasets, and engine settings.
  A checkpoint's encoding name alone cannot identify a changing serializer.

## Primary methods read

### Chessformer — May 2026

[Methods §§3–5 and appendices A/C/D](https://arxiv.org/html/2605.19091).
The encoder assigns one token to each square, concatenating seven previous
boards as feature channels. Its policy uses projected source/destination
representations rather than one global vector with an unrelated output per
action. A learned dynamic geometric attention bias improves controlled chess
ablations. Searchless policy evaluation uses one forward pass; selecting by
successor values requires many calls. History ablations improve from zero to
seven previous boards, with little additional benefit at 31.

For Go, a shared intersection readout is a reasonable experiment, not a direct
reproduction. The paper trains board encoders rather than converting pretrained
Qwen. Its full pairwise bias generator also grows with squared board area;
copying it directly from 64 to 361 tokens is not an inexpensive modification.
Use simple spatial readouts first and measure equal-compute whole-game outcomes.

### Understanding Reasoning from Pretraining to Post-Training — August 2026

[Methods §2, scaling/mechanisms §§3–4, appendix C](https://arxiv.org/html/2607.16097).
Chess-specific autoregressive models are pretrained on game sequences, then
trained on serialized continuation trees and puzzle rewards. Their reasoning
uses move tokens, not natural-language prose. Puzzle verification checks each
solver move against a supplied solution line and ends on a mismatch. Games
intersecting post-training/test positions are removed from pretraining. Better
pretraining losses predict stronger post-RL performance in their controlled
sweep; RL can also strengthen wrong alternatives.

Application: establish a useful Go prior before expensive sparse-outcome RL,
retain state-level decontamination, and compare short verified variations with
prose at equal inference cost. Puzzle solution matching is not an exact verifier
for arbitrary full-game Go move quality. Their scaling fit does not predict the
gain from another hour on our 0.6B checkpoint.

### Queen — October 2026

[Architecture/curriculum §3 and ablations §5/appendix A](https://arxiv.org/html/2610.03695).
Queen joins a frozen expert chess encoder with a language decoder through gated
cross-attention. It first learns programmatically checked board, legality, and
future-state questions, then uses iterative search distillation. Removing either
the expert encoder or adaptation curriculum severely hurts its initialization.
The lighter prefix-projection alternative requires decoder LoRA adaptation.
Its cross-attention bridge itself is large; it is not evidence that ordinary
small-model text SFT alone yields strong play.

Application: test exact transition/capture auxiliary targets before generating
verbose explanations. A frozen KataGo-feature bridge is a possible later design,
but requires a real feature-extraction interface and must be reported as an
expert-encoder system at inference. This is more engineering than the current
bounded readout ablations justify.

### LLM2Vec — 2024, relevant mechanism behind the current attention ablation

[Method §2, controlled representation analysis §3–4, appendix D](https://arxiv.org/html/2404.05961).
The recipe combines bidirectional attention, masked next-token prediction,
and contrastive learning. Simply removing the causal mask worsens most tested
models; adaptation is important. Masked-token targets are predicted from the
preceding position, preserving the pretrained next-token convention. Contrastive
training helps sequence embeddings but can hurt token-level tasks relative to
masked prediction alone. These are text representation results, not Go results.

Application: call the present option a bidirectional ablation, not an LLM2Vec
reproduction. If it underperforms, test a brief spatial adaptation objective or
a causal global-query policy before discarding spatial representations entirely.

### ALLIE — ICLR 2025

[Representation/training §3 and evaluation §4](https://proceedings.iclr.cc/paper_files/paper/2025/file/0ef1afa0daa888d695dcd5e9513bafa3-Paper-Conference.pdf).
ALLIE represents whole chess moves as tokens and trains a decoder with policy,
outcome-value, and thinking-time heads on human game histories. Predicted thinking
time allocates an adaptive search budget. Its objective includes matching human
strength and behavior, rather than maximizing engine strength alone.

Application: finite-action heads and auxiliary values are compatible with an
autoregressive backbone. Do not import the human-time allocation target without
corresponding Go data, or confuse stronger human-move matching with stronger
play. Our categorical head already avoids generating several text tokens for
one action; history reconstruction remains explicit and verifiable externally.

### Spatial Language Modeling — October 2026

[Method §3 and controlled training variants §4](https://arxiv.org/html/2610.12172).
This Push-T robotics study trains from scratch using spatial coordinate tokens.
Random interaction data supervises action-conditioned state predictions while
masking random action targets from the loss. Expert data then trains actions
and future states jointly; inference emits only executable actions. The reported
comparisons support transition supervision in this setting, but use limited
robot trials and differing observation-history budgets across baselines.

Application: Go offers cheap exact transition labels without a simulator-learning
requirement. Random legal actions can provide auxiliary capture/state targets,
but must not become imitation targets for the move policy. Treat this as a
transfer hypothesis, not board-game evidence or proof of language-pretraining
benefits.

## Next bounded experiments

1. **Global-query spatial head.** Keep causal Qwen unchanged. Compute one query
   from the final whole-board token, shared keys from each cell representation,
   and a dot product per intersection; use separate pass/value heads. This gives
   each action access to global context while retaining local correspondence.
   A merely additive linear global term cancels between cell logits, so a
   bilinear interaction or nonlinear joint MLP is necessary. Compare against
   last-token, causal-cell, and bidirectional-cell readouts with matched inputs.
2. **Train-only transition supervision.** Feed a sampled legal action with a
   position and predict captures or next-board cells from verified transitions.
   Measure capture-sensitive accuracy separately from overall cell accuracy,
   which is dominated by unchanged cells. Mask random-action imitation loss.
   Compare equal numbers of policy examples and report total compute.
3. **Teacher quality before expensive search imitation.** Fix positions and
   compare raw prior, low-budget visits, and a stronger teacher. Record both
   target divergence and successor score regret. Increasing maxVisits does not
   turn `prior_mix=1` into a searched policy target. Keep final evaluation's
   stronger teacher separate from label selection.
4. **Cheap knowledge checks before long RL.** Require held-out loss better than
   uniform legal and training-frequency baselines, useful capture/endgame
   diagnostics, and legal complete games. Then compare policy improvement with
   trusted terminal rewards across paired colors and multiple opponents. A
   handful of protocol games validates integration, not strength.

These are proposals with explicit falsification criteria. The corresponding
large training and match studies remain user-run experiments.
