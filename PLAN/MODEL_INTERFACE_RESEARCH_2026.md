# Model interface and experiment design — October 2026

These notes record methods read during this session and the resulting hypotheses.
They supplement the RL, data, and engine reading records. Cross-domain findings
motivate experiments; none establishes that a 0.6B language model already plays
strong 19×19 Go. Versioned primary sources are linked below.

## Papers and implementation decisions

1. **Qwen3 technical report**, §§4.1–4.5. The small-model recipe includes strong-to-weak distillation with both off-policy and on-policy stages. The thinking/non-thinking interface is trained explicitly. We pin Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, disable thinking for the coordinate baseline, and compare teacher warm-start with subsequent learner trajectories. We do not assume its mathematical reasoning transfers to Go. [Report](https://arxiv.org/html/2505.09388v1).

2. **Grandmaster-Level Chess Without Search**, §§2.2–2.3. A fixed cell representation and discrete output distributions avoid unconstrained language generation. Its enormous engine-labeled corpus matters: a compact model alone is not the finding. We implement a 362-action policy and side-to-move value head on Qwen, compare fixed-cell and textual observations, and measure teacher-policy cross-entropy independently of syntax. This is a different architecture and training scale. [Paper](https://arxiv.org/html/2402.04494v2).

3. **Resource-Efficient Model-Free RL for Board Games / KLENT**, §§4.1–4.3, 5.3. Finite action spaces allow exact KL/entropy-regularized policy improvement targets and action sums. The paper trains policy/Q functions with signed game returns and includes fullsize Go. We implement and analytically test the detached target operator and signed advantages, plus actual complete-turn PPO collection. A KataGo-derived target is teacher guidance, not a reproduction of model-free KLENT. [Paper](https://arxiv.org/html/2602.10894v1).

4. **SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning**, §3. The shared policy learns across alternating player roles and generated trajectories. This motivates training all collected decisions with perspective-correct returns, not only a root completion rewarded by a long continuation. Its evaluated games and learning horizon differ substantially from 19×19 Go. The repository now records behavior probabilities, both colors, caps, and adjudication provenance. [Paper](https://arxiv.org/html/2506.24119v3).

5. **Training Large Language Models to Reason in a Continuous Latent Space / Coconut**, §3. Hidden representations are fed back as continuous thought, with staged supervised adaptation. Latent steps still require sequential computation and suitable training targets; this is not free reasoning. We defer a latent-loop architecture until a measured direct-move baseline exists. Increasing latent depth would be an explicit compute-matched arm, not a presumed quality gain. [Paper](https://arxiv.org/html/2412.06769v4).

6. **SPIRAL (search-and-aggregation, 2026)**, §§3.1–3.2. This distinct paper trains multiple search traces together with an aggregation stage and final-result rewards. It motivates testing how compute is divided among candidates versus a single long trace. It does not justify treating independent Go moves from unrelated states as one GRPO group. A shared finite action distribution and separate fixed-budget evaluator provide a cheaper initial coverage measurement. [Paper](https://arxiv.org/html/2606.23595v1).

7. **Constrained Decoding via MCMC**, §§2.1, 3.1. Locally masking next tokens generally differs from sampling the original language model conditioned on an entire valid string. We document this distinction in `MoveTokenTrie` and use a single masked categorical action distribution for PPO, with actual behavior log probabilities. Legal-trie text decoding remains a useful interface benchmark but is not described as globally conditioned inference. [Paper](https://arxiv.org/html/2506.05754v1).

8. **LoGos**, §§2–4 and evaluation framing. The system uses extensive Go annotations, candidate/value/variation supervision, and staged training before RL. Its board representation findings motivate explicit matrices and complete state preservation. The scale and teacher contribution matter; copying the reward name without its data regime would not reproduce the result. We retain teacher provenance, full game histories, and independent match evaluation. [Paper](https://arxiv.org/html/2601.16447v1).

9. **LLM2Vec**, §§2.1–2.2 and 4.1. Removing a decoder's causal mask gives token representations access to future input, but the paper explicitly observes that naive conversion can worsen representations and uses adaptation objectives. We implement an experimental bidirectional Qwen board encoder trained with Go policy/value labels. This is only an inspired ablation: we do not implement or claim the paper's masked-next-token and contrastive recipe. Targets are never placed in the encoder input. [Paper](https://arxiv.org/html/2404.05961v1).

## Controlled model alternatives

- Text output: Qwen chat template, thinking disabled, exact coordinate completion;
  raw and legal-trie decoding measured separately. Text SFT trains only answer
  and EOS positions. Its reduced vocabulary projection is numerically tested
  against full causal LM loss and gradients on a real tiny Qwen.
- Last-token categorical output: one transformer pass, 362 action logits, legal
  mask, and a value scalar. No engine selects the model's action.
- Fixed cells: exactly one existing vocabulary token per intersection, with rules,
  komi, player, and eight recent moves in the prefix. The board is never truncated.
- Spatial readout: one shared linear scorer at each cell plus a separate pass
  score. Compare causal and bidirectional attention explicitly; the causal arm's
  early-cell representations lack later-board context, an intentional limitation
  measured by a gradient regression test.

Full superko history lives in the rules state, and exact legal support is supplied
at both collection and optimization. Recent-move text is context, not a complete
rules implementation. The model still chooses quality among legal actions.

## Measurements and early negative results

The first global-head run reduced held-out policy CE from 6.234 to 5.294, but a
uniform legal policy scores 5.205. This is not evidence of useful generalization.
The next runs reduce head initialization variance and compare spatial readouts.
All experiments record seeds, immutable model revision, data hashes, game counts,
wall time, and non-model baselines. Development and final-test games are isolated
before augmentation, including visible-state symmetry checks.

Text NLL includes syntax and EOS learning and is not directly comparable to a
categorical action cross-entropy. Teacher top-1 agreement is not win rate. Dense
prior/search mixtures and a searched best move are different label definitions;
`policy_argmax` supplies the hard-label control for the same dense distribution.
The final quality check must use fresh fixed-budget engine evaluations of selected
moves and complete color-balanced games, with uncertainty and cap rates visible.
