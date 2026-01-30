# Codebase State Summary

This captures the current state of the Go RL training repository and separates
what is usable, what is environment-sensitive, and what remains unimplemented.

## Completely Usable
- Board logic + GTP conversion in `go_rl/board.py` (covered by `tests/test_board.py`).
- Prompt parsing/building in `go_rl/prompting.py` (covered by `tests/test_prompting.py`).
- SGF parsing + dataset prep in `go_rl/sgf_utils.py`, `go_rl/data_prep.py`,
  and scripts `scripts/build_sft_dataset.py`, `scripts/build_rl_prompts.py`,
  `scripts/prepare_datasets.py` (covered by `tests/test_sgf_utils.py` and
  `tests/test_data_prep.py`).
- Reward math and basic legality checks in `go_rl/rewards.py` (covered by
  `tests/test_rewards.py`).
- Logging/config helpers in `go_rl/logging_utils.py`, `go_rl/run_logging.py`,
  `go_rl/config.py`, `go_rl/training_utils.py`.

## Flaky / Environment-Sensitive / Untested
- GRPO training depends on live KataGo analysis and external binaries/models;
  integration coverage is not present.
  - `go_rl/katago.py`, `scripts/run_grpo.py`, `train_grpo.sh`,
    `config/grpo_9x9.yaml`.
- Multi-GPU support is wired via `accelerate` configs and shell flags but is
  not validated end-to-end; reward evaluation assumes rank-0 execution.
  - `config/accelerate/*.yaml`, `train_sft.sh`, `train_grpo.sh`,
    `scripts/run_grpo.py`.
- 19x19 SFT uses the 9x9 config with a warning; no dedicated 19x19 SFT config
  exists yet.
  - `train_sft.sh`, `config/sft_warm_start.yaml`.
- Strict legality checks are simplified (basic ko check; no superko or full
  ruleset fidelity).
  - `go_rl/board.py`.

## Unimplemented / Unusable Out of Box
- Elo evaluation requires an external match config not in the repo.
  - `scripts/eval_elo.py`.
- `download_sample_sgf.sh` provides guidance only; it does not download SGFs.
- No self-play data generation or model export/deployment scripts are present.

## Notes
- Training requires SGF data, a Linux + NVIDIA GPU environment, and KataGo
  binaries/models/configs provided externally.
- Unit tests exist for core logic, but there are no integration tests for
  training or KataGo analysis.
