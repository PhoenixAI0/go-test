import os
from unittest.mock import mock_open, patch

import pytest

from go_rl.config import (
    FullGameConfig,
    KataGoConfig,
    RewardConfig,
    RolloutConfig,
    SftConfig,
    TrainingConfig,
    _expand_path,
    _load_yaml,
    _validate_path,
    load_config,
)


def test_expand_path_expanduser():
    """Test that ~ is expanded to home directory."""
    path = "~/models/kata.bin.gz"
    expanded = _expand_path(path)
    assert not expanded.startswith("~")
    assert expanded.startswith(os.path.expanduser("~"))


def test_expand_path_expandvars():
    """Test that environment variables are expanded."""
    os.environ["TEST_VAR"] = "/test/path"
    path = "$TEST_VAR/models/kata.bin.gz"
    expanded = _expand_path(path)
    assert expanded.startswith("/test/path")
    del os.environ["TEST_VAR"]


def test_validate_path_rejects_placeholder():
    """Test that placeholder paths are rejected."""
    with pytest.raises(ValueError, match="must be set to a real path"):
        _validate_path("/path/to/katago", "binary_path")


def test_validate_path_rejects_unexpanded_vars():
    """Test that unexpanded variable paths are rejected."""
    with pytest.raises(ValueError, match="must be set to a real path"):
        _validate_path("$HOME/katago", "binary_path")


def test_validate_path_accepts_real_paths():
    """Test that real paths pass validation."""
    # Should not raise
    _validate_path("/usr/bin/katago", "binary_path")
    _validate_path("/home/user/models/kata.bin.gz", "model_path")


def test_katago_config_creates_with_defaults():
    config = KataGoConfig(
        binary_path="/usr/bin/katago",
        model_path="/models/kata.bin.gz",
        config_path="/config/gtp.cfg",
    )
    assert config.binary_path == "/usr/bin/katago"
    assert config.model_path == "/models/kata.bin.gz"
    assert config.config_path == "/config/gtp.cfg"
    assert config.board_size == 9
    assert config.rules == "chinese"
    assert config.komi == 7.5


def test_katago_config_custom_values():
    config = KataGoConfig(
        binary_path="/usr/bin/katago",
        model_path="/models/kata.bin.gz",
        config_path="/config/gtp.cfg",
        board_size=19,
        rules="japanese",
        komi=6.5,
        katago_cuda_visible_devices="1",
        request_timeout_seconds=12.0,
        katago_pool_size=2,
    )
    assert config.board_size == 19
    assert config.rules == "japanese"
    assert config.komi == 6.5
    assert config.katago_cuda_visible_devices == "1"
    assert config.request_timeout_seconds == 12.0
    assert config.katago_pool_size == 2


def test_katago_config_rejects_invalid_runtime_controls():
    with pytest.raises(ValueError, match="request_timeout_seconds"):
        KataGoConfig(
            binary_path="/usr/bin/katago",
            model_path="/models/kata.bin.gz",
            config_path="/config/gtp.cfg",
            request_timeout_seconds=0,
        )
    with pytest.raises(ValueError, match="katago_pool_size"):
        KataGoConfig(
            binary_path="/usr/bin/katago",
            model_path="/models/kata.bin.gz",
            config_path="/config/gtp.cfg",
            katago_pool_size=0,
        )


def test_katago_config_rejects_placeholder():
    with pytest.raises(ValueError, match="must be set to a real path"):
        KataGoConfig(
            binary_path="/path/to/katago",
            model_path="/models/kata.bin.gz",
            config_path="/config/gtp.cfg",
        )


def test_katago_config_expands_home():
    config = KataGoConfig(
        binary_path="~/bin/katago",
        model_path="~/models/kata.bin.gz",
        config_path="~/config/gtp.cfg",
    )
    assert not config.binary_path.startswith("~")
    assert not config.model_path.startswith("~")
    assert not config.config_path.startswith("~")


def test_katago_config_resolves_relative_paths_with_base_dir(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    config = KataGoConfig(
        binary_path="bin/katago",
        model_path="models/kata.bin.gz",
        config_path="katago_analysis_train.cfg",
        config_base_dir=str(config_dir),
    )

    assert config.binary_path == os.path.abspath(config_dir / "bin/katago")
    assert config.model_path == os.path.abspath(config_dir / "models/kata.bin.gz")
    assert config.config_path == os.path.abspath(config_dir / "katago_analysis_train.cfg")


def test_rollout_config_defaults():
    config = RolloutConfig()
    assert config.num_samples == 8
    assert config.temperature == 1.0
    assert config.top_p == 0.9
    assert config.max_tokens == 4
    assert config.stop_at_pass is True


def test_rollout_config_custom_values():
    config = RolloutConfig(
        num_samples=16,
        temperature=0.8,
        top_p=0.95,
        max_tokens=8,
        stop_at_pass=False,
    )
    assert config.num_samples == 16
    assert config.temperature == 0.8
    assert config.top_p == 0.95
    assert config.max_tokens == 8
    assert config.stop_at_pass is False


def test_reward_config_defaults():
    config = RewardConfig()
    assert config.winrate_scale == 1.0
    assert config.legal_bonus == 0.2
    assert config.illegal_penalty == -1.0
    assert config.length_penalty == -0.05
    assert config.full_game_bonus == 0.5
    assert config.scorelead_scale == 0.02
    assert config.terminal_outcome_bonus == 1.0
    assert config.illegal_move_ends_game is True
    assert config.strict_legality is False


def test_reward_config_custom_values():
    config = RewardConfig(
        winrate_scale=2.0,
        legal_bonus=0.5,
        illegal_penalty=-2.0,
        length_penalty=-0.1,
        full_game_bonus=1.0,
        scorelead_scale=0.03,
        terminal_outcome_bonus=1.5,
        illegal_move_ends_game=False,
        strict_legality=True,
    )
    assert config.winrate_scale == 2.0
    assert config.legal_bonus == 0.5
    assert config.illegal_penalty == -2.0
    assert config.length_penalty == -0.1
    assert config.full_game_bonus == 1.0
    assert config.scorelead_scale == 0.03
    assert config.terminal_outcome_bonus == 1.5
    assert config.illegal_move_ends_game is False
    assert config.strict_legality is True


def test_fullgame_config_defaults():
    config = FullGameConfig()
    assert config.enabled is True
    assert config.max_plies == 300
    assert config.termination_two_passes is True
    assert config.opponent_policy == "katago_best"
    assert config.model_color_mode == "alternate"
    assert config.seed_empty_board_prob == 0.4
    assert config.max_seed_prefix_moves == 40
    assert config.max_model_turns == 150


def test_fullgame_config_validation():
    with pytest.raises(ValueError, match="seed_empty_board_prob"):
        FullGameConfig(seed_empty_board_prob=-0.1)
    with pytest.raises(ValueError, match="max_plies"):
        FullGameConfig(max_plies=0)
    with pytest.raises(ValueError, match="max_seed_prefix_moves"):
        FullGameConfig(max_seed_prefix_moves=-1)
    with pytest.raises(ValueError, match="max_model_turns"):
        FullGameConfig(max_model_turns=0)
    with pytest.raises(ValueError, match="opponent_policy"):
        FullGameConfig(opponent_policy="random")
    with pytest.raises(ValueError, match="model_color_mode"):
        FullGameConfig(model_color_mode="green")


def test_training_config_validates_quantization_bits():
    """Test that invalid quantization_bits raises ValueError."""
    with pytest.raises(ValueError, match="quantization_bits must be one of"):
        TrainingConfig(
            model_name="gpt2",
            output_dir="/output",
            quantization_bits=3,  # Invalid
        )


@pytest.mark.parametrize("bits", [0, 4, 8, 16, 32])
def test_training_config_valid_quantization_bits(bits):
    """Test that valid quantization_bits are accepted."""
    config = TrainingConfig(
        model_name="gpt2",
        output_dir="/output",
        quantization_bits=bits,
    )
    assert config.quantization_bits == bits


def test_training_config_expands_paths():
    config = TrainingConfig(
        model_name="gpt2",
        output_dir="~/output",
        log_dir="~/logs",
    )
    assert not config.output_dir.startswith("~")
    assert not config.log_dir.startswith("~")


def test_training_config_defaults():
    config = TrainingConfig(
        model_name="gpt2",
        output_dir="/output",
    )
    assert config.log_dir == "logs/grpo"
    assert config.max_seq_length == 2048
    assert config.batch_size == 1
    assert config.gradient_accumulation == 8
    assert config.learning_rate == 1e-5
    assert config.num_steps == 2000
    assert config.seed == 3407
    assert config.lora_rank == 64
    assert config.lora_alpha == 16
    assert config.lora_dropout == 0.0
    assert config.use_lora is True
    assert config.grpo_mini_batch == 1
    assert config.grpo_logit_chunk_multiplier == 4
    assert config.use_8bit_optimizer is True
    assert config.quantization_bits == 8


def test_sft_config_validates_quantization_bits():
    """Test that invalid quantization_bits raises ValueError."""
    with pytest.raises(ValueError, match="quantization_bits must be one of"):
        SftConfig(
            dataset_path="/data/sft.jsonl",
            model_name="gpt2",
            output_dir="/output",
            quantization_bits=7,  # Invalid
        )


@pytest.mark.parametrize("bits", [0, 4, 8, 16, 32])
def test_sft_config_valid_quantization_bits(bits):
    """Test that valid quantization_bits are accepted."""
    config = SftConfig(
        dataset_path="/data/sft.jsonl",
        model_name="gpt2",
        output_dir="/output",
        quantization_bits=bits,
    )
    assert config.quantization_bits == bits


def test_sft_config_expands_paths():
    config = SftConfig(
        dataset_path="~/data/sft.jsonl",
        model_name="gpt2",
        output_dir="~/output",
        log_dir="~/logs",
    )
    assert not config.dataset_path.startswith("~")
    assert not config.output_dir.startswith("~")
    assert not config.log_dir.startswith("~")


def test_sft_config_defaults():
    config = SftConfig(
        dataset_path="/data/sft.jsonl",
        model_name="gpt2",
        output_dir="/output",
    )
    assert config.log_dir == "logs/sft"
    assert config.max_seq_length == 1024
    assert config.epochs == 1
    assert config.batch_size == 4
    assert config.learning_rate == 5e-5
    assert config.lora_rank == 64
    assert config.lora_alpha == 16
    assert config.lora_dropout == 0.0
    assert config.use_packing is True
    assert config.use_8bit_optimizer is True
    assert config.quantization_bits == 8
    assert config.seed == 3407
    assert config.eval_split_ratio == 0.05
    assert config.eval_steps == 200


def test_sft_config_invalid_eval_split_ratio():
    with pytest.raises(ValueError, match="eval_split_ratio must be in"):
        SftConfig(
            dataset_path="/data/sft.jsonl",
            model_name="gpt2",
            output_dir="/output",
            eval_split_ratio=1.0,
        )


def test_sft_config_invalid_eval_steps():
    with pytest.raises(ValueError, match="eval_steps must be >= 1"):
        SftConfig(
            dataset_path="/data/sft.jsonl",
            model_name="gpt2",
            output_dir="/output",
            eval_steps=0,
        )


def test_load_yaml_success(tmp_path):
    """Test successful YAML loading."""
    yaml_content = """
model_name: gpt2
output_dir: /output
quantization_bits: 8
"""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(yaml_content)

    data = _load_yaml(str(yaml_file))
    assert data["model_name"] == "gpt2"
    assert data["output_dir"] == "/output"
    assert data["quantization_bits"] == 8


def test_load_yaml_missing_file():
    """Test that missing file raises appropriate error."""
    with pytest.raises(FileNotFoundError):
        _load_yaml("/nonexistent/config.yaml")


def test_load_yaml_not_a_dict(tmp_path):
    """Test that non-dict YAML raises ValueError."""
    yaml_content = """
- item1
- item2
"""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(yaml_content)

    with pytest.raises(ValueError, match="must be a mapping"):
        _load_yaml(str(yaml_file))


def test_load_yaml_missing_pyyaml():
    """Test that missing PyYAML dependency raises RuntimeError."""
    with patch("builtins.__import__", side_effect=ImportError("No module named 'yaml'")):
        with pytest.raises(RuntimeError, match="PyYAML is required"):
            _load_yaml("/dummy/path.yaml")


def test_load_config_training(tmp_path):
    """Test loading TrainingConfig from YAML."""
    yaml_content = """
model_name: gpt2
output_dir: /output
quantization_bits: 8
learning_rate: 0.0001
"""
    yaml_file = tmp_path / "training.yaml"
    yaml_file.write_text(yaml_content)

    config = load_config(str(yaml_file), TrainingConfig)
    assert isinstance(config, TrainingConfig)
    assert config.model_name == "gpt2"
    assert config.output_dir == "/output"
    assert config.quantization_bits == 8
    assert config.learning_rate == 0.0001


def test_load_config_sft(tmp_path):
    """Test loading SftConfig from YAML."""
    yaml_content = """
dataset_path: /data/sft.jsonl
model_name: gpt2
output_dir: /output
epochs: 3
"""
    yaml_file = tmp_path / "sft.yaml"
    yaml_file.write_text(yaml_content)

    config = load_config(str(yaml_file), SftConfig)
    assert isinstance(config, SftConfig)
    assert config.dataset_path == "/data/sft.jsonl"
    assert config.model_name == "gpt2"
    assert config.epochs == 3


def test_load_config_katago(tmp_path):
    """Test loading KataGoConfig from YAML."""
    yaml_content = """
binary_path: /usr/bin/katago
model_path: /models/kata.bin.gz
config_path: /config/gtp.cfg
board_size: 19
"""
    yaml_file = tmp_path / "katago.yaml"
    yaml_file.write_text(yaml_content)

    config = load_config(str(yaml_file), KataGoConfig)
    assert isinstance(config, KataGoConfig)
    assert config.binary_path == "/usr/bin/katago"
    assert config.board_size == 19


def test_load_config_katago_resolves_relative_paths_against_yaml_dir(tmp_path):
    config_dir = tmp_path / "nested" / "config"
    config_dir.mkdir(parents=True)
    yaml_file = config_dir / "katago.yaml"
    yaml_file.write_text(
        """
binary_path: bin/katago
model_path: models/kata.bin.gz
config_path: katago_analysis_train.cfg
board_size: 19
"""
    )

    config = load_config(str(yaml_file), KataGoConfig)
    assert isinstance(config, KataGoConfig)
    assert config.binary_path == os.path.abspath(config_dir / "bin/katago")
    assert config.model_path == os.path.abspath(config_dir / "models/kata.bin.gz")
    assert config.config_path == os.path.abspath(config_dir / "katago_analysis_train.cfg")
    assert config.board_size == 19


def test_load_config_reward(tmp_path):
    """Test loading RewardConfig from YAML."""
    yaml_content = """
winrate_scale: 2.0
legal_bonus: 0.5
strict_legality: true
"""
    yaml_file = tmp_path / "reward.yaml"
    yaml_file.write_text(yaml_content)

    config = load_config(str(yaml_file), RewardConfig)
    assert isinstance(config, RewardConfig)
    assert config.winrate_scale == 2.0
    assert config.legal_bonus == 0.5
    assert config.strict_legality is True


def test_load_config_rollout(tmp_path):
    """Test loading RolloutConfig from YAML."""
    yaml_content = """
num_samples: 16
temperature: 0.8
stop_at_pass: false
"""
    yaml_file = tmp_path / "rollout.yaml"
    yaml_file.write_text(yaml_content)

    config = load_config(str(yaml_file), RolloutConfig)
    assert isinstance(config, RolloutConfig)
    assert config.num_samples == 16
    assert config.temperature == 0.8
    assert config.stop_at_pass is False


def test_load_config_fullgame(tmp_path):
    """Test loading FullGameConfig from YAML."""
    yaml_content = """
enabled: true
max_plies: 120
model_color_mode: alternate
seed_empty_board_prob: 0.25
"""
    yaml_file = tmp_path / "fullgame.yaml"
    yaml_file.write_text(yaml_content)

    config = load_config(str(yaml_file), FullGameConfig)
    assert isinstance(config, FullGameConfig)
    assert config.enabled is True
    assert config.max_plies == 120
    assert config.seed_empty_board_prob == 0.25
