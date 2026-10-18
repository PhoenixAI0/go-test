#!/usr/bin/env python3
"""Upload trained models or LoRA adapters to HuggingFace Hub."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from venv_utils import ensure_virtualenv

# Trainer checkpoints hold optimizer state; only the final save belongs on the Hub.
UPLOAD_IGNORE_PATTERNS = ["checkpoint-*", "runs/"]
DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(levelname)s: %(message)s",
    )


def main() -> int:
    ensure_virtualenv()
    parser = argparse.ArgumentParser(
        description="Upload trained model to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload SFT model
  uv run python scripts/upload_to_hub.py \\
      --model outputs/sft_warm_start \\
      --repo-id username/go-rl-9x9-sft \\
      --board-size 9

  # Upload GRPO model with custom settings
  uv run python scripts/upload_to_hub.py \\
      --model outputs/grpo_19x19/final \\
      --repo-id username/go-rl-19x19-grpo \\
      --board-size 19 \\
      --private

Prerequisites:
  1. Login: huggingface-cli login
  2. Verify: huggingface-cli whoami
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to trained model checkpoint or LoRA adapter directory",
    )
    parser.add_argument(
        "--repo-id",
        "-r",
        required=True,
        help="HuggingFace repository ID (username/repo-name)",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        required=True,
        choices=[9, 13, 19],
        help="Board size the model was trained on",
    )
    parser.add_argument(
        "--base-model",
        help=f"Base model ID (default: from adapter_config.json, else {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository",
    )
    parser.add_argument(
        "--push-only",
        action="store_true",
        help="Required when uploading a LoRA adapter",
    )
    parser.add_argument("--model-card", help="Path to custom README.md for model card")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model path not found: {model_path}")
        return 1

    # Check for adapter files (LoRA)
    is_lora = (model_path / "adapter_config.json").exists()
    logger.info(f"Model type: {'LoRA adapter' if is_lora else 'Full model'}")

    base_model = args.base_model
    if base_model is None and is_lora:
        adapter_config = json.loads((model_path / "adapter_config.json").read_text(encoding="utf-8"))
        base_model = adapter_config.get("base_model_name_or_path")
    base_model = base_model or DEFAULT_BASE_MODEL

    from huggingface_hub import HfApi, create_repo
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Create repository
    logger.info(f"Creating repository: {args.repo_id}")
    try:
        create_repo(
            repo_id=args.repo_id,
            private=args.private,
            exist_ok=True,
        )
        logger.info("Repository created (or already exists)")
    except Exception as e:
        logger.error(f"Failed to create repository: {e}")
        return 1

    # Handle LoRA adapter upload
    if is_lora:
        if not args.push_only:
            logger.error("LoRA adapters must be uploaded with --push-only.")
            return 1

        logger.info("Uploading LoRA adapter (push-only mode)")

        try:
            api = HfApi()

            logger.info("Uploading adapter files...")
            api.upload_folder(
                folder_path=str(model_path),
                repo_id=args.repo_id,
                repo_type="model",
                ignore_patterns=UPLOAD_IGNORE_PATTERNS,
            )

            model_card = generate_model_card(
                args.repo_id,
                base_model,
                args.board_size,
                is_lora=True,
                custom_card_path=args.model_card,
            )
            api.upload_file(
                path_or_fileobj=model_card.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=args.repo_id,
                repo_type="model",
            )

            logger.info(f"✓ Model uploaded to: https://huggingface.co/{args.repo_id}")
            return 0

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return 1

    # Upload full model
    logger.info("Uploading model to HuggingFace Hub...")
    try:
        # Load model and tokenizer
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Generate model card
        model_card = generate_model_card(
            args.repo_id,
            base_model,
            args.board_size,
            is_lora=False,
            custom_card_path=args.model_card,
        )

        # Push to hub
        logger.info("Pushing to Hub...")
        model.push_to_hub(args.repo_id, private=args.private)
        tokenizer.push_to_hub(args.repo_id, private=args.private)

        # Upload model card
        api = HfApi()
        api.upload_file(
            path_or_fileobj=model_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
        )

        logger.info("\n✓ Model uploaded successfully!")
        logger.info(f"  URL: https://huggingface.co/{args.repo_id}")
        logger.info("\nTo use this model:")
        logger.info('  from transformers import AutoModelForCausalLM')
        logger.info(f'  model = AutoModelForCausalLM.from_pretrained("{args.repo_id}")')

        return 0

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return 1


def generate_model_card(
    repo_id: str,
    base_model: str,
    board_size: int,
    is_lora: bool = False,
    custom_card_path: str | None = None,
) -> str:
    """Generate model card (README.md) for HuggingFace Hub.

    Args:
        repo_id: Repository ID
        base_model: Base model name
        board_size: Board size
        is_lora: Whether this is a LoRA adapter
        custom_card_path: Path to custom model card

    Returns:
        Model card content
    """
    if custom_card_path:
        return Path(custom_card_path).read_text(encoding="utf-8")

    adapter_type = "LoRA Adapter" if is_lora else "Full Model"

    return f"""---
license: apache-2.0
tags:
- go
- board-games
- reinforcement-learning
- grpo
{'- peft' if is_lora else ''}
- {base_model}
base_model: {base_model}
---

# {repo_id}

This is a {adapter_type.lower()} for playing {board_size}x{board_size} Go, trained using GRPO (Group Relative Policy Optimization).

## Model Details

- **Base Model**: {base_model}
- **Board Size**: {board_size}x{board_size}
- **Training Method**: GRPO with KataGo rewards
- **Type**: {adapter_type}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Generate move
prompt = \"\"\"board_size: 9
to_play: B
board:
.........
.........
.........
.........
.........
.........
.........
.........
.........
move:\"\"\"
messages = [{{"role": "user", "content": prompt}}]
if hasattr(tokenizer, "apply_chat_template"):
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
else:
    inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=10)
prompt_len = int(inputs["input_ids"].shape[-1])
move = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
```

{f'''
## LoRA Adapter Usage

Since this is a LoRA adapter, you need to load it with the base model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base_model, "{repo_id}")
```
''' if is_lora else ''}

## Training

This model was trained using the Go RL framework with:
- Supervised fine-tuning on professional games
- GRPO training with KataGo-based rewards
- Reward shaping for legal moves, game outcomes, and winrate improvement

## Limitations

- Trained specifically for {board_size}x{board_size} board size
- May generate illegal moves (validate before playing)
- Performance depends on temperature and sampling parameters

## Citation

```bibtex
@misc{{go-rl-{board_size}x{board_size},
  author = {{Your Name}},
  title = {{Go RL Model for {board_size}x{board_size}}},
  year = {{2024}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```

## License

Apache 2.0
"""


if __name__ == "__main__":
    sys.exit(main())
