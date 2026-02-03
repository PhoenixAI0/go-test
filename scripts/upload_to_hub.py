#!/usr/bin/env python3
"""Upload trained model to HuggingFace Hub.

This script merges LoRA weights into the base model and uploads to HuggingFace Hub
for easy sharing and deployment.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from venv_utils import ensure_virtualenv, uv_pip_install


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
  python scripts/upload_to_hub.py \\
      --model outputs/sft_warm_start \\
      --repo-id username/go-rl-9x9-sft \\
      --board-size 9

  # Upload GRPO model with custom settings
  python scripts/upload_to_hub.py \\
      --model outputs/grpo_19x19/final \\
      --repo-id username/go-rl-19x19-grpo \\
      --board-size 19 \\
      --private

Prerequisites:
  1. Install huggingface_hub: uv pip install huggingface_hub
  2. Login: huggingface-cli login
  3. Verify: huggingface-cli whoami
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to trained model checkpoint (LoRA adapter directory)",
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
        default="qwen-3-8b",
        help="Base model name (default: qwen-3-8b)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository",
    )
    parser.add_argument(
        "--merge-and-upload",
        action="store_true",
        help="Merge LoRA into base model before uploading",
    )
    parser.add_argument(
        "--push-only",
        action="store_true",
        help="Only push (don't merge), useful for LoRA adapters",
    )
    parser.add_argument(
        "--model-card",
        help="Path to custom README.md for model card",
    )
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

    # Import required libraries
    try:
        from huggingface_hub import HfApi, create_repo
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        logger.warning(f"Missing required library: {e}")
        logger.info("Installing with uv...")
        try:
            uv_pip_install(["huggingface_hub", "transformers"])
        except Exception as install_error:
            logger.error(f"Dependency install failed: {install_error}")
            return 1
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
        logger.info(f"Repository created (or already exists)")
    except Exception as e:
        logger.error(f"Failed to create repository: {e}")
        return 1

    # Handle LoRA adapter upload
    if is_lora and args.push_only:
        logger.info("Uploading LoRA adapter (push-only mode)")

        try:
            api = HfApi()

            # Upload adapter files
            logger.info("Uploading adapter files...")
            api.upload_folder(
                folder_path=str(model_path),
                repo_id=args.repo_id,
                repo_type="model",
            )

            # Generate and upload model card
            model_card = generate_model_card(
                args.repo_id,
                args.base_model,
                args.board_size,
                is_lora=True,
                custom_card_path=args.model_card,
            )

            readme_path = model_path / "README.md"
            readme_path.write_text(model_card, encoding="utf-8")

            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=args.repo_id,
                repo_type="model",
            )

            logger.info(f"✓ Model uploaded to: https://huggingface.co/{args.repo_id}")
            return 0

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return 1

    # Merge LoRA and upload
    if is_lora and args.merge_and_upload:
        logger.info("Merging LoRA weights into base model...")

        try:
            # This is a placeholder - actual implementation would:
            # 1. Load base model
            # 2. Load LoRA adapter
            # 3. Merge weights
            # 4. Save merged model
            # 5. Upload to Hub

            logger.error("Merge and upload not fully implemented yet")
            logger.error("Use --push-only to upload LoRA adapter directly")
            logger.error("Or merge manually using unsloth's merge_and_unload()")
            return 1

        except Exception as e:
            logger.error(f"Merge failed: {e}")
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
            args.base_model,
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
        readme_path = Path("/tmp/README.md")
        readme_path.write_text(model_card, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
        )

        logger.info(f"\n✓ Model uploaded successfully!")
        logger.info(f"  URL: https://huggingface.co/{args.repo_id}")
        logger.info(f"\nTo use this model:")
        logger.info(f'  from transformers import AutoModelForCausalLM')
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
prompt = "Board: 9x9\\nBlack to play\\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=10)
move = tokenizer.decode(outputs[0], skip_special_tokens=True)
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
