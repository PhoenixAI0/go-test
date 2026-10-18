from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

from .config import VALID_QUANTIZATION_BITS

logger = logging.getLogger(__name__)


def quantization_kwargs(bits: int) -> dict[str, bool]:
    if bits not in VALID_QUANTIZATION_BITS:
        raise ValueError(f"quantization_bits must be one of {VALID_QUANTIZATION_BITS}.")
    return {"load_in_4bit": bits == 4, "load_in_8bit": bits == 8}


def add_common_training_args(parser: argparse.ArgumentParser) -> None:
    """Options shared by the SFT and GRPO stage scripts."""
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--log-level", default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases reporting.")
    parser.add_argument("--wandb-project", help="Weights & Biases project name.")
    parser.add_argument("--wandb-entity", help="Weights & Biases entity/team.")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                        help="Weights & Biases mode.")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing for memory efficiency")
    parser.add_argument("--attention-backend", choices=["sdpa", "flash_attention_2", "auto"],
                        default="auto", help="Attention backend (auto=SDPA)")
    parser.add_argument("--enable-compile", action="store_true",
                        help="Enable torch.compile in the trainer")


def default_num_workers() -> int:
    """Compute optimal number of dataloader workers based on CPU count."""
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count // 2 or 1))


def get_distributed_context() -> tuple[int, int, bool]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return local_rank, world_size, int(os.environ.get("RANK", str(local_rank))) == 0


def configure_torch_for_speed(
    enable_compile: bool = False,
    attention_backend: Optional[str] = None,
) -> None:
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not available, skipping torch configuration")
        return

    if not torch.cuda.is_available():
        logger.info("CUDA not available, skipping GPU optimizations")
        return

    # Enable TF32 for float32 matrix multiplications on supported GPUs.
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if attention_backend:
        backend_messages = {
            "sdpa": "Using SDPA attention backend (fastest on new GPUs)",
            "flash_attention_2": "Using Flash Attention 2 backend",
            "xformers": "Using Xformers attention backend (faster, less memory)",
        }
        message = backend_messages.get(attention_backend)
        if message:
            logger.info(message)
        else:
            logger.warning(
                "Unknown attention backend: %s, using default", attention_backend
            )
    else:
        logger.info("Using auto-selected attention backend (SDPA on new GPUs)")

    if enable_compile:
        if hasattr(torch, "compile"):
            logger.info("torch.compile requested for the trainer")
        else:
            logger.warning("torch.compile not available in this PyTorch version")

    logger.info("PyTorch optimization complete: TF32, cuDNN benchmark, high precision matmul enabled")


def get_optimal_attention_backend() -> Optional[str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        # SDPA ships with torch; do not select an uninstalled flash-attn package.
        return "sdpa"

    except (AttributeError, ImportError, RuntimeError) as exc:
        logger.warning(
            "Failed to detect GPU for attention backend selection: %s", exc
        )
        return None


def render_training_prompt(example: dict, tokenizer: object, reasoning_cfg: object,
                           model_name: str) -> dict[str, str]:
    """Keep the original board for scoring while matching SFT's chat format."""
    from .reasoning import build_chat_template_kwargs, detect_reasoning_profile

    profile = detect_reasoning_profile(model_name, tokenizer, reasoning_cfg.profile)
    prompt = example["prompt"]
    return {
        "board_prompt": prompt,
        "prompt": tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True,
            **build_chat_template_kwargs(reasoning_cfg, profile),
        ),
    }


def precision_kwargs(bits: int = 4) -> dict[str, bool]:
    import torch

    mixed = bits != 32 and torch.cuda.is_available()
    bf16 = mixed and torch.cuda.is_bf16_supported()
    return {"bf16": bf16, "fp16": mixed and not bf16}


def load_training_model(config: object, *, attention_backend: str | None = None,
                        gradient_checkpointing: bool = False):
    """Load a base model or continue its saved adapter on the local training device.

    An adapter checkpoint always reloads its original base, then restores trainable
    adapter weights. Wrapping it with a new LoRA would discard the curriculum.
    """
    from pathlib import Path

    import torch
    from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from transformers.utils.peft_utils import find_adapter_config_file

    flags = quantization_kwargs(config.quantization_bits)
    quantized = any(flags.values())
    use_lora = getattr(config, "use_lora", True)
    if quantized and not use_lora:
        raise ValueError("4/8-bit training requires use_lora: true.")
    cuda = torch.cuda.is_available()
    if quantized and not cuda:
        raise ValueError("4/8-bit training requires CUDA; use quantization_bits: 0 for CPU tests.")
    set_seed(config.seed)
    dtype = torch.float32
    if cuda:
        torch.cuda.set_device(get_distributed_context()[0])
        if config.quantization_bits != 32:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    adapter_file = find_adapter_config_file(config.model_name)
    adapter = PeftConfig.from_pretrained(config.model_name) if adapter_file else None
    if adapter and not use_lora:
        raise ValueError("An adapter checkpoint requires use_lora: true.")
    base_name = adapter.base_model_name_or_path if adapter else config.model_name
    if Path(base_name).is_dir():
        base_name = str(Path(base_name).resolve())
    kwargs = {"torch_dtype": dtype, "attn_implementation": attention_backend or "sdpa"}
    if cuda:
        # Never shard a training replica with device_map='auto'.
        kwargs["device_map"] = {"": get_distributed_context()[0]}
    if quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            **flags, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(base_name, **kwargs)
    if config.max_seq_length > getattr(model.config, "max_position_embeddings", float("inf")):
        raise ValueError("max_seq_length exceeds the model's context window.")
    tokenizer_name = config.model_name
    if adapter and Path(config.model_name).is_dir() and not (Path(config.model_name) / "tokenizer_config.json").is_file():
        tokenizer_name = base_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer needs a pad_token or eos_token for batched training.")
        tokenizer.pad_token = tokenizer.eos_token
    if quantized:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    if adapter:
        model = PeftModel.from_pretrained(model, config.model_name, is_trainable=True)
    elif use_lora:
        model = get_peft_model(model, LoraConfig(
            r=config.lora_rank, lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout, target_modules="all-linear",
            bias="none", task_type="CAUSAL_LM",
        ))
    model.config.use_cache = not gradient_checkpointing
    return model, tokenizer
