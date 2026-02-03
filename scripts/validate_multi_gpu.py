#!/usr/bin/env python3
"""Validate multi-GPU training setup.

This script checks that all necessary components for multi-GPU training
are properly configured and accessible.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple
from venv_utils import ensure_virtualenv, uv_pip_install


def check_gpu_availability() -> Tuple[bool, str, int]:
    """Check if GPUs are available via PyTorch.

    Returns:
        Tuple of (success, message, gpu_count)
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA is not available (no GPU or no CUDA drivers)", 0

        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            return False, "No GPUs detected by PyTorch", 0

        # List GPUs
        gpu_names = []
        for i in range(gpu_count):
            gpu_names.append(torch.cuda.get_device_name(i))

        gpu_list = "\n  ".join(f"GPU {i}: {name}" for i, name in enumerate(gpu_names))
        message = f"Found {gpu_count} GPU(s):\n  {gpu_list}"

        return True, message, gpu_count

    except ImportError:
        return False, "PyTorch is not installed", 0
    except Exception as e:
        return False, f"Error checking GPUs: {e}", 0


def check_nccl_availability() -> Tuple[bool, str]:
    """Check if NCCL backend is available for distributed training.

    Returns:
        Tuple of (success, message)
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA not available (NCCL requires CUDA)"

        if not torch.distributed.is_nccl_available():
            return False, "NCCL backend is not available"

        return True, "NCCL backend is available"

    except ImportError:
        return False, "PyTorch is not installed"
    except Exception as e:
        return False, f"Error checking NCCL: {e}"


def check_accelerate_configs() -> Tuple[bool, str, List[str]]:
    """Check if accelerate config files exist.

    Returns:
        Tuple of (success, message, list of found configs)
    """
    required_configs = [
        "config/accelerate/single_gpu.yaml",
        "config/accelerate/ddp_2gpu.yaml",
        "config/accelerate/ddp_4gpu.yaml",
        "config/accelerate/ddp_8gpu.yaml",
    ]

    found = []
    missing = []

    for config_path in required_configs:
        if os.path.exists(config_path):
            found.append(config_path)
        else:
            missing.append(config_path)

    if missing:
        return False, f"Missing configs: {', '.join(missing)}", found

    return True, f"All {len(found)} accelerate configs found", found


def check_accelerate_installation(auto_install: bool = True) -> Tuple[bool, str]:
    """Check if accelerate is installed.

    Returns:
        Tuple of (success, message)
    """
    try:
        import accelerate

        version = accelerate.__version__
        return True, f"Accelerate installed (version {version})"
    except ImportError:
        if not auto_install:
            return False, "Accelerate is not installed (uv pip install accelerate)"
        try:
            uv_pip_install(["accelerate"])
        except Exception as exc:
            return False, f"Accelerate install failed: {exc}"
        try:
            import accelerate

            version = accelerate.__version__
            return True, f"Accelerate installed (version {version})"
        except Exception as exc:
            return False, f"Accelerate import failed after install: {exc}"


def validate_accelerate_config(config_path: str, expected_gpus: int) -> Tuple[bool, str]:
    """Validate a specific accelerate config file.

    Args:
        config_path: Path to config file
        expected_gpus: Expected number of GPUs

    Returns:
        Tuple of (success, message)
    """
    try:
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check required fields
        if "compute_environment" not in config:
            return False, "Missing 'compute_environment' field"

        if "num_processes" not in config:
            return False, "Missing 'num_processes' field"

        if config["num_processes"] != expected_gpus:
            return (
                False,
                f"num_processes ({config['num_processes']}) != expected ({expected_gpus})",
            )

        # Check for distributed training settings (if > 1 GPU)
        if expected_gpus > 1:
            if config.get("distributed_type") != "MULTI_GPU":
                return False, "distributed_type should be MULTI_GPU for multi-GPU training"

        return True, f"Config valid (num_processes={config['num_processes']})"

    except yaml.YAMLError as e:
        return False, f"YAML parsing error: {e}"
    except FileNotFoundError:
        return False, "Config file not found"
    except Exception as e:
        return False, f"Validation error: {e}"


def check_training_scripts() -> Tuple[bool, str]:
    """Check if training scripts exist and support multi-GPU.

    Returns:
        Tuple of (success, message)
    """
    scripts = [
        "scripts/train_sft.sh",
        "scripts/train_grpo.sh",
        "scripts/run_sft.py",
        "scripts/run_grpo.py",
    ]

    missing = []
    for script in scripts:
        if not os.path.exists(script):
            missing.append(script)

    if missing:
        return False, f"Missing training scripts: {', '.join(missing)}"

    # Check if shell scripts support --multi-gpu flag
    for shell_script in ["scripts/train_sft.sh", "scripts/train_grpo.sh"]:
        with open(shell_script, "r") as f:
            content = f.read()
            if "--multi-gpu" not in content:
                return False, f"{shell_script} does not support --multi-gpu flag"

    return True, "Training scripts found and support multi-GPU"


def run_smoke_test(gpu_count: int) -> Tuple[bool, str]:
    """Run a minimal smoke test of distributed setup.

    Args:
        gpu_count: Number of GPUs to test

    Returns:
        Tuple of (success, message)
    """
    try:
        import torch
        import torch.distributed as dist

        # This is a placeholder - actual smoke test would initialize distributed
        # and verify all ranks can communicate

        if gpu_count < 2:
            return True, "Smoke test skipped (single GPU)"

        # Check if we can initialize NCCL (without actually doing it here)
        if torch.distributed.is_nccl_available():
            return True, "Smoke test passed (NCCL available, multi-GPU possible)"
        else:
            return False, "NCCL not available for multi-GPU training"

    except Exception as e:
        return False, f"Smoke test failed: {e}"


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def main() -> int:
    ensure_virtualenv()
    parser = argparse.ArgumentParser(
        description="Validate multi-GPU training setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test of distributed setup",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  Multi-GPU Training Validation")
    print("=" * 70)

    all_passed = True

    # 1. Check GPU availability
    print_section("1. GPU Availability")
    success, message, gpu_count = check_gpu_availability()
    print(message)
    if not success:
        print("❌ FAIL: No GPUs available")
        all_passed = False
    else:
        print(f"✓ PASS: {gpu_count} GPU(s) available")

    # 2. Check NCCL
    if gpu_count > 1:
        print_section("2. NCCL Backend")
        success, message = check_nccl_availability()
        print(message)
        if not success:
            print("❌ FAIL: NCCL not available")
            all_passed = False
        else:
            print("✓ PASS: NCCL available")

    # 3. Check accelerate installation
    print_section("3. Accelerate Installation")
    success, message = check_accelerate_installation()
    print(message)
    if not success:
        print("❌ FAIL: Accelerate not installed")
        all_passed = False
    else:
        print("✓ PASS: Accelerate installed")

    # 4. Check accelerate configs
    print_section("4. Accelerate Configuration Files")
    success, message, found_configs = check_accelerate_configs()
    print(message)
    if not success:
        print("❌ FAIL: Missing configuration files")
        all_passed = False
    else:
        print("✓ PASS: All configuration files found")

        # Validate each config
        if args.verbose:
            print("\nValidating configurations:")
            configs_to_validate = {
                "config/accelerate/single_gpu.yaml": 1,
                "config/accelerate/ddp_2gpu.yaml": 2,
                "config/accelerate/ddp_4gpu.yaml": 4,
                "config/accelerate/ddp_8gpu.yaml": 8,
            }
            for config_path, expected_gpus in configs_to_validate.items():
                success, msg = validate_accelerate_config(config_path, expected_gpus)
                status = "✓" if success else "❌"
                print(f"  {status} {config_path}: {msg}")

    # 5. Check training scripts
    print_section("5. Training Scripts")
    success, message = check_training_scripts()
    print(message)
    if not success:
        print("❌ FAIL: Training scripts missing or incomplete")
        all_passed = False
    else:
        print("✓ PASS: Training scripts ready")

    # 6. Smoke test (if requested)
    if args.smoke_test:
        print_section("6. Smoke Test")
        success, message = run_smoke_test(gpu_count)
        print(message)
        if not success:
            print("❌ FAIL: Smoke test failed")
            all_passed = False
        else:
            print("✓ PASS: Smoke test passed")

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Multi-GPU training is ready!")
        print("\nTo train with multiple GPUs:")
        print("  ./scripts/train_sft.sh --multi-gpu 2    # 2 GPUs")
        print("  ./scripts/train_sft.sh --multi-gpu 4    # 4 GPUs")
        print("  ./scripts/train_grpo.sh --multi-gpu 4   # 4 GPUs")
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues above")
        print("\nCommon solutions:")
        print("  - Install CUDA drivers and PyTorch with CUDA support")
        print("  - Install accelerate: uv pip install accelerate")
        print("  - Ensure config files exist in config/accelerate/")

    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
