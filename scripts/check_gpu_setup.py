#!/usr/bin/env python3
"""Check GPU setup and recommend multi-GPU configuration."""

from __future__ import annotations

import sys


def main() -> None:
    try:
        import torch
    except ImportError:
        print("❌ PyTorch not installed")
        print("Please install PyTorch: pip install torch")
        sys.exit(1)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ No CUDA GPUs detected")
        print("\nPossible reasons:")
        print("  - No NVIDIA GPUs in system")
        print("  - NVIDIA drivers not installed")
        print("  - PyTorch not built with CUDA support")
        print("\nMulti-GPU training requires NVIDIA GPUs with CUDA support")
        sys.exit(1)

    # Get GPU count and details
    num_gpus = torch.cuda.device_count()
    print(f"✅ Found {num_gpus} GPU(s)\n")

    # Print GPU details
    total_vram = 0
    gpu_names = set()
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / 1024**3
        total_vram += vram_gb
        gpu_names.add(props.name)
        print(f"  GPU {i}: {props.name}")
        print(f"    VRAM: {vram_gb:.1f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
        print()

    # Check for homogeneous GPUs
    if len(gpu_names) > 1:
        print("⚠️  WARNING: Mixed GPU types detected")
        print("   For best performance, use identical GPUs")
        print("   Mixed GPUs may cause load imbalance and reduced speedup")
        print()

    # Print total VRAM
    print(f"Total VRAM: {total_vram:.1f} GB")
    print()

    # Recommend configuration
    print("=" * 60)
    print("RECOMMENDED CONFIGURATIONS")
    print("=" * 60)
    print()

    if num_gpus == 1:
        print("Single GPU Configuration:")
        print("  ./train_sft.sh")
        print("  ./train_grpo.sh")
        print()
        print("This is the default mode (no flags needed).")

    elif num_gpus == 2:
        print("Multi-GPU Configuration (2 GPUs):")
        print("  ./train_sft.sh --multi-gpu ddp-2")
        print("  ./train_grpo.sh --multi-gpu ddp-2")
        print()
        print("Expected speedup: ~1.8-1.9x faster than single GPU")
        print()
        if total_vram < 48:
            print("⚠️  Limited VRAM: Consider using --gradient-checkpointing")
            print("  ./train_sft.sh --multi-gpu ddp-2 --gradient-checkpointing")

    elif num_gpus >= 4 and num_gpus < 8:
        print("Multi-GPU Configuration (4 GPUs):")
        print("  ./train_sft.sh --multi-gpu ddp-4")
        print("  ./train_grpo.sh --multi-gpu ddp-4")
        print()
        print("Expected speedup: ~3.5-3.8x faster than single GPU")
        print()
        if total_vram < 96:
            print("⚠️  Limited VRAM: Consider using --gradient-checkpointing")
            print("  ./train_sft.sh --multi-gpu ddp-4 --gradient-checkpointing")

    elif num_gpus >= 8:
        print("Multi-GPU Configuration (8 GPUs):")
        print("  ./train_sft.sh --multi-gpu ddp-8")
        print("  ./train_grpo.sh --multi-gpu ddp-8")
        print()
        print("Expected speedup: ~6.5-7.5x faster than single GPU")
        print()
        if total_vram < 192:
            print("⚠️  Limited VRAM: Consider using --gradient-checkpointing")
            print("  ./train_sft.sh --multi-gpu ddp-8 --gradient-checkpointing")

    print()

    # Memory considerations
    print("=" * 60)
    print("MEMORY RECOMMENDATIONS")
    print("=" * 60)
    print()

    avg_vram = total_vram / num_gpus
    if avg_vram < 24:
        print("⚠️  Limited VRAM per GPU (< 24 GB)")
        print("   Recommendations:")
        print("   - Use --gradient-checkpointing flag")
        print("   - Reduce batch size in config YAML")
        print("   - Reduce max_seq_length in config YAML")
        print()
    elif avg_vram >= 40:
        print("✅ Ample VRAM per GPU (≥ 40 GB)")
        print("   You can:")
        print("   - Increase batch size for faster training")
        print("   - Train without gradient checkpointing")
        print()
    else:
        print("✅ Good VRAM per GPU (24-40 GB)")
        print("   Default settings should work well")
        print()

    # Training tips
    print("=" * 60)
    print("QUICK TIPS")
    print("=" * 60)
    print()
    print("1. Test first with --smoke flag:")
    print("   ./train_sft.sh --multi-gpu ddp-4 --smoke")
    print()
    print("2. Monitor GPU usage during training:")
    print("   watch -n 1 nvidia-smi")
    print()
    print("3. All GPUs should show >80% utilization")
    print("   If not, increase batch size or check data loading")
    print()
    print("4. See full guide: docs/MULTI_GPU_GUIDE.md")
    print()

    # CUDA version info
    print("=" * 60)
    print("SYSTEM INFO")
    print("=" * 60)
    print()
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print()


if __name__ == "__main__":
    main()
