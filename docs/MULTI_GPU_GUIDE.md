# Multi-GPU Training Guide

This guide explains how to use multiple GPUs for training Go-playing language models with this codebase.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## Overview

This codebase supports multi-GPU training using **Distributed Data Parallel (DDP)** via Hugging Face Accelerate. DDP provides near-linear speedup by replicating the model across multiple GPUs and parallelizing the training data.

### Why Multi-GPU Training?

- **Faster training**: Near-linear speedup with more GPUs (2x faster with 2 GPUs, 4x with 4 GPUs)
- **Larger effective batch size**: Process more samples per update
- **Better gradient estimates**: Improved training stability with larger batches

### Supported Configurations

- **DDP with 2 GPUs** (`ddp-2`): 2x speedup
- **DDP with 4 GPUs** (`ddp-4`): 4x speedup
- **DDP with 8 GPUs** (`ddp-8`): 8x speedup

### Unsloth Compatibility

This codebase uses **Unsloth** for efficient fine-tuning with 4-bit quantization. Key points:

- ✅ **DDP is fully supported** with Unsloth's 4-bit quantization
- ❌ **FSDP is NOT supported** by Unsloth (as of 2026)
- ⚠️ **Device map conflicts**: Don't use `device_map="balanced"` with DDP

The good news: Unsloth's 4-bit quantization is so memory-efficient that DDP works well even for 8B parameter models on 24GB GPUs.

## Quick Start

### Check Your GPUs

```bash
# Check available GPUs
nvidia-smi

# Or use the helper script
python scripts/check_gpu_setup.py
```

### Training with Multi-GPU

```bash
# SFT Training with 4 GPUs
./train_sft.sh --multi-gpu ddp-4

# GRPO Training with 2 GPUs
./train_grpo.sh --multi-gpu ddp-2

# With gradient checkpointing for memory efficiency
./train_sft.sh --multi-gpu ddp-4 --gradient-checkpointing
```

### Single GPU (Default)

```bash
# No flags needed for single GPU (backward compatible)
./train_sft.sh
./train_grpo.sh
```

## How It Works

### Distributed Data Parallel (DDP)

DDP works by:

1. **Model Replication**: Each GPU gets a full copy of the model
2. **Data Sharding**: Training data is split across GPUs
3. **Forward Pass**: Each GPU processes its batch independently
4. **Gradient Synchronization**: Gradients are averaged across all GPUs
5. **Weight Update**: All GPUs update their models with the averaged gradients

### Accelerate Integration

Under the hood, the training scripts use Hugging Face Accelerate:

```bash
# Single GPU
python scripts/run_sft.py --config config/sft_warm_start.yaml

# Multi-GPU with DDP
accelerate launch --config_file=config/accelerate/ddp_4gpu.yaml \
    scripts/run_sft.py --config config/sft_warm_start.yaml
```

The shell scripts (`train_sft.sh`, `train_grpo.sh`) handle this automatically based on the `--multi-gpu` flag.

### GRPO Multi-GPU Considerations

For GRPO training, the KataGo analysis client runs **only on the main process** (rank 0):

- Rank 0 generates rollouts, computes rewards using KataGo
- Rewards are broadcasted to other processes
- All processes participate in policy gradient updates

This design avoids conflicts from multiple processes accessing KataGo.

## Performance Tuning

### Batch Size Scaling

With DDP, you can scale your batch size linearly with GPU count:

**Single GPU baseline:**
```yaml
# config/sft_warm_start.yaml
batch_size: 4  # Per-device batch size
```

**Multi-GPU scaling:**
- 2 GPUs: Effective batch size = 4 × 2 = 8
- 4 GPUs: Effective batch size = 4 × 4 = 16
- 8 GPUs: Effective batch size = 4 × 8 = 32

**Optional: Increase per-device batch size:**
```yaml
# For 4 GPUs, you could increase to:
batch_size: 8  # Effective batch size = 8 × 4 = 32
```

### Gradient Accumulation

Gradient accumulation lets you simulate larger batch sizes:

```yaml
# config/grpo_9x9.yaml
batch_size: 1
gradient_accumulation: 8  # Effective batch size = 1 × 8 = 8 per GPU
```

**With 4 GPUs:**
- Per-GPU batch: 1 × 8 = 8
- Total effective batch: 8 × 4 = 32

You can often reduce gradient accumulation when using multiple GPUs:
```yaml
# Single GPU
batch_size: 1
gradient_accumulation: 8  # Effective: 8

# 4 GPUs - reduce accumulation
batch_size: 1
gradient_accumulation: 2  # Effective: 1 × 2 × 4 = 8 (same total)
```

### Gradient Checkpointing

Gradient checkpointing trades compute for memory:

```bash
./train_sft.sh --multi-gpu ddp-4 --gradient-checkpointing
```

**Trade-offs:**
- **Memory savings**: ~40% reduction in VRAM usage
- **Speed cost**: ~20% slower training
- **When to use**: Running out of VRAM, want larger batch sizes

### Mixed Precision

All configurations use `fp16` (16-bit floating point) by default:

```yaml
# config/accelerate/ddp_4gpu.yaml
mixed_precision: fp16
```

This provides:
- ~2x memory savings vs fp32
- ~2x faster computation on modern GPUs
- Minimal accuracy impact with proper loss scaling

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. **Enable gradient checkpointing:**
   ```bash
   ./train_sft.sh --multi-gpu ddp-4 --gradient-checkpointing
   ```

2. **Reduce batch size:**
   ```yaml
   # config/sft_warm_start.yaml
   batch_size: 2  # Reduce from 4
   ```

3. **Reduce sequence length:**
   ```yaml
   max_seq_length: 512  # Reduce from 1024
   ```

### NCCL Timeout Issues

**Symptoms:**
```
RuntimeError: NCCL operation timed out
```

**Solutions:**
1. **Increase timeout:**
   ```bash
   export NCCL_TIMEOUT=3600  # 1 hour
   ./train_sft.sh --multi-gpu ddp-4
   ```

2. **Check GPU communication:**
   ```bash
   # Verify GPUs can communicate
   nvidia-smi topo -m
   ```

### Slow Training

**Expected speedup:**
- 2 GPUs: ~1.8-1.9x faster (not quite 2x due to communication overhead)
- 4 GPUs: ~3.5-3.8x faster
- 8 GPUs: ~6.5-7.5x faster

**If speedup is much lower:**

1. **Check GPU utilization:**
   ```bash
   # In another terminal while training
   watch nvidia-smi
   ```
   All GPUs should show high utilization (>80%)

2. **Increase batch size:** Small batches don't parallelize well
   ```yaml
   batch_size: 8  # Increase from 4
   ```

3. **Check data loading:** May be bottlenecked on CPU
   - Consider using faster storage (SSD vs HDD)
   - Increase dataset preprocessing

### KataGo Multi-GPU Issues (GRPO)

**Symptoms:**
```
RuntimeError: KataGo not initialized
```

**Explanation:**
In multi-GPU GRPO training, only the main process (rank 0) runs KataGo. If you see this error on worker processes, it's expected behavior that should be handled automatically.

**If training fails:**
1. Check that KataGo binary is accessible
2. Verify KATAGO_BIN environment variable is set
3. Test KataGo works: `$KATAGO_BIN version`

### Mixed GPU Types

**Issue:** Using different GPU models (e.g., 2× A100 + 2× V100)

**Problem:** Slower GPUs become bottlenecks, reducing overall speedup

**Solution:** Use homogeneous GPUs or select specific GPUs:
```bash
# Use only GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0,1
./train_sft.sh --multi-gpu ddp-2
```

## Advanced Configuration

### Custom Accelerate Configs

Create custom configurations for your setup:

```bash
# Interactive config creation
accelerate config --config_file config/accelerate/my_custom.yaml
```

Example custom config:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 6  # Custom GPU count
gpu_ids: 0,1,2,3,4,5  # Specific GPU selection
mixed_precision: bf16  # bfloat16 instead of fp16
dynamo_backend: 'NO'
```

Use your custom config:
```bash
# Modify train_sft.sh or call accelerate directly
accelerate launch --config_file=config/accelerate/my_custom.yaml \
    scripts/run_sft.py --config config/sft_warm_start.yaml
```

### Environment Variables

Useful environment variables for multi-GPU training:

```bash
# Select specific GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Increase NCCL timeout for large models
export NCCL_TIMEOUT=7200

# Debug NCCL communication
export NCCL_DEBUG=INFO

# Optimize NCCL for specific topology
export NCCL_P2P_DISABLE=0  # Enable peer-to-peer communication
export NCCL_IB_DISABLE=0   # Enable InfiniBand if available
```

### Monitoring Training

**GPU Utilization:**
```bash
# Real-time monitoring
watch -n 1 nvidia-smi
```

**Weights & Biases:**
```bash
./train_sft.sh --multi-gpu ddp-4 --wandb --wandb-project my-go-rl
```

**TensorBoard (future):**
Coming soon - integrated TensorBoard logging for multi-GPU runs.

### Checkpointing

Checkpoints are saved by the main process only:

```yaml
# config/sft_warm_start.yaml
output_dir: outputs/sft_warm_start  # Saved by rank 0
```

Loading checkpoints works seamlessly across single and multi-GPU:
```bash
# Train with 4 GPUs
./train_sft.sh --multi-gpu ddp-4

# Resume or inference with 1 GPU
python scripts/eval_elo.py --checkpoint outputs/sft_warm_start
```

## Best Practices

1. **Start with single GPU:** Verify training works before scaling
2. **Use smoke tests:** Test multi-GPU with `--smoke` flag first
3. **Scale batch size:** Increase effective batch size with more GPUs
4. **Monitor utilization:** All GPUs should be >80% utilized
5. **Homogeneous GPUs:** Use same GPU models for best performance
6. **Gradient checkpointing:** Use when memory-limited, not by default
7. **W&B logging:** Track multi-GPU runs for performance comparison

## Why No FSDP?

**FSDP (Fully Sharded Data Parallel)** would enable:
- Sharding model parameters across GPUs
- Training larger models than fit in single GPU
- Further memory efficiency

**However:** Unsloth doesn't support FSDP as of 2026.

**Why it's okay:**
- Unsloth's 4-bit quantization provides excellent memory efficiency
- Qwen-3-8B fits comfortably on 24GB GPUs with 4-bit quantization
- DDP provides near-linear speedup for this model size

**Future:** If Unsloth adds FSDP support, we can add configurations for it.

## References

- [Multi-GPU Fine-tuning with Unsloth](https://docs.unsloth.ai/basics/multi-gpu-training-with-unsloth)
- [Multi-GPU Fine-tuning with DDP - Unsloth Docs](https://docs.unsloth.ai/basics/multi-gpu-training-with-unsloth/ddp)
- [Hugging Face Accelerate Documentation](https://huggingface.co/docs/transformers/en/accelerate)
- [TRL Documentation - Distributing Training](https://huggingface.co/docs/trl/main/en/distributing_training)
