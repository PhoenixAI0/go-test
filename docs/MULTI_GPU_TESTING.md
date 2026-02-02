# Multi-GPU Training Testing Guide

This guide explains how to test and validate multi-GPU training setups for the Go RL project.

## Overview

Multi-GPU training uses PyTorch's DistributedDataParallel (DDP) via the Accelerate library. Reward computation with KataGo only runs on rank 0 (the main process) to avoid redundant analysis.

## Prerequisites

### Hardware
- 2 or more NVIDIA GPUs
- GPUs should be of the same model for best performance
- Sufficient VRAM (16GB+ recommended per GPU)

### Software
- CUDA 11.8+ with drivers
- PyTorch with CUDA support
- Accelerate library (`pip install accelerate`)
- NCCL backend (usually included with PyTorch)

## Quick Validation

### Step 1: Run Validation Script

```bash
python scripts/validate_multi_gpu.py --verbose
```

This checks:
- GPU availability
- NCCL backend
- Accelerate installation
- Configuration files
- Training scripts

### Step 2: Check GPU Count

```bash
# Using nvidia-smi
nvidia-smi --list-gpus

# Or using PyTorch
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### Step 3: Verify Accelerate Configs

```bash
ls -la config/accelerate/
# Should show: single_gpu.yaml, ddp_2gpu.yaml, ddp_4gpu.yaml, ddp_8gpu.yaml
```

## Running Multi-GPU Training

### Supervised Fine-Tuning (SFT)

```bash
# 2 GPUs
./train_sft.sh --multi-gpu 2 --smoke

# 4 GPUs
./train_sft.sh --multi-gpu 4

# Auto-detect GPU count
./train_sft.sh --multi-gpu auto
```

### GRPO Training

```bash
# 2 GPUs
./train_grpo.sh --multi-gpu 2 --smoke

# 4 GPUs
./train_grpo.sh --multi-gpu 4
```

## Important Considerations

### Rank 0 Operations

**These operations ONLY run on rank 0 (main process):**
- KataGo analysis and reward computation
- Model checkpointing
- Logging to Weights & Biases
- Console output (to avoid clutter)

**These operations run on ALL ranks:**
- Forward pass (model inference)
- Backward pass (gradient computation)
- Parameter updates

### Batch Size and Gradient Accumulation

With multi-GPU training:
- Effective batch size = `batch_size * num_gpus * gradient_accumulation`
- Each GPU processes `batch_size` samples
- Gradients are synchronized across GPUs before updates

Example:
```yaml
# config/grpo_9x9.yaml
batch_size: 1              # Per GPU
gradient_accumulation: 8   # Steps before update
# With 4 GPUs: effective_batch_size = 1 * 4 * 8 = 32
```

### KataGo Reward Computation

**Critical:** KataGo analysis only runs on rank 0 to avoid:
- Redundant computation
- Resource contention
- File system conflicts

The GRPO script checks `LOCAL_RANK` environment variable:
```python
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
if local_rank == 0:
    # Run KataGo analysis
    reward = compute_reward(...)
else:
    # Skip KataGo, use dummy reward or broadcast from rank 0
    reward = 0.0
```

## Testing Multi-GPU Setup

### Unit Tests

Run multi-GPU configuration tests:

```bash
# Run all multi-GPU tests
pytest tests/test_multi_gpu_setup.py -v

# Run specific test
pytest tests/test_multi_gpu_setup.py::TestMultiGPUConfiguration -v
```

### Smoke Test

Run a minimal training test:

```bash
# SFT smoke test (fast, verifies setup)
./train_sft.sh --multi-gpu 2 --smoke

# GRPO smoke test (includes KataGo, slower)
./train_grpo.sh --multi-gpu 2 --smoke
```

Expected output:
```
Multi-GPU training enabled: 2 GPU(s)
[Rank 0] Starting training...
[Rank 0] KataGo analysis enabled
[Rank 1] Starting training...
[Rank 1] KataGo analysis disabled (worker rank)
...
Training complete!
```

### Integration Test

Run a short training session (100 steps):

```bash
# Modify config to limit steps
vim config/grpo_9x9.yaml
# Set: num_steps: 100

# Run training
./train_grpo.sh --multi-gpu 4

# Check that:
# - All GPUs show activity in nvidia-smi
# - Rank 0 logs KataGo analysis
# - Workers don't log KataGo
# - Training completes without errors
```

## Troubleshooting

### "NCCL error: unhandled system error"

**Causes:**
- NCCL version incompatible with CUDA
- Network issues (if using multi-node)
- Insufficient shared memory

**Solutions:**
```bash
# Update NCCL
conda install -c conda-forge nccl

# Increase shared memory (Docker)
docker run --shm-size=8g ...

# Check NCCL version
python -c "import torch; print(torch.cuda.nccl.version())"
```

### "CUDA out of memory"

**Solutions:**
- Reduce batch size in config
- Enable gradient checkpointing: `--gradient-checkpointing`
- Reduce model size or sequence length
- Use fewer GPUs (counterintuitive but may help)

### "All GPUs show 0% utilization"

**Causes:**
- Training script not using accelerate
- Wrong accelerate config
- Batch size too small

**Solutions:**
```bash
# Verify accelerate config
cat config/accelerate/ddp_4gpu.yaml

# Check distributed environment variables
echo $LOCAL_RANK
echo $WORLD_SIZE

# Run with debug logging
./train_grpo.sh --multi-gpu 4 --debug
```

### "KataGo analysis running on all ranks"

**Problem:** This wastes compute and may cause issues.

**Solution:** Check rank before calling KataGo:
```python
import os

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
if local_rank == 0:
    result = katago_client.analyze_position(board)
```

### "Training hangs after first batch"

**Causes:**
- Deadlock from collective operations
- Rank 0 waiting for other ranks
- KataGo analysis blocking

**Solutions:**
- Ensure all ranks reach the same collective operations (reduce, barrier, etc.)
- Check that KataGo only runs on rank 0
- Add timeouts to KataGo client
- Use `NCCL_DEBUG=INFO` for detailed logs

## Performance Optimization

### Linear Scaling

Ideal multi-GPU scaling:
- 2 GPUs → 2x throughput
- 4 GPUs → 4x throughput
- 8 GPUs → 8x throughput

**Factors affecting scaling:**
- Communication overhead (gradient synchronization)
- Batch size (too small = poor GPU utilization)
- KataGo bottleneck (rank 0 only)
- Data loading (CPU bottleneck)

### Monitoring Performance

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Profile training
nsys profile -o profile python scripts/run_grpo.py ...

# Check throughput
# Look for "samples/sec" in training logs
```

### Optimization Tips

1. **Increase batch size**: Maximize GPU utilization
   ```yaml
   batch_size: 4  # Increase from 1
   ```

2. **Gradient accumulation**: Simulate larger batches
   ```yaml
   gradient_accumulation: 16  # Increase from 8
   ```

3. **Mixed precision**: Reduce memory and increase speed
   ```yaml
   # In accelerate config
   mixed_precision: "fp16"
   ```

4. **Gradient checkpointing**: Trade compute for memory
   ```bash
   ./train_grpo.sh --multi-gpu 4 --gradient-checkpointing
   ```

5. **Pin memory**: Faster CPU→GPU transfers
   ```python
   dataloader = DataLoader(..., pin_memory=True)
   ```

## Environment Variables

Key environment variables for distributed training:

```bash
# Set by accelerate automatically:
LOCAL_RANK=0           # GPU index on current node
RANK=0                 # Global rank across all nodes
WORLD_SIZE=4           # Total number of processes
MASTER_ADDR=localhost  # Address of rank 0
MASTER_PORT=29500      # Port for communication

# Useful for debugging:
NCCL_DEBUG=INFO        # Verbose NCCL logging
CUDA_VISIBLE_DEVICES=0,1,2,3  # Limit visible GPUs

# For KataGo:
KATAGO_BIN=/path/to/katago
KATAGO_MODEL=/path/to/model.bin.gz
KATAGO_CONFIG=/path/to/config.cfg
```

## Best Practices

1. **Always test with smoke test first**
   ```bash
   ./train_grpo.sh --multi-gpu 4 --smoke
   ```

2. **Monitor GPU utilization**
   - All GPUs should show >80% utilization during training
   - If not, increase batch size or reduce model size

3. **Check logs for rank 0 only operations**
   - KataGo analysis should only appear in rank 0 logs
   - Other ranks should skip these operations

4. **Validate configs match GPU count**
   ```bash
   # For 4 GPUs, use ddp_4gpu.yaml
   cat config/accelerate/ddp_4gpu.yaml | grep num_processes
   # Should output: num_processes: 4
   ```

5. **Use gradient accumulation for large effective batch sizes**
   - Helps with memory constraints
   - Maintains training stability

6. **Save checkpoints only on rank 0**
   - Avoid file system conflicts
   - Reduces I/O overhead

## Additional Resources

- [Accelerate Documentation](https://huggingface.co/docs/accelerate/)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [Multi-GPU Training Best Practices](https://huggingface.co/docs/transformers/perf_train_gpu_many)
