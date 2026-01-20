# Go RL Training Quick Start Guide

This guide will help you set up and train a Go-playing AI using reinforcement learning on a fresh server.

## Prerequisites

- **Linux server** with NVIDIA GPU (H100, A100, RTX 40 series, etc.)
- **CUDA 11.8+** installed with drivers
- **At least 80GB VRAM** recommended for full training
- **Internet connection** for downloading models and packages

## 🚀 Quick Start (3 Steps)

### Step 1: Environment Setup

Run this on a fresh server to install all dependencies:

```bash
./setup_environment.sh
```

This will:
- ✅ Install `uv` package manager
- ✅ Create Python virtual environment
- ✅ Install PyTorch with CUDA support
- ✅ Install Unsloth, TRL, and all dependencies
- ✅ Download and configure KataGo (Go analysis engine)
- ✅ Create directory structure
- ✅ Generate `.env` file with KataGo paths

**Time:** ~10-15 minutes

### Step 2: Prepare Training Data

First, add SGF (game record) files to the appropriate directory:

```bash
# For 9x9 Go games
cp your_sgf_files/*.sgf data/sgf_9x9/

# OR for 19x19 Go games
cp your_sgf_files/*.sgf data/sgf_19x19/
```

**Where to get SGF files:**
- [U-Go.net](https://www.u-go.net/gamerecords/) - 9x9 games
- [GoKifu](https://gokifu.com/) - Professional games
- [Online-Go.com](https://www.online-go.com/) - Download your own games
- [Waltheri's Go Pattern Search](https://ps.waltheri.net/) - Historical games

Then run the data preparation script:

```bash
./prepare_data.sh
```

This will:
- ✅ Parse all SGF files
- ✅ Generate SFT training data (`data/sft_9x9.jsonl`)
- ✅ Generate RL prompt data (`data/rl_prompts_9x9.jsonl`)
- ✅ Validate board positions

**Time:** ~1-5 minutes (depends on number of SGF files)

### Step 3: Train Your Model

#### Option A: Supervised Fine-Tuning (Recommended First Step)

Start with SFT warm-start for better initial policy:

```bash
# Smoke test (verifies everything works)
./train_sft.sh --smoke

# Full training
./train_sft.sh
```

**Time:** 
- Smoke test: ~2-5 minutes
- Full training: ~30-60 minutes (depends on dataset size)

#### Option B: Reinforcement Learning (GRPO)

Train with KataGo-based rewards:

```bash
# Smoke test
./train_grpo.sh --smoke

# Full training
./train_grpo.sh
```

**Time:**
- Smoke test: ~5-10 minutes (includes KataGo analysis)
- Full training: ~2-6 hours (depends on num_steps in config)

## 📊 Advanced Usage

### Training Options

Both training scripts support these flags:

```bash
# Run smoke test (minimal training for verification)
./train_sft.sh --smoke
./train_grpo.sh --smoke

# Enable Weights & Biases logging
./train_sft.sh --wandb --wandb-project my-go-project

# Train on 19x19 board
./train_sft.sh --board-size 19x19
./train_grpo.sh --board-size 19x19

# Enable debug logging
./train_sft.sh --debug
```

### Customizing Training

Edit the YAML config files to adjust hyperparameters:

**For SFT:**
```bash
vim config/sft_warm_start.yaml
```

Key parameters:
- `model_name`: Model to fine-tune (default: `qwen-3-8b`)
- `epochs`: Number of training epochs
- `batch_size`: Batch size per device
- `learning_rate`: Learning rate for optimizer

**For GRPO:**
```bash
vim config/grpo_9x9.yaml
```

Key parameters:
- `training.num_steps`: Total training steps
- `training.learning_rate`: Learning rate
- `rollout.num_samples`: Rollout samples per prompt
- `reward.winrate_scale`: Reward scaling for winrate delta
- `reward.illegal_penalty`: Penalty for illegal moves

### Monitoring Training

**Without W&B:**
- Check console output for loss/reward metrics
- View logs in `logs/sft_warm_start/` or `logs/grpo_9x9/`

**With W&B:**
```bash
# Login once
wandb login

# Train with logging
./train_sft.sh --wandb --wandb-project go-rl
```

View training curves at: https://wandb.ai/

## 🎯 Recommended Training Pipeline

### For 9x9 Go (Beginner-Friendly)

1. **Collect Data:** Get 100-1000 9x9 SGF files
2. **SFT Warm Start:** Train for 1 epoch (~30 min)
3. **GRPO Training:** Train for 2000 steps (~3 hours)
4. **Evaluation:** Test against KataGo
5. **Iterate:** Adjust rewards, train more

### For 19x19 Go (Advanced)

1. **Collect Data:** Get 5000-10000 19x19 SGF files
2. **SFT Warm Start:** Train for 1-2 epochs (~1-2 hours)
3. **GRPO Training:** Start with 9x9, promote to 19x19 after gate
4. **Long Training:** May need 10k+ steps for good play

## 📁 Output Files

After training, you'll find:

```
outputs/
├── sft_warm_start/      # SFT checkpoint
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
└── grpo_9x9/            # GRPO checkpoint
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── tokenizer files

logs/
├── sft_warm_start/      # SFT run logs
│   └── sft_warm_start_<timestamp>.json
└── grpo_9x9/            # GRPO run logs
    └── grpo_9x9_<timestamp>.json
```

## 🔧 Troubleshooting

### "CUDA out of memory"
- Reduce `batch_size` in config
- Reduce `gradient_accumulation` in GRPO config
- Use smaller model (e.g., `qwen-3-1.5b`)

### "KataGo not found"
```bash
# Re-run setup
source .env
echo $KATAGO_BIN  # Should show path

# Test KataGo
$KATAGO_BIN version
```

### "No SGF files found"
```bash
# Check directory
ls -la data/sgf_9x9/

# Download sample SGF files
wget -P data/sgf_9x9/ https://example.com/sample.sgf
```

### "Import error: No module named 'unsloth'"
```bash
# Activate environment
source .venv/bin/activate

# Reinstall
uv pip install -r requirements.txt
```

### "Dataset not found"
```bash
# Re-run data preparation
./prepare_data.sh
```

## 🎓 Next Steps

After successful training:

1. **Evaluate Your Model:**
   ```bash
   python scripts/eval_elo.py \
     --katago-bin $KATAGO_BIN \
     --match-config path/to/match.cfg
   ```

2. **Save for Deployment:**
   - See `PLAN/next_steps_1.md` for GGUF export
   - Use `model.save_pretrained()` for HuggingFace format

3. **Experiment with Rewards:**
   - Adjust `reward` section in `config/grpo_9x9.yaml`
   - Try different penalties/bonuses
   - Enable `strict_legality` for ko/suicide checks

4. **Scale Up:**
   - Promote to 19x19 board size
   - Increase training steps
   - Use larger base model

## 📚 Additional Resources

- **Main README:** `README.md` - Full project documentation
- **Agent Guidelines:** `AGENTS.md` - Development conventions
- **Progress Logs:** `PLAN/progress-*.md` - Implementation history
- **Unsloth Docs:** https://docs.unsloth.ai/
- **TRL Docs:** https://huggingface.co/docs/trl/
- **KataGo:** https://github.com/lightvector/KataGo

## 🐛 Getting Help

If you encounter issues:

1. Check logs in `logs/` directory
2. Run smoke tests: `./train_sft.sh --smoke`
3. Verify KataGo: `$KATAGO_BIN version`
4. Check GPU: `nvidia-smi`
5. Open an issue on GitHub with error logs

---

**Happy Training! 🦥🎮**
