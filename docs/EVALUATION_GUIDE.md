# Model Evaluation Guide

This guide explains how to evaluate trained Go models using KataGo match play to estimate Elo ratings.

## Prerequisites

Before running evaluations, you need:

1. **KataGo installed and configured**
   - Download KataGo binary from https://github.com/lightvector/KataGo
   - Download a KataGo neural network model
   - Create a KataGo analysis config file
   - Set environment variables:
     ```bash
     export KATAGO_BIN=/path/to/katago
     export KATAGO_MODEL=/path/to/model.bin.gz
     export KATAGO_CONFIG=/path/to/analysis_config.cfg
     ```

2. **A trained model checkpoint**
   - Output from SFT training (`outputs/sft_warm_start/`)
   - Output from GRPO training (`outputs/grpo_9x9/` or `outputs/grpo_19x19/`)

3. **A GTP engine wrapper for your model**
   - The trained model needs to be wrapped as a GTP engine for match play
   - See "Creating a GTP Wrapper" section below

## Quick Start

### Option 1: Using the Config Generator (Recommended)

Create a match config automatically:

```bash
python scripts/create_match_config.py \
  --output config/my_match.cfg \
  --board-size 9 \
  --num-games 200 \
  --model-command "python scripts/model_to_gtp.py --checkpoint outputs/grpo_9x9/final" \
  --katago-bin $KATAGO_BIN \
  --katago-model $KATAGO_MODEL \
  --katago-config $KATAGO_CONFIG
```

Then run the match:

```bash
python scripts/eval_elo.py \
  --katago-bin $KATAGO_BIN \
  --match-config config/my_match.cfg
```

### Option 2: Using Templates

1. Copy a template:
   ```bash
   cp config/katago_match_9x9.cfg config/my_match.cfg
   ```

2. Edit the config and replace placeholder paths:
   - Replace `/path/to/your/trained/model/gtp/wrapper.py` with your GTP wrapper command
   - Replace `${KATAGO_BIN}`, `${KATAGO_MODEL}`, `${KATAGO_CONFIG}` with actual paths

3. Run the match:
   ```bash
   python scripts/eval_elo.py \
     --katago-bin $KATAGO_BIN \
     --match-config config/my_match.cfg
   ```

## Match Configuration Options

### Key Settings

- **numGamesTotal**: Number of games to play (200-400 recommended for stable Elo)
- **boardSize**: 9 or 19
- **komi**: Usually 7.5 for Chinese rules
- **rules**: chinese, japanese, tromp-taylor, etc.
- **mainTime**: Main time per player in seconds
- **byoYomiTime**: Extra time per move after main time expires
- **alternateColors**: true = players alternate black/white (recommended)

### Time Controls

**9x9 boards (faster games):**
- mainTime = 60 seconds
- byoYomiTime = 5 seconds
- byoYomiPeriods = 5

**19x19 boards (longer games):**
- mainTime = 300 seconds
- byoYomiTime = 10 seconds
- byoYomiPeriods = 5

### Player Configuration

Each player needs:
- **name**: Display name for the player
- **command**: Shell command to launch GTP engine
- **trustPV**: Whether to trust principal variation from this engine

Example:
```
player1 = {
  name = MyTrainedModel
  command = python scripts/model_to_gtp.py --checkpoint outputs/grpo_9x9/final
  trustPV = false
}
```

## Creating a GTP Wrapper

Your trained model needs to communicate via the Go Text Protocol (GTP). You'll need to create a wrapper script that:

1. Loads your trained model
2. Reads GTP commands from stdin
3. Generates moves using your model
4. Writes GTP responses to stdout

**Minimal GTP commands to support:**
- `protocol_version` → return "2"
- `name` → return your model name
- `version` → return version string
- `known_command <cmd>` → return "true" or "false"
- `list_commands` → return list of supported commands
- `quit` → exit cleanly
- `boardsize <size>` → set board size (9 or 19)
- `clear_board` → reset to empty board
- `komi <komi>` → set komi value
- `play <color> <vertex>` → opponent plays move
- `genmove <color>` → generate and return a move
- `showboard` → (optional) display current position

**Example wrapper structure:**

```python
#!/usr/bin/env python3
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from go_rl.board import BoardState
from go_rl.prompting import build_prompt, parse_move

# Load model
model = AutoModelForCausalLM.from_pretrained("path/to/checkpoint")
tokenizer = AutoTokenizer.from_pretrained("path/to/checkpoint")
board = BoardState(size=9)

def genmove(color):
    # Build prompt from current board state
    prompt = build_prompt(board, color)

    # Generate move using model
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=10)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Parse move from model output
    move = parse_move(response)

    # Apply move to board
    board.apply_move(color, move)

    return move

# GTP command loop
while True:
    try:
        line = input().strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        cmd = parts[0]
        args = parts[1:]

        if cmd == "genmove":
            move = genmove(args[0].upper())
            print(f"= {move}\n")
        elif cmd == "play":
            board.apply_move(args[0].upper(), args[1])
            print("=\n")
        elif cmd == "quit":
            print("=\n")
            break
        # ... handle other commands ...

    except Exception as e:
        print(f"? Error: {e}\n")
        sys.stderr.write(f"Error: {e}\n")
```

## Interpreting Results

### Match Log

The match log (`match.log` by default) contains:
- Game results (winner, score, resignation, etc.)
- Win rate for each player
- Estimated Elo difference

**Reading the log:**
```
Player1 vs Player2: 52 wins, 48 losses (52.0% win rate)
Estimated Elo difference: +14 ± 45 (95% confidence)
```

**Elo interpretation:**
- **+0 to +50**: Model is roughly equal to baseline
- **+50 to +150**: Model is noticeably stronger (amateur to SDK level)
- **+150 to +300**: Model is significantly stronger (SDK to dan level)
- **+300+**: Model is much stronger (dan+ level)

Note: Elo differences are relative to your KataGo baseline configuration (visits, playouts, etc.)

### SGF Files

Game records are saved as SGF files in the output directory:
- View games in an SGF viewer (Sabaki, KaTrain, etc.)
- Analyze games with KataGo or other tools
- Check for patterns, mistakes, or illegal moves

### Statistical Significance

For reliable Elo estimates:
- **Minimum**: 100 games (±70 Elo confidence interval)
- **Recommended**: 200-400 games (±45 Elo confidence interval)
- **Ideal**: 400+ games (±30 Elo confidence interval)

Win rate to Elo conversion (approximate):
- 50% → 0 Elo difference
- 55% → +35 Elo
- 60% → +72 Elo
- 70% → +150 Elo
- 80% → +240 Elo
- 90% → +366 Elo

## Troubleshooting

### "KataGo binary not found"
- Check that `--katago-bin` points to the correct path
- Verify the file exists: `ls -l $KATAGO_BIN`
- Make sure the binary is executable: `chmod +x $KATAGO_BIN`

### "Match config contains placeholder paths"
- Edit your match config file
- Replace `/path/to/...` with actual paths
- Replace `${VARIABLE}` with actual values

### "GTP engine failed to respond"
- Test your GTP wrapper manually:
  ```bash
  echo "genmove black" | python scripts/model_to_gtp.py --checkpoint ...
  ```
- Check wrapper logs for errors
- Verify model checkpoint loads correctly

### Games timeout or run very slowly
- Reduce `mainTime` and `byoYomiTime` in match config
- Check if model inference is optimized
- Consider using a smaller/faster baseline (fewer KataGo visits)

### Model plays illegal moves
- Check that GTP wrapper validates moves before returning
- Verify board state is tracked correctly
- Add move validation in the wrapper

## Advanced Topics

### Adjusting KataGo Strength

Control baseline difficulty by adjusting KataGo settings in the match config:

**Weaker baseline (faster evaluation):**
```
player2 = {
  name = KataGoWeak
  command = katago gtp -model model.bin.gz -config config.cfg -override-config maxVisits=100,numSearchThreads=2
  trustPV = true
}
```

**Stronger baseline (harder to beat):**
```
player2 = {
  name = KataGoStrong
  command = katago gtp -model model.bin.gz -config config.cfg -override-config maxVisits=1000,numSearchThreads=8
  trustPV = true
}
```

### Evaluating Multiple Checkpoints

To track training progress, evaluate multiple checkpoints:

```bash
for ckpt in outputs/grpo_9x9/checkpoint-*; do
  python scripts/create_match_config.py \
    --output config/match_$(basename $ckpt).cfg \
    --model-command "python scripts/model_to_gtp.py --checkpoint $ckpt" \
    --board-size 9 \
    --num-games 200

  python scripts/eval_elo.py \
    --katago-bin $KATAGO_BIN \
    --match-config config/match_$(basename $ckpt).cfg \
    --log-file logs/match_$(basename $ckpt).log
done
```

### Head-to-Head Matches

Compare two trained models directly:

```
player1 = {
  name = Model_v1
  command = python scripts/model_to_gtp.py --checkpoint outputs/grpo_9x9/v1
  trustPV = false
}

player2 = {
  name = Model_v2
  command = python scripts/model_to_gtp.py --checkpoint outputs/grpo_9x9/v2
  trustPV = false
}
```

## References

- KataGo documentation: https://github.com/lightvector/KataGo
- GTP specification: https://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html
- Elo rating system: https://en.wikipedia.org/wiki/Elo_rating_system
