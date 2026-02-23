#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "Go RL: Full Training Pipeline"
echo "=========================================="
echo ""
echo "This script runs:"
echo "  1. Environment setup"
echo "  2. Dataset preparation"
echo "  3. SFT (9x9 warm start)"
echo "  4. Evaluation (SFT -> KataGo match, unless --no-eval)"
echo "  5. GRPO 9x9"
echo "  6. Evaluation + promotion gate (before 19x19, unless --no-eval)"
echo "  7. GRPO 19x19 curriculum stage"
echo "  8. Evaluation + promotion gate (before full-game GRPO, unless --no-eval)"
echo "  9. Full-game GRPO 19x19 (conditional)"
echo " 10. Final evaluation before upload (unless --no-eval)"
echo " 11. Optional upload to Hugging Face Hub"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# CLI options
SKIP_SETUP=false
SKIP_DATA=false
SKIP_SFT=false
SKIP_FULLGAME=false
SMOKE_TEST=false
USE_WANDB=false
WANDB_PROJECT="go-rl"
UPLOAD_TO_HUB=false
HF_REPO_ID=""
HF_PRIVATE=false
HF_BASE_MODEL="qwen-3-8b"
RUN_EVAL=true
MODEL_COMMAND_TEMPLATE="${MODEL_GTP_COMMAND_TEMPLATE:-}"
EVAL_GAMES=50
EVAL_MAIN_TIME=60
EVAL_BYOYOMI_TIME=5
EVAL_BYOYOMI_PERIODS=5
EVAL_KOMI=7.5
EVAL_RULES="chinese"
EVAL_SGF_ROOT="match_sgfs"
EVAL_LOG_ROOT="logs/matches"
FULLGAME_SMOKE_PLIES=""

# Promotion gate values (loaded from config/grpo_9x9.yaml)
PROMOTION_METRIC="elo_delta_or_winrate"
PROMOTION_THRESHOLD_ELO="150"
PROMOTION_THRESHOLD_WINRATE="0.55"
PROMOTION_EVAL_GAMES="200"

FULLGAME_PROMOTION_METRIC="winrate"
FULLGAME_PROMOTION_THRESHOLD_ELO="150"
FULLGAME_PROMOTION_THRESHOLD_WINRATE="0.55"
FULLGAME_PROMOTION_EVAL_GAMES="200"

LAST_EVAL_SUMMARY=""
LAST_EVALUATED_MODEL_DIR=""
FULLGAME_GATE_PASSED=false
FULLGAME_COMPLETED=false
FINAL_MODEL_DIR="outputs/grpo_19x19"
FINAL_BOARD_SIZE="19x19"
FINAL_BOARD_NUM=19

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --skip-sft)
            SKIP_SFT=true
            shift
            ;;
        --skip-fullgame)
            SKIP_FULLGAME=true
            shift
            ;;
        --smoke)
            SMOKE_TEST=true
            shift
            ;;
        --wandb)
            USE_WANDB=true
            shift
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --upload-to-hub)
            UPLOAD_TO_HUB=true
            shift
            ;;
        --hf-repo-id)
            HF_REPO_ID="$2"
            shift 2
            ;;
        --hf-private)
            HF_PRIVATE=true
            shift
            ;;
        --hf-base-model)
            HF_BASE_MODEL="$2"
            shift 2
            ;;
        --no-eval)
            RUN_EVAL=false
            shift
            ;;
        --model-command-template)
            MODEL_COMMAND_TEMPLATE="$2"
            shift 2
            ;;
        --eval-games)
            EVAL_GAMES="$2"
            shift 2
            ;;
        --eval-main-time)
            EVAL_MAIN_TIME="$2"
            shift 2
            ;;
        --eval-byoyomi-time)
            EVAL_BYOYOMI_TIME="$2"
            shift 2
            ;;
        --eval-byoyomi-periods)
            EVAL_BYOYOMI_PERIODS="$2"
            shift 2
            ;;
        --eval-komi)
            EVAL_KOMI="$2"
            shift 2
            ;;
        --eval-rules)
            EVAL_RULES="$2"
            shift 2
            ;;
        --eval-sgf-root)
            EVAL_SGF_ROOT="$2"
            shift 2
            ;;
        --eval-log-root)
            EVAL_LOG_ROOT="$2"
            shift 2
            ;;
        --fullgame-smoke-plies)
            FULLGAME_SMOKE_PLIES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-setup               Skip environment setup"
            echo "  --skip-data                Skip data preparation"
            echo "  --skip-sft                 Skip 9x9 SFT stage"
            echo "  --skip-fullgame            Skip full-game GRPO stage"
            echo "  --smoke                    Run smoke tests instead of full training"
            echo "  --wandb                    Enable Weights & Biases logging"
            echo "  --wandb-project NAME       W&B project name (default: go-rl)"
            echo "  --no-eval                  Skip all match evaluations and promotion gate checks"
            echo "  --model-command-template T Model engine command template"
            echo "                             Must contain {MODEL_DIR}; optional {BOARD_SIZE}"
            echo "  --eval-games N             Games for non-gate evaluations (default: 50)"
            echo "  --eval-main-time N         Main time seconds per side (default: 60)"
            echo "  --eval-byoyomi-time N      Byoyomi time seconds (default: 5)"
            echo "  --eval-byoyomi-periods N   Byoyomi periods (default: 5)"
            echo "  --eval-komi K              Komi value for evaluation (default: 7.5)"
            echo "  --eval-rules RULES         Ruleset for evaluation (default: chinese)"
            echo "  --eval-sgf-root DIR        SGF output root (default: match_sgfs)"
            echo "  --eval-log-root DIR        Eval logs/config root (default: logs/matches)"
            echo "  --fullgame-smoke-plies N   Override full-game max plies during step 4"
            echo "  --upload-to-hub            Upload final model to Hugging Face Hub"
            echo "  --hf-repo-id ID            Hugging Face repo id (required with upload)"
            echo "  --hf-private               Create/use private HF repository"
            echo "  --hf-base-model NAME       Base model name for model card"
            echo "  -h, --help                 Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

if [ "$UPLOAD_TO_HUB" = true ] && [ -z "$HF_REPO_ID" ]; then
    echo -e "${RED}Error: --hf-repo-id is required when --upload-to-hub is set${NC}"
    exit 1
fi

if [ "$SMOKE_TEST" = true ] && [ "$EVAL_GAMES" -gt 8 ]; then
    EVAL_GAMES=8
fi

if [ "$RUN_EVAL" = false ]; then
    echo -e "${YELLOW}⚠️  --no-eval enabled: skipping all evaluation and promotion-gate checks.${NC}"
fi

if [ "$SMOKE_TEST" = false ]; then
    echo -e "${YELLOW}⚠️  WARNING: This run may take hours to days depending on stages and evaluation games.${NC}"
fi

START_TIME=$(date +%s)

build_model_command() {
    local model_dir="$1"
    local board_size="$2"
    local command="$MODEL_COMMAND_TEMPLATE"
    command="${command//\{MODEL_DIR\}/$model_dir}"
    command="${command//\{BOARD_SIZE\}/$board_size}"
    echo "$command"
}

load_promotion_gate_from_config() {
    local cfg_path="$1"
    local gate_values=()
    local gate_line
    while IFS= read -r gate_line; do
        gate_values+=("$gate_line")
    done < <(
        python - "$cfg_path" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8")) or {}
gate = cfg.get("promotion_gate") or {}
print(gate.get("metric", "elo_delta_or_winrate"))
print(gate.get("threshold_elo", 150))
print(gate.get("threshold_winrate", 0.55))
print(gate.get("evaluation_games", 200))
PY
    )
    PROMOTION_METRIC="${gate_values[0]:-elo_delta_or_winrate}"
    PROMOTION_THRESHOLD_ELO="${gate_values[1]:-150}"
    PROMOTION_THRESHOLD_WINRATE="${gate_values[2]:-0.55}"
    PROMOTION_EVAL_GAMES="${gate_values[3]:-200}"
}

load_fullgame_promotion_gate_from_config() {
    local cfg_path="$1"
    local gate_values=()
    local gate_line
    while IFS= read -r gate_line; do
        gate_values+=("$gate_line")
    done < <(
        python - "$cfg_path" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8")) or {}
gate = cfg.get("promotion_gate_fullgame") or {}
print(gate.get("metric", "winrate"))
print(gate.get("threshold_elo", 150))
print(gate.get("threshold_winrate", 0.55))
print(gate.get("evaluation_games", 200))
PY
    )
    FULLGAME_PROMOTION_METRIC="${gate_values[0]:-winrate}"
    FULLGAME_PROMOTION_THRESHOLD_ELO="${gate_values[1]:-150}"
    FULLGAME_PROMOTION_THRESHOLD_WINRATE="${gate_values[2]:-0.55}"
    FULLGAME_PROMOTION_EVAL_GAMES="${gate_values[3]:-200}"
}

evaluate_promotion_gate() {
    local summary_json="$1"
    local metric="$2"
    local threshold_elo="$3"
    local threshold_winrate="$4"
    python - "$summary_json" "$metric" "$threshold_elo" "$threshold_winrate" <<'PY'
import json
import sys

summary_path, metric, threshold_elo, threshold_wr = sys.argv[1:]
summary = json.load(open(summary_path, "r", encoding="utf-8"))
elo = float(summary["elo_delta"])
win_rate = float(summary["win_rate"])
threshold_elo = float(threshold_elo)
threshold_wr = float(threshold_wr)

if metric == "elo_delta_or_winrate":
    passed = (elo >= threshold_elo) or (win_rate >= threshold_wr)
elif metric == "elo_delta_and_winrate":
    passed = (elo >= threshold_elo) and (win_rate >= threshold_wr)
elif metric == "elo_delta":
    passed = elo >= threshold_elo
elif metric == "winrate":
    passed = win_rate >= threshold_wr
else:
    print(f"Unknown promotion metric: {metric}", file=sys.stderr)
    sys.exit(2)

if passed:
    print(
        f"Promotion gate PASSED: metric={metric}, "
        f"elo_delta={elo:.2f} (threshold={threshold_elo}), "
        f"win_rate={win_rate:.4f} (threshold={threshold_wr})"
    )
    sys.exit(0)

print(
    f"Promotion gate FAILED: metric={metric}, "
    f"elo_delta={elo:.2f} (threshold={threshold_elo}), "
    f"win_rate={win_rate:.4f} (threshold={threshold_wr})",
    file=sys.stderr,
)
sys.exit(1)
PY
}

run_katago_eval() {
    local stage_label="$1"
    local board_size="$2"
    local model_dir="$3"
    local num_games="$4"

    if [ "$RUN_EVAL" = false ]; then
        echo -e "${YELLOW}Skipping evaluation stage '${stage_label}' (--no-eval).${NC}"
        return 0
    fi

    if [ -z "${KATAGO_BIN:-}" ] || [ -z "${KATAGO_MODEL:-}" ] || [ -z "${KATAGO_CONFIG:-}" ]; then
        echo -e "${RED}Error: KATAGO_BIN, KATAGO_MODEL, and KATAGO_CONFIG must be set for evaluation.${NC}"
        exit 1
    fi
    if [ ! -f "$KATAGO_BIN" ]; then
        echo -e "${RED}Error: KATAGO_BIN not found: $KATAGO_BIN${NC}"
        exit 1
    fi
    if [ ! -f "$KATAGO_MODEL" ]; then
        echo -e "${RED}Error: KATAGO_MODEL not found: $KATAGO_MODEL${NC}"
        exit 1
    fi
    if [ ! -f "$KATAGO_CONFIG" ]; then
        echo -e "${RED}Error: KATAGO_CONFIG not found: $KATAGO_CONFIG${NC}"
        exit 1
    fi

    if [ ! -d "$model_dir" ]; then
        echo -e "${RED}Error: Model directory not found for eval: $model_dir${NC}"
        exit 1
    fi

    local board_size_num="${board_size%x*}"
    local stage_tag="${stage_label//[^a-zA-Z0-9_]/_}"
    local model_name="Model_${stage_tag}"
    local katago_name="KataGoBaseline"
    local model_cmd
    model_cmd="$(build_model_command "$model_dir" "$board_size")"

    mkdir -p "$EVAL_LOG_ROOT" "$EVAL_SGF_ROOT"
    local match_cfg="${EVAL_LOG_ROOT}/match_${stage_tag}.cfg"
    local log_file="${EVAL_LOG_ROOT}/${stage_tag}.log"
    local sgf_dir="${EVAL_SGF_ROOT}/${stage_tag}"
    local summary_json="${EVAL_LOG_ROOT}/${stage_tag}_summary.json"

    echo ""
    echo -e "${YELLOW}=========================================="
    echo "EVALUATION: ${stage_label}"
    echo "==========================================${NC}"
    echo "  Board: ${board_size}"
    echo "  Model: ${model_dir}"
    echo "  Games: ${num_games}"

    uv run python scripts/create_match_config.py \
        --output "$match_cfg" \
        --board-size "$board_size_num" \
        --num-games "$num_games" \
        --model-command "$model_cmd" \
        --katago-bin "$KATAGO_BIN" \
        --katago-model "$KATAGO_MODEL" \
        --katago-config "$KATAGO_CONFIG" \
        --sgf-output-dir "$sgf_dir" \
        --main-time "$EVAL_MAIN_TIME" \
        --byoyomi-time "$EVAL_BYOYOMI_TIME" \
        --byoyomi-periods "$EVAL_BYOYOMI_PERIODS" \
        --komi "$EVAL_KOMI" \
        --rules "$EVAL_RULES" \
        --model-name "$model_name" \
        --katago-name "$katago_name"

    local -a EVAL_CMD=(
        uv run python scripts/eval_elo.py
        --katago-bin "$KATAGO_BIN"
        --match-config "$match_cfg"
        --log-file "$log_file"
        --sgf-dir "$sgf_dir"
        --summary-json "$summary_json"
        --model-name "$model_name"
        --katago-name "$katago_name"
    )
    if [ "$USE_WANDB" = true ]; then
        EVAL_CMD+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-prefix "eval/${stage_tag}" --wandb-run-name "eval_${stage_tag}")
    fi

    "${EVAL_CMD[@]}"
    LAST_EVAL_SUMMARY="$summary_json"
    LAST_EVALUATED_MODEL_DIR="$model_dir"
}

run_train_sft() {
    local board_size="$1"
    local -a SFT_CMD=(./scripts/train_sft.sh --board-size "$board_size")
    if [ "$SMOKE_TEST" = true ]; then
        SFT_CMD+=(--smoke)
    fi
    if [ "$USE_WANDB" = true ]; then
        SFT_CMD+=(--wandb --wandb-project "$WANDB_PROJECT")
    fi
    "${SFT_CMD[@]}"
}

run_train_grpo() {
    local board_size="$1"
    local -a GRPO_CMD=(./scripts/train_grpo.sh --board-size "$board_size")
    if [ "$SMOKE_TEST" = true ]; then
        GRPO_CMD+=(--smoke)
    fi
    if [ "$USE_WANDB" = true ]; then
        GRPO_CMD+=(--wandb --wandb-project "$WANDB_PROJECT")
    fi
    "${GRPO_CMD[@]}"
}

run_train_grpo_fullgame() {
    local -a FULLGAME_CMD=(./scripts/train_grpo_fullgame.sh)
    if [ "$SMOKE_TEST" = true ]; then
        FULLGAME_CMD+=(--smoke)
    fi
    if [ "$USE_WANDB" = true ]; then
        FULLGAME_CMD+=(--wandb --wandb-project "$WANDB_PROJECT")
    fi
    if [ -n "$FULLGAME_SMOKE_PLIES" ]; then
        FULLGAME_CMD+=(--fullgame-smoke-plies "$FULLGAME_SMOKE_PLIES")
    fi
    "${FULLGAME_CMD[@]}"
}

# Step 1: Environment setup
if [ "$SKIP_SETUP" = false ]; then
    echo ""
    echo -e "${YELLOW}=========================================="
    echo "STEP 1: Environment Setup"
    echo "==========================================${NC}"
    ./scripts/setup_environment.sh
else
    echo -e "${YELLOW}Skipping environment setup${NC}"
fi

# Activate environment for subsequent steps
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ ! -f ".venv/bin/activate" ]; then
        echo -e "${RED}Error: .venv not found.${NC}"
        echo "Run ./scripts/setup_environment.sh, or create one manually:"
        echo "  uv venv .venv --python 3.11"
        echo "  source .venv/bin/activate"
        echo "  uv sync --extra dev"
        exit 1
    fi
    source .venv/bin/activate
fi
if [ -f .env ]; then
    source .env
fi

if [ "$RUN_EVAL" = true ]; then
    if [ -z "$MODEL_COMMAND_TEMPLATE" ] && [ -n "${MODEL_GTP_COMMAND_TEMPLATE:-}" ]; then
        MODEL_COMMAND_TEMPLATE="$MODEL_GTP_COMMAND_TEMPLATE"
    fi
    if [ -z "$MODEL_COMMAND_TEMPLATE" ]; then
        echo -e "${RED}Error: evaluation requires --model-command-template (or MODEL_GTP_COMMAND_TEMPLATE env var).${NC}"
        echo "Example:"
        echo "  --model-command-template \"uv run python scripts/model_gtp_engine.py --model {MODEL_DIR}\""
        exit 1
    fi
    if [[ "$MODEL_COMMAND_TEMPLATE" != *"{MODEL_DIR}"* ]]; then
        echo -e "${RED}Error: --model-command-template must include {MODEL_DIR}.${NC}"
        exit 1
    fi
fi

# Step 2: Data preparation
if [ "$SKIP_DATA" = false ]; then
    echo ""
    echo -e "${YELLOW}=========================================="
    echo "STEP 2: Data Preparation"
    echo "==========================================${NC}"
    ./scripts/prepare_data.sh
else
    echo -e "${YELLOW}Skipping data preparation${NC}"
fi

# Step 3: SFT 9x9 warm start
if [ "$SKIP_SFT" = false ]; then
    echo ""
    echo -e "${YELLOW}=========================================="
    echo "STEP 3: SFT Training (9x9 Warm Start)"
    echo "==========================================${NC}"
    run_train_sft "9x9"
else
    echo -e "${YELLOW}Skipping SFT training${NC}"
fi

# Step 4: Evaluate SFT before GRPO
if [ "$SKIP_SFT" = false ]; then
    run_katago_eval "sft9_before_grpo9" "9x9" "outputs/sft_warm_start" "$EVAL_GAMES"
fi

# Step 5: GRPO 9x9
echo ""
echo -e "${YELLOW}=========================================="
echo "STEP 5: GRPO Training (9x9)"
echo "==========================================${NC}"
run_train_grpo "9x9"

# Step 6: Evaluate GRPO 9x9 + promotion gate
if [ "$RUN_EVAL" = true ]; then
    load_promotion_gate_from_config "config/grpo_9x9.yaml"
    GATE_EVAL_GAMES="$PROMOTION_EVAL_GAMES"
    if [ "$SMOKE_TEST" = true ] && [ "$GATE_EVAL_GAMES" -gt 8 ]; then
        GATE_EVAL_GAMES=8
    fi
    run_katago_eval "grpo9_promotion_gate" "9x9" "outputs/grpo_9x9" "$GATE_EVAL_GAMES"
    evaluate_promotion_gate \
        "$LAST_EVAL_SUMMARY" \
        "$PROMOTION_METRIC" \
        "$PROMOTION_THRESHOLD_ELO" \
        "$PROMOTION_THRESHOLD_WINRATE"
else
    echo ""
    echo -e "${YELLOW}=========================================="
    echo "STEP 6: Promotion Gate"
    echo "==========================================${NC}"
    echo -e "${YELLOW}Skipping promotion gate because --no-eval was set.${NC}"
fi

# Step 7: GRPO 19x19 curriculum stage
echo ""
echo -e "${YELLOW}=========================================="
echo "STEP 7: GRPO Training (19x19 Curriculum)"
echo "==========================================${NC}"
run_train_grpo "19x19"

# Step 8: Evaluate 19x19 gate before full-game GRPO
if [ "$RUN_EVAL" = true ]; then
    load_fullgame_promotion_gate_from_config "config/grpo_19x19.yaml"
    FULLGAME_GATE_EVAL_GAMES="$FULLGAME_PROMOTION_EVAL_GAMES"
    if [ "$SMOKE_TEST" = true ] && [ "$FULLGAME_GATE_EVAL_GAMES" -gt 8 ]; then
        FULLGAME_GATE_EVAL_GAMES=8
    fi
    run_katago_eval "grpo19_fullgame_gate" "19x19" "outputs/grpo_19x19" "$FULLGAME_GATE_EVAL_GAMES"
    if evaluate_promotion_gate \
        "$LAST_EVAL_SUMMARY" \
        "$FULLGAME_PROMOTION_METRIC" \
        "$FULLGAME_PROMOTION_THRESHOLD_ELO" \
        "$FULLGAME_PROMOTION_THRESHOLD_WINRATE"; then
        FULLGAME_GATE_PASSED=true
    else
        FULLGAME_GATE_PASSED=false
    fi
else
    echo ""
    echo -e "${YELLOW}=========================================="
    echo "STEP 8: Full-game Promotion Gate"
    echo "==========================================${NC}"
    echo -e "${YELLOW}Skipping full-game promotion gate because --no-eval was set.${NC}"
fi

# Step 9: Full-game GRPO 19x19 (conditional)
echo ""
echo -e "${YELLOW}=========================================="
echo "STEP 9: Full-Game GRPO (19x19)"
echo "==========================================${NC}"
if [ "$SKIP_FULLGAME" = true ]; then
    echo -e "${YELLOW}Skipping full-game GRPO stage (--skip-fullgame).${NC}"
elif [ "$RUN_EVAL" = false ]; then
    echo -e "${YELLOW}Bypassing full-game promotion gate because --no-eval was set.${NC}"
    run_train_grpo_fullgame
    FULLGAME_COMPLETED=true
    FINAL_MODEL_DIR="outputs/grpo_fullgame_19x19"
elif [ "$FULLGAME_GATE_PASSED" != true ]; then
    echo -e "${YELLOW}Skipping full-game GRPO stage because the promotion gate did not pass.${NC}"
else
    run_train_grpo_fullgame
    FULLGAME_COMPLETED=true
    FINAL_MODEL_DIR="outputs/grpo_fullgame_19x19"
fi

# Step 10: Final evaluation before upload
if [ "$RUN_EVAL" = true ]; then
    if [ "$FULLGAME_COMPLETED" = true ]; then
        run_katago_eval "grpo_fullgame_19x19_final" "19x19" "$FINAL_MODEL_DIR" "$EVAL_GAMES"
    elif [ "$LAST_EVALUATED_MODEL_DIR" != "outputs/grpo_19x19" ]; then
        run_katago_eval "grpo19_final" "19x19" "outputs/grpo_19x19" "$EVAL_GAMES"
    else
        echo -e "${YELLOW}Skipping duplicate 19x19 final eval (already evaluated during full-game gate).${NC}"
    fi
fi

# Step 11: Upload to Hugging Face Hub (optional)
if [ "$UPLOAD_TO_HUB" = true ]; then
    echo ""
    echo -e "${YELLOW}=========================================="
    echo "STEP 11: Upload to Hugging Face Hub"
    echo "==========================================${NC}"

    if [ "$RUN_EVAL" = true ] && [ "$LAST_EVALUATED_MODEL_DIR" != "$FINAL_MODEL_DIR" ]; then
        run_katago_eval "pre_upload_final" "$FINAL_BOARD_SIZE" "$FINAL_MODEL_DIR" "$EVAL_GAMES"
    fi

    if [ ! -d "$FINAL_MODEL_DIR" ]; then
        echo -e "${RED}Error: Model directory not found: $FINAL_MODEL_DIR${NC}"
        exit 1
    fi

    UPLOAD_CMD=(
        uv run python scripts/upload_to_hub.py
        --model "$FINAL_MODEL_DIR"
        --repo-id "$HF_REPO_ID"
        --board-size "$FINAL_BOARD_NUM"
        --base-model "$HF_BASE_MODEL"
        --push-only
    )
    if [ "$HF_PRIVATE" = true ]; then
        UPLOAD_CMD+=(--private)
    fi

    echo "Uploading model to Hugging Face..."
    echo "  - Repo: $HF_REPO_ID"
    echo "  - Model path: $FINAL_MODEL_DIR"
    echo "  - Board size: $FINAL_BOARD_NUM"
    "${UPLOAD_CMD[@]}"
fi

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo -e "${GREEN}=========================================="
echo "PIPELINE COMPLETE"
echo "==========================================${NC}"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Artifacts:"
echo "  - SFT model: outputs/sft_warm_start"
echo "  - GRPO 9x9 model: outputs/grpo_9x9"
echo "  - GRPO 19x19 model: outputs/grpo_19x19"
if [ "$FULLGAME_COMPLETED" = true ]; then
    echo "  - Full-game GRPO model: outputs/grpo_fullgame_19x19"
fi
echo "  - Logs: logs/"
if [ "$RUN_EVAL" = true ] && [ -n "$LAST_EVAL_SUMMARY" ]; then
    echo "  - Last evaluation summary: $LAST_EVAL_SUMMARY"
fi
if [ "$UPLOAD_TO_HUB" = true ]; then
    echo "  - Hugging Face: https://huggingface.co/${HF_REPO_ID}"
fi
