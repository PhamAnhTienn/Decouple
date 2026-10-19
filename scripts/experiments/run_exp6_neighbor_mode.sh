#!/usr/bin/env bash
# ==============================================================================
# Experiment 6: Specific reliability — conditional-locality assumption
#
# Trains and evaluates 3 neighbor selection variants at missing_rate=0.5:
#   - complement (SARM, default)
#   - self (self-conditioned)
#   - random
#
# After training, runs the analysis script to generate:
#   - Violin/box plots of specific cosine similarity per variant
#   - Table of downstream performance (F1/MAE)
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Configurable overrides
GPU_ID="${GPU_ID:-0}"
SEEDS="${SEEDS:-1111}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
CONFIG_FILE="${CONFIG_FILE:-config/dp_mosi_config.json}"
DATASET="${DATASET:-mosi}"
MISSING_RATE="${MISSING_RATE:-0.5}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/result/exp6_neighbor_mode}"
mkdir -p "$RESULT_ROOT"

VARIANTS=(complement self random)

for variant in "${VARIANTS[@]}"; do
    run_dir="$RESULT_ROOT/$variant/mr$MISSING_RATE"
    model_dir="$run_dir/models"
    log_dir="$run_dir/logs"
    mkdir -p "$model_dir" "$log_dir"

    echo "============================================================"
    echo "Exp 6 | Variant=$variant | missing_rate=$MISSING_RATE | seeds=[$SEEDS]"
    echo "============================================================"

    python main.py \
        --model dp \
        --dataset "$DATASET" \
        --config_file "$CONFIG_FILE" \
        --data_dir "$DATA_DIR" \
        --data_missing \
        --neighbor_mode "$variant" \
        --missing_rate_eval_test "$MISSING_RATE" \
        --seeds $SEEDS \
        --gpu_ids "$GPU_ID" \
        --num_workers "$NUM_WORKERS" \
        --model_save_dir "$model_dir" \
        --res_save_dir "$run_dir" \
        --log_dir "$log_dir" \
        --mode train
done

echo ""
echo "============================================================"
echo "Training complete. Running analysis..."
echo "============================================================"

python scripts/experiments/analyze_exp6.py \
    --input-root "$RESULT_ROOT" \
    --output-dir "$RESULT_ROOT/figures" \
    --dataset "$DATASET" \
    --seeds $SEEDS \
    --missing-rate "$MISSING_RATE"

echo "Done. Outputs: $RESULT_ROOT/figures"
