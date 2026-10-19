#!/usr/bin/env bash
# ==============================================================================
# Experiment 7: Joint gate γ — why normalized reliability alone isn't enough
#
# Compares Full SARM (with γ gate) vs γ=1 (disable_gamma, only normalized
# reliability weights) under "all-low" stress conditions.
#
# Missing rates: 0.5, 0.7, 0.8, 0.9
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
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/result/exp7_gamma_gate}"
mkdir -p "$RESULT_ROOT"

MISSING_RATES=(0.5 0.7 0.8 0.9)

# ---- Full SARM (γ enabled, default) ----
for rate in "${MISSING_RATES[@]}"; do
    run_dir="$RESULT_ROOT/full_sarm/mr$rate"
    model_dir="$run_dir/models"
    log_dir="$run_dir/logs"
    mkdir -p "$model_dir" "$log_dir"

    echo "============================================================"
    echo "Exp 7 | Full SARM (γ enabled) | missing_rate=$rate | seeds=[$SEEDS]"
    echo "============================================================"

    python main.py \
        --model dp \
        --dataset "$DATASET" \
        --config_file "$CONFIG_FILE" \
        --data_dir "$DATA_DIR" \
        --data_missing \
        --missing_rate_eval_test "$rate" \
        --seeds $SEEDS \
        --gpu_ids "$GPU_ID" \
        --num_workers "$NUM_WORKERS" \
        --model_save_dir "$model_dir" \
        --res_save_dir "$run_dir" \
        --log_dir "$log_dir" \
        --mode train
done

# ---- γ = 1 (disabled gate, only normalized weights) ----
for rate in "${MISSING_RATES[@]}"; do
    run_dir="$RESULT_ROOT/no_gamma/mr$rate"
    model_dir="$run_dir/models"
    log_dir="$run_dir/logs"
    mkdir -p "$model_dir" "$log_dir"

    echo "============================================================"
    echo "Exp 7 | No γ (disable_gamma) | missing_rate=$rate | seeds=[$SEEDS]"
    echo "============================================================"

    python main.py \
        --model dp \
        --dataset "$DATASET" \
        --config_file "$CONFIG_FILE" \
        --data_dir "$DATA_DIR" \
        --data_missing \
        --disable_gamma \
        --missing_rate_eval_test "$rate" \
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

python scripts/experiments/analyze_exp7.py \
    --input-root "$RESULT_ROOT" \
    --output-dir "$RESULT_ROOT/figures" \
    --dataset "$DATASET" \
    --seeds $SEEDS \
    --missing-rates "${MISSING_RATES[@]}"

echo "Done. Outputs: $RESULT_ROOT/figures"
