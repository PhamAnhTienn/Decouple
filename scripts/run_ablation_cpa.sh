#!/bin/bash
# ==============================================================================
# Script: Run CPA Supporting Mechanism Ablation (Ablation 10)
# Configurations:
#   1. Full Model (+ CPA):  lambda_cpa = 0.1
#   2. w/o CPA:             lambda_cpa = 0.0
# Missing Rates: 0.5 (Moderate) and 0.8 (Severe)
# ==============================================================================

DATASET=${1:-"mosi"}
CONFIG_FILE=${2:-"config/dp_mosi_config.json"}
DATA_DIR=${3:-""}
GPU_ID=${4:-0}

MISSING_RATES=(0.5 0.8)
SEEDS=(1111)

# 1. Full Model (+ CPA: lambda_cpa=0.1)
echo "======================================================================"
echo "  RUNNING CONFIG: Full Model (+ CPA) - lambda=0.1"
echo "======================================================================"
for mr in "${MISSING_RATES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        DATA_DIR_ARG=""
        if [ -n "$DATA_DIR" ]; then
            DATA_DIR_ARG="--data_dir ${DATA_DIR}"
        fi
        python main.py \
            --data_missing \
            --model dp \
            --dataset "${DATASET}" \
            --config_file "${CONFIG_FILE}" \
            --mode train \
            --seeds "${seed}" \
            --missing_rate_eval_test "${mr}" \
            --interaction_ablation "full_sarm" \
            --lambda_cpa 0.1 \
            --gpu_ids "${GPU_ID}" \
            --num_workers 0 \
            --model_save_dir "./ablation_cpa/pt" \
            --res_save_dir "./ablation_cpa/result" \
            --log_dir "./ablation_cpa/log/full_cpa" \
            ${DATA_DIR_ARG}
    done
done

# 2. w/o CPA (lambda_cpa=0.0)
echo "======================================================================"
echo "  RUNNING CONFIG: w/o CPA - lambda=0.0"
echo "======================================================================"
for mr in "${MISSING_RATES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        DATA_DIR_ARG=""
        if [ -n "$DATA_DIR" ]; then
            DATA_DIR_ARG="--data_dir ${DATA_DIR}"
        fi
        python main.py \
            --data_missing \
            --model dp \
            --dataset "${DATASET}" \
            --config_file "${CONFIG_FILE}" \
            --mode train \
            --seeds "${seed}" \
            --missing_rate_eval_test "${mr}" \
            --interaction_ablation "full_sarm" \
            --lambda_cpa 0.0 \
            --gpu_ids "${GPU_ID}" \
            --num_workers 0 \
            --model_save_dir "./ablation_cpa/pt" \
            --res_save_dir "./ablation_cpa/result" \
            --log_dir "./ablation_cpa/log/no_cpa" \
            ${DATA_DIR_ARG}
    done
done

echo "Generating summary table..."
python scripts/run_ablation_cpa.py --skip_train --dataset "${DATASET}" --base_save_dir "./ablation_cpa"
