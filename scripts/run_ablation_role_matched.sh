#!/bin/bash
# ==============================================================================
# Script: Run Role-matched Interaction Ablation (Ablation 9)
# Variants:
#   1. full_sarm       (Shared: Consensus, Specific: Directional) - Proposed
#   2. generic         (Shared: Weighted, Specific: Weighted)
#   3. consensus_all   (Shared: Consensus, Specific: Consensus)
#   4. directional_all (Shared: Directional, Specific: Directional)
#   5. role_swapped    (Shared: Directional, Specific: Consensus)
# ==============================================================================

DATASET=${1:-"mosi"}
CONFIG_FILE=${2:-"config/dp_mosi_config.json"}
DATA_DIR=${3:-""}
GPU_ID=${4:-0}

VARIANTS=("full_sarm" "generic" "consensus_all" "directional_all" "role_swapped")
MISSING_RATES=(0.5 0.8)
SEEDS=(1111)

for variant in "${VARIANTS[@]}"; do
    echo "======================================================================"
    echo "  RUNNING INTERACTION ABLATION VARIANT: ${variant}"
    echo "======================================================================"

    for mr in "${MISSING_RATES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "[Variant: ${variant} | MR: ${mr} | Seed: ${seed}]"

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
                --interaction_ablation "${variant}" \
                --gpu_ids "${GPU_ID}" \
                --num_workers 0 \
                --model_save_dir "./ablation_role_matched/pt" \
                --res_save_dir "./ablation_role_matched/result" \
                --log_dir "./ablation_role_matched/log/${variant}" \
                ${DATA_DIR_ARG}
        done
    done
done

echo "Generating summary table..."
python scripts/run_ablation_role_matched.py --skip_train --dataset "${DATASET}" --base_save_dir "./ablation_role_matched"
