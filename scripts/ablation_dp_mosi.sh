#!/bin/bash
# =============================================================================
# Ablation Study for DP model on MOSI dataset
# =============================================================================
# 4 ablation variants, each with 1 seed (1111), missing rates 0.0 to 0.9
#
# Variants:
#   1. w/o L_ort               → ablate_ort=True
#   2. w/o L_cpa               → ablate_cpa=True
#   3. w/o Shared Calibration  → ablate_shared_calibration=True
#   4. w/o Specific Calibration→ ablate_specific_calibration=True
# =============================================================================

set -e

SEED=1116
CONFIG="/home/tien/NCKH/DecAlign/config/dp_mosi_config.json"
MISSING_RATES="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"

# ──────────────────────────────────────────────────────────────────────────────
# Ablation 1: without L_ort
# ──────────────────────────────────────────────────────────────────────────────
ABLATION_NAME="wo_L_ort"
echo "============================================"
echo "  Ablation: ${ABLATION_NAME}"
echo "============================================"

for mr in $MISSING_RATES; do
    echo ">>> [${ABLATION_NAME}] Missing Rate: ${mr} | Seed: ${SEED}"
    python main.py \
        --data_missing \
        --model dp \
        --config_file "${CONFIG}" \
        --dataset mosi \
        --mode train \
        --seeds ${SEED} \
        --model_save_dir ./pt/ablation \
        --res_save_dir ./result/ablation/${ABLATION_NAME} \
        --log_dir ./log/ablation/${ABLATION_NAME} \
        --missing_rate_eval_test ${mr} \
        --config_override ablate_ort=True
done

# # ──────────────────────────────────────────────────────────────────────────────
# # Ablation 2: without L_cpa
# # ──────────────────────────────────────────────────────────────────────────────
# ABLATION_NAME="wo_L_cpa"
# echo "============================================"
# echo "  Ablation: ${ABLATION_NAME}"
# echo "============================================"

# for mr in $MISSING_RATES; do
#     echo ">>> [${ABLATION_NAME}] Missing Rate: ${mr} | Seed: ${SEED}"
#     python main.py \
#         --data_missing \
#         --model dp \
#         --config_file "${CONFIG}" \
#         --dataset mosi \
#         --mode train \
#         --seeds ${SEED} \
#         --model_save_dir ./pt/ablation \
#         --res_save_dir ./result/ablation/${ABLATION_NAME} \
#         --log_dir ./log/ablation/${ABLATION_NAME} \
#         --missing_rate_eval_test ${mr} \
#         --config_override ablate_cpa=True
# done

# ──────────────────────────────────────────────────────────────────────────────
# Ablation 3: without Shared Score & Calibrated Shared Fusion
# ──────────────────────────────────────────────────────────────────────────────
ABLATION_NAME="wo_shared_calibration"
echo "============================================"
echo "  Ablation: ${ABLATION_NAME}"
echo "============================================"

for mr in $MISSING_RATES; do
    echo ">>> [${ABLATION_NAME}] Missing Rate: ${mr} | Seed: ${SEED}"
    python main.py \
        --data_missing \
        --model dp \
        --config_file "${CONFIG}" \
        --dataset mosi \
        --mode train \
        --seeds ${SEED} \
        --model_save_dir ./pt/ablation \
        --res_save_dir ./result/ablation/${ABLATION_NAME} \
        --log_dir ./log/ablation/${ABLATION_NAME} \
        --missing_rate_eval_test ${mr} \
        --config_override ablate_shared_calibration=True
done

# ──────────────────────────────────────────────────────────────────────────────
# Ablation 4: without Specific Score & Calibrated Specific Feature
# ──────────────────────────────────────────────────────────────────────────────
ABLATION_NAME="wo_specific_calibration"
echo "============================================"
echo "  Ablation: ${ABLATION_NAME}"
echo "============================================"

for mr in $MISSING_RATES; do
    echo ">>> [${ABLATION_NAME}] Missing Rate: ${mr} | Seed: ${SEED}"
    python main.py \
        --data_missing \
        --model dp \
        --config_file "${CONFIG}" \
        --dataset mosi \
        --mode train \
        --seeds ${SEED} \
        --model_save_dir ./pt/ablation \
        --res_save_dir ./result/ablation/${ABLATION_NAME} \
        --log_dir ./log/ablation/${ABLATION_NAME} \
        --missing_rate_eval_test ${mr} \
        --config_override ablate_specific_calibration=True
done

echo ""
echo "============================================"
echo "  All ablation experiments completed!"
echo "============================================"
echo "Results saved to: ./result/ablation/<variant>/normal/mosi.csv"
