#!/usr/bin/env bash
# =====================================================================
# Experiment 3: Role-Conditioned Reliability Ablation
# =====================================================================
# Proves that shared and specific representations should NOT share a
# single reliability score — each score must match its semantic role.
#
# Variants (excluding 'full' which already has trained checkpoints):
#   uniform        – r^c = r^s = 1 (no reliability)
#   shared_only    – both modules use r^c
#   specific_only  – both modules use r^s
#   averaged       – both modules use (r^c + r^s) / 2
#   swapped        – shared→r^s, specific→r^c
#
# Missing rates : 0.5, 0.8
# Seeds         : 1111, 1112, 1113
# =====================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/config/dp_mosi_config.json"
MODEL_SAVE_DIR="${PROJECT_ROOT}/pt"
RES_SAVE_DIR="${PROJECT_ROOT}/result"
LOG_DIR="${PROJECT_ROOT}/log"

MODES=("uniform" "shared_only" "specific_only" "averaged" "swapped")
MISSING_RATES=("0.5" "0.8")
SEEDS=("1111" "1112" "1113")

for mode in "${MODES[@]}"; do
    for mr in "${MISSING_RATES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "========================================================"
            echo "  reliability_mode=${mode} | missing_rate=${mr} | seed=${seed}"
            echo "========================================================"

            python "${PROJECT_ROOT}/main.py" \
                --data_missing \
                --model dp \
                --config_file "${CONFIG_FILE}" \
                --dataset mosi \
                --mode train \
                --seeds ${seed} \
                --model_save_dir "${MODEL_SAVE_DIR}" \
                --res_save_dir "${RES_SAVE_DIR}/role_reliability/${mode}" \
                --log_dir "${LOG_DIR}" \
                --missing_rate_eval_test ${mr} \
                --config_override "reliability_mode=${mode}"
        done
    done
done

echo ""
echo "============================================"
echo "  All role-conditioned reliability runs done"
echo "============================================"
