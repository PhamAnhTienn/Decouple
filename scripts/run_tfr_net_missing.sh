#!/bin/bash
# Sequential Benchmark Runner for TFRNet across 5 seeds and missing rates 0.0 to 0.9.
# Usage: ./scripts/run_tfr_net_missing.sh [dataset] [gpu_id] [data_dir]

set -e

DATASET=${1:-mosi}
GPU_ID=${2:-0}
DATA_DIR=${3:-./data}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DECOUPLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_FILE="${DECOUPLE_ROOT}/config/tfr_net_${DATASET}_config.json"
if [ ! -f "${CONFIG_FILE}" ]; then
    CONFIG_FILE="${DECOUPLE_ROOT}/config/tfr_net_mosi_config.json"
fi

SEEDS=(1111 1112 1113 1114 1115)
RATES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

echo "======================================================================"
echo "Starting TFRNet Benchmark: Dataset=${DATASET}, GPU=${GPU_ID}"
echo "Seeds: ${SEEDS[*]}"
echo "Missing Rates: ${RATES[*]}"
echo "======================================================================"

cd "${DECOUPLE_ROOT}"

for seed in "${SEEDS[@]}"; do
    for rate in "${RATES[@]}"; do
        MODEL_SAVE_DIR="./pt/seed_${seed}"
        RES_SAVE_DIR="./result/seed_${seed}"
        LOG_DIR="./log/seed_${seed}"
        CKPT="${MODEL_SAVE_DIR}/tfr_net-${DATASET}-missing${rate}.pth"

        echo ""
        echo ">>> Running Seed: ${seed} | Missing Rate: ${rate}"

        if [ -f "${CKPT}" ]; then
            echo "[Skip] Checkpoint exists: ${CKPT}"
            continue
        fi

        CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
            --model tfr_net \
            --dataset "${DATASET}" \
            --config_file "${CONFIG_FILE}" \
            --data_missing \
            --missing_rate_eval_test "${rate}" \
            --seeds "${seed}" \
            --gpu_ids "${GPU_ID}" \
            --data_dir "${DATA_DIR}" \
            --model_save_dir "${MODEL_SAVE_DIR}" \
            --res_save_dir "${RES_SAVE_DIR}" \
            --log_dir "${LOG_DIR}"
    done
done

echo ""
echo "Benchmark completed successfully!"
