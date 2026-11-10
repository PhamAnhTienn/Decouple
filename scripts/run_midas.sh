#!/bin/bash
# Training script for MIDAS model
# Usage: ./scripts/run_midas.sh [dataset] [gpu_id] [missing_rate]
#   dataset: mosi, mosei, or sims (default: mosi)
#   gpu_id: GPU device ID (default: 0)
#   missing_rate: Missing rate for evaluation (default: 0.0)

set -e

# Default values
DATASET=${1:-mosi}
GPU_ID=${2:-0}
MISSING_RATE=${3:-0.0}
DATA_DIR=${DATA_DIR:-./data}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_FILE="${ROOT_DIR}/config/midas_${DATASET}_config.json"

echo "=================================================="
echo "Training MIDAS on ${DATASET} dataset"
echo "GPU: ${GPU_ID} | Missing Rate: ${MISSING_RATE}"
echo "Config: ${CONFIG_FILE}"
echo "=================================================="

cd "${ROOT_DIR}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    --data_missing \
    --config_file "${CONFIG_FILE}" \
    --model midas \
    --dataset ${DATASET} \
    --data_dir ${DATA_DIR} \
    --mode train \
    --seeds 1111 1112 1113 1114 1115 \
    --model_save_dir ./pt \
    --res_save_dir ./result \
    --log_dir ./log \
    --missing_rate_eval_test ${MISSING_RATE}

echo "Training completed!"

