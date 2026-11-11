#!/bin/bash
# Infer from a single checkpoint across all missing rates (0.0 to 0.9)
#
# Usage:
#   ./scripts/infer_all.sh <model> <dataset> [checkpoint] [gpu_id] [seeds...]
#
# Examples:
#   # DP on MOSI, auto-detect checkpoint, GPU 0, seed 1111
#   ./scripts/infer_all.sh dp mosi
#
#   # DP on MOSI with specific checkpoint and multiple seeds
#   ./scripts/infer_all.sh dp mosi ./pt/dp-mosi-missing0.0.pth 0 1111 1112 1113 1114 1115
#
#   # MIDAS on MOSI, auto-detect, GPU 1
#   ./scripts/infer_all.sh midas mosi "" 1
#
#   # MISA on MOSI, auto-detect, default GPU, specific seeds
#   ./scripts/infer_all.sh misa mosi "" 0 1111 1112 1113

set -e

MODEL=${1:?Usage: $0 <model> <dataset> [checkpoint] [gpu_id] [seeds...]}
DATASET=${2:-mosi}
CHECKPOINT=${3:-""}
GPU_ID=${4:-0}
DATA_DIR=${DATA_DIR:-./data}

# Remaining args are seeds (default: 1111)
shift 4 2>/dev/null || true
if [ $# -gt 0 ]; then
    SEEDS="$@"
else
    SEEDS="1111"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=================================================="
echo "Inference: ${MODEL} on ${DATASET}"
echo "GPU: ${GPU_ID} | Seeds: ${SEEDS}"
if [ -n "${CHECKPOINT}" ]; then
    echo "Checkpoint: ${CHECKPOINT}"
else
    echo "Checkpoint: auto-detect"
fi
echo "=================================================="

cd "${ROOT_DIR}"

CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python infer.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --data_dir ${DATA_DIR} \
    --seeds ${SEEDS} \
    --model_save_dir ./pt \
    --res_save_dir ./result \
    --log_dir ./log"

# Add checkpoint if specified
if [ -n "${CHECKPOINT}" ]; then
    CMD="${CMD} --checkpoint ${CHECKPOINT}"
fi

# Add config file if model-specific one exists
CONFIG_FILE="${ROOT_DIR}/config/${MODEL}_${DATASET}_config.json"
if [ -f "${CONFIG_FILE}" ]; then
    CMD="${CMD} --config_file ${CONFIG_FILE}"
fi

# Add --config_override data_missing=True
CMD="${CMD} --config_override data_missing=True"

eval ${CMD}

echo "Inference completed!"

