#!/bin/bash
# ==============================================================================
# Sequential Benchmark Runner for TFRNet on CH-SIMS (SIMS)
# Loop structure: For each seed, runs missing rates 0.0 -> 0.9
# Usage:
#   ./scripts/run_tfr_net_sims.sh [gpu_id] [data_dir] [seeds] [batch_size]
# Examples:
#   # Run only seed 1111:
#   ./scripts/run_tfr_net_sims.sh 0 ./data "1111" 16
#
#   # Run 5 seeds (1111 to 1115):
#   ./scripts/run_tfr_net_sims.sh 0 ./data "1111 1112 1113 1114 1115" 16
# ==============================================================================

set -e

GPU_ID=${1:-0}
DATA_DIR=${2:-./data}
SEEDS_ARG=${3:-"1111"}
BATCH_SIZE=${4:-16}
DATASET="sims"

# Split seeds argument into array
read -ra SEEDS <<< "${SEEDS_ARG}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DECOUPLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_FILE="${DECOUPLE_ROOT}/config/tfr_net_sims_config.json"
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "[ERROR] Config file not found: ${CONFIG_FILE}"
    exit 1
fi

RATES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

echo "======================================================================"
echo "Starting TFRNet Benchmark: Dataset=${DATASET}, GPU=${GPU_ID}"
echo "Config: ${CONFIG_FILE}"
echo "Seeds: ${SEEDS[*]}"
echo "Missing Rates: ${RATES[*]}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Data Dir: ${DATA_DIR}"
echo "======================================================================"

cd "${DECOUPLE_ROOT}"

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "######################################################################"
    echo "  RUNNING SEED: ${seed} (Missing Rates: 0.0 -> 0.9)"
    echo "######################################################################"

    MODEL_SAVE_DIR="./pt/seed_${seed}"
    RES_SAVE_DIR="./result/seed_${seed}"
    LOG_DIR="./log/seed_${seed}"

    mkdir -p "${MODEL_SAVE_DIR}" "${RES_SAVE_DIR}" "${LOG_DIR}"

    for rate in "${RATES[@]}"; do
        CKPT="${MODEL_SAVE_DIR}/tfr_net-${DATASET}-missing${rate}.pth"

        echo ""
        echo ">>> [Seed: ${seed}] | Missing Rate: ${rate}"

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
            --log_dir "${LOG_DIR}" \
            --batch_size "${BATCH_SIZE}"
    done
done

echo ""
echo "======================================================================"
echo "All training completed!"
echo "Generating summary table..."
echo "======================================================================"

python scripts/run_tfr_net_missing.py \
    --dataset "${DATASET}" \
    --config_file "${CONFIG_FILE}" \
    --data_dir "${DATA_DIR}" \
    --seeds "${SEEDS[@]}" \
    --missing_rates "${RATES[@]}" \
    --resume

echo "Done!"
