#!/bin/bash
set -e

# Activate environment
eval "$(conda shell.bash hook)"
conda activate baseline

MISSING_RATES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for mr in "${MISSING_RATES[@]}"; do
    echo "=============================================="
    echo "Extracting DMD Uncertainty for MR=${mr}"
    echo "=============================================="
    python main.py \
        --data_missing \
        --seeds 1111 \
        --model dmd \
        --config_file "config/dmd_mosi_config.json" \
        --dataset mosi \
        --mode uncertainty \
        --missing_rate_eval_test "${mr}"
done

echo "Finished all DMD uncertainty extractions!"
