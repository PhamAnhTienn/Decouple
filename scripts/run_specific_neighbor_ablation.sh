#!/usr/bin/env bash
# Train/evaluate Complement, Self-conditioned, and Random specific-neighbour
# selection at r = 0.0, 0.5, 0.8, then create the coherence figure/table.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Optional overrides, e.g. GPU_ID=1 SEEDS="1111 1112 1113" bash scripts/...
GPU_ID="${GPU_ID:-0}"
SEEDS="${SEEDS:-1111}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
CONFIG_FILE="${CONFIG_FILE:-config/dp_mosi_config.json}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/result/specific_neighbor_ablation}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RESULT_ROOT/.mplconfig}"
mkdir -p "$RESULT_ROOT" "$MPLCONFIGDIR"

VARIANTS=(complement self random)
MISSING_RATES=(0.0 0.5 0.8)

for variant in "${VARIANTS[@]}"; do
    for rate in "${MISSING_RATES[@]}"; do
        run_dir="$RESULT_ROOT/$variant/mr$rate"
        model_dir="$run_dir/models"
        log_dir="$run_dir/logs"
        mkdir -p "$model_dir" "$log_dir"

        echo "============================================================"
        echo "Variant=$variant | missing rate=$rate | seeds=[$SEEDS]"
        echo "============================================================"

        # Training also evaluates the best checkpoint.  main.py rebuilds the
        # train bank after that reload, so this test uses train-bank neighbours.
        python main.py \
            --model dp \
            --dataset mosi \
            --config_file "$CONFIG_FILE" \
            --data_dir "$DATA_DIR" \
            --data_missing \
            --neighbor_mode "$variant" \
            --missing_rate_eval_test "$rate" \
            --seeds $SEEDS \
            --gpu_ids "$GPU_ID" \
            --num_workers "$NUM_WORKERS" \
            --model_save_dir "$model_dir" \
            --res_save_dir "$run_dir" \
            --log_dir "$log_dir" \
            --mode train
    done
done

# The figure/table use one seed at a time.  With multiple seeds, invoke this
# once per seed (or aggregate their emitted CSVs explicitly in a follow-up).
read -r -a seed_list <<< "$SEEDS"
python scripts/analyze_specific_neighbor_ablation.py \
    --input-root "$RESULT_ROOT" \
    --output-dir "$RESULT_ROOT/summary" \
    --dataset mosi \
    --seed "${seed_list[0]}" \
    --missing-rates "${MISSING_RATES[@]}"

echo "Done. Outputs: $RESULT_ROOT/summary"