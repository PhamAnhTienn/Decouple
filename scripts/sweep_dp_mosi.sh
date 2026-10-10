#!/bin/bash
# ============================================================================
# Hyperparameter sweep for DP model on MOSI (missing_rate=0.5)
#
# Sweeps one hyperparameter at a time (others fixed at their defaults):
#   λ_CPA  ∈ {0, 0.05, 0.1*, 0.2}     config key: lambda_cpa    (default: 0.1)
#   K_p    ∈ {3, 5, 7*, 9}             config key: nmb_prototypes (default: 7)
#   τ      ∈ {0.25, 0.5, 1*, 2}        config key: tau_c          (default: 1.0)
#   K_n    ∈ {5, 10*, 20, 50}          config key: K_neighbors    (default: 10)
#
# * = default value → skipped (reuse existing results)
#
# Seeds: 1111 1112 1113
# Dataset: mosi, missing_rate: 0.5
# Results saved to: ./result/ablation/<param_name>_<value>/
# ============================================================================

set -e  # exit on error

SEEDS="1111 1112 1113"
DATASET="mosi"
MISSING_RATE=0.5
CONFIG_FILE="/home/tien/NCKH/DecAlign/config/dp_mosi_config.json"
MODEL="dp"

run_experiment() {
    local desc="$1"
    local res_dir="$2"
    shift 2
    local overrides=("$@")

    mkdir -p "$res_dir"

    for seed in $SEEDS; do
        echo "========================================"
        echo " $desc | Seed: $seed"
        echo " Results → $res_dir"
        echo "========================================"

        python main.py \
            --data_missing \
            --model $MODEL \
            --config_file "$CONFIG_FILE" \
            --dataset $DATASET \
            --mode train \
            --seeds $seed \
            --model_save_dir ./pt \
            --res_save_dir "$res_dir" \
            --log_dir ./log \
            --missing_rate_eval_test $MISSING_RATE \
            "${overrides[@]}"
    done
}

# ============================================================================
# 1. Sweep λ_CPA (lambda_cpa) — default: 0.1 → skip 0.1
# ============================================================================
echo "############################################################"
echo "# Sweep λ_CPA (lambda_cpa)"
echo "############################################################"

for lambda_cpa in 0 0.05 0.2; do
    run_experiment "lambda_cpa=$lambda_cpa" \
        "./result/ablation/lambda_cpa_${lambda_cpa}" \
        --config_override "lambda_cpa=$lambda_cpa"
done

# ============================================================================
# 2. Sweep K_p (nmb_prototypes) — default: 7 → skip 7
# ============================================================================
echo "############################################################"
echo "# Sweep K_p (nmb_prototypes)"
echo "############################################################"

for kp in 3 5 9; do
    run_experiment "nmb_prototypes=$kp" \
        "./result/ablation/nmb_prototypes_${kp}" \
        --config_override "nmb_prototypes=$kp"
done

# ============================================================================
# 3. Sweep τ (tau_c) — default: 1.0 → skip 1.0
# ============================================================================
echo "############################################################"
echo "# Sweep τ (tau_c)"
echo "############################################################"

for tau in 0.25 0.5 2; do
    run_experiment "tau_c=$tau" \
        "./result/ablation/tau_c_${tau}" \
        --config_override "tau_c=$tau"
done

# ============================================================================
# 4. Sweep K_n (K_neighbors) — default: 10 → skip 10
# ============================================================================
echo "############################################################"
echo "# Sweep K_n (K_neighbors)"
echo "############################################################"

for kn in 5 20 50; do
    run_experiment "K_neighbors=$kn" \
        "./result/ablation/K_neighbors_${kn}" \
        --config_override "K_neighbors=$kn"
done

echo ""
echo "############################################################"
echo "# All sweeps complete!"
echo "############################################################"
