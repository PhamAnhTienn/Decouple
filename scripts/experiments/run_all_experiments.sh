#!/bin/bash
# =============================================================================
# Run all DecAlign / SARM experiments
# Usage:  bash scripts/experiments/run_all_experiments.sh [DATASET] [GPU_ID]
#
# Checkpoints expected in: ./checkpoint/dp-{dataset}-missing{rate}.pth
# Results saved to:        ./result/experiments/exp{N}/
# =============================================================================

set -e

DATASET=${1:-mosi}
GPU=${2:-0}
DATA_DIR="./data"
CKPT_DIR="./checkpoint"
WORKERS=4
SEED=1111

echo "========================================"
echo " DecAlign Experiment Suite"
echo " Dataset: $DATASET | GPU: $GPU"
echo "========================================"

# ── Exp 4: Reliability Validity ──
echo ""
echo ">>> Exp 4: Reliability Validity"
python scripts/experiments/exp04_reliability_validity.py \
    --dataset $DATASET \
    --data_dir $DATA_DIR \
    --checkpoint_dir $CKPT_DIR \
    --missing_rates 0.0 0.2 0.5 0.7 \
    --gpu_ids $GPU \
    --num_workers $WORKERS \
    --seed $SEED

# ── Exp 5: Selective / Asymmetric Corruption ──
echo ""
echo ">>> Exp 5: Selective / Asymmetric Corruption"
python scripts/experiments/exp05_selective_corruption.py \
    --dataset $DATASET \
    --data_dir $DATA_DIR \
    --checkpoint_dir $CKPT_DIR \
    --base_missing_rate 0.0 \
    --high_rate 0.8 \
    --low_rate 0.2 \
    --gpu_ids $GPU \
    --num_workers $WORKERS \
    --seed $SEED

# ── Exp 8: Sender–Receiver β Behaviour ──
echo ""
echo ">>> Exp 8: Sender-Receiver β (all pairs)"
python scripts/experiments/exp08_sender_receiver_beta.py \
    --dataset $DATASET \
    --data_dir $DATA_DIR \
    --checkpoint_dir $CKPT_DIR \
    --all_pairs \
    --rates 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
    --gpu_ids $GPU \
    --num_workers $WORKERS \
    --seed $SEED

# ── Exp 12: Non-redundancy of uᶜ and uˢ ──
echo ""
echo ">>> Exp 12: Non-redundancy"
python scripts/experiments/exp12_nonredundancy.py \
    --dataset $DATASET \
    --data_dir $DATA_DIR \
    --checkpoint_dir $CKPT_DIR \
    --missing_rates 0.0 0.2 0.5 0.7 \
    --gpu_ids $GPU \
    --num_workers $WORKERS \
    --seed $SEED

# ── Exp 14: Efficiency & Reproducibility ──
echo ""
echo ">>> Exp 14: Efficiency"
python scripts/experiments/exp14_efficiency.py \
    --dataset $DATASET \
    --data_dir $DATA_DIR \
    --checkpoint_dir $CKPT_DIR \
    --missing_rate 0.0 \
    --gpu_ids $GPU \
    --num_workers $WORKERS \
    --seed $SEED

echo ""
echo "========================================"
echo " All experiments complete!"
echo " Results in: ./result/experiments/"
echo "========================================"
