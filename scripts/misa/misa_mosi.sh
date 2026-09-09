#!/bin/bash
for seed in 1111 1112 1113 1114 1115; do
    python main.py \
        --data_missing \
        --config_file "/home/tien/NCKH/DecAlign/config/misa_mosi_config.json" \
        --model misa \
        --dataset mosi \
        --mode train \
        --seeds $seed \
        --model_save_dir ./pt \
        --res_save_dir ./result \
        --log_dir ./log \
        --missing_rate_eval_test 0.1
done
