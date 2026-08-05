#!/bin/bash
python main.py \
    --data_missing \
    --model misa \
    --config_file "/home/tien/NCKH/DecAlign/config/misa_mosi_config.json" \
    --dataset mosi \
    --mode train \
    --seeds 1111 1112 1113 1114 1115 \
    --model_save_dir ./pt \
    --res_save_dir ./result \
    --log_dir ./log \
    --missing_rate_eval_test 0.0
