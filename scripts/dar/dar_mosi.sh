#!/bin/bash
python main.py \
    --data_missing \
    --model dar \
    --config_file "/home/tien/NCKH/DecAlign/config/dar_mosi_config.json" \
    --dataset mosi \
    --mode train \
    --seeds 1111 \
    --model_save_dir ./pt \
    --res_save_dir ./result \
    --log_dir ./log \
    --missing_rate_eval_test 0.0

python main.py \
    --data_missing \
    --model dar \
    --config_file "/home/tien/NCKH/DecAlign/config/dar_sims_config.json" \
    --dataset sims \
    --mode train \
    --seeds 1111 \
    --model_save_dir ./pt \
    --res_save_dir ./result \
    --log_dir ./log \
    --missing_rate_eval_test 0.0