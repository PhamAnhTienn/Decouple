#!/bin/bash
python main-dmd.py \
    --data_missing \
    --model dmd \
    --dataset mosi \
    --mode train \
    --seeds 1111 \
    --model_save_dir ./pt \
    --res_save_dir ./result \
    --log_dir ./log \
    --missing_rate_eval_test 0.5