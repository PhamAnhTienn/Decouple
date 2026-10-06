python main.py \
    --data_missing \
    --model dp \
    --config_file "/home/tien/NCKH/DecAlign/config/dp_mosi_config.json" \
    --dataset mosi \
    --mode train \
    --seeds 1111 \
    --model_save_dir ./pt \
    --res_save_dir ./result \
    --log_dir ./log \
    --missing_rate_eval_test 0.0


python main.py \
    --data_missing \
    --seeds 1111 \
    --model dp \
    --config_file "/home/tien/NCKH/DecAlign/config/dp_mosi_config.json" \
    --dataset mosi \
    --mode uncertainty \
    --missing_rate_eval_test 0.0

for seed in 1111 1112 1113 1114 1115; do
    python main.py \
        --data_missing \
        --model dp \
        --config_file "/home/tien/NCKH/DecAlign/config/dp_mosi_config.json" \
        --dataset mosi \
        --mode train \
        --seeds $seed \
        --model_save_dir ./pt \
        --res_save_dir ./result \
        --log_dir ./log \
        --missing_rate_eval_test 0.9
done