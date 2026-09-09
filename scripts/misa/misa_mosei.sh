for missing_rate in 0.0 0.1 0.2 0.3 0.4 0.5; do
    for seed in 1111 1112 1113 1114 1115; do
        python main.py \
            --data_missing \
            --config_file "/home/tien/NCKH/DecAlign/config/misa_mosei_config.json" \
            --model misa \
            --dataset mosei \
            --mode train \
            --seeds $seed \
            --model_save_dir ./pt \
            --res_save_dir ./result \
            --log_dir ./log \
            --missing_rate_eval_test $missing_rate
    done
done
