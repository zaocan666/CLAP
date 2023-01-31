DIR_EXPERIMENT="$PWD/runs/tmp"  # NOTE: experiment logs are written here

CUDA_VISIBLE_DEVICES=3 python main_impute.py \
            --dataset eicu \
            --target_dir="$DIR_EXPERIMENT" \
            --test_missing_ratio 0.15 \
            --impute_method collaborate \
			--class_dim=32 \
            --beta=2.0 \
            --batch_size=256 \
            --initial_learning_rate=0.001 \
            --num_training_samples_lr 400 \
            --eval_freq=25 \
            --end_epoch=150 \
            --k_single 0.05 \
            --calc_nll \
            --eval_lr \
            --calc_mse \
