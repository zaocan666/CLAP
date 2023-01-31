DIR_DATA="/data/"
DIR_EXPERIMENT="$PWD/runs/tmp"  # NOTE: experiment logs are written here

CUDA_VISIBLE_DEVICES=3 python main_impute.py \
            --dataset mnistsvhntext \
            --dir_data "$DIR_DATA/mnistsvhntext" \
            --target_dir="$DIR_EXPERIMENT" \
            --test_missing_ratio 0.33 \
            --impute_method poe \
			--class_dim=20 \
            --beta=0.1 \
            --batch_size=256 \
            --initial_learning_rate=0.001 \
            --eval_freq=25 \
            --data_multiplications=20 \
            --num_hidden_layers=1 \
            --end_epoch=150 \
            --class_per_user 6 \
            --client_num 10 \
            --k_single 0 \
            --calc_nll \
            --eval_lr \
            --use_clf \