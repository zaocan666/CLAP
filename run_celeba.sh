DIR_DATA="/data/"
DIR_EXPERIMENT="$PWD/runs/tmp"  # NOTE: experiment logs are written here

CUDA_VISIBLE_DEVICES=4 python main_impute.py \
            --dataset celeba \
            --dir_data "$DIR_DATA/CelebA" \
            --target_dir="$DIR_EXPERIMENT" \
            --test_missing_ratio 0.5 \
            --impute_method joint_elbo \
			--class_dim=32 \
            --beta=0.4 \
            --beta_style=2.0 \
            --beta_content=1.0 \
            --batch_size=256 \
            --initial_learning_rate=0.0001 \
            --eval_freq=25 \
            --factorized_representation \
            --end_epoch=200 \
            --mm_vae_save vae \
            --client_num 5 \
            --k_single 0.05 \
            --calc_nll \
            --eval_lr \
            --use_clf \