LIKELIHOOD="laplace"
DIR_DATA="/data/"
DIR_EXPERIMENT="$PWD/runs/tmp"  # NOTE: experiment logs are written here

CUDA_VISIBLE_DEVICES=2 python main_impute.py \
            --dataset polymnist \
            --unimodal-datapaths-train "$DIR_DATA/MMNIST/train" \
            --unimodal-datapaths-test "$DIR_DATA/MMNIST/test" \
            --target_dir="$DIR_EXPERIMENT" \
            --impute_method moe \
            --class_dim=512 \
            --beta=0.1 \
            --likelihood_mnist=$LIKELIHOOD \
            --batch_size=256 \
            --initial_learning_rate=0.0005 \
            --eval_freq=25 \
            --num_hidden_layers=1 \
            --end_epoch=300 \
            --mm_vae_save vae \
            --class_per_user 6 \
            --client_num 5 \
            --k_single 0 \
            --calc_nll \
            --eval_lr \
            --use_clf \
