

import argparse

parser = argparse.ArgumentParser()

# TRAINING
parser.add_argument('--batch_size', type=int, default=256, help="batch size for training")
parser.add_argument('--initial_learning_rate', type=float, default=0.001, help="starting learning rate")
parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")
parser.add_argument('--start_epoch', type=int, default=0, help="flag to set the starting epoch for training")
parser.add_argument('--end_epoch', type=int, default=1000, help="flag to indicate the final epoch of training")
parser.add_argument('--test_only', default=False, action="store_true", help="")

# DATA DEPENDENT
parser.add_argument('--dataset', type=str, default='eicu', help="[polymnist, mnistsvhntext]")
parser.add_argument('--class_dim', type=int, default=512, help="dimension of common factor latent space")

# SAVE and LOAD
parser.add_argument('--mm_vae_save', type=str, default='vae', help="model save for vae_bimodal")
parser.add_argument('--load_saved', action="store_true", default=False, help="flag to indicate if a saved model will be loaded")
parser.add_argument('--checkpoint_pth', type=str, default='', help="")

# DIRECTORIES
# clfs
parser.add_argument('--dir_clf', type=str, default='//trained_classifiers', help="directory where clf is stored")
# data
parser.add_argument('--dir_data', type=str, default='/eicu_fedmainfold', help="directory where data is stored")
# experiments
parser.add_argument('--dir_experiment', type=str, default='/tmp/exp', help="directory to save generated samples in")

# EVALUATION
parser.add_argument('--calc_mse', default=False, action="store_true",
                    help="flag to indicate if mse should be calculated")
parser.add_argument('--use_clf', default=False, action="store_true",
                    help="flag to indicate if generates samples should be classified")
parser.add_argument('--calc_nll', default=False, action="store_true",
                    help="flag to indicate calculation of nll")
parser.add_argument('--eval_lr', default=False, action="store_true",
                    help="flag to indicate evaluation of lr")
parser.add_argument('--save_figure', default=False, action="store_true",
                    help="flag to indicate if figures should be saved to disk (in addition to tensorboard logs)")
parser.add_argument('--eval_freq', type=int, default=25, help="frequency of evaluation of latent representation of generative performance (in number of epochs)")
parser.add_argument('--num_training_samples_lr', type=int, default=500,
                    help="number of training samples to train the lr clf")
parser.add_argument('--test_set_ratio', type=float, default=1.0,
                    help="test set ratio")

#multimodal
parser.add_argument('--impute_method', type=str, default='joint_elbo', help='choose method for training the model')
parser.add_argument('--poe_unimodal_elbos', type=bool, default=True, help="unimodal_klds")
parser.add_argument('--factorized_representation', action='store_true', default=False, help="factorized_representation")

# LOSS TERM WEIGHTS
parser.add_argument('--beta', type=float, default=5.0, help="default weight of sum of weighted divergence terms")
parser.add_argument('--beta_style', type=float, default=1.0, help="default weight of sum of weighted style divergence terms")
parser.add_argument('--beta_content', type=float, default=1.0, help="default weight of sum of weighted content divergence terms")
parser.add_argument('--k_single', type=float, default=1.0, help="")

# impute
parser.add_argument('--test_missing_ratio', type=float, default=0.4, help="default missing ratio of test set")

# federated
parser.add_argument('--client_num', type=int, default=5, help="")
parser.add_argument('--class_per_user', type=int, default=6, help="")

# search
parser.add_argument("--seed", type=int, default=1313, help="Random seed (--i is always added to it)")
parser.add_argument("--ssh", action="store_true", help="whether is run by search")
parser.add_argument("--debug", action="store_true", help="debug or not")
parser.add_argument('--host_name', type=str, default='host', help="name of the host")
parser.add_argument("--target_dir", type=str, default="runs/tmp", help="model name")

