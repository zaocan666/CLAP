# CLAP: Collaborative Adaptation for Checkerboard Learning
the implementation of CLAP.

## introduction for files
* mmnist: dataset, networks and running files for the PolyMNIST dataset.
* mnistsvhntext: dataset, networks and running files for the MNIST-SVHN-TEXT dataset.
* celeba: dataset, networks and running files for the CelebA dataset.
* eicu: dataset, networks and running files for the eICU dataset.
* modalities: modality interface of the datasets.
* eval_metrics: evaluation of the model, including reconstruction MSE, generation coherence, log-likelihood and latent space classification.
* run_script/run_epochs_vaes.py: training script of all methods for all the datasets.
* utils: needed function for the implementation.

## requirements
the needed libraries are in requirements.txt

## dataset preparation:
The preparation of the PolyMNIST, MNIST-SVHN-TEXT, CelebA datasets follows https://github.com/thomassutter/MoPoE.

### eICU
eICU data set is not directly available and detailed description is in the paper https://arxiv.org/abs/2108.08435

## experiments
The scripts for all the datasets are provided as run_{dataset_name}.sh, change the variable 'dataset' to the dataset name in ```main_impute.py``` and run the script.

    sh run_polymnist.sh
    sh run_mst.sh
    sh run_celeba.sh
    sh run_eicu.sh

the parameters in the script file are as follows:
### parameters
- impute_method: method used to run the script, can be 'poe', 'moe', 'joint_elob', or 'collaborate'.
- beta: regularization parameter in the kl divergence, $\beta$.
- class_dim: dimension of the latent vector.
- factorized_representation: whether to use modality-specific latent space, only used in the CelebA dataset.
- client_num: number of clients.
- calc_nll: whether to evaluate with log-likelihoood.
- eval_lr: whether to evaluate with latent space classification.
- use_clf: whether to evaluate with generation coherence.
- calc_mse: whether to evaluate with reconstruction MSE.