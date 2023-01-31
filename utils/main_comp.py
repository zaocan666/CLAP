import json
import glob
import glog as logger
import os
import torch

from utils.utils import setup_seed, set_log_file, print_args
from utils.filehandling import create_dir_structure
from mmnist.experiment import MMNISTExperiment_impute
from mnistsvhntext.experiment import MNISTSVHNText_impute
from eicu.experiment import eicuExperiment_impute
from celeba.experiment import CelebaExperiment_impute

def main_start(FLAGS):
    create_dir_structure(FLAGS)

    setup_seed(FLAGS.seed)
    # args.target_dir = '_'.join([args.target_dir, args.dataset, args.fed_alg])
    if not FLAGS.debug:
        log_name = 'test.log' if FLAGS.test_only else 'train.log'
        FLAGS.log_pth = os.path.join(FLAGS.target_dir, log_name)
        set_log_file(FLAGS.log_pth, file_only=FLAGS.ssh)
    else:
        logger.info('------------------Debug--------------------')
    # logger.setLevel('DEBUG')

    print_args(FLAGS)
    with open(os.path.join(FLAGS.target_dir, 'args.json'), "w") as f:
        json.dump(vars(FLAGS), f, indent = 2)

    use_cuda = torch.cuda.is_available()
    FLAGS.device = torch.device('cuda' if use_cuda else 'cpu')

    alphabet_path = os.path.join(os.getcwd(), 'alphabet.json')
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))

    if FLAGS.dataset=='polymnist':
        FLAGS.unimodal_datapaths_train = glob.glob(FLAGS.unimodal_datapaths_train+'/*')
        FLAGS.unimodal_datapaths_test = glob.glob(FLAGS.unimodal_datapaths_test+'/*')

        # postprocess flags
        assert len(FLAGS.unimodal_datapaths_train) == len(FLAGS.unimodal_datapaths_test)
        FLAGS.num_mods = len(FLAGS.unimodal_datapaths_train)  # set number of modalities dynamically    

        if FLAGS.div_weight_uniform_content is None:
            FLAGS.div_weight_uniform_content = 1 / (FLAGS.num_mods + 1)
        FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content]
        if FLAGS.div_weight is None:
            FLAGS.div_weight = 1 / (FLAGS.num_mods + 1)
        FLAGS.alpha_modalities.extend([FLAGS.div_weight for _ in range(FLAGS.num_mods)])
        logger.info("alpha_modalities:" + str(FLAGS.alpha_modalities))

        mst = MMNISTExperiment_impute(FLAGS, alphabet,)

    elif FLAGS.dataset=='mnistsvhntext':
        FLAGS.num_mods = 3
        FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content, FLAGS.div_weight_m1_content,
                              FLAGS.div_weight_m2_content, FLAGS.div_weight_m3_content];
        FLAGS.num_features = len(alphabet);
        mst = MNISTSVHNText_impute(FLAGS, alphabet,)

    elif FLAGS.dataset=='celeba':
        FLAGS.num_mods = 2
        FLAGS.num_features = len(alphabet);
        FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content, FLAGS.div_weight_m1_content,
                              FLAGS.div_weight_m2_content];
        mst = CelebaExperiment_impute(FLAGS, alphabet)
        
    elif FLAGS.dataset=='eicu':
        FLAGS.client_num = 14
        FLAGS.num_mods = 13
        FLAGS.alpha_modalities = [1 / (FLAGS.num_mods + 1)]*(FLAGS.num_mods+1)
        logger.info("alpha_modalities:" + str(FLAGS.alpha_modalities))
        mst = eicuExperiment_impute(FLAGS, alphabet)
        
    return FLAGS, mst

def main_end(FLAGS):
    os.makedirs( os.path.join(FLAGS.target_dir, "done"), exist_ok = True)
    