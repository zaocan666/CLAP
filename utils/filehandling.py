
import os
import shutil
from datetime import datetime
import glog as logger

def create_dir(dir_name):
    os.makedirs(dir_name, exist_ok=True)

def get_str_experiments(flags):
    dateTimeObj = datetime.now()
    dateStr = dateTimeObj.strftime("%Y_%m_%d_%H_%M_%S_%f")
    str_experiments = flags.dataset + '_' + dateStr;
    return str_experiments


def create_dir_structure_testing(exp):
    flags = exp.flags;
    for k, label_str in enumerate(exp.labels):
        dir_gen_eval_label = os.path.join(flags.dir_gen_eval, label_str)
        create_dir(dir_gen_eval_label)
        dir_inference_label = os.path.join(flags.dir_inference, label_str)
        create_dir(dir_inference_label)


def create_dir_structure(flags, train=True):
    if train and (not flags.ssh):
        if flags.debug:
            str_experiments = 'debug'
        else:
            str_experiments = get_str_experiments(flags)
        flags.target_dir = os.path.join(flags.target_dir, str_experiments)
        # flags.str_experiment = str_experiments;
    else:
        flags.target_dir = flags.target_dir;

    logger.info(flags.target_dir)
    if train:
        create_dir(flags.target_dir)

    flags.dir_checkpoints = os.path.join(flags.target_dir, 'checkpoints')
    if train:
        create_dir(flags.dir_checkpoints)

    flags.dir_logs = os.path.join(flags.target_dir, 'logs')
    if train:
        create_dir(flags.dir_logs)
    logger.info(flags.dir_logs)

    flags.dir_logs_clf = os.path.join(flags.target_dir, 'logs_clf')
    if train:
        create_dir(flags.dir_logs_clf)

    flags.dir_gen_eval = os.path.join(flags.target_dir, 'generation_evaluation')
    if train:
        create_dir(flags.dir_gen_eval)

    flags.dir_inference = os.path.join(flags.target_dir, 'inference')
    if train:
        create_dir(flags.dir_inference)

    flags.dir_plots = os.path.join(flags.target_dir, 'plots')
    if train:
        create_dir(flags.dir_plots)
    flags.dir_swapping = os.path.join(flags.dir_plots, 'swapping')
    if train:
        create_dir(flags.dir_swapping)

    flags.dir_random_samples = os.path.join(flags.dir_plots, 'random_samples')
    if train:
        create_dir(flags.dir_random_samples)

    flags.dir_cond_gen = os.path.join(flags.dir_plots, 'cond_gen')
    if train:
        create_dir(flags.dir_cond_gen)
    return flags;
