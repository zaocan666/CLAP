import os
import random
import sys
import numpy as np
import json
import torch
import torch.nn as nn
from collections import defaultdict
import glog as logger

from utils import text as text

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    logger.info('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        logger.info()

def reweight_weights(w):
    w = w / w.sum();
    return w;

def mixture_component_selection(flags, mus, logvars, w_modalities=None):
    #if not defined, take pre-defined weights
    num_components = mus.shape[0];
    num_samples = mus.shape[1];
    if w_modalities is None:
        w_modalities = torch.Tensor(flags.alpha_modalities).to(flags.device);
    idx_start = [];
    idx_end = []
    for k in range(0, num_components):
        if k == 0:
            i_start = 0;
        else:
            i_start = int(idx_end[k-1]);
        if k == w_modalities.shape[0]-1:
            i_end = num_samples;
        else:
            i_end = i_start + int(torch.floor(num_samples*w_modalities[k]));
        idx_start.append(i_start);
        idx_end.append(i_end);
    idx_end[-1] = num_samples;
    mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])]);
    logvar_sel = torch.cat([logvars[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])]);
    return [mu_sel, logvar_sel];


def calc_elbo(exp, modality, recs, klds):
    flags = exp.flags;
    mods = exp.modalities;
    s_weights = exp.style_weights;
    r_weights = exp.rec_weights;
    kld_content = klds['content'];
    if modality == 'joint':
        w_style_kld = 0.0;
        w_rec = 0.0;
        klds_style = klds['style']
        for k, m_key in enumerate(recs.keys()):
                w_style_kld += s_weights[m_key] * klds_style[m_key];
                w_rec += r_weights[m_key] * recs[m_key];
        kld_style = w_style_kld;
        rec_error = w_rec;
    else:
        beta_style_mod = s_weights[modality];
        rec_weight_mod = 1.0;
        kld_style = beta_style_mod * klds['style'][modality];
        rec_error = rec_weight_mod * recs[modality];
    div = flags.beta_content * kld_content + flags.beta_style * kld_style;
    elbo = rec_error + flags.beta * div;
    return elbo, div, rec_error


def save_and_log_flags(flags):
    filename_flags_rar = os.path.join(flags.target_dir, 'flags.rar')
    torch.save(flags, filename_flags_rar);
    str_args = '';
    for k, key in enumerate(sorted(flags.__dict__.keys())):
        str_args = str_args + '\n' + key + ': ' + str(flags.__dict__[key]);
    return str_args;


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(torch.nn.Module):
    def __init__(self, ndims):
        super(Unflatten, self).__init__()
        self.ndims = ndims

    def forward(self, x):
        return x.view(x.size(0), *self.ndims)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
    
def set_log_file(fname, file_only=False):
    # set log file
    # simple tricks for duplicating logging destination in the logging module such as:
    # logging.getLogger().addHandler(logging.FileHandler(filename))
    # does NOT work well here, because python Traceback message (not via logging module) is not sent to the file,
    # the following solution (copied from : https://stackoverflow.com/questions/616645) is a little bit
    # complicated but simulates exactly the "tee" command in linux shell, and it redirects everything
    if file_only:
        # we only output messages to file, and stdout/stderr receives nothing.
        # this feature is designed for executing the script via ssh:
        # since ssh has a windowing kind of flow control, i.e., if the controller does not read data from a
        # ssh channel and its buffer fills up, the execution machine will not be able to write anything into the
        # channel and the process will be set to sleeping (S) status until someone reads all data from the channel.
        # this is not desired since we do not want to read stdout/stderr from the controller machine.
        # so, here we use a simple solution: disable output to stdout/stderr and only output messages to log file.
        logger.logger.handlers[0].stream = logger.handler.stream = sys.stdout = sys.stderr = open(fname, 'w', buffering=1)
    else:
        # we output messages to both file and stdout/stderr
        import subprocess
        tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
        os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def print_args(args):
    logger.info('-------- args -----------')
    for k,v in vars(args).items():
        logger.info('%s: '%k+str(v))
    logger.info('-------------------------')

def mask_modalities(data_all, observed_mask, modalities_names):
    data_d = {}
    for k, m_key in enumerate(data_all.keys()):
        k_ind = modalities_names.index(m_key)
        if observed_mask[k_ind]==1:
            data_d[m_key] = data_all[m_key]
    return data_d

def avg_dict(scalar_dict_list, weights=None):
    if not weights:
        weights = [1.0]*(len(scalar_dict_list))
    weights = np.array(weights)/np.sum(weights)

    avg_dict = defaultdict(lambda:0)
    weights_sum = defaultdict(lambda:0)
    for i in range(len(scalar_dict_list)):
        for k in scalar_dict_list[i].keys():
            avg_dict[k] += scalar_dict_list[i][k]*weights[i]
            weights_sum[k] += weights[i]
    
    for k in scalar_dict_list[i].keys():
        avg_dict[k] /= weights_sum[k]

    return dict(avg_dict)

def get_invisible_modalities(observed_mask, modalities_names):
    invisible_modalities = []
    for k in range(observed_mask.shape[0]):
        if observed_mask[k]==0:
            invisible_modalities.append(modalities_names[k])
    return invisible_modalities

def longest_ind(s):
    lens = [len(i) for i in s]
    return np.argmax(lens)
