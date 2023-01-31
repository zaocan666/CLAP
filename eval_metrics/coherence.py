
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import glog as logger
from collections import defaultdict

import utils.utils as utils
from utils.meter import Meter

def predict_cond_gen_samples(exp, labels, cond_samples):
    labels = np.reshape(labels, (labels.shape[0], len(exp.labels)));
    clfs = exp.clfs;
    outputs_mods = {}
    for key in clfs.keys():
        if key in cond_samples.keys():
            mod_cond_gen = cond_samples[key];
            mod_clf = clfs[key];
            attr_hat = mod_clf(mod_cond_gen);
            outputs_mods[key]=attr_hat
        else:
            logger.info(str(key) + 'not existing in cond_gen_samples');
            raise KeyError()
    return outputs_mods;


def calculate_coherence(exp, samples):
    clfs = exp.clfs;
    mods = exp.modalities;
    # TODO: make work for num samples NOT EQUAL to batch_size
    c_labels = dict();
    num_sample = samples[list(samples.keys())[0]].shape[0]
    
    pred_mods = np.zeros((len(samples.keys()), num_sample, len(exp.labels)))
    for k, m_key in enumerate(samples.keys()):
        mod = mods[m_key];
        clf_mod = clfs[mod.name];
        samples_mod = samples[mod.name];
        attr_mod = clf_mod(samples_mod);
        output_prob_mod = attr_mod.cpu().data.numpy();
        if exp.flags.dataset=='celeba':
            pred_mod = (output_prob_mod>0.5).astype(int)
        else:
            pred_mod = np.argmax(output_prob_mod, axis=1).astype(int)[:,None];
        pred_mods[k] = pred_mod;
    for j, l_key in enumerate(exp.labels):
        coh_mods = np.all(pred_mods[:, :, j] == pred_mods[0, :, j], axis=0)
        coherence = np.sum(coh_mods.astype(int))/float(num_sample);
        c_labels[l_key] = coherence;
    return c_labels;

def test_generation_all(epoch, exp):
    mods = exp.modalities;
    mm_vae = exp.model;

    with torch.no_grad():
        gen_perf_all = {'cond': {l_key: [] for l_key in ['mean']}, 'random': []}
        for c_i in range(exp.flags.client_num):
            subset_used = exp.clients_subsets[c_i]
            gen_perf = {'cond': {}, 'random': Meter()}

            d_loader = DataLoader(exp.test_impute_dataset[c_i], batch_size=exp.flags.batch_size,
                                shuffle=False,
                                num_workers=8, drop_last=False);
            
            outputs_all = defaultdict(list)
            labels_all = []
            for iteration, batch in enumerate(d_loader):
                batch_all = batch[0]
                batch_l = batch[1]
                observed_mask = batch[2]
                labels_all.append(batch_l)

                rand_gen = mm_vae.generate(num_samples=observed_mask.shape[0]);
                coherence_random = calculate_coherence(exp, rand_gen);
                coherence_random = {'mean': np.mean(list(coherence_random.values()))}
                gen_perf['random']._update(coherence_random, batch_size=batch[2].shape[0])

                batch_d = utils.mask_modalities(batch_all, observed_mask[0], exp.modalities_names)
                batch_d = {k: Variable(batch_d[k]).to(exp.flags.device) for k in batch_d.keys()}

                inferred = mm_vae.inference(batch_d, subset_used);
                subset_names = inferred['subsets_in_fusion']
                longest_subset = subset_names[utils.longest_ind(subset_names)]
                output = mm_vae.cond_generation({longest_subset: inferred['subsets'][longest_subset]})[longest_subset]
                clf_outputs = predict_cond_gen_samples(exp,
                                                batch_l,
                                                output);
                for k in clf_outputs.keys():
                    outputs_all[k].append(clf_outputs[k])

            labels_all = torch.cat(labels_all, dim=0)
            clf_cg = {}
            for k in outputs_all.keys():
                eval_labels = {}
                outputs_all[k] = torch.cat(outputs_all[k], dim=0)
                for l, label_str in enumerate(exp.labels):
                    score = exp.eval_label(outputs_all[k].cpu().data.numpy(), labels_all,
                                        index=l);
                    eval_labels[label_str] = score;
                clf_cg[k] = np.mean(list(eval_labels.values()))
            gen_perf['cond']['mean'] = clf_cg

            client_random = gen_perf['random'].get_scalar_dict('global_avg')
            gen_perf_all['random'].append(client_random)
            for j, l_key in enumerate(gen_perf['cond'].keys()):
                gen_perf_all['cond'][l_key].append(gen_perf['cond'][l_key])

    result = {'random': utils.avg_dict(gen_perf_all['random'], exp.client_test_dataset_len), 'cond':{}}
    for j, l_key in enumerate(gen_perf_all['cond'].keys()):
        result['cond'][l_key] = utils.avg_dict(gen_perf_all['cond'][l_key], exp.client_test_dataset_len)
    
    return result;
