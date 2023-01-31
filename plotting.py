import sys
import os

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import utils
from utils import plot
from utils.BaseExperiment import BaseExperiment_impute

def generate_plots(exp, epoch):
    plots = dict();

    plots['random'] = generate_random_samples_plots(exp, epoch);
    plots['impute_fig'] = generate_impute_fig_vaes(exp, epoch)
    return plots;


def generate_random_samples_plots(exp, epoch):
    model = exp.model;
    mods = exp.modalities;
    num_samples = 100;
    random_samples = model.generate(num_samples)
    random_plots = dict();
    for k, m_key_in in enumerate(mods.keys()):
        mod = mods[m_key_in];
        samples_mod = random_samples[m_key_in];
        rec = torch.zeros(exp.plot_img_size,
                          dtype=torch.float32).repeat(num_samples,1,1,1);
        for l in range(0, num_samples):
            rand_plot = mod.plot_data(samples_mod[l:l+1])[0];
            rec[l, :, :, :] = rand_plot;
        random_plots[m_key_in] = rec;

    for k, m_key in enumerate(mods.keys()):
        fn = os.path.join(exp.flags.dir_random_samples, 'random_epoch_' +
                             str(epoch).zfill(4) + '_' + m_key + '.png');
        mod_plot = random_plots[m_key];
        p = plot.create_fig(fn, mod_plot, 10, save_figure=exp.flags.save_figure);
        random_plots[m_key] = p;
    return random_plots;
    
def clip_pic(t):
    t[torch.where(t>1)]=1
    t[torch.where(t<0)]=0
    t[torch.isnan(t)]=0
    return t

def generate_impute_fig_vaes(exp:BaseExperiment_impute, epoch):
    test_samples = exp.test_samples
    model = exp.model
    modalities_names = exp.modalities_names
    mu_dim = exp.mu_dim

    plot_out = torch.zeros(exp.plot_img_size, dtype=torch.float32).repeat(len(test_samples)*len(modalities_names),1,1,1)
    plot_gt = torch.zeros_like(plot_out)
    for s_ind, s in enumerate(test_samples):
        sample_ind, sample_all, observed_mask, c_ind = s
        subset_used = exp.clients_subsets[c_ind]
        sample = utils.mask_modalities(sample_all, observed_mask, exp.modalities_names)
        sample = {k: sample[k].unsqueeze(0).to(exp.flags.device) for k in sample.keys()}
        with torch.no_grad():
            latents = model.inference(sample, subset_used)
            subset_names = latents['subsets_in_fusion']
            longest_subset = subset_names[utils.longest_ind(subset_names)]

            styles = model.get_random_style_dists(1)
            for k in sample.keys():
                styles[k] = latents['modalities'][k + '_style']
            l_dec = {'content': latents['subsets'][longest_subset],
                    'style': styles};
            output = model.generate_from_distribution(l_dec)

        for m_i, m_k in enumerate(modalities_names):
            pic_m = exp.modalities[m_k].plot_data(output[m_k].cpu())[0]
            pic_m_gt = exp.modalities[m_k].plot_data(sample_all[m_k].cpu().unsqueeze(0))[0]

            pic_m = clip_pic(pic_m)
            if observed_mask[m_i]==0:
                pic_m = plot.draw_border(pic_m)
                pic_m_gt = plot.draw_border(pic_m_gt)

            
            plot_out[s_ind + m_i*len(test_samples)] = pic_m
            plot_gt[s_ind + m_i*len(test_samples)] = pic_m_gt


    fn_out = os.path.join(exp.flags.dir_cond_gen, 'out_epoch_%d.png'%epoch);
    plot_out_grid = plot.create_fig(fn_out, plot_out, len(test_samples), save_figure=exp.flags.save_figure)
    fn_gt = os.path.join(exp.flags.dir_cond_gen, 'recon_gt_epoch_%d.png'%epoch);
    plot_gt_grid = plot.create_fig(fn_gt, plot_gt, len(test_samples), save_figure=exp.flags.save_figure)

    return {'impute_out': plot_out_grid, 'impute_recon_gt': plot_gt_grid}
