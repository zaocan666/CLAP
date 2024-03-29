
import numpy as np
from scipy.special import logsumexp
from itertools import cycle
import math

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import utils
from utils.likelihood import log_mean_exp
from utils.likelihood import gaussian_log_pdf
from utils.likelihood import unit_gaussian_log_pdf
from utils.likelihood import get_latent_samples
from utils.likelihood import get_dyn_prior
from utils.likelihood import log_marginal_estimate
from utils.likelihood import log_joint_estimate

from divergence_measures.mm_div import alpha_poe
from divergence_measures.mm_div import poe
from utils.meter import Meter
from utils.BaseExperiment import BaseExperiment_impute

LOG2PI = float(np.log(2.0 * math.pi))



#at the moment: only marginals and joint
def calc_log_likelihood_batch(exp, latents, s_dist, subset, batch, num_imp_samples=10):
    flags = exp.flags;
    model = exp.model;
    mods = exp.modalities;

    n_total_samples = s_dist[0].shape[0]*num_imp_samples;

    if flags.factorized_representation:
        enc_mods = latents['modalities'];
        style = model.get_random_style_dists(flags.batch_size);
        for m, mod in enumerate(subset):
            if (enc_mods[mod.name + '_style'][0] is not None
                and enc_mods[mod.name + '_style'][1] is not None):
                style[mod.name] = enc_mods[mod.name + '_style'];
    else:
        style = None;

    l_subset = {'content': s_dist, 'style': style};
    mod_names = mods.keys()
    l = get_latent_samples(model, l_subset, num_imp_samples, mod_names);

    l_style_rep = l['style'];
    l_content_rep = l['content'];

    c = {'mu': l_content_rep['mu'].view(n_total_samples, -1),
         'logvar': l_content_rep['logvar'].view(n_total_samples, -1),
         'z': l_content_rep['z'].view(n_total_samples, -1)}
    l_lin_rep = {'content': c,
                 'style': dict()};
    for m, m_key in enumerate(l_style_rep.keys()):
        if flags.factorized_representation:
            s = {'mu': l_style_rep[mod.name]['mu'].view(n_total_samples, -1),
                 'logvar': l_style_rep[mod.name]['logvar'].view(n_total_samples, -1),
                 'z': l_style_rep[mod.name]['z'].view(n_total_samples, -1)}
            l_lin_rep['style'][m_key] = s;
        else:
            l_lin_rep['style'][m_key] = None;

    l_dec = {'content': l_lin_rep['content']['z'],
             'style': dict()};
    for m, m_key in enumerate(l_style_rep.keys()):
        if flags.factorized_representation:
            l_dec['style'][m_key] = l_lin_rep['style'][m_key]['z'];
        else:
            l_dec['style'][m_key] = None;

    gen = model.generate_sufficient_statistics_from_latents(l_dec);

    l_lin_rep_style = l_lin_rep['style'];
    l_lin_rep_content = l_lin_rep['content'];
    ll = dict();
    for k, m_key in enumerate(batch.keys()):
        mod = mods[m_key];
        # compute marginal log-likelihood
        if mod in subset:
            style_mod = l_lin_rep_style[mod.name];
        else:
            style_mod = None;
        ll_mod = log_marginal_estimate(flags,
                                       num_imp_samples,
                                       gen[mod.name],
                                       batch[mod.name],
                                       style_mod,
                                       l_lin_rep_content)
        ll[mod.name] = ll_mod;

    ll_joint = log_joint_estimate(flags, num_imp_samples, gen, batch,
                                  l_lin_rep_style,
                                  l_lin_rep_content);
    ll['joint'] = ll_joint;
    return ll;


def estimate_likelihoods(exp):
    model = exp.model;
    mods = exp.modalities;
    bs_normal = exp.flags.batch_size;
    exp.flags.batch_size = 64;
    d_loader = DataLoader(exp.dataset_test,
                          batch_size=exp.flags.batch_size,
                          shuffle=True,
                          num_workers=8, drop_last=True);

    subsets = exp.subsets;
    lhoods = dict()
    for k, s_key in enumerate(subsets.keys()):
        if s_key != '':
            lhoods[s_key] = dict();
            for m, m_key in enumerate(mods.keys()):
                lhoods[s_key][m_key] = [];
            lhoods[s_key]['joint'] = [];

    for iteration, batch in enumerate(d_loader):
        batch_d = batch[0];
        for m, m_key in enumerate(mods.keys()):
            batch_d[m_key] = batch_d[m_key].to(exp.flags.device);

        latents = model.inference(batch_d);
        for k, s_key in enumerate(subsets.keys()):
            if s_key != '':
                subset = subsets[s_key];
                ll_batch = calc_log_likelihood_batch(exp, latents,
                                                     latents['subsets'][s_key], subset,
                                                     batch_d, num_imp_samples=12)
                for l, m_key in enumerate(ll_batch.keys()):
                    lhoods[s_key][m_key].append(ll_batch[m_key].item());

    for k, s_key in enumerate(lhoods.keys()):
        lh_subset = lhoods[s_key];
        for l, m_key in enumerate(lh_subset.keys()):
            mean_val = np.mean(np.array(lh_subset[m_key]))
            lhoods[s_key][m_key] = mean_val;
    exp.flags.batch_size = bs_normal;
    return lhoods;


def estimate_likelihoods_all(exp: BaseExperiment_impute):
    model = exp.model;
    mods = exp.modalities;
    bs_normal = exp.flags.batch_size;
    exp.flags.batch_size = 64;

    lhoods_all = []
    for c_i in range(exp.flags.client_num):
        subset_used = exp.clients_subsets[c_i]
        d_loader = DataLoader(exp.test_impute_dataset[c_i], batch_size=exp.flags.batch_size,
                            shuffle=False,
                            num_workers=8, drop_last=False);

        lhoods = Meter()

        for iteration, batch in enumerate(d_loader):
            batch_all = batch[0]
            batch_l = batch[1]
            observed_mask = batch[2]
            batch_all = {k: Variable(batch_all[k]).to(exp.flags.device) for k in batch_all.keys()}
            batch_d = utils.mask_modalities(batch_all, observed_mask[0], exp.modalities_names)            

            latents = model.inference(batch_d, subset_used);
            subset_names = latents['subsets_in_fusion']
            longest_subset = subset_names[utils.longest_ind(subset_names)]
            dis = latents['subsets'][longest_subset]
            subset = exp.subsets[longest_subset]
            ll_batch = calc_log_likelihood_batch(exp, latents,
                                                dis, subset,
                                                batch_all,
                                                num_imp_samples=12)
            lhoods._update(ll_batch, batch_size=batch[2].shape[0])

        client_lhoods = lhoods.get_scalar_dict('global_avg')
        lhoods_all.append(client_lhoods)

    lhoods_all = utils.avg_dict(lhoods_all, exp.client_test_dataset_len)
    exp.flags.batch_size = bs_normal;
    return lhoods_all;
