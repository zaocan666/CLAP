import copy
import sys, os
import numpy as np
from itertools import cycle
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import glog as logger
from collections import OrderedDict, defaultdict
import pickle

from divergence_measures.kl_div import calc_kl_divergence

from eval_metrics.coherence import test_generation_all
from eval_metrics.representation import train_clf_lr_all_subsets, test_clf_lr_all_subsets
from eval_metrics.likelihood import estimate_likelihoods_all
from eval_metrics.test_mse import test_mse_all

from plotting import generate_plots

from utils import utils
from utils.TBLogger import TBLogger
from utils.BaseExperiment import BaseExperiment_impute
from utils.meter import Meter

# global variables
SEED = None 
SAMPLE1 = None
if SEED is not None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED) 


def calc_log_probs(exp, rec, batch_d):
    mods = exp.modalities;
    log_probs = dict()
    weighted_log_prob = 0.0;
    for m, m_key in enumerate(mods.keys()):
        if m_key in batch_d.keys():
            mod = mods[m_key]
            log_probs[mod.name] = -mod.calc_log_prob(rec[mod.name],
                                                    batch_d[mod.name],
                                                    batch_d[mod.name].shape[0]);
            weighted_log_prob += exp.rec_weights[mod.name]*log_probs[mod.name];
    return log_probs, weighted_log_prob;


def calc_klds(exp, result):
    latents = result['latents']['subsets'];
    klds = dict();
    for m, key in enumerate(latents.keys()):
        mu, logvar = latents[key];
        klds[key] = calc_kl_divergence(mu, logvar,
                                       norm_value=exp.flags.batch_size)
    return klds;


def calc_klds_style(exp, result, batch_d, single_prefix='_single'):
    mods = exp.modalities;
    latents = result['latents']['modalities'];
    klds = dict();
    # for m, key in enumerate(latents.keys()):
    #     if key.endswith('style'):
    for m, m_key in enumerate(mods.keys()):
        if m_key in batch_d.keys():
            mu, logvar = latents[m_key+'_style'+single_prefix];
            klds[m_key+'_style'] = calc_kl_divergence(mu, logvar,
                                           norm_value=mu.shape[0])
    return klds;


def calc_style_kld(exp, klds, batch_d):
    mods = exp.modalities;
    style_weights = exp.style_weights;
    weighted_klds = 0.0;
    for m, m_key in enumerate(mods.keys()):
        if m_key in batch_d.keys():
            weighted_klds += style_weights[m_key]*klds[m_key+'_style'];
    return weighted_klds;

def tmp_dict_func(f, d):
    out = {}
    for k in d.keys():
        if type(d[k][0])!=type(None):
            out[k] = [f(d[k][0]), f(d[k][1])]
        else:
            out[k] = [None, None]
    return out

def basic_routine_epoch(exp:BaseExperiment_impute, batch, mm_vae, modality_name=None, subset_used=None, train=False):
    # set up weights
    beta_style = exp.flags.beta_style;
    beta_content = exp.flags.beta_content;
    beta = exp.flags.beta;
    rec_weight = 1.0;

    mods = exp.modalities;
    
    batch_all = batch[0];
    batch_l = batch[1];
    observed_mask = batch[2]
    batch_d = utils.mask_modalities(batch_all, observed_mask[0], exp.modalities_names)
    batch_d = {k: Variable(batch_d[k]).to(exp.flags.device) for k in batch_d.keys()}
    results = mm_vae(batch_d, subset_used);

    log_probs, weighted_log_prob = calc_log_probs(exp, results['rec'], batch_d);
    group_divergence = results['joint_divergence'];
    if exp.flags.impute_method == 'collaborate' and exp.flags.k_single!=0:
        if exp.flags.factorized_representation:
            klds_style_single = calc_klds_style(exp, results, batch_d, single_prefix='_single');
            kld_style_single = calc_style_kld(exp, klds_style_single, batch_d);
        else:
            kld_style_single = 0

        log_probs_single, weighted_log_prob_single = calc_log_probs(exp, results['rec_single'], batch_d)
        group_divergence_single = results['joint_divergence_single'];
        kld_weighted_single = beta_style * kld_style_single + beta_content * group_divergence_single
        total_loss_single = rec_weight * weighted_log_prob_single + beta * kld_weighted_single
    else:
        weighted_log_prob_single = kld_weighted_single = total_loss_single = 0

    klds = calc_klds(exp, results);
    if exp.flags.factorized_representation:
        klds_style = calc_klds_style(exp, results, batch_d, single_prefix='');

    if (exp.flags.impute_method=='jsd' or exp.flags.impute_method=='moe'
        or exp.flags.impute_method=='joint_elbo' or exp.flags.impute_method=='vae'
        or exp.flags.impute_method=='collaborate'):
        if exp.flags.factorized_representation:
            kld_style = calc_style_kld(exp, klds_style, batch_d);
        else:
            kld_style = 0.0;
        kld_content = group_divergence;
        kld_weighted = beta_style * kld_style + beta_content * kld_content;
        total_loss_subsets = rec_weight * weighted_log_prob + beta * kld_weighted;
        total_loss = total_loss_subsets + exp.flags.k_single*total_loss_single
    elif exp.flags.impute_method=='poe':
        klds_joint = {'content': group_divergence,
                      'style': dict()};
        elbos = dict();
        kld_weighted = 0
        weighted_log_prob = 0
        for m, m_key in enumerate(batch_d.keys()):
            mod = mods[m_key];
            if exp.flags.factorized_representation:
                kld_style_m = klds_style[m_key + '_style'];
            else:
                kld_style_m = 0.0;
            klds_joint['style'][m_key] = kld_style_m;
            if exp.flags.poe_unimodal_elbos:
                i_batch_mod = {m_key: batch_d[m_key]};
                r_mod = mm_vae(i_batch_mod, subset_used);
                log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                                  batch_d[m_key],
                                                  batch_d[m_key].shape[0]);
                log_prob = {m_key: log_prob_mod};
                klds_mod = {'content': klds[m_key],
                            'style': {m_key: kld_style_m}};
                elbo_mod, div, rec_error = utils.calc_elbo(exp, m_key, log_prob, klds_mod);
                elbos[m_key] = elbo_mod;

                kld_weighted += div.item()/(len(batch_d.keys())+1)
                weighted_log_prob += rec_error.item()/(len(batch_d.keys())+1)

        elbo_joint, div, rec_error = utils.calc_elbo(exp, 'joint', log_probs, klds_joint);
        elbos['joint'] = elbo_joint;
        total_loss = sum(elbos.values())

        kld_weighted += div.item()/(len(batch_d.keys())+1)
        weighted_log_prob += rec_error.item()/(len(batch_d.keys())+1)

        total_loss_subsets = 0

    out_basic_routine = dict();
    out_basic_routine['results'] = results;
    out_basic_routine['log_probs'] = log_probs;
    out_basic_routine['total_loss'] = total_loss;
    out_basic_routine['total_loss_subsets'] = total_loss_subsets;
    out_basic_routine['klds'] = klds;
    out_basic_routine['kld_weighted'] = kld_weighted;
    out_basic_routine['weighted_log_prob'] = weighted_log_prob;

    out_basic_routine['weighted_log_prob_single'] = weighted_log_prob_single;
    out_basic_routine['kld_weighted_single'] = kld_weighted_single;
    out_basic_routine['total_loss_single'] = exp.flags.k_single*total_loss_single;
    return out_basic_routine;

def print_log(name, iteration, total_iter, loss, klds, log_probs, joint_divergence):
    str_out = '%s Iteration %d/%d; loss: %.4f; joint_divergence: %.4f; '%(name, iteration, total_iter, loss, joint_divergence)
    
    str_out += 'kld:'
    for k in klds.keys():
        str_out += ' %s(%.4f)'%(k, klds[k].item())
    
    str_out += '; log probs:'
    for k in log_probs.keys():
        str_out += ' %s(%.4f)'%(k, log_probs[k].item())
    
    logger.info(str_out)

def dict2str(out_dict):
    str_out = ''
    for i, k in enumerate(out_dict.keys()):
        str_out += str(k)+':'
        if isinstance(out_dict[k], dict):
            for k_k in out_dict[k].keys():
                str_out += ' %.4f (%s)'%(out_dict[k][k_k], k_k)
        else:
            str_out += '%.4f'%(out_dict[k])
        
        str_out += '; '
    return str_out

def train(epoch, exp: BaseExperiment_impute, tb_logger: TBLogger, modality_name=None):
    exp.pickle_record['train'][epoch] = {}

    # clients_sample = random.choices(list(range(exp.flags.client_num)), k=round(exp.flags.client_num*exp.flags.client_sample))
    clients_sample = list(range(exp.flags.client_num))
    state_dict_list = []
    client_log_dict_list = []
    k_clients_subsets = []
    k_clients_single = []
    for c_i in clients_sample:
        subset_used = exp.clients_subsets[c_i]
        model = copy.deepcopy(exp.model)
        model.train()
        optimizer = exp.optimizer_class(model.parameters(), lr=exp.flags.initial_learning_rate, weight_decay=1e-6)
        d_loader = DataLoader(exp.train_impute_dataset[c_i], batch_size=exp.flags.batch_size,
                            shuffle=True,
                            num_workers=8, drop_last=True);

        meter_client = Meter()
        for iteration, batch in enumerate(d_loader):
            basic_routine = basic_routine_epoch(exp, batch, model, modality_name, subset_used, train=True);
            results = basic_routine['results'];
            total_loss = basic_routine['total_loss'];
            klds = basic_routine['klds'];
            log_probs = basic_routine['log_probs'];
            # backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            meter_client._update({'loss': total_loss, 'kld': basic_routine['kld_weighted'], 'log_prob': basic_routine['weighted_log_prob'],
                                        'loss_sin': basic_routine['total_loss_single'], 'kld_sin': basic_routine['kld_weighted_single'], 
                                        'log_prob_sin': basic_routine['weighted_log_prob_single'], 'loss_subsets': basic_routine['total_loss_subsets']},
                                    batch_size=batch[2].shape[0])

        invisible_modalities = utils.get_invisible_modalities(exp.observed_mask[c_i], exp.modalities_names)
        state_dict_list.append(model.get_visible_state_dict(exclude_modalities=invisible_modalities, subset_used=subset_used))

        logger.info('Train epoch %d, client %d: '%(epoch, c_i) + str(meter_client))
        logger_dict = meter_client.get_scalar_dict('global_avg')
        tb_logger.write_dict({'train/'+k+'_client%d'%c_i:logger_dict[k] for k in logger_dict.keys()}, step=epoch)
        tb_logger.write_latent_distr('train', results['latents'], step=epoch);
        client_log_dict_list.append(logger_dict)
        exp.pickle_record['train'][epoch][c_i] = logger_dict

        k_clients_subsets.append(logger_dict['loss_subsets'])
        k_clients_single.append(logger_dict['loss_sin'])
    
    avg_log_dict = utils.avg_dict(client_log_dict_list, exp.client_train_dataset_len)
    logger.info('-------Train epoch %d all: '%(epoch) + dict2str(avg_log_dict))
    tb_logger.write_dict({'train/'+k+'_all':avg_log_dict[k] for k in avg_log_dict.keys()}, step=epoch)

    if exp.flags.impute_method == 'collaborate':
        reweight_f = lambda x: utils.reweight_weights(10+utils.reweight_weights(np.array(x)))*len(x)
        k_clients_subsets = reweight_f(k_clients_subsets)
        k_clients_single = reweight_f(k_clients_single)
    else:
        k_clients_subsets = [1]*len(clients_sample)
        k_clients_single = [1]*len(clients_sample)
    
    update_state = OrderedDict()
    state_count = defaultdict(lambda:0)
    for i, c_i in enumerate(clients_sample):
        local_state = state_dict_list[i]
        for key in local_state.keys():
            if key.split('.')[0]=='encoders':
                k_weight = (k_clients_subsets[i]+k_clients_single[i])/2
            elif key.split('.')[0]=='decoders':
                k_weight = k_clients_subsets[i]
            elif key.split('.')[0]=='decoders_single':
                k_weight = k_clients_single[i]
            else:
                raise KeyError()

            if key not in update_state.keys():
                update_state[key] = local_state[key]*k_weight
            else:
                update_state[key] += local_state[key]*k_weight
            state_count[key] += 1
    
    for key in state_count.keys():
        update_state[key] = update_state[key]/state_count[key]
    
    exp.model.load_state_dict(update_state)

def test(epoch, exp, tb_logger, modality_name=None):
    with torch.no_grad():
        mm_vae = exp.model;
        mm_vae.eval();
        exp.model = mm_vae;

        exp.pickle_record['test'][epoch] = {}
        exp.pickle_record['test_metric'][epoch] = {}

        # set up weights
        beta_style = exp.flags.beta_style;
        beta_content = exp.flags.beta_content;
        beta = exp.flags.beta;
        rec_weight = 1.0;

        client_log_dict_list = []
        for c_i in range(exp.flags.client_num):
            subset_used = exp.clients_subsets[c_i]
            d_loader = DataLoader(exp.test_impute_dataset[c_i], batch_size=exp.flags.batch_size,
                                shuffle=False,
                                num_workers=8, drop_last=False);

            meter_client = Meter()
            for iteration, batch in enumerate(d_loader):
                basic_routine = basic_routine_epoch(exp, batch, mm_vae, modality_name, subset_used, train=False);
                results = basic_routine['results'];
                total_loss = basic_routine['total_loss'];
                klds = basic_routine['klds'];
                log_probs = basic_routine['log_probs'];

                # tb_logger.write_testing_logs(results, {'loss_' + str(modality_name) if modality_name else '': total_loss.data.item()}, log_probs, klds);
                # print_log('Test', iteration, len(d_loader), total_loss.data.item(), basic_routine['klds'], 
                #     basic_routine['log_probs'], basic_routine['results']['joint_divergence'])

                meter_client._update({'loss': total_loss, 'kld': basic_routine['kld_weighted'], 'log_prob': basic_routine['weighted_log_prob'],
                                        'loss_sin': basic_routine['total_loss_single'], 'kld_sin': basic_routine['kld_weighted_single'], 
                                        'log_prob_sin': basic_routine['weighted_log_prob_single']},
                                    batch_size=batch[2].shape[0])

            logger.info('Test epoch %d, client %d: '%(epoch, c_i) + str(meter_client))
            logger_dict = meter_client.get_scalar_dict('global_avg')
            tb_logger.write_dict({'test/'+k+'_client%d'%c_i:logger_dict[k] for k in logger_dict.keys()}, step=epoch)
            tb_logger.write_latent_distr('test', results['latents']);
            client_log_dict_list.append(logger_dict)

            exp.pickle_record['test'][epoch][c_i] = logger_dict

        avg_log_dict = utils.avg_dict(client_log_dict_list, exp.client_test_dataset_len)
        logger.info('-------Test epoch %d all: '%(epoch) + dict2str(avg_log_dict))
        tb_logger.write_dict({'test/'+k+'_all':avg_log_dict[k] for k in avg_log_dict.keys()}, step=epoch)
        tb_logger.write_latent_distr('test', results['latents'], step=epoch);
        
        if (epoch + 1) % exp.flags.eval_freq == 0 or (epoch + 1) == exp.flags.end_epoch or exp.flags.test_only:
            test_metric(epoch, exp, tb_logger)

def test_metric(epoch, exp, tb_logger):
    exp.model = exp.model.eval()
    if exp.flags.dataset!='eicu':
        plots = generate_plots(exp, epoch);
        tb_logger.write_plots(plots, epoch);

    if epoch not in exp.pickle_record['test_metric'].keys():
        exp.pickle_record['test_metric'][epoch] = {}

    if exp.flags.calc_mse:
        mse_eval = test_mse_all(epoch, exp)
        tb_logger.write_scalars('MSE', mse_eval, step=epoch)
        logger.info('-------Test epoch %d: '%(epoch) + dict2str({'MSE': mse_eval}))

    if exp.flags.eval_lr:
        clf_lr = train_clf_lr_all_subsets(exp);
        lr_eval = test_clf_lr_all_subsets(epoch, clf_lr, exp);
        tb_logger.write_lr_eval(lr_eval, step=epoch);
        logger.info('-------Test epoch %d: '%(epoch) + dict2str({'latent classification': lr_eval}))
        exp.pickle_record['test_metric'][epoch]['latent_classification'] = lr_eval

    if exp.flags.use_clf:
        with torch.no_grad():
            gen_eval = test_generation_all(epoch, exp);
        tb_logger.write_coherence_logs_all(gen_eval, step=epoch);
        logger.info('-------Test epoch %d, Coherence: '%(epoch) + 'random | ' + dict2str(gen_eval['random']) + ' cond | ' + dict2str(gen_eval['cond']))
        exp.pickle_record['test_metric'][epoch]['coherence'] = gen_eval

    if exp.flags.calc_nll:
        with torch.no_grad():
            lhoods = estimate_likelihoods_all(exp);
        # tb_logger.write_lhood_logs(lhoods);
        tb_logger.write_scalars('Likelihoods', lhoods, step=epoch)
        logger.info('-------Test epoch %d: '%(epoch) + dict2str({'Likelihoods': lhoods}))
        exp.pickle_record['test_metric'][epoch]['likelihoods'] = lhoods

def run_epochs_vae(exp, modality_name=None):
    # initialize summary writer
    writer = SummaryWriter(exp.flags.dir_logs)
    # tb_logger = TBLogger(exp.flags.str_experiment, writer)
    tb_logger = TBLogger(os.path.basename(exp.flags.dir_experiment), writer)
    str_flags = utils.save_and_log_flags(exp.flags);
    tb_logger.writer.add_text('FLAGS', str_flags, 0)

    if exp.flags.test_only:
        # test(-1, exp, tb_logger, modality_name);
        test_metric(-1, exp, tb_logger);
        return

    last_model_pth = None
    logger.info('training epochs progress:')
    for epoch in range(exp.flags.start_epoch, exp.flags.end_epoch):
        # utils.printProgressBar(epoch, exp.flags.end_epoch)
        logger.info('------Epoch %d/%d---------'%(epoch, exp.flags.end_epoch))
        # one epoch of training and testing
        train(epoch, exp, tb_logger, modality_name);
        with torch.no_grad():
            test(epoch, exp, tb_logger, modality_name);
        # save checkpoints after every 5 epochs

        with open(os.path.join(exp.flags.target_dir, 'pickle.pkl'), "wb") as f:
            pickle.dump(exp.pickle_record, f)

        if (epoch + 1) % 5 == 0 or (epoch + 1) == exp.flags.end_epoch:
            if last_model_pth:
                os.system('rm -r %s'%last_model_pth)
            dir_network_epoch = os.path.join(exp.flags.dir_checkpoints, str(epoch).zfill(4));
            if not os.path.exists(dir_network_epoch):
                os.makedirs(dir_network_epoch);
            # exp.model.save_networks()
            model_name = exp.flags.mm_vae_save
            # if exp.flags.method=='vae':
            #     model_name += '_%s'%modality_name
            torch.save(exp.model.state_dict(),
                       os.path.join(dir_network_epoch, model_name))
            last_model_pth = dir_network_epoch
