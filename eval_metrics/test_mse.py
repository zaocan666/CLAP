import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import glog as logger
from utils.meter import Meter
import utils.utils as utils

def test_mse_all(epoch, exp):
    mods = exp.modalities;
    mm_vae = exp.model;

    gen_perf_all = []
    for c_i in range(exp.flags.client_num):
        subset_used = exp.clients_subsets[c_i]
        gen_perf = Meter()

        d_loader = DataLoader(exp.test_impute_dataset[c_i], batch_size=exp.flags.batch_size,
                            shuffle=False,
                            num_workers=8, drop_last=False);

        for iteration, batch in enumerate(d_loader):
            batch_all = batch[0]
            batch_l = batch[1]
            observed_mask = batch[2]

            batch_all = {k: Variable(batch_all[k]).to(exp.flags.device) for k in batch_all.keys()}
            batch_d = utils.mask_modalities(batch_all, observed_mask[0], exp.modalities_names)

            with torch.no_grad():
                inferred = mm_vae.inference(batch_d, subset_used);
                subset_names = inferred['subsets_in_fusion']
                longest_subset = subset_names[utils.longest_ind(subset_names)]
                output = mm_vae.cond_generation({longest_subset: inferred['subsets'][longest_subset]})[longest_subset]
                mse_cg = mse_cond_gen_samples(exp, batch_all, output);
            gen_perf._update(mse_cg, batch_size=batch[2].shape[0])

        client_cond_key = gen_perf.get_scalar_dict('global_avg')
        gen_perf_all.append(client_cond_key)

    result = utils.avg_dict(gen_perf_all, exp.client_test_dataset_len)
    
    return result;

def mse_cond_gen_samples(exp, batch_all, cond_samples):
    out_dict = {}
    for key in batch_all.keys():
        if key in cond_samples.keys():
            mod_cond_gen = cond_samples[key];
            mod_batch = batch_all[key]
            mse = ((mod_cond_gen-mod_batch)**2).mean().item()
            out_dict[key] = mse
        else:
            logger.info(str(key) + 'not existing in cond_gen_samples');
            raise KeyError()
    return out_dict;
    