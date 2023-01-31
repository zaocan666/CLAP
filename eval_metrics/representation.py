import sys
import os
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

import utils.utils as utils
from utils.meter import Meter
from eicu.flags import EICU_SUBSETS

def train_clf_lr_all_subsets(exp):
    mm_vae = exp.model;
    mm_vae.eval();

    clf_all = []
    for c_i in range(exp.flags.client_num):
        subset_used = exp.clients_subsets[c_i]
        d_loader = DataLoader(exp.train_impute_dataset[c_i], batch_size=exp.flags.batch_size,
                            shuffle=True,
                            num_workers=8, drop_last=False);

        bs = exp.flags.batch_size;
        n_train_samples = exp.flags.num_training_samples_lr
        labels = []
        data_train = defaultdict(list)

        for it, batch in enumerate(d_loader):
            batch_all = batch[0]
            batch_l = batch[1]
            observed_mask = batch[2]

            batch_d = utils.mask_modalities(batch_all, observed_mask[0], exp.modalities_names)
            batch_d = {k: Variable(batch_d[k]).to(exp.flags.device) for k in batch_d.keys()}

            inferred = mm_vae.inference(batch_d, subset_used)
            
            if exp.flags.dataset=='eicu':
                subset_names = list(filter(lambda x:x in inferred['subsets'].keys(), EICU_SUBSETS))
            else:
                subset_names = inferred['subsets'].keys()

            for k in subset_names:
                data_train[k].append(inferred['subsets'][k][0].cpu().data.numpy())
            labels.append(batch_l.data.numpy())
            if (it+1)*bs>=n_train_samples:
                break
        
        labels = np.concatenate(labels, axis=0)[:n_train_samples]
        clf_client = {}
        for k in subset_names:
            data_train[k] = np.concatenate(data_train[k], axis=0)[:n_train_samples]
            assert data_train[k].shape[0]==n_train_samples
            clf_lr = train_clf_lr(exp, data_train[k], labels)
            clf_client[k] = clf_lr
        
        clf_all.append(clf_client)
    return clf_all;
 

def test_clf_lr_all_subsets(epoch, clf_lr, exp):
    mm_vae = exp.model;
    mm_vae.eval();

    client_eval_list = []
    for c_i in range(exp.flags.client_num):
        subset_used = exp.clients_subsets[c_i]
        d_loader = DataLoader(exp.test_impute_dataset[c_i], batch_size=exp.flags.batch_size,
                            shuffle=False,
                            num_workers=8, drop_last=False);

        outputs_all = defaultdict(dict)
        labels_all = []
        for iteration, batch in enumerate(d_loader):
            batch_all = batch[0]
            batch_l = batch[1]
            observed_mask = batch[2]
            labels_all.append(batch_l)

            batch_d = utils.mask_modalities(batch_all, observed_mask[0], exp.modalities_names)
            batch_d = {k: Variable(batch_d[k]).to(exp.flags.device) for k in batch_d.keys()}
            inferred = mm_vae.inference(batch_d, subset_used)

            if exp.flags.dataset=='eicu':
                subset_names = list(filter(lambda x:x in inferred['subsets'].keys(), EICU_SUBSETS))
            else:
                subset_names = inferred['subsets'].keys()

            for k in subset_names:
                data_test = inferred['subsets'][k][0].cpu().data.numpy()
                outputs = predict_latent_representations(exp,
                                                    clf_lr[c_i][k],
                                                    data_test); # {label: [1,2,3]}
                for label_k in outputs.keys():
                    if label_k not in outputs_all[k].keys():
                        outputs_all[k][label_k] = []
                    outputs_all[k][label_k].append(outputs[label_k])
        
        labels_all = torch.cat(labels_all, dim=0)
        labels_all = torch.reshape(labels_all, (labels_all.shape[0], len(exp.labels))).numpy();
        client_eval = {}
        for subset_k in outputs_all.keys():
            eval_all_labels = {}
            for l, label_str in enumerate(exp.labels):
                gt = labels_all[:, l]
                outputs_all[subset_k][label_str] = np.concatenate(outputs_all[subset_k][label_str])
                eval_label_rep = exp.eval_metric(gt.ravel(), outputs_all[subset_k][label_str].ravel());
                eval_all_labels[label_str] = eval_label_rep;
            client_eval[subset_k] = np.mean(list(eval_all_labels.values()))
        client_eval_list.append(client_eval)
    
    lr_eval = utils.avg_dict(client_eval_list, exp.client_test_dataset_len)
    lr_eval['mean'] = np.mean(list(lr_eval.values()))
    modality_len_f = lambda x:len(x.split('_'))
    for i in range(1, max([modality_len_f(x) for x in lr_eval.keys()])+1):
        len_i_keys = list(filter(lambda x:modality_len_f(x)==i, lr_eval.keys()))
        if len(len_i_keys)>0:
            lr_eval['mean_%d'%i] = np.mean([lr_eval[k] for k in len_i_keys])
    return lr_eval

def predict_latent_representations(exp, clf_lr, data):
    output_all_labels = dict()
    for l, label_str in enumerate(exp.labels):
        clf_lr_label = clf_lr[label_str];
        if exp.flags.dataset=='celeba' or exp.flags.dataset=='eicu':
            y_pred_rep = clf_lr_label.predict_proba(data)[:,1];
        else:
            y_pred_rep = clf_lr_label.predict(data);
        output_all_labels[label_str] = y_pred_rep.ravel()

    return output_all_labels;


def train_clf_lr(exp, data, labels):
    labels = np.reshape(labels, (labels.shape[0], len(exp.labels)));
    clf_lr_labels = dict();
    for l, label_str in enumerate(exp.labels):
        gt = labels[:, l];
        clf_lr_s = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter=1000);
        clf_lr_s.fit(data, gt.ravel());
        clf_lr_labels[label_str] = clf_lr_s;
    return clf_lr_labels;





