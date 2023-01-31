
from abc import ABC, abstractmethod
import pickle
import random
import glog as logger
import torch
import os
from itertools import chain, combinations
import glob
import glog as logger
from torch.utils.data import DataLoader
from collections import defaultdict
from torch.optim import Adam
import numpy as np

from utils.BaseMMVae import BaseMMVae
from utils.dataset import Client_Dataset, get_observed_mask, noniid, celeba_split

def get_subsets(modalities):
    num_mods = len(list(modalities.keys()));

    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3)
    (1,2,3)
    """
    xs = list(modalities) # list(self.modalities.keys())
    # note we return an iterator rather than a list
    subsets_list = chain.from_iterable(combinations(xs, n) for n in
                                        range(len(xs)+1))
    subsets = dict();
    for k, mod_names in enumerate(subsets_list):
        mods = [];
        for l, mod_name in enumerate(sorted(mod_names)):
            mods.append(modalities[mod_name])
        key = '_'.join(sorted(mod_names));
        subsets[key] = mods;
    return subsets;

class BaseExperiment_impute(ABC):
    def __init__(self, flags, alphabet):
        self.flags = flags
        self.alphabet = alphabet
        self.num_modalities = flags.num_mods
        self.set_dataset()
        self.modalities_names = self.dataset_train.modalities_names
        self.model = self.set_model()
     
        self.set_impute_dataset()
        self.client_train_dataset_len = [len(self.train_impute_dataset[i]) for i in range(len(self.test_impute_dataset))]
        self.client_test_dataset_len = [len(self.test_impute_dataset[i]) for i in range(len(self.test_impute_dataset))]
        self.test_samples = self.get_test_samples()
                
        self.optimizer_class = self.set_optimizer_class()
        self.mu_dim = flags.class_dim
        self.rec_weights = self.set_rec_weights()
        self.style_weights = self.set_style_weights()
        self.clfs = self.set_clfs()
        self.clients_subsets = self.get_client_subset(self.subsets, self.observed_mask)

        self.pickle_record = {"train": {}, "test": {}, "test_metric": {},
                            "clients_train_len": {c: len(self.train_impute_dataset[c]) for c in range(len(self.test_impute_dataset))},
                            "clients_test_len": {c: len(self.test_impute_dataset[c]) for c in range(len(self.test_impute_dataset))}}
    
    @abstractmethod
    def set_dataset(self):
        pass;
    
    @abstractmethod
    def get_test_samples(self):
        pass;

    @abstractmethod
    def get_modality(self, modality_name):
        pass
    
    @abstractmethod
    def set_clfs(self):
        pass

    def set_impute_dataset(self):
        pth = "./data/latent_missing_%s_%s.pk"%(self.flags.dataset, str(self.flags.test_missing_ratio))
        if not os.path.exists(pth):
            observed_mask = get_observed_mask(self.flags.client_num, self.num_modalities, self.flags.test_missing_ratio)
            with open(pth, "wb") as f:
                pickle.dump([observed_mask], f)
        else:
            with open(pth, "rb") as f:
                observed_mask = pickle.load(f)[0]
        self.observed_mask = observed_mask
        assert observed_mask.shape[0]==self.flags.client_num

        f_data = self.get_data
        if self.flags.dataset != 'eicu':
            train_labels_list = f_data(self.dataset_train)
            test_labels_list = f_data(self.dataset_test)
        torch.cuda.empty_cache()
        
        if self.flags.dataset=='celeba':
            train_ind = celeba_split(train_labels_list, self.flags.client_num)
            test_ind = celeba_split(test_labels_list, self.flags.client_num)
        elif self.flags.dataset=='eicu':
            def get_ind_list(train_str):
                dict_users = np.load(os.path.join(self.flags.dir_data, "dict_users_%s.npy"%train_str), allow_pickle = True).tolist()
                user_list = sorted(list(dict_users.keys()))
                ind_list = [dict_users[i] for i in user_list]
                assert self.flags.client_num==len(user_list)
                return ind_list
            train_ind = get_ind_list('train')
            test_ind = get_ind_list('test')
        else:
            train_ind, rand_set_all = noniid(train_labels_list, num_users=self.flags.client_num, shard_per_user=self.flags.class_per_user, num_classes=self.num_classes, rand_set_all=[])
            test_ind, _ = noniid(test_labels_list, num_users=self.flags.client_num, shard_per_user=self.flags.class_per_user, num_classes=self.num_classes, rand_set_all=rand_set_all)

        self.train_impute_dataset = []
        self.test_impute_dataset = []
        for c_i in range(self.flags.client_num):
            self.train_impute_dataset.append(Client_Dataset(self.dataset_train, train_ind[c_i], observed_mask[c_i], 'train'))
            self.test_impute_dataset.append(Client_Dataset(self.dataset_test, test_ind[c_i], observed_mask[c_i], 'test',))

    def get_data(self, dataset):
        if self.flags.dataset=='polymnist':
            dp = list(dataset.file_paths.keys())[0]
            label = [int(dataset.file_paths[dp][index].split(".")[-2]) for index in range(len(dataset))] 
        elif self.flags.dataset=='celeba':
            label = dataset.attributes
        elif self.flags.dataset=='mnistsvhntext':
            label = []
            for i in range(len(dataset)):
                label.append(int(dataset.labels_mnist[dataset.mnist_idx[i]]))

        return label

    def set_model(self):
        mods = {modality_name: self.get_modality(modality_name) for modality_name in self.modalities_names}
        self.modalities = mods
        self.subsets = get_subsets(mods)
        model = BaseMMVae(self.flags, mods, self.subsets)
        model = model.to(self.flags.device)
        
        if self.flags.load_saved:
            model.load_state_dict(torch.load(self.flags.checkpoint_pth))
            logger.info('Load checkpoint: %s'%self.flags.checkpoint_pth)
        return model

    def set_optimizer_class(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info('num parameters: %.2f M'%(float(total_params)/1e6))
        # optimizer = Adam(self.model.parameters(), lr=self.flags.initial_learning_rate, weight_decay=1e-6)
        optimizer_class = Adam
        return optimizer_class

    def get_client_subset(self, subsets, observed_mask):
        subset_all = []
        for c_i in range(self.flags.client_num):
            subset_i = []
            for c_j in range(self.flags.client_num):
                if c_i==c_j:
                    continue
                overlap_inds = np.where((observed_mask[c_i]*observed_mask[c_j])>0)[0]
                if len(overlap_inds)>0:
                    subset_name = '_'.join(sorted([self.modalities_names[i] for i in overlap_inds]))
                    assert subset_name in subsets.keys()
                    subset_i.append(subset_name)

            subset_all.append(sorted(list(set(subset_i))))
        return subset_all
