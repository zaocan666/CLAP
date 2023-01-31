import random
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pickle
import utils.utils as utils

def get_observed_mask(num, T, missing_ratio):
    observed_mask_ = np.zeros([num, T]) # [B, mod_num]
    for b in range(num):
        while True:
            observed_rand_i = np.random.rand(T)
            
            num_masked = round(T * missing_ratio)
            observed_mask_i = np.array([1]*T) # [mod_num]
            observed_mask_i[torch.Tensor(observed_rand_i).topk(num_masked).indices.numpy()] = 0
            assert observed_mask_i.sum()>0 and observed_mask_i.sum()<T

            if observed_mask_i.sum()>0 and observed_mask_i.sum()<T:
                break
        observed_mask_[b] = observed_mask_i
    return observed_mask_

class Client_Dataset(Dataset):
    def __init__(self, dataset, inds, observed_mask, train_type) -> None:
        self.dataset = dataset
        self.inds = inds
        self.observed_mask = torch.Tensor(observed_mask)
        self.train_type = train_type

    def __len__(self):
        return len(self.inds)
    
    def __getitem__(self, i):
        index = self.inds[i]
        x, label = self.dataset[index]
        observed_mask = self.observed_mask
        return x, label, observed_mask

def noniid(label_list, num_users, shard_per_user, num_classes, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    count = 0
    for i in range(len(label_list)):
        label = label_list[i]
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)
            count += 1

    shard_per_class = int(shard_per_user * num_users / num_classes)
    samples_per_user = int( count/num_users )
    # whether to sample more test samples per user

    for label in idxs_dict.keys():
        x = idxs_dict[label]
        random.shuffle(x)
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(label_list)[value])
        test.append(x)
    all_len = [len(dict_users[k]) for k in dict_users.keys()]

    ind_users_list = [dict_users[i] for i in range(num_users)]
    return ind_users_list, rand_set_all


def celeba_split(attributes, num_clients):
    data_len = len(attributes)

    inds = list(range(data_len))
    random.shuffle(inds)

    num_each_client = data_len//num_clients

    attr = attributes['Male'].values
    attr_n_inds = np.where(attr<0)[0]
    attr_p_inds = np.where(attr>0)[0]
    random.shuffle(attr_n_inds)
    random.shuffle(attr_p_inds)

    client_weights = list(range(num_clients*2, num_clients*3))
    client_weights = utils.reweight_weights(np.array(client_weights))      
    start_n = 0
    start_p = 0
    client_inds = []
    for c in range(num_clients):
        attr_n_c_num = round(len(attr_n_inds)*client_weights[c])
        attr_n_c_inds = attr_n_inds[start_n:start_n+attr_n_c_num]
        start_n = start_n + attr_n_c_num

        attr_p_c_num = num_each_client-attr_n_c_num
        attr_p_c_inds = attr_p_inds[start_p:start_p+attr_p_c_num]
        start_p = start_p + attr_p_c_num
        
        client_inds.append(np.concatenate([attr_n_c_inds, attr_p_c_inds], axis=0))

    return client_inds
