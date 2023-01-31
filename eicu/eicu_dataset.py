import os
import torch
from torch.utils.data import Dataset
import numpy as np

class eicuDataset(Dataset):
    def __init__(self, data_dir):
        self.data_all = np.load(os.path.join(data_dir, "data_new.npy"), allow_pickle = True)
        self.features = torch.Tensor([self.data_all[idx][0] for idx in range(len(self.data_all))])
        self.labels = torch.Tensor([self.data_all[idx][1] for idx in range(len(self.data_all))])
        self.modalities_names = ['m%d'%i for i in range(self.features.shape[2])]
    
    def __getitem__(self, index):
        sample_dict = {'m%d'%i: self.features[index, :, i] for i in range(self.features.shape[2])}
        label = self.labels[index]
        return sample_dict, label

class eicuDataset_part(Dataset):
    def __init__(self, data_all, data_dir, train_str, clf_modality):
        dict_users = np.load(os.path.join(data_dir, "dict_users_%s.npy"%train_str), allow_pickle = True).tolist()
        all_idx = np.concatenate([dict_users[k] for k in dict_users.keys()])

        self.features = torch.Tensor([data_all[idx][0] for idx in all_idx])[:, :, clf_modality]
        self.labels = torch.Tensor([data_all[idx][1] for idx in all_idx]).long()

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x,y