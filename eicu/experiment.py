import torch
from sklearn.metrics import roc_auc_score
import numpy as np
from utils.BaseExperiment import BaseExperiment_impute
from eicu.eicu_dataset import eicuDataset
from modalities.Modality import Modality
from eicu.networks.eicu_coder import eicu_encoder, eicu_decoder
from eicu.flags import EICU_SUBSETS

class eicuExperiment_impute(BaseExperiment_impute):
    dataset_name = 'celeba'
    plot_img_size = torch.Size((3, 64, 64))
    labels = ['auc']

    def __init__(self, flags, alphabet):
        super().__init__(flags, alphabet)
        self.eval_metric = roc_auc_score        
        for s in EICU_SUBSETS:
            assert s in self.subsets.keys()

    def set_dataset(self):
        d_train = eicuDataset(self.flags.dir_data)
        d_eval = d_train
        self.dataset_train = d_train;
        self.dataset_test = d_eval;

    def get_modality(self, modality_name):
        mod = Modality(modality_name, 
                    eicu_encoder(self.flags),
                    eicu_decoder(self.flags),
                    self.flags.class_dim,
                    self.flags.style_dim,
                    lhood_name=self.flags.likelihood)
        return mod

    def get_test_samples(self, num_images=10):
        return None

    def set_clfs(self):
        return None

    def get_prediction_from_attr(self, attr, index=None):
        raise NotImplementedError()
        
    def eval_label(self, values, labels, index):
        raise NotImplementedError()

    def set_rec_weights(self):
        rec_weights = dict()
        for k, m_key in enumerate(self.modalities.keys()):
            rec_weights[m_key] = 1.0/len(self.modalities.keys())
        return rec_weights

    def set_style_weights(self):
        weights = {m: self.flags.beta_style for m in self.modalities.keys()}
        return weights

    def get_prediction_from_attr_random(self, values, index=None):
        raise NotImplementedError
