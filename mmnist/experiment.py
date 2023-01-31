import os

import random
import numpy as np 

import torch
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import accuracy_score
import glog as logger
from PIL import ImageFont

from modalities.CMNIST import CMNIST

from mmnist.MMNISTDataset import MMNISTDataset

from mmnist.networks.ConvNetworkImgClfCMNIST import ClfImg as ClfImgCMNIST

from mmnist.networks.ConvNetworksImgCMNIST import EncoderImg, DecoderImg

from utils.BaseExperiment import BaseExperiment_impute


class MMNISTExperiment_impute(BaseExperiment_impute):
    data_batch_size = 10000
    dataset_name = 'MMNIST'
    plot_img_size = torch.Size((3, 28, 28))   
    num_classes = 10
    labels = ['digit']

    def __init__(self, flags, alphabet):
        super().__init__(flags, alphabet)
        self.eval_metric = accuracy_score

    def get_modality(self, modality_name):
        mod = CMNIST(modality_name, EncoderImg(self.flags, modality_name),
                        DecoderImg(self.flags), self.flags.class_dim,
                        self.flags.style_dim, self.flags.likelihood_mnist)
        return mod
    
    def set_dataset(self):
        transform = transforms.Compose([transforms.ToTensor()])
        train = MMNISTDataset(self.flags.unimodal_datapaths_train, transform=transform)
        test = MMNISTDataset(self.flags.unimodal_datapaths_test, transform=transform, sample_ratio=self.flags.test_set_ratio)
        self.dataset_train = train
        self.dataset_test = test


    def get_test_samples(self, num_images=10):
        # dataset = self.dataset_train
        dataset = self.dataset_test
        samples = []
        for i in range(num_images):
            while True:
                c_ind = random.randint(0, self.flags.client_num-1)
                dataset = self.test_impute_dataset[c_ind]
                n_test = len(dataset)
                ix = random.randint(0, n_test-1)
                sample, target, observed_mask = dataset[ix]
                if target == i:
                    for k, key in enumerate(sample):
                        sample[key] = sample[key]
                    samples.append([ix, sample, observed_mask, c_ind])
                    break
        return samples

    def set_clfs(self):
        clfs = {m: None for m in self.modalities_names}
        if self.flags.use_clf:
            pretrained_classifier_path = os.path.join(self.flags.dir_clf, 'trained_clfs_polyMNIST')
            for m in self.modalities_names:
                model_clf = ClfImgCMNIST()
                model_clf.load_state_dict(torch.load(os.path.join(pretrained_classifier_path, 'pretrained_img_to_digit_clf_'+m)))
                model_clf = model_clf.to(self.flags.device)
                model_clf = model_clf.eval()
                clfs[m] = model_clf
            for m, clf in clfs.items():
                if clf is None:
                    raise ValueError("Classifier is 'None' for modality %s" % str(m))
        return clfs

    def get_prediction_from_attr(self, attr, index=None):
        pred = np.argmax(attr, axis=1).astype(int)
        return pred
        
    def eval_label(self, values, labels, index):
        pred = self.get_prediction_from_attr(values)
        return self.eval_metric(labels, pred)
    
    def set_rec_weights(self):
        rec_weights = dict()
        for k, m_key in enumerate(self.modalities.keys()):
            rec_weights[m_key] = 1.0/len(self.modalities.keys())
        return rec_weights

    def set_style_weights(self):
        weights = {m: self.flags.beta_style for m in self.modalities.keys()}
        return weights