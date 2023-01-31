import os

import random
import numpy as np 

import PIL.Image as Image
from PIL import ImageFont

import torch
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import accuracy_score

from modalities.MNIST import MNIST
from modalities.SVHN import SVHN
from modalities.Text import Text

from mnistsvhntext.SVHNMNISTDataset import SVHNMNIST
from mnistsvhntext.networks.ConvNetworkImgClfMNIST import ClfImg as ClfImgMNIST
from mnistsvhntext.networks.ConvNetworkImgClfSVHN import ClfImgSVHN
from mnistsvhntext.networks.ConvNetworkTextClf import ClfText as ClfText

from mnistsvhntext.networks.ConvNetworksImgMNIST import EncoderImg, DecoderImg
from mnistsvhntext.networks.ConvNetworksImgSVHN import EncoderSVHN, DecoderSVHN
from mnistsvhntext.networks.ConvNetworksTextMNIST import EncoderText, DecoderText

from utils.BaseExperiment import BaseExperiment_impute

class MNISTSVHNText_impute(BaseExperiment_impute):
    data_batch_size = 10000
    dataset_name = 'mnistsvhntext'
    plot_img_size = torch.Size((3, 28, 28))   
    num_classes = 10
    labels = ['digit']

    def __init__(self, flags, alphabet):
        self.font = ImageFont.truetype('FreeSerif.ttf', 38)
        super().__init__(flags, alphabet)
        self.eval_metric = accuracy_score        

    def get_transform_mnist(self):
        transform_mnist = transforms.Compose([transforms.ToTensor(),
                                              transforms.ToPILImage(),
                                              transforms.Resize(size=(28, 28), interpolation=Image.BICUBIC),
                                              transforms.ToTensor()])
        return transform_mnist;


    def get_transform_svhn(self):
        transform_svhn = transforms.Compose([transforms.ToTensor()])
        return transform_svhn;

    def set_dataset(self):
        transform_mnist = self.get_transform_mnist();
        transform_svhn = self.get_transform_svhn();
        transforms = [transform_mnist, transform_svhn];
        train = SVHNMNIST(self.flags,
                          self.alphabet,
                          train=True,
                          transform=transforms)
        test = SVHNMNIST(self.flags,
                         self.alphabet,
                         train=False,
                         transform=transforms)
        self.dataset_train = train;
        self.dataset_test = test;

    def get_modality(self, modality_name):
        if modality_name=='mnist':
            mod = MNIST('mnist', EncoderImg(self.flags), DecoderImg(self.flags),
                        self.flags.class_dim, self.flags.style_m1_dim, 'laplace');
        elif modality_name=='svhn':
            mod = SVHN('svhn', EncoderSVHN(self.flags), DecoderSVHN(self.flags),
                        self.flags.class_dim, self.flags.style_m2_dim, 'laplace',
                        self.plot_img_size);
        elif modality_name=='text':
            mod = Text('text', EncoderText(self.flags), DecoderText(self.flags),
                        self.flags.class_dim, self.flags.style_m3_dim, 'categorical',
                        self.flags.len_sequence,
                        self.alphabet,
                        self.plot_img_size,
                        self.font);
        else:
            raise KeyError()
        return mod

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
        model_clf_m1 = None;
        model_clf_m2 = None;
        model_clf_m3 = None;
        if self.flags.use_clf:
            model_clf_m1 = ClfImgMNIST();
            model_clf_m1.load_state_dict(torch.load(os.path.join(self.flags.dir_clf, 'trained_clfs_mst',
                                                                 self.flags.clf_save_m1)))
            model_clf_m1 = model_clf_m1.to(self.flags.device);
            model_clf_m1 = model_clf_m1.eval()

            model_clf_m2 = ClfImgSVHN();
            model_clf_m2.load_state_dict(torch.load(os.path.join(self.flags.dir_clf, 'trained_clfs_mst',
                                                                 self.flags.clf_save_m2)))
            model_clf_m2 = model_clf_m2.to(self.flags.device);
            model_clf_m2 = model_clf_m2.eval()

            model_clf_m3 = ClfText(self.flags);
            model_clf_m3.load_state_dict(torch.load(os.path.join(self.flags.dir_clf, 'trained_clfs_mst', 
                                                                 self.flags.clf_save_m3)))
            model_clf_m3 = model_clf_m3.to(self.flags.device);
            model_clf_m3 = model_clf_m3.eval()

        clfs = {'mnist': model_clf_m1,
                'svhn': model_clf_m2,
                'text': model_clf_m3}
        return clfs;

    def get_prediction_from_attr(self, attr, index=None):
        pred = np.argmax(attr, axis=1).astype(int)
        return pred
        
    def eval_label(self, values, labels, index):
        pred = self.get_prediction_from_attr(values)
        return self.eval_metric(labels, pred)
    

    def set_rec_weights(self):
        rec_weights = dict();
        ref_mod_d_size = self.modalities['svhn'].data_size.numel();
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key];
            numel_mod = mod.data_size.numel()
            rec_weights[mod.name] = float(ref_mod_d_size/numel_mod)
        return rec_weights;


    def set_style_weights(self):
        weights = dict();
        weights['mnist'] = self.flags.beta_m1_style;
        weights['svhn'] = self.flags.beta_m2_style;
        weights['text'] = self.flags.beta_m3_style;
        return weights;
