

import os 
import random
import numpy as np 

import PIL.Image as Image
from PIL import ImageFont 
import torch
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import average_precision_score

from modalities.CelebaImg import Img
from modalities.CelebaText import Text
from celeba.CelebADataset import CelebaDataset
from celeba.networks.ConvNetworkImgClfCelebA import ClfImg as ClfImg
from celeba.networks.ConvNetworkTextClfCelebA import ClfText as ClfText

from celeba.networks.ConvNetworksImgCelebA import EncoderImg, DecoderImg
from celeba.networks.ConvNetworksTextCelebA import EncoderText, DecoderText

from utils.BaseExperiment import BaseExperiment_impute


LABELS = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
          'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair',
          'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
          'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
          'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
          'Mouth_Slightly_Open', 'Mustache',
          'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
          'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
          'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
          'Wearing_Earrings', 'Wearing_Hat',
          'Wearing_Lipstick', 'Wearing_Necklace',
          'Wearing_Necktie', 'Young'];


class CelebaExperiment_impute(BaseExperiment_impute):
    dataset_name = 'celeba'
    plot_img_size = torch.Size((3, 64, 64))
    labels = LABELS

    def __init__(self, flags, alphabet):
        self.font = ImageFont.truetype('FreeSerif.ttf', 38)
        super().__init__(flags, alphabet)
        self.eval_metric = average_precision_score

    def get_transform_celeba(self):
        offset_height = (218 - self.flags.crop_size_img) // 2
        offset_width = (178 - self.flags.crop_size_img) // 2
        crop = lambda x: x[:, offset_height:offset_height + self.flags.crop_size_img,
                         offset_width:offset_width + self.flags.crop_size_img]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(crop),
                                        transforms.ToPILImage(),
                                        transforms.Resize(size=(self.flags.img_size,
                                                                self.flags.img_size),
                                                          interpolation=Image.BICUBIC),
                                        transforms.ToTensor()])

        return transform;

    def set_dataset(self):
        transform = self.get_transform_celeba();
        d_train = CelebaDataset(self.flags, self.alphabet, partition=0, transform=transform)
        d_eval = CelebaDataset(self.flags, self.alphabet, partition=1, transform=transform)
        self.dataset_train = d_train;
        self.dataset_test = d_eval;

    def get_modality(self, modality_name):
        if modality_name=='img':
            mod = Img(EncoderImg(self.flags),
                   DecoderImg(self.flags),
                   self.plot_img_size,
                   style_dim=self.flags.style_img_dim);
        elif modality_name=='text':
            mod = Text(EncoderText(self.flags),
                    DecoderText(self.flags),
                    self.flags.len_sequence,
                    self.alphabet,
                    self.plot_img_size,
                    self.font,
                    style_dim=self.flags.style_text_dim);
        else:
            raise KeyError()
        return mod

    def get_test_samples(self, num_images=10):
        dataset = self.dataset_test
        samples = []
        for i in range(num_images):
            c_ind = random.randint(0, self.flags.client_num-1)
            dataset = self.test_impute_dataset[c_ind]
            n_test = len(dataset)
            ix = random.randint(0, n_test-1)
            sample, target, observed_mask = dataset[ix]
            
            for k, key in enumerate(sample):
                sample[key] = sample[key]
            samples.append([ix, sample, observed_mask, c_ind])
        return samples

    def set_clfs(self):
        model_clf_m1 = None;
        model_clf_m2 = None;
        if self.flags.use_clf:
            model_clf_m1 = ClfImg(self.flags);
            model_clf_m1.load_state_dict(torch.load(os.path.join(self.flags.dir_clf, 'trained_clfs_celeba',
                                                                 self.flags.clf_save_m1)))
            model_clf_m1 = model_clf_m1.to(self.flags.device);
            model_clf_m1 = model_clf_m1.eval()

            model_clf_m2 = ClfText(self.flags);
            model_clf_m2.load_state_dict(torch.load(os.path.join(self.flags.dir_clf, 'trained_clfs_celeba',
                                                                 self.flags.clf_save_m2)))
            model_clf_m2 = model_clf_m2.to(self.flags.device);
            model_clf_m2 = model_clf_m2.eval()

        clfs = {'img': model_clf_m1,
                'text': model_clf_m2}
        return clfs;

    def get_prediction_from_attr(self, attr, index=None):
        pred = np.argmax(attr, axis=1).astype(int)
        return pred
        
    def eval_label(self, values, labels, index):
        pred = self.get_prediction_from_attr(values)
        return self.eval_metric(labels, pred)

    def set_rec_weights(self):
        rec_weights = dict();
        ref_mod_d_size = self.modalities['img'].data_size.numel()/3;
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key];
            numel_mod = mod.data_size.numel()
            rec_weights[mod.name] = float(ref_mod_d_size/numel_mod)
        return rec_weights;


    def set_style_weights(self):
        weights = dict();
        weights['img'] = self.flags.beta_m1_style;
        weights['text'] = self.flags.beta_m2_style;
        return weights;
    
    def get_prediction_from_attr(self, values):
        return values.ravel();


    def get_prediction_from_attr_random(self, values, index=None):
        return values[:,index] > 0.5;


    def eval_label(self, values, labels, index):
        pred = values[:,index];
        gt = labels[:,index];
        try:
            ap = self.eval_metric(gt, pred) 
        except ValueError:
            raise ValueError()
        return ap;