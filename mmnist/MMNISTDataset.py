import numpy as np
import argparse
import torch
import os
import glob

from torch.utils.data import Dataset
from torchvision.utils import save_image
from torchvision import datasets, transforms
from PIL import Image


class MMNISTDataset(Dataset):
    """Multimodal MNIST Dataset."""

    def __init__(self, unimodal_datapaths, transform=None, target_transform=None, sample_ratio=1.0):
        """
            Args: unimodal_datapaths (list): list of paths to weakly-supervised unimodal datasets with samples that
                    correspond by index. Therefore the numbers of samples of all datapaths should match.
                transform: tranforms on colored MNIST digits.
                target_transform: transforms on labels.
        """
        super().__init__()
        self.num_modalities = len(unimodal_datapaths)
        self.unimodal_datapaths = unimodal_datapaths
        self.transform = transform
        self.target_transform = target_transform

        # save all paths to individual files
        self.file_paths = {dp: [] for dp in self.unimodal_datapaths}
        for dp in unimodal_datapaths:
            files = glob.glob(os.path.join(dp, "*.png"))
            self.file_paths[dp] = files
        
        self.file_paths = {dp.split('/')[-1]: self.file_paths[dp] for dp in self.file_paths.keys()}
        self.modalities_names = sorted(self.file_paths.keys())
        
        d1 = list(self.file_paths.keys())[0]
        if sample_ratio<1:
            num_files = len(self.file_paths[d1])
            inds = np.random.choice(np.arange(num_files), round(num_files*sample_ratio))
            for dp in self.file_paths.keys():
                self.file_paths[dp] = [self.file_paths[dp][i] for i in inds]

        num_files = len(self.file_paths[d1])
        # assert that each modality has the same number of images
        for files in self.file_paths.values():
            assert len(files) == num_files
        self.num_files = num_files

    def __getitem__(self, index):
        """
        Returns a tuple (images, labels) where each element is a list of
        length `self.num_modalities`.
        """
        files = {dp: self.file_paths[dp][index] for dp in self.file_paths.keys()}
        images = {dp: Image.open(files[dp]) for dp in files.keys()}
        label = [int(files[dp].split(".")[-2]) for dp in files.keys()][0]

        # transforms
        if self.transform:
            images = {dp: self.transform(images[dp]) for dp in files.keys()}
        if self.target_transform:
            label = self.transform(label)

        images_dict = images
        return images_dict, label   # NOTE: for MMNIST, labels are shared across modalities, so can take one value

    def __len__(self):
        return self.num_files
