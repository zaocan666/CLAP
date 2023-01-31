

import torch
from PIL import ImageFont 

from modalities.Modality import Modality

from utils import utils
from utils import plot
from utils.save_samples import write_samples_text_to_file
from utils.text import tensor_to_text

font = ImageFont.truetype('FreeSerif.ttf', 38)

class Text(Modality):
    def __init__(self, enc, dec, len_sequence, alphabet, plotImgSize, font, style_dim):
        self.name = 'text';
        self.likelihood_name = 'categorical';
        self.alphabet = alphabet;
        self.len_sequence = len_sequence;
        self.data_size = torch.Size([len_sequence]);
        self.plot_img_size = plotImgSize;
        self.gen_quality_eval = False;
        self.file_suffix = '.txt';
        self.encoder = enc;
        self.decoder = dec;
        self.likelihood = self.get_likelihood(self.likelihood_name);
        self.style_dim = style_dim


    def save_data(self, d, fn, args):
        write_samples_text_to_file(tensor_to_text(self.alphabet,
                                                  d.unsqueeze(0)),
                                   fn);

 
    def plot_data(self, d):
        out = plot.text_to_pil(d, self.plot_img_size,
                               self.alphabet, font,
                              w=256, h=256, linewidth=16).unsqueeze(0)
        return out;
