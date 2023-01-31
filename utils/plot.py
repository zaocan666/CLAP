import textwrap
import torch
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchvision import transforms
import numpy as np
from PIL import ImageDraw
from PIL import ImageFont
from PIL import Image
import cv2

from utils import text as text

def tensor2np(img:torch.Tensor):
    img_np = np.array(img.mul(255).byte()).transpose(1,2,0)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_np

def draw_border(img:torch.Tensor, color=(0,0,255)):
    img_np = tensor2np(img)
    img_draw = cv2.rectangle(img_np, pt1=(0,0), pt2=(img_np.shape[0], img_np.shape[1]), color=color, thickness=4)
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
    img_t = transforms.ToTensor()(img_draw)
    return img_t

def create_fig(fn, img_data, num_img_row, save_figure=False):
    if save_figure:
        save_image(img_data.data.cpu(), fn, nrow=num_img_row);
    grid = make_grid(img_data, nrow=num_img_row);
    # plot = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy();
    plot = tensor2np(grid)
    return plot;


def text_to_pil(t, imgsize, alphabet, font, w=128, h=128, linewidth=8):
    blank_img = torch.ones([imgsize[0], w, h]);
    pil_img = transforms.ToPILImage()(blank_img.cpu()).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    text_sample = text.tensor_to_text(alphabet, t)[0]
    text_sample = ''.join(text_sample).translate({ord('*'): None})
    lines = textwrap.wrap(''.join(text_sample), width=linewidth)
    y_text = h
    num_lines = len(lines);
    for l, line in enumerate(lines):
        width, height = font.getsize(line)
        draw.text((0, (h/2) - (num_lines/2 - l)*height), line, (0, 0, 0), font=font)
        y_text += height
    if imgsize[0] == 3:
        text_pil = transforms.ToTensor()(pil_img.resize((imgsize[1], imgsize[2]),
                                                        Image.ANTIALIAS));
    else:
        text_pil = transforms.ToTensor()(pil_img.resize((imgsize[1], imgsize[2]),
                                                        Image.ANTIALIAS).convert('L'));
    return text_pil;


def text_to_pil_celeba(t, imgsize, alphabet, font, w=256, h=256):
    blank_img = torch.ones([3, w, h]);
    pil_img = transforms.ToPILImage()(blank_img.cpu()).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    text_sample = text.tensor_to_text(alphabet, t)[0]
    text_sample = ''.join(text_sample).translate({ord('*'): None})
    lines = textwrap.wrap(text_sample, width=16)
    y_text = h
    num_lines = len(lines);
    for l, line in enumerate(lines):
        width, height = font.getsize(line)
        draw.text((0, (h/2) - (num_lines/2 - l)*height), line, (0, 0, 0), font=font)
        y_text += height
    text_pil = transforms.ToTensor()(pil_img.resize((imgsize[1], imgsize[2]),
                                                    Image.ANTIALIAS));
    return text_pil;


def text_to_pil_mimic(t, imgsize, alphabet, font, w=512, h=512):
    blank_img = torch.ones([1, w, h]);
    pil_img = transforms.ToPILImage()(blank_img.cpu()).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    text_sample = text.tensor_to_text(alphabet, t)[0]
    text_sample = ''.join(text_sample).translate({ord('*'): None})
    lines = textwrap.wrap(text_sample, width=64)
    y_text = h
    num_lines = len(lines);
    for l, line in enumerate(lines):
        width, height = font.getsize(line)
        draw.text((0, (h/2) - (num_lines/2 - l)*height), line, (0, 0, 0), font=font)
        y_text += height
    text_pil = transforms.ToTensor()(pil_img.resize((imgsize[1], imgsize[2]),
                                                    Image.ANTIALIAS).convert('L'));
    return text_pil;
