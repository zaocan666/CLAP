
import argparse
from utils.BaseFlags import parser as parser

EICU_SUBSETS = ['m11_m12_m4_m7_m8', 'm1_m2_m4_m7_m8_m9', 'm1_m11_m12_m2_m4', 'm11_m5_m7', 'm11_m12_m2_m3_m4_m5_m7', 'm0_m12_m3_m4_m5', 
                'm11_m2_m3_m4_m7_m8_m9', 'm0_m12_m4_m8', 'm0_m11_m2_m3_m4_m7_m8', 'm0_m1_m11_m2_m3_m4_m5_m7_m8_m9', 'm0_m1_m2_m3_m4_m7_m8']

# DATA DEPENDENT
# to be set by experiments themselves
parser.add_argument('--clf_modality', type=int, default=0, help="dimension of varying factor latent space")
parser.add_argument('--clf_learning_rate', type=float, default=0.01, help="starting learning rate")
parser.add_argument('--clf_batchsize', type=int, default=256, help="starting learning rate")

parser.add_argument('--likelihood', type=str, default='laplace', help="output distribution")
parser.add_argument('--style_dim', type=int, default=0, help="style dimensionality")  # TODO: use modality-specific style dimensions?
