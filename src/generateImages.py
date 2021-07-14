import argparse
import os

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt, prod
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset
from torchvision.utils import save_image, make_grid
from utils import get_mean, kl_divergence


def gen(modelFile):
    model = torch.load_state_dict(torch.load(modelFile))
    N=64
    samples_list = model.generate(N)
    for i, samples_list in enumerate(samples_list):
        samples = samples_list.data
        samples = samples.view(N,*samples.size()[1:])
        save_image(samples,
                   '{}/Generazione_{}_{}.png'.format("./", i, modelFile),
                   nrow=int(sqrt(N)))

#torch.load("2021-07-08T16:33:42.124737_u0_mzjw/model.rar", map_location = torch.device("cpu"))



if __name__=="__main__"():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=argparse.FileType('r'), nargs=1)
    args = parser.parse_args()
    gen(args.model)
    print("Done")