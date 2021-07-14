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
import models.minimalmodel

def gen(modelFile):
    #ldd_mdl = torch.load(modelFile) if torch.cuda.is_available() else torch.load(modelFile,map_location = torch.device("cpu"))
    #model = minimalmodel.MinimalModel(*args)
    print(modelFile)

    model = torch.load(str(modelFile[0]),map_location = torch.device("cpu"))
    N=64
    K=9

    model.eval()
    with torch.no_grad():
        pz = model.vaes[0].pz(*model.vaes[0].pz_params)
        latents = pz.rsample(torch.Size([N]))
        px_z = model.vaes[0].px_z(*model.vaes[0].dec(latents))
        data = px_z.sample(torch.Size([K]))
    ret=  data.view(-1, *data.size()[3:])

    # wrangle things so they come out tiled
    samples = ret.view(K, N, *ret.size()[1:]).transpose(0, 1)  # N x K x 1 x 28 x 28
    s = [make_grid(t, nrow=int(sqrt(K)), padding=0) for t in samples]
    save_image(torch.stack(s), '../experiments/Generazione.png', nrow=int(sqrt(N)))

    '''
    with torch.no_grad():
        pz = model.vaes[0].pz(*model.pz_params)
        latents = pz.rsample(torch.Size([N]))
        px_z = model.vaes[0].px_z(*model.vaes[0].dec(latents))
        data = px_z.sample(torch.Size([N]))
    samples_list =  data.view(-1, *data.size()[3:])
    print(len(samples_list))


    # wrangle things so they come out tiled
    samples = samples_list.view(K, N, *samples_list.size()[1:]).transpose(0, 1)  # N x K x 1 x 28 x 28
    s = [make_grid(t, nrow=int(sqrt(K)), padding=0) for t in samples]

    save_image(torch.stack(samples), '../experiments/Generazione_{}.png'.format(i), nrow=int(sqrt(N)))

    
    for i, samples_list in enumerate(samples_list):
        samples = samples_list.data
        samples = samples.view(1,*samples.size()[1:])
        save_image(samples,'../experiments/Generazione_{}.png'.format(i),nrow=int(sqrt(N)))
'''
#torch.load("2021-07-08T16:33:42.124737_u0_mzjw/model.rar", map_location = torch.device("cpu"))

#python generateImages.py --modelPth ~/Downloads/test_models/all/model_epoch_6.pt


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelPth', nargs=1)
    args = parser.parse_args()
    gen(args.modelPth)
    print("Done")