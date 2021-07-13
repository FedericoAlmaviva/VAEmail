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

from vis import plot_embeddings, plot_kls_df
from models.vae_mnist import MNIST
from models.vae_svhn import SVHN

class MinimalModel(nn.Module):
    def __init__(self,params):
        super(MinimalModel, self).__init__()
        self.vaes = nn.ModuleList([MNIST(params), SVHN(params)])
        self.params = params
        self.vaes[0].llik_scaling = prod(self.vaes[1].dataSize) / prod(self.vaes[0].dataSize) \
            if params.llik_scaling == 0 else params.llik_scaling

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda'):
        if not (os.path.exists('../data/train-ms-mnist-idx.pt')
                and os.path.exists('../data/train-ms-svhn-idx.pt')
                and os.path.exists('../data/test-ms-mnist-idx.pt')
                and os.path.exists('../data/test-ms-svhn-idx.pt')):
            raise RuntimeError('Generate transformed indices with the script in bin')
        # get transformed indices
        t_mnist = torch.load('../data/train-ms-mnist-idx.pt')
        t_svhn = torch.load('../data/train-ms-svhn-idx.pt')
        s_mnist = torch.load('../data/test-ms-mnist-idx.pt')
        s_svhn = torch.load('../data/test-ms-svhn-idx.pt')

        # load base datasets
        t1, s1 = self.vaes[0].getDataLoaders(batch_size, shuffle, device)
        t2, s2 = self.vaes[1].getDataLoaders(batch_size, shuffle, device)

        train_mnist_svhn = TensorDataset([
            ResampleDataset(t1.dataset, lambda d, i: t_mnist[i], size=len(t_mnist)),
            ResampleDataset(t2.dataset, lambda d, i: t_svhn[i], size=len(t_svhn))
        ])
        test_mnist_svhn = TensorDataset([
            ResampleDataset(s1.dataset, lambda d, i: s_mnist[i], size=len(s_mnist)),
            ResampleDataset(s2.dataset, lambda d, i: s_svhn[i], size=len(s_svhn))
        ])

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        train = DataLoader(train_mnist_svhn, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(test_mnist_svhn, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train, test

    def forward(self, x, K=1):
        qz_xs, zss = [], []
        # initialise cross-modal matrix
        px_zs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        for m, vae in enumerate(self.vaes):
            qz_x, px_z, zs = vae(x[m], K=K)
            qz_xs.append(qz_x)
            zss.append(zs)
            px_zs[m][m] = px_z  # fill-in diagonal
        for e, zs in enumerate(zss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal
                    px_zs[e][d] = vae.px_z(*vae.dec(zs))
        return qz_xs, px_zs, zss