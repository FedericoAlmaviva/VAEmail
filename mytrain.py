from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import MNIST
import util
import myModel

def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))

'''
def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    model = model.MVAE(checkpoint['n_latents'])
    model.load_state_dict(checkpoint['state_dict'])
    return model
'''

def elbo_loss(recon_image, image, recon_text, text, mu, logvar,
              lambda_image=1.0, lambda_text=1.0, annealing_factor=1):
    """Bimodal ELBO loss function.

    @param recon_image: torch.Tensor
                        reconstructed image
    @param image: torch.Tensor
                  input image
    @param recon_text: torch.Tensor
                       reconstructed text probabilities
    @param text: torch.Tensor
                 input text (one-hot)
    @param mu: torch.Tensor
               mean of latent distribution
    @param logvar: torch.Tensor
                   log-variance of latent distribution
    @param lambda_image: float [default: 1.0]
                         weight for image BCE
    @param lambda_text: float [default: 1.0]
                       weight for text BCE
    @param annealing_factor: integer [default: 1]
                             multiplier for KL divergence term
    @return ELBO: torch.Tensor
                  evidence lower bound
    """
    image_bce, text_bce = 0, 0  # default params
    if recon_image is not None and image is not None:
        image_bce = torch.sum(binary_cross_entropy_with_logits(
            recon_image.view(-1, 1 * 28 * 28),
            image.view(-1, 1 * 28 * 28)), dim=1)

    if recon_text is not None and text is not None:
        text_bce = torch.sum(cross_entropy(recon_text, text), dim=1)

    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    ELBO = torch.mean(lambda_image * image_bce + lambda_text * text_bce
                      + annealing_factor * KLD)
    return ELBO

def binary_cross_entropy_with_logits(input, target):
    """Sigmoid Activation + Binary Cross Entropy

    @param input: torch.Tensor (size N)
    @param target: torch.Tensor (size N)
    @return loss: torch.Tensor (size N)
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(
            target.size(), input.size()))

    return (torch.clamp(input, 0) - input * target
            + torch.log(1 + torch.exp(-torch.abs(input))))


def cross_entropy(input, target, eps=1e-6):
    """k-Class Cross Entropy (Log Softmax + Log Loss)

    @param input: torch.Tensor (size N x K)
    @param target: torch.Tensor (size N x K)
    @param eps: error to add (default: 1e-6)
    @return loss: torch.Tensor (size N)
    """
    if not (target.size(0) == input.size(0)):
        raise ValueError(
            "Target size ({}) must be the same as input size ({})".format(
                target.size(0), input.size(0)))

    log_input = F.log_softmax(input + eps, dim=1)
    y_onehot = Variable(log_input.data.new(log_input.size()).zero_())
    y_onehot = y_onehot.scatter(1, target.unsqueeze(1), 1)
    loss = y_onehot * log_input
    return -loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    execution_id = util.get_executionName()
    args = util.get_argparser().parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    if not args.cuda:
        print("WARNING: won't run on GPU")
    train_loader = torch.utils.data.DataLoader(
        MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True)
    N_mini_batches = len(train_loader)
    test_loader = torch.utils.data.DataLoader(
        MNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=False)
    if not os.path.isdir('./'+execution_id):
        os.makedirs('./'+execution_id)
        os.makedirs('./'+ execution_id+"/trained_models")
    util.saveargs(execution_id, args)

    model = myModel.MVAE(args.n_latents)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if(args.cuda):
        model.cuda()

    def train(epoch):
        model.train()
        train_loss_meter = AverageMeter()

        # NOTE: is_paired is 1 if the example is paired
        for batch_idx, (image, text) in enumerate(train_loader):
            if epoch < args.annealing_epochs:
                # compute the KL annealing factor for the current mini-batch in the current epoch
                annealing_factor = (float(batch_idx + (epoch - 1) * N_mini_batches + 1) /
                                    float(args.annealing_epochs * N_mini_batches))
            else:
                # by default the KL annealing factor is unity
                annealing_factor = 1.0

            if args.cuda:
                image = image.cuda()
                text = text.cuda()

            image = Variable(image)
            text = Variable(text)
            batch_size = len(image)

            # refresh the optimizer
            optimizer.zero_grad()

            # pass data through model
            recon_image_1, recon_text_1, mu_1, logvar_1 = model(image, text)
            recon_image_2, recon_text_2, mu_2, logvar_2 = model(image)
            recon_image_3, recon_text_3, mu_3, logvar_3 = model(text=text)

            # compute ELBO for each data combo
            joint_loss = elbo_loss(recon_image_1, image, recon_text_1, text, mu_1, logvar_1,
                                   lambda_image=args.lambda_image, lambda_text=args.lambda_text,
                                   annealing_factor=annealing_factor)
            image_loss = elbo_loss(recon_image_2, image, None, None, mu_2, logvar_2,
                                   lambda_image=args.lambda_image, lambda_text=args.lambda_text,
                                   annealing_factor=annealing_factor)
            text_loss = elbo_loss(None, None, recon_text_3, text, mu_3, logvar_3,
                                  lambda_image=args.lambda_image, lambda_text=args.lambda_text,
                                  annealing_factor=annealing_factor)
            train_loss = joint_loss + image_loss + text_loss
            train_loss_meter.update(train_loss.data, batch_size)

            # compute gradients and take step
            train_loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAnnealing-Factor: {:.3f}'.format(
                    epoch, batch_idx * len(image), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), train_loss_meter.avg, annealing_factor))
        st='====> Epoch: {}\tLoss: {:.4f}\tAnnealing-Factor: {:.3f}'.format(epoch, train_loss_meter.avg,annealing_factor)
        print(st)
        util.logEpoch(execution_id,st)


    def test(epoch):
        model.eval()
        test_loss_meter = AverageMeter()

        for batch_idx, (image, text) in enumerate(test_loader):
            if args.cuda:
                image = image.cuda()
                text = text.cuda()

            image = Variable(image, volatile=True)
            text = Variable(text, volatile=True)
            batch_size = len(image)

            recon_image_1, recon_text_1, mu_1, logvar_1 = model(image, text)
            recon_image_2, recon_text_2, mu_2, logvar_2 = model(image)
            recon_image_3, recon_text_3, mu_3, logvar_3 = model(text=text)

            joint_loss = elbo_loss(recon_image_1, image, recon_text_1, text, mu_1, logvar_1)
            image_loss = elbo_loss(recon_image_2, image, None, None, mu_2, logvar_2)
            text_loss = elbo_loss(None, None, recon_text_3, text, mu_3, logvar_3)
            test_loss = joint_loss + image_loss + text_loss
            test_loss_meter.update(test_loss.data, batch_size)

        print('====> Test Loss: {:.4f}'.format(test_loss_meter.avg))
        return test_loss_meter.avg


    best_loss = sys.maxunicode
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test_loss = test(epoch)
        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)
        # save the best model and current model
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'n_latents': args.n_latents,
            'optimizer': optimizer.state_dict(),
        }, is_best, folder='./trained_models')
