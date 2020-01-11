# -------------------------------------------------------------------------------
# Titre : vae.py
# Projet : Granolar
# Description : train, test and reconstruct the database
# -------------------------------------------------------------------------------

import torch
from torch import optim
import numpy as np

from src.vae_model import VAE_Model

GRAIN_SIZE = 512
CHANNEL = 1
LEARNING_RATE = 1e-3
# Reconstruction + KL divergence losses summed over all elements and batch
beta = 0  # value for train testing


def loss_function_2(recon_x, x, mu_z, log_var_z):
    # BCE = torch.nn.functional.binary_cross_entropy(recon_x, x.view(16, 1, -1), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_var_z - mu_z.pow(2) - log_var_z.exp())
    return KLD


def loss_function(x, mu_z, log_var_z, mu_recon, log_var_recon):
    # p(z | x) = - log(sigma) - 0.5 * log(2*pi) - (x - mu)^2 / 2 * sigma ^ 2
    recon_loss = torch.sum(log_var_recon - 0.5 * np.log(2 * np.pi)
                           + ((x.view(-1, GRAIN_SIZE) - mu_recon).pow(2)) / (2 * torch.exp(log_var_recon).pow(2)))
    # Kullback-Leibler divergence
    KLD = -0.5 * torch.sum(1 + log_var_z - mu_z.pow(2) - log_var_z.exp())
    return recon_loss + beta * KLD


class VAE:
    # device is gpu if possible
    def __init__(self, train_loader, test_loader, batch_size=128, seed=1, cuda=False):
        # TODO: get GRAIN_SIZE and CHANNEL from train and/or test_dataset
        channel = 1
        torch.manual_seed(seed)
        # device is gpu if possible
        self.device = torch.device("cuda" if cuda else "cpu")
        # send the model to device
        self.model = VAE_Model(batch_size, channel, GRAIN_SIZE).to(self.device)
        # set the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.batch_size = batch_size
        self.epoch = 0
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self, log_interval=10):
        self.model.train()
        self.epoch += 1
        train_loss = 0
        # for each batch
        for batch_idx, data in enumerate(self.train_loader):
            print("DATA_LEN:", len(data))
            print("DATA:", data)
            data = data.to(self.device)
            self.optimizer.zero_grad()
            # get the variables
            data_recon, mu_z, log_var_z, mu_recon, log_var_recon = self.model(data)
            loss = loss_function(data, mu_z, log_var_z, mu_recon, log_var_recon)
            # loss = loss_function_2(data_recon, data, mu_z, log_var_z)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            # print
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, batch_idx * len(data), len(self.train_loader.dataset),
                                100. * batch_idx / len(self.train_loader),
                                loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            self.epoch, train_loss / len(self.train_loader.dataset)))
        return train_loss

    def test(self, log_interval=10):
        self.model.eval()
        self.epoch += 1
        test_loss = 0
        comparison = None
        with torch.no_grad():
            # for i, (data, _) in enumerate(self.test_loader):
            for batch_idx, data in enumerate(self.test_loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                # get the variables
                data_recon, mu_z, log_var_z, mu_recon, log_var_recon = self.model(data)
                # mu_z, log_var_z, mu_recon, log_var_recon = self.model(data)
                test_loss += loss_function(data_recon, mu_z, log_var_z, mu_recon, log_var_recon)
                # loss = loss_function_2(data_recon, data, mu_z, log_var_z)
                test_loss += test_loss.item()
                self.optimizer.step()
                # print
                if batch_idx % log_interval == 0:
                    print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss test: {:.6f}'.format(
                        self.epoch, batch_idx * len(data), len(self.test_loader.dataset),
                                    100. * batch_idx / len(self.test_loader),
                                    test_loss.item() / len(data)))
        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(
            self.epoch, test_loss / len(self.test_loader.dataset)))
        return test_loss

    def create_sample(self):
        with torch.no_grad():
            latent_sample = torch.randn(self.model.batch_size, self.model.latent_size).to(self.device)
            sample = self.model.decode(latent_sample)
            return sample

