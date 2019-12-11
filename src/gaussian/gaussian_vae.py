#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional


def loss_function(x, mu_z, logvar_z, mu, logvar, beta):
    # p(z | x) = - log(sigma) - 0.5 * log(2*pi) - (x - mu)^2 / 2 * sigma ^ 2
    recon_loss = torch.sum(logvar_z + 0.918938533204672741780 + ((x.view(-1, 784) - mu_z).pow(2)) / (2 * torch.exp(logvar_z).pow(2)))
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * KLD


class VAE_gauss(nn.Module):
    def __init__(self):
        super(VAE_gauss, self).__init__()

        # input encoder layer
        self.fc1 = nn.Linear(784, 400)
        # gaussian encoder output
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        # first decoder layer
        self.fc3 = nn.Linear(20, 400)
        # gaussian decoder output
        self.fc41 = nn.Linear(400, 784)
        self.fc42 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = functional.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = functional.relu(self.fc3(z))
        return self.fc41(h3), self.fc42(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        sample_z = self.reparameterize(mu, logvar)
        mu_z, logvar_z = self.decode(sample_z)
        return mu_z, logvar_z, mu, logvar