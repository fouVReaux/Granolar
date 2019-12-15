# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from torch.nn import functional


SIZE_IO = 832


class VAE_Model(nn.Module):
    def __init__(self):
        super(VAE_Model, self).__init__()

        # encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(SIZE_IO, 400, kernel_size=5, stride=1),
            nn.ReLU(),
            torch.nn.BatchNorm1d(400),
            nn.Conv1d(400, 200, kernel_size=8, stride=2),
            nn.ReLU(),
            torch.nn.BatchNorm1d(200),
            nn.Conv1d(200, 100, kernel_size=10, stride=4),
            nn.ReLU(),
            torch.nn.BatchNorm1d(100),
            nn.Conv1d(100, 20, kernel_size=13, stride=4),
            nn.ReLU())
        # gaussian encoder
        self.fc1 = nn.Linear(20, 1000)
        self.fc2 = nn.Linear(1000, 10)

        # gaussian decoder
        self.fc3 = nn.Linear(10, 1000)
        self.fc4 = nn.Linear(1000, 20)
        # gaussian decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(20, 100, kernel_size=13, stride=4),
            nn.ReLU(),
            torch.nn.BatchNorm1d(100),
            nn.ConvTranspose1d(100, 200, kernel_size=10, stride=4),
            nn.ReLU(),
            torch.nn.BatchNorm1d(200),
            nn.ConvTranspose1d(200, 400, kernel_size=8, stride=2),
            nn.ReLU(),
            torch.nn.BatchNorm1d(400),
            nn.ConvTranspose1d(400, SIZE_IO, kernel_size=5, stride=1),
            nn.ReLU())

        print(self.encoder)
        print('size_encoder:', len(self.encoder))
        print(self.decoder)
        print('size_decoder:', len(self.decoder))

    def encode(self, signal):
        x = self.encoder(signal)
        return self.mu(x), self.log_var(x)

    def decode(self, z):
        h3 = functional.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, SIZE_IO))
        z = self.reparameterize(mu, log_var)
        mu_z, log_var_z = self.decode(z)
        return mu_z, log_var_z, mu, log_var

