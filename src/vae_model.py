# -------------------------------------------------------------------------------
# Titre : vae_model.py
# Projet : Granolar
# Description : modeling of the VAE
# -------------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from torch.nn import functional


SIZE_IO = 1
BATCH_SIZE = 128


class VAE_Model(nn.Module):
    def __init__(self):
        super(VAE_Model, self).__init__()

        # encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(SIZE_IO, 400, kernel_size=5, stride=1),
            torch.nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Conv1d(400, 200, kernel_size=8, stride=2),
            torch.nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Conv1d(200, 100, kernel_size=10, stride=4),
            torch.nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Conv1d(100, 20, kernel_size=13, stride=4),
            torch.nn.BatchNorm1d(100),
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
            torch.nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.ConvTranspose1d(100, 200, kernel_size=10, stride=4),
            torch.nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.ConvTranspose1d(200, 400, kernel_size=8, stride=2),
            torch.nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.ConvTranspose1d(400, SIZE_IO, kernel_size=5, stride=1),
            nn.ReLU())

        print(self.encoder)
        print('size_encoder:', len(self.encoder))
        print(self.decoder)
        print('size_decoder:', len(self.decoder))

    def encode(self, signal):
        x = self.encoder(signal)
        x = x.view(-1, BATCH_SIZE * SIZE_IO * 1)
        mu_z = functional.relu(self.fc1(x))
        mu_z = self.fc2(mu_z)
        log_var_z = functional.relu(self.fc1(x))
        log_var_z = self.fc2(log_var_z)
        return mu_z, log_var_z

    def decode(self, z):
        x = functional.relu(self.fc3(z))
        x = functional.relu(self.fc4(x))
        mu_x = functional.relu(self.decoder(x))
        mu_x = torch.tanh(self.fc4(mu_x))
        log_x = functional.relu(self.decoder(x))
        log_x = torch.tanh(self.fc4(log_x))
        return mu_x.view(-1, SIZE_IO), log_x.view(-1, SIZE_IO)

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, SIZE_IO))
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

