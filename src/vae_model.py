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

        # channels = 1, 64, 32, 16 encoder = [conv(channels[i-1], channels [i] ...) for i in range(len(channels))]
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=3),
            torch.nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=8, stride=2, padding=5),
            torch.nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=10, stride=4, padding=5),
            torch.nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=13, stride=4, padding=7),
            torch.nn.BatchNorm1d(8),
            nn.ReLU())
        # gaussian encoder
        self.fc1 = nn.Linear(128, 27)
        self.fc2 = nn.Linear(128, 27)

        # gaussian decoder
        self.fc3 = nn.Linear(27, 128)
        self.fc4 = nn.Linear(27, 128)
        # gaussian decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 100, kernel_size=13, stride=4, padding=7),
            torch.nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.ConvTranspose1d(100, 50, kernel_size=10, stride=4),
            torch.nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.ConvTranspose1d(50, 20, kernel_size=8, stride=2),
            torch.nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.ConvTranspose1d(20, 1, kernel_size=5, stride=1),
            nn.ReLU())

        print(self.encoder)
        print('size_encoder:', len(self.encoder))
        print(self.decoder)
        print('size_decoder:', len(self.decoder))

    def encode(self, signal):
        x = self.encoder(signal)
        x = x.view(-1, BATCH_SIZE * SIZE_IO * 1)
        fc1_x = self.fc1(x)
        mu_z = functional.relu(self.fc2(fc1_x))
        log_var_z = functional.relu(self.fc2(fc1_x))
        return mu_z, log_var_z

    def decode(self, z):
        x = functional.relu(self.fc3(z))
        x = functional.relu(self.fc4(x))
        decoded_x = self.decoder(x)
        mu_x = functional.relu(decoded_x)
        mu_x = torch.tanh(self.fc4(mu_x))
        log_x = functional.relu(decoded_x)
        log_x = torch.tanh(self.fc4(log_x))
        return mu_x.view(-1, SIZE_IO), log_x.view(-1, SIZE_IO)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        print('size mu:', mu.size(), 'size log_var:', log_var.size())
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

