# -------------------------------------------------------------------------------
# Titre : vae_model.py
# Projet : Granolar
# Description : modeling of the VAE
# -------------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from torch.nn import functional

import math

class VAE_Model(nn.Module):
    def __init__(self, batch_size, channel, grain_size):
        super(VAE_Model, self).__init__()
        self.BATCH_SIZE = batch_size
        self.CHANNEL = channel
        self.GRAIN_SIZE = grain_size
        # encode layers
        channels = [1, 64, 32, 16, 1]
        kernel_sizes = [5, 8, 10, 13]
        strides = [1, 2, 4, 4]
        self.encoder = nn.Sequential()
        l_out = grain_size
        for i, (c_in, c_out, kernel, stride) in enumerate(zip(channels[:-1], channels[1:], kernel_sizes, strides)):
            padding = math.ceil(kernel/2)
            self.encoder.add_module("enc_conv_"+str(i), nn.Conv1d(c_in, c_out, kernel_size=kernel, stride=stride, padding=padding))
            self.encoder.add_module("enc_norm_"+str(i), nn.BatchNorm1d(c_out))
            self.encoder.add_module("enc_relu_"+str(i), nn.ReLU())
            l_out = (l_out + 2 * padding - (kernel - 1) - 1) // stride + 1
            print('l_encode', l_out)
        # self.encoder = nn.Sequential(
        #     nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=3),
        #     torch.nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 32, kernel_size=8, stride=2, padding=5),
        #     torch.nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Conv1d(32, 16, kernel_size=10, stride=4, padding=5),
        #     torch.nn.BatchNorm1d(16),
        #     nn.ReLU(),
        #     nn.Conv1d(16, 8, kernel_size=13, stride=4, padding=7),
        #     torch.nn.BatchNorm1d(8),
        #     nn.ReLU())
        # gaussian encoder
        self.fc1 = nn.Linear(batch_size, l_out)
        # Always have same dimensions
        self.fc21 = nn.Linear(l_out, l_out)
        self.fc22 = nn.Linear(l_out, l_out)

        # gaussian decoder
        self.fc3 = nn.Linear(l_out, batch_size)
        self.fc4 = nn.Linear(batch_size, batch_size)
        # gaussian decoder layers
        channels = channels[::-1]
        kernel_sizes = kernel_sizes[::-1]
        strides = strides[::-1]
        self.decoder = nn.Sequential()
        for i, (c_in, c_out, kernel, stride) in enumerate(zip(channels[:-1], channels[1:], kernel_sizes, strides)):
            padding = math.ceil(kernel / 2)
            self.decoder.add_module("dec_conv_" + str(i),
                                    nn.ConvTranspose1d(c_in, c_out, kernel_size=kernel, stride=stride, padding=padding))
            self.decoder.add_module("dec_norm_" + str(i), nn.BatchNorm1d(c_out))
            self.decoder.add_module("dec_relu_" + str(i), nn.ReLU())
            l_out = (l_out - 1) * stride - 2 * padding + kernel
            print('l decode', l_out)

        # gaussian decoder layers
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose1d(1, 128, kernel_size=13, stride=4, padding=7),
        #     torch.nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(128, 50, kernel_size=10, stride=4),
        #     torch.nn.BatchNorm1d(50),
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(50, 20, kernel_size=8, stride=2),
        #     torch.nn.BatchNorm1d(20),
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(20, 1, kernel_size=5, stride=1),
        #     nn.ReLU())

        print(self.encoder)
        print('size_encoder:', len(self.encoder))
        print(self.decoder)
        print('size_decoder:', len(self.decoder))

    def encode(self, signal):
        x = self.encoder(signal)
        x = x.view(-1, self.BATCH_SIZE * self.CHANNEL * 1)
        fc1_x = functional.relu(self.fc1(x))
        return self.fc21(fc1_x), self.fc22(fc1_x)

    def decode(self, z):
        x = functional.relu(self.fc3(z))
        x = functional.relu(self.fc4(x))
        decoded_x = self.decoder(x.unsqueeze(1))
        mu, log_var = self.decoder(x)
        print('size mu_recon:', mu.size, 'size log_var_recon', log_var.size)
        return decoded_x, mu, log_var

    def reparameterize(self, mu_z, log_var_z):
        std = torch.exp(0.5 * log_var_z)
        eps = torch.randn_like(std)
        z = mu_z + eps * std
        return z

    def forward(self, x):
        mu_z, log_var_z = self.encode(x)
        print('size mu:', mu_z.size(), 'size log_var:', log_var_z.size())
        z = self.reparameterize(mu_z, log_var_z)
        mu_recon, log_var_recon = self.decode(z.view(1, -1))
        return mu_z, log_var_z, mu_recon, log_var_recon
