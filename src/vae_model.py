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
        self.batch_size = batch_size
        self.channel = channel
        self.grain_size = grain_size
        # encode layers
        channels = [1, 64, 32, 16, 1]
        kernel_sizes = [5, 8, 10, 13]
        strides = [1, 2, 4, 4]
        paddings = [3, 4, 5, 7]
        self.encoder = nn.Sequential()
        l_out = grain_size
        for i, (c_in, c_out, kernel, stride, padding) in enumerate(zip(channels[:-1], channels[1:], kernel_sizes, strides, paddings)):
            # padding = math.ceil(kernel/2)
            self.encoder.add_module("enc_conv_"+str(i), nn.Conv1d(c_in, c_out, kernel_size=kernel, stride=stride, padding=padding))
            self.encoder.add_module("enc_norm_"+str(i), nn.BatchNorm1d(c_out))
            self.encoder.add_module("enc_relu_"+str(i), nn.ReLU())
            l_out = (l_out + 2 * padding - (kernel - 1) - 1) // stride + 1
            print('l_encode', l_out)
        # gaussian encoder
        self.latent_size = l_out
        self.encoder_fc = nn.Linear(batch_size, l_out)
        # Always have same dimensions
        self.enc_mu = nn.Linear(l_out, l_out)
        self.enc_log_var = nn.Linear(l_out, l_out)

        # gaussian decoder
        self.decoder_fc = nn.Linear(l_out, batch_size)
        # gaussian decoder layers
        channels = channels[::-1]
        kernel_sizes = kernel_sizes[::-1]
        strides = strides[::-1]
        paddings = [6, 5, 4, 1]
        self.decoder = nn.Sequential()
        for i, (c_in, c_out, kernel, stride, padding) in enumerate(zip(channels[:-1], channels[1:], kernel_sizes, strides, paddings)):
            # padding = math.ceil(kernel / 2)
            self.decoder.add_module("dec_conv_" + str(i),
                                    nn.ConvTranspose1d(c_in, c_out, kernel_size=kernel, stride=stride, padding=padding))
            self.decoder.add_module("dec_norm_" + str(i), nn.BatchNorm1d(c_out))
            self.decoder.add_module("dec_relu_" + str(i), nn.ReLU())
            l_out = (l_out - 1) * stride - 2 * padding + kernel
            print('l decode', l_out)
        self.decoder.add_module("dec_tanh", nn.Tanh())
        self.dec_mu = nn.Linear(batch_size, batch_size)
        self.dec_log_var = nn.Linear(batch_size, batch_size)

        print(self.encoder)
        print('size_encoder:', len(self.encoder))
        print(self.decoder)
        print('size_decoder:', len(self.decoder))

    def encode(self, data):
        x = self.encoder(data)
        x = x.view(-1, self.batch_size * self.channel * 1)
        fc_x = functional.relu(self.encoder_fc(x))
        return self.enc_mu(fc_x), self.enc_log_var(fc_x)

    def decode(self, z):
        fc_z = self.decoder_fc(z)
        x = functional.relu(fc_z)
        data_recon = self.decoder(x.unsqueeze(1))
        mu_recon = self.dec_mu(data_recon)
        log_var_recon = self.dec_log_var(data_recon)
        print('size mu_recon:', mu_recon.size, 'size log_var_recon', log_var_recon.size)
        return data_recon, mu_recon, log_var_recon

    def reparameterize(self, mu_z, log_var_z):
        std = torch.exp(0.5 * log_var_z)
        eps = torch.randn_like(std)
        z = mu_z + eps * std
        return z

    def forward(self, data):
        mu_z, log_var_z = self.encode(data)
        print('size mu:', mu_z.size(), 'size log_var:', log_var_z.size())
        z = self.reparameterize(mu_z, log_var_z)
        data_recon, mu_recon, log_var_recon = self.decode(z.view(289, 1))
        return data_recon, mu_z, log_var_z, mu_recon, log_var_recon
