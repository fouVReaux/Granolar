# -------------------------------------------------------------------------------
# Title : vae_model.py
# Project : Granolar
# Description : modeling of the VAE
# Author : Ninon Devis
# -------------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F


class VAE_Model(nn.Module):
    def __init__(self, batch_size, channel, grain_size):
        """
        Initialization of inner class' attributes
        :param batch_size: initialises the batch_size
        :param channel: sets the number  of channels
        :param grain_size: sets the size of the sound grain
        """
        super(VAE_Model, self).__init__()
        self.batch_size = batch_size
        self.channel = channel
        self.grain_size = grain_size
        # encode layers
        channels = [1, 64, 32, 16, 1]
        kernel_sizes = [5, 8, 10, 13]
        strides = [1, 2, 4, 4]
        paddings = [3, 4, 5, 7]
        # encoder built as a sequential
        self.encoder = nn.Sequential().cuda()
        self.input_pad = 16
        l_out = self.grain_size
        for i, (c_in, c_out, kernel, stride, padding) in enumerate(zip(channels[:-1], channels[1:], kernel_sizes, strides, paddings)):
            # padding = math.ceil(kernel/2)
            self.encoder.add_module("enc_conv_"+str(i), nn.Conv1d(c_in, c_out, kernel_size=kernel, stride=stride, padding=padding))
            self.encoder.add_module("enc_norm_"+str(i), nn.BatchNorm1d(c_out))
            self.encoder.add_module("enc_relu_"+str(i), nn.ReLU())
            l_out = (l_out + 2 * padding - (kernel - 1) - 1) // stride + 1
            print('l_encode', l_out)

        self.l_enc = l_out
        # gaussian encoder
        self.latent_size = 16
        self.encoder_fc = nn.Linear(self.l_enc, self.latent_size).cuda()
        # always have same dimensions
        self.enc_mu = nn.Linear(self.latent_size, self.latent_size).cuda()
        self.enc_log_var = nn.Linear(self.latent_size, self.latent_size).cuda()

        # gaussian decoder
        self.decoder_fc = nn.Linear(self.latent_size, self.l_enc).cuda()
        # gaussian decoder layers built as a mirror of the encoder
        channels = channels[::-1]
        kernel_sizes = kernel_sizes[::-1]
        strides = strides[::-1]
        paddings = [6, 5, 4, 1]
        # decoder built as a sequential
        self.decoder = nn.Sequential().cuda()
        for i, (c_in, c_out, kernel, stride, padding) in enumerate(zip(channels[:-1], channels[1:], kernel_sizes, strides, paddings)):
            # padding = math.ceil(kernel / 2)
            self.decoder.add_module("dec_conv_" + str(i),
                                    nn.ConvTranspose1d(c_in, c_out, kernel_size=kernel, stride=stride, padding=padding))
            self.decoder.add_module("dec_norm_" + str(i), nn.BatchNorm1d(c_out))
            self.decoder.add_module("dec_relu_" + str(i), nn.ReLU())
            l_out = (l_out - 1) * stride - 2 * padding + kernel
            print('l decode', l_out)
        self.decoder.add_module("dec_tanh", nn.Tanh())
        self.dec_mu = nn.Linear(self.grain_size, self.grain_size).cuda()
        self.dec_log_var = nn.Linear(self.grain_size, self.grain_size).cuda()

        # print the architecture of the model
        print(self.encoder)
        print('size_encoder:', len(self.encoder))
        print(self.decoder)
        print('size_decoder:', len(self.decoder))

    def encode(self, data):
        """
        Defines the encoding function
        :param data: inputs
        :return: encoded gaussian parameters
        """
        x = self.encoder(data).cuda()
        x = x.view(self.batch_size, self.l_enc).cuda()
        fc_x = F.relu(self.encoder_fc(x)).cuda()

        return self.enc_mu(fc_x), self.enc_log_var(fc_x).cuda()

    def decode(self, z):
        """
        Defines the decoding function
        :param z: variable of the latent space
        :return: reconstructed parameters
        """
        print(z.shape)
        fc_z = self.decoder_fc(z).cuda()
        x = F.relu(fc_z).cuda()
        data_recon = self.decoder(x.unsqueeze(1)).cuda()
        mu_recon = self.dec_mu(data_recon).cuda()
        log_var_recon = self.dec_log_var(data_recon).cuda()

        return data_recon, mu_recon, log_var_recon

    def reparameterize(self, mu_z, log_var_z):
        """
        Turns the gaussian parameters into latent space variable
        :param mu_z: mean of the latent space
        :param log_var_z: standard deviation of the latent space
        :return: latent space variable
        """
        std = torch.exp(0.5 * log_var_z).cuda()
        eps = torch.randn_like(std).cuda()
        z = (mu_z + eps * std).cuda()

        return z

    def forward(self, data):
        """
        Computes output tensors from input tensors
        :param data: input tensors
        :return: output tensors
        """
        mu_z, log_var_z = self.encode(data)
        print('size mu:', mu_z.size(), 'size log_var:', log_var_z.size())
        z = self.reparameterize(mu_z, log_var_z).cuda()
        data_recon, mu_recon, log_var_recon = self.decode(z)

        return data_recon, mu_z, log_var_z, mu_recon, log_var_recon
