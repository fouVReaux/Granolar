# -------------------------------------------------------------------------------
# Title : comparision.py
# Project : Granolar
# Description : plot some grains and the reconstruction of those grains in order to check the training
# Author : Ninon Devis
# -------------------------------------------------------------------------------

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os
import torch
import matplotlib.pyplot as plt
import loader as loader
import vae_model

from loss_train_test import VAE

sample_rate = 44100
data_dir = 'database/raw'

# get the arguments, if not on command line, the arguments are the default
parser = argparse.ArgumentParser(description='Granular VAE')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

cuda = not args.no_cuda and torch.cuda.is_available()

# main code
if __name__ == "__main__":
    if not os.path.exists('../results/'):
        os.makedirs('../results')

    # Get the database for train and test
    train_loader, test_loader = loader.get_data_loaders(data_dir, batch_size=args.batch_size, sr=sample_rate)
    vae = VAE(train_loader, test_loader, batch_size=args.batch_size, seed=args.seed, cuda=cuda)
    losses = []
    datas = None
    recons = None
    # Compute and store the loss, create a sample every epochs
    for epoch in range(args.epochs):
        loss, datas, recons = vae.train()
        losses.append(loss)
        test_loss = vae.test()
        sample = vae.create_sample()
        # plot the grains and reconstruction of the grains (a modifier)
        fig, (input_plot, reconstruction_plot) = plt.subplots(1, 2)
        plt.suptitle('comparison')
        input_plot.plot(datas[1][0][0].detach().numpy())
        input_plot.set_title('Input')
        reconstruction_plot.plot(recons[1][0][0].detach().numpy())
        reconstruction_plot.set_title('Reconstruction')
        plt.show()

# sound reconstruction using librosa

    # Plot the loss evolution
    plt.plot(losses)
    plt.show()
    print("genius")



# # plot the latent space in 3D: to achieve, maybe use correlation matrix of latent spaces??
#     z = vae_model.encoded(datas)
#     plt.figure(figsize=(6, 6))
#     plt.scatter(z[:, 0],z[:, 1], c=np.argmax(y_sample, 1))
#     plt.colorbar()
#     plt.grid()
#
# x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
# plt.figure(figsize=(6, 6))
# plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
# plt.colorbar()
# plt.show()
#
# # sound reconstruction using librosa
#
#     # Plot the loss evolution
#     plt.plot(losses)
#     plt.show()
#     print("genius")
