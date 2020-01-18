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
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import loader as loader
import librosa
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
    concat = None
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
        for recon in recons:
            for i in range(recon.size()[0]):
                new_rec = recon[i][0].detach().numpy()
                if concat is None:
                    concat = new_rec
                else:
                    concat = np.concatenate((concat, new_rec))
            print('concat', concat)
            librosa.output.write_wav('./results/reconstruction.wav', concat, sample_rate, norm=False)

    # Plot the loss evolution
    plt.plot(losses)
    plt.show()
    print("genius")

