# -------------------------------------------------------------------------------
# Titre : vae_pytorch.py
# Projet : Granolar
# Description : get the arguments, load the dataset and compile the main code
# -------------------------------------------------------------------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os
import torch
import matplotlib.pyplot as plt
import src.loader as loader

from src.vae import VAE

sample_rate = 44100
data_dir = 'database/raw'

# get the arguments, if not on command line, the arguments are the default
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
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

    train_loader, test_loader = loader.get_data_loaders(data_dir, batch_size=args.batch_size, sr=sample_rate)
    vae = VAE(train_loader, test_loader, batch_size=args.batch_size, seed=args.seed, cuda=cuda)
    losses = []
    for epoch in range(args.epochs):
        loss = vae.train()
        losses.append(loss)
        test_loss = vae.test()
        sample = vae.create_sample()
    plt.plot(losses)
    plt.show()
    print("ok")
