#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import torch.utils.data
from torch import optim
from utils import parse_args, load_MNIST_datasets
from gaussian_vae import VAE_gauss
from gauss_train_test import train, test


args = parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

# device is gpu if possible
device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader, test_loader = load_MNIST_datasets(args, kwargs)

# send the model to device
model = VAE_gauss().to(device)

# set the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# main code
if __name__ == "__main__":
    if not os.path.exists('results/'):
        os.makedirs('results')
    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, device, optimizer, args)
        test(epoch, model, test_loader, device, args)
        """
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
        """
