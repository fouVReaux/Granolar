# -------------------------------------------------------------------------------
# Title : vae_pytorch.py
# Project : Granolar
# Description : Working VAE on MNIST Database
# Author : Ninon Devis
# -------------------------------------------------------------------------------

import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import trials.vae_lowlevel_MNIST.input_data as input_data

mnist = input_data.read_data_sets('../MNIST_Handwritten_Digit_Recognition/MNIST_Dataset', one_hot=True)
batch_size = 64
X_dim = mnist.train.images.shape[1]
h_dim = 128
Z_dim = 100
Y_dim = X_dim
c = 0
lr = 1e-3


def init_weights(size):
    # default implementation :
    # return torch.zeros(*size, requires_grad=True)
    # Xavier !
    in_dim = size[0]
    stddev = 1. / np.sqrt(in_dim / 2.)
    W = torch.mul(torch.randn(*size), stddev)
    W.requires_grad = True
    return W


def init_bias(dim):
    return torch.zeros(dim, requires_grad=True)


# =============================== Q(z|X) ======================================

Wxh_Q = init_weights(size=[X_dim, h_dim])
bh_Q = init_bias(h_dim)

Whz_mu = init_weights(size=[h_dim, Z_dim])
bhz_mu = init_bias(Z_dim)

Whz_logsigma = init_weights(size=[h_dim, Z_dim])
bz_logsigma = init_bias(Z_dim)


def Q(X):
    h = nn.relu(X @ Wxh_Q + bh_Q.repeat(X.size(0), 1))
    z_mu = h @ Whz_mu + bhz_mu.repeat(h.size(0), 1)
    z_logsigma = h @ Whz_logsigma + bz_logsigma.repeat(h.size(0), 1)
    return z_mu, z_logsigma


def sample_z(mu, log_var):
    eps = torch.randn(batch_size, Z_dim)
    return mu + torch.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

Wzh_P = init_weights(size=[Z_dim, h_dim])
bh_P = init_bias(h_dim)

Why = init_weights(size=[h_dim, Y_dim])
by = init_bias(Y_dim)


def P(z):
    h = nn.relu(z @ Wzh_P + bh_P.repeat(z.size(0), 1))
    return nn.sigmoid(h @ Why + by.repeat(h.size(0), 1))


# =============================== TRAINING ====================================

params = [Wxh_Q, bh_Q, Whz_mu, bhz_mu, Whz_logsigma, bz_logsigma,
          Wzh_P, bh_P, Why, by]

solver = optim.Adam(params, lr=lr)

for it in range(100000):
    X, _ = mnist.train.next_batch(batch_size)
    X = torch.from_numpy(X)

    # Forward
    z_mu, z_logsigma = Q(X)
    z = sample_z(z_mu, z_logsigma)
    Y = P(z)

    # Loss
    recon_loss = nn.binary_cross_entropy(Y, X, size_average=False) / batch_size
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_logsigma) + z_mu**2 - 1. - z_logsigma, 1))
    loss = recon_loss + kl_loss

    # Backward
    loss.backward()

    # Update
    solver.step()
    solver.zero_grad()

    # Print and plot sometimes
    if it % 1000 == 0:
        print('Iter-{}; Loss: {:.4}'.format(it, loss.item()))

        samples = P(z).data.numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('results/'):
            os.makedirs('results/')

        plt.savefig('out/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
        c += 1
        plt.close(fig)
