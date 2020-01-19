# -------------------------------------------------------------------------------
# Title : loss_train_test.py
# Project : Granolar
# Description : compute the loss, train, test and reconstruct the database
# Author : Ninon Devis
# -------------------------------------------------------------------------------

import torch
from torch import optim
import numpy as np
import os
from torch.nn import functional as F


from vae_model import VAE_Model

GRAIN_SIZE = 512
CHANNEL = 1
LEARNING_RATE = 1e-3
beta = 4  # value for train testing


def loss_function(x, mu_z, log_var_z, mu_recon, log_var_recon):
    """
    Computation of the global loss divided between the latent loss and the generative loss
    :param x: data
    :param mu_z: mean of the latent space
    :param log_var_z: standard deviation of the latent space
    :param mu_recon: mean of the reconstruction
    :param log_var_recon: reconstruction's standard deviation
    :return: global loss
    """
    # reconstruction loss: p(z | x) = - log(sigma) - 0.5 * log(2*pi) - (x - mu)^2 / 2 * sigma ^ 2
    recon_loss = torch.sum(log_var_recon - 0.5 * np.log(2 * np.pi)
                           + ((x.view(-1, GRAIN_SIZE) - mu_recon).pow(2)) / (2 * torch.exp(log_var_recon).pow(2))).cuda()
    # recon_loss = F.nn.MSELoss()

    # latent loss: Kullback-Leibler divergence
    KLD = (-0.5 * torch.sum(1 + log_var_z - mu_z.pow(2) - log_var_z.exp())).cuda()
    return recon_loss + beta * KLD


class VAE:
    # device is gpu if possible
    def __init__(self, train_loader, test_loader, batch_size=60, seed=1, cuda=False, output_path='./saved_model/data.pth'):
        """
        Initialization of inner class' attributes
        :param train_loader: sets the train loader
        :param test_loader: sets the test loader
        :param batch_size: initialises the batch_size
        :param seed: sets the random seed
        :param cuda: sends the computation tu GPU if possible
        :param output_path: defines the path for the model saving
        """
        channel = 1
        torch.manual_seed(seed)
        # device is gpu if possible
        self.device = torch.device("cuda" if cuda else "cpu")
        # send the model to device
        self.model = VAE_Model(batch_size, channel, GRAIN_SIZE).to(self.device)
        # set the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.batch_size = batch_size
        self.epoch = 0
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss = self.train()
        self.path = output_path

    def run(self, data):
        data.to(self.device)
        self.optimizer.zero_grad()
        return self.model(data)


    def train(self, log_interval=10):
        """
        Trains the model every epochs by calculating the loss
        :param log_interval: needed just to print every interval declared
        :return: the new loss computed from the training
        """
        self.model.train()
        self.epoch += 1
        train_loss = 0
        datas = []
        recons = []
        # for each batch
        for batch_idx, data in enumerate(self.train_loader):
            print("DATA_LEN:", len(data))
            print("DATA:", data)
            datas.append(data)
            data_recon, mu_z, log_var_z, mu_recon, log_var_recon = self.run(data)
            recons.append(data_recon)
            # compute the loss function
            loss = loss_function(data, mu_z, log_var_z, mu_recon, log_var_recon).cuda()
            # compute loss' gradient for every parameter
            loss.backward()
            # increment the average loss for each sample
            train_loss += loss.item()
            print('train_loss', train_loss)
            self.optimizer.step()
            # print
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, batch_idx * len(data), len(self.train_loader.dataset),
                                100. * batch_idx / len(self.train_loader),
                                loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            self.epoch, train_loss / len(self.train_loader.dataset)))

        # # Uncomment if need to print model's state_dict
        # print("Model's state_dict:")
        # for param_tensor in self.model.state_dict():
        #     print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

        return train_loss, datas, recons

    def test(self, log_interval=10):
        """
        Tests the model every epochs by calculating the loss
        :param log_interval: needed just to print every interval declared
        :return: the new loss computed from the testing
        """
        self.model.eval()
        self.epoch += 1
        test_loss = 0
        comparison = None
        with torch.no_grad():
            # for i, (data, _) in enumerate(self.test_loader):
            for batch_idx, data in enumerate(self.test_loader):
                # get the variables
                data_recon, mu_z, log_var_z, mu_recon, log_var_recon = self.run(data)
                # compute the loss function
                test_loss = loss_function(data_recon, mu_z, log_var_z, mu_recon, log_var_recon).cuda()
                # increment the average loss for each sample
                test_loss += test_loss.item()
                self.optimizer.step()
                # print
                if batch_idx % log_interval == 0:
                    print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss test: {:.6f}'.format(
                        self.epoch, batch_idx * len(data), len(self.test_loader.dataset),
                                    100. * batch_idx / len(self.test_loader), test_loss.item() / len(data)))
        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(
            self.epoch, test_loss / len(self.test_loader.dataset)))
        return test_loss

    def save_training(self):
        """
        Saves the model trained into the files 'saved_model'
        """
        print('[DEBUG] Saving training .. ')
        if not os.path.exists('./saved_model'):
            print("[DEBUG] Creating saving path: '{}'.. ".format(self.path))
            os.makedirs('./saved_model')

        try:
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.loss,
            }, self.path)
        except:
            print('[ERROR] Something happened while saving data set :( ... ')
            print('[ERROR] Data set HAS NOT been saved ! ')

    def resume_training(self):
        """
        Restores the model when it is trained
        """
        print('[DEBUG] Restoring training .. ')
        model = self.model
        optimizer = self.optimizer
        try:
            checkpoint = torch.load(self.path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.loss = checkpoint['loss']
            model.train()
        except:
            print('[ERROR] Something bad happened while restoring data set :( ... ')
            print('[ERROR] Data set HAS NOT been restored ! ')

    def create_sample(self):
        """
        Rebuilds directly sample from the latent space
        :return: new samples with same attributes as inputs
        """
        with torch.no_grad():
            latent_sample = torch.randn(self.model.batch_size, self.model.latent_size).to(self.device).cuda()
            sample = self.model.decode(latent_sample)
            return sample



