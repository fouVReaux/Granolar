import torch
from torch import optim
from torchvision.utils import save_image

from src.vae_model import VAE_Model

# Reconstruction + KL divergence losses summed over all elements and batch
beta = 0.5


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


def loss_function(recon_x, x, mu, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + beta * KLD


class VAE:
    # device is gpu if possible
    def __init__(self, train_dataset, test_dataset, batch_size=128, seed=1, no_cuda=False):
        cuda = not no_cuda and torch.cuda.is_available()
        torch.manual_seed(seed)
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
        # device is gpu if possible
        self.device = torch.device("cuda" if cuda else "cpu")
        # send the model to device
        self.model = VAE_Model().to(self.device)
        # set the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.batch_size = batch_size
        self.epoch = 0
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **self.kwargs)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **self.kwargs)

    def train(self, log_interval=10):
        self.model.train()
        self.epoch += 1
        train_loss = 0
        # for each batch
        for batch_idx, data in enumerate(self.train_loader):
            print("ELEM_LEN:", len(data[3]))
            print("DATA_LEN:", len(data))
            print("DATA:", data)
            data = data.to(self.device)
            self.optimizer.zero_grad()
            # get the variables
            recon_batch, mu, log_var = self.model(data)
            # define the loss function
            loss = loss_function(recon_batch, data, mu, log_var)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            # print
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              self.epoch, train_loss / len(self.train_loader.dataset)))

    def test(self):
        self.model.eval()
        test_loss = 0
        comparison = None
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.to(self.device)
                recon_batch, mu, log_var = self.model(data)
                test_loss += loss_function(recon_batch, data, mu, log_var).item()
                # affichage
                # if i == 0:
                #     n = min(data.size(0), 8)
                #     comparison = torch.cat([data[:n],
                #                           recon_batch.view(self.batch_size, 1, 28, 28)[:n]])
                #     save_image(comparison.cpu(),
                #              'results/reconstruction_' + str(self.epoch) + '.png', nrow=n)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss

    def create_sample(self):
        with torch.no_grad():
            sample = torch.randn(64, 20).to(self.device)
            sample = self.model.decode(sample).cpu()
            return sample

