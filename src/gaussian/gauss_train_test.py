import torch
from torchvision.utils import save_image
from gaussian_vae import loss_function


def train(n_epoch, model, train_loader, device, optimizer, args):
    model.train()
    train_loss = 0
    # for each batch
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        # get the variables
        mu_z, logvar_z, mu, logvar = model(data)

        # define the loss function
        loss = loss_function(data, mu_z, logvar_z, mu, logvar, args.beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # affichage
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                n_epoch, batch_idx * len(data), len(train_loader.dataset),
                         100. * batch_idx / len(train_loader),
                         loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        n_epoch, train_loss / len(train_loader.dataset)))


def test(n_epoch, model, test_loader, device, args):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            mu_z, logvar_z, mu, logvar = model(data)
            test_loss += loss_function(data, mu_z, logvar_z, mu, logvar, args.beta).item()
            # affichage
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        mu_z.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(n_epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))