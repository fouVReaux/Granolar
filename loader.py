# -------------------------------------------------------------------------------
# Title : loader.py
# Project : Granolar
# Description : load the database and create appropriate grains for the VAE
# Author : Constance Douwes & Ninon Devis
# -------------------------------------------------------------------------------

from os import walk
from os.path import join
from natsort import natsorted
import librosa 
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


data_dir = 'database/raw'
output_dir = 'out'
sample_rate = 16000
l_grain = 512

batch_plot = 4
test_split = .2
shuffle_dataset = True
life_seed = 42


class Grains(Dataset):
    def __init__(self, path, sr, batch_size, l_grain, max_files):
        self.grains = []
        n_files = 0
        for root, dirs, files in walk(path, topdown=False):
            file_names = natsorted([join(root, file_name) for file_name in files if not file_name.startswith('.')])
            for file_name in file_names:
                (seq, sr) = librosa.load(file_name, sr=sr, mono=True)  #, duration=4)
                (seq_t, index) = librosa.effects.trim(seq)
                n_gr = int(seq_t.size/l_grain)
                nb_batches = int(n_gr / batch_size)
                for batch in range(32):  # range(nb_batches * n_gr):
                    sub_grains = torch.from_numpy(seq_t).type(torch.float)[batch*l_grain:(batch+1)*l_grain].unsqueeze(0)
                    self.grains.append(sub_grains)
                n_files += 1
                if max_files is not None and n_files == max_files:
                    break

    def __len__(self):
        return len(self.grains)

    def __getitem__(self, index):
        return self.grains[index]


def get_data_loaders(data_dir, batch_size=16, sr=22050, l_grain=512, max_files=None):
    dataset = Grains(data_dir, sr, batch_size, l_grain, max_files)
    dataset_size = len(dataset)
    # Compute indices for train/test split
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(life_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    # Create corresponding subsets
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    # np.save(output_dir + '/' + 'train_indices', train_indices)
    # np.save(output_dir + '/' + 'test_indices', val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler, drop_last=True, num_workers=3)  # pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              sampler=test_sampler, drop_last=True, num_workers=3)  # pin_memory=True)
    return train_loader, test_loader


# main code
if __name__ == "__main__":
    i = 0
    n = 2
    train_loader, test_loader = get_data_loaders(data_dir, sr=sample_rate, l_grain=l_grain)
    for grains in train_loader:
        # Plotting the n_th batch
        if i == n:
            for j in range(batch_plot):
                grain = grains[0].float()
                plt.figure()
                plt.plot(grain[j, :].numpy())
        i += 1
    plt.show()
