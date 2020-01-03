from os import walk
from os.path import join
from natsort import natsorted
import librosa 
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

def Grain_DatasetLoader(path, sr=22050, l_grain=2**11, max_files=None):
    grains = []
    n_files = 0
    for root, dirs, files in walk(path, topdown=False):
            file_names = natsorted([join(root, file_name) for file_name in files if not file_name.startswith('.')])
            for file_name in file_names:
                (seq, sr) = librosa.load(file_name, sr=sr, mono=True, duration = 4)
                (seq_t, index) = librosa.effects.trim(seq)
                n_gr = int(seq_t.size/l_grain)
                grains.append(torch.from_numpy(seq_t).type(torch.float)[:(n_gr*l_grain)].view(n_gr,l_grain))
                n_files += 1
                if max_files is not None and n_files==max_files:
                    break
    grains = torch.cat(grains)
    dataset = torch.utils.data.TensorDataset(grains)

    return dataset


data_dir = 'NINON/TECHNOOOOO'
output_dir = 'OUTPUT'
sample_rate = 16000
l_grain = 512

dataset = Grain_DatasetLoader(data_dir, sr=sample_rate, l_grain=l_grain, max_files=None)



batch_size = 4
test_split = .2
shuffle_dataset = True
random_seed= 42
        
dataset_size = len(dataset)
indices = list(range(dataset_size))

split = int(np.floor(test_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler, drop_last=True, num_workers=3)#, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=test_sampler, drop_last=True, num_workers=3)#, pin_memory=True)

np.save(output_dir + '_' + 'train_indices', train_indices)
np.save(output_dir + '_' + 'test_indices', val_indices)

i = 0
n = 10
for grains in train_loader:
    #Plotting the n_th batch :-)
    if i == n:
        for j in range(batch_size):
            grain = grains[0].float()
            plt.figure()
            plt.plot(grain[j,:].numpy())
    i = i+1