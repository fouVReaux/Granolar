#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First recover the saved model, then run the granular script
@author: mano
"""

from __future__ import print_function
import argparse
import os
import torch
import matplotlib.pyplot as plt
import loader as loader

from loss_train_test import VAE

sample_rate = 44100
data_dir = 'database/raw'

# get the arguments, if not on command line, the arguments are the default
parser = argparse.ArgumentParser(description='Granular VAE')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

train_loader, test_loader = loader.get_data_loaders(data_dir, batch_size=args.batch_size, sr=sample_rate)
vae = VAE(train_loader, test_loader, batch_size=args.batch_size, seed=args.seed,cuda=False)
vae.resume_training()  


from pyo import *
import numpy as np
#Port to receive OSC messages.
port=9001

#a=torch.empty(vae.model.latent_size)
#a.fill_(.5)
#a=a.repeat(vae.model.batch_size,1)

#Initialize the first sampling point at all coordinates = .5
a=torch.empty(vae.model.batch_size,vae.model.latent_size)
a.fill_(.5)
sample=vae.model.decode(a)
grain=sample[0].flatten().detach().numpy()

#Initialize the pyo audio server.
s=Server().boot()

#Load the grain on a pyo Table and loop it.
snd=DataTable(size=grain.shape[0],init=grain.tolist())
snd_array=np.asarray(snd.getBuffer())

#Choice of table to smooth the grains
env = HannTable()
#pos = Phasor(snd.getRate()*.25, 0, snd.getSize())
#dur = Noise(.001, 1)

#Granulator object allows the control over the grains
g = Granulator(snd, env, .5, 0, 1, mul=.1).out()
g.ctrl()
def changeCoords(address,*args):
    if address=="/coord":
        print ("Coords are:",args)
        a.fill_(args[0])
        sample=vae.model.decode(a)
        grain=sample[1].flatten().detach().numpy()
        snd.replace(grain.tolist())

scan = OscDataReceive(port=port, address="*", function=changeCoords)



#Start the audio.
s.start()

#use s.stop() to stop the audio, s.shutdown() to terminate the server
#s.gui(locals())