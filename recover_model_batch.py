#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loads grains from a linear trajectory starting at the zero point,
with the end point given by a 16-dimensional tuple by OSC.
All grains are loaded in a pyo table, which is updated each time a new point is chosen (attention! replacement takes considerable time, depending on the number of points chosen)
Minimum number of points is equal to the batch size of the model.
The code first recovers the trained model, so be sure to run vae_main.py first to save the model.

Besides the choice of linear trajectory, the user can control the granulator properties like pitch, number of grains etc.
@author: mano
"""
#The first part of the code loads the trained model.
from __future__ import print_function
import argparse
import os
import torch
import matplotlib.pyplot as plt
import loader as loader

from loss_train_test import VAE

sample_rate = 44100
data_dir = 'database/raw'

# Get the arguments, if not on command line, the arguments are the default
parser = argparse.ArgumentParser(description='Granular VAE')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--epochs conda install -c conda-forge wxpython ', type=int, default=200, metavar='N',
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
vae.model.eval()

#In this part a trajectory is initialized and the server is started.
from pyo import *
import numpy as np
#Port to receive OSC messages.
port=9001
#How long will the table be (samples *batch_size*512)
table_size=50 
#a=torch.empty(vae.model.latent_size)
#a.fill_(.5)
#a=a.repeat(vae.model.batch_size,1)

#Initialize the trajectory as a line from the zero point to the one with coordinates 0.5 in all dimensions
args=tuple(np.ones(16)-0.5)
a=torch.ones(vae.model.batch_size*table_size,vae.model.latent_size)
for i in np.arange(vae.model.batch_size*table_size):
    a[i]=torch.from_numpy(np.asarray(args))/vae.model.batch_size *i
sample=torch.empty(table_size,512*vae.model.batch_size)
for i in np.arange(table_size):
    sample[i]=vae.model.decode(a[i:i+vae.model.batch_size])[1].detach().flatten()
grain=sample.flatten().detach().numpy()

#Initialize the pyo audio server.
s=Server().boot()

#Load the grain on a pyo Table and loop it.
snd=DataTable(size=grain.shape[0],init=grain.tolist())

snd.view()

#Choice of table to smooth the grains
env = HannTable()
pos = Phasor(snd.getRate()*.25, 0, snd.getSize())
dur = Noise(.001, .1)

#Granulator object allows having control over the grains
g = Granulator(snd, env, 1, pos, dur, mul=.1).out()
g.ctrl()

#Function to be called whenever a new OSC message is received

def changeCoords(address,*args):
    if address=="/coord":
        print ("Coords are:",args)
        a=torch.ones(vae.model.batch_size*table_size,vae.model.latent_size)
        for i in np.arange(vae.model.batch_size*table_size):
            a[i]=torch.from_numpy(np.asarray(args))/vae.model.batch_size *i
        sample=torch.empty(table_size,512*vae.model.batch_size)
        for i in np.arange(table_size):
            sample[i]=vae.model.decode(a[i:i+vae.model.batch_size])[1].detach().flatten()
        grain=sample.flatten().detach().numpy()
        snd.replace(grain.tolist())
#Scans for new OSC messages in the specified port and calls the above function.

scan = OscDataReceive(port=port, address="*", function=changeCoords)



#Start the audio.
s.start()

#use s.stop() to stop the audio, s.shutdown() to terminate the server

#Uncomment the next line to show the Granulator gui (attention! it tends to crash when quitting, when run through an IDE)
s.gui(locals())