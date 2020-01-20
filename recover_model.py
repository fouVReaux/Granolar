#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loads batches of grains in real time to a table to loop and manipulate.

The points chosen have all the same coordinates in all dimensions, 
and are chosen by a float from 0 to 1 sent by OSC.

Running the decoder and updating the table adds a considerable lag between choice of grains, so be careful with the speed of incoming OSC messages.


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
vae.model.eval()


from pyo import *
import numpy as np
#Port to receive OSC messages.
port=9001


#Initialize the first sampling point at all coordinates = .5
a=torch.empty(vae.model.batch_size,vae.model.latent_size)
a.fill_(.5)
sample=vae.model.decode(a)
grain=sample[1].flatten().detach().numpy()

#Initialize the pyo audio server.
s=Server().boot()

#Load the grain on a pyo Table and loop it.
snd=DataTable(size=grain.shape[0],init=grain.tolist())
snd_array=np.asarray(snd.getBuffer())

#Choice of table to smooth the grains
env = HannTable()

#Granulator object allows control over the grains
g = Granulator(snd, env, .5, 0, 1, mul=.1).out()
g.ctrl()

#Function to be called whenever a new OSC message is received
def changeCoords(address,*args):
    if address=="/coord":
        print ("Coords are:",args)
        a.fill_(args[0])
        sample=vae.model.decode(a)
        grain=sample[1].flatten().detach().numpy()
        snd.replace(grain.tolist())
#Scans for new OSC messages in the specified port and calls the above function.
scan = OscDataReceive(port=port, address="*", function=changeCoords)



#Start the audio.
s.start()

#use s.stop() to stop the audio, s.shutdown() to terminate the server
#Comment the next line to hide the GUI when running in an IDE (pyo gui can crash if not)
s.gui(locals())