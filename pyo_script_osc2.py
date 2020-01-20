#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for granular synthesis of the VAE decoder's output, controlled by OSC.
The model needs to be in memory, with the name vae.model
Takes OSC messages of type float with address /coord at port 9001

@author: mano
"""

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
g = Granulator(snd, env, 1, 0, 1, mul=.1).out()

def changeCoords(address,*args):
    if address=="/coord":
        print ("Coords are:",args)
        a.fill_(args[0])
        sample=vae.model.decode(a)
        grain=sample[0].flatten().detach().numpy()
        snd.replace(grain.tolist())

scan = OscDataReceive(port=port, address="*", function=changeCoords)



#Start the audio.
s.start()

#use s.stop() to stop the audio, s.shutdown() to terminate the server
