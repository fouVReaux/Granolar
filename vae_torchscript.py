# -------------------------------------------------------------------------------
# Title : vae_torchscript.py
# Project : Granolar
# Description : get the vae and torchscript working well together
# Author : Alice Rixte
# -------------------------------------------------------------------------------

# !/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import torch
from vae_model import VAE_Model

BATCH_SIZE = 62
GRAIN_SIZE = 512
CHANNEL = 1


if __name__ == "__main__":

    device = torch.device("cpu")
    model = VAE_Model(BATCH_SIZE, CHANNEL, GRAIN_SIZE).to(device)
    trained_vae = torch.load('./saved_model/data.pth')
    model.load_state_dict(trained_vae['model_state_dict'])

  

