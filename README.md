# Granolar

Granolar is a machine learning project that reconstructs grains of sound, it is executable with Python3.

## Installation

See the requirements folder for the compulsory libraries

```bash
pip3 install -r requirements.txt
```

## To train the model

Run

```bash
python3 vae_main.py
```

## To load the trained model and start the synth
Run one of the following scripts:

\- recover_model.py

\- recover_model_batch.py

You need to have pyo installed, and also wxpython to have the pyo gui. Just run:

```bash
pip3 install pyo
pip3 install wxpython
```
The file simple_OSC_control.pd is a Pure Data patch for sending simple OSC messages to control the synth. To control recover_model.py you need to send one element tuples of floats in the range (0,1). To control recover_model_batch.py you need to send tuples of 16 floats in the range of (0,1).

Attention: Depending on the operating system and how the scripts are run, pyo could not work at all or crash. Running pyo on Windows doesn't work most of the time, depending on the computer. When running from an IDE you are advised to turn off the pyo GUI, as it can crash when you close it.

## Content

this archive contains:

\- database: folder with the database that will be use for the training

\- mnist_vae: folder with the VAE apply to the mnist database

\- references: folder with the documents of reference

\- saved_model: folder containing the trained model of VAE

\- trials_scripts: folder containing all the failed scripts

\- loader.py: loader of the database

\- loss_train_test.py: compute the loss, train, test and save the model

\- vae_main.py: main project script

\- vae_model.py: architecture of the model

\- comparison.py: plot inputs and outputs, concatenate the output and convert it in .wav

\- recover_model.py , recover_model_batch.py: granular synthesis of the VAE decoder's output, controlled by OSC

\- simple_OSC_control.pd: Pure Data OSC controller for the synth

