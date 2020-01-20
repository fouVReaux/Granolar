# Granolar

Granolar is a machine learning project that reconstruct grains of sound, it is executable with Python3.

## Installation

See the requirements folder for the compulsory libraries

```bash
pip3 install -r requirements.txt
```

## To train the model, run

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

\- pyo_script_osc2.py: granular synthesis of the VAE decoder's output, controlled by OSC

