#-------------------------------------------------------------------------------
# Titre : DataBase.py
# Projet : Granolar
# Description : Prototype
#-------------------------------------------------------------------------------
import librosa
import numpy as np
import os

class TchouTchou:
    def __init__(self):
        self.train_list=[]
        self.database_path=""
        self.path_list=""
        pass
    # Setter -------------------------------------------------------------------
    def set_database_path(self, path):
        pass
    def set_train_list(self, path):
        pass
    # Getter -------------------------------------------------------------------
    def get_train_list(self):
        pass
    def get_database_path(self, path):
        pass
    # generate -----------------------------------------------------------------
    def generate_train_list(self, path2save):
        pass
    # Train : TchouTchou -------------------------------------------------------
    def train(self):
        pass
if __name__ == "__main__":
    train = TchouTchou()
    train.set_database_path("/path")
    train.generate_train_list("/path")
    train.train()
