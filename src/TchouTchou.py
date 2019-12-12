#-------------------------------------------------------------------------------
# Titre : DataBase.py
# Projet : Granolar
# Description : Prototype
#-------------------------------------------------------------------------------
import librosa
import numpy as np
import csv
import os

class TchouTchou:
    def __init__(self):
        self.train_list=[]
        self.database_path=""
        self.path_list=""
        pass
    # Setter -------------------------------------------------------------------
    def set_database_path(self, path):
        self.database_path = path
        pass
    # Getter -------------------------------------------------------------------
    def get_train_list(self):
        """
        give the train list
        """
        return self.train_list
    def get_database_path(self, path):
        """
        give the DataBase path used
        """
        return self.database_path
    # generate -----------------------------------------------------------------
    def generate_train_list(self):
        """
        Generate a train list with 70% of the sample file present the database_path
        """
        path = self.database_path+"/"
        file_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        n=0
        train_list = [""] * int(len(file_list)*0.7)
        while (n!=int(len(file_list)*0.7)):
            train_list[n]=file_list[np.random.randint(len(file_list),size=1)[0]]#[>(n_list*0.7)]
            n = n+1
        self.train_list = train_list
        return train_list
    # save ---------------------------------------------------------------------
    def save_train_list(self, file_name='train_list.csv'):
        """
        save the train list in a csv file named train_list.csv
        this file is located in the /sample directory
        """
        with open(self.database_path+"/"+file_name, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter='\n',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(self.train_list)
        pass
    # Load ---------------------------------------------------------------------
    def load_train_list(self, file_name='train_list.csv'):
        """
        Load in an array all the names inside the train_list.csv
        """
        self.train_list=[]
        with open(self.database_path+"/"+file_name, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\n')
            for row in spamreader:
                self.train_list.append(', '.join(row))
        return self.train_list
    # Train : TchouTchou -------------------------------------------------------
    def train(self):
        
        pass
if __name__ == "__main__":
    train = TchouTchou()
    train.set_database_path("../database/slices")
    train.generate_train_list()
    train.save_train_list()
    train.load_train_list()
