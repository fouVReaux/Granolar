# -------------------------------------------------------------------------------
# Title : database.py
# Project : Granolar
# Description : slice the database in wav of .npz
# Authors : Cyril Lavrat
# -------------------------------------------------------------------------------

import librosa
import os

class DataBase:
    def __init__(self):
        self.sr = 0
        self.slices_path = ""
        self.database_path = ""
        self.slices_duration = 0

    # Setter -------------------------------------------------------------------
    def set_sr(self, sr):
        """
        set the sampling rate of the database
        """
        self.sr = sr

    def set_slices_path(self, slices_path):
        """
        set the slice path
        """
        self.slices_path = slices_path

    def set_database_path(self, database_path):
        """
        set the database path
        """
        self.database_path = database_path

    def set_slices_duration(self, duration):
        """
        set the duration of a slice (in second)
        """
        self.slices_duration = duration

    # Slicer -------------------------------------------------------------------
    def slice_database(self):
        """
        slice all files into the database directory
        """
        onlyfiles = [f for f in os.listdir(self.database_path) if os.path.isfile(os.path.join(self.database_path, f))]

        for file in onlyfiles:
            self.slice_file(file)
        return

    def slice_file(self, file_name):
        """
        slice a given file and save it in the slice directory
        """
        # create a directory for the slices
        path = os.getcwd()
        path = self.slices_path+"/"

        # get the samplig rate of the file_name
        sr = librosa.get_samplerate(self.database_path+"/"+file_name)
        print("\t> load of the file : ",file_name)
        # Set the frame parameters to be equivalent to the librosa defaults
        # in the file's native sampling rate
        frame_length = (2048 * sr) // 22050
        hop_length = (512 * sr) // 22050

        size_slice = int(self.slices_duration*128/0.3)
        # Stream the data input
        stream = librosa.stream(self.database_path+"/"+file_name,
                                block_length=size_slice,
                                frame_length=frame_length,
                                hop_length=hop_length,
                                mono=True)
        # slice and save the data into output file
        index = 0
        for y in stream:
            # resample our lovely data
            yrs = librosa.resample(y, sr, self.sr)
            # save data into a sweety file of duration [second]
            librosa.output.write_wav(path+str(index)+"_"+file_name, yrs, self.sr, norm=False)
            index +=1
        print("\t\tNumber of slices :", index)
        return index



if __name__ == "__main__":
    #---------------------------------------------------------------------------
    # To test this code add some audio (format wav) in the raw directory
    # run python Database.py in order to slice all the wav file into subfile of
    # 5 seconds
    #---------------------------------------------------------------------------
    db = DataBase()
    db.set_database_path('../database/raw')
    db.set_slices_path('../database/slices')
    db.set_sr(22050)
    db.set_slices_duration(0.5)
    db.slice_database()
