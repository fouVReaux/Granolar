# -------------------------------------------------------------------------------
# Title : database.py
# Project : Granolar
# Description : slice the database in wav of .npz
# Authors : Ninon Devis & Cyril Lavrat
# -------------------------------------------------------------------------------
import librosa
import numpy as np
import os

# Audio
from torch.utils.data import Dataset

SAMPLING_RATE = 11025
SLICE_DURATION = 0.01
AUDIO_FORMAT = "npz"
# PATH
DEFAULT_DATABASE_PATH = '../database/raw'
DEFAULT_SLICES_PATH = '../database/slices'


class DataBase(Dataset):
    def __init__(self, audio_files_path=DEFAULT_DATABASE_PATH, sampling_rate=SAMPLING_RATE, slice_duration=SLICE_DURATION,
                 transform=None, save_slices=False, slices_path=DEFAULT_SLICES_PATH, audio_format=AUDIO_FORMAT):
        self.sr = sampling_rate
        self.slices_path = slices_path
        self.database_path = audio_files_path
        self.slices_duration = slice_duration
        self.save_slices = save_slices
        self.type = audio_format
        self.slices = []
        self.transform = transform
        # generate the slices
        only_files = [f for f in os.listdir(self.database_path) if os.path.isfile(os.path.join(self.database_path, f))]
        for file in only_files:
            self.slices += self.__slice_file__(file)

    # Override abstract methods from Dataset -----------------------------------
    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        item = self.slices[index]
        if self.transform is not None:
            item = self.transform(item)
        # return data as a tuple of data, metadata (null)
        return item

    # Slicer -------------------------------------------------------------------
    def __slice_file__(self, file_name):
        """
        PRIVATE: slice a given file and save it in the slice directory. Should not be called directly
        """
        # create a directory for the slices
        path = self.slices_path + "/"

        # get the sampling rate of the file_name
        sr = librosa.get_samplerate(self.database_path + "/" + file_name)
        print("\t> load of the file : ", file_name)
        # Set the frame parameters to be equivalent to the librosa defaults
        # in the file's native sampling rate
        frame_length = (2048 * sr) // 22050
        hop_length = (512 * sr) // 22050

        size_slice = int(self.slices_duration * 128 / 0.3)
        # Stream the data input
        stream = librosa.stream(self.database_path + "/" + file_name,
                                block_length=size_slice,
                                frame_length=frame_length,
                                hop_length=hop_length,
                                mono=True)
        # slice and save the data into output file
        index = 0
        slices = []
        for y in stream:
            # resample our lovely data
            yrs = librosa.resample(y, sr, self.sr)
            if self.save_slices:
                # save data into a sweety file of duration [second]
                if self.type == "wav":
                    librosa.output.write_wav(path + str(index) + "_" + file_name, yrs, self.sr, norm=False)
                elif self.type == "npz":
                    np.savez_compressed(path + str(index) + "_" + file_name + '.npz', self.sr, yrs)
            index += 1
            slices.append(yrs)
        print("\t\tNumber of slices :", index)
        print('Size of slices:', len(yrs))
        return slices


if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # To test this code add some audio (format wav) in the raw directory
    # run python database.py in order to slice all the wav file into subfile of
    # 5 seconds
    # ---------------------------------------------------------------------------
    db = DataBase(save_slices=False)
