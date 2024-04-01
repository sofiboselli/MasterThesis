import numpy as np
import os
from torch.utils.data import Dataset
import torch


def findfiles(search_dir, prefix):
    matching_files = []

    for dirpath, _, filenames in os.walk(search_dir):
        for filename in filenames:
            if filename.startswith(prefix):
                full_path = os.path.join(dirpath, filename)
                matching_files.append(full_path)

    return matching_files

def window_data(data, window_length, hop):
    new_data = np.empty(((data.shape[1] - window_length) // hop,data.shape[0], window_length))
    #print(new_data.shape)
    for i in range(new_data.shape[0]):
        new_data[i, :, :] = data[:,i * hop : i * hop + window_length]
    return new_data

class C_Dataset(Dataset):
    def __init__(self, mode, seconds=3, overlap=0.75, fs=64, path="DTUFiles/"):
        super().__init__()

        self.seconds = seconds
        self.overlap = 1 - overlap
        self.fs = fs
        self.path = path
        self.mode = mode

        self.times = self.seconds * self.fs

        self.files = findfiles(self.path, self.mode)
        self.indices = []
        total_windows = 0
        for file in self.files:
            eeg_data = np.load(file)['EEG']
            file_windows = (eeg_data.shape[1] - int(self.seconds * self.fs)) // int(self.seconds * self.fs*self.overlap)
            self.indices.extend([(file, i) for i in range(file_windows)])
            total_windows += file_windows
        self.total_windows = total_windows

    def __len__(self):
        return self.total_windows

    def __getitem__(self, index):
        file, window_idx = self.indices[index]
        start_index = window_idx * int(self.seconds * self.fs * self.overlap)
        end_index = start_index + int(self.seconds * self.fs)

        eeg = np.load(file)['EEG'][:, start_index:end_index]
        attended = np.load(file)['attended'][start_index:end_index]
        masker = np.load(file)['masker'][start_index:end_index]

        print(attended.shape)

        eeg = (eeg - eeg.mean(axis=1, keepdims=True)) / eeg.std(axis=1, keepdims=True)
        attended = (attended - attended.mean(axis=0, keepdims=True)) / attended.std(axis=0, keepdims=True)
        masker = (masker - masker.mean(axis=0, keepdims=True)) / masker.std(axis=0, keepdims=True)

        win_eeg = torch.from_numpy(eeg).float()
        win_att = torch.from_numpy(attended).float()
        win_mas = torch.from_numpy(masker).float()

        return win_eeg, win_att, win_mas

