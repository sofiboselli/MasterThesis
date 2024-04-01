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

class DTU_Dataset(Dataset):

    def __init__(self,mode,seconds=4,overlap=0.75,fs=64,path="DTUFiles/"):
        super().__init__()

        self.seconds = seconds
        self.overlap = 1-overlap
        self.fs = fs
        self.path = path
        self.mode = mode

        self.win_len = (3200 - int(self.seconds*self.fs)) // int(self.seconds*self.fs*self.overlap)
        self.times = self.seconds*self.fs

        self.files = findfiles(self.path,self.mode)

    def __len__(self):
        return self.win_len*len(self.files)
    
    def __getitem__(self, index):


        sub = index//(self.win_len)
        ind = int(index%(self.win_len))
        self.data = np.load(self.files[sub])
        self.eeg = self.data['EEG']
        self.att = self.data['attended']
        self.mas = self.data['masker']

        self.eeg = (self.eeg - self.eeg.mean(axis=1,keepdims=True))/self.eeg.std(axis=1,keepdims=True)
        self.att = np.expand_dims((self.att - self.att.mean(axis=0,keepdims=True))/self.att.std(axis=0,keepdims=True),0)
        self.mas = np.expand_dims((self.mas - self.mas.mean(axis=0,keepdims=True))/self.mas.std(axis=0,keepdims=True),0)
        
        win_eeg = torch.from_numpy(np.reshape(window_data(self.eeg,int(self.seconds*self.fs),int(self.seconds*self.fs*self.overlap)),(self.win_len,64,self.times))[ind]).float()
        win_att = torch.from_numpy(np.reshape(window_data(self.att,int(self.seconds*self.fs),int(self.seconds*self.fs*self.overlap)),(self.win_len,self.times))[ind]).float()
        win_mas = torch.from_numpy(np.reshape(window_data(self.mas,int(self.seconds*self.fs),int(self.seconds*self.fs*self.overlap)),(self.win_len,self.times))[ind]).float()


        
        return win_eeg,win_att,win_mas
            
            
