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

class EEG_Dataset(Dataset):

    def __init__(self,seconds=4,overlap=0.75,fs=64,path="EriksholmFiles/",files=[""]):
        super().__init__()

        self.seconds = seconds
        self.overlap = 1-overlap
        self.fs = fs
        self.path = path

        self.win_len = (int(self.fs*33) - int(self.seconds*self.fs)) // int(self.seconds*self.fs*self.overlap)
        self.times = self.seconds*self.fs

        self.files = files

    def __len__(self):
        return self.win_len*len(self.files)
    
    def __getitem__(self, index):


        sub = index//(self.win_len)
        ind = int(index%(self.win_len))
        numb = int(self.files[sub].split("_")[1])
        rr = np.array([numb])
        rr = torch.Tensor(rr).to(torch.int64)

        self.data = np.load(self.files[sub])
        self.eeg = self.data['EEG']
        self.att = self.data['attended']
        self.mas = self.data['masker']
        self.loc = self.data['loc']
        win_loc = torch.Tensor(np.array([self.loc]))

        
        
        win_eeg = torch.from_numpy(np.reshape(window_data(self.eeg,int(self.seconds*self.fs),int(self.seconds*self.fs*self.overlap)),(self.win_len,self.eeg.shape[0],self.times))[ind]).float()
        win_att = torch.from_numpy(np.reshape(window_data(self.att,int(self.seconds*self.fs),int(self.seconds*self.fs*self.overlap)),(self.win_len,self.att.shape[0],self.times))[ind]).float()
        win_mas = torch.from_numpy(np.reshape(window_data(self.mas,int(self.seconds*self.fs),int(self.seconds*self.fs*self.overlap)),(self.win_len,self.mas.shape[0],self.times))[ind]).float()


        win_eeg = (win_eeg - win_eeg.mean(dim=1,keepdim=True))/win_eeg.std(dim=1,keepdim=True)
        win_att =  (win_att - win_att.mean(dim=1,keepdim=True))/win_att.std(dim=1,keepdim=True)
        win_mas =  (win_mas - win_mas.mean(dim=1,keepdim=True))/win_mas.std(dim=1,keepdim=True)
        




        
        return win_eeg,win_att,win_mas,win_loc,rr
            
            
