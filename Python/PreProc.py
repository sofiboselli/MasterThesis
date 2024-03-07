import numpy as np
import librosa
from pymatreader import read_mat

def load_and_spec(filenames,n_splits=33):
    out = []
    for fname in filenames:
        spectro = []
        y,_ = librosa.load(fname,sr=22050)
        y = np.split(y,n_splits)
        for sp in y:
            sp = (sp - np.mean(sp))/np.std(sp)
            spec = librosa.feature.melspectrogram(y=sp,n_mels=80,fmin=0,fmax=8000,norm='slaney',n_fft=2048,hop_length=512)
            spec = np.log(np.clip(spec,1e-5,None))
            spec = (spec - (-0.096733235))/4.0452905
            spectro.append(spec)
        out.append(spectro)
    out = np.array(out)
    return out.reshape((out.shape[0]*out.shape[1], out.shape[2],out.shape[3]))
""
def load_data(n_subs):
    X = []
    att = []
    mas = []
    for i in range(1,n_subs):
        if i == 14:
            continue
        dir = "C:/Users/gauta/Thesis/ExJobb/ID" + str(1000+i) +"/PreprocIV_ID" +str(1000+i)+ "_mast.mat"
        temp_data = read_mat(dir)
        data = np.vstack([np.array(temp_data['ICAcleaned_data_allsess'][i]['trial'])[:,0:64,256*20:256*53] for i in range(4)])
        X.append(data)
        attended = ["C:/Users/gauta/Thesis/ExJobb/Files_Audio/" + temp_data['ICAcleaned_data_allsess'][k]['targetfiles'][i].split('\\')[-1] for k in range(4) for i in range(20)]
        masker = ["C:/Users/gauta/Thesis/ExJobb/Files_Audio/" + temp_data['ICAcleaned_data_allsess'][k]['maskerfiles'][i].split('\\')[-1] for k in range(4) for i in range(20)]
        att.append(attended)
        mas.append(masker)
    X = np.vstack(X)
    attended = np.hstack(att)
    masker = np.hstack(mas)

    return (X,attended,masker)


def load_and_split(filenames):
    out = []
    for fname in filenames:
        s = []
        y,_ = librosa.load(fname,sr=24064)
        y = np.split(y,33)
        for k in y:
            s.append((k-np.mean(k))/np.std(k))
        y = np.vstack(s)
        out.append(y)
    out = np.array(out)
    return out.reshape((out.shape[0]*out.shape[1], out.shape[2]))


def spect(filenames):
    out = []
    for fname in filenames:
        spectro = []
        y,_ = librosa.load(fname,sr=22050)
        #y = y/np.max(y)
        #y = (y - np.mean(y))/np.std(y)
        y = np.split(y,33)
        for sp in y:
            sp = (sp - np.mean(sp))/np.std(sp)
            spec = np.abs(librosa.stft(sp,n_fft=2048,hop_length=512))**2
            spec = np.log(np.clip(spec,1e-5,None))
            
            spectro.append(spec)
        out.append(spectro)
    out = np.array(out)
    return out.reshape((out.shape[0]*out.shape[1], out.shape[2],out.shape[3]))


def prep_EEG(eeg):
    eeg = np.transpose(eeg,(0,2,1))
    EEG = []
    #[b,a] = butter(3,[1*2,32*2],btype = "bandpass",fs=256)
    for ee in eeg:
        #x = filtfilt(b,a,eeg[i],axis=1)
        e = np.split(ee,33)
        for i in range(33):
            e[i] = (e[i]-np.mean(e[i]))/np.std(e[i])

        EEG.append(np.transpose(e,(0,2,1)))

    EEG = np.array(EEG)
    EEG = EEG.reshape((EEG.shape[0]*EEG.shape[1], EEG.shape[2],EEG.shape[3]))
    return EEG