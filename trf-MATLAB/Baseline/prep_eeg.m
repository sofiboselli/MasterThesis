function EEG = prep_eeg(eeg)

 x = eeg(1:64,20*256+1:53*256)';
 [b,a] = butter(3,[1 8]/(64/2),'bandpass');

 EEG = resample(x,64,256);

 EEG = filtfilt(b,a,EEG);

 EEG = normalize(EEG);


end
