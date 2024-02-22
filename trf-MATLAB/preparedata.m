function [resp,stim,stim2] = preparedata(eeg,att,mas,ds)


eegp = permute(eeg,[3 2 1]);
eeg_ds = downsample(eegp,256/ds);
[b,a] = butter(3,[1 8]/(ds/2));
k = filter(b,a,eeg_ds);


%k = bandpass(eeg_ds,[2 8],64);
%size(k)
eeg_ds = normalize(k);
%eeg_ds = normalize(eeg_ds,'range');

%eeg_ds = permute(eeg_ds,[3 2 1]);

resp = nt_mat2trial(eeg_ds)';
stim = num2cell(att',1)';
stim2 = num2cell(mas',1)';
end