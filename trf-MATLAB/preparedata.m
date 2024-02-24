function [resp,stim,stim2] = preparedata(eeg,att,mas,ds,bpFilter)


eegp = permute(eeg,[3 2 1]);

for i = 1:size(eegp,3)
    temp = squeeze(eegp(:,:,i));
    %temp = temp./std(temp,0,1);
    [b,a] = butter(3,[1 9]/256*2);
    temp = filter(b,a,double(temp));
    temp = downsample(temp,256/ds);
    temp = normalize(temp);
    eeg_ds(:,:,i) = temp;
end


%eeg_ds = downsample(eegp,256/64);
%eegp = eeg_ds;


%k = bandpass(eeg_ds,[2 8],64);
%size(k)
%eeg_ds = normalize(eeg_ds);
%eeg_ds = normalize(eeg_ds,'range');

%eeg_ds = permute(eeg_ds,[3 2 1]);

for j = 1:20
    stim{j} = att(j,:)';
    stim2{j} = mas(j,:)';
stim = stim';
stim2 = stim2';
resp = nt_mat2trial(eeg_ds)';
%sum(sum(resp{1} - squeeze(eeg_ds(:,:,1))))

end