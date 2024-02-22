function [eeg,att,mas] = loadsubject(data,attended,masker,n)

eeg = data(n*4+1:(n+1)*4,:,:,:);


eeg = permute(eeg,[2 1 3 4]);

eeg_size = size(eeg);

eeg = reshape(eeg, [eeg_size(1)*eeg_size(2),66,8448]);

att = attended(n*4+1:(n+1)*4,:);
mas = masker(n*4+1:(n+1)*4,:);

att = att';
mas = mas';

att = reshape(att, [eeg_size(1)*eeg_size(2) 1]);
mas = reshape(mas, [eeg_size(1)*eeg_size(2) 1]);

end