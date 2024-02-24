function [eeg,att,mas] = loadblock(data,attended,masker,n)

eeg = data(n+1,:,:,:);
att = attended(n+1,:);
mas = masker(n+1,:);

eeg = squeeze(eeg);

end