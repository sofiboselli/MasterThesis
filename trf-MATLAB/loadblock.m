function [eeg,att,mas] = loadblock(data,attended,masker,n)

eeg = data(n,:,:,:);
att = attended(n,:);
mas = masker(n,:);

end