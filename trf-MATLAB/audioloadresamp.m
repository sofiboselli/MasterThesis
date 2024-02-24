function [attended,masker] = audioloadresamp(att,mas,ds,bpfilt)

attended = [];
masker = [];

for i = 1:size(att,1)
    attended =  [attended; resample_envelope(att{i},ds,bpfilt)'];
    masker = [masker; resample_envelope(mas{i},ds,bpfilt)'];
end

end