function [attended,masker] = audioloadresamp(att,mas,ds)

attended = [];
masker = [];

for i = 1:size(att,1)
    attended =  [attended; resample_envelope(att{i},ds)'];
    masker = [masker; resample_envelope(mas{i},ds)'];
end

end