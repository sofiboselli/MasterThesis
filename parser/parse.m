clear all;
clc;

direct = dir;
dirs = zeros([1,length(dir)]);

l = string([]);

dirpath = direct.folder;
names = {direct.name};
for j = 1:numel(names)
    if names{j}(1)=='I'
        m = dirpath + "\" + names{j} + "\PreprocIV_" + names{j} + "_mast.mat";
        l(end+1) = m;
    else
        continue;
    end
end

%%

data = zeros([31*4,20,66,8448]);
label = zeros([31*4,20]);
sound_attended = strings([31*4,20]);
sound_masker = strings([31*4,20]);
d = zeros([20,66,8448]);
for k =progress(1:numel(l))
    s = load(l(k));
    ica = s.ICAcleaned_data_allsess;
    for j = 1:numel(ica)
        for z = 1:20
            d(z,:,:) = ica{j}.trial{z}(:,5*256+1:38*256);
        end
        data(4*(k-1)+j,:,:,:) = d;
        label(4*(k-1)+j,:) = ica{j}.trialid(:,6,:);
        a = ica{j}.targetfiles;
        m = ica{j}.maskerfiles;
        sound_attended(4*(k-1)+j,:) = cellfun(@myfun,a);
        sound_masker(4*(k-1)+j,:) = cellfun(@myfun,m);
    end
end

sound_attended = dirpath+ "\Files_Audio\" + sound_attended;
sound_masker = dirpath +"\Files_Audio\" + sound_masker;
sound_attended = convertStringsToChars(sound_attended);
sound_masker = convertStringsToChars(sound_masker);


%%

save('att.mat',"sound_attended",'-mat')
save('mas.mat',"sound_masker",'-mat')
save('X.mat',"data",'-v7.3')
save('y.mat',"label")

